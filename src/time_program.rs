//! A formula for T: the expression-DAG system, pointed at the time axis.
//!
//! A [`crate::formula::TimeMod`] drives a scalar with one of seven fixed shapes.
//! A [`TimeProgram`] replaces that fixed vocabulary with an evolved
//! [`crate::formula::OpNode`] DAG — the same representation the fractals
//! themselves use — evaluated as a function of `t` instead of as a function of
//! the iterate.
//!
//! # The two leaves, reinterpreted
//!
//! Everything here follows from giving `op::Z` and `op::C` new meanings. With
//! `x = freq·t + phase`:
//!
//! | leaf | in a fractal | here |
//! |------|--------------|------|
//! | `c`  | the parameter | the **phasor** `e^{2πi·x}` |
//! | `z`  | the iterate   | the **ramp** `(x, 0)` |
//! | `k`  | a constant    | a constant |
//!
//! The consequence worth the whole design: **a DAG that reads only the phasor is
//! exactly periodic.** It is a function on the unit circle — a Fourier-type
//! series in `e^{2πit}` — so at an integer `freq` the clip loops seamlessly with
//! no analysis, no windowing and no special cases. Reading the ramp buys
//! one-way travel and gives up the loop, and [`Profile::loops`] says which of
//! the two a given program is.
//!
//! # Three things this deliberately does not add
//!
//! 1. **No evaluator.** [`crate::formula::f64_impl::eval_program`] is called
//!    directly, exactly as `fractal::dag_escape_pixel_f64` calls it.
//! 2. **No VM mirrors.** A time program runs once per FRAME on the host, not
//!    once per pixel, so — unlike a new opcode — it never touches
//!    `fractal.wgsl`, `dd.rs`, or the `define_precision!` macro. That is also
//!    why it could safely grow ops the pixel VM does not have.
//! 3. **No genetic operators.** `genome::random_program`/`mutate_program`/
//!    `crossover_program`/`strip_dead` are already topology-safe and cap-aware;
//!    evolving time programs is those functions with a different fitness.
//!
//! # Why normalisation is the crux
//!
//! A random DAG containing `EXP`, `DIV`, `RECIP` or `LOG` has unbounded range,
//! so its raw output cannot drive a genome scalar directly. Every evaluation
//! therefore samples the program across one clip, subtracts the mean, and
//! rescales peak magnitude to 1 before applying `amp`.
//!
//! Subtracting the mean is not cosmetic. `TimeMod` promises that its value is an
//! OFFSET added to the genome's own scalar, so `amp = 0` is *exactly* the stored
//! fractal and a modulation can be dialled from "off" to "wild" without the
//! fractal's identity jumping. A DAG with a large DC term would break that
//! promise the moment it was attached. Removing the DC keeps it.

use serde::{Deserialize, Serialize};

use crate::formula::{op, reachable_from_root, ModTarget, OpNode};

/// Samples per cycle used to find the DC offset and peak magnitude.
///
/// Frozen: changing it changes the normalisation, and therefore changes how
/// every already-saved `.nn` carrying a time program renders. Treat it the way
/// the opcode numbers are treated.
pub const NORM_SAMPLES_PER_CYCLE: usize = 64;

/// Samples per cycle used by the smoothness gate. Higher than the normalisation
/// rate on purpose: the defect it looks for is a pole *between* two coarse
/// samples, which a coarse sampling is structurally blind to.
pub const SMOOTH_SAMPLES_PER_CYCLE: usize = 256;

/// Hard ceiling on samples per evaluation, so a genome carrying an absurd
/// `freq` costs bounded time rather than hanging a render.
pub const MAX_SAMPLES: usize = 4096;

/// Smallest peak-to-DC ratio that counts as an animation rather than rounding
/// error. Relative, not absolute: normalisation rescales any nonzero variation
/// up to full amplitude, so an absolute floor would happily promote the last
/// bits of f64 noise into a full-amplitude "modulation".
pub const MIN_TRAVEL_REL: f64 = 1e-6;

/// Largest tolerated single step between adjacent samples, as a fraction of the
/// normalised range (peak magnitude is 1, so the diameter is at most 2).
///
/// This is the parameter-space analogue of `time_explore::MAX_LEVEL_JUMP`, and
/// it exists for the same reason that one does — except it costs no rendering at
/// all, which is what makes a GA over these programs affordable. A smooth
/// sinusoid sampled at [`SMOOTH_SAMPLES_PER_CYCLE`] steps by about `2π/256` ≈
/// 0.025 per sample; a pole steps by nearly the full diameter. The gap between
/// those is wide enough that the threshold is not delicate.
///
/// The sample count scales with `freq` (see [`sample_count`]) so this stays a
/// property of the program's shape rather than of how fast it is played.
pub const MAX_STEP_FRACTION: f64 = 0.25;

/// How close `freq` must be to a whole number for the clip to close the loop.
pub const LOOP_FREQ_TOLERANCE: f32 = 1e-3;

/// Largest offset a channel may apply, as a fraction of the rendered view's
/// half-extent.
///
/// A modulation is only an animation while it moves the picture by less than a
/// frame. Half a half-extent per clip is a generous ceiling for a morph and a
/// hard stop against a cut.
pub const MAX_OFFSET_FRACTION: f32 = 0.5;

/// The deepest zoom at which a `frames`-frame clip can still animate SMOOTHLY.
///
/// # Why there is a limit, and why it is not a tuning problem
///
/// Every animatable scalar on a `Genome` is an `f32`. The smallest change
/// representable in an `f32` of magnitude ~1 is one ULP, about 1.2e-7. A view at
/// zoom `z` spans `2/z` in each direction, and [`MAX_OFFSET_FRACTION`] of that
/// is all a modulation may travel. So the number of DISTINCT values a
/// modulation can take within a frame is `MAX_OFFSET_FRACTION · 2 / (z · ULP)`,
/// and a clip of `frames` frames needs at least that many to move a little each
/// frame instead of jumping.
///
/// The answer is far shallower than it first looks. "One ULP still fits inside
/// the frame" gives z ≈ 1.7e7 — but that is one single step for the whole clip,
/// which is a cut, not an animation. For 40 frames it is z ≈ 2.1e5.
///
/// # Measured
///
/// On a real shot (2026-09-10) from zoom 0.147 to 3.14e11 — 41 doublings — the
/// time search returned `0 winners · 5 flash · 4 static` at every generation.
/// That split is the signature: large amplitudes cut the frame, small ones round
/// to nothing, and no amplitude between them survives because there is no
/// representable value between them. Capping the search at the one-ULP depth was
/// not enough and produced the same result; the smooth limit is what made the
/// search viable. At that shot's end zoom one ULP is 18,716 screen-widths.
///
/// So this is not an amplitude-range problem. Past this depth the camera can
/// keep going but the formula must hold still, which is what the offset clamp in
/// `Genome::at_time_in_view` makes happen gracefully, and what
/// `time_ga::depth_views` refuses to search past.
pub fn animatable_zoom_limit(frames: u32) -> f64 {
    let steps = frames.max(1) as f64;
    2.0 * MAX_OFFSET_FRACTION as f64 / (steps * f32::EPSILON as f64)
}

/// Scale `(re, im)` down so its magnitude is at most `cap`. Never scales up.
pub fn clamp_offset(re: f32, im: f32, cap: f32) -> (f32, f32) {
    if !cap.is_finite() {
        return (re, im);
    }
    let mag = (re * re + im * im).sqrt();
    if mag <= cap || mag == 0.0 || !mag.is_finite() {
        (re, im)
    } else {
        let k = cap / mag;
        (re * k, im * k)
    }
}

/// One evolved time channel: a DAG, what it drives, and how hard.
///
/// The scalars mean the same things they mean on [`crate::formula::TimeMod`],
/// with one difference — `amp` is applied AFTER normalisation, so it is the peak
/// offset in the target's own units regardless of what the DAG's raw range
/// happens to be.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeProgram {
    pub target: ModTarget,
    /// The program. Evaluated with the phasor as `c` and the ramp as `z`.
    pub prog: Vec<OpNode>,
    /// Peak offset applied to the target, after normalisation.
    #[serde(default)]
    pub amp: f32,
    /// Cycles over the full clip. 1.0 = exactly one loop.
    #[serde(default = "default_freq")]
    pub freq: f32,
    /// Where in the cycle the clip starts, in turns.
    #[serde(default)]
    pub phase: f32,
}

fn default_freq() -> f32 {
    1.0
}

impl TimeProgram {
    pub fn new(target: ModTarget, prog: Vec<OpNode>, amp: f32) -> Self {
        TimeProgram { target, prog, amp, freq: 1.0, phase: 0.0 }
    }

    /// The complex offset at `t ∈ [0,1)`, normalised and scaled by `amp`.
    ///
    /// Returns `(0, 0)` for any program the gates reject, so an unusable
    /// program renders the fractal unchanged rather than producing garbage.
    pub fn eval(&self, t: f32) -> (f32, f32) {
        let Some(norm) = self.norm() else { return (0.0, 0.0) };
        let (re, im) = raw(&self.prog, self.freq, self.phase, t);
        let (re, im) = norm.apply(re, im);
        (self.amp * re as f32, self.amp * im as f32)
    }

    /// The DC offset and scale factor this program is normalised by, or `None`
    /// when it is not usable at all (non-finite, or constant).
    ///
    /// Recomputed on demand rather than cached in a field. It is pure and
    /// deterministic, a few thousand flops against a multi-megapixel render, and
    /// a stored copy could silently go stale the moment the program is edited.
    pub fn norm(&self) -> Option<Norm> {
        normalize(&self.prog, self.freq, self.phase)
    }

    /// Measure the program and decide whether it is usable, without rendering
    /// a single pixel. See [`Profile`].
    pub fn profile(&self) -> Profile {
        profile(&self.prog, self.freq, self.phase)
    }

    /// `f(t) = …` in the same infix notation `Genome::formula_expr` uses, with
    /// the leaves named for what they mean here.
    pub fn expr(&self) -> String {
        if self.prog.is_empty() {
            return "f(t) = 0".into();
        }
        format!(
            "f(t) = {}",
            crate::genome::render_node_with(&self.prog, self.prog.len() - 1, 0, "t", "e^iτt")
        )
    }

    /// One-line summary for manifests, logs and the viewer's channel list.
    pub fn label(&self) -> String {
        format!(
            "{} amp={:.3} freq={:.2} phase={:.2}  {}",
            self.target.label(),
            self.amp,
            self.freq,
            self.phase,
            self.expr()
        )
    }
}

/// The DC offset and inverse peak magnitude a program is normalised by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm {
    pub dc_re: f64,
    pub dc_im: f64,
    /// Multiplier that takes the DC-removed signal to unit peak magnitude.
    pub scale: f64,
}

impl Norm {
    #[inline]
    pub fn apply(&self, re: f64, im: f64) -> (f64, f64) {
        ((re - self.dc_re) * self.scale, (im - self.dc_im) * self.scale)
    }
}

/// What a program measures, and which gate (if any) rejects it.
///
/// Every field here comes from evaluating the DAG a few hundred times — no
/// rendering, no ffmpeg, no GPU. That is deliberate: it is what lets a genetic
/// search discard most of its population for free, before the expensive
/// clip-based gates in `time_explore` ever run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    /// Every sample was finite.
    pub finite: bool,
    /// Peak magnitude of the DC-removed signal, relative to the DC level.
    /// Answers "does this function actually go anywhere".
    pub travel_rel: f64,
    /// Largest single step between adjacent normalised samples. Catches poles.
    pub max_step: f64,
    /// The clip returns to where it started: the program never reads the ramp,
    /// and `freq` is a whole number.
    pub loops: bool,
    /// Which gate stopped it, if any.
    pub rejected: Option<&'static str>,
}

impl Profile {
    pub fn passed(&self) -> bool {
        self.rejected.is_none()
    }
}

/// How many samples one pass takes, so that the sampling RATE per cycle is what
/// `per_cycle` says regardless of `freq`.
///
/// Without this the gates would measure how fast a program is played rather than
/// what shape it is: at `freq = 8` a fixed 256 samples gives only 32 per cycle,
/// and a perfectly smooth sinusoid would step 8× further between samples and
/// read as a discontinuity.
pub fn sample_count(per_cycle: usize, freq: f32) -> usize {
    let cycles = if freq.is_finite() { freq.abs().max(1.0) as f64 } else { 1.0 };
    ((per_cycle as f64 * cycles).ceil() as usize).clamp(per_cycle, MAX_SAMPLES)
}

/// The program's raw complex value at `t`, before normalisation.
///
/// Both leaves read the same phase argument `x = freq·t + phase`: the phasor is
/// `e^{2πi·x}`, the ramp is `x`. Sharing `x` is what keeps `freq` meaning
/// "cycles over the clip" for a program built from either leaf, or both.
pub fn raw(prog: &[OpNode], freq: f32, phase: f32, t: f32) -> (f64, f64) {
    if prog.is_empty() {
        return (0.0, 0.0);
    }
    let x = freq as f64 * t as f64 + phase as f64;
    let theta = std::f64::consts::TAU * x;
    crate::formula::f64_impl::eval_program(prog, x, 0.0, theta.cos(), theta.sin())
}

/// Sample one full clip at `n` evenly spaced points over `t ∈ [0,1)`.
pub fn samples(prog: &[OpNode], freq: f32, phase: f32, n: usize) -> Vec<(f64, f64)> {
    let n = n.max(2);
    (0..n)
        .map(|i| raw(prog, freq, phase, i as f32 / n as f32))
        .collect()
}

/// The DC offset and scale a program normalises by, or `None` when it is
/// non-finite or constant.
pub fn normalize(prog: &[OpNode], freq: f32, phase: f32) -> Option<Norm> {
    let n = sample_count(NORM_SAMPLES_PER_CYCLE, freq);
    let s = samples(prog, freq, phase, n);
    if s.is_empty() || s.iter().any(|(re, im)| !re.is_finite() || !im.is_finite()) {
        return None;
    }
    let inv = 1.0 / s.len() as f64;
    let dc_re = s.iter().map(|v| v.0).sum::<f64>() * inv;
    let dc_im = s.iter().map(|v| v.1).sum::<f64>() * inv;
    let peak = s
        .iter()
        .map(|&(re, im)| ((re - dc_re).powi(2) + (im - dc_im).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    let dc_mag = (dc_re * dc_re + dc_im * dc_im).sqrt();
    if !peak.is_finite() || peak <= MIN_TRAVEL_REL * (1.0 + dc_mag) {
        return None;
    }
    Some(Norm { dc_re, dc_im, scale: 1.0 / peak })
}

/// Whether `prog`'s live subgraph reads the ramp leaf.
///
/// Uses `reachable_from_root` because evolved DAGs are roughly half intron: a
/// `Z` node sitting in a dead subtree does not stop the clip from looping, and
/// reporting otherwise would reject good programs for a node nothing reads.
pub fn reads_ramp(prog: &[OpNode]) -> bool {
    let live = reachable_from_root(prog);
    prog.iter().zip(live).any(|(n, alive)| alive && n.op == op::Z)
}

/// Measure a program and apply the four free gates, in order.
///
/// Order matters for the same reason it matters in `time_explore::gate`: the
/// reason reported is the FIRST failure, not the only one.
pub fn profile(prog: &[OpNode], freq: f32, phase: f32) -> Profile {
    let loops = !reads_ramp(prog) && (freq - freq.round()).abs() <= LOOP_FREQ_TOLERANCE;

    if prog.is_empty() {
        return Profile { finite: true, travel_rel: 0.0, max_step: 0.0, loops, rejected: Some("empty") };
    }

    let n = sample_count(NORM_SAMPLES_PER_CYCLE, freq);
    let s = samples(prog, freq, phase, n);
    if s.iter().any(|(re, im)| !re.is_finite() || !im.is_finite()) {
        return Profile { finite: false, travel_rel: 0.0, max_step: f64::INFINITY, loops,
                         rejected: Some("non-finite") };
    }

    let inv = 1.0 / s.len() as f64;
    let dc_re = s.iter().map(|v| v.0).sum::<f64>() * inv;
    let dc_im = s.iter().map(|v| v.1).sum::<f64>() * inv;
    let peak = s
        .iter()
        .map(|&(re, im)| ((re - dc_re).powi(2) + (im - dc_im).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    let dc_mag = (dc_re * dc_re + dc_im * dc_im).sqrt();
    let travel_rel = peak / (1.0 + dc_mag);
    if !peak.is_finite() || travel_rel <= MIN_TRAVEL_REL {
        return Profile { finite: true, travel_rel, max_step: 0.0, loops, rejected: Some("constant") };
    }

    // Re-sample finer for the smoothness gate: the whole point is to look
    // between the normalisation samples.
    let norm = Norm { dc_re, dc_im, scale: 1.0 / peak };
    let fine = samples(prog, freq, phase, sample_count(SMOOTH_SAMPLES_PER_CYCLE, freq));
    if fine.iter().any(|(re, im)| !re.is_finite() || !im.is_finite()) {
        return Profile { finite: false, travel_rel, max_step: f64::INFINITY, loops,
                         rejected: Some("non-finite") };
    }
    // Includes the wrap from the last sample back to the first: a program that
    // is smooth throughout but lands somewhere else at t=1 produces a visible
    // cut at the loop point, and that is the same defect.
    let mut max_step = 0.0f64;
    for i in 0..fine.len() {
        let a = norm.apply(fine[i].0, fine[i].1);
        let b = norm.apply(fine[(i + 1) % fine.len()].0, fine[(i + 1) % fine.len()].1);
        // The wrap step only counts for a program that claims to loop; a
        // one-shot ramp is expected to end away from where it began.
        if i + 1 == fine.len() && !loops {
            continue;
        }
        max_step = max_step.max(((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt());
    }
    let rejected = if max_step > MAX_STEP_FRACTION { Some("discontinuous") } else { None };

    Profile { finite: true, travel_rel, max_step, loops, rejected }
}

/// Node cap for a randomly generated time program.
///
/// Well below `N_SLOTS`: unlike a fractal formula, a time function's whole job
/// is to be legible as motion, and a 24-node graph of nested `EXP`/`DIV` is
/// overwhelmingly likely to be a spike the smoothness gate throws out anyway.
/// Smaller programs also make the free gates a cheaper filter, which is the
/// point of having them.
pub const DEFAULT_MAX_NODES: usize = 10;
pub const DEFAULT_MAX_DEPTH: usize = 4;

/// Frequencies a random program draws from, weighted toward one cycle.
///
/// Whole numbers only: at a non-integer `freq` even a phasor-only program ends
/// the clip somewhere other than where it started, which reads as a cut at the
/// loop point. A fractional frequency is something to set deliberately, not to
/// stumble into.
pub const LOOP_FREQS: &[f32] = &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0];

/// A random time program that passes the free gates, or `None` if `tries`
/// attempts all failed.
///
/// Reuses `genome::random_program` outright — the DAG is the same object here as
/// it is in a fractal, so growing one is the same problem. The only time-specific
/// step is `prefer_loop`, which rewrites every ramp leaf into a phasor leaf: a
/// program built only from the phasor is periodic by construction, so this turns
/// "does it loop" from something to test into something to decide.
pub fn random(
    rng: &mut impl rand::Rng, target: ModTarget, amp: f32, prefer_loop: bool, tries: usize,
) -> Option<TimeProgram> {
    for _ in 0..tries.max(1) {
        let exotic = rng.random_bool(0.4);
        let mut prog =
            crate::genome::random_program(rng, DEFAULT_MAX_NODES, DEFAULT_MAX_DEPTH, exotic);
        if prefer_loop {
            for n in prog.iter_mut() {
                if n.op == op::Z {
                    n.op = op::C;
                }
            }
        }
        prog = crate::genome::strip_dead(&prog);
        let freq = if prefer_loop {
            LOOP_FREQS[rng.random_range(0..LOOP_FREQS.len())]
        } else {
            0.25 + rng.random::<f32>() * 3.75
        };
        let tp = TimeProgram { target, prog, amp, freq, phase: rng.random::<f32>() };
        if tp.profile().passed() {
            return Some(tp);
        }
    }
    None
}

/// Evaluate a whole channel list at `t`, summing per target.
///
/// Split out so `Genome::at_time`, the viewer's plot and the GA all agree on
/// what a stack of channels does.
pub fn offsets_at(progs: &[TimeProgram], t: f32) -> Vec<(ModTarget, f32, f32)> {
    progs.iter().map(|tp| {
        let (re, im) = tp.eval(t);
        (tp.target, re, im)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: u8, a: u8, b: u8) -> OpNode {
        OpNode { op, a, b, kre: 0.0, kim: 0.0 }
    }
    fn konst(kre: f32, kim: f32) -> OpNode {
        OpNode { op: op::CONST, a: 0, b: 0, kre, kim }
    }

    /// `c` alone: the unit-circle phasor. The canonical looping program.
    fn phasor() -> Vec<OpNode> {
        vec![node(op::C, 0, 0)]
    }

    /// `z` alone: the ramp. The canonical one-shot program.
    fn ramp() -> Vec<OpNode> {
        vec![node(op::Z, 0, 0)]
    }

    #[test]
    fn phasor_program_is_exactly_periodic() {
        let tp = TimeProgram::new(ModTarget::JuliaC, phasor(), 1.0);
        let a = tp.eval(0.0);
        let b = tp.eval(1.0);
        assert!((a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5,
                "phasor must close the loop: {a:?} vs {b:?}");
        assert!(tp.profile().loops, "a program reading only the phasor loops");
    }

    #[test]
    fn ramp_program_does_not_loop_but_is_still_usable() {
        let tp = TimeProgram::new(ModTarget::JuliaC, ramp(), 1.0);
        let p = tp.profile();
        assert!(!p.loops, "a program reading the ramp cannot loop");
        assert!(p.passed(), "not looping is advisory, not a rejection: {p:?}");
    }

    #[test]
    fn a_dead_ramp_node_does_not_stop_the_loop() {
        // z is present but unreachable from the root, which is `c`.
        let prog = vec![node(op::Z, 0, 0), node(op::C, 0, 0)];
        assert!(!reads_ramp(&prog), "an intron must not count as reading the ramp");
        assert!(profile(&prog, 1.0, 0.0).loops);
    }

    #[test]
    fn normalisation_removes_dc_and_scales_to_unit_peak() {
        // c + (10 + 0i): a unit circle centred far from the origin.
        let prog = vec![node(op::C, 0, 0), konst(10.0, 0.0), node(op::ADD, 0, 1)];
        let n = normalize(&prog, 1.0, 0.0).expect("finite, non-constant");
        assert!((n.dc_re - 10.0).abs() < 1e-6, "DC should be the offset, got {}", n.dc_re);
        assert!((n.scale - 1.0).abs() < 1e-6, "unit circle already has peak 1, got {}", n.scale);

        // The peak magnitude of the normalised signal is 1 regardless of offset.
        let tp = TimeProgram::new(ModTarget::JuliaC, prog, 1.0);
        let peak = (0..64)
            .map(|i| {
                let (re, im) = tp.eval(i as f32 / 64.0);
                (re * re + im * im).sqrt()
            })
            .fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 1e-3, "amp=1 must mean peak offset 1, got {peak}");
    }

    #[test]
    fn amp_scales_the_offset_linearly() {
        let mut tp = TimeProgram::new(ModTarget::JuliaC, phasor(), 1.0);
        let one = tp.eval(0.125);
        tp.amp = 0.25;
        let quarter = tp.eval(0.125);
        assert!((quarter.0 - one.0 * 0.25).abs() < 1e-6);
        assert!((quarter.1 - one.1 * 0.25).abs() < 1e-6);
    }

    #[test]
    fn amp_zero_is_exactly_no_offset() {
        // The invariant TimeMod already promises, which the DC removal exists
        // to preserve for a DAG with a large constant term.
        let prog = vec![node(op::C, 0, 0), konst(1000.0, -7.0), node(op::ADD, 0, 1)];
        let tp = TimeProgram { target: ModTarget::JuliaC, prog, amp: 0.0, freq: 1.0, phase: 0.0 };
        for i in 0..16 {
            let (re, im) = tp.eval(i as f32 / 16.0);
            assert_eq!((re, im), (0.0, 0.0), "amp=0 must be the identity at every t");
        }
    }

    #[test]
    fn a_constant_program_is_rejected() {
        let prog = vec![konst(0.5, 0.5)];
        let p = profile(&prog, 1.0, 0.0);
        assert_eq!(p.rejected, Some("constant"), "{p:?}");
        assert!(normalize(&prog, 1.0, 0.0).is_none());
        // And it renders as no offset at all rather than as garbage.
        let tp = TimeProgram::new(ModTarget::Bailout, prog, 1.0);
        assert_eq!(tp.eval(0.3), (0.0, 0.0));
    }

    #[test]
    fn a_pole_is_rejected_as_discontinuous() {
        // 1/(c - 1): blows up as the phasor passes through +1, i.e. at t = 0.
        let prog = vec![node(op::C, 0, 0), konst(1.0, 0.0), node(op::SUB, 0, 1),
                        node(op::RECIP, 2, 0)];
        let p = profile(&prog, 1.0, 0.0);
        assert!(!p.passed(), "a pole must not pass: {p:?}");
        assert!(p.max_step > MAX_STEP_FRACTION, "step {} should exceed {MAX_STEP_FRACTION}", p.max_step);
    }

    #[test]
    fn a_smooth_sinusoid_passes_at_every_reasonable_freq() {
        // sin of the phasor — smooth, bounded, and periodic.
        let prog = vec![node(op::C, 0, 0), node(op::SIN, 0, 0)];
        for freq in [1.0f32, 2.0, 5.0, 12.0] {
            let p = profile(&prog, freq, 0.0);
            assert!(p.passed(), "freq {freq} should pass but got {p:?}");
            assert!(p.max_step < MAX_STEP_FRACTION,
                    "freq {freq} stepped {} — the gate must measure shape, not playback rate",
                    p.max_step);
        }
    }

    #[test]
    fn sample_count_holds_the_rate_per_cycle() {
        assert_eq!(sample_count(64, 1.0), 64);
        assert_eq!(sample_count(64, 4.0), 256);
        assert_eq!(sample_count(64, 0.25), 64, "below one cycle, sample the whole clip");
        assert_eq!(sample_count(64, 1.0e9), MAX_SAMPLES, "cost must stay bounded");
        assert_eq!(sample_count(64, f32::NAN), 64, "a non-finite freq must not hang");
    }

    #[test]
    fn non_finite_output_is_rejected_not_propagated() {
        // log(0) at t where the phasor equals its own constant.
        let prog = vec![node(op::C, 0, 0), node(op::C, 0, 0), node(op::SUB, 0, 1),
                        node(op::RECIP, 2, 0)];
        let p = profile(&prog, 1.0, 0.0);
        assert!(!p.passed(), "1/0 must be rejected: {p:?}");
        let tp = TimeProgram::new(ModTarget::JuliaC, prog, 1.0);
        let (re, im) = tp.eval(0.4);
        assert!(re.is_finite() && im.is_finite(), "a rejected program must still eval finitely");
    }

    #[test]
    fn empty_program_is_inert() {
        let tp = TimeProgram::new(ModTarget::JuliaC, Vec::new(), 1.0);
        assert_eq!(tp.eval(0.5), (0.0, 0.0));
        assert_eq!(tp.profile().rejected, Some("empty"));
    }

    #[test]
    fn round_trips_through_json() {
        let tp = TimeProgram {
            target: ModTarget::ProgConst { node: 3 },
            prog: vec![node(op::C, 0, 0), node(op::SIN, 0, 0)],
            amp: 0.125,
            freq: 2.0,
            phase: 0.25,
        };
        let s = serde_json::to_string(&tp).unwrap();
        let back: TimeProgram = serde_json::from_str(&s).unwrap();
        assert_eq!(tp, back);
    }

    #[test]
    fn omitted_scalars_take_their_defaults() {
        // Forward compatibility in the direction that matters: a hand-written
        // or older program that names only what it must.
        let back: TimeProgram =
            serde_json::from_str(r#"{"target":"Bailout","prog":[{"op":1,"a":0,"b":0}]}"#).unwrap();
        assert_eq!(back.amp, 0.0);
        assert_eq!(back.freq, 1.0, "freq must default to one cycle, not zero");
        assert_eq!(back.phase, 0.0);
    }

    #[test]
    fn random_programs_pass_their_own_gates_and_loop_when_asked() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260910);
        let mut made = 0;
        for _ in 0..200 {
            let Some(tp) = random(&mut rng, ModTarget::JuliaC, 0.1, true, 8) else { continue };
            made += 1;
            let p = tp.profile();
            assert!(p.passed(), "random() must not return a program it would reject: {p:?}");
            assert!(p.loops, "prefer_loop must guarantee a loop, not merely favour one");
            assert!(tp.prog.len() <= DEFAULT_MAX_NODES + 1,
                    "grew past the cap: {} nodes", tp.prog.len());
            // And the promise the whole normalisation exists for.
            let (a, b) = (tp.eval(0.0), tp.eval(1.0));
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4,
                    "a looping program must close: {a:?} vs {b:?}  {}", tp.expr());
        }
        assert!(made > 150, "only {made}/200 attempts produced a usable program — the \
                             generator is fighting the gates");
    }

    #[test]
    fn the_animatable_depth_limit_gives_one_representable_step_per_frame() {
        for frames in [12u32, 40, 48] {
            let z = animatable_zoom_limit(frames);
            // At the limit the offset cap is exactly `frames` ULPs wide, so the
            // clip has one distinct representable value per frame.
            let cap = MAX_OFFSET_FRACTION as f64 * 2.0 / z;
            let steps = cap / f32::EPSILON as f64;
            assert!((steps - frames as f64).abs() < 1e-6, "{frames} frames gave {steps} steps");
        }
        // More frames demand a shallower zoom, never a deeper one.
        assert!(animatable_zoom_limit(40) < animatable_zoom_limit(12));
        // And the measured 41-doubling shot is far past it either way.
        assert!(animatable_zoom_limit(40) < 3.14e11);
        // "one step for the whole clip" is the loosest it can be — and is a cut.
        assert!((animatable_zoom_limit(1) - 2.0 * MAX_OFFSET_FRACTION as f64
                 / f32::EPSILON as f64).abs() < 1.0);
    }

    #[test]
    fn clamping_only_ever_shrinks_an_offset() {
        // Inside the cap: untouched, exactly.
        assert_eq!(clamp_offset(0.1, 0.0, 0.5), (0.1, 0.0));
        // Outside: scaled to the cap, direction preserved.
        let (re, im) = clamp_offset(3.0, 4.0, 1.0);
        assert!(((re * re + im * im).sqrt() - 1.0).abs() < 1e-6);
        assert!((re / im - 3.0 / 4.0).abs() < 1e-6, "direction must be preserved");
        // Degenerate inputs must not produce NaN.
        assert_eq!(clamp_offset(0.0, 0.0, 0.5), (0.0, 0.0));
        assert_eq!(clamp_offset(1.0, 2.0, f32::INFINITY), (1.0, 2.0), "no cap = no clamp");
    }

    #[test]
    fn expr_names_the_leaves_for_the_time_axis() {
        let tp = TimeProgram::new(ModTarget::JuliaC, vec![node(op::C, 0, 0), node(op::SIN, 0, 0)], 1.0);
        let e = tp.expr();
        assert!(e.contains("e^iτt"), "the phasor leaf must not print as `c`: {e}");
        assert!(!e.contains(" c)") && !e.starts_with("f(t) = c"), "{e}");
    }
}
