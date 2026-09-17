//! A camera motion driven by an actual physics simulation instead of a
//! closed-form curve: the "projection" (`Slice`) is a projectile orbiting
//! the fractal's own mass — a center of mass built by treating every
//! sampled point's escape time as its mass ("the highest iteration has the
//! highest mass"), reusing the exact same `quat_escape` every other
//! renderer in this project uses. That's what makes this generalize to any
//! `QuatFormula`, not just the one it's demoed on (`bulb`).
//!
//! Deliberately a POINT-MASS approximation (Newton's shell theorem: outside
//! a mass distribution, gravity only depends on its total mass and center,
//! not its exact shape) rather than a full N-body field integration — the
//! latter would mean resampling the whole R,A,B volume at every simulation
//! step, both unnecessary here and far too slow to do 100+ times per clip.
//!
//! The orbit starts circular — exactly enough tangential velocity to stay
//! bound, no per-clip tuning needed for that part — and decays inward over
//! the clip via a small velocity-damping term each step, standing in for
//! atmospheric drag: the "free energy lowers progressively so it falls more
//! and more toward the mass" request. Every frame's `Slice` faces the
//! (fixed, precomputed once) center of mass via `quat_motion::look_at_basis`.

use rayon::prelude::*;

use crate::formula::{mod_value, ModShape, ModTarget, TimeMod};
use crate::quat_fractal::{quat_escape, PerspectiveCamera, QuatFormula, Slice, TimeAxis};
use crate::quat_motion::{add, cross, dot, look_at_basis, normalize, scale, sub, Vec3};

pub struct ProjectileParams {
    pub formula: QuatFormula,
    /// Which quaternion component is time — see `TimeAxis`. The other
    /// three are what the mass field/orbit both treat as "spatial".
    pub time_axis: TimeAxis,
    /// The time-driven component ramps LINEARLY time_val0 -> time_val1
    /// across the clip, same t=i/n convention as everywhere else — a real
    /// 4th-dimension drift running alongside the 3D orbital physics, not a
    /// frozen background parameter. The mass centroid (and therefore the
    /// whole orbit) is still computed ONCE, from time_val0 only — see
    /// `simulate_projectile`'s doc comment for why that's a deliberate
    /// simplification, not an oversight.
    pub time_val0: f64,
    pub time_val1: f64,
    /// How the time value moves between `time_val0` and `time_val1` — reuses
    /// this project's existing `ModShape` vocabulary (`formula.rs`, already
    /// driving the 2D time-modulation system) rather than inventing a new
    /// one. `Ramp` (the default) is a straight `time_val0 -> time_val1`
    /// sweep — the exact old behavior, byte-for-byte, see `time_value_at`.
    /// Every other shape instead treats `[time_val0, time_val1]` as a
    /// symmetric range: center = midpoint, amplitude = half-width, and
    /// oscillates within it — e.g. `Sine` breathes back and forth rather
    /// than sweeping one-way. See the module doc comment on why this alone
    /// can't break the "solid of revolution" formulas' angular degeneracy —
    /// it turns a frozen frame into a pulsing one, not a structured one.
    pub time_shape: ModShape,
    /// Cycles over the full clip for periodic shapes (ignored by `Ramp`).
    pub time_freq: f64,
    /// Phase offset in turns (0..1) for periodic shapes (ignored by `Ramp`).
    pub time_phase: f64,
    pub domain_extent: f64,
    pub max_iter: u32,
    pub bailout: f64,
    /// Grid resolution (per axis) for the one-time mass-centroid precompute.
    pub mass_samples: u32,
    /// Exponent on escape_time when weighting mass — higher concentrates
    /// the effective mass more tightly around genuinely bounded points
    /// rather than the slowly-escaping halo around them.
    pub mass_power: f64,
    /// Defines the initial orbital plane (perpendicular to this axis) —
    /// same convention as `OrbitParams::axis`. Central gravity keeps the
    /// whole trajectory confined to that one fixed plane forever (position
    /// and velocity both start, and stay, perpendicular to `axis`), so
    /// `dot(pos - center, axis)` is exactly 0 for every frame. If `axis` is
    /// itself a coordinate basis vector (e.g. `(0,1,0)`), that dot product
    /// degenerates to a single raw spatial coordinate (A, in that example) —
    /// which then stays EXACTLY frozen at the center's value for the entire
    /// clip, a real dullness bug Carl caught by eye. Callers should default
    /// this to something with no component equal to 0 OR 1 — a `1` still
    /// reads as a suspiciously round, near-axis-aligned choice even though
    /// it technically avoids exact freezing (Carl's own correction, after
    /// the first fix used `(1,0.6,0.3)`) — so every raw coordinate
    /// genuinely moves without one dominating. The CLI's default is
    /// `(0.65,0.42,0.83)`.
    pub axis: Vec3,
    pub start_radius: f64,
    /// Effective GM (gravitational parameter) — an artistic strength knob,
    /// not literally derived from the (unitless) mass-weight sum, since
    /// that sum has no natural physical unit to convert from.
    pub mu: f64,
    /// Fraction of velocity lost per second of SIMULATED time (not
    /// wall-clock/video time) — this is what makes the orbit decay inward.
    pub damping_per_sec: f64,
    /// Simulated seconds advanced per rendered frame.
    pub sim_dt: f64,
    /// Softens the 1/r² singularity as the projectile falls close to the
    /// mass; without this, a close pass produces an unplayable whiplash.
    pub softening: f64,
    pub zoom: f64,
    /// `None` (default) = orthographic, unchanged from before this existed.
    /// `Some(cam)` renders each frame's `Slice` through a genuine
    /// perspective camera instead — see `PerspectiveCamera`'s doc comment
    /// for why `cam.tilt_u`/`tilt_v` (not just `cam.distance`) are what
    /// actually matter for turning a centered circle into a visible
    /// ellipse for the spherically-symmetric-under-`TimeAxis::R` formulas.
    pub perspective: Option<PerspectiveCamera>,
}

/// One-time precompute: samples a dense grid over the 3 SPATIAL axes (see
/// `time_axis`) at fixed time value, weights each point by
/// `escape_time^mass_power`, and returns the weighted centroid (in the same
/// spatial coordinates, not quaternion components). This is the ONLY place
/// the fractal's shape enters the simulation — everything after this
/// treats it as a single point mass.
#[allow(clippy::too_many_arguments)]
pub fn compute_mass_centroid(
    formula: QuatFormula,
    time_axis: TimeAxis,
    time_val: f64,
    domain_extent: f64,
    samples: u32,
    max_iter: u32,
    bailout_sq: f64,
    mass_power: f64,
) -> Vec3 {
    let n = samples.max(2) as usize;
    let half = domain_extent;
    let denom = (n as f64 - 1.0).max(1.0);
    let total = n * n * n;
    let (wsum, wx, wy, wz) = (0..total)
        .into_par_iter()
        .map(|idx| {
            let ix = idx % n;
            let iy = (idx / n) % n;
            let iz = idx / (n * n);
            let x = -half + 2.0 * half * (ix as f64) / denom;
            let y = -half + 2.0 * half * (iy as f64) / denom;
            let z = -half + 2.0 * half * (iz as f64) / denom;
            let et = quat_escape(formula, time_axis.assemble((x, y, z), time_val), max_iter, bailout_sq) as f64;
            let w = et.max(0.0).powf(mass_power);
            (w, w * x, w * y, w * z)
        })
        .reduce(
            || (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2, x.3 + y.3),
        );
    if wsum < 1e-12 {
        (0.0, 0.0, 0.0)
    } else {
        (wx / wsum, wy / wsum, wz / wsum)
    }
}

/// Integrates the projectile forward one step per requested frame
/// (semi-implicit Euler with velocity damping and softened gravity) and
/// records the look-at `Slice` for each step, with the time-driven
/// component ramping time_val0->time_val1 in parallel. Precomputed all at
/// once (rather than sampled at an arbitrary `t`) because the physics is
/// inherently sequential — see `SliceMotion::Trajectory`.
///
/// The mass centroid is computed ONCE, from the time value at `t=0`
/// (`time_val0` itself for the default `Ramp` shape; the range's center for
/// any oscillating shape, since that's what frame 0 actually renders — see
/// `time_value_at`), even though the time value goes on to drift across the
/// clip — a deliberate
/// simplification, not an oversight: recomputing it every frame would mean
/// the "gravity" a formula whose escape-time field genuinely depends on the
/// time axis (e.g. Celtic/Mandelbrot under `TimeAxis::R`, or any formula
/// under `TimeAxis::C`) is subtly wrong for every frame after the first —
/// but the alternative (an orbit whose own center silently drifts) is a
/// stranger, harder-to-reason-about motion than "orbit a fixed point while
/// a separate, independent dimension also drifts", which is exactly the
/// same relationship `OrbitParams`/`PanZoomParams` already have with their
/// own c0->c1 ramps: the ramp never feeds back into the geometric motion.
/// The time-driven component's value at `t ∈ [0,1)`. `Ramp` reproduces the
/// original `time_val0 + (time_val1-time_val0)*t` formula exactly (kept as
/// its own branch, not routed through the general shape path, specifically
/// so this stays byte-identical for existing callers/tests). Every other
/// shape treats `[time_val0, time_val1]` as a center/amplitude pair driving
/// `formula::mod_value` — see `ProjectileParams::time_shape`.
fn time_value_at(params: &ProjectileParams, t: f64) -> f64 {
    match params.time_shape {
        ModShape::Ramp => params.time_val0 + (params.time_val1 - params.time_val0) * t,
        shape => {
            let center = (params.time_val0 + params.time_val1) / 2.0;
            let amp = (params.time_val1 - params.time_val0) / 2.0;
            let m = TimeMod {
                target: ModTarget::Bailout, // unused by mod_value — same placeholder formula.rs's own blend_fraction uses for a "shape only" TimeMod
                shape,
                amp: amp as f32,
                freq: params.time_freq as f32,
                phase: params.time_phase as f32,
            };
            center + mod_value(&m, t as f32) as f64
        }
    }
}

pub fn simulate_projectile(params: &ProjectileParams, n_frames: u32) -> Vec<(Slice, f64)> {
    let bailout_sq = params.bailout * params.bailout;
    let center = compute_mass_centroid(
        params.formula,
        params.time_axis,
        time_value_at(params, 0.0),
        params.domain_extent,
        params.mass_samples,
        params.max_iter,
        bailout_sq,
        params.mass_power,
    );

    let axis = normalize(params.axis);
    let reference = if dot(axis, (1.0, 0.0, 0.0)).abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    let e1 = normalize(add(reference, scale(axis, -dot(reference, axis))));
    let e2 = normalize(cross(axis, e1));

    let r0 = params.start_radius.max(1e-6);
    let mut pos = add(center, scale(e1, r0));
    // Circular orbital velocity: bound (well under escape velocity) by
    // construction, no per-scene tuning needed to avoid immediately flying
    // off — this is "just enough force for it not to exit the system."
    let v_circ = (params.mu / r0).max(0.0).sqrt();
    let mut vel = scale(e2, v_circ);

    let n = n_frames.max(1);
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let t = i as f64 / n as f64;
        let time_val = time_value_at(params, t);
        let forward = normalize(sub(center, pos));
        let (basis_u, basis_v) = look_at_basis(forward, axis);
        out.push((
            Slice { origin: pos, basis_u, basis_v, zoom: params.zoom, camera: params.perspective },
            time_val,
        ));

        let r_vec = sub(center, pos);
        let r2 = dot(r_vec, r_vec) + params.softening * params.softening;
        let r = r2.sqrt();
        let accel = scale(r_vec, params.mu / (r2 * r));
        vel = add(vel, scale(accel, params.sim_dt));
        let damp = (1.0 - params.damping_per_sec).clamp(0.0, 1.0).powf(params.sim_dt);
        vel = scale(vel, damp);
        pos = add(pos, scale(vel, params.sim_dt));
    }
    out
}

/// Cheap per-frame stats for judging whether a trajectory stays visually
/// alive across its whole length — the same "is this frame worth looking
/// at" question `time_explore.rs`'s gates already answer for the 2D
/// time-formula system (blank/dead/static), applied here to a *simulated*
/// trajectory before any real rendering happens, so a long clip's pacing
/// can be checked and retuned in seconds instead of by rendering it.
#[derive(Copy, Clone, Debug)]
pub struct FrameProbe {
    /// Fraction of probe pixels that escaped almost immediately (empty/background).
    pub escaped_frac: f32,
    /// Fraction of probe pixels that never escaped (flat interior, no boundary detail).
    pub interior_frac: f32,
    /// Mean normalized (escape_time/max_iter) value — a coarse brightness/detail proxy.
    pub mean_norm: f32,
}

/// A run of consecutive frames sharing the same "boring" classification —
/// what actually matters for a long clip isn't a few isolated dead frames
/// (unremarkable, one blink) but a SUSTAINED stretch of them.
#[derive(Clone, Debug)]
pub struct DeadRun {
    pub start_frame: usize,
    pub len: usize,
    pub kind: &'static str, // "blank" | "flat" | "static"
}

pub struct TrajectoryReport {
    pub probes: Vec<FrameProbe>,
    /// Frame-to-frame mean absolute change in normalized escape time —
    /// near-zero means the picture barely moved between frames.
    pub deltas: Vec<f32>,
    pub dead_runs: Vec<DeadRun>,
    pub longest_dead_run: usize,
    pub blank_frames: usize,
    pub flat_frames: usize,
    pub static_frames: usize,
}

impl TrajectoryReport {
    /// A compact one-line-per-bucket timeline: `.` alive, `_` blank
    /// (nothing there yet), `#` flat (solid interior, no detail), `=`
    /// static (not blank/flat, but barely changing frame to frame).
    /// Buckets take the WORST frame in their span, so a short dead patch
    /// isn't averaged away and hidden.
    pub fn ascii_timeline(&self, buckets: usize) -> String {
        let n = self.probes.len();
        if n == 0 || buckets == 0 {
            return String::new();
        }
        let mut out = String::with_capacity(buckets);
        for b in 0..buckets {
            let lo = b * n / buckets;
            let hi = ((b + 1) * n / buckets).max(lo + 1).min(n);
            let mut ch = '.';
            for i in lo..hi {
                let p = &self.probes[i];
                let d = self.deltas.get(i).copied().unwrap_or(1.0);
                let this = if p.escaped_frac > 0.97 {
                    '_'
                } else if p.interior_frac > 0.97 {
                    '#'
                } else if d < 0.01 {
                    '='
                } else {
                    '.'
                };
                // worst-of: '_' and '#' outrank '=' outrank '.'
                let rank = |c: char| match c {
                    '_' | '#' => 2,
                    '=' => 1,
                    _ => 0,
                };
                if rank(this) > rank(ch) {
                    ch = this;
                }
            }
            out.push(ch);
        }
        out
    }
}

/// Runs the physics sim, then renders a tiny (`probe_res`^2) escape-time
/// buffer per frame — cheap enough to check a multi-thousand-frame
/// trajectory in well under a second — and classifies each frame plus
/// looks for sustained "boring" runs.
///
/// `delta_window` controls how far back "frame-to-frame" change is
/// measured — comparing to the IMMEDIATELY PREVIOUS frame (window=1) is the
/// wrong question for a long, high-frame-rate clip: a smooth slow-motion
/// pass naturally moves very little in a single frame, so a window of 1
/// flags genuinely fine motion as "static". Pass a window matched to a
/// perceptible timescale instead — roughly your target fps (so it compares
/// against "about a second ago"), not frame count.
pub fn probe_trajectory(params: &ProjectileParams, n_frames: u32, probe_res: u32, delta_window: usize) -> TrajectoryReport {
    let bailout_sq = params.bailout * params.bailout;
    let trajectory = simulate_projectile(params, n_frames);
    let window = delta_window.max(1);

    // One pass: render each frame's tiny probe buffer once, derive its
    // stats, and diff it against the buffer from `window` frames back (a
    // small ring buffer, not the whole clip — probe_res^2 * window is
    // trivial even for a long clip).
    let max = params.max_iter as f32;
    let mut probes = Vec::with_capacity(trajectory.len());
    let mut deltas = Vec::with_capacity(trajectory.len());
    let mut history: std::collections::VecDeque<Vec<f32>> = std::collections::VecDeque::with_capacity(window + 1);
    for (slice, c) in &trajectory {
        let et = crate::quat_fractal::render_quat_frame(
            params.formula, params.time_axis, slice, *c, probe_res, probe_res, params.max_iter, bailout_sq,
        );
        let n = et.len().max(1) as f32;
        let escaped = et.iter().filter(|&&v| v < max * 0.05).count() as f32 / n;
        let interior = et.iter().filter(|&&v| v >= max * 0.999).count() as f32 / n;
        let mean_norm = et.iter().sum::<f32>() / n / max;
        probes.push(FrameProbe { escaped_frac: escaped, interior_frac: interior, mean_norm });

        let d = if history.len() >= window {
            let p = &history[0]; // exactly `window` frames back
            et.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / n / max
        } else {
            1.0 // not enough history yet: treat as "changed" rather than falsely static
        };
        deltas.push(d);
        history.push_back(et);
        if history.len() > window {
            history.pop_front();
        }
    }

    let mut dead_runs = Vec::new();
    let mut run_start = 0usize;
    let mut run_kind: Option<&'static str> = None;
    let (mut blank_frames, mut flat_frames, mut static_frames) = (0usize, 0usize, 0usize);
    let classify = |p: &FrameProbe, d: f32| -> Option<&'static str> {
        if p.escaped_frac > 0.97 {
            Some("blank")
        } else if p.interior_frac > 0.97 {
            Some("flat")
        } else if d < 0.01 {
            Some("static")
        } else {
            None
        }
    };
    for (i, p) in probes.iter().enumerate() {
        let kind = classify(p, deltas[i]);
        match kind {
            Some("blank") => blank_frames += 1,
            Some("flat") => flat_frames += 1,
            Some("static") => static_frames += 1,
            _ => {}
        }
        if kind != run_kind {
            if let Some(k) = run_kind {
                dead_runs.push(DeadRun { start_frame: run_start, len: i - run_start, kind: k });
            }
            run_start = i;
            run_kind = kind;
        }
    }
    if let Some(k) = run_kind {
        dead_runs.push(DeadRun { start_frame: run_start, len: probes.len() - run_start, kind: k });
    }
    let longest_dead_run = dead_runs.iter().map(|r| r.len).max().unwrap_or(0);

    TrajectoryReport { probes, deltas, dead_runs, longest_dead_run, blank_frames, flat_frames, static_frames }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(u: Vec3, v: Vec3, eps: f64) -> bool {
        (u.0 - v.0).abs() < eps && (u.1 - v.1).abs() < eps && (u.2 - v.2).abs() < eps
    }

    #[test]
    fn mass_centroid_is_symmetric_in_a_b_for_mandelbrot() {
        // Mandelbrot's escape time depends only on R and rho=sqrt(A^2+B^2+C^2)
        // (the "solid of revolution" finding — see quat_fractal module
        // docs), so over a domain cube centered at the origin, the mass
        // field is symmetric under A->-A and B->-B independently. The
        // centroid's A,B components must be ~0 regardless of where the
        // mass concentrates along R.
        let (r, a, b) = compute_mass_centroid(QuatFormula::Mandelbrot, TimeAxis::C, 0.0, 1.6, 40, 60, 16.0, 3.0);
        assert!(a.abs() < 0.05, "A={a} should be ~0");
        assert!(b.abs() < 0.05, "B={b} should be ~0");
        // Sanity: the cardioid body sits at negative R, so the centroid
        // shouldn't be exactly at the domain center either.
        assert!(r.abs() > 1e-6, "R={r} should not be exactly 0");
    }

    #[test]
    fn mass_centroid_stays_within_domain_bounds() {
        for f in QuatFormula::ALL {
            let (r, a, b) = compute_mass_centroid(f, TimeAxis::C, 0.0, 1.6, 24, 40, 16.0, 3.0);
            assert!(r.abs() <= 1.6 + 1e-9, "{}: R={r} out of bounds", f.name());
            assert!(a.abs() <= 1.6 + 1e-9, "{}: A={a} out of bounds", f.name());
            assert!(b.abs() <= 1.6 + 1e-9, "{}: B={b} out of bounds", f.name());
        }
    }

    fn base_params() -> ProjectileParams {
        ProjectileParams {
            formula: QuatFormula::Mandelbrot,
            time_axis: TimeAxis::C,
            time_val0: 0.0,
            time_val1: 0.0,
            time_shape: ModShape::Ramp,
            time_freq: 1.0,
            time_phase: 0.0,
            domain_extent: 1.6,
            max_iter: 40,
            bailout: 4.0,
            mass_samples: 24,
            mass_power: 3.0,
            axis: (0.0, 1.0, 0.0),
            start_radius: 2.5,
            mu: 1.0,
            damping_per_sec: 0.0,
            sim_dt: 0.05,
            softening: 0.05,
            zoom: 1.5,
            perspective: None,
        }
    }

    #[test]
    fn trajectory_has_the_requested_frame_count() {
        let out = simulate_projectile(&base_params(), 37);
        assert_eq!(out.len(), 37);
    }

    #[test]
    fn trajectory_is_finite_and_basis_stays_orthonormal() {
        let out = simulate_projectile(&base_params(), 50);
        for (slice, c) in &out {
            assert!(c.is_finite());
            for v in [slice.origin, slice.basis_u, slice.basis_v] {
                assert!(v.0.is_finite() && v.1.is_finite() && v.2.is_finite());
            }
            let nu = dot(slice.basis_u, slice.basis_u).sqrt();
            let nv = dot(slice.basis_v, slice.basis_v).sqrt();
            assert!((nu - 1.0).abs() < 1e-6, "|basis_u|={nu}");
            assert!((nv - 1.0).abs() < 1e-6, "|basis_v|={nv}");
            assert!(dot(slice.basis_u, slice.basis_v).abs() < 1e-6, "basis_u·basis_v not ~0");
        }
    }

    #[test]
    fn without_damping_the_orbit_stays_roughly_bound_over_a_short_window() {
        let mut p = base_params();
        p.damping_per_sec = 0.0;
        let out = simulate_projectile(&p, 40);
        let center = compute_mass_centroid(p.formula, p.time_axis, p.time_val0, p.domain_extent, p.mass_samples, p.max_iter, p.bailout * p.bailout, p.mass_power);
        for (slice, _) in &out {
            let d = dot(sub(slice.origin, center), sub(slice.origin, center)).sqrt();
            assert!(d < p.start_radius * 3.0, "orbit drifted to d={d}, expected to stay roughly bound near r0={}", p.start_radius);
        }
    }

    #[test]
    fn with_damping_the_projectile_falls_inward() {
        let mut p = base_params();
        p.damping_per_sec = 0.3;
        let out = simulate_projectile(&p, 60);
        let center = compute_mass_centroid(p.formula, p.time_axis, p.time_val0, p.domain_extent, p.mass_samples, p.max_iter, p.bailout * p.bailout, p.mass_power);
        let dist = |o: Vec3| dot(sub(o, center), sub(o, center)).sqrt();
        let d_first = dist(out.first().unwrap().0.origin);
        let d_last = dist(out.last().unwrap().0.origin);
        assert!(d_last < d_first, "expected inward decay: d_first={d_first} d_last={d_last}");
    }

    #[test]
    fn time_value_ramps_linearly_across_the_clip_independent_of_the_orbit() {
        let mut p = base_params();
        p.time_val0 = -2.0;
        p.time_val1 = 6.0;
        let out = simulate_projectile(&p, 40);
        let (_, t_first) = out[0];
        let (_, t_quarter) = out[10];
        let (_, t_last) = out[39];
        assert!((t_first - (-2.0)).abs() < 1e-9, "t_first={t_first}");
        // frame 10 of 40 => t=0.25 => -2.0 + 0.25*8.0 = 0.0
        assert!((t_quarter - 0.0).abs() < 1e-9, "t_quarter={t_quarter}");
        // frame 39 of 40 => t=0.975, not quite time_val1 (matches the
        // t=i/n, never-reaches-1.0 convention used everywhere else)
        assert!(t_last < 6.0 && t_last > 5.5, "t_last={t_last}");
    }

    #[test]
    fn time_ramp_does_not_change_the_orbit_shape() {
        // The ramp is an independent side-channel (matches OrbitParams/
        // PanZoomParams' own c0->c1 convention) — it must not perturb the
        // spatial trajectory at all, only the paired time value.
        let mut fixed = base_params();
        fixed.time_val0 = 1.0;
        fixed.time_val1 = 1.0;
        let mut ramped = base_params();
        ramped.time_val0 = 1.0;
        ramped.time_val1 = 9.0;
        let out_fixed = simulate_projectile(&fixed, 30);
        let out_ramped = simulate_projectile(&ramped, 30);
        for i in 0..30 {
            let (sf, _) = out_fixed[i];
            let (sr, _) = out_ramped[i];
            assert!(approx(sf.origin, sr.origin, 1e-12), "frame {i}: origins diverged {:?} vs {:?}", sf.origin, sr.origin);
        }
    }

    #[test]
    fn probe_trajectory_detects_the_blank_to_alive_transition() {
        // Starting far outside the structure (start_radius=2.5, well beyond
        // where bulb has any structure) means the first frames MUST probe
        // as blank; falling inward with damping must eventually leave that
        // state. If this ever stops being true, the probe's own
        // classification logic broke, not just the physics.
        let mut p = base_params();
        p.formula = QuatFormula::Bulb;
        p.start_radius = 2.5;
        p.mu = 25.0;
        p.damping_per_sec = 0.15;
        let report = probe_trajectory(&p, 80, 24, 1);
        assert_eq!(report.probes.len(), 80);
        assert_eq!(report.deltas.len(), 80);
        assert!(report.probes[0].escaped_frac > 0.97, "first frame should be blank");
        let later_alive = report.probes[60..].iter().any(|p| p.escaped_frac < 0.9);
        assert!(later_alive, "expected the trajectory to reach visible structure by frame 60+");
    }

    #[test]
    fn sine_shape_oscillates_around_the_center_instead_of_sweeping_one_way() {
        let mut p = base_params();
        p.time_val0 = -1.0;
        p.time_val1 = 3.0; // center=1.0, amp=2.0
        p.time_shape = ModShape::Sine;
        assert!((time_value_at(&p, 0.0) - 1.0).abs() < 1e-6, "t=0 should sit at the center (sin(0)=0)");
        // Quarter cycle in: sin(pi/2)=1 -> exactly the peak (center+amp).
        assert!((time_value_at(&p, 0.25) - 3.0).abs() < 1e-5, "t=0.25 should reach time_val1 (the peak)");
        // The value must never leave [time_val0, time_val1] for a sine.
        for i in 0..40 {
            let t = i as f64 / 40.0;
            let v = time_value_at(&p, t);
            assert!(v >= p.time_val0 - 1e-6 && v <= p.time_val1 + 1e-6, "t={t} v={v} left [{}, {}]", p.time_val0, p.time_val1);
        }
    }

    #[test]
    fn ramp_shape_is_unchanged_from_the_original_linear_formula() {
        // Regression guard: Ramp must stay byte-identical to the pre-ModShape
        // behavior (time_val0 + (time_val1-time_val0)*t), not get routed
        // through the generic center/amplitude path other shapes use.
        let mut p = base_params();
        p.time_val0 = -2.0;
        p.time_val1 = 6.0;
        for i in 0..10 {
            let t = i as f64 / 10.0;
            let expected = -2.0 + 8.0 * t;
            assert!((time_value_at(&p, t) - expected).abs() < 1e-9, "t={t}");
        }
    }

    #[test]
    fn mass_centroid_uses_the_frame_zero_value_not_time_val0_for_oscillating_shapes() {
        // For Sine, frame 0 actually renders the CENTER (sin(0)=0), not
        // time_val0 — the centroid must be computed at what's actually shown
        // first, or the "gravity" is centered on a scene the clip never
        // starts at.
        let mut p = base_params();
        p.formula = QuatFormula::Bulb; // a formula with real directional structure
        p.time_val0 = -1.0;
        p.time_val1 = 1.0; // center=0.0
        p.time_shape = ModShape::Sine;
        let center_via_helper = compute_mass_centroid(
            p.formula, p.time_axis, time_value_at(&p, 0.0), p.domain_extent, p.mass_samples, p.max_iter, p.bailout * p.bailout, p.mass_power,
        );
        let center_at_zero = compute_mass_centroid(
            p.formula, p.time_axis, 0.0, p.domain_extent, p.mass_samples, p.max_iter, p.bailout * p.bailout, p.mass_power,
        );
        assert!(approx(center_via_helper, center_at_zero, 1e-9));
    }

    #[test]
    fn perspective_camera_is_threaded_into_every_frame_slice() {
        let mut p = base_params();
        p.perspective = Some(PerspectiveCamera { tilt_u: 0.5, tilt_v: 0.2, distance: 1.5 });
        let out = simulate_projectile(&p, 10);
        for (slice, _) in &out {
            let cam = slice.camera.expect("expected a perspective camera on every frame");
            assert!((cam.tilt_u - 0.5).abs() < 1e-12);
            assert!((cam.tilt_v - 0.2).abs() < 1e-12);
            assert!((cam.distance - 1.5).abs() < 1e-12);
        }
    }

    #[test]
    fn axis_aligned_with_a_basis_vector_freezes_that_coordinate() {
        // Root-cause proof for the dullness Carl reported: a central force
        // keeps position and velocity confined to the plane perpendicular to
        // `axis` forever, so dot(pos-center, axis) == 0 at every frame. When
        // `axis` is itself a raw coordinate basis vector, that dot product
        // IS the raw coordinate — it never moves from the center's value.
        let mut p = base_params();
        p.axis = (0.0, 1.0, 0.0); // the old default
        p.damping_per_sec = 0.0;
        let out = simulate_projectile(&p, 60);
        let y0 = out[0].0.origin.1;
        for (slice, _) in &out {
            assert!((slice.origin.1 - y0).abs() < 1e-9, "y={} should stay exactly frozen at {y0} with a basis-aligned axis", slice.origin.1);
        }
    }

    #[test]
    fn asymmetric_axis_does_not_freeze_any_single_spatial_coordinate() {
        let mut p = base_params();
        p.axis = (0.65, 0.42, 0.83); // the CLI default — no component is 0 or 1
        p.damping_per_sec = 0.0;
        let out = simulate_projectile(&p, 60);
        let (x0, y0, z0) = out[0].0.origin;
        let varies = |get: fn(Vec3) -> f64, v0: f64| out.iter().any(|(s, _)| (get(s.origin) - v0).abs() > 1e-6);
        assert!(varies(|v| v.0, x0), "x should vary across the clip");
        assert!(varies(|v| v.1, y0), "y should vary across the clip");
        assert!(varies(|v| v.2, z0), "z should vary across the clip");
    }

    #[test]
    fn ascii_timeline_has_the_requested_length() {
        let p = base_params();
        let report = probe_trajectory(&p, 50, 16, 1);
        let line = report.ascii_timeline(20);
        assert_eq!(line.chars().count(), 20);
    }
}
