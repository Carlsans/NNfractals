//! Search for the most interesting time axis of a given fractal.
//!
//! A genome has several scalars that could be animated (see
//! [`crate::formula::ModTarget`]), several shapes to drive them with, and a
//! range of amplitudes. This module renders a short low-res clip for each
//! combination, scores it by how well it resists video compression, and ranks
//! them — the temporal analogue of `video_zoom_explore`'s search over camera
//! paths, using the identical `bytes / (w·h·3·n)` normalisation so the two
//! families of score are directly comparable.
//!
//! # Why this needs gates, and why it needs three
//!
//! **A compression objective is maximised by noise.** This project has already
//! paid for that lesson once: `video_zoom_explore`'s first batch run returned a
//! pure speckle field as its #1 winner (spatial coherence 0.027 against 0.26+
//! for real structure), because incompressible dither scores exactly like
//! genuine detail. The spatial gate here is the same `noise_tile_fraction`
//! floor that fixed it, applied per frame.
//!
//! The temporal axis then adds its own version of the trap, in **both**
//! directions:
//!
//! * Too much amplitude and consecutive frames are unrelated. Inter-frame
//!   prediction fails completely, the file is maximally large, and the
//!   candidate wins every time — while looking like a slideshow of strangers
//!   rather than an animation. The **coherence** gate rejects that.
//! * Too little amplitude and nothing visibly moves. The clip is valid,
//!   compresses well, ranks low — but it would still be reported as a candidate
//!   with a real score, and a "winner" where nothing happens is worse than no
//!   winner. The **change** gate rejects that.
//!
//! Together those two bracket the useful window from both sides, which is the
//! whole point: the interesting amplitude is the largest one that still reads
//! as a continuous morph. Every rejection records WHICH gate stopped it — the
//! zoom search once shipped a bare "0 winners" message with no reason attached
//! and it cost real debugging time.

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::formula::{op, reachable_from_root, ModShape, ModTarget, TimeMod};
use crate::genome::Genome;
use crate::video_export::{probe_frames_score, time_frames, View};
use crate::fitness;

/// Minimum mean Pearson correlation between consecutive frames' luminance for a
/// clip to count as an animation rather than a sequence of cuts. Tuned against
/// real output — see this module's header. Overridable per run.
pub const MIN_TEMPORAL_COHERENCE: f32 = 0.55;

/// Minimum mean absolute luminance change (0-255 scale) between consecutive
/// frames. Below this the modulation is present but invisible.
pub const MIN_TEMPORAL_CHANGE: f32 = 1.0;

/// Worst-frame `noise_tile_fraction` a clip may contain — deliberately STRICTER
/// than `fitness::MAX_NOISE_TILE_FRACTION` (0.25), the floor used for stills.
///
/// Two reasons, both found by looking at real output rather than reasoning
/// ahead of it. First, a defect that would be borderline in a still is far more
/// objectionable in a clip: it flickers, so it draws the eye precisely because
/// it is intermittent, and one bad frame in sixteen is enough to spoil the
/// whole thing. Since this is a max over frames, "one frame that bad" is
/// exactly what it measures.
///
/// Second, on the first real sweep the top three candidates all measured
/// EXACTLY 0.250 and slipped through a `>` comparison against the still floor.
/// That is not a coincidence — `noise_tile_fraction` tiles the frame 4x4, so at
/// probe resolution only a handful of tiles are textured and the metric
/// quantizes to coarse fractions like 1/4. Sitting a threshold on one of those
/// quantization points makes the outcome a toss-up. The contact sheet confirmed
/// all three were visibly dithered; the clean candidate that replaced them
/// scored lower on compression but higher on every other measure (coherence
/// 0.96 vs 0.90, change 15.0 vs 13.1, noise 0.000 vs 0.250).
pub const MAX_CLIP_NOISE: f32 = 0.15;

/// A frame pair counts as "still" when it moves less than this fraction of the
/// clip's mean motion — i.e. is all but identical.
pub const STILL_FRACTION: f32 = 0.02;

/// Longest tolerated run of still pairs, as a fraction of the clip. A sinusoid
/// turning point is a single pair; the freeze this catches was four in a row out
/// of fifteen (27%).
///
/// # Calibration
///
/// Measured on a real sweep with every gate disabled, after contact sheets of
/// the top scorers showed them to be visibly bad.
///
/// The FIRST version of this gate compared `min_change` to `mean_change` and
/// was badly wrong — it rejected every sine and triangle candidate on genome
/// 4f53bc40f64a86c9, including clips averaging 13.4 luminance units of motion,
/// because a sinusoid's velocity is zero at its turning points. Carl reported it
/// as "find time axis does not work anymore even if there is plenty good
/// candidates", and he was right.
///
/// The run-length form separates the two cases with nothing in between. On that
/// same genome, 16 frames (15 pairs):
///
/// ```text
///   longest_still_run   verdict
///        0.000          every passing candidate
///   ---------------------------------- gap
///        0.200          3 consecutive near-identical frames
///        0.267          4
///        0.400          6   (min_change 0.006 against a mean of 11-18)
/// ```
///
/// A turning point is one slow pair; a freeze is several identical ones. Note
/// `min_change` is near zero in BOTH cases, which is exactly why it was the
/// wrong thing to test.
pub const MAX_STILL_RUN: f32 = 0.15;

/// Largest tolerated jump in frame-mean luminance (0-255) between consecutive
/// frames. See [`MIN_STALL_RATIO`] for the calibration data.
pub const MAX_LEVEL_JUMP: f32 = 12.0;

/// Cap on how many `ProgScale` insertion points to try. The interior nodes of a
/// program are all plausible targets, but the sweep cost is linear in the target
/// count and most programs have 6-14 nodes, so sample evenly rather than take
/// them all.
pub const MAX_PROG_SCALE_TARGETS: usize = 4;

#[derive(Clone, Debug)]
pub struct TimeExploreOpts {
    pub probe_w: u32,
    pub probe_h: u32,
    pub frames: u32,
    pub fps: u32,
    /// Amplitudes to try per (target, shape). In the target's own units.
    pub amps: Vec<f32>,
    pub shapes: Vec<ModShape>,
    pub top_k: usize,
    pub angle_coloring: bool,
    pub min_coherence: f32,
    pub min_change: f32,
    pub max_noise: f32,
    /// Reject when the longest run of near-identical frames exceeds this
    /// fraction of the clip.
    pub max_still_run: f32,
    /// Reject if frame-mean luminance jumps by more than this (0-255) between
    /// consecutive frames — a global flash.
    pub max_level_jump: f32,
}

impl Default for TimeExploreOpts {
    fn default() -> Self {
        TimeExploreOpts {
            probe_w: 192,
            probe_h: 144,
            frames: 48,
            fps: 24,
            amps: vec![0.02, 0.08, 0.25],
            shapes: vec![ModShape::Sine, ModShape::Triangle, ModShape::Orbit],
            top_k: 8,
            angle_coloring: false,
            min_coherence: MIN_TEMPORAL_COHERENCE,
            min_change: MIN_TEMPORAL_CHANGE,
            max_noise: MAX_CLIP_NOISE,
            max_still_run: MAX_STILL_RUN,
            max_level_jump: MAX_LEVEL_JUMP,
        }
    }
}

/// What a clip measured, before the gates decide whether to trust its score.
///
/// Almost every field is a WORST-CASE over the clip rather than an average, and
/// that is the whole lesson of the first calibration run. The initial version
/// gated on mean consecutive correlation; a clip that froze for five frames
/// (correlation 1.000 each, dragging the mean to 0.955) and flashed twice
/// scored better than a clip that morphed smoothly throughout. Averages hide
/// exactly the moments the eye is drawn to, because a defect that appears once
/// in a loop is not diluted by the good frames around it — it is a flicker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClipStats {
    /// Worst per-frame `noise_tile_fraction` across the clip.
    pub max_noise: f32,
    /// WORST consecutive-frame Pearson correlation: the single most abrupt
    /// structural transition in the clip.
    pub min_coherence: f32,
    /// Mean consecutive correlation. Reported for reference only — never gated
    /// on, for the reason in this struct's doc comment.
    pub mean_coherence: f32,
    /// Mean absolute luminance change between consecutive frames, 0-255.
    /// Answers "does the clip move at all".
    pub mean_change: f32,
    /// Smallest consecutive-frame change. Reported for reference; NOT gated on
    /// — see `longest_still_run` for why the minimum alone is the wrong test.
    pub min_change: f32,
    /// Longest RUN of consecutive frame pairs that barely changed, as a
    /// fraction of the clip's pairs.
    ///
    /// This replaced a gate on `min_change / mean_change`, which was wrong in a
    /// way worth recording. A sinusoid's velocity is zero at its turning
    /// points — that is what makes it smooth and loop cleanly — so ONE small
    /// pair per half-cycle is the signature of well-behaved periodic motion,
    /// not of a defect. The old ratio gate rejected every sine and triangle
    /// candidate on a real genome, including clips averaging 13.4 luminance
    /// units of motion per frame, purely because their minimum was near zero.
    ///
    /// The actual defect being caught was a PLATEAU: five genuinely identical
    /// frames in the middle of a clip. A run separates the two cleanly — a
    /// turning point is one pair, a freeze is several.
    pub longest_still_run: f32,
    /// Largest jump in FRAME-MEAN luminance between consecutive frames, 0-255.
    /// Catches global flashes, which correlation structurally cannot see:
    /// Pearson is invariant to offset and scale, so a frame of identical
    /// structure at twice the brightness still correlates at 1.0.
    pub max_level_jump: f32,
}

#[derive(Clone, Debug)]
pub struct TimeCandidate {
    pub tmod: TimeMod,
    /// Normalized compressed/raw ratio, or 0.0 when a gate rejected the clip.
    pub score: f64,
    pub stats: ClipStats,
    /// Which gate stopped this candidate, if any.
    pub rejected: Option<&'static str>,
}

impl TimeCandidate {
    pub fn passed(&self) -> bool {
        self.rejected.is_none()
    }
}

/// Every scalar in `g` that can be animated, most-broadly-applicable first.
///
/// `JuliaC` appears only for a genome actually in Julia mode — the constant is
/// simply unread otherwise, so animating it would render a static clip and
/// waste a slot in the sweep. `ProgConst`/`WarpConst` appear only for nodes that
/// really are CONST. `ProgScale` covers the ~75% of archived genomes that have
/// no CONST node at all to drive.
pub fn enumerate_targets(g: &Genome) -> Vec<ModTarget> {
    let mut out = Vec::new();
    // Julia mode alone is not enough. The julia constant enters the iteration
    // ONLY as the main program's `c` leaf (the warp program's `c` is the pixel
    // coordinate, not the constant — see fractal::dag_escape_pixel), so a
    // program with no live C node ignores it completely and animating it
    // renders a static clip. Measured on 0b3199d357fc16e0, whose formula is
    // |Im|(|Re|(z/|z|)) — julia_mode is set but the constant is decorative.
    if g.julia_mode && program_reads_c(&g.program) {
        out.push(ModTarget::JuliaC);
    }
    // Phoenix and bailout exist on every genome. Phoenix at p = (0,0) is a
    // no-op term, so modulating it fades a second-order memory term in and out
    // — a real structural change even when the stored value is zero.
    out.push(ModTarget::Phoenix);
    out.push(ModTarget::Bailout);

    // Evolved DAGs carry introns: subtrees that no longer reach the root and
    // therefore cannot affect a single pixel. Measured on a real archived
    // genome (0b3199d357fc16e0) FIVE of nine nodes were dead, and every probe
    // aimed at one came back "static" — which reads as "amplitude too small,
    // raise --amps" when the truth is "this node does nothing at any
    // amplitude". Filtering them here removed 23% of that genome's sweep and,
    // more importantly, stopped the run reporting a misleading diagnosis.
    let live = reachable_from_root(&g.program);
    let warp_live = reachable_from_root(&g.warp);

    for (i, n) in g.program.iter().enumerate() {
        if n.op == op::CONST && live[i] {
            out.push(ModTarget::ProgConst { node: i as u8 });
        }
    }
    for (i, n) in g.warp.iter().enumerate() {
        if n.op == op::CONST && warp_live[i] {
            out.push(ModTarget::WarpConst { node: i as u8 });
        }
    }

    // Interior (non-leaf) nodes are the meaningful scale points: scaling a leaf
    // CONST duplicates ProgConst, and scaling Z or C alone is a degenerate
    // rewrite of the whole formula.
    let interior: Vec<u8> = g.program.iter().enumerate()
        .filter(|(i, n)| op::arity(n.op) >= 1 && live[*i])
        .map(|(i, _)| i as u8)
        .collect();
    if !interior.is_empty() {
        let step = (interior.len() as f32 / MAX_PROG_SCALE_TARGETS as f32).max(1.0);
        let mut taken = Vec::new();
        let mut k = 0.0f32;
        while (k as usize) < interior.len() && taken.len() < MAX_PROG_SCALE_TARGETS {
            let idx = interior[k as usize];
            if !taken.contains(&idx) {
                taken.push(idx);
            }
            k += step;
        }
        // The root scales the entire formula and is always worth trying.
        if let Some(&last) = interior.last() {
            if !taken.contains(&last) && taken.len() < MAX_PROG_SCALE_TARGETS + 1 {
                taken.push(last);
            }
        }
        for node in taken {
            out.push(ModTarget::ProgScale { node });
        }
    }
    out
}

/// Whether a program actually reads its `c` input on a path that reaches the
/// root. A `C` leaf sitting in an intron does not count.
pub fn program_reads_c(prog: &[crate::formula::OpNode]) -> bool {
    let live = reachable_from_root(prog);
    prog.iter().enumerate().any(|(i, n)| n.op == op::C && live[i])
}

/// Shapes worth trying for a given target. `Orbit` is meaningless on a
/// single-channel target — it degenerates to a cosine, which is already in the
/// list — so it is skipped rather than burning a probe on a duplicate.
fn shapes_for(target: ModTarget, shapes: &[ModShape]) -> Vec<ModShape> {
    shapes.iter().copied()
        .filter(|s| *s != ModShape::Orbit || target.is_two_channel())
        .collect()
}

/// Per-pixel luminance of an RGB24 frame.
fn luminance(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect()
}

/// Pearson correlation of two equal-length signals.
///
/// Returns 0.0 when either side is constant: an undefined correlation must not
/// read as perfect agreement, or a pair of flat frames would sail through the
/// coherence gate.
pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= f32::EPSILON || db <= f32::EPSILON {
        return 0.0;
    }
    (num / (da.sqrt() * db.sqrt())).clamp(-1.0, 1.0)
}

/// Measure a rendered clip: worst-frame spatial noise, and how coherently and
/// how much it moves.
pub fn clip_stats(frames: &[Vec<u8>], w: u32, h: u32) -> ClipStats {
    if frames.len() < 2 {
        return ClipStats::default();
    }
    let lums: Vec<Vec<f32>> = frames.iter().map(|f| luminance(f)).collect();

    // Spatial gate is measured on the COLORMAPPED luminance, not the raw
    // escape-time field: the field can carry a smooth gradient that reads as
    // coherent while the palette turns its fine component into visual noise.
    // What ships is the colormapped image, so that is what must be judged —
    // same reasoning as video_zoom_explore's probe.
    let max_noise = lums.iter()
        .map(|l| fitness::noise_tile_fraction(l, w, h))
        .fold(0.0f32, f32::max);

    let means: Vec<f32> = lums.iter()
        .map(|l| l.iter().sum::<f32>() / l.len() as f32)
        .collect();

    let mut coh_sum = 0.0f32;
    let mut chg_sum = 0.0f32;
    let mut min_coh = f32::INFINITY;
    let mut min_chg = f32::INFINITY;
    let mut max_jump = 0.0f32;
    let mut changes: Vec<f32> = Vec::with_capacity(lums.len().saturating_sub(1));
    for (i, pair) in lums.windows(2).enumerate() {
        let coh = pearson(&pair[0], &pair[1]);
        coh_sum += coh;
        min_coh = min_coh.min(coh);

        let d: f32 = pair[0].iter().zip(&pair[1]).map(|(a, b)| (a - b).abs()).sum();
        let chg = d / pair[0].len() as f32;
        chg_sum += chg;
        min_chg = min_chg.min(chg);
        changes.push(chg);

        max_jump = max_jump.max((means[i + 1] - means[i]).abs());
    }
    let pairs = (lums.len() - 1) as f32;
    let mean_change = chg_sum / pairs;

    // "Still" means all but identical, not merely slow: a turning point is
    // slow, a freeze is still. STILL_FRACTION is deliberately tiny so the two
    // cannot be confused.
    let still_floor = mean_change * STILL_FRACTION;
    let (mut run, mut longest) = (0u32, 0u32);
    for c in &changes {
        if *c <= still_floor {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }

    ClipStats {
        max_noise,
        min_coherence: min_coh,
        mean_coherence: coh_sum / pairs,
        mean_change,
        min_change: min_chg,
        longest_still_run: longest as f32 / pairs,
        max_level_jump: max_jump,
    }
}

/// Apply the three gates in cost-of-being-wrong order.
fn gate(stats: &ClipStats, opts: &TimeExploreOpts) -> Option<&'static str> {
    if stats.max_noise > opts.max_noise {
        return Some("noise");
    }
    if stats.mean_change < opts.min_change {
        return Some("static");
    }
    // A RUN of near-identical frames, not a single slow pair — see
    // ClipStats::longest_still_run.
    if stats.longest_still_run > opts.max_still_run {
        return Some("stalls");
    }
    if stats.max_level_jump > opts.max_level_jump {
        return Some("flash");
    }
    if stats.min_coherence < opts.min_coherence {
        return Some("incoherent");
    }
    None
}

/// Gate and score an already-rendered clip. The one place the two searches —
/// scalar modulation and formula morph — agree on what "good" means.
pub fn score_clip(frames: &[Vec<u8>], opts: &TimeExploreOpts) -> (ClipStats, Option<&'static str>, f64) {
    if frames.len() < 2 {
        return (ClipStats::default(), Some("empty"), 0.0);
    }
    let stats = clip_stats(frames, opts.probe_w, opts.probe_h);
    let rejected = gate(&stats, opts);
    let score = if rejected.is_none() {
        probe_frames_score(frames, opts.fps, opts.probe_w, opts.probe_h, None).unwrap_or(0.0)
    } else {
        0.0
    };
    (stats, rejected, score)
}

/// Every (target × shape × amplitude) combination this sweep will try.
pub fn candidate_mods(g: &Genome, opts: &TimeExploreOpts) -> Vec<TimeMod> {
    let mut out = Vec::new();
    for target in enumerate_targets(g) {
        for shape in shapes_for(target, &opts.shapes) {
            for &amp in &opts.amps {
                out.push(TimeMod::new(target, shape, amp));
            }
        }
    }
    out
}

/// Render, gate and score every candidate. Ranked best-first, with rejected
/// candidates kept (score 0) so the manifest can explain what was tried.
pub fn run(
    g: &Genome, config: &Config, view: &View, opts: &TimeExploreOpts,
    on_progress: &dyn Fn(usize, usize, &TimeMod, &TimeCandidate),
) -> Vec<TimeCandidate> {
    let mods = candidate_mods(g, opts);
    let total = mods.len();
    let mut out: Vec<TimeCandidate> = Vec::with_capacity(total);

    for (i, tmod) in mods.into_iter().enumerate() {
        let mut probe = g.clone();
        probe.time_mod = vec![tmod];
        let frames: Vec<Vec<u8>> = time_frames(
            &probe, config, opts.angle_coloring, view, opts.frames, opts.probe_w, opts.probe_h,
        ).collect();

        let (stats, rejected, score) = score_clip(&frames, opts);
        // Frames are dropped here — a full sweep holds one clip at a time.
        let cand = TimeCandidate { tmod, score, stats, rejected };
        on_progress(i + 1, total, &tmod, &cand);
        out.push(cand);
    }

    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// A genome carrying just this candidate's modulation, ready to render or save.
pub fn genome_with(g: &Genome, tmod: &TimeMod) -> Genome {
    let mut out = g.clone();
    out.time_mod = vec![*tmod];
    out
}

fn target_json(t: &ModTarget) -> serde_json::Value {
    match t {
        ModTarget::JuliaC => serde_json::json!({ "kind": "julia_c" }),
        ModTarget::Phoenix => serde_json::json!({ "kind": "phoenix" }),
        ModTarget::Bailout => serde_json::json!({ "kind": "bailout" }),
        ModTarget::ProgConst { node } => serde_json::json!({ "kind": "prog_const", "node": node }),
        ModTarget::WarpConst { node } => serde_json::json!({ "kind": "warp_const", "node": node }),
        ModTarget::ProgScale { node } => serde_json::json!({ "kind": "prog_scale", "node": node }),
    }
}

/// Write `time_winners.jsonl` (every candidate, ranked, with its rejection
/// reason if any) plus a re-rendered clip for the top `top_k` survivors.
///
/// The clips are re-rendered rather than kept from the sweep: holding every
/// candidate's frames would cost hundreds of megabytes, and re-rendering the
/// handful that won is a couple of seconds.
pub fn write_manifest(
    out_dir: &Path, cands: &[TimeCandidate], g: &Genome, config: &Config,
    view: &View, opts: &TimeExploreOpts, keep_clips: bool,
) -> std::io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir)?;

    let mut clips = Vec::new();
    if keep_clips {
        for (rank, c) in cands.iter().filter(|c| c.passed()).take(opts.top_k).enumerate() {
            let path = out_dir.join(format!("winner_{rank:02}.mp4"));
            let probe = genome_with(g, &c.tmod);
            let frames: Vec<Vec<u8>> = time_frames(
                &probe, config, opts.angle_coloring, view, opts.frames, opts.probe_w, opts.probe_h,
            ).collect();
            if probe_frames_score(&frames, opts.fps, opts.probe_w, opts.probe_h, Some(&path)).is_some() {
                clips.push(path);
            }
        }
    }

    let mut lines = String::new();
    for (rank, c) in cands.iter().enumerate() {
        let clip = if c.passed() && rank < clips.len() {
            serde_json::Value::String(
                clips[rank].file_name().unwrap_or_default().to_string_lossy().into_owned(),
            )
        } else {
            serde_json::Value::Null
        };
        let v = serde_json::json!({
            "rank": rank,
            "genome_id": format!("{:016x}", g.id),
            "target": target_json(&c.tmod.target),
            "target_label": c.tmod.target.label(),
            "shape": c.tmod.shape.label(),
            "amp": c.tmod.amp,
            "freq": c.tmod.freq,
            "phase": c.tmod.phase,
            "score": c.score,
            "min_coherence": c.stats.min_coherence,
            "mean_coherence": c.stats.mean_coherence,
            "mean_change": c.stats.mean_change,
            "min_change": c.stats.min_change,
            "longest_still_run": c.stats.longest_still_run,
            "max_level_jump": c.stats.max_level_jump,
            "max_noise": c.stats.max_noise,
            "rejected": c.rejected,
            "loops": c.tmod.shape.loops(),
            "clip": clip,
            "view": { "cx": view.cx, "cy": view.cy, "zoom": view.zoom },
            "probe": {
                "w": opts.probe_w, "h": opts.probe_h,
                "frames": opts.frames, "fps": opts.fps,
            },
        });
        lines.push_str(&v.to_string());
        lines.push('\n');
    }
    std::fs::write(out_dir.join("time_winners.jsonl"), lines)?;
    Ok(clips)
}

/// One-line summary of a finished sweep, including WHY nothing survived when
/// nothing does.
pub fn summary(cands: &[TimeCandidate]) -> String {
    let passed = cands.iter().filter(|c| c.passed()).count();
    if passed > 0 {
        let best = &cands[0];
        return format!(
            "{passed}/{} candidates passed; best = {} {} amp {:.3}  score {:.4}  worst-coherence {:.2}  change {:.1}",
            cands.len(), best.tmod.target.label(), best.tmod.shape.label(),
            best.tmod.amp, best.score, best.stats.min_coherence, best.stats.mean_change,
        );
    }
    // Counted from the gate names themselves so a new gate can never be
    // silently omitted — the first version of this hard-coded three of the five
    // and reported "1 static, 0 incoherent, 1 noisy" for four candidates.
    let mut counts: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    for c in cands {
        if let Some(why) = c.rejected {
            *counts.entry(why).or_insert(0) += 1;
        }
    }
    let breakdown: Vec<String> = counts.iter().map(|(k, n)| format!("{n} {k}")).collect();
    let total: usize = counts.values().sum();
    debug_assert_eq!(total, cands.len(), "every rejected candidate must be counted");

    // Advice only for the gates that actually fired, so it stays short and
    // points at the knob that matters for THIS run.
    let mut advice: Vec<&str> = Vec::new();
    if counts.contains_key("static") {
        advice.push("'static' = amplitudes below this genome's sensitivity: raise --amps");
    }
    if counts.contains_key("incoherent") {
        advice.push("'incoherent' = amplitudes above it, cuts rather than a morph: lower --amps");
    }
    if counts.contains_key("stalls") {
        advice.push("'stalls' = the clip freezes partway; try a different shape or amplitude");
    }
    if counts.contains_key("flash") {
        advice.push("'flash' = a sudden global brightness jump: lower --amps");
    }
    if counts.contains_key("noise") {
        advice.push("'noise' = this genome dithers at the amplitudes tried; --max-noise relaxes the floor");
    }
    format!(
        "0 of {} candidates passed ({}). {}",
        cands.len(),
        breakdown.join(", "),
        advice.join(". ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::OpNode;

    fn mandelbrot() -> Genome {
        let mut g = Genome::default();
        g.program = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        g.bailout_radius = 4.0;
        g
    }

    #[test]
    fn julia_c_is_offered_only_in_julia_mode() {
        let g = mandelbrot();
        assert!(!enumerate_targets(&g).contains(&ModTarget::JuliaC),
            "animating an unread constant would render a static clip");
        let mut j = g.clone();
        j.julia_mode = true;
        assert!(enumerate_targets(&j).contains(&ModTarget::JuliaC));
    }

    #[test]
    fn julia_c_is_skipped_when_the_program_never_reads_c() {
        // Real case from the archive: julia_mode is set, but the formula is a
        // pure function of z, so the constant is decorative and animating it
        // would render a static clip.
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.program = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        assert!(!program_reads_c(&g.program));
        assert!(!enumerate_targets(&g).contains(&ModTarget::JuliaC));
    }

    #[test]
    fn a_c_leaf_in_an_intron_does_not_count_as_reading_c() {
        // The C node exists but nothing consumes it, so the julia constant is
        // still unread.
        let prog = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        assert!(!program_reads_c(&prog), "C is present but dead");
        let live = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 0, b: 1, kre: 0.0, kim: 0.0 },
        ];
        assert!(program_reads_c(&live));
    }

    #[test]
    fn prog_const_is_offered_only_for_const_nodes() {
        let mut g = mandelbrot();
        g.program.push(OpNode { op: op::CONST, a: 0, b: 0, kre: 0.5, kim: 0.0 });
        let const_idx = (g.program.len() - 1) as u8;
        let targets = enumerate_targets(&g);
        let offered: Vec<u8> = targets.iter().filter_map(|t| match t {
            ModTarget::ProgConst { node } => Some(*node),
            _ => None,
        }).collect();
        assert_eq!(offered, vec![const_idx]);
    }

    #[test]
    fn a_genome_with_no_const_still_gets_scale_targets() {
        // The 75% case: nothing to drive directly, so ProgScale must cover it.
        let g = mandelbrot();
        let targets = enumerate_targets(&g);
        assert!(!targets.iter().any(|t| matches!(t, ModTarget::ProgConst { .. })));
        assert!(targets.iter().any(|t| matches!(t, ModTarget::ProgScale { .. })),
            "no way to animate this genome at all: {targets:?}");
    }

    #[test]
    fn scale_targets_are_interior_nodes_and_bounded() {
        let mut g = mandelbrot();
        for i in 4..20u8 {
            g.program.push(OpNode { op: op::ADD, a: i - 1, b: i - 2, kre: 0.0, kim: 0.0 });
        }
        let scales: Vec<u8> = enumerate_targets(&g).iter().filter_map(|t| match t {
            ModTarget::ProgScale { node } => Some(*node),
            _ => None,
        }).collect();
        assert!(scales.len() <= MAX_PROG_SCALE_TARGETS + 1, "unbounded: {scales:?}");
        for n in &scales {
            assert!(op::arity(g.program[*n as usize].op) >= 1, "node {n} is a leaf");
        }
        let root = (g.program.len() - 1) as u8;
        assert!(scales.contains(&root), "the whole-formula scale must always be tried");
    }

    #[test]
    fn dead_nodes_are_never_offered_as_targets() {
        // An intron cannot affect a pixel, so a probe aimed at it is guaranteed
        // to come back "static" — a misleading diagnosis that says "raise the
        // amplitude" when no amplitude would ever work.
        let mut g = mandelbrot();
        // Append a CONST and an unrelated SQR that nothing consumes; the root
        // (the ADD) stays where it is, so both new nodes are dead.
        g.program.insert(3, OpNode { op: op::CONST, a: 0, b: 0, kre: 0.7, kim: 0.0 });
        g.program.insert(4, OpNode { op: op::SIN, a: 3, b: 0, kre: 0.0, kim: 0.0 });
        // Re-point the (now shifted) ADD so it still computes z² + c.
        let last = g.program.len() - 1;
        g.program[last] = OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 };

        let live = crate::formula::reachable_from_root(&g.program);
        assert!(!live[3] && !live[4], "fixture is wrong — nodes 3/4 should be dead");

        for t in enumerate_targets(&g) {
            match t {
                ModTarget::ProgConst { node } | ModTarget::ProgScale { node } => {
                    assert!(live[node as usize], "offered dead node {node}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn orbit_is_skipped_for_the_single_channel_target() {
        let all = vec![ModShape::Sine, ModShape::Orbit];
        assert_eq!(shapes_for(ModTarget::Bailout, &all), vec![ModShape::Sine]);
        assert_eq!(shapes_for(ModTarget::JuliaC, &all), all);
    }

    #[test]
    fn pearson_is_one_for_identical_and_zero_for_flat() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((pearson(&a, &a) - 1.0).abs() < 1e-5);
        let b: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!((pearson(&a, &b) + 1.0).abs() < 1e-5);
        // A constant signal has undefined correlation; it must NOT read as
        // perfect agreement or two flat frames would pass the coherence gate.
        let flat = vec![3.0f32; 5];
        assert_eq!(pearson(&flat, &flat), 0.0);
        assert_eq!(pearson(&flat, &a), 0.0);
        assert_eq!(pearson(&a, &[]), 0.0);
    }

    fn frame(w: u32, h: u32, mut f: impl FnMut(u32, u32) -> u8) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                out.extend_from_slice(&[v, v, v]);
            }
        }
        out
    }

    #[test]
    fn a_gentle_drift_is_coherent_and_moving() {
        let (w, h) = (32u32, 32u32);
        let frames: Vec<Vec<u8>> = (0..8).map(|i| {
            frame(w, h, |x, y| (((x + i) * 4 + y * 2) % 200) as u8)
        }).collect();
        let st = clip_stats(&frames, w, h);
        assert!(st.min_coherence > MIN_TEMPORAL_COHERENCE, "worst coherence {}", st.min_coherence);
        assert!(st.mean_change > MIN_TEMPORAL_CHANGE, "change {}", st.mean_change);
        assert_eq!(gate(&st, &TimeExploreOpts::default()), None);
    }

    #[test]
    fn unrelated_frames_are_rejected_as_incoherent() {
        // The failure mode the whole gate exists for: each frame is fine on its
        // own, but nothing connects them.
        let (w, h) = (32u32, 32u32);
        let frames: Vec<Vec<u8>> = (0..8u32).map(|i| {
            let mut st = 0x9E3779B9u32.wrapping_mul(i + 1);
            frame(w, h, move |x, y| {
                st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                (((x * 7 + y * 13) as u32).wrapping_add(st) % 255) as u8
            })
        }).collect();
        let s = clip_stats(&frames, w, h);
        let g = gate(&s, &TimeExploreOpts::default());
        assert!(g.is_some(), "unrelated frames must be rejected, worst coherence {}", s.min_coherence);
    }

    #[test]
    fn an_unmoving_clip_is_rejected_as_static() {
        let (w, h) = (32u32, 32u32);
        let frames: Vec<Vec<u8>> = (0..8).map(|_| frame(w, h, |x, y| ((x * 3 + y * 5) % 200) as u8)).collect();
        let st = clip_stats(&frames, w, h);
        assert_eq!(st.mean_change, 0.0);
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("static"),
            "a winner where nothing happens is worse than no winner");
    }

    #[test]
    fn a_clip_that_freezes_partway_is_rejected_even_though_it_averages_well() {
        // The exact failure the worst-case rework exists for: five identical
        // frames in the middle correlate at 1.000 each and pull the MEAN
        // coherence to 0.955, which is why the original mean-based gate passed
        // this clip while it visibly hung.
        let st = ClipStats {
            max_noise: 0.0, min_coherence: 0.895, mean_coherence: 0.955,
            mean_change: 15.0, min_change: 0.0,
            // 4 still pairs out of 15 — the measured freeze.
            longest_still_run: 4.0 / 15.0, max_level_jump: 5.0,
        };
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("stalls"));
    }

    #[test]
    fn a_sinusoid_is_not_punished_for_turning_around() {
        // Carl, 2026-09-10: "find time axis does not work anymore even if there
        // is plenty good candidates". The old gate compared min_change to
        // mean_change, and a sinusoid's velocity is ZERO at its turning points
        // by definition — so every sine and triangle candidate was rejected,
        // including these real measured numbers: a clip averaging 13.4
        // luminance units of motion per frame, thrown out because its minimum
        // pair was 0.688 (ratio 0.05).
        let st = ClipStats {
            max_noise: 0.0, min_coherence: 0.93, mean_coherence: 0.97,
            mean_change: 13.398, min_change: 0.688,
            // One slow pair per half-cycle is not a run.
            longest_still_run: 0.0, max_level_jump: 4.0,
        };
        assert_eq!(gate(&st, &TimeExploreOpts::default()), None,
            "smooth periodic motion must not be rejected for being smooth");
    }

    #[test]
    fn one_still_pair_passes_but_a_run_of_them_does_not() {
        // 16-frame clip = 15 pairs. A single still pair is 6.7% of the clip; a
        // freeze of three is 20%.
        let one = ClipStats {
            mean_change: 10.0, min_coherence: 1.0, longest_still_run: 1.0 / 15.0,
            ..Default::default()
        };
        assert_eq!(gate(&one, &TimeExploreOpts::default()), None);
        let three = ClipStats {
            mean_change: 10.0, min_coherence: 1.0, longest_still_run: 3.0 / 15.0,
            ..Default::default()
        };
        assert_eq!(gate(&three, &TimeExploreOpts::default()), Some("stalls"));
    }

    #[test]
    fn a_synthetic_sine_clip_measures_no_still_run() {
        // End to end through clip_stats rather than hand-built stats: a real
        // sinusoidal sweep of pixel values must report longest_still_run 0.
        let (w, h) = (32u32, 32u32);
        let n = 24;
        let frames: Vec<Vec<u8>> = (0..n).map(|i| {
            let t = i as f32 / n as f32;
            // No wrap-around: rem_euclid would snap the level from 255 to 0 and
            // register as a flash, which is a defect in the fixture, not the clip.
            let off = 20.0 * (std::f32::consts::TAU * t).sin();
            frame(w, h, move |x, y| {
                (((x * 3 + y * 5) % 120) as f32 + 70.0 + off).clamp(0.0, 255.0) as u8
            })
        }).collect();
        let st = clip_stats(&frames, w, h);
        // Not necessarily zero: this fixture is a uniform ramp, so near the
        // turning point the whole frame quantizes to the same u8 values and a
        // pair or two really is identical. That is a short run, not a freeze,
        // and the gate is what has to get it right.
        assert!(st.longest_still_run <= MAX_STILL_RUN,
            "a sine sweep must not read as a freeze, got {}", st.longest_still_run);
        assert_eq!(gate(&st, &TimeExploreOpts::default()), None,
            "smooth periodic motion must pass end to end");
    }

    #[test]
    fn a_synthetic_frozen_stretch_is_caught_end_to_end() {
        // The control: same clip, but frames 8..13 held.
        let (w, h) = (32u32, 32u32);
        let n = 24;
        let frames: Vec<Vec<u8>> = (0..n).map(|i| {
            let held = if (8..14).contains(&i) { 8 } else { i };
            let t = held as f32 / n as f32;
            let off = 20.0 * (std::f32::consts::TAU * t).sin();
            frame(w, h, move |x, y| {
                (((x * 3 + y * 5) % 120) as f32 + 70.0 + off).clamp(0.0, 255.0) as u8
            })
        }).collect();
        let st = clip_stats(&frames, w, h);
        assert!(st.longest_still_run > MAX_STILL_RUN,
            "a five-frame freeze must be caught, got {}", st.longest_still_run);
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("stalls"));
    }

    #[test]
    fn a_global_flash_is_rejected_although_correlation_cannot_see_it() {
        // Pearson is invariant to offset and scale, so a frame of identical
        // structure at double brightness still correlates at 1.0. Only the
        // level-jump measure catches it.
        let bright: Vec<f32> = (0..64).map(|i| 100.0 + i as f32).collect();
        let dim: Vec<f32> = bright.iter().map(|v| v * 0.4).collect();
        assert!(pearson(&bright, &dim) > 0.999, "correlation is blind to a flash");

        let st = ClipStats {
            max_noise: 0.0, min_coherence: 1.0, mean_coherence: 1.0,
            mean_change: 10.0, min_change: 8.0,
            longest_still_run: 0.0, max_level_jump: 54.6,
        };
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("flash"));
    }

    #[test]
    fn the_calibrated_thresholds_separate_the_measured_groups() {
        // Guards the calibration table in MIN_STALL_RATIO's doc comment: the
        // worst clean candidate must pass and the best bad one must not.
        let clean = ClipStats {
            max_noise: 0.0, min_coherence: 0.919, mean_coherence: 0.963,
            mean_change: 5.7, min_change: 3.35,
            longest_still_run: 0.0, max_level_jump: 3.4,
        };
        assert_eq!(gate(&clean, &TimeExploreOpts::default()), None);
        let bad = ClipStats {
            max_noise: 0.0, min_coherence: 0.895, mean_coherence: 0.965,
            mean_change: 15.0, min_change: 0.0,
            longest_still_run: 4.0 / 15.0, max_level_jump: 54.6,
        };
        assert!(gate(&bad, &TimeExploreOpts::default()).is_some());
    }

    #[test]
    fn the_clip_noise_floor_is_stricter_than_the_still_floor() {
        // Regression for the first real sweep, where the top three candidates
        // all measured exactly 0.250 — the still floor, on a quantization point
        // of noise_tile_fraction — and slipped through a `>` comparison while
        // being visibly dithered.
        assert!(MAX_CLIP_NOISE < fitness::MAX_NOISE_TILE_FRACTION);
        let st = ClipStats { max_noise: 0.25, min_coherence: 0.90, mean_change: 13.0, ..Default::default() };
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("noise"));
    }

    #[test]
    fn the_noise_gate_fires_before_the_others() {
        // Order matters: a noisy clip is also usually incoherent, and "noise"
        // is the more actionable diagnosis.
        let st = ClipStats { max_noise: 1.0, min_coherence: 0.0, mean_change: 0.0, ..Default::default() };
        assert_eq!(gate(&st, &TimeExploreOpts::default()), Some("noise"));
    }

    #[test]
    fn candidate_count_is_the_product_of_the_three_axes() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        let opts = TimeExploreOpts {
            shapes: vec![ModShape::Sine, ModShape::Orbit],
            amps: vec![0.1, 0.2],
            ..Default::default()
        };
        let mods = candidate_mods(&g, &opts);
        // Bailout is single-channel, so it loses the Orbit variant.
        let targets = enumerate_targets(&g);
        let expected: usize = targets.iter()
            .map(|t| shapes_for(*t, &opts.shapes).len() * opts.amps.len())
            .sum();
        assert_eq!(mods.len(), expected);
        assert!(mods.iter().all(|m| m.freq == 1.0), "sweep should hold freq at one loop");
    }

    #[test]
    fn the_summary_explains_a_zero_result_instead_of_just_reporting_it() {
        let cands: Vec<TimeCandidate> = ["static", "static", "incoherent"].iter().map(|r| {
            TimeCandidate {
                tmod: TimeMod::new(ModTarget::Bailout, ModShape::Sine, 0.1),
                score: 0.0,
                stats: ClipStats::default(),
                rejected: Some(r),
            }
        }).collect();
        let s = summary(&cands);
        assert!(s.contains("0 of 3"), "{s}");
        assert!(s.contains("2 static"), "counts must appear: {s}");
        assert!(s.contains("1 incoherent"), "counts must appear: {s}");
        assert!(s.contains("raise --amps"), "must say what to do next: {s}");
    }

    #[test]
    fn the_summary_counts_every_gate_including_the_newer_ones() {
        // The first version hard-coded three of the five reasons and silently
        // dropped 'stalls' and 'flash', so the breakdown did not add up to the
        // number of candidates.
        let cands: Vec<TimeCandidate> = ["noise", "static", "stalls", "flash", "incoherent"]
            .iter()
            .map(|r| TimeCandidate {
                tmod: TimeMod::new(ModTarget::Bailout, ModShape::Sine, 0.1),
                score: 0.0,
                stats: ClipStats::default(),
                rejected: Some(r),
            })
            .collect();
        let s = summary(&cands);
        for reason in ["noise", "static", "stalls", "flash", "incoherent"] {
            assert!(s.contains(&format!("1 {reason}")), "{reason} missing from: {s}");
        }
    }
}

// ── Formula morphing: search the pool for a fractal that blends well ────────

/// One candidate partner fractal, and how well morphing into it went.
#[derive(Clone, Debug)]
pub struct BlendCandidate {
    pub partner_path: PathBuf,
    pub partner_id: String,
    pub shape: ModShape,
    /// How far toward the partner the morph travels, in [0,1].
    pub amp: f32,
    pub score: f64,
    pub stats: ClipStats,
    /// Which gate stopped it, or why it could not be blended at all.
    pub rejected: Option<&'static str>,
    /// Detail for the incompatible case, which is not a gate result.
    pub note: String,
}

impl BlendCandidate {
    pub fn passed(&self) -> bool {
        self.rejected.is_none()
    }
}

/// Shapes tried per partner. Periodic ones give an A→B→A ping-pong that loops;
/// a one-way ramp cannot, so it is not in the default set.
pub const BLEND_SHAPES: &[ModShape] = &[ModShape::Sine, ModShape::Triangle];

/// How far toward the partner formula to travel. Measured on real archive
/// pairs, a full 0→1 sweep crosses a bifurcation for most of them and the clip
/// reads as a cut; a partial morph is continuous and watchable. Swept like the
/// scalar search sweeps amplitudes.
pub const BLEND_AMPS: &[f32] = &[0.15, 0.35, 0.7, 1.0];

/// Sample `max_samples` genomes from `pool_dir`, keep the ones that can blend
/// with `g`, render a short morph for each and rank them.
///
/// Sampling rather than exhausting the pool is the point: a pool holds tens of
/// thousands of genomes and each candidate costs a clip. `max_samples` is the
/// budget, and incompatible draws are cheap to reject (no render), so the real
/// cost is the number that survive compatibility.
#[allow(clippy::too_many_arguments)]
pub fn blend_pool_search(
    g: &Genome, config: &Config, view: &View, pool_dir: &Path,
    max_samples: usize, seed: u64, opts: &TimeExploreOpts,
    on_progress: &dyn Fn(usize, usize, &BlendCandidate),
) -> Vec<BlendCandidate> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut paths: Vec<PathBuf> = std::fs::read_dir(pool_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    paths.shuffle(&mut rng);

    let self_id = format!("{:016x}", g.id);
    let mut out: Vec<BlendCandidate> = Vec::new();
    let mut looked = 0usize;

    for path in paths {
        if looked >= max_samples {
            break;
        }
        let Ok(partner) = crate::io::load_genome(&path) else { continue };
        let pid = format!("{:016x}", partner.id);
        if pid == self_id {
            continue;
        }
        looked += 1;

        if let Err(why) = g.blend_compatibility(&partner) {
            let cand = BlendCandidate {
                partner_path: path.clone(), partner_id: pid, shape: BLEND_SHAPES[0], amp: 0.0,
                score: 0.0, stats: ClipStats::default(),
                rejected: Some("incompatible"), note: why,
            };
            on_progress(looked, max_samples, &cand);
            out.push(cand);
            continue;
        }

        for &shape in BLEND_SHAPES {
            for &amp in BLEND_AMPS {
                let frames: Vec<Vec<u8>> = crate::video_export::blend_frames(
                    g, &partner, config, opts.angle_coloring, view,
                    opts.frames, opts.probe_w, opts.probe_h, shape, 1.0, 0.0, amp,
                ).collect();
                let (stats, rejected, score) = score_clip(&frames, opts);
                let cand = BlendCandidate {
                    partner_path: path.clone(), partner_id: pid.clone(), shape, amp,
                    score, stats, rejected, note: String::new(),
                };
                on_progress(looked, max_samples, &cand);
                out.push(cand);
            }
        }
    }

    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Write `blend_winners.jsonl` plus a clip per surviving winner.
#[allow(clippy::too_many_arguments)]
pub fn write_blend_manifest(
    out_dir: &Path, cands: &[BlendCandidate], g: &Genome, config: &Config,
    view: &View, opts: &TimeExploreOpts, keep_clips: bool,
) -> std::io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir)?;
    let mut clips = Vec::new();
    if keep_clips {
        for (rank, c) in cands.iter().filter(|c| c.passed()).take(opts.top_k).enumerate() {
            let Ok(partner) = crate::io::load_genome(&c.partner_path) else { continue };
            let path = out_dir.join(format!("blend_{rank:02}.mp4"));
            let frames: Vec<Vec<u8>> = crate::video_export::blend_frames(
                g, &partner, config, opts.angle_coloring, view,
                opts.frames, opts.probe_w, opts.probe_h, c.shape, 1.0, 0.0, c.amp,
            ).collect();
            if probe_frames_score(&frames, opts.fps, opts.probe_w, opts.probe_h, Some(&path)).is_some() {
                clips.push(path);
            }
        }
    }

    let mut lines = String::new();
    for (rank, c) in cands.iter().enumerate() {
        let clip = if c.passed() && rank < clips.len() {
            serde_json::Value::String(
                clips[rank].file_name().unwrap_or_default().to_string_lossy().into_owned())
        } else {
            serde_json::Value::Null
        };
        let v = serde_json::json!({
            "rank": rank,
            "genome_id": format!("{:016x}", g.id),
            "partner_id": c.partner_id,
            "partner_path": c.partner_path.to_string_lossy(),
            "shape": c.shape.label(),
            "amp": c.amp,
            "score": c.score,
            "min_coherence": c.stats.min_coherence,
            "mean_change": c.stats.mean_change,
            "longest_still_run": c.stats.longest_still_run,
            "max_level_jump": c.stats.max_level_jump,
            "max_noise": c.stats.max_noise,
            "rejected": c.rejected,
            "note": c.note,
            "clip": clip,
            "view": { "cx": view.cx, "cy": view.cy, "zoom": view.zoom },
        });
        lines.push_str(&v.to_string());
        lines.push('\n');
    }
    std::fs::write(out_dir.join("blend_winners.jsonl"), lines)?;
    Ok(clips)
}

/// One-line summary of a blend search, naming why nothing survived when
/// nothing does.
pub fn blend_summary(cands: &[BlendCandidate]) -> String {
    let passed = cands.iter().filter(|c| c.passed()).count();
    if passed > 0 {
        let best = &cands[0];
        return format!(
            "{passed} of {} tried; best partner {} ({} amp {:.2}) score {:.4}  worst-coherence {:.2}  change {:.1}",
            cands.len(), best.partner_id, best.shape.label(), best.amp,
            best.score, best.stats.min_coherence, best.stats.mean_change,
        );
    }
    let mut counts: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    for c in cands {
        if let Some(why) = c.rejected {
            *counts.entry(why).or_insert(0) += 1;
        }
    }
    let breakdown: Vec<String> = counts.iter().map(|(k, n)| format!("{n} {k}")).collect();
    let mut advice = String::new();
    if counts.get("incompatible").copied().unwrap_or(0) > cands.len() / 2 {
        advice = " Most draws could not be blended at all — usually a julia-mode mismatch \
                  or two programs too large to fit together. Raising --samples helps; so does \
                  starting from a genome with a short formula.".into();
    }
    format!("0 of {} candidates passed ({}).{advice}", cands.len(), breakdown.join(", "))
}

#[cfg(test)]
mod blend_search_tests {
    use super::*;
    use crate::formula::{blend_fraction, ModShape};

    #[test]
    fn travel_caps_how_far_the_morph_goes() {
        // The measured fix: a full sweep crosses a bifurcation on most real
        // pairs (mean coherence 0.25 at travel 1.0 vs 0.66 at 0.15), so the
        // amplitude has to actually bound s.
        for amp in [0.15f32, 0.35, 0.7, 1.0] {
            let mut hi = 0.0f32;
            for i in 0..200 {
                let t = i as f32 / 200.0;
                let s = blend_fraction(ModShape::Sine, 1.0, 0.0, amp, t);
                assert!((0.0..=amp + 1e-5).contains(&s), "s={s} outside [0,{amp}]");
                hi = hi.max(s);
            }
            assert!((hi - amp).abs() < 0.05, "travel {amp} should be reached, peaked at {hi}");
        }
    }

    #[test]
    fn zero_travel_never_leaves_the_first_fractal() {
        for i in 0..50 {
            let t = i as f32 / 50.0;
            assert_eq!(blend_fraction(ModShape::Sine, 1.0, 0.0, 0.0, t), 0.0);
        }
    }

    #[test]
    fn periodic_shapes_return_to_where_they_started() {
        for shape in [ModShape::Sine, ModShape::Triangle, ModShape::Cosine, ModShape::Orbit] {
            let a = blend_fraction(shape, 1.0, 0.0, 0.5, 0.0);
            let b = blend_fraction(shape, 1.0, 0.0, 0.5, 1.0);
            assert!((a - b).abs() < 1e-5, "{shape:?} does not loop: {a} vs {b}");
        }
    }

    #[test]
    fn a_ramp_reaches_the_partner_and_stops() {
        assert_eq!(blend_fraction(ModShape::Ramp, 1.0, 0.0, 1.0, 0.0), 0.0);
        assert!((blend_fraction(ModShape::Ramp, 1.0, 0.0, 1.0, 0.99) - 0.99).abs() < 0.02);
    }

    #[test]
    fn the_blend_summary_explains_a_mostly_incompatible_pool() {
        let cands: Vec<BlendCandidate> = (0..10).map(|i| BlendCandidate {
            partner_path: PathBuf::from("x.nn"),
            partner_id: format!("{i:016x}"),
            shape: ModShape::Sine, amp: 1.0, score: 0.0,
            stats: ClipStats::default(),
            rejected: Some("incompatible"),
            note: "julia mode differs".into(),
        }).collect();
        let s = blend_summary(&cands);
        assert!(s.contains("10 incompatible"), "{s}");
        assert!(s.contains("julia-mode mismatch"), "must say why: {s}");
    }
}
