//! Video-compression-driven zoom exploration ("Video-Zoom Explore") — finds
//! (start → end) zoom paths whose rendered, compressed preview video is as
//! large/entropic as possible (`video_export::probe_video_score`), capped
//! just before the DD precision wall. Chess-like in the literal sense: at
//! each real decision point, look `VideoZoomOpts::lookahead_plies` moves
//! ahead — branch once, extend each branch's own continuation greedily, and
//! commit to only the first move of whichever continuation scores best —
//! rather than just re-ranking the immediate options on their own account.
//!
//! Built entirely on top of `explore.rs`/`vae_explore.rs`'s already-`pub`/
//! `pub(crate)` surface (`coarse_scan`, `rank_by_zscore`, `apply_offset`,
//! `Candidate`, `Logger`, `ScoreMethod`, `save_shot`) — no changes to either
//! file. See `[[project-video-zoom-explore]]`-equivalent plan notes for the
//! full design rationale (why `vae_explore`'s canvas engine, not
//! `explore::sweep`; why normalized ratios, not raw byte counts).

use crate::config::Config;
use crate::explore::{apply_offset, rank_by_zscore, save_shot, Candidate, Logger, ScoreMethod};
use crate::fitness;
use crate::genome::Genome;
use crate::render_gpu;
use crate::vae_explore::{coarse_scan, ZoneGate, COARSE_SAMPLE_RES};
use crate::video_export::{
    effective_max_iter, lerp_view, needs_dd_with_margin, needs_f64, probe_video_score,
    probe_video_score_keep, render_escape_times, CapturedView, View, DD_MARGIN_ULPS_PIXELATE,
    VIDEO_FRAME_ALLOW_DD,
};
use std::path::Path;

/// Backtracking-DFS visit budget, `max_depth * BACKTRACK_BUDGET_FACTOR` —
/// same constant/shape as `vae_explore::recursive_drill`'s own budget
/// model. Kept as a small local duplicate rather than importing
/// `vae_explore`'s copy: the two engines are otherwise independent (this
/// one tracks full `Vec<CapturedView>` paths for later queuing;
/// `recursive_drill` only returns a save-count), and this project already
/// tolerates this kind of small duplication over reaching into an
/// already-proven module for 3 lines (see `vae_explore::test_config` vs.
/// `video_export::chain_test_config`).
const BACKTRACK_BUDGET_FACTOR: usize = 3;

/// Full-process trace, written to `<out_dir>/vz_trace.log` on EVERY run.
/// Deliberately always-on and file-backed rather than stderr-gated: Carl
/// runs this from the viewer GUI, where a subprocess's stderr only reaches
/// a 500-line capped, ephemeral log panel — useless for after-the-fact
/// analysis. Tracing is what found the sibling depth-accounting bug after
/// several scoring theories failed, so it earns its keep.
static TRACE: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();

pub fn init_trace(out_dir: &Path) {
    let _ = std::fs::create_dir_all(out_dir);
    if let Ok(f) = std::fs::File::create(out_dir.join("vz_trace.log")) {
        let _ = TRACE.set(std::sync::Mutex::new(f));
    }
}

pub fn trace_write(line: &str) {
    use std::io::Write;
    if let Some(m) = TRACE.get()
        && let Ok(mut f) = m.lock() {
        let _ = writeln!(f, "{line}");
    }
}

macro_rules! vztrace {
    ($($arg:tt)*) => { crate::video_zoom_explore::trace_write(&format!($($arg)*)) };
}

/// Cap on how many candidates one `"level_scanning"` log line lists — a
/// real, reproducible bug (not theoretical): `coarse_scan` routinely
/// returns 200-243 candidates per level (confirmed from real runs), and
/// logging every one of them, unlike `vae_explore.rs`'s own convention of
/// truncating to `top_k` (typically 6) BEFORE logging, meant the viewer's
/// scan overlay — which draws one bordered square per logged candidate —
/// tiled the ENTIRE visible canvas with squares on a real run (Carl's
/// screenshot, 2026-08-13). An initial cap of 30 was still visually dense
/// (confirmed via a live screenshot, 2026-08-13 follow-up): `coarse_scan`'s
/// `sweep_positions` grid means nearby cells tend to score similarly, so
/// "top N by rank" clusters into overlapping adjacent squares rather than
/// spreading out — a count problem AND a clustering problem. 12 keeps
/// enough of the z-score-ranked (best-first) list to still diagnose a run
/// (this is how the earlier entropy-plateau bug in this module was found)
/// without the mesh-of-boxes look. The viewer ALSO independently caps how
/// many it draws (`viewer.rs`'s own constant, in case this one is ever
/// raised again for diagnostic reasons without remembering the overlay
/// impact) — the actual finalist selection below is unaffected either way,
/// it still filters/takes from the FULL, uncapped list.
const LOG_CANDIDATE_CAP: usize = 12;

/// How many of `cheap_funnel`'s cheap, broad pre-filter survivors get a
/// REAL file-size-entropy render — see `cheap_funnel`'s doc comment for why
/// the pre-filter's own histogram-entropy ranking isn't trusted as the
/// deciding signal (Carl's explicit instruction, 2026-08-13, after real
/// hand-picked zooms turned out visibly better than this search's own
/// output: "Please always use file size entropy since it is the most
/// effective"). Bounded rather than scoring coarse_scan's FULL raw output
/// (200+ candidates) — each of these costs a real small render + PNG
/// encode, cheap individually but not free at scale.
/// Canvas resolution for this pipeline — deliberately NOT
/// `vae_explore::CANVAS_RES` (4095). `coarse_scan` strided-samples every
/// crop down to `COARSE_SAMPLE_RES` (128) and the smallest
/// `CANVAS_SCAN_SCALES` entry is 0.125, so 128/0.125 = 1024 is the
/// smallest canvas that fully feeds the sampling — anything larger renders
/// pixels that are then thrown away. Measured A/B (2026-08-14, depth 3,
/// only this value differing): **4095 → 1006s, best chain 1 leg, end zoom
/// 2.04; 1024 → 141s, best chain 9 legs, end zoom 1.67e4** — i.e. 7.1x
/// faster AND strictly better results, so this is not a speed/quality
/// tradeoff.
///
/// Two compounding reasons it's also BETTER, not merely cheaper: (1) at
/// 4095 a 0.125-scale crop is 512px subsampled 4:1 down to 128, which
/// aliases exactly the fine high-frequency detail these metrics exist to
/// measure, whereas at 1024 the same crop is 128px 1:1; (2)
/// `needs_f64(view, canvas_res)`'s threshold scales with width, so a
/// smaller canvas stays on the fast GPU path ~4x deeper in zoom before
/// falling to the CPU/f64 tier this codebase documents as the dominant
/// cost (20-40s per render).
const VIDEO_ZOOM_CANVAS_RES: u32 = 1024;

const FILE_SIZE_ENTROPY_CANDIDATES: usize = 40;

/// Minimum separation between two winners' endpoints, in frame-widths of
/// the shallower one, for them to count as genuinely different results.
/// `drill_chain_generic` returns EVERY terminus of the DFS — but because
/// backtracking pushes siblings at the same remaining depth while the
/// winner descends one trunk, most of those termini are prefixes or
/// extensions of the SAME path. Measured on a real run (2026-08-14): all
/// five presented winners ended at the identical (cx, cy) 1.814053,
/// 1.567634, differing only in stop depth — i.e. one zoom path shown five
/// times, with nothing to actually choose between. That is almost
/// certainly the bulk of Carl's "all 10 winners unsuccessful" (if the one
/// trunk is mediocre, every winner is mediocre) and of "picking by hand
/// would give a much more complex result" (a human picks genuinely
/// different places). Same spatial-diversity idea this project already
/// applies elsewhere — `explore::MIN_DIVERSITY_DISTANCE`,
/// `select_diverse_latent.py`, vae-explore's zone dedup.
const WINNER_MIN_SEPARATION_FRAMES: f64 = 0.75;

/// True when `b`'s endpoint is close enough to `a`'s to be the same result.
/// Compares at the SHALLOWER endpoint's frame size, so "same place, just
/// deeper" (the exact duplicate shape observed) counts as a duplicate.
fn same_endpoint(a: &CapturedView, b: &CapturedView) -> bool {
    let frame = 2.0 / a.zoom.min(b.zoom);
    let dx = (a.cx_dd() - b.cx_dd()).hi;
    let dy = (a.cy_dd() - b.cy_dd()).hi;
    (dx * dx + dy * dy).sqrt() < frame * WINNER_MIN_SEPARATION_FRAMES
}

/// Resolution for `FILE_SIZE_ENTROPY_CANDIDATES`'s fresh per-candidate
/// crops — reuses `vae_explore::COARSE_SAMPLE_RES`, the resolution
/// `coarse_scan` already samples at internally for its own (pre-filter
/// only) metrics, rather than inventing an unrelated number.
const FILE_SIZE_PROBE_RES: u32 = COARSE_SAMPLE_RES as u32;

#[derive(Clone, Copy, Debug)]
pub struct ProbeSize {
    pub w: u32,
    pub h: u32,
    pub steps: u32,
    pub fps: u32,
}

#[derive(Clone)]
pub struct VideoZoomOpts {
    /// Real committed plies (mirrors `vae_explore`'s `depth`).
    pub max_depth: usize,
    /// Immediate branches evaluated per real decision point.
    pub finalists_per_level: usize,
    /// Moves looked ahead per branch before committing to the first move.
    pub lookahead_plies: usize,
    /// DD-boundary gate reference — the REAL intended export's width, NOT
    /// the probe width. `needs_dd`'s threshold moves with width, so this
    /// must reflect the video that will actually be exported later for
    /// "stop just before the DD zone" to mean anything.
    pub final_export_width: u32,
    /// Output HEIGHT the finished video will be rendered at. Only used to
    /// derive the frame aspect for validation — but that matters: a chain
    /// validated at the waypoints' captured square aspect is not the chain
    /// that gets exported at 1080x1920, it is a different crop of the
    /// fractal. Validating the wrong region is how a noise-tailed chain
    /// passed every gate (Carl, 2026-08-15).
    pub final_export_height: u32,
    /// Per-level canvas render resolution (mirrors `vae_explore::CANVAS_RES`).
    pub canvas_res: u32,
    /// Tiny, cheap — used `finalists_per_level` times per real node visit.
    pub lookahead_probe: ProbeSize,
    /// Bigger — used once per surviving completed chain.
    pub final_probe: ProbeSize,
    pub top_winners: usize,
    /// Loose absolute floor on a candidate's raw `Candidate.score` (the
    /// active `ScoreMethod` applied to TRUE whole-crop metrics, before any
    /// z-score ranking) — a candidate below this is ineligible regardless
    /// of how it compares to its neighbors. Without SOME absolute floor,
    /// `rank_by_zscore`'s own floor (`min_score_fraction`, passed as 0.0
    /// here — see `cheap_funnel`) is purely RELATIVE to the current level's
    /// own best candidate: if EVERY candidate nearby is bad, the "best of a
    /// bad bunch" still gets returned as a finalist, and the search keeps
    /// drilling deeper into a neighborhood with nothing worthwhile in it,
    /// never recognizing a true dead end.
    ///
    /// This alone is NOT sufficient for `ScoreMethod::Entropy`/`GatedEntropy`
    /// specifically, which is why `gate` (below) also exists — see its doc
    /// comment for the real, measured failure mode this floor alone missed.
    /// Kept as a light backstop regardless (cheap, harmless, method-generic).
    pub min_score: f32,
    /// Floor on file-size entropy, as a FRACTION of the seed view's own
    /// file-size entropy — the search may never descend into a region this
    /// much less interesting than where the user started it. Load-bearing:
    /// without it, `file_size_score` was only ever used to SORT candidates,
    /// never to reject them, so a level where every candidate is terrible
    /// still returned its top `finalists_per_level` — the search never saw
    /// an empty result, never registered a `DeadEnd`, and never backtracked.
    /// Confirmed as the cause of a real failure (Carl, 2026-08-13: all 10
    /// winners bad, "after about 1/4 of the zoom the algo gets stuck in a
    /// low entropy zone and keeps zooming in on it") — the winning chain's
    /// zoom trace showed exactly that: a good opening, then a long tail of
    /// timid minimal 2× steps grinding deeper in one spot.
    ///
    /// MEASURED CALIBRATION (overnight sweep, 2026-08-14, `scripts/vz_sweep.sh`,
    /// on the genome from Carl's failing run — ratio / legs / richness
    /// retention): 0.0 (control, floor disabled) → 14 legs, 0.69 retention
    /// i.e. reproduces the original bug; 0.6 → 5 legs, 0.86; **0.7 → 1 leg,
    /// 0.8 → 1 leg, both `BudgetExhausted`** — i.e. the floor rejected
    /// EVERYTHING and the search couldn't descend at all. The first shipped
    /// default of 0.70 was therefore actively broken (a zoom video needs
    /// depth; one leg is not a zoom), which is why this is now 0.45 and is
    /// paired with `min_file_size_step_ratio`. Root cause of that failure:
    /// image richness declines naturally with zoom depth, so demanding a
    /// candidate stay at 70-80% of the SEED's richness forever is simply
    /// unsatisfiable past the first few levels.
    ///
    /// Kept seed-anchored (rather than purely per-parent) because a
    /// per-parent-only ratio permits unbounded cumulative drift — chains here
    /// reach 14 legs via sideways backtracking even at `max_depth` 5, and
    /// 0.85^14 ≈ 0.10, i.e. boiling-frog all the way into a dead zone. The
    /// hybrid gets both properties: this bounds TOTAL degradation, while
    /// `min_file_size_step_ratio` bounds PER-STEP degradation. Also why this
    /// is a RATIO and not an absolute constant: the
    /// meaningful scale of `png_compression_entropy` varies by formula,
    /// colormap and probe resolution, so anything hardcoded here would need
    /// per-formula recalibration; the seed is a view the user themselves
    /// chose to explore from, which makes it the natural reference.
    ///
    /// Catches a case the `gate` (edge_density/intricacy) structurally
    /// cannot: banded/striped regions have HIGH edge density (they sail
    /// through that gate) but are highly periodic, so they compress well
    /// and score low here — visually repetitive, exactly what "boring zoom"
    /// means for this feature.
    pub min_file_size_ratio: f32,
    /// Second half of the hybrid floor: a candidate must also be at least
    /// this fraction of ITS OWN PARENT's file-size entropy. Blocks sudden
    /// cliffs (stepping straight off rich structure into a smooth basin)
    /// which the seed anchor alone can't catch once the chain has legitimately
    /// declined. See `min_file_size_ratio` for why both are needed.
    pub min_file_size_step_ratio: f32,
    /// Minimum zoom multiplier a candidate must represent over the current
    /// view — i.e. every committed move must be a REAL jump, not a nudge.
    /// `vae_explore::CANVAS_SCAN_SCALES`' shallowest tier (0.5 → a 2×
    /// step) let the search creep: on the genome from Carl's failing run,
    /// 9 of the winning chain's 14 steps were minimal 2× moves, so
    /// consecutive "decisions" barely relocated the view at all — which is
    /// literally the reported symptom ("keeps zooming in on it"). Measured
    /// (overnight sweep, 2026-08-14): no setting of the file-size floors
    /// fixes this, because the resulting richness decline is a gentle
    /// cumulative slide (0.82-0.88 per step, never a cliff) rather than
    /// anything a per-step or seed-anchored floor can catch — every floor
    /// value landed in one of two buckets, deep-but-degrading (14 legs,
    /// 0.69 retention) or shallow-but-clean (5 legs, 0.86). Raising the
    /// minimum step is a different lever entirely: it makes grinding in
    /// place structurally impossible instead of trying to detect it after
    /// the fact.
    ///
    /// HOWEVER — measured, and it did NOT improve richness retention:
    /// 2.0 (= old behavior) → 14 legs/0.69 retention; 4.0 → 11 legs/0.65;
    /// 8.0 → 9 legs/0.71. Default therefore stays at **2.0**, i.e. no
    /// behavior change, rather than shipping an unvalidated one. The reason
    /// nothing helped turned out to be the sweeps' own premise: probing the
    /// BEST AVAILABLE candidate at each depth showed the richness CEILING
    /// itself falls from 0.83 near the surface to 0.50-0.73 deep in this
    /// genome, so achievable retention is ~0.78 and the search was already
    /// getting ~0.71 of it. Most of the apparent "degradation" is intrinsic
    /// to zooming deep in this fractal, not a search defect. Raising this
    /// to 8.0 is still worth trying if a zoom FEELS like it crawls — it
    /// makes each step a decisive jump — but expect a different pacing, not
    /// higher richness. Applied as a filter here rather than by editing
    /// `CANVAS_SCAN_SCALES`, which is shared with the separate (and
    /// working) `vae-explore` pipeline.
    pub min_step_zoom: f64,
    /// Minimum richness ANY rendered frame of a winner's video may have.
    /// The other gates all score WAYPOINTS; this scores the frames actually
    /// exported, sampled along each leg with the same `lerp_view` the
    /// exporter uses. Load-bearing because a leg's two endpoints can both be
    /// rich while the straight-line camera path between them crosses empty
    /// space — measured on a real depth-15 run: every waypoint scored
    /// 0.77-0.94 yet the exported video was solid interior colour from frame
    /// 8 on. Waypoint-based gating structurally cannot see that; only
    /// sampling the interpolated path can. A winner with any frame below
    /// this is dropped outright rather than presented.
    pub min_frame_richness: f32,
    /// Independent structural floor, reusing `vae_explore`'s own
    /// already-calibrated gate type. NEEDED because `min_score` alone can't
    /// catch every degenerate case: `fitness::entropy_score_fast` is
    /// Shannon entropy over a 32-bin escape-time histogram, and once a crop
    /// has collapsed to essentially two occupied bins (e.g. deep in a
    /// smooth gradient near the max-iteration boundary — very ordinary deep
    /// in a fractal, not a rendering bug), entropy plateaus at close to
    /// `1 bit / log2(32) = 0.2` REGARDLESS of whether there's any real
    /// structure — confirmed on a real run (`--method entropy`): from
    /// zoom≈93,634× onward, every level's best candidate scored between
    /// 0.1999 and 0.2000 while `edge_density` collapsed from 0.28 to
    /// 0.01 and `intricacy` hit exactly 0.0 over the same span, and the
    /// deepest point rendered to a visually flat 512×512 PNG (5KB vs. 56KB
    /// for the run's actual best-kept zone). `edge_density`/`intricacy`
    /// don't share entropy's few-bins blind spot (they measure local pixel
    /// variation directly, not histogram spread), so gating on them catches
    /// what a same-metric floor structurally cannot. Defaults mirror
    /// `cmd_vae_explore`'s own already-tuned values for this exact
    /// `coarse_scan`-produced candidate shape.
    pub gate: ZoneGate,
    /// How close to the f64 precision floor a chain may zoom, in ULPs of
    /// pixel step. This is the search's DD-boundary gate; see
    /// [`video_export::needs_dd_with_margin`].
    ///
    /// Defaults to [`DD_MARGIN_ULPS_PIXELATE`] (1.0), NOT the viewer's
    /// conservative 4.0: the exporter pins `allow_dd = false`, so a video
    /// chain never escalates to DD and the only thing stopping at 4 ULPs
    /// buys is 4× less zoom. Carl asked for depth explicitly, accepting
    /// visible f64 pixelation in the final frames (2026-08-17).
    pub dd_margin_ulps: f64,
}

/// Longest contiguous run of LEGS whose every sampled frame passed, given
/// per-frame pass flags spread evenly across `n_legs`. Returns the inclusive
/// leg range, or `None` when no whole leg survives.
///
/// A leg counts only if ALL of its frames pass, so a partly-bad leg is never
/// half-included. Split out as a pure function purely so the trimming policy
/// is unit-testable without a genome, GPU, or ffmpeg — the same reason
/// `drill_chain_generic` is generic over its callbacks.
pub fn usable_leg_span(good: &[bool], n_legs: usize) -> Option<(usize, usize)> {
    if n_legs == 0 || good.is_empty() { return None; }
    let per = good.len() as f64 / n_legs as f64;
    let (mut best, mut cur_lo, mut cur_len) = (None::<(usize, usize)>, 0usize, 0usize);
    for l in 0..n_legs {
        let a = (l as f64 * per).floor() as usize;
        let b = ((((l + 1) as f64 * per).ceil()) as usize).min(good.len());
        let ok = a < b && good[a..b].iter().all(|&g| g);
        if ok {
            if cur_len == 0 { cur_lo = l; }
            cur_len += 1;
            if best.is_none_or(|(blo, bhi)| cur_len > bhi - blo + 1) {
                best = Some((cur_lo, cur_lo + cur_len - 1));
            }
        } else {
            cur_len = 0;
        }
    }
    best
}

/// The zoom at which the search's precision gate stops a chain, for a given
/// export width and ULP margin. Reporting only, but worth having exact: a
/// chain that ends at 1e10 against a 1.7e13 wall ran out of *interesting
/// structure*, while one ending at 1.7e13 ran out of *precision* — and those
/// call for opposite fixes.
///
/// Assumes the square views the search actually produces
/// (`View::new_square` pins `aspect: 1.0`, so `bounds()` spans `4/zoom`) and
/// a coordinate magnitude of 1.0, which `needs_dd_with_margin` clamps up to
/// for the sub-unit coordinates fractal views live at. Pinned against the
/// real gate by `dd_wall_zoom_agrees_with_the_gate`.
pub fn dd_wall_zoom(width: u32, margin_ulps: f64) -> f64 {
    4.0 / (width.max(1) as f64 * f64::EPSILON * margin_ulps.max(f64::MIN_POSITIVE))
}

impl Default for VideoZoomOpts {
    fn default() -> Self {
        VideoZoomOpts {
            // Measured, not guessed: across 12 real runs the deep chains all
            // terminated `DepthReached` at exactly the depth cap, having
            // reached zoom 7.6e7-1.3e10 — the deepest still 179x SHORT of the
            // f64 wall (2.3e12 at 1080px). Steps average 3.1-4.4x zoom each,
            // so ~25 plies are needed for the precision gate rather than this
            // cap to be what ends a chain. 30 leaves headroom for slower
            // steppers; chains that genuinely run out still stop on their own.
            max_depth: 30,
            finalists_per_level: 3,
            lookahead_plies: 2,
            final_export_width: 1280,
            final_export_height: 720,
            canvas_res: VIDEO_ZOOM_CANVAS_RES,
            top_winners: 10,
            lookahead_probe: ProbeSize { w: 128, h: 96, steps: 12, fps: 24 },
            final_probe: ProbeSize { w: 320, h: 240, steps: 48, fps: 24 },
            min_score: 0.15,
            min_file_size_ratio: 0.45,
            min_file_size_step_ratio: 0.80,
            min_step_zoom: 2.0,
            min_frame_richness: 0.30,
            gate: ZoneGate { max_intricacy: 0.30, min_edge_density: 0.05 },
            dd_margin_ulps: DD_MARGIN_ULPS_PIXELATE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndedReason {
    /// Reached `max_depth` real plies with something still eligible.
    DepthReached,
    /// The cheap funnel found nothing above the score floor at this node,
    /// while it (or a shallower ancestor) was still comfortably inside the
    /// DD boundary.
    DeadEnd,
    /// Nothing eligible remained because every candidate here would need
    /// DD precision at the real export width — the intended "end just
    /// before the DD zone" stop condition.
    DdBoundary,
    /// Hit the total-node-visit budget before exhausting depth or
    /// candidates (backtracking explored a lot of dead branches).
    BudgetExhausted,
}

pub struct Winner {
    pub seed_id: usize,
    /// root..leaf, ready to drop straight into `QueueItem::waypoints`.
    pub chain: Vec<CapturedView>,
    /// Sum of each real step's winning lookahead-line ratio — a cheap
    /// signal for pre-sorting survivors before paying for `final_probe`,
    /// not itself the reported score.
    pub cheap_score: f64,
    /// The "official" score: one higher-quality probe over the WHOLE
    /// committed chain. `None` until `run` fills it in for the survivors
    /// of the cheap pre-sort.
    pub final_probe_ratio: Option<f64>,
    pub ended_reason: EndedReason,
}

/// Same two independent reasons the GPU/f32 batch path can't take a render
/// that `vae_explore::recursion_level` already checks inline: `needs_f64`
/// (precision) and a dispatch-size ceiling (`render_gpu::MAX_WORKGROUPS_PER_DIM`)
/// `needs_f64` knows nothing about. Duplicated here (not imported) for the
/// same reason as `BACKTRACK_BUDGET_FACTOR` above — 3 lines, independent
/// engine.
/// "File size entropy" for one view: render small, colormap, PNG-encode,
/// return compressed/raw ratio — `fitness::png_compression_entropy`, this
/// project's own canonical "is this visually interesting" metric (the same
/// one the GA optimizes against), which Carl asked to make the deciding
/// signal here ("always use file size entropy since it is the most
/// effective", 2026-08-13).
fn file_size_entropy(genome: &Genome, config: &Config, view: &View, res: u32, export_w: u32) -> f32 {
    // Render at the VIEW's aspect, not a forced square. `render_escape_times`
    // maps `view.bounds()` straight onto the w×h grid with no letterboxing,
    // so squashing a portrait view (aspect 0.5625) into res×res stretches it
    // 1.78x horizontally — which moves horizontally-adjacent pixels closer
    // together in fractal space and INFLATES the lag-1 horizontal
    // correlation the noise gate is built on. Measured consequence: frames
    // whose real 1080x1920 render has coherence 0.003 (pure dither) scored
    // 0.43 richness here and sailed through the gate, while the exporter's
    // own aspect-preserving `render_save` correctly read them as noise.
    // Candidate views from `coarse_scan` are square, so this is a no-op for
    // them and only changes the frame-validation path — which is exactly the
    // path that was wrong.
    let (pw, ph) = if view.aspect >= 1.0 {
        (res, ((res as f64 / view.aspect).round() as u32).max(1))
    } else {
        (((res as f64 * view.aspect).round() as u32).max(1), res)
    };
    // Optional supersampling of the PROBE, matching what the exporter now
    // does. Keyframe interpolation warps (resamples) most shipped frames,
    // which suppresses aliasing — so a point-sampled probe is PESSIMISTIC
    // about the video that actually gets made, and rejects fine-detail
    // fractals as "noise". Measured on a genome the gate refused outright
    // (2026-08-16): coherence 0.153 point-sampled vs 0.732 at 4x — the
    // structure was real, the sampling was not.
    //
    // Off by default: it costs the SQUARE of the factor on every probe, and
    // probes dominate a search. Opt in with NNFRACTALS_PROBE_SUPERSAMPLE.
    let ss = std::env::var("NNFRACTALS_PROBE_SUPERSAMPLE").ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&f| (1..=4).contains(&f))
        .unwrap_or(1);
    let (rw, rh) = (pw * ss, ph * ss);
    // `export_w`, NOT the probe's own width — the SAME rule already applied to
    // `needs_dd` (which takes `final_export_width`), for the same reason:
    // `needs_f64`'s threshold scales with width, so a narrow probe stays in
    // f32 far longer than the wide export does. At FILE_SIZE_PROBE_RES=128 a
    // portrait frame is only 72px wide and switches to f64 at zoom ~4096,
    // while the 1080px export switches at ~273. In that window the probe
    // renders in f32 and its accumulated iteration error decorrelates
    // adjacent pixels into dither, which the noise gate below correctly
    // identifies as noise and scores 0.0 — rejecting a frame the exporter
    // renders perfectly cleanly in f64.
    //
    // Measured on f12 (2026-08-17): the probe rejected a frame at zoom 2294
    // with richness EXACTLY 0.0, truncating a 23-leg chain to 5 legs. The
    // identical frame rendered at the real 1080x1920 export geometry scored
    // noisy-tile fraction 0.000 — VERDICT ALIVE across the whole span. The
    // `.max(rw)` keeps the stricter of the two, so a probe wider than the
    // export never silently loses precision either.
    let use_f64 = needs_f64(view, rw.max(export_w));
    // Render on EXACTLY the exporter's terms — same precision tier
    // (`VIDEO_FRAME_ALLOW_DD`) and same zoom-scaled iteration depth
    // (`effective_max_iter`). Both were previously wrong here, and each
    // caused the same class of bug: a score describing an image the video
    // will never contain.
    //   * allow_dd:true let the probe use double-double precision the
    //     exporter never uses, so it scored rich frames past the f64 wall.
    //   * a fixed max_iter made the probe blind to iteration starvation —
    //     the actual cause of Carl's flat-tailed videos, since past ~1e11
    //     zoom every pixel hits the cap and the frame goes single-colour.
    let eff_iter = effective_max_iter(view, config.rendering.max_iter);
    let raw = render_escape_times(
        genome, config, view, rw, rh, eff_iter, use_f64, VIDEO_FRAME_ALLOW_DD,
    );
    // Box-average back down to pw x ph so every downstream metric sees the
    // anti-aliased field, at the resolution they were calibrated for.
    let field: Vec<f32> = if ss > 1 {
        let (pw_u, ph_u, ss_u) = (pw as usize, ph as usize, ss as usize);
        let rw_u = rw as usize;
        let n = (ss_u * ss_u) as f32;
        (0..ph_u * pw_u).map(|i| {
            let (x, y) = (i % pw_u, i / pw_u);
            let mut sum = 0.0f32;
            for oy in 0..ss_u {
                for ox in 0..ss_u {
                    sum += raw[(y * ss_u + oy) * rw_u + (x * ss_u + ox)];
                }
            }
            sum / n
        }).collect()
    } else { raw };
    // `multiscale_entropy`, NOT plain `png_compression_entropy` — this is
    // the fix for Carl's persistent report that the search "still focuses
    // on a non-entropic zone" (2026-08-14). Raw compression entropy is
    // MAXIMIZED by noise: granular dither/aliasing in a visually flat
    // region is incompressible, so it scores as high as genuine structure.
    // That's not a subtle effect here — it's the metric's defining failure
    // mode, and this project already solved it: `multiscale_entropy`
    // geometric-means the full-res score with a 4x-average-pooled coarse
    // one, so noise (which averages to near-uniform, collapsing the coarse
    // term) is penalised while structure (complex at every scale) keeps
    // both terms high. The GA itself scores on this for the same reason;
    // using the raw version here was the mistake.
    // HARD noise gate, before any entropy scoring. A compression-ratio
    // objective is maximised by random dither, so without this the search
    // actively seeks noise out: on Carl's first batch run the #1 winner was
    // a pure speckle field (spatial coherence 0.027 vs 0.26+ for real
    // structure). `multiscale_entropy` alone does NOT catch it — dense
    // speckle survives its 4x pooling — so this is a separate, independent
    // floor, the same way `ZoneGate` is an absolute floor next to the
    // relative z-score ranking.
    //
    // Measured on the COLORMAPPED luminance, not the raw escape-time field.
    // The field can carry a smooth large-scale gradient that reads as
    // coherent while the palette turns its chaotic fine component into
    // visual noise: on a real chain the field scored 0.46 richness (gate
    // passed) for frames whose shipped RGB measured 0.068 coherence — pure
    // dither. What ships is the colormapped image, so that is what has to be
    // judged. Same principle as matching the exporter's precision tier and
    // iteration depth above.
    let rgb = crate::colormap::apply_colormap(&field, eff_iter, &config.rendering.colormap);
    let lum: Vec<f32> = rgb.chunks_exact(3)
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect();
    // TILED, not whole-frame: a frame that is half smooth gradient and half
    // dither averages to a healthy whole-frame score (a real one measured
    // 0.564 while being visibly half garbage). See `noise_tile_fraction`.
    if fitness::noise_tile_fraction(&lum, pw, ph) > fitness::MAX_NOISE_TILE_FRACTION {
        return 0.0;
    }
    // `eff_iter`, not the base — the colormap normalisation cap must match
    // the cap the field was actually computed with, exactly as `render_save`
    // does for the shipped frame.
    fitness::multiscale_entropy(&field, pw, ph, eff_iter, &config.rendering.colormap)
}

fn canvas_needs_cpu_tier(view: &View, canvas_res: u32) -> bool {
    let pixel_count = canvas_res as u64 * canvas_res as u64;
    let gpu_dispatch_limit = render_gpu::WG_SIZE as u64 * render_gpu::MAX_WORKGROUPS_PER_DIM as u64;
    needs_f64(view, canvas_res) || pixel_count > gpu_dispatch_limit
}

/// One node's immediate candidates: fresh canvas render (same CANVAS_RES-
/// tier precision handling `vae_explore::recursion_level` uses — this is
/// deliberately `vae_explore`'s canvas/`coarse_scan` engine, never
/// `explore::sweep`, whose `GPU_MAX_ZOOM` cap sits ~10⁹× shallower than
/// where the DD wall actually is) → `coarse_scan` → two-tier selection:
///
/// 1. **Cheap, broad pre-filter** (free — reuses metrics `coarse_scan`
///    already computed): z-score rank, then drop anything below
///    `opts.min_score` OR outside `opts.gate` OR past `opts.final_export_width`'s
///    DD boundary (see those fields' doc comments — this is what lets a
///    genuinely bad neighborhood register as a dead end instead of always
///    finding a "least-bad" survivor), then keep only the top
///    `FILE_SIZE_ENTROPY_CANDIDATES`. This is a safety net, NOT the
///    deciding signal — coarse_scan's histogram-entropy-based ranking is
///    proven blind to a real degenerate case (see `VideoZoomOpts::gate`'s
///    doc comment) and, per Carl's real-world comparison against hand-picked
///    zooms, produces visibly less interesting results than compression-
///    based scoring even where it isn't outright wrong.
/// 2. **File-size entropy — the actual deciding signal** (Carl's explicit
///    instruction, 2026-08-13: "Please always use file size entropy since
///    it is the most effective"): a fresh small render + PNG-compression-
///    ratio per tier-1 survivor, the exact same "compress it for real, use
///    the resulting ratio" recipe as `fitness::png_compression_entropy`
///    (this project's own canonical, already-proven "is this visually
///    interesting" metric — used by the GA's fitness function and
///    Auto-Select) applied per-candidate here. Ranks survivors by this,
///    keeps the top `opts.finalists_per_level`.
///
/// Logs a `"level_scanning"` event in the same name/shape `vae_explore.rs`
/// already uses (plus `file_size_score`/`coarse_score`/entropy/edge_density/
/// intricacy per candidate, which `vae_explore.rs`'s version doesn't log —
/// added so a bad run is diagnosable straight from the log instead of
/// needing to re-render candidates by hand, PLUS `is_real_step` — see
/// `zoom_level`'s doc comment for why that distinction matters) so the
/// viewer's existing scan overlay (polls the log file for exactly this
/// event) works against this stage unmodified; `gate_pass` there now means
/// "ended up in the actual finalist set; not just cleared the tier-1 floor.
/// Called once for the immediate ply and again for every greedy lookahead
/// extension step in `zoom_level` — the overlay showing that lookahead
/// activity too is free, desired feedback, not a leak.
#[allow(clippy::too_many_arguments)]
fn cheap_funnel(
    genome: &Genome, config: &Config, current: &View, method: ScoreMethod,
    opts: &VideoZoomOpts, log: &mut Logger, is_real_step: bool, seed_floor: f32,
) -> Vec<View> {
    // Hybrid floor: the stricter of the run-wide seed anchor (bounds TOTAL
    // degradation) and a per-step anchor off this node's own parent (bounds
    // per-step cliffs). Either alone is measurably wrong — see
    // `min_file_size_ratio`'s doc comment for the sweep data.
    let parent_score = file_size_entropy(genome, config, current, FILE_SIZE_PROBE_RES, opts.final_export_width);
    let file_size_floor = seed_floor.max(parent_score * opts.min_file_size_step_ratio);
    let use_f64 = canvas_needs_cpu_tier(current, opts.canvas_res);
    let canvas_field = render_escape_times(
        genome, config, current, opts.canvas_res, opts.canvas_res,
        config.rendering.max_iter, use_f64, true,
    );
    let candidates: Vec<Candidate> =
        coarse_scan(&canvas_field, opts.canvas_res, opts.canvas_res, current, config, method);
    let total_candidates = candidates.len();
    let ranked = rank_by_zscore(&candidates, 0.0);

    let broad: Vec<(Candidate, View)> = ranked
        .into_iter()
        .map(|c| {
            let v = apply_offset(current, c.dx, c.dy, c.zoom);
            (c, v)
        })
        .filter(|(c, v)| {
            c.score >= opts.min_score
                && c.metrics.edge_density >= opts.gate.min_edge_density
                && c.metrics.intricacy <= opts.gate.max_intricacy
                && v.zoom >= current.zoom * opts.min_step_zoom
                && !needs_dd_with_margin(v, opts.final_export_width, opts.dd_margin_ulps)
        })
        .take(FILE_SIZE_ENTROPY_CANDIDATES)
        .collect();

    let mut scored: Vec<(Candidate, View, f32)> = broad
        .into_iter()
        .map(|(c, v)| {
            let file_size_score = file_size_entropy(genome, config, &v, FILE_SIZE_PROBE_RES, opts.final_export_width);
            (c, v, file_size_score)
        })
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    // THE gate, not just a ranking — see `min_file_size_ratio`'s doc
    // comment. Applied after sorting so the log below still shows the
    // best-scoring rejects (what the search *considered* and turned down
    // is the diagnostic that matters when a run dead-ends early).
    let n_above_floor = scored.iter().filter(|(_, _, s)| *s >= file_size_floor).count();

    vztrace!("[TRACE cheap_funnel] scan_view=({:.9},{:.9})@{:.4e} parent_score={:.4} floor={:.4} real={} cands={} above_floor={}",
        current.cx, current.cy, current.zoom, parent_score, file_size_floor, is_real_step, total_candidates, n_above_floor);
    for (i, (_, v, sc)) in scored.iter().take(3).enumerate() {
        vztrace!("[TRACE cheap_funnel]    cand[{i}] ({:.9},{:.9})@{:.4e} score={:.4} {}",
            v.cx, v.cy, v.zoom, sc, if *sc >= file_size_floor { "PASS" } else { "reject" });
    }
    log.log(&serde_json::json!({
        "event": "level_scanning",
        "seed_cx": current.cx, "seed_cy": current.cy, "seed_zoom": current.zoom,
        "is_real_step": is_real_step,
        "total_candidates": total_candidates,
        "file_size_floor": file_size_floor,
        "parent_file_size_score": parent_score,
        "seed_floor": seed_floor,
        "n_above_floor": n_above_floor,
        "candidates": scored.iter().take(LOG_CANDIDATE_CAP).enumerate().map(|(i, (c, v, file_size_score))| serde_json::json!({
            "cx": v.cx, "cy": v.cy, "zoom": v.zoom,
            "file_size_score": file_size_score, "coarse_score": c.score,
            "entropy": c.metrics.entropy, "edge_density": c.metrics.edge_density,
            "intricacy": c.metrics.intricacy,
            "gate_pass": *file_size_score >= file_size_floor && i < opts.finalists_per_level,
        })).collect::<Vec<_>>(),
    }));

    scored
        .into_iter()
        .filter(|(_, _, s)| *s >= file_size_floor)
        .take(opts.finalists_per_level)
        .map(|(_, v, _)| v)
        .collect()
}

/// Ranks `current`'s immediate children by looking `opts.lookahead_plies`
/// moves ahead before committing to the first move — the chess-engine
/// pattern of "search several plies deep, play only the first move, then
/// re-search after actually reaching the new position." Only the immediate
/// ply branches (`opts.finalists_per_level` candidates); each candidate's
/// own continuation is then extended GREEDILY — single best (DD-eligible)
/// child only, no further branching — for the remaining plies, so cost
/// grows as `finalists_per_level * lookahead_plies`, not exponentially. A
/// line that runs out of eligible continuations early (dead end, or every
/// remaining option needs DD) is scored on whatever partial line it has —
/// this is what makes "stop just before the DD zone" apply at every
/// simulated ply, not only the committed one. Returns immediate children
/// ranked by their own line's normalized probe score, best first.
///
/// `is_real_step` distinguishes an actual decision point (`drill_chain_generic`
/// is about to commit to whichever child wins) from a hypothetical
/// lookahead-extension call (evaluating a candidate line that may never be
/// taken) — both call this same function, and without the distinction
/// there's no way to tell, from the log alone, which scan activity is real
/// progress vs. exploratory noise. This matters beyond diagnostics: the
/// viewer's live-progress panel (zone count + "most recently found" marker)
/// is entirely driven by `vae_explore`'s save-a-numbered-file-per-zone
/// convention, which this module has no equivalent of (nothing is saved
/// until the very end) — so without a distinguishable "a real move just
/// happened" signal, a healthy, correctly-progressing search can look
/// indistinguishable from a stuck one purely because the viewer has nothing
/// positive to show, only a cloud of scan squares including many that will
/// never be taken. Only the immediate-ply call is real; every lookahead
/// extension is definitionally hypothetical.
#[allow(clippy::too_many_arguments)]
fn zoom_level(
    genome: &Genome, config: &Config, angle_coloring: bool, current: &View,
    method: ScoreMethod, opts: &VideoZoomOpts, log: &mut Logger, is_real_step: bool,
    file_size_floor: f32,
) -> Vec<(View, f64)> {
    let immediate = cheap_funnel(genome, config, current, method, opts, log, is_real_step, file_size_floor);

    let mut scored: Vec<(View, f64)> = immediate
        .into_iter()
        .map(|candidate| {
            let mut line = vec![CapturedView::from_view(current), CapturedView::from_view(&candidate)];
            let mut tail = candidate.clone();
            for _ply in 1..opts.lookahead_plies {
                match cheap_funnel(genome, config, &tail, method, opts, log, false, file_size_floor).into_iter().next() {
                    Some(next) => {
                        line.push(CapturedView::from_view(&next));
                        tail = next;
                    }
                    None => break,
                }
            }
            let ratio = probe_video_score(
                genome, config, angle_coloring, &line,
                opts.lookahead_probe.steps, opts.lookahead_probe.fps,
                opts.lookahead_probe.w, opts.lookahead_probe.h,
            )
            .unwrap_or(0.0);
            (candidate, ratio)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Fires once per REAL decision point (never for a lookahead extension),
    // exactly when `drill_chain_generic` is about to commit to `scored[0]`
    // — see this function's doc comment for why the viewer needs this
    // distinct signal. Logs the winner even though a later backtrack could
    // in principle still abandon it — same "show the current best guess"
    // convention a live chess-engine eval display uses, and correct for
    // this feature's purpose (reassuring the user real progress is
    // happening), not a claim the winner is final.
    if is_real_step {
        for (i, (v, r)) in scored.iter().enumerate() {
            vztrace!("[TRACE zoom_level] finalist[{i}] ({:.9},{:.9})@{:.4e} video_ratio={:.6}", v.cx, v.cy, v.zoom, r);
        }
    }
    if is_real_step && let Some((winner, ratio)) = scored.first() {
        log.log(&serde_json::json!({
            "event": "committed_move",
            "cx": winner.cx, "cy": winner.cy, "zoom": winner.zoom, "lookahead_ratio": ratio,
        }));
    }

    scored
}

/// Generic backtracking-DFS — same LIFO-stack / `budget = depth *
/// BACKTRACK_BUDGET_FACTOR` shape as `vae_explore::recursive_drill`, but
/// (a) tracks the full `Vec<CapturedView>` path alongside each stack entry
/// (`recursive_drill` only needs a save-count, since it saves as a side
/// effect instead) and a running sum of each step's winning score, and
/// (b) generalized over an `expand`/`is_dd_limited` callback pair so this
/// is unit-testable with a synthetic tree — no Genome/GPU/ffmpeg needed.
/// `expand(&view)` must return this node's children ranked BEST FIRST;
/// every non-winner child becomes a same-level backtrack sibling (every
/// finalist `zoom_level` produces has already paid for a video probe, so
/// there's no reason to discard some of them with a second, smaller cap —
/// unlike `vae_explore`'s separate `top_k`/`MAX_ALTERNATES` knobs).
pub(crate) fn drill_chain_generic<F, G>(
    seed: View, max_depth: usize, mut expand: F, mut is_dd_limited: G,
) -> Vec<(Vec<CapturedView>, f64, EndedReason)>
where
    F: FnMut(&View) -> Vec<(View, f64)>,
    G: FnMut(&View) -> bool,
{
    let budget = max_depth.saturating_mul(BACKTRACK_BUDGET_FACTOR).max(max_depth);
    let mut stack: Vec<(usize, View, Vec<CapturedView>, f64)> =
        vec![(max_depth, seed.clone(), vec![CapturedView::from_view(&seed)], 0.0)];
    let mut out = Vec::new();
    let mut visited = 0usize;

    while let Some((remaining, view, chain, score_sum)) = stack.pop() {
        if remaining == 0 {
            out.push((chain, score_sum, EndedReason::DepthReached));
            continue;
        }
        if visited >= budget {
            out.push((chain, score_sum, EndedReason::BudgetExhausted));
            continue;
        }
        visited += 1;

        vztrace!("[TRACE drill] POP view=({:.9},{:.9})@{:.4e} remaining={} chain_len={} visited={}",
            view.cx, view.cy, view.zoom, remaining, chain.len(), visited);
        let children = expand(&view);
        if children.is_empty() {
            let reason = if is_dd_limited(&view) { EndedReason::DdBoundary } else { EndedReason::DeadEnd };
            out.push((chain, score_sum, reason));
            continue;
        }

        // Siblings pushed first (bottom of this batch) at the SAME
        // remaining depth, so they're only tried once the winner's whole
        // subtree below is exhausted — best-first, depth-first descent.
        for (child_view, child_score) in children.iter().skip(1).rev() {
            let mut c = chain.clone();
            c.push(CapturedView::from_view(child_view));
            // `remaining - 1`, NOT `remaining`: a backtrack sibling is an
            // ALTERNATIVE to the winner at the same tree depth, not a step
            // before it. Pushing at unchanged `remaining` while still
            // appending the child to the chain made depth and path length
            // disagree — every lateral move granted a free extra level, so
            // chain length was bounded only by the visit budget
            // (`depth * BACKTRACK_BUDGET_FACTOR`) instead of by `max_depth`.
            // Traced on a real run (2026-08-14): `--depth 3` produced a
            // 10-waypoint chain zooming to 1.5e5 — nine steps on a
            // three-step budget. That overshoot is what drove chains past
            // rich structure into flat interior, i.e. Carl's "completely
            // flat after not even 1/4 of the zoom"; no scoring change could
            // fix it because each individual step passed its own gate.
            stack.push((remaining - 1, child_view.clone(), c, score_sum + child_score));
        }
        let (winner_view, winner_score) = &children[0];
        let mut c = chain;
        c.push(CapturedView::from_view(winner_view));
        vztrace!("[TRACE drill] COMMIT from=({:.9},{:.9})@{:.4e} -> ({:.9},{:.9})@{:.4e} chain_len={} remaining={}",
            view.cx, view.cy, view.zoom, winner_view.cx, winner_view.cy, winner_view.zoom, c.len(), remaining - 1);
        stack.push((remaining - 1, winner_view.clone(), c, score_sum + winner_score));
    }
    out
}

/// Thin concrete wrapper: wires `zoom_level` in as `expand` and
/// `needs_dd(_, final_export_width)` as `is_dd_limited`, converts completed
/// paths into `Winner`s. Drops any path that never made a single real move
/// (e.g. the seed itself already needs DD at the target export width).
#[allow(clippy::too_many_arguments)]
pub fn explore_from_seed(
    genome: &Genome, config: &Config, angle_coloring: bool, seed: View,
    method: ScoreMethod, opts: &VideoZoomOpts, seed_id: usize, log: &mut Logger,
) -> Vec<Winner> {
    // Progress signal for the viewer (Carl's request, 2026-08-13: "give me
    // an idea of the progress of the search") — logs the total visit
    // BUDGET this seed can spend (same formula `drill_chain_generic` uses
    // internally) once, up front, so the viewer can show "N of BUDGET real
    // moves committed" by just counting this seed's own `committed_move`
    // events afterward, without needing to duplicate the budget formula
    // itself or guess at how close to done a run is.
    let budget = opts.max_depth.saturating_mul(BACKTRACK_BUDGET_FACTOR).max(opts.max_depth);
    // Absolute file-size-entropy floor for this whole seed's descent,
    // anchored to the seed's OWN score — see `min_file_size_ratio`.
    let seed_score = file_size_entropy(genome, config, &seed, FILE_SIZE_PROBE_RES, opts.final_export_width);
    let file_size_floor = seed_score * opts.min_file_size_ratio;
    log.log(&serde_json::json!({
        "event": "seed_started", "seed_id": seed_id,
        "max_depth": opts.max_depth, "budget": budget,
        "seed_file_size_score": seed_score, "file_size_floor": file_size_floor,
    }));

    let results = drill_chain_generic(
        seed,
        opts.max_depth,
        // Every call `drill_chain_generic` makes through this closure is,
        // by construction, evaluating a real decision point (it's the only
        // place `zoom_level` is invoked as `expand` — every OTHER call
        // happens inside `zoom_level` itself, for lookahead extensions).
        |v| zoom_level(genome, config, angle_coloring, v, method, opts, log, true, file_size_floor),
        |v| needs_dd_with_margin(v, opts.final_export_width, opts.dd_margin_ulps),
    );
    results
        .into_iter()
        .filter(|(chain, _, _)| chain.len() >= 2)
        .map(|(chain, cheap_score, ended_reason)| Winner {
            seed_id, chain, cheap_score, final_probe_ratio: None, ended_reason,
        })
        .collect()
}

/// Runs `explore_from_seed` over every given seed, cheap-pre-sorts by
/// summed lookahead scores, truncates to `top_winners * 3` slack, then
/// renders ONE higher-quality probe per survivor over its *whole*
/// committed chain (`probe_video_score_keep`, so the clip survives on disk
/// for review) — a chain through the actually-discovered intermediate
/// points, not a naive straight-line start→end lerp. Re-sorts by that
/// "official" score and truncates to `top_winners`. Scratch clips for
/// non-surviving pre-sort candidates are never written in the first place
/// (only survivors get a `final_probe_*` call at all).
#[allow(clippy::too_many_arguments)]
pub fn run(
    genome: &Genome, config: &Config, angle_coloring: bool, seeds: &[View],
    method_arg: &str, opts: &VideoZoomOpts, out_dir: &Path, log: &mut Logger,
) -> Vec<Winner> {
    init_trace(out_dir);
    let exe = std::env::current_exe().ok();
    let built = exe.as_ref()
        .and_then(|e| std::fs::metadata(e).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    vztrace!("=== video-zoom-explore trace ===");
    vztrace!("[BUILD] exe={:?} mtime_epoch={:?}  (confirm this matches the intended build)", exe, built);
    vztrace!("[CONFIG] max_depth={} finalists={} lookahead_plies={} canvas_res={} final_export_width={}",
        opts.max_depth, opts.finalists_per_level, opts.lookahead_plies, opts.canvas_res, opts.final_export_width);
    // The zoom this run is allowed to reach before the precision gate stops
    // it — logged as an absolute number so a chain that ends short is
    // immediately distinguishable from one that ran into the f64 wall.
    vztrace!("[CONFIG] dd_margin_ulps={} -> f64 wall at zoom {:.3e} (width {})",
        opts.dd_margin_ulps,
        dd_wall_zoom(opts.final_export_width, opts.dd_margin_ulps),
        opts.final_export_width);
    vztrace!("[CONFIG] min_score={} seed_ratio={} step_ratio={} min_step_zoom={} gate(max_intr={}, min_edge={})",
        opts.min_score, opts.min_file_size_ratio, opts.min_file_size_step_ratio, opts.min_step_zoom,
        opts.gate.max_intricacy, opts.gate.min_edge_density);
    vztrace!("[CONFIG] lookahead_probe={}x{}@{}f/{}fps final_probe={}x{}@{}f/{}fps method={} n_seeds={} top_winners={}",
        opts.lookahead_probe.w, opts.lookahead_probe.h, opts.lookahead_probe.steps, opts.lookahead_probe.fps,
        opts.final_probe.w, opts.final_probe.h, opts.final_probe.steps, opts.final_probe.fps,
        method_arg, seeds.len(), opts.top_winners);
    vztrace!("[CONFIG] max_iter={} colormap={}", config.rendering.max_iter, config.rendering.colormap);

    let mut all: Vec<Winner> = Vec::new();
    for (seed_id, seed) in seeds.iter().enumerate() {
        let method = match method_arg {
            "mixed" => ScoreMethod::ALL[seed_id % ScoreMethod::ALL.len()],
            s => ScoreMethod::parse(s).unwrap_or(ScoreMethod::GatedEntropy),
        };
        all.extend(explore_from_seed(genome, config, angle_coloring, seed.clone(), method, opts, seed_id, log));
    }
    all.sort_by(|a, b| b.cheap_score.partial_cmp(&a.cheap_score).unwrap_or(std::cmp::Ordering::Equal));
    let slack = opts.top_winners.max(1) * 3;
    all.truncate(slack);

    let _ = std::fs::create_dir_all(out_dir);
    for (i, w) in all.iter_mut().enumerate() {
        let scratch = out_dir.join(format!("_vz_scratch_{i:04}.mp4"));
        w.final_probe_ratio = probe_video_score_keep(
            genome, config, angle_coloring, &w.chain,
            opts.final_probe.steps, opts.final_probe.fps, opts.final_probe.w, opts.final_probe.h, &scratch,
        );
    }
    for (i, w) in all.iter().enumerate() {
        vztrace!("[SURVIVOR {i}] legs={} final_ratio={:?} ended={:?}", w.chain.len()-1, w.final_probe_ratio, w.ended_reason);
        for (j, wp) in w.chain.iter().enumerate() {
            // Re-score each waypoint as the video will actually see it, so a
            // flat leg is visible directly in the log with no re-rendering.
            let v = View { cx: wp.cx, cx_lo: wp.cx_lo, cy: wp.cy, cy_lo: wp.cy_lo, zoom: wp.zoom, aspect: wp.aspect };
            let rich = file_size_entropy(genome, config, &v, FILE_SIZE_PROBE_RES, opts.final_export_width);
            let dd = needs_dd_with_margin(&v, opts.final_export_width, opts.dd_margin_ulps);
            vztrace!("[SURVIVOR {i}]    wp[{j}] ({:.9},{:.9})@{:.4e} richness={:.4}{}{}",
                wp.cx, wp.cy, wp.zoom, rich,
                if rich < 0.15 { "  <<< FLAT" } else { "" },
                if dd { "  <<< PAST-DD (video renders f64-capped)" } else { "" });
            // Sample INSIDE the leg: the exported video interpolates between
            // waypoints (`lerp_view`, the same call the exporter makes), so a
            // leg can pass through flat territory even when both of its
            // endpoints are rich. Waypoint-only logging would miss exactly
            // that, and it is a live explanation for "video goes flat" with
            // healthy-looking chain scores.
            if j + 1 < w.chain.len() {
                for t in [0.25_f64, 0.5, 0.75] {
                    let mid = lerp_view(&w.chain[j], &w.chain[j + 1], t);
                    let r = file_size_entropy(genome, config, &mid, FILE_SIZE_PROBE_RES, opts.final_export_width);
                    vztrace!("[SURVIVOR {i}]       leg{j}+t={t:.2} ({:.9},{:.9})@{:.4e} richness={:.4}{}",
                        mid.cx, mid.cy, mid.zoom, r, if r < 0.15 { "  <<< FLAT MID-LEG" } else { "" });
                }
            }
        }
    }
    // Per-frame validation of the ACTUAL exported path (Carl's requirement,
    // 2026-08-14: "verify every frame for valid entropy"). Samples each leg
    // with the same `lerp_view` the exporter calls, so a chain whose camera
    // flies through empty space between two rich waypoints is caught here
    // and dropped, instead of being presented as a winner and only failing
    // when he watches it.
    const FRAME_SAMPLES_PER_LEG: usize = 8;
    // TRUNCATE at the first bad leg rather than discarding the whole chain.
    //
    // A 15-leg chain that is clean for 6 legs and then enters a chaotic
    // region is a perfectly good 6-leg zoom video; throwing it away turns a
    // usable result into nothing. Measured on a real genome (2026-08-15):
    // every one of 18 candidate chains was clean to roughly zoom 900 and
    // noisy beyond it, so a drop-only policy returned NO-WINNERS while a
    // truncating one returns real videos that simply stop before the mess.
    //
    // Validates the EXACT frames the exporter will render, via its own
    // generator (`chain_frame_views`) at the real output geometry.
    // Re-deriving them with a bare `lerp_view` over the raw waypoints
    // validated a DIFFERENT crop — waypoints are captured square while the
    // export applies the output aspect — so a chain could pass every gate
    // and still ship noise.
    let mut truncated: Vec<Winner> = Vec::new();
    for mut w in all.into_iter() {
        let n_legs = w.chain.len().saturating_sub(1);
        if n_legs == 0 { continue; }
        let frames = crate::video_export::chain_frame_views(
            &w.chain, (FRAME_SAMPLES_PER_LEG * n_legs) as u32,
            opts.final_export_width, opts.final_export_height, false, false,
        );
        if frames.is_empty() { continue; }

        // Score EVERY frame, then keep the longest contiguous run of good
        // LEGS — rather than only the clean prefix, stopping at the first
        // bad frame.
        //
        // Prefix-only was wrong at BOTH ends, and both failures are real:
        //
        //  * BAD OPENING (Carl, 2026-08-17): a seed view that is zoomed OUT
        //    (zoom 0.22) is legitimately sparse and scored 0.1497 against the
        //    0.30 floor, so `leg == 0` dropped the chain outright. All 30
        //    survivors of a 20-leg search reaching zoom 9.8e11 — every
        //    waypoint after the first scoring 0.71-0.85 — were discarded
        //    because of the establishing shot alone. Every chain shares the
        //    seed, so this cannot fail partially: it always returns ZERO
        //    winners, which reads as "the search found nothing".
        //
        //  * BAD TAIL: past the f64 precision limit `View::bounds()` collapses
        //    and frames degenerate. Keeping the last WORKING zoom is what Carl
        //    asked for, instead of failing the chain for continuing too far.
        //
        // A leg is kept only if ALL of its frames pass, so a partially-bad leg
        // is never half-included.
        let mut worst = f32::INFINITY;
        let good: Vec<bool> = frames.iter().map(|v| {
            let r = file_size_entropy(genome, config, v, FILE_SIZE_PROBE_RES, opts.final_export_width);
            if r < worst { worst = r; }
            r >= opts.min_frame_richness
        }).collect();

        let Some((lo, hi)) = usable_leg_span(&good, n_legs) else {
            vztrace!("[FRAME-VALIDATE] legs={n_legs} worst_frame_richness={worst:.4} -> DROP (no usable leg)");
            continue;
        };
        let best_len = hi - lo + 1;
        if best_len == n_legs {
            vztrace!("[FRAME-VALIDATE] legs={n_legs} worst_frame_richness={worst:.4} -> KEEP (all clean)");
        } else {
            // Waypoints lo..=hi+1 bound the kept legs. Trim the tail first so
            // the head trim's indices stay valid.
            w.chain.truncate(hi + 2);
            w.chain.drain(..lo);
            // Only a trimmed TAIL means the chain stopped early; a trimmed
            // head just means the opening was sparse, so the original reason
            // (DepthReached / DdBoundary) still describes how it ended.
            if hi + 1 < n_legs { w.ended_reason = EndedReason::DeadEnd; }
            vztrace!("[FRAME-VALIDATE] legs={n_legs} -> KEEP legs {lo}..={hi} ({best_len} clean, \
                     trimmed {} from head / {} from tail), zoom {:.4e} -> {:.4e}",
                lo, n_legs - 1 - hi, w.chain[0].zoom, w.chain[w.chain.len() - 1].zoom);
        }
        truncated.push(w);
    }
    let mut all = truncated;

    all.sort_by(|a, b| {
        b.final_probe_ratio.unwrap_or(0.0)
            .partial_cmp(&a.final_probe_ratio.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Spatial dedup, best-first — see `WINNER_MIN_SEPARATION_FRAMES`.
    // Without this the "winners" list is one trunk repeated at different
    // stop depths, which makes the whole present-the-winners step useless.
    let mut diverse: Vec<Winner> = Vec::new();
    for w in all {
        let Some(end) = w.chain.last() else { continue };
        if diverse.iter().all(|k: &Winner| !same_endpoint(k.chain.last().expect("non-empty"), end)) {
            diverse.push(w);
            if diverse.len() >= opts.top_winners.max(1) { break; }
        }
    }
    diverse
}

/// Writes `video_zoom_winners.jsonl` (one manifest per run, truncated not
/// appended — unlike vae-explore's growing corpus, this is a single ranked
/// result set), a per-winner still thumbnail (`explore::save_shot` on the
/// chain's deepest view), and renames each surviving winner's scratch clip
/// (written by `run`, named by its pre-truncation index — matched here via
/// `final_probe_ratio` identity since that's the only handle `Winner`
/// carries back to it) to `winner_%04d.mp4`. Deletes any leftover
/// `_vz_scratch_*.mp4` from `run`'s pre-sort losers, which never got a
/// `final_probe_*` call and so never had a scratch file written for them
/// in the first place — this cleans up only genuine same-run leftovers
/// from a previous invocation's truncated tail, if `out_dir` is reused.
pub fn write_winners_manifest(
    out_dir: &Path, winners: &[Winner], genome: &Genome, config: &Config, angle_coloring: bool,
) -> std::io::Result<()> {
    let mut log = Logger::new(&out_dir.join("video_zoom_winners.jsonl"))?;
    for (rank, w) in winners.iter().enumerate() {
        let last = w.chain.last().expect("filtered to len >= 2 in explore_from_seed");
        let end_view = View {
            cx: last.cx, cx_lo: last.cx_lo, cy: last.cy, cy_lo: last.cy_lo,
            zoom: last.zoom, aspect: last.aspect,
        };
        let thumb_path = out_dir.join(format!("winner_{rank:04}_end.png"));
        // The thumbnail must be in the colouring the search actually
        // scored and the export will actually use — a gallery of standard
        // -palette thumbnails for an angle-coloured chain shows the user a
        // video that will not be produced.
        if angle_coloring {
            let rgb = crate::video_export::render_save(
                genome, config, &end_view, 512, 512, true, crate::video_export::VIDEO_FRAME_ALLOW_DD);
            let _ = crate::io::save_png(&rgb, 512, 512, &thumb_path);
        } else {
            save_shot(genome, config, &end_view, 512, &thumb_path);
        }

        let final_path = out_dir.join(format!("winner_{rank:04}.mp4"));
        // `run` names scratch clips by pre-truncation index, not by this
        // manifest's final `rank` — hunt the one whose modify time is
        // newest among not-yet-claimed scratch files is fragile, so
        // instead `run` is required to hand back winners in the same
        // relative order it wrote scratch files for its slack list, and
        // this loop simply consumes them in that same order.
        let _ = std::fs::rename(out_dir.join(format!("_vz_scratch_{rank:04}.mp4")), &final_path);

        log.log(&serde_json::json!({
            "event": "winner", "rank": rank, "seed_id": w.seed_id,
            "n_legs": w.chain.len() - 1,
            "final_probe_ratio": w.final_probe_ratio,
            "cheap_score": w.cheap_score,
            "ended_reason": format!("{:?}", w.ended_reason),
            "chain": w.chain.iter().map(|c| serde_json::json!({
                "cx": c.cx, "cx_lo": c.cx_lo, "cy": c.cy, "cy_lo": c.cy_lo, "zoom": c.zoom, "aspect": c.aspect,
            })).collect::<Vec<_>>(),
            "preview_mp4": format!("winner_{rank:04}.mp4"),
            "thumb_png": format!("winner_{rank:04}_end.png"),
            // Recorded so a later render reproduces what was SEARCHED. The
            // video probe scores a real encode, and angle colouring changes
            // the encoded bytes substantially — a chain picked under one
            // colouring was not evaluated under the other.
            "angle_coloring": angle_coloring,
        }));
    }
    // Anything still named `_vz_scratch_*` at this point belongs to a
    // stale previous run in the same `out_dir` (this run's own scratch
    // files were all consumed by the rename loop above, in order).
    if let Ok(entries) = std::fs::read_dir(out_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("_vz_scratch_") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_export::DD_MARGIN_ULPS;

    fn v(id: i64) -> View {
        // Encode a small integer node id as `cx` so the synthetic `expand`
        // closures below can key off it without needing a real fractal.
        View { cx: id as f64, cx_lo: 0.0, cy: 0.0, cy_lo: 0.0, zoom: 1.0, aspect: 1.0 }
    }
    fn id_of(view: &View) -> i64 {
        view.cx as i64
    }

    /// The probe must make its f32-vs-f64 decision on the EXPORT's width, not
    /// its own. `needs_f64`'s threshold scales with width, so judging by a
    /// 72px probe leaves it in f32 across a zoom window where the 1080px
    /// export is already in f64 — and f32 iteration error there reads as
    /// dither to the noise gate, scoring a perfectly good frame 0.0. This is
    /// the same rule `needs_dd` already follows via `final_export_width`.
    #[test]
    fn probe_precision_follows_the_export_width_not_the_probe_width() {
        // A portrait frame at FILE_SIZE_PROBE_RES is this narrow:
        let probe_w = (FILE_SIZE_PROBE_RES as f64 * (1080.0 / 1920.0)).round() as u32;
        let export_w = 1080u32;
        // Zoom inside the window where the two widths disagree (the real f12
        // rejection was at 2294).
        let v = View { cx: 0.713, cx_lo: 0.0, cy: -0.626, cy_lo: 0.0, zoom: 2294.0, aspect: 0.5625 };
        assert!(!needs_f64(&v, probe_w), "probe width alone would stay in f32 here — the bug");
        assert!(needs_f64(&v, export_w), "the real export is already in f64 here");
        // What the fixed call computes: the stricter of the two.
        assert!(needs_f64(&v, probe_w.max(export_w)),
            "probe must follow the export into f64, or it judges an image the export never renders");
    }

    /// Carl's real failure (2026-08-17): a zoomed-OUT seed view is
    /// legitimately sparse, so leg 0 fails while the other 19 legs are rich.
    /// The old prefix-only policy dropped the WHOLE chain — and since every
    /// chain shares the seed, that returned zero winners from a search that
    /// had actually reached zoom 9.8e11.
    #[test]
    fn sparse_opening_leg_is_trimmed_not_fatal() {
        let n_legs = 20;
        let mut good = vec![true; n_legs * 8];
        for g in good.iter_mut().take(8) { *g = false; }   // leg 0 only
        assert_eq!(usable_leg_span(&good, n_legs), Some((1, 19)),
            "a bad opening must trim the head, never discard the chain");
    }

    /// The f64 limit case Carl asked for: keep the last WORKING zoom rather
    /// than failing the chain for having continued past the precision wall.
    #[test]
    fn degenerate_tail_keeps_the_last_working_zoom() {
        let n_legs = 10;
        let mut good = vec![true; n_legs * 8];
        for g in good.iter_mut().skip(6 * 8) { *g = false; }  // legs 6..9 dead
        assert_eq!(usable_leg_span(&good, n_legs), Some((0, 5)));
    }

    #[test]
    fn usable_leg_span_edges() {
        // All clean -> the whole span.
        assert_eq!(usable_leg_span(&vec![true; 40], 5), Some((0, 4)));
        // Nothing usable -> None (caller drops the chain).
        assert_eq!(usable_leg_span(&vec![false; 40], 5), None);
        // A partly-bad leg is never half-counted: one bad frame kills its leg
        // and splits the run, so the longer SIDE wins.
        let mut good = vec![true; 80];
        good[3 * 8 + 2] = false;               // one frame inside leg 3 of 10
        assert_eq!(usable_leg_span(&good, 10), Some((4, 9)),
            "tail run (6 legs) beats head run (3 legs)");
        // A GENUINE tie (two 4-leg runs) resolves to the earlier, shallower
        // one — `cur_len > best_len` is strict, so the first wins. Preferring
        // the shallower half on a tie is the safe bias: it is the side whose
        // coordinates carry the most precision headroom.
        let mut g2 = vec![true; 80];
        for i in 4 * 8..6 * 8 { g2[i] = false; }   // legs 4 AND 5 bad
        assert_eq!(usable_leg_span(&g2, 10), Some((0, 3)));
    }

    /// `dd_wall_zoom` is a closed form duplicating `needs_dd_with_margin`'s
    /// inequality, so it can silently drift from the gate it claims to
    /// describe. Pin it: just under the reported wall must pass the real
    /// gate, just over must fail it.
    #[test]
    fn dd_wall_zoom_agrees_with_the_gate() {
        for &w in &[640u32, 1080, 1280, 3840] {
            for &margin in &[1.0f64, 2.0, 4.0] {
                let wall = dd_wall_zoom(w, margin);
                let at = |zoom: f64| View::new_square(0.0, 0.0, zoom);
                assert!(!needs_dd_with_margin(&at(wall * 0.99), w, margin),
                    "w={w} margin={margin}: just below the wall must still be f64-renderable");
                assert!(needs_dd_with_margin(&at(wall * 1.01), w, margin),
                    "w={w} margin={margin}: just above the wall must trip the gate");
            }
        }
    }

    /// The whole point of the 1.0 default: it must actually buy depth over
    /// the viewer's conservative 4.0, and in the right direction.
    #[test]
    fn pixelate_margin_reaches_deeper_than_the_viewer_margin() {
        let (w, smooth, pixel) = (1080u32, DD_MARGIN_ULPS, DD_MARGIN_ULPS_PIXELATE);
        let deep = dd_wall_zoom(w, pixel);
        assert!(deep > dd_wall_zoom(w, smooth), "1 ULP must permit more zoom than 4 ULP");
        assert!((deep / dd_wall_zoom(w, smooth) - 4.0).abs() < 1e-9, "expected exactly 4x more zoom");
        // A view past the conservative wall but inside the pixelation wall is
        // exactly the region this change unlocks: rejected before, kept now.
        let between = View::new_square(0.0, 0.0, dd_wall_zoom(w, smooth) * 2.0);
        assert!(needs_dd_with_margin(&between, w, smooth), "would have been rejected at 4 ULP");
        assert!(!needs_dd_with_margin(&between, w, pixel), "must be accepted at 1 ULP");
    }

    #[test]
    fn drill_chain_generic_follows_best_child_first() {
        // A simple binary tree: node n's children are 2n+1 (best, score 1.0)
        // and 2n+2 (worse, score 0.5). Greedy best-first descent from 0
        // through depth 3 should walk 0 -> 1 -> 3 -> 7.
        let expand = |view: &View| -> Vec<(View, f64)> {
            let n = id_of(view);
            vec![(v(2 * n + 1), 1.0), (v(2 * n + 2), 0.5)]
        };
        let results = drill_chain_generic(v(0), 3, expand, |_| false);
        let best = results.iter().find(|(_, _, r)| *r == EndedReason::DepthReached).expect("a depth-reached path must exist");
        let ids: Vec<i64> = best.0.iter().map(|cv| cv.cx as i64).collect();
        assert_eq!(ids, vec![0, 1, 3, 7]);
    }

    #[test]
    fn drill_chain_generic_backtracks_to_sibling_on_dead_end() {
        // Node 1's only child is a dead end at depth 2; node 2 (the
        // sibling) has a real line that reaches full depth. Both a
        // terminated DeadEnd chain and the sibling's longer completed
        // chain must appear in the output. Note: a backtracked sibling is
        // pushed at the SAME `remaining` its parent had (a fresh budget,
        // not one-level-deeper) — the proven `recursive_drill` semantic
        // this mirrors — so node 2's own line needs a full `max_depth`
        // (3) more moves from itself, not 2, to reach DepthReached.
        let expand = |view: &View| -> Vec<(View, f64)> {
            match id_of(view) {
                0 => vec![(v(1), 1.0), (v(2), 0.5)], // node 1 preferred (higher score)
                1 => vec![(v(10), 1.0)],             // node 1's only child
                10 => vec![],                        // dead end
                2 => vec![(v(20), 1.0)],
                20 => vec![(v(21), 1.0)],
                21 => vec![(v(210), 1.0)],
                _ => vec![],
            }
        };
        let results = drill_chain_generic(v(0), 3, expand, |_| false);
        let dead_end = results.iter().find(|(chain, _, r)| {
            *r == EndedReason::DeadEnd && chain.last().map(|cv| cv.cx as i64) == Some(10)
        });
        assert!(dead_end.is_some(), "expected the greedy line through node 10 to terminate as DeadEnd");
        let sibling_success = results.iter().find(|(chain, _, r)| {
            *r == EndedReason::DepthReached && chain.iter().any(|cv| cv.cx as i64 == 2)
        });
        assert!(sibling_success.is_some(), "expected backtracking to the sibling (node 2) to reach full depth");
    }

    #[test]
    fn drill_chain_generic_distinguishes_dd_boundary_from_dead_end() {
        // `expand` always returns empty (every node is a leaf); `is_dd_limited`
        // is true only for node 0 itself. The seed's own termination reason
        // must reflect that distinction — this directly exercises "end just
        // before the DD zone" without needing a real fractal.
        let expand = |_: &View| -> Vec<(View, f64)> { vec![] };
        let results = drill_chain_generic(v(0), 2, expand, |view: &View| id_of(view) == 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, EndedReason::DdBoundary);

        let results_not_limited = drill_chain_generic(v(0), 2, expand, |_| false);
        assert_eq!(results_not_limited[0].2, EndedReason::DeadEnd);
    }

    #[test]
    fn drill_chain_generic_respects_visit_budget() {
        // Every node has 4 children. Depth must be deep enough that the
        // TREE is bigger than the budget, otherwise the search legitimately
        // finishes on depth and never reports BudgetExhausted — that's the
        // correct behavior since siblings consume a depth level (they are
        // alternatives at the same tree depth, not extra free levels; see
        // `drill_chain_generic`'s sibling push). At depth 3 the budget is
        // `3 * BACKTRACK_BUDGET_FACTOR = 9` visits while a full 4-ary tree
        // of depth 3 has 4+16+64 nodes, so the budget genuinely binds.
        use std::cell::RefCell;
        let visits = RefCell::new(0usize);
        let expand = |view: &View| -> Vec<(View, f64)> {
            *visits.borrow_mut() += 1;
            let n = id_of(view);
            (0..4).map(|k| (v(n * 4 + k + 1), 1.0 - k as f64 * 0.1)).collect()
        };
        let results = drill_chain_generic(v(0), 3, expand, |_| false);
        let exhausted = results.iter().any(|(_, _, r)| *r == EndedReason::BudgetExhausted);
        assert!(exhausted, "a branching factor this high at depth 3 must exceed the visit budget");
        assert!(*visits.borrow() <= 3 * BACKTRACK_BUDGET_FACTOR, "visits={}", visits.borrow());
        // Depth is now a HARD bound on path length — the bug this guards
        // against let lateral moves grant free extra levels, producing
        // 10-waypoint chains from `--depth 3`.
        for (chain, _, _) in &results {
            assert!(chain.len() <= 4, "depth 3 must cap chains at 4 waypoints, got {}", chain.len());
        }
    }
}
