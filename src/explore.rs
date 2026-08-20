//! Multi-metric, diversity-aware fractal exploration engine.
//!
//! Headless generalization of the interactive viewer's "Auto-Select" drill
//! (`src/bin/viewer.rs`'s `find_interesting_square`/`sweep_candidates`/
//! `pick_by_scale_normalized_entropy`): same grid×scale sweep + per-scale
//! z-score selection + weak-pass widen retry, but (a) a configurable choice
//! of richness metric/score instead of hardcoded PNG-compression entropy,
//! and (b) many independent starting seeds whose own best-scoring round is
//! then filtered down to a visually DIVERSE final set (via a dihedral-
//! invariant structure fingerprint + greedy farthest-point selection),
//! instead of one single deepest point.
//!
//! Shared by `src/bin/explorer.rs` (headless CLI: batch runs, the 24h
//! "hidden gems" tiled search, archive curation) and `src/bin/viewer.rs`
//! (an "Explore" button that runs this against whatever genome is
//! currently loaded, not just a hardcoded formula) — moved here from
//! explorer.rs for the same reason `video_export.rs` exists: a rendering
//! subsystem two binaries both need shouldn't be duplicated between them.

use crate::config::Config;
use crate::dd::Dd;
use crate::fractal::{
    correlation, dihedral_variants, edge_density, field_intricacy, local_edge_contrast, local_intricacy_contrast,
    structure_vec, LOCAL_CONTRAST_MIN, LOCAL_EDGE_DEGENERATE_OVERRIDE, LOCAL_TILE_GRID,
};
use crate::genome::Genome;
use crate::io::save_png;
use crate::colormap::apply_colormap;
use crate::render_gpu;
use crate::video_export::{needs_f64, render_escape_times, View};
use std::path::Path;

// ── Metrics ──────────────────────────────────────────────────────────────

pub const SWEEP_RES: u32 = 64;
/// Grid side length for every scale tier's candidate layout (both the wide
/// seed-picking sweep and `vae_explore::coarse_scan`'s per-level scan share
/// this). Raised from 5 (Carl's report, 2026-08-10: "zones of interest are
/// randomly ignored") — a 5×5 grid genuinely can miss thin, off-grid
/// fractal boundary detail between sample points. Safe to raise because the
/// two stages that actually use it are cheap regardless of grid density:
/// the wide sweep is one batched GPU dispatch at `SWEEP_RES` (tiny, tens of
/// KB per candidate) and `coarse_scan` only crops/scores an ALREADY-
/// rendered canvas (no new render). The expensive step — a fresh precise
/// render per surviving candidate — is bounded by `top_k`/`n_seeds`
/// independent of grid density, so a denser grid improves which
/// candidates make it into that expensive step without making the step
/// itself any more expensive.
const GRID: usize = 9;
pub const SCALES: &[f64] = &[0.5, 0.25, 0.125, 0.0625];
/// Seed-picking scale set for a WIDE search (see `EXPLORE_WIDE_RADIUS`) —
/// unlike `SCALES` (always a fraction of the current view, so it can only
/// ever crop TIGHTER than what's already on screen), this spans several
/// orders of magnitude OUT (up to 4096x shallower than the current view)
/// as well as in. A small zoom-out alone isn't enough: fractal detail lives
/// near a boundary/Julia set, not throughout the whole plane, and a real
/// case confirmed the gap can be huge — a genome explored at zoom≈2244 was
/// smooth rainbow banding with NO fractal structure anywhere within 2x of
/// that depth in any direction (confirmed by probing the same genome from
/// zoom 1 up to 2244: its actual structure lived at zoom≈1-50 and had
/// already dissolved into smooth bands by zoom≈150), so a sweep that can
/// only reach 2x out from a 45-130x-too-deep starting point could never
/// have found it. The largest entries collapse to ~1 effective grid
/// position each once the tile exceeds the search window (harmless —
/// still gets a well-placed sample near the current center at that shallower
/// depth) rather than actually needing a proportionally huge search radius.
/// Only used for seed-PICKING (`pick_seeds`); `drill`'s own per-round
/// refinement always uses plain `SCALES`, since narrowing in on an
/// already-good seed is exactly what should stay tight.
pub const WIDE_SCALES: &[f64] = &[4096.0, 1024.0, 256.0, 64.0, 16.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625];
/// `pick_seeds` radius multiplier for a wide search — the viewer's Explore
/// button used to search seeds only within `radius_mult=1.0` of the
/// current view (i.e., strictly inside what's already on screen), which
/// meant a current view that happened to be entirely smooth/non-fractal
/// (confirmed in practice: 8/8 results were plain rainbow banding, no
/// escape possible because there was nothing else IN REACH to find) could
/// never produce a good result no matter how many seeds or rounds ran.
/// 10x searches a window ten times the current viewport's extent around
/// its center — cheap on the GPU-batched sweep (one dispatch regardless of
/// candidate count), so "search much further away" costs little even
/// though it looks like a big multiplier.
pub const EXPLORE_WIDE_RADIUS: f64 = 10.0;
const WIDEN_RADIUS: f64 = 1.6;
const MIN_VALID_FRACTION: f64 = 0.2; // below this, retry the sweep wider
const MIN_SCORE_FRACTION: f32 = 0.5; // candidate must reach this fraction of the sweep's own best raw score
const FP_PS: usize = 12; // fingerprint pooling size for diversity comparison
/// GPU sweep rendering (`render_gpu::render_batch_dag`) uses plain f32 view
/// bounds — no DD tier, unlike the CPU path elsewhere in this project.
/// Capped well inside `needs_f64`'s own safe range (that threshold works
/// out to roughly 8,000-10,000x for Mandelbrot-scale coordinates; this
/// stays comfortably clear of the ragged edge where f32 escape-time
/// renders start aliasing) so every candidate the GPU sweep considers is
/// genuinely resolvable at this precision — not a shallower proxy for a
/// deeper answer. A round whose every scale would exceed this is skipped
/// entirely (empty candidate set), which stops `drill` naturally.
pub const GPU_MAX_ZOOM: f64 = 3000.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub entropy: f32,
    pub edge_density: f32,
    pub intricacy: f32,
    pub degenerate: bool,
}

/// Same "mostly one color" containment signal `viewer.rs::view_is_degenerate`
/// uses, checked on colormapped RGB (not raw escape times) for the same
/// reason: iteration-count saturation near the interior can map many
/// distinct raw values to the identical top color.
fn rgb_is_degenerate(rgb: &[u8]) -> bool {
    if rgb.len() < 3 { return true; }
    let n = rgb.len() / 3;
    let mut counts: std::collections::HashMap<(u8, u8, u8), u32> = std::collections::HashMap::new();
    for i in 0..n {
        *counts.entry((rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    (max_count as f32 / n as f32) > 0.95
}

/// Same PNG-compression-ratio entropy `viewer.rs::rgb_compression_entropy`/
/// `fitness::png_compression_entropy` use.
fn rgb_compression_entropy(rgb: &[u8], w: u32, h: u32) -> f32 {
    let mut buf = std::io::Cursor::new(Vec::with_capacity(8192));
    image::write_buffer_with_format(&mut buf, rgb, w, h, image::ColorType::Rgb8, image::ImageFormat::Png).unwrap_or(());
    let png_bytes = buf.into_inner().len() as f32;
    let raw_bytes = (w * h * 3) as f32;
    png_bytes / raw_bytes
}

// ── Scoring methods ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreMethod { Entropy, EdgeDensity, GatedEntropy, GatedEdgeDensity }

impl ScoreMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "entropy" => Some(Self::Entropy),
            "edge" => Some(Self::EdgeDensity),
            "gated-entropy" => Some(Self::GatedEntropy),
            "gated-edge" => Some(Self::GatedEdgeDensity),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Entropy => "entropy",
            Self::EdgeDensity => "edge",
            Self::GatedEntropy => "gated-entropy",
            Self::GatedEdgeDensity => "gated-edge",
        }
    }
    pub const ALL: [ScoreMethod; 4] = [Self::Entropy, Self::EdgeDensity, Self::GatedEntropy, Self::GatedEdgeDensity];
}

/// Same ramp-up/ramp-down shape already validated (twice, empirically) this
/// project for `fractal_recursion_score` and `fractal::wormhole_search`:
/// reject smooth/monotone fields (too low — a scale-invariant ramp
/// trivially maximizes raw entropy/edge-density without being an actual
/// fractal boundary) AND reject pixel-noise fields (too high — noise also
/// trivially maximizes raw entropy/edge-density without showing organized
/// structure), full credit in between. LO/HI/CEIL_LO unchanged from
/// (and originally copied from) `fractal::WORMHOLE_INTRIC_*` — a SEPARATE
/// copy, not shared code, so tuning this doesn't touch wormhole search.
///
/// `CEIL_HI` raised 0.40 → 0.60, 2026-08-11, with real calibration data
/// (Carl found a genuinely attractive circular/mandala structure on
/// 9aa9885ad83f97cc that `debug-sweep` showed scoring entropy=1.01 (near
/// max), edge=0.74, intricacy=0.492, yet `score=0.0000` under BOTH gated
/// methods — the ceiling was hard-zeroing it). Checked what actually
/// separates structure from noise on THIS metric before picking a number,
/// not guessing: a 150-sample check of this genome's own already-
/// successfully-explored corpus topped out at intricacy=0.305 (median
/// 0.198) — this structure's 0.492 is a genuine outlier even among
/// "good" content, not typical. 10 trials of pure uniform random noise at
/// the same 128×128 resolution measured 0.655-0.661, consistently — the
/// mandala (0.492) sits clearly below that. 0.60 leaves the structure a
/// real, nonzero (though still discounted — ramp_down≈0.28 there, not
/// full credit) chance under gated scoring, while keeping a real gap
/// (~0.06) below where pure noise actually measures, so the gate's
/// original noise-rejection purpose isn't gutted. Calibrated against ONE
/// real example plus a synthetic noise/existing-corpus baseline — not an
/// exhaustive sweep across many formulas, worth revisiting if a different
/// genome's legitimately-interesting content turns out to sit above 0.60
/// too.
fn intricacy_gate(intricacy: f32) -> f32 {
    const LO: f32 = 0.010;
    const HI: f32 = 0.030;
    const CEIL_LO: f32 = 0.22;
    const CEIL_HI: f32 = 0.60;
    let ramp_up = ((intricacy - LO) / (HI - LO)).clamp(0.0, 1.0);
    let ramp_down = 1.0 - ((intricacy - CEIL_LO) / (CEIL_HI - CEIL_LO)).clamp(0.0, 1.0);
    ramp_up * ramp_down
}

pub(crate) fn score(m: &Metrics, method: ScoreMethod) -> f32 {
    if m.degenerate { return 0.0; }
    match method {
        ScoreMethod::Entropy => m.entropy,
        ScoreMethod::EdgeDensity => m.edge_density,
        ScoreMethod::GatedEntropy => m.entropy * intricacy_gate(m.intricacy),
        ScoreMethod::GatedEdgeDensity => m.edge_density * intricacy_gate(m.intricacy),
    }
}

// ── Sweep (generalized viewer.rs sweep_candidates + pick_by_scale_normalized_entropy) ──

#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub(crate) scale_idx: usize,
    pub(crate) dx: f64,
    pub(crate) dy: f64,
    pub(crate) zoom: f64,
    pub(crate) metrics: Metrics,
    pub(crate) score: f32,
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() { return (0.0, 0.0); }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    (mean, var.sqrt())
}

/// Every candidate position for one sweep — scale/grid layout unchanged
/// from the CPU design, just split out so all positions can be batched
/// through one GPU dispatch instead of rendered one at a time.
pub(crate) struct Pos { pub(crate) scale_idx: usize, pub(crate) dx: f64, pub(crate) dy: f64, pub(crate) zoom: f64 }

pub(crate) fn sweep_positions(cur_w: f64, cur_h: f64, radius_mult: f64, max_zoom: f64, scales: &[f64]) -> Vec<Pos> {
    let base_half_x = (cur_w / 2.0) * radius_mult;
    let base_half_y = (cur_h / 2.0) * radius_mult;
    let mut out = Vec::new();
    for (scale_idx, &scale) in scales.iter().enumerate() {
        let side = cur_h * scale;
        if side <= 0.0 { continue; }
        let half = side / 2.0;
        let new_zoom = 4.0 / side;
        if new_zoom > max_zoom { continue; }
        // Per-scale search window, not the shared `base_half` directly: a
        // scale bigger than `radius_mult` itself (WIDE_SCALES' largest
        // entries, when zooming out far past the base search radius) would
        // otherwise make `(2*base_half - side).max(0.0)` collapse to 0 for
        // every grid point — all GRID*GRID candidates land on the exact
        // same position, so `rank_by_zscore` sees zero within-bucket
        // variance and scores every one of them a flat 0.0, regardless of
        // whether that single position is actually good. Confirmed this
        // way in practice: a real "wide" search's largest scales (the ones
        // that could reach back to a genome's only structured zoom range)
        // never won a single seed slot because of exactly this collapse.
        // Scaling the search window WITH the tile size keeps real grid
        // spread — and so a meaningful z-score — at every scale.
        let search_half_x = base_half_x.max(side);
        let search_half_y = base_half_y.max(side);
        for iy in 0..GRID {
            for ix in 0..GRID {
                let t_x = if GRID > 1 { ix as f64 / (GRID - 1) as f64 } else { 0.5 };
                let t_y = if GRID > 1 { iy as f64 / (GRID - 1) as f64 } else { 0.5 };
                let dx = -search_half_x + half + t_x * (2.0 * search_half_x - side).max(0.0);
                let dy = -search_half_y + half + t_y * (2.0 * search_half_y - side).max(0.0);
                out.push(Pos { scale_idx, dx, dy, zoom: new_zoom });
            }
        }
    }
    out
}

fn field_to_candidate(p: &Pos, field: &[f32], config: &Config, method: ScoreMethod) -> Candidate {
    let rgb = apply_colormap(field, config.rendering.max_iter, &config.rendering.colormap);
    let (local_edge, edge_contrast) = local_edge_contrast(field, SWEEP_RES as usize, SWEEP_RES as usize, LOCAL_TILE_GRID);
    let (local_intric, intric_contrast) = local_intricacy_contrast(field, SWEEP_RES as usize, SWEEP_RES as usize, LOCAL_TILE_GRID);
    // Same relaxation as `vae_explore::score_position`: a whole-patch
    // dominant-color verdict (`rgb_is_degenerate`) can still miss a small,
    // genuinely interesting localized feature (a crisp circular boundary,
    // say) that gets diluted into invisibility by a large uniform
    // surrounding area — see `fractal::tile_stats`'s doc comment. Checked
    // against the raw field's own local structure (not the colormapped
    // rgb) since that's what these operate on either way.
    let degenerate = rgb_is_degenerate(&rgb) && local_edge < LOCAL_EDGE_DEGENERATE_OVERRIDE;
    // `m` carries the TRUE whole-patch values, ALWAYS — see
    // `vae_explore::score_position`'s doc comment for why this matters: a
    // real bug (caught by a live re-test, not shipped) wrote the boosted
    // values straight into `Metrics` here, which is also what gates
    // (`max_intricacy`/`min_edge_density`, wherever a caller reads
    // `Candidate.metrics` directly) check — inflating intricacy past the
    // ceiling for candidates that would otherwise have passed.
    let m = Metrics {
        entropy: if degenerate { 0.0 } else { rgb_compression_entropy(&rgb, SWEEP_RES, SWEEP_RES) },
        edge_density: edge_density(field, SWEEP_RES as usize, SWEEP_RES as usize),
        intricacy: field_intricacy(field, SWEEP_RES as usize, SWEEP_RES as usize),
        degenerate,
    };
    let mut s = score(&m, method);
    // Rescuing the RANKING is gated on CONTRAST, not a bare local max —
    // see `vae_explore::score_position`'s doc comment for the full story:
    // an earlier version blended raw `local_edge_density` straight into
    // the score and caused a REAL, observed regression (a live run stuck
    // reaching only 32x deeper than its base zoom after 500+ zones) — a
    // wide/shallow crop has a much higher a-priori chance of containing
    // SOME locally-okay tile purely because it covers more ground, so
    // rewarding "best tile anywhere" made the search systematically
    // prefer staying shallow. `contrast` (max minus the tiles' own mean)
    // only fires for a genuine standout, not generic wide-view variance.
    // Rescored via a SEPARATE boosted Metrics that never escapes this
    // function, so it can't leak into gating.
    if edge_contrast >= LOCAL_CONTRAST_MIN || intric_contrast >= LOCAL_CONTRAST_MIN {
        let boosted = Metrics { edge_density: m.edge_density.max(local_edge), intricacy: m.intricacy.max(local_intric), ..m };
        s = s.max(score(&boosted, method));
    }
    Candidate { scale_idx: p.scale_idx, dx: p.dx, dy: p.dy, zoom: p.zoom, metrics: m, score: s }
}

/// One GPU dispatch for every candidate in the sweep — the whole point of
/// concentrating on the GPU-solvable (f32, `GPU_MAX_ZOOM`-capped) range:
/// batching means a full 100-candidate sweep costs one dispatch instead of
/// 100 separate small renders, which is what makes covering far more seeds
/// (genuine diversity across the whole visible set) affordable.
fn sweep(
    genome: &Genome, config: &Config, view: &View, cur_w: f64, cur_h: f64,
    radius_mult: f64, method: ScoreMethod, scales: &[f64],
) -> Vec<Candidate> {
    let positions = sweep_positions(cur_w, cur_h, radius_mult, GPU_MAX_ZOOM, scales);
    if positions.is_empty() { return Vec::new(); }

    // Absolute f32 bounds per candidate — the offset-from-DD-center pattern
    // used everywhere else in this project isn't needed here specifically
    // BECAUSE `GPU_MAX_ZOOM` keeps every candidate well inside f32's own
    // precision floor; this is the one place in the codebase where a bare
    // f32 position is the correct choice, not an oversight.
    let bounds: Vec<(f32, f32, f32, f32)> = positions.iter().map(|p| {
        let cx = (view.cx_dd() + Dd::from_f64(p.dx)).hi as f32;
        let cy = (view.cy_dd() + Dd::from_f64(p.dy)).hi as f32;
        let half = (2.0 / p.zoom) as f32;
        (cx - half, cx + half, cy - half, cy + half)
    }).collect();

    let items: Vec<render_gpu::DagItem> = positions.iter().map(|_| render_gpu::dag_item(genome)).collect();
    let fields = render_gpu::render_batch_dag(&items, &bounds, SWEEP_RES, SWEEP_RES, config.rendering.max_iter);

    positions.iter().zip(fields.iter()).map(|(p, field)| field_to_candidate(p, field, config, method)).collect()
}

/// CPU/DD-tiered sweep — same scale×grid layout as `sweep`, but per-candidate
/// via `render_escape_times` (the offset-from-DD-center pattern used
/// everywhere else in this project) instead of one GPU batch, and with no
/// `GPU_MAX_ZOOM` ceiling: `needs_f64`/DD tiering inside `render_escape_times`
/// keeps every candidate correctly resolved no matter how deep. Slower per
/// candidate than the batched GPU path — used only for the deep-drill phase
/// of a "hidden gems" search, on the minority of seeds that already cleared
/// a shallow quality bar, not for wide coverage sweeps.
fn sweep_cpu(
    genome: &Genome, config: &Config, view: &View, cur_w: f64, cur_h: f64,
    radius_mult: f64, method: ScoreMethod, scales: &[f64],
) -> Vec<Candidate> {
    let positions = sweep_positions(cur_w, cur_h, radius_mult, f64::INFINITY, scales);
    positions.iter().map(|p| {
        let cx = view.cx_dd() + Dd::from_f64(p.dx);
        let cy = view.cy_dd() + Dd::from_f64(p.dy);
        let cand_view = View { cx: cx.hi, cx_lo: cx.lo, cy: cy.hi, cy_lo: cy.lo, zoom: p.zoom, aspect: 1.0 };
        let use_f64 = needs_f64(&cand_view, SWEEP_RES);
        let field = render_escape_times(genome, config, &cand_view, SWEEP_RES, SWEEP_RES, config.rendering.max_iter, use_f64, true);
        field_to_candidate(p, &field, config, method)
    }).collect()
}

/// Every valid candidate ranked by per-scale Z-SCORE — "how good is this,
/// for a square of its size" — instead of raw score, which structurally
/// favors wider scales regardless of which candidate is actually more
/// interesting (same bias `viewer.rs`'s own `pick_by_scale_normalized_entropy`
/// documents and fixes; confirmed to matter here too, the hard way: an
/// early version of seed-picking ranked by raw score directly and 6 of 8
/// seeds landed at the identical wide scale index, several in genuinely
/// boring exterior gradient bands, because raw edge/entropy at the widest
/// scale beats every narrower, more genuinely interesting candidate almost
/// by construction). Sorted best first; candidates below
/// `min_score_fraction` of the sweep's own best raw score are excluded
/// first, so z-score normalization can't flip that bias into elevating the
/// least-bad member of a scale group that's uniformly worse than what's
/// available elsewhere in the same sweep. `MIN_SCORE_FRACTION` (0.5) is
/// appropriate for a single drill round's winner; wide seed-picking wants
/// coverage across the whole visible fractal, so `pick_seeds` passes `0.0`.
pub(crate) fn rank_by_zscore(candidates: &[Candidate], min_score_fraction: f32) -> Vec<Candidate> {
    let valid: Vec<&Candidate> = candidates.iter().filter(|c| !c.metrics.degenerate).collect();
    if valid.is_empty() { return Vec::new(); }
    let global_max = valid.iter().map(|c| c.score).fold(f32::MIN, f32::max);
    let min_score = global_max * min_score_fraction;

    // Derived from the candidates themselves, not the global `SCALES`
    // constant — `pick_seeds`'s wide search can generate candidates from a
    // longer scale list (`WIDE_SCALES`), and indexing `stats[c.scale_idx]`
    // against the wrong (shorter) length would panic.
    let n_scales = valid.iter().map(|c| c.scale_idx).max().map_or(0, |m| m + 1);
    let stats: Vec<(f32, f32)> = (0..n_scales)
        .map(|i| mean_std(&valid.iter().filter(|c| c.scale_idx == i).map(|c| c.score).collect::<Vec<_>>()))
        .collect();
    let zscore = |c: &Candidate| -> f32 {
        let (mean, std) = stats[c.scale_idx];
        if std < 1e-6 { 0.0 } else { (c.score - mean) / std }
    };

    let mut ranked: Vec<Candidate> = valid.into_iter().filter(|c| c.score >= min_score).copied().collect();
    ranked.sort_by(|a, b| zscore(b).partial_cmp(&zscore(a)).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

fn pick_best(candidates: &[Candidate]) -> Option<Candidate> {
    rank_by_zscore(candidates, MIN_SCORE_FRACTION).into_iter().next()
}

/// One sweep round, widening the search net once if most of the grid came
/// back degenerate (same rationale as viewer.rs's `find_interesting_square`:
/// a sweep confined strictly inside the current frame can't recover detail
/// that's just outside it). `use_gpu` picks the batched-f32-shallow backend
/// (fast, `GPU_MAX_ZOOM`-capped) or the per-candidate DD-tiered backend
/// (slow, unlimited depth) — see `sweep`/`sweep_cpu`.
fn sweep_round(genome: &Genome, config: &Config, view: &View, method: ScoreMethod, use_gpu: bool) -> (Vec<Candidate>, Option<Candidate>) {
    let cur_w = 4.0 / view.zoom * view.aspect;
    let cur_h = 4.0 / view.zoom;
    let sweep_fn = if use_gpu { sweep } else { sweep_cpu };
    let mut candidates = sweep_fn(genome, config, view, cur_w, cur_h, 1.0, method, SCALES);
    if candidates.is_empty() { return (candidates, None); } // every scale past GPU_MAX_ZOOM (GPU backend only)
    let total = candidates.len(); // NOT a fixed constant — GPU_MAX_ZOOM can skip whole scales
    let valid = candidates.iter().filter(|c| !c.metrics.degenerate).count();
    if (valid as f64) < (total as f64) * MIN_VALID_FRACTION {
        candidates.extend(sweep_fn(genome, config, view, cur_w, cur_h, WIDEN_RADIUS, method, SCALES));
    }
    let winner = pick_best(&candidates);
    (candidates, winner)
}

pub fn apply_offset(view: &View, dx: f64, dy: f64, zoom: f64) -> View {
    let cx = view.cx_dd() + Dd::from_f64(dx);
    let cy = view.cy_dd() + Dd::from_f64(dy);
    View { cx: cx.hi, cx_lo: cx.lo, cy: cy.hi, cy_lo: cy.lo, zoom, aspect: view.aspect }
}

// ── Logging ──────────────────────────────────────────────────────────────

pub struct Logger {
    file: std::fs::File,
    pub verbose: bool, // whether candidate() actually writes — see gems mode's doc comment
}

impl Logger {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: std::fs::File::create(path)?, verbose: true })
    }
    /// Append instead of truncate — needed for a multi-hour run that must
    /// survive being restarted partway through without losing its own
    /// history (see `cmd_gems`).
    pub fn append(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: std::fs::OpenOptions::new().create(true).append(true).open(path)?, verbose: true })
    }
    pub fn log(&mut self, value: &serde_json::Value) {
        use std::io::Write;
        if let Ok(mut line) = serde_json::to_string(value) {
            line.push('\n');
            let _ = self.file.write_all(line.as_bytes());
        }
    }
    fn candidate(&mut self, seed: usize, round: usize, view: &View, c: &Candidate, method: ScoreMethod) {
        if !self.verbose { return; }
        let cand_view = apply_offset(view, c.dx, c.dy, c.zoom);
        self.log(&serde_json::json!({
            "event": "candidate", "seed": seed, "round": round, "scale_idx": c.scale_idx,
            "cx": cand_view.cx, "cy": cand_view.cy, "zoom": c.zoom,
            "entropy": c.metrics.entropy, "edge_density": c.metrics.edge_density,
            "intricacy": c.metrics.intricacy, "degenerate": c.metrics.degenerate,
            "score": c.score, "method": method.name(),
        }));
    }
}

// ── Drill ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RoundResult {
    pub view: View,
    pub metrics: Metrics,
    pub score: f32,
    pub round: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn drill(
    genome: &Genome, config: &Config, start: View, max_rounds: usize, method: ScoreMethod,
    seed: usize, log: &mut Logger, use_gpu: bool, round_offset: usize,
) -> Vec<RoundResult> {
    let mut history = Vec::new();
    let mut view = start;
    for round in 0..max_rounds {
        let (candidates, winner) = sweep_round(genome, config, &view, method, use_gpu);
        for c in &candidates { log.candidate(seed, round_offset + round, &view, c, method); }
        let Some(w) = winner else { break };
        view = apply_offset(&view, w.dx, w.dy, w.zoom);
        log.log(&serde_json::json!({
            "event": "round_winner", "seed": seed, "round": round_offset + round,
            "cx": view.cx, "cy": view.cy, "zoom": view.zoom, "score": w.score, "gpu": use_gpu,
        }));
        history.push(RoundResult { view: view.clone(), metrics: w.metrics, score: w.score, round: round_offset + round });
    }
    history
}

// ── Diversity ────────────────────────────────────────────────────────────

/// All 8 dihedral (rotation/mirror) orientations of a candidate's
/// contrast-normalized structure fingerprint — NOT just the plain one.
/// Necessary specifically because the classic Mandelbrot has exact
/// conjugate symmetry across the real axis (f(cx, -cy) mirrors f(cx, cy)):
/// two seeds straddling the axis routinely converge on genuine mirror-image
/// pairs, which a plain position-for-position correlation scores as
/// UNRELATED (every pixel moved), not as the near-duplicate a human would
/// immediately recognize them as. Confirmed in practice — a real run's
/// "diverse" set included several such pairs before this fix. Same
/// technique `fractal::wormhole_search` already uses to match copies that
/// appear rotated or flipped relative to the reference.
pub fn fingerprint(genome: &Genome, config: &Config, view: &View) -> Vec<Vec<f32>> {
    let use_f64 = needs_f64(view, SWEEP_RES);
    let field = render_escape_times(genome, config, view, SWEEP_RES, SWEEP_RES, config.rendering.max_iter, use_f64, true);
    let base = structure_vec(&field, SWEEP_RES as usize, SWEEP_RES as usize, FP_PS);
    dihedral_variants(&base, FP_PS)
}

/// Best correlation between `a` (its plain/identity form) and ANY of `b`'s
/// 8 orientations — comparing one side's full orientation set against the
/// other's plain form already covers "related in any orientation" (the
/// dihedral group is closed), so only one side needs expanding.
pub fn best_orientation_correlation(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    b.iter().map(|bv| correlation(&a[0], bv)).fold(f32::MIN, f32::max)
}

/// Candidate shot must reach at least this fraction of the best shot's own
/// score to be eligible at all — a diverse SELECTION of otherwise-boring
/// views isn't the goal; diversity only matters among views that already
/// earned a place. Confirmed necessary in practice: an early run without
/// this floor filled its final 8-shot set with 6 genuinely rich finds and 2
/// literal near-uniform exterior crops, just because there were 8 seeds to
/// fill 8 slots.
pub const MIN_SHOT_SCORE_FRACTION: f32 = 0.25;
/// Stop adding candidates once the best remaining option's own minimum
/// distance to everything already selected drops below this — two
/// candidates this close are the same wormhole neighborhood in different
/// clothes, not two different discoveries. Raised from an initial 0.15
/// after a real 40-seed run: with mirror pairs correctly caught (see
/// `fingerprint`'s doc comment), 0.15 still let through shots that, while
/// not literal duplicates, were mostly minor variations on one theme —
/// technically distinct, not what "genuine diversity" means to a human
/// looking at the results. 0.3 curates down to compositions that actually
/// read as different at a glance.
pub const MIN_DIVERSITY_DISTANCE: f32 = 0.3;
/// Same idea as `MIN_DIVERSITY_DISTANCE`, but for `select_diverse_latent`'s
/// L2 distance between L2-normalized 128-d novelty-model embeddings (range
/// [0,2] — NOT the same scale as the pixel-fingerprint distance above,
/// which is a correlation-derived [0,~1]). Matches `explorer.rs::cmd_curate`'s
/// own default, calibrated the same way (checked real pairwise distances on
/// an actual run: genuinely distinct gems measured ~0.9-1.3 apart against
/// this project's production novelty model).
pub const MIN_DIVERSITY_DISTANCE_LATENT: f32 = 0.9;

/// Greedy farthest-point selection: start with the highest-scoring shot,
/// then repeatedly add whichever remaining candidate maximizes its minimum
/// distance (1 - correlation of fingerprints) to everything already picked
/// — the same principle the GA's own novelty archive uses for behavioral
/// diversity, applied here to visual structure instead.
pub fn select_diverse(shots: &[(usize, RoundResult, Vec<Vec<f32>>)], k: usize) -> Vec<usize> {
    if shots.is_empty() { return Vec::new(); }
    let best_score = shots.iter().map(|s| s.1.score).fold(f32::MIN, f32::max);
    let min_score = best_score * MIN_SHOT_SCORE_FRACTION;

    let mut remaining: Vec<usize> = (0..shots.len()).filter(|&i| shots[i].1.score >= min_score).collect();
    if remaining.is_empty() { return Vec::new(); }
    remaining.sort_by(|&a, &b| shots[b].1.score.partial_cmp(&shots[a].1.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected = vec![remaining.remove(0)];

    while selected.len() < k && !remaining.is_empty() {
        let next = remaining.iter().enumerate()
            .map(|(pos, &i)| {
                let dist = selected.iter().map(|&s| 1.0 - best_orientation_correlation(&shots[i].2, &shots[s].2)).fold(f32::MAX, f32::min);
                (pos, dist)
            })
            .max_by(|&(_, da), &(_, db)| da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal));
        match next {
            Some((pos, dist)) if dist >= MIN_DIVERSITY_DISTANCE => selected.push(remaining.remove(pos)),
            _ => break,
        }
    }
    selected
}

/// Same greedy farthest-point shape as `select_diverse`, but over arbitrary
/// (score, latent-embedding) pairs instead of dihedral pixel fingerprints —
/// shared by `cmd_curate` (explorer.rs) and the viewer's Explore button,
/// both of which embed candidates via the trained novelty model
/// (`novelty::NoveltyScorer::embed_blocking`). Confirmed necessary for
/// Explore specifically, not just a nice-to-have: a real run's pixel-
/// fingerprint diversity selection kept 38 "diverse" shots of which only 2
/// were genuinely different compositions — the other 36 were smooth color
/// banding at different angles/widths, which DOES move every pixel to a
/// different position (so the pooling fingerprint sees them as unrelated)
/// while still reading as "the same boring thing" to a human, exactly the
/// perceptual-vs-pixel gap `cmd_curate` already fixed this same way for
/// curation. `candidates` must be pre-sorted best-first by whatever the
/// caller's own score means — index 0 is always the first pick, matching
/// `select_diverse`'s "start from the best, then diversify" shape.
pub fn select_diverse_latent(embeddings: &[Vec<f32>], min_dist: f32, k: usize) -> Vec<usize> {
    if embeddings.is_empty() { return Vec::new(); }
    let mut remaining: Vec<usize> = (1..embeddings.len()).collect();
    let mut selected: Vec<usize> = vec![0];
    while selected.len() < k && !remaining.is_empty() {
        let next = remaining.iter().enumerate()
            .map(|(pos, &i)| {
                let dist = selected.iter().map(|&s| l2_dist(&embeddings[i], &embeddings[s])).fold(f32::MAX, f32::min);
                (pos, dist)
            })
            .max_by(|&(_, da), &(_, db)| da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal));
        match next {
            Some((pos, dist)) if dist >= min_dist => selected.push(remaining.remove(pos)),
            _ => break,
        }
    }
    selected
}

pub fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

// ── Seeds ────────────────────────────────────────────────────────────────

fn same_region(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
    let ref_zoom = a.2.min(b.2);
    let tol = (2.0 / ref_zoom) * 0.3;
    (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol && (a.2 / b.2).max(b.2 / a.2) < 2.0
}

/// Diagnostic-only: exposes `sweep`'s FULL candidate list (not just the
/// final picks `pick_seeds` returns) for direct numeric/visual inspection
/// — added 2026-08-11 to investigate why a specific, visually rich
/// circular structure never got picked as a seed despite being within
/// `pick_seeds`' own wide search radius. Returns (absolute cx, cy, zoom,
/// metrics, score), sorted best-first.
pub fn debug_sweep_candidates(
    genome: &Genome, config: &Config, view: &View, radius_mult: f64, method: ScoreMethod, scales: &[f64],
) -> Vec<(f64, f64, f64, Metrics, f32)> {
    let cur_w = 4.0 / view.zoom * view.aspect;
    let cur_h = 4.0 / view.zoom;
    let mut candidates = sweep(genome, config, view, cur_w, cur_h, radius_mult, method, scales);
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.into_iter().map(|c| {
        let v = apply_offset(view, c.dx, c.dy, c.zoom);
        (v.cx, v.cy, c.zoom, c.metrics, c.score)
    }).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn pick_seeds(
    genome: &Genome, config: &Config, view: &View, method: ScoreMethod, n: usize, log: &mut Logger,
    radius_mult: f64, scales: &[f64],
) -> Vec<View> {
    let cur_w = 4.0 / view.zoom * view.aspect;
    let cur_h = 4.0 / view.zoom;
    let candidates = sweep(genome, config, view, cur_w, cur_h, radius_mult, method, scales);
    for c in &candidates { log.candidate(usize::MAX, 0, view, c, method); }

    // Same per-scale z-score ranking `pick_best` uses for a single drill
    // round's winner — see `rank_by_zscore`'s doc comment for why raw-score
    // ranking here specifically produced bad seeds in practice. Unlike
    // `pick_best`, no min-score floor: seed-picking wants coverage across
    // the whole visible fractal, not just the neighborhood of the single
    // best raw score found anywhere in the sweep.
    let ranked = rank_by_zscore(&candidates, 0.0);

    // Guarantee at least `n / scales.len()` seeds from EVERY scale tier
    // actually present, not just whichever tiers happen to win the global
    // z-score ranking. Z-score is normalized WITHIN each tier's own
    // mean/std, so it only measures "how much does this stand out from its
    // own neighbors" — it has no way to compare ABSOLUTE quality across
    // tiers, and confirmed in practice that's not a safe thing to leave
    // implicit: a genome whose only real fractal structure lived many
    // scales away from a bad starting view still lost every seed slot to
    // smooth-but-sharp-banded content elsewhere, because banding can
    // produce comparably extreme WITHIN-tier z-scores (sharp band edges
    // score high on edge_density/intricacy despite having no real
    // structure) — global ranking alone had no way to know it should have
    // looked at the shallower tiers harder. Explicit per-tier floors don't
    // depend on getting that comparison right.
    let mut kept: Vec<(f64, f64, f64)> = Vec::new();
    let n_scales = scales.len().max(1);
    let per_scale_floor = (n / n_scales).max(1);
    for scale_idx in 0..n_scales {
        let mut taken = 0;
        for c in ranked.iter().filter(|c| c.scale_idx == scale_idx) {
            if taken >= per_scale_floor { break; }
            let pos = (c.dx, c.dy, c.zoom);
            if kept.iter().any(|&k| same_region(pos, k)) { continue; }
            kept.push(pos);
            taken += 1;
        }
    }
    // Fill any budget left over (a tier without `per_scale_floor` distinct
    // candidates, or `n` not evenly divisible by `n_scales`) from whatever
    // ranks next-best globally.
    for c in &ranked {
        if kept.len() >= n { break; }
        let pos = (c.dx, c.dy, c.zoom);
        if kept.iter().any(|&k| same_region(pos, k)) { continue; }
        kept.push(pos);
    }
    kept.truncate(n);

    kept.into_iter().map(|(dx, dy, zoom)| apply_offset(view, dx, dy, zoom)).collect()
}

// ── High-level orchestration ─────────────────────────────────────────────

/// Seeds `view` with one method, drills each `max_rounds`, and collects
/// every seed's best round + fingerprint (UNfiltered — no diversity
/// selection yet). `seed_id_offset` keeps ids unique when a caller merges
/// several methods' shots together before selecting (see
/// `explore_diverse_mixed`).
/// Seeds `view`, drills each seed `max_rounds` deep, and returns every
/// seed's FULL round history (unlike `collect_shots`, no "pick the best
/// round by raw score" — that decision is left to the caller). Shared by
/// `collect_shots` (picks best-by-score, for the CLI's simpler pipelines)
/// and `collect_shots_mixed` (picks best-by-AESTHETIC instead, for the
/// viewer — see `collect_shots_mixed`'s doc comment for why raw score isn't
/// trusted with this decision there).
#[allow(clippy::too_many_arguments)]
fn collect_shots_full(
    genome: &Genome, config: &Config, view: &View, method: ScoreMethod,
    n_seeds: usize, max_rounds: usize, log: &mut Logger, seed_id_offset: usize,
    radius_mult: f64, scales: &[f64],
) -> Vec<(usize, Vec<RoundResult>)> {
    let seeds = pick_seeds(genome, config, view, method, n_seeds, log, radius_mult, scales);
    seeds.iter().enumerate()
        .map(|(i, seed_view)| {
            let seed_id = seed_id_offset + i;
            let history = drill(genome, config, seed_view.clone(), max_rounds, method, seed_id, log, true, 0);
            (seed_id, history)
        })
        .filter(|(_, history)| !history.is_empty())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn collect_shots(
    genome: &Genome, config: &Config, view: &View, method: ScoreMethod,
    n_seeds: usize, max_rounds: usize, log: &mut Logger, seed_id_offset: usize,
    radius_mult: f64, scales: &[f64],
) -> Vec<(usize, RoundResult, Vec<Vec<f32>>)> {
    collect_shots_full(genome, config, view, method, n_seeds, max_rounds, log, seed_id_offset, radius_mult, scales)
        .into_iter()
        .filter_map(|(seed_id, history)| {
            let best = history.into_iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))?;
            log.log(&serde_json::json!({
                "event": "seed_best", "seed": seed_id, "round": best.round, "method": method.name(),
                "cx": best.view.cx, "cy": best.view.cy, "zoom": best.view.zoom, "score": best.score,
                "entropy": best.metrics.entropy, "edge_density": best.metrics.edge_density, "intricacy": best.metrics.intricacy,
            }));
            let fp = fingerprint(genome, config, &best.view);
            Some((seed_id, best, fp))
        })
        .collect()
}

/// Full pipeline: pick up to `n_seeds` diverse starting points from a wide
/// sweep of `view`, drill each `max_rounds` deep, then filter the results
/// down to a genuinely diverse, genuinely good final set (may be smaller
/// than `n_seeds` — the quality/diversity floors decide the count, not the
/// request). Returns `(seed_id, best round, dihedral fingerprint)` triples,
/// best score first. This is the single-method entry point `explorer run`
/// uses — see this module's doc comment, and `explore_diverse_mixed` for
/// why the viewer's "Explore" button uses that instead.
pub fn explore_diverse(
    genome: &Genome, config: &Config, view: &View, method: ScoreMethod,
    n_seeds: usize, max_rounds: usize, log: &mut Logger,
) -> Vec<(usize, RoundResult, Vec<Vec<f32>>)> {
    let shots = collect_shots(genome, config, view, method, n_seeds, max_rounds, log, 0, 1.0, SCALES);
    let selected = select_diverse(&shots, shots.len());
    selected.into_iter().map(|i| shots[i].clone()).collect()
}

/// Same pipeline as `explore_diverse`, but seeds are split evenly across
/// several scoring methods instead of one fixed method. Confirmed necessary
/// the hard way (a real 72,000-tile "hidden gems" run under `edge` alone):
/// a single scoring function systematically rewards ONE visual family
/// (edge_density favors dense radiating starbursts almost everywhere on the
/// Mandelbrot boundary), so every seed's local search converges on that
/// same family regardless of position — no amount of diversity filtering
/// afterward can inject variety that was never in the candidate pool.
/// Mixing methods pulls different seeds toward genuinely different kinds
/// of structure (confirmed visually: entropy found a dense repeating
/// texture, gated-entropy found a delicate spiral, from the same tile grid
/// `edge` alone had been finding only starbursts from).
/// `radius_mult`/`scales` control how far `pick_seeds` looks for a starting
/// point relative to `view` — pass `(1.0, explore::SCALES)` for the original
/// "search within what's on screen" behavior, or `(EXPLORE_WIDE_RADIUS,
/// WIDE_SCALES)` for a search that can escape a `view` that's uniformly
/// boring everywhere within it (see their doc comments). `drill`'s own
/// round-to-round refinement is unaffected either way — always narrow,
/// always `SCALES`.
#[allow(clippy::too_many_arguments)]
pub fn explore_diverse_mixed(
    genome: &Genome, config: &Config, view: &View, methods: &[ScoreMethod],
    n_seeds: usize, max_rounds: usize, log: &mut Logger, radius_mult: f64, scales: &[f64],
) -> Vec<(usize, RoundResult, Vec<Vec<f32>>)> {
    if methods.is_empty() { return Vec::new(); }
    let per_method = (n_seeds / methods.len()).max(1);
    let mut shots: Vec<(usize, RoundResult, Vec<Vec<f32>>)> = Vec::new();
    for (mi, &method) in methods.iter().enumerate() {
        shots.extend(collect_shots(genome, config, view, method, per_method, max_rounds, log, mi * per_method, radius_mult, scales));
    }
    let selected = select_diverse(&shots, shots.len());
    selected.into_iter().map(|i| shots[i].clone()).collect()
}

/// Like `explore_diverse_mixed`'s seed collection, but returns every seed's
/// FULL round history instead of picking one "best" round per seed by raw
/// search score — for a caller that wants to make that pick itself using a
/// more reliable signal (the viewer's Explore button uses the trained
/// aesthetic ensemble). Confirmed necessary, not just nice-to-have:
/// `drill`'s own greedy per-round descent already uses raw score to decide
/// which direction to zoom, so even when a seed's DESCENT passed near
/// genuinely rich structure, raw score alone (not just at the final pick,
/// but at every round along the way) could favor a more boring round by the
/// end — real case: entropy/edge_density/intricacy, computed at
/// `SWEEP_RES=64`, rated a strikingly rich region (nested rings + fine
/// marbled texture) BELOW plain smooth color banding nearby on
/// edge_density (0.68 vs 0.82, gap widening at higher render resolution —
/// not fixable by more pixels), while the trained aesthetic ensemble
/// correctly preferred it (5.68 vs 5.48). No quality floor or fingerprint
/// here either — the caller has the full history to work with.
#[allow(clippy::too_many_arguments)]
pub fn collect_shots_mixed(
    genome: &Genome, config: &Config, view: &View, methods: &[ScoreMethod],
    n_seeds: usize, max_rounds: usize, log: &mut Logger, radius_mult: f64, scales: &[f64],
) -> Vec<(usize, Vec<RoundResult>)> {
    if methods.is_empty() { return Vec::new(); }
    let per_method = (n_seeds / methods.len()).max(1);
    let mut shots: Vec<(usize, Vec<RoundResult>)> = Vec::new();
    for (mi, &method) in methods.iter().enumerate() {
        shots.extend(collect_shots_full(genome, config, view, method, per_method, max_rounds, log, mi * per_method, radius_mult, scales));
    }

    // Bonus seed from the trained navigation model (see
    // `nav_predicted_offset`'s doc comment) — added ON TOP of the
    // requested `n_seeds` grid-sweep seeds above, never in place of one,
    // so an early/data-starved model can't starve the proven heuristic
    // seeds of coverage; worst case (model unavailable or its prediction
    // fails the sanity clamp) this is simply a no-op. `usize::MAX - 1`
    // keeps its id out of the range any `mi * per_method` offset above
    // can reach.
    if let Some((dx, dy, zoom)) = nav_predicted_offset(genome, config, view) {
        let seed_view = apply_offset(view, dx, dy, zoom);
        let seed_id = usize::MAX - 1;
        let history = drill(genome, config, seed_view, max_rounds, methods[0], seed_id, log, true, 0);
        if !history.is_empty() {
            shots.push((seed_id, history));
        }
    }

    shots
}

/// Render `view` at `res`×`res` and save as PNG — the shared "take a
/// screenshot of a discovered spot" primitive.
pub fn save_shot(genome: &Genome, config: &Config, view: &View, res: u32, path: &Path) {
    let use_f64 = needs_f64(view, res);
    let field = render_escape_times(genome, config, view, res, res, config.rendering.max_iter, use_f64, true);
    let rgb = apply_colormap(&field, config.rendering.max_iter, &config.rendering.colormap);
    save_png(&rgb, res, res, path).expect("save screenshot");
}

/// Renders `view` once and asks the trained navigation model (see
/// [[project-nav-imitation-model]]) where Carl would zoom next FROM there
/// — genuinely learned from his own past choices, not another hand-tuned
/// structural heuristic (entropy/edge_density/intricacy/aesthetic ensemble
/// all tried this project, each with confirmed real blind spots). Returns
/// the same `(dx, dy, zoom)` offset-from-`view` shape `find_interesting_
/// square` (viewer.rs) does, so a caller can use it identically — apply
/// via `apply_offset(view, dx, dy, zoom)` when a concrete `View` is
/// needed. `None` if the sidecar/model isn't available (not yet trained)
/// or the prediction fails a basic sanity check — the model was trained
/// only on zoom-IN actions (`drag_zoom`/`zoom_in_*`), so a prediction that
/// doesn't actually deepen the zoom, or lands far outside the current
/// frame, isn't usable; callers fall back to their own heuristic in that
/// case, same graceful-degrade contract as every other optional-sidecar
/// feature in this project. Shared by the viewer's Auto-Select and
/// `collect_shots_mixed`'s bonus seed below — one render + one sidecar
/// call either way, no duplicated predict-then-validate logic.
pub fn nav_predicted_offset(genome: &Genome, config: &Config, view: &View) -> Option<(f64, f64, f64)> {
    let mut predictor = crate::nav_predict::NavPredictor::new(None)?;
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let tmp_path = std::env::temp_dir().join(format!("nav_predict_{}_{nanos}.png", std::process::id()));
    save_shot(genome, config, view, 224, &tmp_path);
    let result = predictor.predict_blocking(tmp_path.clone());
    let _ = std::fs::remove_file(&tmp_path);
    let (u, v, log_zoom_ratio) = result?;
    if !(-1.5..=1.5).contains(&u) || !(-1.5..=1.5).contains(&v) || !log_zoom_ratio.is_finite() || log_zoom_ratio <= 0.0 {
        return None;
    }
    let (dx, dy, zoom) = crate::nav_predict::to_offset(view, u, v, log_zoom_ratio);
    if !zoom.is_finite() { return None; }
    Some((dx, dy, zoom))
}

/// Clone `config` with `max_iter` bumped — boundary detail an evolved
/// genome's own `eval_max_iter` rarely needs this much of (pre-selected for
/// interest already, at a resolution/depth tuned for GA-loop speed, not
/// exploration quality) but a deep multi-round drill routinely does, to
/// resolve filament detail past the first few zoom rounds.
pub fn explore_config(base: &Config) -> Config {
    let mut cfg = base.clone();
    cfg.rendering.max_iter = cfg.rendering.max_iter.max(1000);
    cfg
}
