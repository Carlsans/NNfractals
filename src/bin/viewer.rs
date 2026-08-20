//! NNFractals interactive viewer (egui/eframe).
//!
//! Keyboard shortcuts:
//!   W/A/S/D               Translate view (+Shift=2×, +Alt=½, +Ctrl+Shift=10 radii, +Ctrl+Alt=1/10 radius)
//!   Up/Down arrows        Zoom in/out (same modifiers as WASD)
//!   Left/Right arrows     Cycle palette
//!   Drag (left btn)       Zoom into selection (aspect-locked)
//!   Right-click           Zoom out ×2
//!   Backspace / Ctrl+Z    Undo zoom
//!   R                     Reset view
//!   H / ?                 Toggle help
//!   Ctrl+S                Save PNG
//!   Q / Esc               Quit
use std::io::{BufRead as _, Read as _, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use eframe::egui::{self, Color32, ColorImage, Key, TextureHandle, TextureOptions};
use serde::{Deserialize, Serialize};

use nnfractals::config::Config;
use nnfractals::dd::Dd;
#[cfg(feature = "wgpu-backend")]
use nnfractals::explore::{self, ScoreMethod};
use nnfractals::genome::Genome;
use nnfractals::io::{load_genome, save_genome, save_png};
use nnfractals::aesthetic::AestheticScorer;
use nnfractals::novelty::NoveltyScorer;
#[cfg(feature = "wgpu-backend")]
use nnfractals::render_gpu;
use nnfractals::video_export::{View, effective_max_iter, needs_f64, needs_dd, render_cpu, render_save, save_pool, CapturedView};

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_UNDO: usize = 20;
const MIN_SEL_PX: f32 = 12.0;
// Minimum zoom (zoom = 2/half_y). No practical zoom-out limit — the floor is just
// the smallest positive f64, kept only so half_y = 2/zoom stays finite (avoids
// div-by-zero / NaN). The old 0.05 floor is what capped zoom-out at the ±40 wall.
const MIN_ZOOM: f64 = f64::MIN_POSITIVE;
const MAX_ZOOM: f64 = 1.0e30;

const RATIOS: &[(&str, f64, f64)] = &[
    ("1:1",  1.0, 1.0),
    ("4:3",  4.0, 3.0),
    ("3:2",  3.0, 2.0),
    ("16:9", 16.0, 9.0),
    ("2:1",  2.0, 1.0),
];

const COLORMAPS: &[&str] = &[
    "turbo", "inferno", "viridis", "plasma", "magma", "earth", "neon", "grayscale",
];

// `explorer vae-explore --method`/`--select-by` choices, mirrored here so the
// Explore Options window can offer them as a combo box instead of free text.
const EO_METHODS: &[&str] = &["mixed", "entropy", "edge", "gated-entropy", "gated-edge"];
const EO_SELECT_BY: &[&str] = &["max-error", "min-error", "random"];
// Independent of whatever cap the log-writing side uses (currently
// `video_zoom_explore::LOG_CANDIDATE_CAP`) — a real run tiled the entire
// canvas with one bordered square per logged candidate before that
// existed (Carl's screenshot, 2026-08-13). Keeping this cap here too means
// the overlay stays sane even if the log-side cap is ever raised again for
// diagnostic reasons without remembering the visual impact.
const EO_SCAN_OVERLAY_MAX: usize = 12;

// Known-good "1. Grow corpus" parameter sets, empirically found across a
// real multi-fractal exploration session (2026-08-08/10) rather than
// guessed — see [[project-complex-nn-weekend-research]] memory. Fields:
// (name, hover text, iterations, n_seeds, recursion_depth, top_k, patience, min_improvement).
// Standard is what every one of Mandelbrot/Celtic Mandelbrot/genome_9fd45e
// grew cleanly with; Crash-resistant is what Burning Ship specifically
// needed after Standard's `pick_seeds` found only 6 usable seeds near its
// reference view and iteration 0 couldn't clear the internal 20-zone
// training floor — more top-k/recursion-depth extracts more zones from
// the SAME few seeds instead of (uselessly, it turned out) asking for more.
type EoPreset = (&'static str, &'static str, usize, usize, usize, usize, usize, &'static str);
const EO_PRESETS: &[EoPreset] = &[
    ("Standard", "Worked cleanly for most formulas this session.", 20, 60, 4, 6, 15, "0.02"),
    ("Crash-resistant (few seeds found)",
     "If growth crashes with \"only N images found... need at least 20\" — \
      pick_seeds found too few usable seeds near this view; more top-k/depth \
      extracts more zones from THOSE seeds instead of asking for more seeds \
      (raising n_seeds alone did not help when this happened on Burning Ship).",
     20, 60, 5, 15, 15, "0.02"),
];

// ── Outer-limit search ───────────────────────────────────────────────────────
//
// "How far can you zoom out before the fractal stops filling the frame" —
// searched independently per axis since evolved formulas aren't guaranteed
// to be symmetric. Deliberately reuses the SAME `render_cpu` path interactive
// display already uses (rather than fractal.rs's `render_bounds`, which only
// understands the legacy 58-basis representation, not DAG programs — nearly
// every real saved genome is DAG) so DAG/legacy/julia/phoenix/warp genomes
// all work automatically. Always forced to the f32 tier (`use_f64=false`):
// this search only ever zooms OUT to shallow/moderate ranges, nowhere near
// where f64/DD precision would matter.

const OUTER_LIMIT_RES: u32 = 96;

/// Cheap "is this render mostly one uniform color" check on colormapped RGB
/// output — the search's containment signal. Checked on final pixels rather
/// than raw escape times so it stays representation-agnostic (see above) and
/// arguably matches what a human would call "empty" more directly than a
/// raw escape-time threshold would.
fn view_is_degenerate(rgb: &[u8]) -> bool {
    if rgb.len() < 3 { return true; }
    let n = rgb.len() / 3;
    let mut counts: std::collections::HashMap<(u8, u8, u8), u32> = std::collections::HashMap::new();
    for i in 0..n {
        let px = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        *counts.entry(px).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    (max_count as f32 / n as f32) > 0.95
}

/// Grow a half-extent from `start`, doubling until the render (bounds derived
/// from `extents(g)` → (half_x, half_y), centered at `cx,cy`) goes degenerate,
/// then binary-search-refine between the last non-degenerate and first
/// degenerate step. Returns the largest integer half-extent still
/// non-degenerate, or `None` if even `start` is already degenerate (e.g.
/// triggered from an already-blank view — reported honestly rather than
/// guessed at).
fn search_limit(
    genome: &Genome, config: &Config, cx: f64, cy: f64, start: f64,
    extents: impl Fn(f64) -> (f64, f64),
) -> Option<i32> {
    let render_at = |g: f64| -> bool {
        let (half_x, half_y) = extents(g);
        let view = View {
            cx, cx_lo: 0.0, cy, cy_lo: 0.0,
            zoom: 2.0 / half_y, aspect: half_x / half_y,
        };
        let rgb = render_cpu(genome, config, &view, OUTER_LIMIT_RES, OUTER_LIMIT_RES,
                              config.optimization.eval_max_iter, false, false, true);
        view_is_degenerate(&rgb)
    };

    if render_at(start) { return None; }

    let mut lo = start;
    let mut hi = start;
    loop {
        hi *= 2.0;
        if hi > 1.0e9 { return Some(lo.floor().max(1.0) as i32); } // safety cap
        if render_at(hi) { break; }
        lo = hi;
    }
    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        if render_at(mid) { hi = mid; } else { lo = mid; }
        if hi - lo < 1.0 { break; }
    }
    Some(lo.floor().max(1.0) as i32)
}

/// Runs `search_limit` growing one axis while holding the other at a fixed
/// reference extent, shrinking that reference (starting at 2× `base`) until
/// the STARTING candidate is actually non-degenerate. A reference extent
/// that's too large is just as fatal to the search as one that's too small:
/// it swamps the frame's pixel count with uniform empty background
/// regardless of what the searched axis is doing, which is what made an
/// earlier fixed `8×` multiplier fail outright (caught by
/// `outer_limit_finds_xy_for_classic_mandelbrot` below).
fn search_with_shrinking_reference(
    genome: &Genome, config: &Config, cx: f64, cy: f64,
    start_half: f64, base: f64, grow_x: bool,
) -> Option<i32> {
    let mut reference = base * 2.0;
    for _ in 0..6 {
        let result = if grow_x {
            search_limit(genome, config, cx, cy, start_half, |g| (g, reference))
        } else {
            search_limit(genome, config, cx, cy, start_half, |g| (reference, g))
        };
        if result.is_some() { return result; }
        reference *= 0.5;
    }
    None
}

/// Three independent searches: X (Y held at a self-calibrated reference
/// extent so it's never the bottleneck), Y (mirror), and combined XY (both
/// grow together, square). XY runs first and seeds the reference extent
/// used by X/Y from its own result — no fractal-scale guessing needed.
fn outer_limit_search(genome: &Genome, config: &Config, cx: f64, cy: f64, start_half: f64) -> OuterLimitResult {
    let xy = search_limit(genome, config, cx, cy, start_half, |g| (g, g));
    let base = xy.map(|v| v as f64).unwrap_or(start_half * 4.0);
    let x = search_with_shrinking_reference(genome, config, cx, cy, start_half, base, true);
    let y = search_with_shrinking_reference(genome, config, cx, cy, start_half, base, false);
    OuterLimitResult { cx, cy, x, y, xy }
}

// ── Auto-select ───────────────────────────────────────────────────────────────
//
// "Which square part of the current frame looks most interesting" — scans a
// grid of candidate square sub-regions (square in PLANE units, which is what
// makes them square on SCREEN too, since the aspect setting is exactly what
// keeps plane and pixel aspect matched), at SEVERAL sizes at once (so a
// single click can land on a small tucked-away detail just as easily as a
// broad region — "a more complete sweep... smaller squares/higher zoom"),
// and picks whichever candidate has the highest PNG-compression-ratio
// entropy (`rgb_compression_entropy` below) — the SAME metric this project
// already uses everywhere else to judge a fractal's actual richness (the
// save-gate's Stage 1 prefilter, `structured_ent`, `fitness::
// png_compression_entropy`), rather than a bespoke local-gradient score.
// Compression ratio reflects genuine GLOBAL information content, so it
// isn't fooled the way a local gradient-energy sum can be by e.g. one hard
// edge dominating an otherwise mostly-flat candidate. Degenerate candidates
// (uniform color) are excluded outright via `view_is_degenerate` as a cheap
// pre-filter (skips the PNG-encode cost on obviously-blank candidates),
// same containment signal the outer-limit search uses.
//
// Two earlier versions tried: (1) local gradient energy (`auto_palette_score`)
// alone, and (2) ranking by "unusualness" (distance from the sweep's mean
// color) on top of various detail pre-filters. Both kept drifting onto
// near-empty zones — (2) because color-outlier-ness is symmetric (emptier
// registers as just as "unusual" as richer, however tightly the detail bars
// were set), and (1), it turned out, wasn't a reliable enough richness
// signal on its own either.
//
// Switching to compression entropy fixed THAT, but a real 16-round logged
// session (see `log_auto_select_round`) then showed two more things going
// on, both confirmed from the data rather than guessed at:
//
// 1. The winner was the widest (2×) candidate in EVERY round, never once
//    4×/8×/16×, and mean/max entropy dropped monotonically with scale —
//    already true at round 0 (zoom ~7, nowhere near any precision limit).
//    Raw compression entropy at a FIXED render resolution structurally
//    rewards wider views: they simply capture more macro-scale variation
//    in the same pixel budget, independent of whether a narrower candidate
//    is actually the more interesting square. Fixed by comparing candidates
//    via a per-scale Z-SCORE (how good is this, for a square of ITS size)
//    instead of raw entropy, so narrower scales compete on equal footing —
//    with an absolute floor tied to the sweep's own best raw entropy so
//    z-score normalization can't flip that around into picking the
//    least-bad member of a scale bucket that's uniformly worse than what
//    was found elsewhere in the same sweep.
// 2. Every candidate was rendered with `use_f64: false` regardless of zoom
//    depth. By round 10 (zoom > ~7,000) the degenerate rate among 16×
//    candidates had climbed to over 50% and entropy had collapsed to a flat
//    ~0.06 — not because that area was actually empty, but because f32
//    precision was exhausted well past where the interactive viewer itself
//    would have upgraded to f64. Fixed by using `needs_f64` per candidate,
//    same as every other render path in this file already does.

const AUTO_SELECT_RES: u32 = 64;
const AUTO_SELECT_GRID: usize = 5;
/// Candidate square sizes, as a fraction of the current view's y-extent —
/// 2×/4×/8×/16× zoom options all considered together in one sweep.
const AUTO_SELECT_SCALES: &[f64] = &[0.5, 0.25, 0.125, 0.0625];
/// A candidate must reach at least this fraction of the SWEEP's own best raw
/// entropy (across every scale) to be eligible — self-calibrating floor (not
/// a fixed magic number) so per-scale z-score normalization can't end up
/// picking the least-bad member of a scale bucket that's uniformly worse
/// than what was actually found elsewhere in the sweep.
const AUTO_SELECT_MIN_ENTROPY_FRACTION: f32 = 0.5;

/// Same PNG-compression-ratio entropy metric `fitness::png_compression_entropy`
/// uses (`compressed_bytes / raw_bytes`; ~0.3 boring/flat, ~0.9+ rich detail)
/// — entering at the already-colormapped RGB stage (`render_cpu` applies the
/// colormap internally) instead of raw escape times, which keeps this
/// DAG-compatible the same way `view_is_degenerate`/`auto_palette_score`
/// already are (`fractal::render_bounds`, which `png_compression_entropy`'s
/// usual callers feed with escape times, only understands the legacy
/// formula representation).
fn rgb_compression_entropy(rgb: &[u8], w: u32, h: u32) -> f32 {
    let mut buf = std::io::Cursor::new(Vec::with_capacity(8192));
    image::write_buffer_with_format(
        &mut buf, rgb, w, h, image::ColorType::Rgb8, image::ImageFormat::Png,
    ).unwrap_or(());
    let png_bytes = buf.into_inner().len() as f32;
    let raw_bytes = (w * h * 3) as f32;
    png_bytes / raw_bytes
}

/// Returns `(cx, cy, zoom)` of the square sub-region of `view` with the
/// highest `rgb_compression_entropy` across all `AUTO_SELECT_SCALES`, or
/// `None` if every candidate at every scale came back degenerate (e.g.
/// already zoomed into a uniform area). `zoom` is `4.0 / side` (View's own
/// `half_y = 2.0/zoom`, and `side` is the FULL new span, i.e. `2 * half_y`
/// — so `zoom = 2.0 / (side / 2.0) = 4.0 / side`), meant to be applied with
/// the CALLER's own aspect preserved (not forced to 1.0) — the square
/// becomes the new frame's full height, width follows the current aspect.
/// TEMPORARY diagnostic instrumentation: reported as "inconsistent — after a
/// while the algorithm decided to take a non-optimal path." Rather than
/// guess at a fifth revision, log every candidate considered each round
/// (position, scale-derived zoom, entropy score, degenerate flag) plus the
/// winner and the parent view it searched from, to `auto_select_log.jsonl`
/// — one JSON line per round, same manual-format!-string idiom
/// `optimizer.rs::log_gen_metrics` already uses for this kind of lightweight
/// structured telemetry. Remove once a real sequence has been captured and
/// analyzed. Best-effort (a logging failure must never break auto-select
/// itself).
fn log_auto_select_round(
    view: &View,
    candidates: &[(f64, f64, f64, f32, bool)], // (dx, dy, zoom, entropy, degenerate) — offsets from parent center
    winner: Option<(f64, f64, f64, f32)>,      // (dx, dy, zoom, entropy)
) {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let cand_json: Vec<String> = candidates.iter()
        .map(|(dx, dy, zoom, entropy, degenerate)| format!(
            "{{\"dx\":{dx:.6e},\"dy\":{dy:.6e},\"zoom\":{zoom:.6e},\"entropy\":{entropy:.4},\"degenerate\":{degenerate}}}"
        ))
        .collect();
    let winner_json = match winner {
        Some((dx, dy, zoom, entropy)) => format!(
            "{{\"dx\":{dx:.6e},\"dy\":{dy:.6e},\"zoom\":{zoom:.6e},\"entropy\":{entropy:.4}}}"
        ),
        None => "null".to_string(),
    };
    let line = format!(
        "{{\"t\":{t},\"parent\":{{\"cx\":{:.6},\"cy\":{:.6},\"zoom\":{:.6e}}},\"winner\":{winner_json},\"candidates\":[{}]}}\n",
        view.cx, view.cy, view.zoom, cand_json.join(","),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("auto_select_log.jsonl") {
        use std::io::Write;
        let _ = f.write_all(line.as_bytes());
    }
}

/// Mean and population standard deviation of `values`, as `(mean, std)`.
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() { return (0.0, 0.0); }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    (mean, var.sqrt())
}

/// Pick the winning `(scale_idx, cx, cy, zoom, entropy)` candidate: entropy
/// is normalized to a Z-SCORE within its OWN scale group before comparing
/// across groups — "how good is this, for a square of its size" — instead
/// of comparing raw entropy directly, which a real logged session showed
/// structurally favors wider scales regardless of which candidate is
/// actually more interesting (see the module doc comment above). Candidates
/// below `AUTO_SELECT_MIN_ENTROPY_FRACTION` of the sweep's own best raw
/// entropy are excluded first, so z-score normalization can't flip that
/// bias into picking the least-bad member of a scale group that's uniformly
/// worse than what's available elsewhere in the same sweep. `n_scales` is
/// `AUTO_SELECT_SCALES.len()` (passed explicitly so this stays unit-testable
/// without needing a real render).
fn pick_by_scale_normalized_entropy(
    candidates: &[(usize, f64, f64, f64, f32)],
    n_scales: usize,
) -> Option<(usize, f64, f64, f64, f32)> {
    if candidates.is_empty() { return None; }
    let global_max = candidates.iter().map(|c| c.4).fold(f32::MIN, f32::max);
    let min_entropy = global_max * AUTO_SELECT_MIN_ENTROPY_FRACTION;

    let stats: Vec<(f32, f32)> = (0..n_scales)
        .map(|i| {
            let entropies: Vec<f32> = candidates.iter()
                .filter(|c| c.0 == i)
                .map(|c| c.4)
                .collect();
            mean_std(&entropies)
        })
        .collect();
    let zscore = |scale_idx: usize, entropy: f32| -> f32 {
        let (mean, std) = stats[scale_idx];
        if std < 1e-6 { 0.0 } else { (entropy - mean) / std }
    };

    candidates.iter()
        .filter(|c| c.4 >= min_entropy)
        .max_by(|a, b| zscore(a.0, a.4).partial_cmp(&zscore(b.0, b.4)).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
}

/// Returns `(dx, dy, zoom)`: the winning candidate square's center as an
/// OFFSET from `view`'s own center, plus its zoom. Offsets — not absolute
/// coordinates — because at deep zoom `view.bounds()` (plain f64 `cx -
/// half_x`) silently collapses to zero width once `half_x` drops below
/// ~0.5 ULP of `cx` (confirmed empirically: at zoom 2^55 with a typical
/// deep-zoom coordinate, `bounds()` returns an exact zero-width window
/// while the true width is still ~1e-16). That's precisely the zoom depth
/// DD exists for, so this function must never materialize an absolute
/// f64 position — everything stays relative-to-parent (well within f64
/// precision, since it's bounded by the current view's own span) until
/// the caller adds the offset onto the view's DD center via `cx_dd()`.
// A candidate square's whole search area was, until now, always confined
// STRICTLY INSIDE the frame that produced it — every candidate's edges stay
// within the parent's own square. That means once a round commits to a
// square, every later round can only ever search deeper INSIDE it: if the
// coarse grid that picked that square was slightly off-target (easy at
// extreme zoom, where the true boundary is thinner than a 5x5 grid can
// resolve), there is no way to recover — every subsequent round just
// searches closer to the same already-wrong spot. `radius_mult` > 1.0 lets
// candidates extend outside the naive frame to give a weak round a second,
// wider chance before giving up (see `find_interesting_square`).
const AUTO_SELECT_WIDEN_RADIUS: f64 = 1.6;

/// Sweeps `AUTO_SELECT_SCALES` × `AUTO_SELECT_GRID`² candidate squares
/// around `view`, offsets sampled from a search area `radius_mult` times
/// the naive frame half-width/height. Returns (scored candidates, full log
/// entries including degenerate ones).
fn sweep_candidates(
    genome: &Genome, config: &Config, view: &View,
    cur_w: f64, cur_h: f64, radius_mult: f64, allow_dd: bool,
) -> (Vec<(usize, f64, f64, f64, f32)>, Vec<(f64, f64, f64, f32, bool)>) {
    let search_half_x = (cur_w / 2.0) * radius_mult;
    let search_half_y = (cur_h / 2.0) * radius_mult;

    let mut candidates: Vec<(usize, f64, f64, f64, f32)> = Vec::new();
    let mut log_candidates: Vec<(f64, f64, f64, f32, bool)> = Vec::new();

    for (scale_idx, &scale) in AUTO_SELECT_SCALES.iter().enumerate() {
        let side = cur_h * scale;
        if side <= 0.0 { continue; }
        let half = side / 2.0;
        let new_zoom = 4.0 / side;
        for iy in 0..AUTO_SELECT_GRID {
            for ix in 0..AUTO_SELECT_GRID {
                let t_x = if AUTO_SELECT_GRID > 1 { ix as f64 / (AUTO_SELECT_GRID - 1) as f64 } else { 0.5 };
                let t_y = if AUTO_SELECT_GRID > 1 { iy as f64 / (AUTO_SELECT_GRID - 1) as f64 } else { 0.5 };
                let dx = -search_half_x + half + t_x * (2.0 * search_half_x - side).max(0.0);
                let dy = -search_half_y + half + t_y * (2.0 * search_half_y - side).max(0.0);
                // Candidate center = parent's DD center + a plain-f64 offset.
                // The offset itself is always well within f64 precision (it's
                // bounded by the current frame's span); only the ADDITION to
                // the astronomically precise parent center needs to be DD.
                let cand_cx = view.cx_dd() + Dd::from_f64(dx);
                let cand_cy = view.cy_dd() + Dd::from_f64(dy);
                let cand_view = View {
                    cx: cand_cx.hi, cx_lo: cand_cx.lo,
                    cy: cand_cy.hi, cy_lo: cand_cy.lo,
                    zoom: new_zoom, aspect: 1.0,
                };
                // Precision must track zoom depth like every other render
                // path here — candidates at the deepest scale can be well
                // past f32's precision floor even when the parent view
                // itself is still shallow (see the doc comment above).
                let use_f64 = needs_f64(&cand_view, AUTO_SELECT_RES);
                let rgb = render_cpu(genome, config, &cand_view, AUTO_SELECT_RES, AUTO_SELECT_RES,
                                      config.optimization.eval_max_iter, use_f64, false, allow_dd);
                if view_is_degenerate(&rgb) {
                    log_candidates.push((dx, dy, new_zoom, 0.0, true));
                    continue;
                }
                let entropy = rgb_compression_entropy(&rgb, AUTO_SELECT_RES, AUTO_SELECT_RES);
                log_candidates.push((dx, dy, new_zoom, entropy, false));
                candidates.push((scale_idx, dx, dy, new_zoom, entropy));
            }
        }
    }
    (candidates, log_candidates)
}

fn find_interesting_square(genome: &Genome, config: &Config, view: &View, allow_dd: bool) -> Option<(f64, f64, f64)> {
    let cur_w = 4.0 / view.zoom * view.aspect;
    let cur_h = 4.0 / view.zoom;

    let (mut candidates, mut log_candidates) = sweep_candidates(genome, config, view, cur_w, cur_h, 1.0, allow_dd);

    // Weak pass (most of the grid degenerate): the normal sweep is confined
    // strictly inside the current frame, so it can never recover detail
    // that's just outside — retry once with a wider net (see
    // AUTO_SELECT_WIDEN_RADIUS) before conceding the round.
    let total = AUTO_SELECT_SCALES.len() * AUTO_SELECT_GRID * AUTO_SELECT_GRID;
    if candidates.len() * 5 < total {
        let (wide, wide_log) = sweep_candidates(genome, config, view, cur_w, cur_h, AUTO_SELECT_WIDEN_RADIUS, allow_dd);
        candidates.extend(wide);
        log_candidates.extend(wide_log);
    }

    let winner = pick_by_scale_normalized_entropy(&candidates, AUTO_SELECT_SCALES.len());
    log_auto_select_round(view, &log_candidates, winner.map(|(_, dx, dy, zoom, e)| (dx, dy, zoom, e)));
    winner.map(|(_, dx, dy, zoom, _)| (dx, dy, zoom))
}

// ── Render channel ────────────────────────────────────────────────────────────

struct RenderRequest {
    view:       View,
    w:          u32,
    h:          u32,
    preview:    bool,
    generation: u64,
    colormap:   String,
    angle_coloring: bool,
    allow_dd:   bool,
    // Set when the genome itself changed (IPC load); otherwise None = keep current.
    genome:     Option<Genome>,
}

struct RenderResult {
    pixels:     Vec<u8>,  // RGB flat (3 bytes/pixel)
    w:          u32,
    h:          u32,
    is_preview: bool,
    complete:   bool,
    generation: u64,
}

/// Progress messages from a background hi-res save thread → the UI status line.
enum SaveMsg {
    Started { w: u32, h: u32 },
    Done(PathBuf),
    Failed(String),
}

/// One `explore::explore_diverse_mixed` result, already saved to disk and
/// (if the novelty sidecar was available) scored — see `start_explore`.
struct ExploreResult {
    // Companion .nn always sits next to this PNG with the same stem
    // (`start_explore` writes both from one `stem` variable) — not stored
    // separately, it'd only ever be `png_path.with_extension("nn")`.
    png_path: PathBuf,
    cx: f64, cy: f64, zoom: f64,
    score:   f32,
    novelty: Option<f32>,
}

enum ExploreMsg {
    Done { results: Vec<ExploreResult>, out_dir: PathBuf },
    Failed(String),
}

/// Streamed status from one stage of the "Explore Options" pipeline (VAE
/// growth / complex-export / diversity-selection / clustering — see
/// `App::show_explore_options_window`) — each stage shells out to either
/// the `explorer` binary or a `scripts/*.py` script via `spawn_explore_stage`,
/// same background-thread-plus-mpsc shape every other multi-second
/// operation in this file already uses (`save_rx`/`explore_rx`), just
/// generalized to an arbitrary subprocess instead of an in-process render.
enum ExploreOpsMsg {
    /// One line of the subprocess's stdout/stderr — logged verbatim rather
    /// than parsed, since the underlying tools (vae-explore in particular)
    /// print a lot of free-form progress text (`iteration N: ...`, canvas
    /// render timing, etc.) that's genuinely useful to see live, not just
    /// noise to filter.
    Line(String),
    Done,
    Failed(String),
}

/// One entry from `video_zoom_winners.jsonl` (see `video_zoom_explore::write_winners_manifest`),
/// loaded into the "5. Video-Zoom Explore" gallery once that stage's
/// `ExploreOpsMsg::Done` fires. `chain` is kept around (not just the score)
/// specifically so "Queue this winner" can hand it straight to
/// `QueueItem::waypoints` without re-reading the manifest.
struct VideoZoomWinnerUi {
    rank: usize,
    n_legs: usize,
    final_probe_ratio: Option<f64>,
    ended_reason: String,
    chain: Vec<CapturedView>,
    thumb: Option<TextureHandle>,
    preview_mp4: PathBuf,
}

#[derive(Clone, Copy, Default)]
struct OuterLimitResult {
    // Center the search actually ran at — kept alongside the result (rather
    // than assuming the view's current center still matches) so applying the
    // result to the view stays correct even if the user panned away while
    // the background search was running.
    cx: f64,
    cy: f64,
    x:  Option<i32>,
    y:  Option<i32>,
    xy: Option<i32>,
}

/// Drives the multi-round "Auto-Select" sequence: search the current view
/// in the background, apply the winning square once found, then pause for
/// `AUTO_SELECT_WAIT` (so the user can actually see each step) before
/// either starting the next queued round or going idle.
enum AutoSelectState {
    Idle,
    /// `(dx, dy, zoom)` — dx/dy are offsets from the searched view's own
    /// center, not absolute coordinates (see `find_interesting_square`).
    Searching(mpsc::Receiver<Option<(f64, f64, f64)>>),
    /// Found a square; holding it highlighted in red on the CURRENT
    /// (not-yet-zoomed) view for `AUTO_SELECT_PREVIEW` before actually
    /// applying the zoom, so the user can see what's about to happen.
    /// `dx`/`dy` are offsets from the current view's center (not absolute
    /// coordinates — see `find_interesting_square`'s doc comment for why).
    Previewing { dx: f64, dy: f64, zoom: f64, until: std::time::Instant },
    Waiting(std::time::Instant),
}

const AUTO_SELECT_PREVIEW: std::time::Duration = std::time::Duration::from_secs(1);
const AUTO_SELECT_WAIT: std::time::Duration = std::time::Duration::from_secs(3);
// How many times a single round will step back to the previous zoom level
// and retry before finally reporting "nothing found."
const AUTO_SELECT_MAX_BACKTRACK: u32 = 3;

/// Drives the "Wormhole" sequence: search the CURRENT view in the
/// background for a smaller embedded copy of itself (fractal::
/// wormhole_search), preview-highlight it, then jump. Structurally
/// identical to `AutoSelectState` (same reasoning: single consumption
/// point for queued clicks, preview-before-apply so the user can see what
/// they're about to jump to) — kept as a separate enum/state rather than
/// merged because the two searches are conceptually different actions
/// (maximize local detail vs. find a self-similar echo) that should stay
/// independently clickable/queueable.
enum WormholeState {
    Idle,
    /// `(dx, dy, zoom)` offsets from the searched view's own center — see
    /// `WormholeMatch`'s doc comment (fractal.rs) for why never absolute.
    Searching(mpsc::Receiver<Option<(f64, f64, f64)>>),
    Previewing { dx: f64, dy: f64, zoom: f64, until: std::time::Instant },
    Waiting(std::time::Instant),
}

// ── Preferences ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct ViewerPrefs {
    last_save_width:  u32,
    last_save_height: u32,
    ratio_label:      String,
    colormap:         String,
    window_width:     u32,
    window_height:    u32,
    /// Output folder for hi-res saves. Empty = next to the loaded .nn file.
    /// Remembered across sessions once the user picks one.
    #[serde(default)]
    save_dir:         String,

    // ── Zoom-video export — last-used knobs become the new defaults. The
    // captured start/end POINTS themselves are intentionally not part of
    // this struct: they belong to whichever genome is currently loaded and
    // must not leak into the next one. ──
    #[serde(default = "default_video_steps")]  video_steps:  u32,
    #[serde(default = "default_video_fps")]    video_fps:    u32,
    #[serde(default = "default_video_width")]  video_width:  u32,
    #[serde(default = "default_video_height")] video_height: u32,
    /// Render only every Nth frame, warping the rest from keyframes. 1 =
    /// off (every frame rendered exactly). Persisted like the other video
    /// settings so a chosen speed/accuracy trade survives a restart.
    #[serde(default = "default_video_keyframe_stride")] video_keyframe_stride: u32,
    #[serde(default)] video_invert_coords: bool,
    #[serde(default)] video_invert_range:  bool,
}

fn default_video_steps()  -> u32 { 60 }
fn default_video_fps()    -> u32 { 30 }
fn default_video_width()  -> u32 { 1280 }
fn default_video_height() -> u32 { 720 }
// 16, not 1 (off). Measured on a real 2400-frame 1080x1920 chain: ~8x faster
// AND slightly cleaner, because warping resamples each frame and that
// suppresses the aliasing speckle point-sampling produces. The speedup only
// shows at realistic length — a 240-frame benchmark reported just 1.8x,
// because the per-keyframe padded render is amortised over far fewer warped
// frames. Every video in the 2026-08 batch from f05 on used this and all
// verified clean. Set the KF field to 1 in the video row to disable.
fn default_video_keyframe_stride() -> u32 { 16 }

impl Default for ViewerPrefs {
    fn default() -> Self {
        Self {
            last_save_width:  1920,
            last_save_height: 1080,
            ratio_label:      "1:1".into(),
            colormap:         "turbo".into(),
            window_width:     1024,
            window_height:    768,
            save_dir:         String::new(),
            video_steps:  default_video_steps(),
            video_fps:    default_video_fps(),
            video_width:  default_video_width(),
            video_height: default_video_height(),
            video_keyframe_stride: default_video_keyframe_stride(),
            video_invert_coords: false,
            video_invert_range:  false,
        }
    }
}

impl ViewerPrefs {
    fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save(&self, path: &Path) {
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }
}

// ── Application ───────────────────────────────────────────────────────────────

struct App {
    genome:       Genome,
    config:       Config,
    nn_path:      PathBuf,

    view:         View,
    default_view: View,
    view_stack:   Vec<View>,
    // Navigation-history logging (see [[project-fractal-explorer]] memory /
    // scripts/mine_nav_history.py) — `None` degrades to inert, same
    // contract as the optional aesthetic/novelty sidecars, rather than
    // failing the viewer if the log file can't be opened.
    nav_log: Option<explore::Logger>,
    // Held-arrow-key zoom spans multiple frames (press → repeated
    // no-push zoom while held → release); the "before" view is captured
    // once at press and logged as a single nav event at release, not
    // per-frame (which would flood the log).
    pending_arrow_zoom: Option<(View, &'static str)>,

    req_tx:          mpsc::Sender<RenderRequest>,
    res_rx:          mpsc::Receiver<RenderResult>,
    render_gen:      u64,
    displayed_gen:   u64,
    render_complete: bool,

    texture: Option<TextureHandle>,
    // fractal display area within the window, updated each frame
    frac_rect:  egui::Rect,
    prev_frac_dims: (u32, u32),  // track size changes to avoid redundant re-renders

    drag_start: Option<egui::Pos2>,
    // True for the CURRENT drag (latched at drag_started, since the user
    // could release Shift mid-drag) when Shift was held — routes
    // drag_stopped to commit_mark (queue a saliency training example)
    // instead of commit_selection (zoom navigate). Carl's request,
    // 2026-08-10: "when pressing the shift key and holding the left mouse
    // button, I could select multiple zones of interest."
    mark_drag: bool,
    // Zones marked this way, queued in memory (cx, cy, zoom) until
    // "Save marks" writes them to explorer_out/saliency_manual_marks/ as
    // .nn files (same format a normal saved zone uses) for
    // `retrain-saliency`/`saliency-data --vae-model` to pick up.
    eo_manual_marks: Vec<(f64, f64, f64)>,

    show_help: bool,
    show_save: bool,
    save_w_str: String,
    save_h_str: String,
    save_dir_str: String,

    ratio_idx:    usize,
    colormap_idx: usize,
    // Cosmetic: color by bailout exit-angle (arg z) instead of the normal
    // escape-time palette. DAG genomes only; never affects saved .nn/.png
    // files (try_save/force_save in optimizer.rs always use the standard
    // colormap). Silently inert at deep zoom (DD/f64 paths).
    angle_coloring: bool,
    // Closest known-formula match for the currently loaded genome — computed
    // once per genome load (fractal::known_formula_match), not persisted
    // from here (that happens at save time in optimizer.rs). Empty = no
    // match above threshold.
    known_formula_label: String,
    known_formula_score: f32,

    // XY bound fields — stored so they survive across frames while being edited
    xmin_str:  String,
    xmax_str:  String,
    ymin_str:  String,
    ymax_str:  String,
    sync_xy:   bool,  // true = view changed externally, refresh strings next frame

    prefs:      ViewerPrefs,
    prefs_path: PathBuf,

    // Single-instance IPC: new genome paths arrive here when another launch delegates to us
    ipc_rx: mpsc::Receiver<PathBuf>,

    // Auto-palette: background thread sends winner palette index when done
    auto_pal_rx:   Option<mpsc::Receiver<usize>>,
    auto_pal_busy: bool,

    // True when the currently displayed texture came from a preview render
    displayed_is_preview: bool,

    // Continuous zoom animation — toggled with the Z key
    zoom_anim: bool,

    // egui context clone so background threads (saves) can request a repaint.
    egui_ctx: egui::Context,
    // Hi-res save progress. Worker threads report Started/Done/Failed here; their
    // JoinHandles are kept so a slow save is never lost when the window closes
    // (joined in on_exit). saves_active drives the toolbar "saving…" indicator.
    save_tx:      mpsc::Sender<SaveMsg>,
    save_rx:      mpsc::Receiver<SaveMsg>,
    save_jobs:    Vec<thread::JoinHandle<()>>,
    saves_active: usize,
    save_status:  String,

    // 'b' toggles a grayscale view; remembers the palette to restore on toggle-off.
    binary_prev_idx: Option<usize>,

    // ── Outer-limit finder ──────────────────────────────────────────────
    outer_limit_rx:     Option<mpsc::Receiver<OuterLimitResult>>,
    outer_limit_busy:   bool,
    outer_limit_result: Option<OuterLimitResult>,

    // ── Auto-select ───────────────────────────────────────────────────────
    auto_select_state:  AutoSelectState,
    // Extra rounds requested by clicking again while a sequence is already
    // running/waiting — each click adds one more round rather than
    // restarting or stacking redundant concurrent searches.
    auto_select_queued: u32,
    // Set when a search comes back with nothing sufficiently interesting
    // (`find_interesting_square` returned `None`) — shown next to the
    // button until the next search starts.
    auto_select_message: String,
    // Backtrack attempts left for the CURRENT round. A round that finds
    // nothing (even after find_interesting_square's own internal widened
    // retry) undoes the last committed zoom(s) and searches again from
    // there, rather than giving up immediately — a fixed local grid can
    // walk itself into a dead end at extreme depth, and the only way out is
    // to step back and try a different direction. Reset to
    // AUTO_SELECT_MAX_BACKTRACK at the start of every new round (queue
    // consumption), NOT on each backtrack retry, so one bad round can't eat
    // into the budget of the next.
    auto_select_backtracks_left: u32,
    // How many view_stack levels the NEXT backtrack pops. Starts at 1 and
    // grows with each consecutive failure — since the search is fully
    // deterministic, backing up exactly one level and retrying from the
    // identical view just re-finds the identical doomed candidate forever
    // (confirmed empirically: a naive single-level backtrack oscillates
    // between the same two zoom levels indefinitely instead of exploring
    // anything new). Reset to 1 alongside the backtrack budget.
    auto_select_backtrack_depth: usize,
    // The zoom depth AT the most recent failure within the current
    // failure chain. A round only counts as genuine recovery — and gets to
    // reset the backtrack budget/depth — once it clears this ceiling;
    // merely succeeding isn't enough, since the deterministic search can
    // "succeed" its way right back into the same wall it just backed away
    // from. f64::INFINITY when not currently in a failure chain.
    auto_select_stuck_ceiling: f64,

    // ── Wormhole ─────────────────────────────────────────────────────────
    // Search for a smaller self-similar copy of the CURRENT view embedded
    // inside it, and jump there (see fractal::wormhole_search). Mirrors
    // Auto-Select's state machine exactly (search in background → preview
    // highlight → apply → pause), so repeated clicks chain into successive
    // self-similar jumps the same way repeated Auto-Select clicks chain
    // zoom rounds. No backtracking (unlike Auto-Select): a genuine
    // wormhole match is comparatively rare, so "found nothing" is reported
    // honestly rather than retried from a stepped-back view.
    wormhole_state:   WormholeState,
    wormhole_queued:  u32,
    wormhole_message: String,

    // "Wormhole Video": builds a multi-leg chain (find_wormhole_chain) from
    // the current view in a background thread — several seconds per leg,
    // same reason the single jump above runs off the UI thread — then adds
    // it straight to the export queue (mirrors add_to_queue's own file-copy
    // + queue.json write, just with a Vec<CapturedView> chain instead of a
    // single manually-set start/end pair). Separate from wormhole_state
    // (that one just jumps the live view; this one never touches it).
    wormhole_video_rx:      Option<mpsc::Receiver<Result<(), String>>>,
    wormhole_video_busy:    bool,
    wormhole_video_message: String,

    // ── Explore ──────────────────────────────────────────────────────────
    // "Find diverse/unusual spots within THIS formula": runs
    // explore::explore_diverse_mixed from the current view in a background
    // thread (same reason every other multi-second search here does —
    // several seeds × several rounds would freeze the UI thread), saves a
    // PNG + a matching .nn (genome clone with view_cx/cy/zoom baked in, so
    // each result is independently loadable later — see start_explore's
    // doc comment) per result, then postprocesses with the trained novelty
    // model (novelty::NoveltyScorer — DINOv2 + VICReg head, scored against
    // the real archive at config.output.save_dir, not just this batch) to
    // rank results by how visually unusual they are, most novel first.
    explore_rx:      Option<mpsc::Receiver<ExploreMsg>>,
    explore_busy:    bool,
    explore_message: String,

    // ── Explore Options window (VAE growth / complex-export / diversity
    // selection / clustering — the complex-VAE latent-space exploration
    // pipeline built in scripts/*.py + `explorer vae-explore`/`complex-
    // export`, 2026-08-09) ──────────────────────────────────────────────
    // Clicking "Explore" opens this window instead of immediately running
    // the classic explore_diverse_mixed search (still reachable from a
    // button inside it) — Carl's request: "when clicking explore, open a
    // context window with all the options." One shared busy/stage/message/
    // rx rather than four independent ones: the four stages are meant to
    // run in sequence (grow → export → diversify/cluster) against the SAME
    // out_dir, and only one heavy GPU/CPU job should run at a time anyway
    // (established discipline throughout this project).
    show_explore_options: bool,
    eo_out_dir_str: String,
    // Growth (`explorer vae-explore`)
    eo_iterations:      String,
    eo_n_seeds:         String,
    eo_recursion_depth: String,
    eo_top_k:           String,
    eo_canvas_res:      String,
    eo_method_idx:      usize, // index into EO_METHODS
    eo_select_by_idx:   usize, // index into EO_SELECT_BY
    eo_patience:        String,
    eo_min_improvement: String,
    // Complex-export (`explorer complex-export`)
    eo_export_res:   String,
    eo_export_limit: String,
    // Diversity selection (scripts/select_diverse_latent.py)
    eo_model_path: String,
    eo_top_n:      String,
    // scripts/train_saliency.py checkpoint, defaulted to
    // vae_explore::SALIENCY_DEFAULT_MODEL_PATH (Carl's request, 2026-08-10:
    // "use the saliency model by default") — empty (or a nonexistent path)
    // means "don't pass --saliency-model", the pre-default grid-only
    // behavior. Separate from `eo_model_path` (stages 3/4's VAE checkpoint
    // for diversity selection/clustering) since this is a different model
    // for a different stage.
    eo_saliency_model_path: String,
    // Clustering (scripts/cluster_latent.py)
    eo_min_cluster_size:  String, // empty = auto-sweep
    eo_reps_per_cluster:  String,
    eo_noise_sample:      String,
    // Shared job state
    eo_busy:      bool,
    eo_stage:     String,
    eo_log:       Vec<String>,
    eo_message:   String,
    eo_rx:        Option<mpsc::Receiver<ExploreOpsMsg>>,
    // PID of the currently-running stage's top-level process (the
    // `explorer` binary or `python3`, whichever spawn_explore_stage last
    // launched) — set so the window's Cancel button has something to
    // kill. Not process-group-aware: killing this PID doesn't guarantee
    // a grandchild subprocess (e.g. `explorer vae-explore`'s own
    // internal `python3 train_autoencoder.py` call) dies with it, same
    // simple kill-by-pid approach launcher.rs's own stop button already
    // uses elsewhere in this project.
    eo_child_pid: Option<u32>,
    // True from the moment Cancel is clicked until the killed process's
    // Failed message is drained — lets that message read "cancelled"
    // instead of a raw (and misleading) "FAILED: exited with signal 9".
    eo_cancelling: bool,
    // Live "what is actually happening" preview (Carl's request,
    // 2026-08-10): while a stage runs, periodically scans eo_out_dir_str
    // for the most-recently-modified saved zone PNG and shows it as a
    // thumbnail, plus a live zone count (counted by .nn files, not by
    // parsing log text — robust even if a log line format changes).
    eo_preview_texture:   Option<TextureHandle>,
    eo_preview_stem:      String,
    eo_preview_last_scan: Option<std::time::Instant>,
    eo_zone_count:        usize,
    // (cx, cy, zoom) of the latest saved zone's baked-in view, read from
    // its .nn sibling — drawn as a red square over the main fractal canvas
    // (Carl's request, 2026-08-10) so a running scan is visible in-place,
    // not just as a disconnected thumbnail. `None` whenever nothing's been
    // saved yet or the newest zone's genome failed to load.
    eo_preview_view:      Option<(f64, f64, f64)>,
    // This level's full top-K candidate set (cx, cy, zoom, gate_pass),
    // parsed from the newest "level_scanning" line in vae_explore_log.jsonl
    // — logged by the Rust side BEFORE the expensive precise render, so
    // these show up as "currently being scanned" ahead of (and independent
    // of) whichever of them end up actually saved. Carl's follow-up request
    // (2026-08-10): the single red "last saved" square wasn't enough — he
    // wanted the in-flight, not-yet-validated candidates too.
    eo_scan_candidates:   Vec<(f64, f64, f64, bool)>,
    // This level's seed/canvas view (cx, cy, zoom) — same "level_scanning"
    // log line as `eo_scan_candidates`. Without this, the candidate/saved
    // squares only ever show up if the main canvas ALREADY happens to be
    // looking at the same place the scan currently is, which in practice
    // is almost never true (each seed jumps to a different location/zoom
    // entirely) — Carl reported exactly this: log showing real progress,
    // "but nothing on the screen." `eo_follow_scan` (default on) uses this
    // to auto-navigate the main view there so the overlay is actually
    // meaningful, not just theoretically correct.
    eo_scan_seed_view:    Option<(f64, f64, f64)>,
    eo_follow_scan:       bool,
    // Live view of what the (optional) saliency net predicts — loaded from
    // `saliency_heatmap_latest.png`, overwritten every level by
    // `recursion_level` whenever `--saliency-model` was passed (Carl's
    // request, 2026-08-10: "I want the result of the 2d conv to be visible
    // in the viewer... help me debug and know what the program is doing").
    // `None` the whole run if no saliency model was used — this is a
    // debug view of an optional, unverified feature, not something every
    // run produces. Tracked by mtime (not a stem, since the filename is
    // always the same, overwritten in place) so it only reloads when the
    // file actually changed.
    eo_saliency_texture:  Option<TextureHandle>,
    eo_saliency_mtime:    Option<std::time::SystemTime>,
    // Preset index into EO_PRESETS — combo box at the top of "1. Grow
    // corpus", applies a whole known-good parameter set at once instead
    // of leaving Carl to hand-tune 9 fields (his own feedback, 2026-08-
    // 10, after a session of real trial-and-error finding these).
    eo_preset_idx: usize,

    // "5. Video-Zoom Explore" (`explorer video-zoom-explore`) — shares
    // `eo_out_dir_str`/`eo_busy`/`eo_message`/the scan overlay with every
    // other stage; these are just this stage's own numeric knobs plus its
    // results gallery. `eo_vz_method_idx` reuses EO_METHODS/the same
    // ComboBox pattern stage 1 already uses — the "dropdown list" Carl
    // asked for. Final export width is deliberately NOT a field here — it
    // reads `video_w_str` (the "Add to Queue" section's own width field)
    // directly, since the DD-boundary gate only means anything relative to
    // the video that will actually get exported later.
    eo_vz_depth:       String,
    eo_vz_finalists:   String,
    eo_vz_lookahead:   String,
    eo_vz_method_idx:  usize, // index into EO_METHODS
    eo_vz_top_winners: String,
    eo_vz_n_seeds:     String,
    // Absolute floor on a candidate's raw score (see
    // `video_zoom_explore::VideoZoomOpts::min_score`'s doc comment) — added
    // after a real run spent 40% of its budget drilling into a near-flat
    // zone because relative z-score ranking alone never recognizes "this
    // whole neighborhood is bad," only "this is the best of what's here."
    eo_vz_min_score:   String,
    eo_vz_winners:     Vec<VideoZoomWinnerUi>,
    // Live progress (Carl's request, 2026-08-13: "give me an idea of the
    // progress of the search") — parsed from the latest `"seed_started"`
    // event (`seed_id`/`budget`) plus a count of `"committed_move"` events
    // logged after it. `None` until the first `"seed_started"` line
    // appears. Not a simple fraction-of-total-runtime estimate — depth-
    // first backtracking means later moves aren't uniformly paced — but a
    // real, honest "N of BUDGET real moves spent so far" count.
    eo_vz_progress: Option<(usize, usize, usize)>, // (seed_id, committed_count, budget)

    // Manual gate on the f64→DD precision escalation (off by default — stay
    // on f64 at any zoom depth until the user explicitly opts in). Past a
    // certain depth, whether a direction still has anything left to resolve
    // depends on the formula's own escape-iteration count, not just on
    // arithmetic precision — auto-escalating produces a flat render that
    // looks broken but isn't, which is confusing enough that manual control
    // beats a threshold guessing when DD is actually worth the cost.
    manual_dd: bool,

    // ── Zoom-video export (queued — actually rendered by nnfractals-queue) ──
    video_start: Option<CapturedView>,
    video_end:   Option<CapturedView>,
    video_steps_str: String,
    video_fps_str:   String,
    video_w_str:     String,
    video_kf_str:    String,
    video_h_str:     String,
    // Feedback for the last "Add to Queue" click ("Added to queue ✓" or an
    // error) — no progress/rendering state here anymore, that all lives in
    // the queue window now.
    video_status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext, nn_path: PathBuf, ipc_rx: mpsc::Receiver<PathBuf>) -> anyhow::Result<Self> {
        // Project-root-relative, NOT CWD-relative (`nnfractals::project_root`)
        // — a GUI launch (desktop file / file-manager double-click) can have
        // almost any working directory, unlike a terminal-invoked CLI tool.
        // Previously a bare "config.toml" silently fell back to
        // `default_config()` whenever CWD wasn't the project root — no
        // crash, but wrong colormap/max_iter/etc. defaults with no
        // indication why (Carl, 2026-08-11, hit the SAME root cause as a
        // hard crash in `queue.rs`, which has no such fallback).
        let config = Config::load(&nnfractals::project_root().join("config.toml"))
            .unwrap_or_else(|_| default_config());
        let genome = load_genome(&nn_path)?;
        let (known_formula_label, known_formula_score) =
            nnfractals::fractal::known_formula_match(&genome)
                .map(|(n, s)| (n.to_string(), s))
                .unwrap_or_default();

        let default_view = View::new_square(
            genome.view_cx as f64,
            genome.view_cy as f64,
            genome.view_zoom.max(0.1) as f64,
        );

        let prefs_path = nn_path.parent().unwrap_or(Path::new("."))
            .join("viewer_prefs.toml");
        let prefs = ViewerPrefs::load(&prefs_path);
        let save_dir_str = if prefs.save_dir.is_empty() {
            nn_path.parent().unwrap_or(Path::new(".")).to_string_lossy().into_owned()
        } else {
            prefs.save_dir.clone()
        };

        // Find colormap index from prefs
        let colormap_idx = COLORMAPS.iter().position(|&c| c == prefs.colormap)
            .unwrap_or(0);
        let ratio_idx = RATIOS.iter().position(|(label, _, _)| *label == prefs.ratio_label)
            .unwrap_or(0);

        // Sync config colormap from prefs
        let mut config = config;
        config.rendering.colormap = COLORMAPS[colormap_idx].to_string();

        let ctx = cc.egui_ctx.clone();
        // Requests are UNBOUNDED: a bounded channel silently drops requests once
        // full (try_send), which at deep zoom — where render_cpu is slow and the
        // worker stays busy — left render_gen ahead of any queued request, so the
        // worker parked on recv() and the view froze until the next user event.
        // The worker coalesces to the newest request each loop, so no backlog builds.
        let (req_tx, req_rx) = mpsc::channel::<RenderRequest>();
        let (res_tx, res_rx) = mpsc::sync_channel::<RenderResult>(4);

        {
            let base_genome  = genome.clone();
            let base_config  = config.clone();
            thread::spawn(move || {
                let base_full_iter = base_config.rendering.max_iter;
                let mut config = base_config; // mutable: colormap + iter updated per-request
                let mut genome = base_genome; // mutable: replaced when IPC loads a new file
                let mut pending = req_rx.recv().ok();
                while let Some(req) = pending.take() {
                    let mut latest = req;
                    while let Ok(newer) = req_rx.try_recv() { latest = newer; }

                    // Update genome when a new file was loaded via IPC
                    if let Some(new_g) = latest.genome { genome = new_g; }
                    // Apply the palette from this request (may have changed since startup)
                    config.rendering.colormap = latest.colormap.clone();

                    // Check precision need regardless of whether this is a preview request.
                    // f32/GPU previews are wrong once pixel size drops below f32 precision
                    // (~zoom 1000), so we force f64 for previews at deep zoom too.
                    let use_dd       = needs_dd(&latest.view, latest.w);
                    let use_f64      = use_dd || needs_f64(&latest.view, latest.w);
                    // "effective_preview": true only when f32/GPU is accurate enough
                    let gpu_ok       = !use_f64;
                    let eff_preview  = latest.preview && gpu_ok;

                    // Resolution cap: the render thread controls this so that render_cpu
                    // always returns exactly (rw × rh) pixels with no internal re-sizing.
                    let (rw, rh) = if eff_preview {
                        // GPU/f32 preview at 1/4 resolution — fast and correct at normal zoom
                        ((latest.w / 4).max(1), (latest.h / 4).max(1))
                    } else if use_dd {
                        // DD is ~4–8× slower than f64.  Previews cap at 400px so continuous
                        // zoom stays responsive; the settled full render goes to display
                        // resolution (capped at 1600px) so deep-zoom shots are crisp, not
                        // a stretched 400px blur.  DD coords stay sub-pixel-distinct to ~1e30,
                        // so resolution — not precision — is what governs sharpness here.
                        let cap = if latest.preview { 400u32 } else { 1600u32 };
                        let long = latest.w.max(latest.h).max(1);
                        if long > cap {
                            ((latest.w * cap / long).max(1), (latest.h * cap / long).max(1))
                        } else { (latest.w, latest.h) }
                    } else if use_f64 {
                        // At deep zoom, a "preview" request becomes a quick f64 render
                        // at reduced resolution (180px) — still faster than full 720px
                        let cap = if latest.preview { 180u32 } else { 720 };
                        let long = latest.w.max(latest.h).max(1);
                        if long > cap {
                            ((latest.w * cap / long).max(1), (latest.h * cap / long).max(1))
                        } else { (latest.w, latest.h) }
                    } else {
                        (latest.w, latest.h)
                    };

                    // Step sequence:
                    //   eff_preview        → GPU/f32 single-pass (very fast)
                    //   preview + deep zoom → f64 quick 2-step (correct but still fast)
                    //   full render        → progressive 4-step
                    // Iteration depth scales with zoom (same measured law the
                    // video exporter uses — see `video_export::ZOOM_ITER_GAIN`).
                    // A FIXED cap makes every pixel in a deep viewport reach
                    // the cap and share one escape time, so the image goes
                    // flat single-colour past ~1e11 zoom. That was the cause
                    // of Carl's flat-tailed zoom videos, and the interactive
                    // viewer had exactly the same defect: deep zoom looked
                    // empty here too, which is misleading precisely when
                    // picking a deep target by hand.
                    //
                    // Assigned into `config` as well as passed as the step
                    // cap because `render_cpu` takes its COLORMAP
                    // normalisation cap from `config.rendering.max_iter` —
                    // raising compute depth alone would clamp every escape
                    // time above the base to one colour and change nothing.
                    let full_iter = effective_max_iter(&latest.view, base_full_iter);
                    config.rendering.max_iter = full_iter;
                    let quick_steps: &[u32] = &[8, 64];
                    let single_step = [full_iter];
                    let full_steps = [8u32, 24, 64, full_iter];
                    let steps: &[u32] = if eff_preview {
                        &single_step
                    } else if latest.preview {
                        quick_steps
                    } else {
                        &full_steps
                    };

                    for (i, &iter) in steps.iter().enumerate() {
                        let is_last = i == steps.len() - 1;
                        let pixels  = render_cpu(&genome, &config, &latest.view,
                                                 rw, rh, iter.min(full_iter), use_f64,
                                                 latest.angle_coloring, latest.allow_dd);
                        if res_tx.send(RenderResult {
                            pixels, w: rw, h: rh,
                            is_preview: latest.preview,
                            complete: is_last,
                            generation: latest.generation,
                        }).is_err() { return; }
                        ctx.request_repaint();
                        if is_last { break; }
                        if let Ok(newer) = req_rx.try_recv() {
                            pending = Some(newer);
                            break;
                        }
                    }
                    if pending.is_none() {
                        pending = req_rx.recv().ok();
                    }
                }
            });
        }

        let initial_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(prefs.window_width as f32, prefs.window_height as f32),
        );

        let (xmin, xmax, ymin, ymax) = default_view.bounds();
        let (save_tx, save_rx) = mpsc::channel::<SaveMsg>();
        let video_steps_str = prefs.video_steps.to_string();
        let video_fps_str   = prefs.video_fps.to_string();
        let video_w_str     = prefs.video_width.to_string();
        let video_kf_str    = prefs.video_keyframe_stride.to_string();
        let video_h_str     = prefs.video_height.to_string();
        let mut app = Self {
            genome, config, nn_path,
            view: default_view.clone(),
            default_view,
            view_stack: Vec::new(),
            nav_log: explore::Logger::append(Path::new("nav_log.jsonl")).ok(),
            pending_arrow_zoom: None,
            req_tx, res_rx,
            render_gen: 0, displayed_gen: 0, render_complete: true,
            texture: None,
            frac_rect: initial_rect,
            prev_frac_dims: (0, 0),
            drag_start: None,
            mark_drag: false,
            eo_manual_marks: Vec::new(),
            show_help: false,
            show_save: false,
            save_w_str: prefs.last_save_width.to_string(),
            save_h_str: prefs.last_save_height.to_string(),
            save_dir_str,
            ratio_idx, colormap_idx,
            angle_coloring: false,
            known_formula_label,
            known_formula_score,
            xmin_str: format!("{:.6}", xmin),
            xmax_str: format!("{:.6}", xmax),
            ymin_str: format!("{:.6}", ymin),
            ymax_str: format!("{:.6}", ymax),
            sync_xy: false,
            prefs, prefs_path,
            ipc_rx,
            auto_pal_rx: None,
            auto_pal_busy: false,
            zoom_anim: false,
            displayed_is_preview: false,
            egui_ctx: cc.egui_ctx.clone(),
            save_tx, save_rx,
            save_jobs: Vec::new(),
            saves_active: 0,
            save_status: String::new(),
            binary_prev_idx: None,
            outer_limit_rx: None,
            outer_limit_busy: false,
            outer_limit_result: None,
            auto_select_state: AutoSelectState::Idle,
            auto_select_queued: 0,
            auto_select_message: String::new(),
            auto_select_backtracks_left: 0,
            auto_select_backtrack_depth: 1,
            auto_select_stuck_ceiling: f64::INFINITY,
            wormhole_state: WormholeState::Idle,
            wormhole_queued: 0,
            wormhole_message: String::new(),
            wormhole_video_rx: None,
            wormhole_video_busy: false,
            wormhole_video_message: String::new(),
            explore_rx: None,
            explore_busy: false,
            explore_message: String::new(),
            show_explore_options: false,
            eo_out_dir_str: String::new(),
            eo_iterations: "20".to_string(),
            eo_n_seeds: "60".to_string(),
            eo_recursion_depth: "4".to_string(),
            eo_top_k: "6".to_string(),
            eo_canvas_res: "4095".to_string(),
            eo_method_idx: 0,
            eo_select_by_idx: 0,
            eo_patience: "15".to_string(),
            eo_min_improvement: "0.02".to_string(),
            eo_export_res: "512".to_string(),
            eo_export_limit: "4000".to_string(),
            eo_model_path: "explorer_out/complex_ae/complex_vae_tuned_et.pt".to_string(),
            eo_top_n: "30".to_string(),
            eo_saliency_model_path: nnfractals::vae_explore::SALIENCY_DEFAULT_MODEL_PATH.to_string(),
            eo_min_cluster_size: String::new(),
            eo_reps_per_cluster: "400".to_string(),
            eo_noise_sample: "20".to_string(),
            eo_busy: false,
            eo_stage: String::new(),
            eo_log: Vec::new(),
            eo_message: String::new(),
            eo_rx: None,
            eo_child_pid: None,
            eo_cancelling: false,
            eo_preview_texture: None,
            eo_preview_stem: String::new(),
            eo_preview_last_scan: None,
            eo_zone_count: 0,
            eo_preview_view: None,
            eo_scan_candidates: Vec::new(),
            eo_scan_seed_view: None,
            eo_follow_scan: true,
            eo_saliency_texture: None,
            eo_saliency_mtime: None,
            eo_preset_idx: 0,
            eo_vz_depth: "5".to_string(),
            eo_vz_finalists: "3".to_string(),
            eo_vz_lookahead: "2".to_string(),
            eo_vz_method_idx: 0,
            eo_vz_top_winners: "10".to_string(),
            eo_vz_n_seeds: "1".to_string(),
            eo_vz_min_score: "0.15".to_string(),
            eo_vz_winners: Vec::new(),
            eo_vz_progress: None,
            manual_dd: false,
            video_start: None,
            video_end: None,
            video_steps_str, video_fps_str, video_w_str, video_h_str, video_kf_str,
            video_status: String::new(),
        };
        // Set initial aspect ratio from prefs
        app.apply_ratio(ratio_idx, false);
        app.request_render(false);
        Ok(app)
    }

    fn request_render(&mut self, preview: bool) {
        let w = self.frac_rect.width().round() as u32;
        let h = self.frac_rect.height().round() as u32;
        if w == 0 || h == 0 { return; }
        self.render_gen += 1;
        let _ = self.req_tx.send(RenderRequest {
            view: self.view.clone(), w, h, preview,
            generation: self.render_gen,
            colormap: self.config.rendering.colormap.clone(),
            angle_coloring: self.angle_coloring,
            allow_dd: self.manual_dd,
            genome: None,
        });
    }

    fn push_view(&mut self) -> View {
        let old = self.view.clone();
        if self.view_stack.len() >= MAX_UNDO { self.view_stack.remove(0); }
        self.view_stack.push(old.clone());
        old
    }

    fn view_json(v: &View) -> serde_json::Value {
        serde_json::json!({"cx": v.cx, "cx_lo": v.cx_lo, "cy": v.cy, "cy_lo": v.cy_lo, "zoom": v.zoom, "aspect": v.aspect})
    }

    /// Logs one navigation action (`before` → current `self.view`) to
    /// `nav_log.jsonl` — see [[project-fractal-explorer]] memory /
    /// scripts/mine_nav_history.py for why: a model trained on Carl's own
    /// real navigation choices, not another hand-tuned heuristic. Silently
    /// inert if the log couldn't be opened (`nav_log: None`).
    fn log_nav_event(&mut self, action: &'static str, before: &View) {
        let Some(log) = self.nav_log.as_mut() else { return };
        let genome_id = self.nn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        log.log(&serde_json::json!({
            "event": "nav", "t": t, "action": action, "genome_id": genome_id,
            "before": Self::view_json(before), "after": Self::view_json(&self.view),
        }));
    }

    fn undo_zoom(&mut self) {
        if let Some(prev) = self.view_stack.pop() {
            let before = self.view.clone();
            self.view = prev;
            self.sync_xy = true;
            self.request_render(true);
            self.log_nav_event("undo", &before);
        }
    }

    fn zoom_out(&mut self) {
        let before = self.push_view();
        self.view.zoom = (self.view.zoom * 0.5).clamp(MIN_ZOOM, MAX_ZOOM);
        self.sync_xy = true;
        self.request_render(true);
        self.log_nav_event("zoom_out_rightclick", &before);
    }

    fn current_aspect(&self) -> f64 {
        let (_, rw, rh) = RATIOS[self.ratio_idx];
        rw / rh
    }

    // Change the display aspect ratio, keeping cy and the y range.
    fn apply_ratio(&mut self, idx: usize, save_prefs: bool) {
        self.ratio_idx = idx;
        let (_, rw, rh) = RATIOS[idx];
        let new_asp = rw / rh;
        self.view.aspect = new_asp;
        self.sync_xy = true;
        if save_prefs {
            self.prefs.ratio_label = RATIOS[idx].0.to_string();
            self.prefs.save(&self.prefs_path);
        }
    }

    fn set_colormap(&mut self, idx: usize) {
        self.colormap_idx = idx;
        self.config.rendering.colormap = COLORMAPS[idx].to_string();
        self.prefs.colormap = COLORMAPS[idx].to_string();
        self.prefs.save(&self.prefs_path);
        self.request_render(false);
    }

    fn update_view_from_bounds(&mut self, xmin: f64, xmax: f64, ymin: f64, ymax: f64) {
        if xmax <= xmin || ymax <= ymin { return; }
        // User typed explicit f64 coordinates — reset the lo parts
        self.view.cx     = (xmin + xmax) / 2.0;
        self.view.cx_lo  = 0.0;
        self.view.cy     = (ymin + ymax) / 2.0;
        self.view.cy_lo  = 0.0;
        let yrange       = ymax - ymin;
        let xrange       = xmax - xmin;
        self.view.zoom   = (4.0 / yrange).clamp(MIN_ZOOM, MAX_ZOOM);
        self.view.aspect = xrange / yrange;
        self.sync_xy = true;
    }

    fn poll_render(&mut self, ctx: &egui::Context) -> bool {
        let mut got = false;
        while let Ok(res) = self.res_rx.try_recv() {
            if res.generation >= self.displayed_gen {
                let image = ColorImage::from_rgb([res.w as usize, res.h as usize], &res.pixels);
                self.texture = Some(ctx.load_texture("fractal", image, TextureOptions::LINEAR));
                self.render_complete      = res.complete;
                self.displayed_gen        = res.generation;
                self.displayed_is_preview = res.is_preview;
                got = true;
            }
        }
        got
    }

    // Load a new genome into the viewer (IPC single-instance path).
    fn load_new_genome(&mut self, path: PathBuf) {
        match load_genome(&path) {
            Ok(genome) => {
                self.genome = genome;
                let (label, score) = nnfractals::fractal::known_formula_match(&self.genome)
                    .map(|(n, s)| (n.to_string(), s))
                    .unwrap_or_default();
                self.known_formula_label = label;
                self.known_formula_score = score;
                let dv = View::new_square(
                    self.genome.view_cx as f64,
                    self.genome.view_cy as f64,
                    self.genome.view_zoom.max(0.1) as f64,
                );
                self.nn_path = path;
                self.default_view = dv.clone();
                self.view = dv;
                self.view.aspect = self.current_aspect();
                self.view_stack.clear();
                self.sync_xy = true;
                // Send genome alongside the render request so the thread picks it up
                self.request_render_genome(false);
            }
            Err(e) => eprintln!("[viewer] IPC load failed: {e}"),
        }
    }

    fn request_render_genome(&mut self, preview: bool) {
        let w = self.frac_rect.width().round() as u32;
        let h = self.frac_rect.height().round() as u32;
        if w == 0 || h == 0 { return; }
        self.render_gen += 1;
        let _ = self.req_tx.send(RenderRequest {
            view: self.view.clone(), w, h, preview,
            generation: self.render_gen,
            colormap: self.config.rendering.colormap.clone(),
            angle_coloring: self.angle_coloring,
            allow_dd: self.manual_dd,
            genome: Some(self.genome.clone()),
        });
    }

    // Spawn a thread that renders the fractal at 64×64 with every palette and
    // picks the winner by gradient energy (sum of squared pixel differences).
    fn start_auto_palette(&mut self) {
        if self.auto_pal_busy { return; }
        let (tx, rx) = mpsc::channel::<usize>();
        let genome = self.genome.clone();
        let mut config = self.config.clone();
        let view = self.view.clone();
        self.auto_pal_busy = true;
        self.auto_pal_rx = Some(rx);
        thread::spawn(move || {
            let iter = config.rendering.max_iter.min(128);
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (i, &cmap) in COLORMAPS.iter().enumerate() {
                config.rendering.colormap = cmap.to_string();
                let rgb = render_cpu(&genome, &config, &view, 64, 64, iter, false, false, true);
                let score = auto_palette_score(&rgb, 64, 64);
                if score > best_score { best_score = score; best_idx = i; }
            }
            let _ = tx.send(best_idx);
        });
    }

    /// Spawn the outer-limit search (X/Y/XY) centered on the CURRENT view —
    /// mirrors "keep zooming out until it stops filling the frame." See
    /// `outer_limit_search`'s doc comment for the algorithm.
    fn start_outer_limit_search(&mut self) {
        if self.outer_limit_busy { return; }
        let (tx, rx) = mpsc::channel::<OuterLimitResult>();
        let genome = self.genome.clone();
        let config = self.config.clone();
        let (cx, cy) = (self.view.cx, self.view.cy);
        let start_half = (2.0 / self.view.zoom).max(1e-6);
        self.outer_limit_busy = true;
        self.outer_limit_rx = Some(rx);
        self.outer_limit_result = None;
        thread::spawn(move || {
            let result = outer_limit_search(&genome, &config, cx, cy, start_half);
            let _ = tx.send(result);
        });
    }

    /// Kick off one Auto-Select round: search the CURRENT view in the
    /// background for its most interesting square, transitioning to
    /// `Searching`. The result is applied (and the next round, if any,
    /// queued after a pause) by the poll in `ui()`.
    fn start_auto_select_round(&mut self) {
        self.auto_select_backtracks_left = AUTO_SELECT_MAX_BACKTRACK;
        self.auto_select_backtrack_depth = 1;
        self.auto_select_stuck_ceiling = f64::INFINITY;
        self.spawn_auto_select_search();
    }

    /// Pops up to `levels` entries off `view_stack` (stopping early if it
    /// runs out), landing on whichever view that reaches. Unlike
    /// `undo_zoom`, always pops at least as many levels as requested in one
    /// go — a multi-level backtrack must land on a genuinely different view
    /// before the next search runs, not flicker through each intermediate
    /// one.
    fn backtrack_view(&mut self, levels: usize) {
        for _ in 0..levels {
            match self.view_stack.pop() {
                Some(v) => self.view = v,
                None => break,
            }
        }
        self.sync_xy = true;
        self.request_render(false);
    }

    /// Spawns one search from the current view without touching the
    /// backtrack budget — used both for a fresh round (after
    /// `start_auto_select_round` resets it) and for a backtrack retry
    /// against a stepped-back view (which must NOT get a fresh budget, or
    /// one bad round could retry forever).
    fn spawn_auto_select_search(&mut self) {
        self.auto_select_message.clear();
        let (tx, rx) = mpsc::channel::<Option<(f64, f64, f64)>>();
        let genome = self.genome.clone();
        let config = self.config.clone();
        let view = self.view.clone();
        let allow_dd = self.manual_dd;
        thread::spawn(move || {
            // Try the trained navigation model first (see
            // [[project-nav-imitation-model]]) — one render + one sidecar
            // call, far cheaper than the heuristic grid sweep below, and
            // genuinely learned from Carl's own past choices instead of a
            // hand-tuned entropy proxy. Falls back to the existing sweep
            // whenever the model isn't usable (not yet trained, sidecar
            // unavailable, or its prediction fails the sanity clamp) —
            // same graceful-degrade contract as every other optional-
            // sidecar feature in this project, so Auto-Select never goes
            // fully dark just because a model file is missing.
            let result = Self::nav_predict_square(&genome, &config, &view)
                .or_else(|| find_interesting_square(&genome, &config, &view, allow_dd));
            let _ = tx.send(result);
        });
        self.auto_select_state = AutoSelectState::Searching(rx);
    }

    /// One render of the CURRENT view + one `nav_predict_sidecar.py` call,
    /// converted to the same `(dx, dy, zoom)` offset-from-`view` shape
    /// `find_interesting_square` returns. `None` on any failure (sidecar/
    /// model unavailable, or a prediction that fails the sanity clamp)
    /// — the caller falls back to the heuristic sweep in that case, never
    /// surfaced as an error to Carl. Thin wrapper: the actual predict+
    /// clamp+convert logic lives in `explore::nav_predicted_offset`,
    /// shared with `collect_shots_mixed`'s bonus seed.
    fn nav_predict_square(genome: &Genome, config: &Config, view: &View) -> Option<(f64, f64, f64)> {
        explore::nav_predicted_offset(genome, config, view)
    }

    /// Spawns a wormhole search from the current view. Always DD-capable
    /// internally (`fractal::wormhole_search`'s own render calls, not
    /// gated by `manual_dd`) — the search's whole job is to explore
    /// whatever scale a match happens to sit at, and it needs to behave
    /// identically here and in `backfill_wormhole` (the batch scanner) for
    /// the score shown in the browser to mean the same thing as a live
    /// jump; making it depend on an unrelated interactive-rendering toggle
    /// would break that.
    fn spawn_wormhole_search(&mut self) {
        self.wormhole_message.clear();
        let (tx, rx) = mpsc::channel::<Option<(f64, f64, f64)>>();
        let genome = self.genome.clone();
        let config = self.config.clone();
        let view = self.view.clone();
        thread::spawn(move || {
            let result = nnfractals::fractal::wormhole_search(&genome, &config, &view)
                .map(|m| (m.dx, m.dy, m.zoom));
            let _ = tx.send(result);
        });
        self.wormhole_state = WormholeState::Searching(rx);
    }

    fn show_toolbar(&mut self, ui: &mut egui::Ui) {
        let win_h = ui.ctx().input(|i| i.viewport_rect().height());
        let toolbar_h = (win_h * 0.055).clamp(28.0, 58.0);
        let font_size = (toolbar_h * 0.55).clamp(12.0, 28.0);

        // Content-sized (no exact_size) so the wrapped toolbar can grow to
        // multiple rows when the window is narrower than the menu.
        egui::Panel::top("toolbar")
            .show(ui, |ui| {
                // Scale all button text to match toolbar height
                {
                    let style = ui.style_mut();
                    style.text_styles.insert(
                        egui::TextStyle::Button,
                        egui::FontId::proportional(font_size),
                    );
                    style.text_styles.insert(
                        egui::TextStyle::Body,
                        egui::FontId::proportional(font_size * 0.85),
                    );
                    style.text_styles.insert(
                        egui::TextStyle::Monospace,
                        egui::FontId::monospace(font_size * 0.80),
                    );
                }

                // Wraps onto extra rows when the window width < menu width.
                ui.horizontal_wrapped(|ui| {
                        // ── Translation arrows ──────────────────────────────
                        if ui.button("←").on_hover_text("A — left (Shift=2×, Alt=½)").clicked() {
                            self.do_translate(-1.0, 0.0);
                        }
                        if ui.button("↑").on_hover_text("W — up").clicked() {
                            self.do_translate(0.0, -1.0);
                        }
                        if ui.button("↓").on_hover_text("S — down").clicked() {
                            self.do_translate(0.0, 1.0);
                        }
                        if ui.button("→").on_hover_text("D — right").clicked() {
                            self.do_translate(1.0, 0.0);
                        }

                        ui.separator();

                        // ── Zoom / reset ────────────────────────────────────
                        if ui.button("+").on_hover_text("Up — zoom in").clicked() {
                            self.do_zoom(true, 1.0);
                        }
                        if ui.button("-").on_hover_text("Down — zoom out").clicked() {
                            self.do_zoom(false, 1.0);
                        }
                        if ui.button("R").on_hover_text("R — reset view").clicked() {
                            self.view = self.default_view.clone();
                            self.view.aspect = self.current_aspect();
                            self.view_stack.clear();
                            self.sync_xy = true;
                            self.request_render(false);
                        }

                        ui.separator();

                        // ── Depth + precision mode (kept on the left so it stays
                        //    visible at any zoom; the right side scrolls off in DD) ──
                        let z = self.view.zoom;
                        let depth_str = if z < 10.0 {
                            format!("{:.2}×", z)
                        } else if z < 1.0e6 {
                            format!("{:.0}×", z)
                        } else {
                            format!("1e{:.1}", z.log10())
                        };
                        ui.label(depth_str).on_hover_text("Current zoom depth");
                        let w_mode = self.frac_rect.width() as u32;
                        if needs_dd(&self.view, w_mode) {
                            ui.colored_label(Color32::from_rgb(255, 160, 50), "DD")
                                .on_hover_text("Double-double precision (deep zoom)");
                        } else if needs_f64(&self.view, w_mode) {
                            ui.colored_label(Color32::from_rgb(100, 200, 255), "f64")
                                .on_hover_text("f64 precision");
                        }
                        if self.zoom_anim {
                            ui.colored_label(Color32::from_rgb(100, 255, 100), "Z")
                                .on_hover_text("Auto-zoom animation active");
                        }

                        ui.separator();

                        // ── XY coordinate fields ────────────────────────────
                        let (xmin, xmax, ymin, ymax) = self.view.bounds();
                        if self.sync_xy {
                            // At deep zoom f64 bounds can round xmin==xmax at 6 decimal places.
                            // Use enough digits so the two bounds are visually distinct.
                            let w = self.frac_rect.width() as u32;
                            let prec = if needs_dd(&self.view, w) { 15usize }
                                       else if needs_f64(&self.view, w) { 10 }
                                       else { 6 };
                            self.xmin_str = format!("{:.prec$}", xmin);
                            self.xmax_str = format!("{:.prec$}", xmax);
                            self.ymin_str = format!("{:.prec$}", ymin);
                            self.ymax_str = format!("{:.prec$}", ymax);
                            self.sync_xy = false;
                        }

                        let w_check = self.frac_rect.width() as u32;
                        let field_w = font_size * if needs_dd(&self.view, w_check) { 9.5 }
                                                   else if needs_f64(&self.view, w_check) { 7.5 }
                                                   else { 5.5 };
                        ui.label("x:");
                        let rx = ui.add(egui::TextEdit::singleline(&mut self.xmin_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if rx.lost_focus() {
                            if let Ok(v) = self.xmin_str.trim().parse::<f64>() {
                                let (_, cx, cy, cy2) = self.view.bounds();
                                let before = self.push_view();
                                self.update_view_from_bounds(v, cx, cy, cy2);
                                self.request_render(false);
                                self.log_nav_event("bounds_edit", &before);
                            }
                        }
                        let rx = ui.add(egui::TextEdit::singleline(&mut self.xmax_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if rx.lost_focus() {
                            if let Ok(v) = self.xmax_str.trim().parse::<f64>() {
                                let (cx, _, cy, cy2) = self.view.bounds();
                                let before = self.push_view();
                                self.update_view_from_bounds(cx, v, cy, cy2);
                                self.request_render(false);
                                self.log_nav_event("bounds_edit", &before);
                            }
                        }

                        ui.label("y:");
                        let ry = ui.add(egui::TextEdit::singleline(&mut self.ymin_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if ry.lost_focus() {
                            if let Ok(v) = self.ymin_str.trim().parse::<f64>() {
                                let (cx, cx2, _, cy2) = self.view.bounds();
                                let before = self.push_view();
                                self.update_view_from_bounds(cx, cx2, v, cy2);
                                self.request_render(false);
                                self.log_nav_event("bounds_edit", &before);
                            }
                        }
                        let ry = ui.add(egui::TextEdit::singleline(&mut self.ymax_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if ry.lost_focus() {
                            if let Ok(v) = self.ymax_str.trim().parse::<f64>() {
                                let (cx, cx2, cy, _) = self.view.bounds();
                                let before = self.push_view();
                                self.update_view_from_bounds(cx, cx2, cy, v);
                                self.request_render(false);
                                self.log_nav_event("bounds_edit", &before);
                            }
                        }

                        ui.separator();

                        // ── Aspect ratio ────────────────────────────────────
                        let ratio_label = RATIOS[self.ratio_idx].0;
                        egui::ComboBox::from_id_salt("ratio")
                            .selected_text(ratio_label)
                            .show_ui(ui, |ui| {
                                for (i, (label, _, _)) in RATIOS.iter().enumerate() {
                                    if ui.selectable_label(i == self.ratio_idx, *label).clicked() {
                                        self.apply_ratio(i, true);
                                        self.request_render(false);
                                    }
                                }
                            });

                        ui.separator();

                        // ── Palette ─────────────────────────────────────────
                        if ui.button("<").on_hover_text("Left — previous palette").clicked() {
                            let n = COLORMAPS.len();
                            self.set_colormap((self.colormap_idx + n - 1) % n);
                        }
                        ui.label(COLORMAPS[self.colormap_idx]);
                        if ui.button(">").on_hover_text("Right — next palette").clicked() {
                            self.set_colormap((self.colormap_idx + 1) % COLORMAPS.len());
                        }
                        if self.auto_pal_busy {
                            ui.colored_label(Color32::YELLOW, "...");
                        } else if ui.button("auto")
                            .on_hover_text("Pick best palette by visual gradient score")
                            .clicked()
                        {
                            self.start_auto_palette();
                        }

                        // Cosmetic angle-coloring toggle — DAG genomes only (no
                        // exit-angle data for legacy formulas); never affects
                        // saved .nn/.png files, purely a display option.
                        ui.add_enabled_ui(self.genome.uses_program(), |ui| {
                            if ui.checkbox(&mut self.angle_coloring, "∠")
                                .on_hover_text("Color by bailout exit angle instead of the \
                                                normal escape-time palette — cosmetic, DAG \
                                                genomes only, never affects saved files.")
                                .changed()
                            {
                                self.request_render(false);
                            }
                        });

                        // Closest known-formula match — discovery/curiosity only,
                        // computed once per genome load (see App::new/load_new_genome).
                        // Stable for the lifetime of a loaded genome, so (unlike the
                        // status bar's render-state indicator, see its own doc comment
                        // below) showing/hiding this never depends on anything that
                        // changes mid-render — no risk of the same layout feedback loop.
                        if !self.known_formula_label.is_empty() {
                            ui.separator();
                            let phoenix = self.genome.phoenix_re != 0.0 || self.genome.phoenix_im != 0.0;
                            let label = if phoenix {
                                format!("≈ {} + Phoenix", self.known_formula_label)
                            } else {
                                format!("≈ {}", self.known_formula_label)
                            };
                            ui.colored_label(Color32::from_rgb(180, 180, 255), label)
                                .on_hover_text(format!(
                                    "Closest known-formula match (discovery only, r={:.2}) \
                                     — computed from the base program, ignoring julia/warp/view.",
                                    self.known_formula_score,
                                ));
                        }

                        ui.separator();

                        // ── Help / Save ─────────────────────────────────────
                        let help_label = if self.show_help { "x Help" } else { "? Help" };
                        if ui.button(help_label).clicked() {
                            self.show_help = !self.show_help;
                        }
                        if ui.button("Save").on_hover_text("Ctrl+S").clicked() {
                            self.show_save = true;
                        }

                });
            });
    }

    /// Thin status line drawn directly *below* the toolbar (its own panel), so
    /// the render/save indicators no longer crowd or scroll off the menu.
    ///
    /// Always shown at a fixed height (blank when idle) rather than
    /// conditionally reserving/releasing its strip: this panel's own content
    /// depends on render state (`rendering`), so letting its height vary was
    /// a feedback loop — hiding it grew the fractal panel below, which was
    /// detected as a resize and triggered a fresh render, which re-showed the
    /// bar, shrinking the panel again, triggering another "resize"-render,
    /// forever (very visible on the CPU renderer, where each cycle takes
    /// seconds instead of milliseconds).
    fn show_status_bar(&mut self, ui: &mut egui::Ui) {
        let rendering = !self.render_complete || self.displayed_gen < self.render_gen;
        egui::Panel::top("save_status").show(ui, |ui| {
            ui.horizontal(|ui| {
                // Always draw this label (transparent when idle) rather than
                // omitting it: a min-height on the row isn't enough to pin
                // its height exactly, since a *populated* text label is a
                // couple pixels taller than a bare minimum — leaving a small
                // per-frame height delta between "rendering"/"idle" that was
                // enough to keep re-triggering the resize/render loop above.
                // Same widget in every state, just invisible, guarantees an
                // identical row height regardless of render state.
                let color = if rendering { Color32::YELLOW } else { Color32::TRANSPARENT };
                ui.colored_label(color, "rendering...");
                // Save feedback: active count in blue while rendering to disk,
                // else the last outcome (green Saved / red FAILED).
                if self.saves_active > 0 {
                    ui.colored_label(
                        Color32::LIGHT_BLUE,
                        format!("💾 saving {}…", self.saves_active),
                    );
                }
                if !self.save_status.is_empty() {
                    let col = if self.save_status.starts_with("Save FAILED") {
                        Color32::LIGHT_RED
                    } else if self.save_status.starts_with("Saved") {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_BLUE // "Rendering …"
                    };
                    ui.colored_label(col, &self.save_status);
                }
            });
        });
    }

    /// Outer-limit finder + zoom-video export. A new bottom panel, separate
    /// from the toolbar/status bar above — called before `show_fractal_panel`
    /// so the central panel's remaining-space layout accounts for it.
    fn show_bottom_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("bottom_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                // ── Outer limit ──────────────────────────────────────────
                if self.outer_limit_busy {
                    ui.colored_label(Color32::YELLOW, "finding limit...");
                } else if ui.button("Find Outer Limit")
                    .on_hover_text("Search outward from the current view for the largest \
                                    integer half-extent that still contains most of the \
                                    fractal — independently for X, Y, and combined XY.")
                    .clicked()
                {
                    self.start_outer_limit_search();
                }
                if let Some(r) = self.outer_limit_result {
                    let fmt = |v: Option<i32>| v.map(|n| n.to_string()).unwrap_or_else(|| "—".to_string());
                    ui.monospace(format!("X:±{}  Y:±{}  XY:±{}", fmt(r.x), fmt(r.y), fmt(r.xy)));
                }

                // ── Manual DD toggle ────────────────────────────────────────
                // Off by default: render caps at f64 no matter how deep you
                // zoom. Past a certain depth, whether a direction still has
                // detail to resolve depends on the formula's own escape
                // dynamics, not just precision — DD can't show detail in a
                // direction that escapes too fast for its own chaos to have
                // separated nearby pixels yet at that depth. Auto-escalating
                // produces a flat render that looks broken but isn't, so
                // this stays a deliberate choice instead of a threshold guess.
                if ui.checkbox(&mut self.manual_dd, "DD")
                    .on_hover_text("Allow double-double precision past f64's ~10¹¹ zoom limit. \
                                    Off by default — rendering caps at f64 (correct, but visibly \
                                    pixelated/blocky once you're deeper than that). Toggle on to \
                                    push further; if a direction still looks flat with DD on, \
                                    that direction has run out of resolvable detail at your \
                                    current iteration budget, not a rendering bug.")
                    .changed()
                {
                    self.request_render(false);
                }

                ui.separator();

                // ── Auto-select ───────────────────────────────────────────
                let label = if self.auto_select_queued > 0 {
                    format!("Auto-Select (+{})", self.auto_select_queued)
                } else {
                    "Auto-Select".to_string()
                };
                if ui.button(label)
                    .on_hover_text("Zoom into the most visually interesting square part of the \
                                    current view. Click again while it's running/waiting to queue \
                                    another round — each round pauses a few seconds afterward so \
                                    you can watch the zoom evolve.")
                    .clicked()
                {
                    // Just record the request — a single place (the poll in
                    // `ui()`) is the only thing that ever starts a round, so
                    // one click can never double-fire.
                    self.auto_select_queued += 1;
                }
                match &self.auto_select_state {
                    AutoSelectState::Searching(_) => { ui.colored_label(Color32::YELLOW, "selecting…"); }
                    AutoSelectState::Previewing { until, .. } => {
                        let left = until.saturating_duration_since(std::time::Instant::now()).as_secs_f32();
                        ui.colored_label(Color32::RED, format!("zooming in {left:.1}s…"));
                    }
                    AutoSelectState::Waiting(deadline) => {
                        let left = deadline.saturating_duration_since(std::time::Instant::now()).as_secs_f32();
                        ui.colored_label(Color32::from_rgb(100, 200, 255), format!("next in {left:.0}s…"));
                    }
                    AutoSelectState::Idle => {
                        if !self.auto_select_message.is_empty() {
                            ui.colored_label(Color32::from_rgb(255, 160, 50), &self.auto_select_message);
                        }
                    }
                }

                ui.separator();

                // ── Wormhole ────────────────────────────────────────────
                let wh_label = if self.wormhole_queued > 0 {
                    format!("Wormhole (+{})", self.wormhole_queued)
                } else {
                    "Wormhole".to_string()
                };
                if ui.button(wh_label)
                    .on_hover_text("Find a smaller copy of the current view's own structure \
                                    embedded inside it, and jump there. A real match is only \
                                    approximately similar (exact self-similarity doesn't exist \
                                    in these formulas) — each jump changes what you're looking \
                                    at, not just how zoomed in you are. Click again while \
                                    running to queue another jump. Genuine matches are less \
                                    common than Auto-Select targets, so \"nothing found\" is a \
                                    normal, honest answer here.")
                    .clicked()
                {
                    self.wormhole_queued += 1;
                }
                match &self.wormhole_state {
                    WormholeState::Searching(_) => { ui.colored_label(Color32::YELLOW, "searching…"); }
                    WormholeState::Previewing { until, .. } => {
                        let left = until.saturating_duration_since(std::time::Instant::now()).as_secs_f32();
                        ui.colored_label(Color32::from_rgb(230, 80, 230), format!("jumping in {left:.1}s…"));
                    }
                    WormholeState::Waiting(deadline) => {
                        let left = deadline.saturating_duration_since(std::time::Instant::now()).as_secs_f32();
                        ui.colored_label(Color32::from_rgb(100, 200, 255), format!("next in {left:.0}s…"));
                    }
                    WormholeState::Idle => {
                        if !self.wormhole_message.is_empty() {
                            ui.colored_label(Color32::from_rgb(255, 160, 50), &self.wormhole_message);
                        }
                    }
                }
                if ui.add_enabled(!self.wormhole_video_busy, egui::Button::new("Wormhole Video"))
                    .on_hover_text(format!(
                        "Build a {}-leg video diving through successive wormhole jumps from \
                         the current view, then queue it — each leg is its own bounded, \
                         tractable zoom (see the Wormhole button), so the chain reaches far \
                         deeper overall than any single anchor's precision/iteration budget \
                         could alone. Uses the Steps/FPS/Res below. Takes a while to search \
                         (several seconds per leg) before anything shows up in the queue.",
                        Self::WORMHOLE_VIDEO_LEGS,
                    ))
                    .clicked()
                {
                    let steps = self.video_steps_str.trim().parse::<u32>().unwrap_or(60).max(2);
                    let fps   = self.video_fps_str.trim().parse::<u32>().unwrap_or(30).max(1);
                    let w     = self.video_w_str.trim().parse::<u32>().unwrap_or(1280).max(64);
                    let h     = self.video_h_str.trim().parse::<u32>().unwrap_or(720).max(64);
                    self.prefs.video_steps  = steps;
                    self.prefs.video_fps    = fps;
                    self.prefs.video_width  = w;
                    self.prefs.video_height = h;
                    self.prefs.save(&self.prefs_path);
                    self.start_wormhole_video(steps, fps, w, h);
                }
                if self.wormhole_video_busy {
                    ui.colored_label(Color32::YELLOW, "building chain…");
                } else if !self.wormhole_video_message.is_empty() {
                    let col = if self.wormhole_video_message.starts_with("Wormhole video queued") {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::from_rgb(255, 160, 50)
                    };
                    ui.colored_label(col, &self.wormhole_video_message);
                }

                if ui.button("Explore")
                    .on_hover_text(
                        "Open the exploration options: quick classic search, or the complex-\
                         VAE pipeline (grow a corpus / complex-export / diversity-select / \
                         cluster) against the current genome."
                    )
                    .clicked()
                {
                    if self.eo_out_dir_str.trim().is_empty() {
                        self.eo_out_dir_str = self.save_out_dir()
                            .join("vae_explore")
                            .join(format!("{:016x}", self.genome.id))
                            .to_string_lossy()
                            .into_owned();
                    }
                    self.show_explore_options = true;
                }
                if self.explore_busy {
                    ui.colored_label(Color32::YELLOW, "exploring…");
                } else if !self.explore_message.is_empty() {
                    let col = if self.explore_message.starts_with("Found") {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::from_rgb(255, 160, 50)
                    };
                    ui.colored_label(col, &self.explore_message);
                }

                if !self.eo_manual_marks.is_empty() {
                    ui.separator();
                    ui.colored_label(Color32::from_rgb(0, 220, 220), format!("{} zone(s) marked", self.eo_manual_marks.len()))
                        .on_hover_text(
                            "Zones marked with Shift+drag on the canvas, for saliency-net \
                             training. Save them, then use \"Retrain saliency model\" in \
                             Explore Options to fold them into a new checkpoint."
                        );
                    if ui.button("Save marks")
                        .on_hover_text("Writes each marked zone as a .nn file into explorer_out/saliency_manual_marks/.")
                        .clicked()
                    { self.save_manual_marks(); }
                    if ui.button("Clear marks").clicked() { self.eo_manual_marks.clear(); }
                }

                ui.separator();

                // ── Zoom-video export ────────────────────────────────────
                ui.label("Video:");
                if ui.button("Set Start")
                    .on_hover_text("Capture the current view as the video's start point")
                    .clicked()
                {
                    self.video_start = Some(CapturedView::from_view(&self.view));
                }
                match &self.video_start {
                    Some(s) => { ui.monospace(format!("start ({:.4},{:.4}) @{:.2e}×", s.cx, s.cy, s.zoom)); }
                    None => { ui.colored_label(Color32::GRAY, "start: —"); }
                }
                if ui.button("Set End")
                    .on_hover_text("Capture the current view as the video's end point")
                    .clicked()
                {
                    self.video_end = Some(CapturedView::from_view(&self.view));
                }
                match &self.video_end {
                    Some(e) => { ui.monospace(format!("end ({:.4},{:.4}) @{:.2e}×", e.cx, e.cy, e.zoom)); }
                    None => { ui.colored_label(Color32::GRAY, "end: —"); }
                }

                ui.label("Steps:");
                ui.add(egui::TextEdit::singleline(&mut self.video_steps_str).desired_width(45.0));
                ui.label("FPS:");
                ui.add(egui::TextEdit::singleline(&mut self.video_fps_str).desired_width(35.0));
                ui.label("Res:");
                ui.add(egui::TextEdit::singleline(&mut self.video_w_str).desired_width(55.0));
                ui.label("×");
                ui.add(egui::TextEdit::singleline(&mut self.video_h_str).desired_width(55.0));
                ui.label("KF:");
                ui.add(egui::TextEdit::singleline(&mut self.video_kf_str).desired_width(35.0))
                    .on_hover_text("Keyframe stride: render only every Nth frame and warp the \
                                    rest from the two keyframes around it. 1 = render every \
                                    frame exactly. 16 measured ~8x faster on a real 2400-frame \
                                    chain, and slightly CLEANER — resampling softens the \
                                    aliasing speckle that point-sampling produces.");

                if ui.checkbox(&mut self.prefs.video_invert_coords, "inv.coords")
                    .on_hover_text("Swap which captured point's position feeds the start vs. end")
                    .changed()
                { self.prefs.save(&self.prefs_path); }
                if ui.checkbox(&mut self.prefs.video_invert_range, "inv.range")
                    .on_hover_text("Swap which captured point's zoom feeds the start vs. end")
                    .changed()
                { self.prefs.save(&self.prefs_path); }

                let ready = self.video_start.is_some() && self.video_end.is_some();
                if ui.add_enabled(ready, egui::Button::new("Add to Queue"))
                    .on_hover_text("Queues this export in the video-export window (opened/focused \
                                    automatically) — rendering happens there, in the background, \
                                    so you can keep exploring other fractals here.")
                    .clicked()
                {
                    let steps = self.video_steps_str.trim().parse::<u32>().unwrap_or(60).max(2);
                    let fps   = self.video_fps_str.trim().parse::<u32>().unwrap_or(30).max(1);
                    let w     = self.video_w_str.trim().parse::<u32>().unwrap_or(1280).max(64);
                    let h     = self.video_h_str.trim().parse::<u32>().unwrap_or(720).max(64);
                    let kf    = self.video_kf_str.trim().parse::<u32>().unwrap_or_else(|_| default_video_keyframe_stride()).max(1);
                    self.prefs.video_steps  = steps;
                    self.prefs.video_fps    = fps;
                    self.prefs.video_width  = w;
                    self.prefs.video_height = h;
                    self.prefs.video_keyframe_stride = kf;
                    self.prefs.save(&self.prefs_path);
                    self.add_to_queue(steps, fps, w, h);
                }
                if !self.video_status.is_empty() {
                    let col = if self.video_status.starts_with("Add to queue FAILED") {
                        Color32::LIGHT_RED
                    } else {
                        Color32::LIGHT_GREEN
                    };
                    ui.colored_label(col, &self.video_status);
                }
            });
        });
    }

    fn show_fractal_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ui, |ui| {
                let avail = ui.available_size();
                let asp   = self.current_aspect() as f32;
                let (fw, fh) = if avail.x / avail.y >= asp {
                    (avail.y * asp, avail.y)
                } else {
                    (avail.x, avail.x / asp)
                };

                let offset_x = (avail.x - fw) / 2.0;
                let offset_y = (avail.y - fh) / 2.0;
                let panel_min = ui.min_rect().min;
                let frac_min = egui::Pos2::new(panel_min.x + offset_x, panel_min.y + offset_y);
                let new_rect = egui::Rect::from_min_size(frac_min, egui::Vec2::new(fw, fh));

                // Trigger re-render if fractal area dimensions changed
                let new_dims = (fw.round() as u32, fh.round() as u32);
                if new_dims != self.prev_frac_dims && new_dims.0 > 0 && new_dims.1 > 0 {
                    self.frac_rect = new_rect;
                    self.prev_frac_dims = new_dims;
                    self.request_render(false);
                }
                self.frac_rect = new_rect;

                // Draw fractal texture
                if let Some(tex) = &self.texture {
                    let uv  = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0));
                    ui.painter().image(tex.id(), new_rect, uv, Color32::WHITE);
                }

                // Interaction area (drag + click)
                let resp = ui.allocate_rect(
                    new_rect,
                    egui::Sense::click_and_drag(),
                );

                // Right-click → zoom out (also cancels any in-progress selection)
                if resp.secondary_clicked() {
                    self.drag_start = None;
                    self.zoom_out();
                }

                // Drag → selection rectangle (plain) or a queued saliency
                // training mark (Shift held) — latched at drag_started,
                // not drag_stopped, since the user could release Shift
                // mid-drag.
                if resp.drag_started() {
                    self.drag_start = resp.interact_pointer_pos();
                    self.mark_drag = ui.ctx().input(|i| i.modifiers.shift);
                }
                if resp.drag_stopped() {
                    if let (Some(start), Some(end)) = (self.drag_start.take(), resp.interact_pointer_pos()) {
                        if self.mark_drag {
                            self.commit_mark(start, end, fw, fh);
                        } else {
                            self.commit_selection(start, end, fw, fh);
                        }
                    }
                    self.drag_start = None;
                }
                // A plain click (press+release without a drag) cancels a pending selection —
                // "click away" to abort instead of leaving the viewer stuck in selection mode.
                if resp.clicked() {
                    self.drag_start = None;
                }
                // Safety net: if we think a selection is in progress but no mouse button is
                // actually held (e.g. the release happened off-widget and drag_stopped never
                // fired), clear it so the viewer can't get stuck drawing a selection forever.
                if self.drag_start.is_some() && !ui.ctx().input(|i| i.pointer.any_down()) {
                    self.drag_start = None;
                }

                // Draw selection overlay — cyan while Shift-marking, the
                // usual white+amber otherwise, so the mode is obvious
                // during the drag itself, not just after releasing.
                if let (Some(start), Some(cur)) = (self.drag_start, ui.ctx().input(|i| i.pointer.latest_pos())) {
                    let (sel_rect, ok) = selection_rect(start, cur, fw / fh);
                    if ok {
                        let inner = if self.mark_drag { Color32::from_rgb(0, 220, 220) } else { Color32::from_rgb(255, 200, 0) };
                        let painter = ui.painter();
                        painter.rect_stroke(sel_rect, 0.0, egui::Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Middle);
                        painter.rect_stroke(sel_rect.shrink(1.5), 0.0, egui::Stroke::new(1.0, inner), egui::StrokeKind::Middle);
                    }
                }

                // Persistent overlay of every queued (not-yet-saved) mark —
                // otherwise a mark is invisible again the instant the drag
                // ends, with no way to tell what's already queued.
                if !self.eo_manual_marks.is_empty() {
                    let half_x_frame = 2.0 / self.view.zoom * self.view.aspect;
                    let half_y_frame = 2.0 / self.view.zoom;
                    let to_px = |ox: f64, oy: f64| -> egui::Pos2 {
                        egui::Pos2::new(
                            frac_min.x + (0.5 + ox / (2.0 * half_x_frame)) as f32 * fw,
                            frac_min.y + (0.5 + oy / (2.0 * half_y_frame)) as f32 * fh,
                        )
                    };
                    for &(mx, my, mzoom) in &self.eo_manual_marks {
                        let dd_dx = Dd::from_f64(mx) - self.view.cx_dd();
                        let dd_dy = Dd::from_f64(my) - self.view.cy_dd();
                        let half = 2.0 / mzoom;
                        let rect = egui::Rect::from_two_pos(
                            to_px(dd_dx.hi - half, dd_dy.hi - half),
                            to_px(dd_dx.hi + half, dd_dy.hi + half),
                        );
                        let painter = ui.painter();
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(4.0, Color32::WHITE), egui::StrokeKind::Middle);
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(2.0, Color32::from_rgb(0, 220, 220)), egui::StrokeKind::Middle);
                    }
                }

                // Auto-select preview: highlight the candidate square in red
                // on the CURRENT (not-yet-zoomed) view before actually
                // zooming into it. Mapped via the current view's own span
                // (never `self.view.bounds()`, which collapses to zero width
                // at deep zoom — see `find_interesting_square`) — dx/dy are
                // offsets from screen center, always well within f64 precision.
                if let AutoSelectState::Previewing { dx, dy, zoom, .. } = &self.auto_select_state {
                    let half_y_cand = 2.0 / zoom;
                    let half_x_cand = half_y_cand; // square selection, aspect 1.0
                    let half_x_frame = 2.0 / self.view.zoom * self.view.aspect;
                    let half_y_frame = 2.0 / self.view.zoom;
                    let to_px = |ox: f64, oy: f64| -> egui::Pos2 {
                        egui::Pos2::new(
                            frac_min.x + (0.5 + ox / (2.0 * half_x_frame)) as f32 * fw,
                            frac_min.y + (0.5 + oy / (2.0 * half_y_frame)) as f32 * fh,
                        )
                    };
                    let rect = egui::Rect::from_two_pos(
                        to_px(dx - half_x_cand, dy - half_y_cand),
                        to_px(dx + half_x_cand, dy + half_y_cand),
                    );
                    ui.painter().rect_stroke(rect, 0.0, egui::Stroke::new(2.5, Color32::RED), egui::StrokeKind::Middle);
                }

                // Wormhole preview: same mapping as Auto-Select's, distinct
                // color (magenta) so the two are visually distinguishable
                // when either might be running.
                if let WormholeState::Previewing { dx, dy, zoom, .. } = &self.wormhole_state {
                    let half_y_cand = 2.0 / zoom;
                    let half_x_cand = half_y_cand;
                    let half_x_frame = 2.0 / self.view.zoom * self.view.aspect;
                    let half_y_frame = 2.0 / self.view.zoom;
                    let to_px = |ox: f64, oy: f64| -> egui::Pos2 {
                        egui::Pos2::new(
                            frac_min.x + (0.5 + ox / (2.0 * half_x_frame)) as f32 * fw,
                            frac_min.y + (0.5 + oy / (2.0 * half_y_frame)) as f32 * fh,
                        )
                    };
                    let rect = egui::Rect::from_two_pos(
                        to_px(dx - half_x_cand, dy - half_y_cand),
                        to_px(dx + half_x_cand, dy + half_y_cand),
                    );
                    ui.painter().rect_stroke(rect, 0.0, egui::Stroke::new(2.5, Color32::from_rgb(230, 80, 230)), egui::StrokeKind::Middle);
                }

                // Explore-scan preview: the most recently saved zone from a
                // running "Grow corpus" stage, mapped onto the CURRENT view
                // the same way as Auto-Select's square above — offset (in
                // complex-plane units) from the view's own DD center, not
                // absolute bounds (see the DD bounds invariant note on
                // `find_interesting_square`). Only shown while the stage is
                // actually running; only visible where it geometrically
                // overlaps what's on screen right now (a zone far outside
                // the current view, or from a much shallower/deeper
                // recursion level, simply won't appear — that's accurate,
                // not a bug).
                if self.eo_busy {
                    let half_x_frame = 2.0 / self.view.zoom * self.view.aspect;
                    let half_y_frame = 2.0 / self.view.zoom;
                    let to_px = |ox: f64, oy: f64| -> egui::Pos2 {
                        egui::Pos2::new(
                            frac_min.x + (0.5 + ox / (2.0 * half_x_frame)) as f32 * fw,
                            frac_min.y + (0.5 + oy / (2.0 * half_y_frame)) as f32 * fh,
                        )
                    };

                    // Saliency heatmap tint, drawn UNDERNEATH the squares
                    // below (so a validated/scanning square still stands
                    // out clearly on top of it): stretches the raw
                    // grayscale heatmap texture across the CURRENT seed's
                    // own frame extent, tinted translucent red so bright
                    // (high-predicted-interest) cells glow red and dark
                    // cells stay invisible. Carl's request, 2026-08-10: "I
                    // want the result of the 2d conv to be visible in the
                    // viewer... help me debug and know what the program is
                    // doing." Only present at all if a --saliency-model was
                    // actually passed for this run.
                    if let (Some(tex), Some((sx, sy, szoom))) = (&self.eo_saliency_texture, self.eo_scan_seed_view) {
                        let dd_dx = Dd::from_f64(sx) - self.view.cx_dd();
                        let dd_dy = Dd::from_f64(sy) - self.view.cy_dd();
                        let half = 2.0 / szoom;
                        let rect = egui::Rect::from_two_pos(
                            to_px(dd_dx.hi - half, dd_dy.hi - half),
                            to_px(dd_dx.hi + half, dd_dy.hi + half),
                        );
                        let uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0));
                        ui.painter().image(tex.id(), rect, uv, Color32::from_rgba_unmultiplied(255, 60, 40, 130));
                    }

                    // This level's whole top-K candidate set — logged by
                    // Rust BEFORE the expensive precise render, so these
                    // appear as "currently being scanned" ahead of (and
                    // independent of) whichever end up actually validated
                    // and saved. Orange = passed the cheap coarse gate (will
                    // likely be saved, barring a later dedup rejection);
                    // dim gray = already rejected on coarse metrics (won't
                    // be rendered/saved at all). Carl's follow-up request,
                    // 2026-08-10: the single "last saved" square wasn't
                    // enough, he wanted the in-flight ones too.
                    for &(cx, cy, zoom, gate_pass) in &self.eo_scan_candidates {
                        let dd_dx = Dd::from_f64(cx) - self.view.cx_dd();
                        let dd_dy = Dd::from_f64(cy) - self.view.cy_dd();
                        let half = 2.0 / zoom;
                        let rect = egui::Rect::from_two_pos(
                            to_px(dd_dx.hi - half, dd_dy.hi - half),
                            to_px(dd_dx.hi + half, dd_dy.hi + half),
                        );
                        let col = if gate_pass { Color32::from_rgb(255, 165, 0) } else { Color32::from_gray(180) };
                        // White halo underneath so the line stays visible against
                        // busy/similarly-colored fractal backgrounds, not just flat
                        // ones — a plain 1.5px colored stroke was reported "almost
                        // invisible" in practice.
                        let painter = ui.painter();
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(5.0, Color32::WHITE), egui::StrokeKind::Middle);
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(2.5, col), egui::StrokeKind::Middle);
                    }

                    // Explore-scan preview: the most recently VALIDATED
                    // (saved) zone, mapped onto the CURRENT view the same
                    // way as Auto-Select's square above — offset (in
                    // complex-plane units) from the view's own DD center,
                    // not absolute bounds (see the DD bounds invariant note
                    // on `find_interesting_square`). Drawn last/on top of
                    // the orange candidates so a validated zone always
                    // stands out clearly at its own position.
                    if let Some((zx, zy, zzoom)) = self.eo_preview_view {
                        let dd_dx = Dd::from_f64(zx) - self.view.cx_dd();
                        let dd_dy = Dd::from_f64(zy) - self.view.cy_dd();
                        let half_y_cand = 2.0 / zzoom;
                        let half_x_cand = half_y_cand; // saved zones are square (aspect 1.0)
                        let rect = egui::Rect::from_two_pos(
                            to_px(dd_dx.hi - half_x_cand, dd_dy.hi - half_y_cand),
                            to_px(dd_dx.hi + half_x_cand, dd_dy.hi + half_y_cand),
                        );
                        let painter = ui.painter();
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(6.0, Color32::WHITE), egui::StrokeKind::Middle);
                        painter.rect_stroke(rect, 0.0, egui::Stroke::new(3.0, Color32::RED), egui::StrokeKind::Middle);
                    }
                }

            });
    }

    fn do_translate(&mut self, dx_sign: f64, dy_sign: f64) {
        self.do_translate_scaled(dx_sign, dy_sign, 1.0);
    }

    fn do_translate_scaled(&mut self, dx_sign: f64, dy_sign: f64, scale: f64) {
        let half_x = 2.0 / self.view.zoom * self.view.aspect;
        let half_y = 2.0 / self.view.zoom;
        // /6.0 (was /3.0) halves the WASD pan step; modifier_scale (Alt=½, etc.)
        // still multiplies on top, so WASD and WASD+Alt both halve proportionally.
        let step_x = half_x / 6.0 * scale;
        let step_y = half_y / 6.0 * scale;
        let before = self.push_view();
        // DD-accurate center update: f64 step added to dd center preserves precision at deep zoom
        self.view.set_cx_dd(self.view.cx_dd() + Dd::from_f64(dx_sign * step_x));
        self.view.set_cy_dd(self.view.cy_dd() + Dd::from_f64(dy_sign * step_y));
        self.sync_xy = true;
        self.request_render(true);
        self.log_nav_event("pan", &before);
    }

    fn do_zoom(&mut self, zoom_in: bool, scale: f64) {
        let factor = (1.5_f64).powf(scale);
        let before = self.push_view();
        self.apply_zoom(zoom_in, factor);
        self.log_nav_event(if zoom_in { "zoom_in_btn" } else { "zoom_out_btn" }, &before);
    }

    /// Zoom without pushing to undo stack — for continuous key_down zoom.
    fn do_zoom_nopush(&mut self, zoom_in: bool, scale: f64) {
        let factor = (1.5_f64).powf(scale);
        self.apply_zoom(zoom_in, factor);
    }

    fn apply_zoom(&mut self, zoom_in: bool, factor: f64) {
        if zoom_in {
            self.view.zoom = (self.view.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        } else {
            self.view.zoom = (self.view.zoom / factor).clamp(MIN_ZOOM, MAX_ZOOM);
        }
        self.sync_xy = true;
        self.request_render(true);
    }

    fn commit_selection(&mut self, start: egui::Pos2, end: egui::Pos2, fw: f32, fh: f32) {
        // Guard against a zero/negative-size panel or non-finite pointer coords, which
        // would otherwise produce a NaN/∞ zoom and wedge the render thread.
        if !(fw > 0.0 && fh > 0.0) || start.any_nan() || end.any_nan() {
            return;
        }
        let (sel_rect, ok) = selection_rect(start, end, fw / fh);
        if !ok { return; }

        // Compute the selection center as an offset from the current dd center.
        // Doing it this way (not via pixel_to_fractal which uses hi-only bounds)
        // keeps the lo parts intact at extreme zoom.
        let half_x = 2.0 / self.view.zoom * self.view.aspect;
        let half_y = 2.0 / self.view.zoom;
        let sel_cx_px = ((sel_rect.min.x + sel_rect.max.x) * 0.5 - self.frac_rect.min.x) as f64;
        let sel_cy_px = ((sel_rect.min.y + sel_rect.max.y) * 0.5 - self.frac_rect.min.y) as f64;
        let dx = (sel_cx_px / fw as f64 - 0.5) * 2.0 * half_x;
        let dy = (sel_cy_px / fh as f64 - 0.5) * 2.0 * half_y;

        // Clamp the selection fraction so a sliver can't jump zoom by a huge factor
        // (which at deep zoom makes the DD render take ~forever and looks like a hang).
        let sel_frac = (sel_rect.width() / fw).clamp(0.02, 1.0);
        let new_zoom = (self.view.zoom / sel_frac as f64).clamp(MIN_ZOOM, MAX_ZOOM);
        if !new_zoom.is_finite() { return; }

        let before = self.push_view();
        self.view.set_cx_dd(self.view.cx_dd() + Dd::from_f64(dx));
        self.view.set_cy_dd(self.view.cy_dd() + Dd::from_f64(dy));
        self.view.zoom = new_zoom;
        self.sync_xy = true;
        self.request_render(true);
        self.log_nav_event("drag_zoom", &before);
    }

    /// Shift+drag counterpart to `commit_selection` — same selection-rect
    /// -> (dx, dy, zoom) geometry, but queues the result into
    /// `eo_manual_marks` instead of navigating there. Carl's request,
    /// 2026-08-10: a way to point out fractal regions the saliency net is
    /// currently missing, to feed back into training.
    fn commit_mark(&mut self, start: egui::Pos2, end: egui::Pos2, fw: f32, fh: f32) {
        if !(fw > 0.0 && fh > 0.0) || start.any_nan() || end.any_nan() {
            return;
        }
        let (sel_rect, ok) = selection_rect(start, end, fw / fh);
        if !ok { return; }

        let half_x = 2.0 / self.view.zoom * self.view.aspect;
        let half_y = 2.0 / self.view.zoom;
        let sel_cx_px = ((sel_rect.min.x + sel_rect.max.x) * 0.5 - self.frac_rect.min.x) as f64;
        let sel_cy_px = ((sel_rect.min.y + sel_rect.max.y) * 0.5 - self.frac_rect.min.y) as f64;
        let dx = (sel_cx_px / fw as f64 - 0.5) * 2.0 * half_x;
        let dy = (sel_cy_px / fh as f64 - 0.5) * 2.0 * half_y;

        let sel_frac = (sel_rect.width() / fw).clamp(0.02, 1.0);
        let mark_zoom = (self.view.zoom / sel_frac as f64).clamp(MIN_ZOOM, MAX_ZOOM);
        if !mark_zoom.is_finite() { return; }

        let mark_cx = (self.view.cx_dd() + Dd::from_f64(dx)).hi;
        let mark_cy = (self.view.cy_dd() + Dd::from_f64(dy)).hi;
        self.eo_manual_marks.push((mark_cx, mark_cy, mark_zoom));
    }

    /// Writes every queued mark as a `.nn` file (same format a normal
    /// saved zone uses — genome clone with view_cx/cy/zoom baked in) into
    /// `explorer_out/saliency_manual_marks/`, append-mode numbered like
    /// every other resumable output in this project. Uses the CURRENT
    /// `self.genome` for every mark in this batch — correct for the
    /// expected workflow (mark a few spots on one genome, save, repeat),
    /// not necessarily correct if the genome was switched mid-batch
    /// without saving first.
    fn save_manual_marks(&mut self) {
        if self.eo_manual_marks.is_empty() {
            self.eo_message = "no marks queued — Shift+drag on the canvas first".to_string();
            return;
        }
        let dir = PathBuf::from("explorer_out/saliency_manual_marks");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.eo_message = format!("cannot create {}: {e}", dir.display());
            return;
        }
        let mut next: usize = std::fs::read_dir(&dir)
            .map(|rd| rd.filter_map(|e| e.ok())
                .filter_map(|e| e.path().file_stem().and_then(|s| s.to_str().and_then(|s| s.strip_prefix("mark_")).map(str::to_string)))
                .filter_map(|n| n.parse::<usize>().ok())
                .max().map_or(0, |m| m + 1))
            .unwrap_or(0);
        let mut saved = 0usize;
        for &(cx, cy, zoom) in &self.eo_manual_marks {
            let mut g = self.genome.clone();
            g.view_cx = cx as f32;
            g.view_cy = cy as f32;
            g.view_zoom = zoom as f32;
            if save_genome(&g, &dir.join(format!("mark_{next:04}.nn"))).is_ok() {
                next += 1;
                saved += 1;
            }
        }
        self.eo_manual_marks.clear();
        self.eo_message = format!("saved {saved} mark(s) to {} — use Retrain saliency model to fold them in", dir.display());
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        // Track whether a text field is capturing input (blocks WASD / palette / undo)
        let any_focused = ctx.memory(|m| m.focused().is_some());

        ctx.input(|i| {
            let mods  = i.modifiers;
            let scale = modifier_scale(&mods);

            // Q / Esc → quit (always active)
            if i.key_pressed(Key::Q) || i.key_pressed(Key::Escape) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }

            // Arrow Up/Down: zoom — always active even when a text field has focus.
            // UP/DOWN don't navigate single-line TextEdit fields, so this is safe.
            // Blocked keys (WASD, palette arrows) that DO conflict are handled below.
            let zoom_pressed_in  = i.key_pressed(Key::ArrowUp);
            let zoom_pressed_out = i.key_pressed(Key::ArrowDown);
            // A held key repeats `do_zoom_nopush` every frame (no per-frame push,
            // see its doc comment) — capture `before` once at press and log the
            // whole gesture as a single nav event at release, not per-frame.
            if zoom_pressed_in  { let before = self.push_view(); self.pending_arrow_zoom = Some((before, "zoom_in_key")); }
            if zoom_pressed_out { let before = self.push_view(); self.pending_arrow_zoom = Some((before, "zoom_out_key")); }
            if i.key_down(Key::ArrowUp)   { self.do_zoom_nopush(true,  0.03 * scale); }
            if i.key_down(Key::ArrowDown)  { self.do_zoom_nopush(false, 0.03 * scale); }
            if (i.key_released(Key::ArrowUp) || i.key_released(Key::ArrowDown))
                && !i.key_down(Key::ArrowUp) && !i.key_down(Key::ArrowDown)
            {
                if let Some((before, action)) = self.pending_arrow_zoom.take() {
                    self.log_nav_event(action, &before);
                }
            }

            // Z: toggle zoom animation — always active
            if i.key_pressed(Key::Z) { self.zoom_anim = !self.zoom_anim; }

            // Keys below conflict with text editing — skip when a field has focus
            if any_focused { return; }

            // Space: quick-save at the default resolution into the default folder
            if i.key_pressed(Key::Space) { self.quick_save(); }

            // WASD translation
            if i.key_pressed(Key::W) { self.do_translate_scaled( 0.0, -1.0, scale); }
            if i.key_pressed(Key::S) { self.do_translate_scaled( 0.0,  1.0, scale); }
            if i.key_pressed(Key::A) { self.do_translate_scaled(-1.0,  0.0, scale); }
            if i.key_pressed(Key::D) { self.do_translate_scaled( 1.0,  0.0, scale); }

            // B: toggle grayscale view (press again to restore the previous palette)
            if i.key_pressed(Key::B) {
                let gray = COLORMAPS.iter().position(|&c| c == "grayscale").unwrap_or(0);
                if self.colormap_idx == gray {
                    let restore = self.binary_prev_idx.take().unwrap_or(0);
                    self.set_colormap(restore);
                } else {
                    self.binary_prev_idx = Some(self.colormap_idx);
                    self.set_colormap(gray);
                }
            }

            // Arrow left/right: palette (blocked when field focused — conflicts with cursor movement)
            if i.key_pressed(Key::ArrowLeft) {
                let n = COLORMAPS.len();
                self.set_colormap((self.colormap_idx + n - 1) % n);
            }
            if i.key_pressed(Key::ArrowRight) {
                self.set_colormap((self.colormap_idx + 1) % COLORMAPS.len());
            }

            // R: reset
            if i.key_pressed(Key::R) {
                let before = self.view.clone();
                self.view = self.default_view.clone();
                self.view.aspect = self.current_aspect();
                self.view_stack.clear();
                self.sync_xy = true;
                self.request_render(false);
                self.log_nav_event("reset", &before);
            }

            // H / ?: help
            if i.key_pressed(Key::H) {
                self.show_help = !self.show_help;
            }

            // Backspace / Ctrl+Z: undo
            if i.key_pressed(Key::Backspace)
                || (mods.ctrl && i.key_pressed(Key::Z))
            {
                self.undo_zoom();
            }

            // Ctrl+S: save
            if mods.ctrl && i.key_pressed(Key::S) {
                self.show_save = true;
            }
        });
    }

    fn show_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help { return; }
        egui::Window::new("Controls")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("help_grid").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    let rows: &[(&str, &str)] = &[
                        ("W/A/S/D",          "Translate (Shift=2x, Alt=1/2, Ctrl+Shift=10r, Ctrl+Alt=r/10)"),
                        ("↑ / ↓ (hold)",     "Zoom in / out continuously (same modifiers)"),
                        ("Z",                "Toggle auto-zoom animation toward center"),
                        ("← / →",            "Previous / next palette"),
                        ("Drag (left btn)",  "Zoom into selection"),
                        ("Shift+Drag",       "Mark a zone of interest for saliency-net training (queued, top toolbar)"),
                        ("Right-click",      "Zoom out x2"),
                        ("Backspace/Ctrl+Z", "Undo zoom"),
                        ("R",                "Reset view"),
                        ("H / ?",            "Toggle this help"),
                        ("Ctrl+S",           "Save PNG (dialog)"),
                        ("Space",            "Quick-save at default resolution/folder"),
                        ("Q / Esc",          "Quit"),
                        ("Status: f64",      "Using f64 precision (deep zoom)"),
                        ("Status: DD",       "Double-double precision (~10^30 depth limit)"),
                    ];
                    for (key, desc) in rows {
                        ui.monospace(*key);
                        ui.label(*desc);
                        ui.end_row();
                    }
                });
                if ui.button("Close").clicked() {
                    self.show_help = false;
                }
            });
    }

    /// Render an sw×sh PNG on a background thread and write it into `out_dir`.
    /// Shared by the Save dialog and the spacebar quick-save.
    fn spawn_save(&mut self, sw: u32, sh: u32, out_dir: PathBuf) {
        let genome  = self.genome.clone();
        let config  = self.config.clone();
        let view    = self.view.clone();
        let nn_path = self.nn_path.clone();
        let tx      = self.save_tx.clone();
        let ctx     = self.egui_ctx.clone();
        let angle_coloring = self.angle_coloring;
        let allow_dd = self.manual_dd;
        self.saves_active += 1;
        // Logged at click time (Carl's decision to save), not when the
        // background render finishes — a completed save is a much higher-
        // confidence "this destination was good" signal than a navigation
        // step abandoned mid-exploration, worth recording even without the
        // eventual on-disk path (that's only known after collision-avoidance
        // renaming, inside the background thread).
        if let Some(log) = self.nav_log.as_mut() {
            let genome_id = self.nn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            log.log(&serde_json::json!({
                "event": "save", "t": t, "genome_id": genome_id, "view": Self::view_json(&self.view),
            }));
        }
        self.save_status = format!("Rendering {sw}×{sh} PNG…");
        // Captured NOW (click time), not when the render finishes: renders vary
        // wildly in duration (deep-zoom DD renders are much slower than shallow
        // ones), so a fast save queued after a slow one can finish first and get
        // an earlier on-disk mtime — inverting file-manager "sort by modified"
        // order relative to the order you actually saved them in. Stamping the
        // finished file with the click time keeps that order intact.
        let click_time = std::time::SystemTime::now();
        let handle = thread::spawn(move || {
            let _ = tx.send(SaveMsg::Started { w: sw, h: sh });
            ctx.request_repaint();
            // Confine to the dedicated save pool so a deep-zoom save can't starve the
            // interactive render's cores (which would freeze zooming until save finished).
            let rgb  = save_pool().install(|| render_save(&genome, &config, &view, sw, sh, angle_coloring, allow_dd));
            let stem = nn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("fractal");
            if let Err(e) = std::fs::create_dir_all(&out_dir) {
                let _ = tx.send(SaveMsg::Failed(format!("cannot create {}: {e}", out_dir.display())));
                ctx.request_repaint();
                return;
            }
            // Collision-safe name: the cx/cy/zoom in the filename are rounded, so two
            // nearby views would otherwise map to the same path and silently overwrite.
            // Append _2, _3, … when the target already exists.
            let base = format!("{stem}_cx{:.4}_cy{:.4}_z{:.2}_{sw}x{sh}", view.cx, view.cy, view.zoom);
            let mut out = out_dir.join(format!("{base}.png"));
            let mut n = 2;
            while out.exists() {
                out = out_dir.join(format!("{base}_{n}.png"));
                n += 1;
            }
            match save_png(&rgb, sw, sh, &out) {
                Ok(_) => {
                    // Best-effort: if this fails (unusual filesystem, permissions),
                    // the file still exists with its real write-time mtime — just
                    // not in click order. Not worth failing the save over.
                    if let Ok(f) = std::fs::File::open(&out) {
                        let _ = f.set_modified(click_time);
                    }
                    let _ = tx.send(SaveMsg::Done(out));
                }
                Err(e) => { let _ = tx.send(SaveMsg::Failed(e.to_string())); }
            }
            ctx.request_repaint();
        });
        self.save_jobs.push(handle);
    }

    /// Output folder for saves: the remembered folder, else the loaded .nn's dir.
    fn save_out_dir(&self) -> PathBuf {
        let s = self.save_dir_str.trim();
        if s.is_empty() {
            self.nn_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            PathBuf::from(s)
        }
    }

    /// Spawn the zoom-video export: renders `steps` frames interpolating from
    /// the captured start to end point (respecting the invert prefs) and
    /// encodes them via `ffmpeg` in one background thread — mirrors
    /// `spawn_save`'s dedicated-pool + progress-channel pattern.
    /// Copy the current genome's `.nn` into `video_queue/`, append a job to
    /// `video_queue/queue.json`, then wake (or launch) the queue window.
    /// Actual rendering happens entirely in `nnfractals-queue` — this is
    /// just a fast file-copy + JSON append, no thread needed.
    fn add_to_queue(&mut self, steps: u32, fps: u32, w: u32, h: u32) {
        let keyframe_stride = self.prefs.video_keyframe_stride.max(1);
        let (Some(start), Some(end)) = (self.video_start, self.video_end) else { return };

        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{:x}", d.as_nanos()))
            .unwrap_or_else(|_| format!("{:016x}", self.genome.id));
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let genome_label = self.nn_path.file_stem().and_then(|s| s.to_str())
            .unwrap_or("fractal").to_string();

        let qdir = nnfractals::video_export::queue_dir();
        if let Err(e) = std::fs::create_dir_all(&qdir) {
            self.video_status = format!("Add to queue FAILED: {e}");
            return;
        }
        let nn_filename = format!("{id}.nn");
        if let Err(e) = std::fs::copy(&self.nn_path, qdir.join(&nn_filename)) {
            self.video_status = format!("Add to queue FAILED: {e}");
            return;
        }

        let item = nnfractals::video_export::QueueItem {
            id,
            nn_filename,
            genome_label,
            start,
            end,
            steps,
            fps,
            width: w,
            height: h,
            invert_coords: self.prefs.video_invert_coords,
            invert_range: self.prefs.video_invert_range,
            colormap: self.config.rendering.colormap.clone(),
            angle_coloring: self.angle_coloring,
            output_dir: self.save_out_dir().to_string_lossy().into_owned(),
            status: nnfractals::video_export::QueueStatus::Pending,
            output_path: None,
            error: None,
            created_at,
            waypoints: Vec::new(),
            chain_label: None,
            keyframe_stride,
        };
        let mut items = nnfractals::video_export::load_queue();
        items.push(item);
        nnfractals::video_export::save_queue(&items);

        wake_or_launch_queue_window();
        self.video_status = "Added to queue ✓".to_string();
    }

    /// Builds a wormhole chain from the CURRENT view in a background thread
    /// (each leg is its own multi-second search — see `spawn_wormhole_search`'s
    /// doc comment for why this can't run on the UI thread) and, once found,
    /// queues it exactly like `add_to_queue` does — copy the genome into
    /// `queue_dir()`, append a `QueueItem`, wake/launch the queue window —
    /// just with a `waypoints` chain instead of a single manually-set
    /// start/end pair. Never touches the live view (unlike the single-jump
    /// Wormhole button); this only ever produces a queued video.
    const WORMHOLE_VIDEO_LEGS: usize = 4;
    fn start_wormhole_video(&mut self, steps: u32, fps: u32, w: u32, h: u32) {
        if self.wormhole_video_busy { return; }
        self.wormhole_video_busy = true;
        self.wormhole_video_message.clear();

        let (tx, rx) = mpsc::channel::<Result<(), String>>();
        let genome = self.genome.clone();
        let config = self.config.clone();
        let view = self.view.clone();
        let nn_path = self.nn_path.clone();
        let genome_id = self.genome.id;
        let invert_coords = self.prefs.video_invert_coords;
        let invert_range = self.prefs.video_invert_range;
        let colormap = self.config.rendering.colormap.clone();
        let angle_coloring = self.angle_coloring;
        let output_dir = self.save_out_dir().to_string_lossy().into_owned();
        let genome_label = self.nn_path.file_stem().and_then(|s| s.to_str())
            .unwrap_or("fractal").to_string();

        // Captured BEFORE the move: the queue item is built inside the
        // thread, where `self` is not available.
        let keyframe_stride = self.prefs.video_keyframe_stride.max(1);
        thread::spawn(move || {
            let waypoints = nnfractals::video_export::find_wormhole_chain(
                &genome, &config, &view, Self::WORMHOLE_VIDEO_LEGS,
            );
            if waypoints.len() < 2 {
                let _ = tx.send(Err(
                    "No confident self-similar copy found from the current view — try a different location.".to_string(),
                ));
                return;
            }

            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| format!("{:x}", d.as_nanos()))
                .unwrap_or_else(|_| format!("{:016x}", genome_id));
            let created_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let qdir = nnfractals::video_export::queue_dir();
            if let Err(e) = std::fs::create_dir_all(&qdir) {
                let _ = tx.send(Err(format!("Add to queue FAILED: {e}")));
                return;
            }
            let nn_filename = format!("{id}.nn");
            if let Err(e) = std::fs::copy(&nn_path, qdir.join(&nn_filename)) {
                let _ = tx.send(Err(format!("Add to queue FAILED: {e}")));
                return;
            }

            let item = nnfractals::video_export::QueueItem {
                id,
                nn_filename,
                genome_label,
                start: waypoints[0],
                end: *waypoints.last().unwrap(),
                steps, fps, width: w, height: h,
                invert_coords, invert_range, colormap, angle_coloring,
                output_dir,
                status: nnfractals::video_export::QueueStatus::Pending,
                output_path: None,
                error: None,
                created_at,
                waypoints,
                chain_label: Some("wormhole chain".to_string()),
            keyframe_stride,
            };
            let mut items = nnfractals::video_export::load_queue();
            items.push(item);
            nnfractals::video_export::save_queue(&items);

            wake_or_launch_queue_window();
            let _ = tx.send(Ok(()));
        });
        self.wormhole_video_rx = Some(rx);
    }

    /// How many seeds "Explore" drills before diversity-filtering. Runs on a
    /// click, in a background thread (doesn't block the UI), and Carl
    /// explicitly asked for this to be thorough rather than resource-
    /// conscious after a run confined to a single boring view returned
    /// nothing but rainbow banding in all 8 results — raised well past the
    /// original "modest, click-scale" sizing (was 12/5). Split across all 4
    /// scoring methods (see `explore::explore_diverse_mixed`'s doc comment
    /// for why a single fixed method systematically narrows what's found).
    const EXPLORE_SEEDS: usize = 80;
    const EXPLORE_ROUNDS: usize = 8;

    /// Runs `explore::explore_diverse_mixed` from the CURRENT view against
    /// the CURRENTLY LOADED genome (any formula, not just Mandelbrot — the
    /// search engine itself never assumed which one) in a background
    /// thread, same reason every other multi-second search in this file
    /// does. Each surviving result gets saved as a PNG **and** a matching
    /// `.nn` (a clone of the loaded genome with view_cx/cy/zoom overwritten
    /// to the discovered spot — plain f32, no DD: `explore_diverse_mixed`
    /// never leaves the GPU-solvable range, `explore::GPU_MAX_ZOOM`, so
    /// nothing is lost), so every result is independently reloadable later,
    /// not just a screenshot. Postprocessed with the trained novelty model
    /// (`novelty::NoveltyScorer` — frozen DINOv2 + a VICReg-trained
    /// projection head, scored against the REAL archive at
    /// `config.output.save_dir` when its `.novelty_cache.npz` exists, not
    /// just this batch) to rank results by how visually unusual they
    /// are — the search's own score/diversity already picked a good,
    /// varied set; novelty answers a different question ("which of these
    /// looks least like anything already in the archive"), which is what
    /// "find the most unusual" needs.
    ///
    /// Seed-picking searches WIDE (`explore::EXPLORE_WIDE_RADIUS`/
    /// `WIDE_SCALES`, not `(1.0, explore::SCALES)`) — a real run confined to
    /// the current view alone got stuck: the whole visible frame was smooth
    /// rainbow banding with no fractal structure anywhere within it, so
    /// every one of 8 results came back equally bad no matter how many
    /// seeds/rounds ran, because there was nothing better IN REACH to find.
    /// A wide sweep can back out and land somewhere else entirely if the
    /// current view itself is a bad neighborhood; `drill`'s own per-seed
    /// refinement afterward is unaffected — still narrow, still `SCALES`.
    fn start_explore(&mut self) {
        if self.explore_busy { return; }
        self.explore_busy = true;
        self.explore_message.clear();

        let (tx, rx) = mpsc::channel::<ExploreMsg>();
        let genome = self.genome.clone();
        let config = self.config.clone();
        let view = self.view.clone();
        let out_dir = self.save_out_dir().join("explored").join(
            self.nn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("fractal"),
        );

        thread::spawn(move || {
            if let Err(e) = std::fs::create_dir_all(&out_dir) {
                let _ = tx.send(ExploreMsg::Failed(format!("cannot create {}: {e}", out_dir.display())));
                return;
            }
            let mut log = match explore::Logger::new(&out_dir.join("explore_log.jsonl")) {
                Ok(l) => l,
                Err(e) => { let _ = tx.send(ExploreMsg::Failed(format!("cannot open log: {e}"))); return; }
            };
            let shots = explore::collect_shots_mixed(
                &genome, &config, &view, &ScoreMethod::ALL, Self::EXPLORE_SEEDS, Self::EXPLORE_ROUNDS, &mut log,
                explore::EXPLORE_WIDE_RADIUS, explore::WIDE_SCALES,
            );
            if shots.is_empty() {
                let _ = tx.send(ExploreMsg::Failed(
                    "No genuinely diverse, good-quality spot found from the current view — try a different location or zoom level.".to_string(),
                ));
                return;
            }
            // Same quality floor `select_diverse` applies internally —
            // against each SEED's own best round (not every round
            // individually: a seed that ever reached a good round earned
            // its place, even if `drill` wandered off by its last round).
            let global_best = shots.iter().flat_map(|(_, h)| h.iter().map(|r| r.score)).fold(f32::MIN, f32::max);
            let min_score = global_best * explore::MIN_SHOT_SCORE_FRACTION;
            let eligible: Vec<&(usize, Vec<explore::RoundResult>)> = shots.iter()
                .filter(|(_, h)| h.iter().any(|r| r.score >= min_score))
                .collect();

            let mut novelty = NoveltyScorer::new(Path::new(&config.output.save_dir));
            let mut aesthetic = AestheticScorer::new();
            let id_base = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);

            // Render + aesthetic-score EVERY ROUND of every eligible seed —
            // not just each seed's best-BY-RAW-SCORE round — and keep only
            // the best-BY-AESTHETIC one per seed. Confirmed necessary, not
            // just the final ranking step: `drill`'s own round-to-round
            // descent already picks its next direction by raw score, so a
            // seed's descent can pass NEAR genuinely rich structure and
            // still end up somewhere more boring by its last round if raw
            // score doesn't track it — same root cause as the final-
            // ranking fix below, just one level deeper. `select_diverse`'s
            // pixel fingerprint fallback (novelty sidecar unavailable)
            // still falls back to plain best-by-score, since it has no
            // access to a render-based signal either.
            struct Rendered { seed_id: usize, view: View, score: f32, aesthetic: Option<f32>, png_path: PathBuf, nn_path: PathBuf }
            let mut rendered: Vec<Rendered> = Vec::with_capacity(eligible.len());
            for (seed_id, history) in &eligible {
                let mut seed_best: Option<Rendered> = None;
                for (round_idx, r) in history.iter().enumerate() {
                    let stem = format!("explore_{id_base:x}_seed{seed_id}_r{round_idx:02}");
                    let png_path = out_dir.join(format!("{stem}.png"));
                    let mut g = genome.clone();
                    g.view_cx = r.view.cx as f32;
                    g.view_cy = r.view.cy as f32;
                    g.view_zoom = r.view.zoom as f32;
                    // Render from the SAME f32-precision view the .nn will
                    // store — not the full f64 `r.view` — so the PNG and
                    // its eventual companion .nn can never disagree.
                    // Confirmed necessary the hard way: near a chaotic
                    // escape-time boundary at real max_iter, even a single
                    // f32 ULP position difference (exactly what `as f32`
                    // truncation introduces) can flip the WHOLE render to
                    // something unrecognizable — 3 of 8 real test shots
                    // diverged this way (one by 953 iteration-units)
                    // before this fix, in the equivalent PNG/.nn pairing
                    // this project does elsewhere.
                    let snapped_view = View::new_square(g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64);
                    explore::save_shot(&genome, &config, &snapped_view, 960, &png_path);
                    let aesthetic_ensemble = aesthetic.as_mut().and_then(|a| a.score_blocking(png_path.clone())).map(|s| s.ensemble());
                    let is_better = seed_best.as_ref().is_none_or(|b| aesthetic_ensemble.unwrap_or(f32::MIN) > b.aesthetic.unwrap_or(f32::MIN));
                    if is_better {
                        if let Some(prev) = &seed_best { let _ = std::fs::remove_file(&prev.png_path); }
                        seed_best = Some(Rendered {
                            seed_id: *seed_id, view: snapped_view, score: r.score,
                            aesthetic: aesthetic_ensemble, png_path, nn_path: out_dir.join(format!("{stem}.nn")),
                        });
                    } else {
                        let _ = std::fs::remove_file(&png_path);
                    }
                }
                if let Some(best) = seed_best {
                    let mut g = genome.clone();
                    g.view_cx = best.view.cx as f32;
                    g.view_cy = best.view.cy as f32;
                    g.view_zoom = best.view.zoom as f32;
                    if let Err(e) = save_genome(&g, &best.nn_path) {
                        let _ = tx.send(ExploreMsg::Failed(format!("save {}: {e}", best.nn_path.display())));
                        return;
                    }
                    rendered.push(best);
                }
            }
            // Re-rank by the trained aesthetic ensemble, not the cheap
            // search score, before diversity selection — `sweep`/`drill`'s
            // entropy/edge_density/intricacy are computed at SWEEP_RES=64
            // and confirmed unreliable at exactly the case that matters
            // most: a real genome had a strikingly rich region (nested
            // rings + fine marbled texture) that measured WORSE on
            // edge_density than plain smooth color banding nearby (0.68 vs
            // 0.82, and the gap widened at higher render resolution too —
            // not a resolution problem, edge_density structurally favors a
            // few bold sharp transitions over many fine ones). The trained
            // aesthetic ensemble got this one right (5.68 vs 5.48) where
            // every cheap structural metric didn't, so it — not raw search
            // score — decides which candidate `select_diverse_latent`
            // treats as "best" (its index-0 seed, and the anchor the rest
            // of the greedy diversity search branches out from).
            rendered.sort_by(|a, b| b.aesthetic.unwrap_or(f32::MIN).partial_cmp(&a.aesthetic.unwrap_or(f32::MIN)).unwrap_or(std::cmp::Ordering::Equal));

            // Diversity selection: latent distance (via the trained novelty
            // model) instead of `select_diverse`'s pixel fingerprint — see
            // `explore::select_diverse_latent`'s doc comment. A pixel
            // fingerprint sees "moved to a different position" whenever
            // ANY pixel differs, which a smooth color-banded genome can
            // trivially satisfy (a band at a different angle IS pixel-
            // different) while still reading as "the same boring thing" to
            // a human — confirmed in practice: a real run's pixel-
            // fingerprint selection kept 38 "diverse" shots of which only 2
            // were genuinely different compositions. Falls back to the
            // pixel-fingerprint method if the novelty sidecar isn't
            // available — degrades, doesn't hard-fail (same contract as
            // every other novelty-optional path in this project).
            let keep: Vec<usize> = match novelty.as_mut() {
                Some(n) => {
                    let embeds: Vec<Option<(f32, Vec<f32>)>> =
                        rendered.iter().map(|r| n.embed_blocking(r.png_path.clone())).collect();
                    let idx: Vec<usize> = (0..rendered.len()).filter(|&i| embeds[i].is_some()).collect();
                    let vecs: Vec<Vec<f32>> = idx.iter().map(|&i| embeds[i].as_ref().unwrap().1.clone()).collect();
                    let picked = explore::select_diverse_latent(&vecs, explore::MIN_DIVERSITY_DISTANCE_LATENT, vecs.len());
                    picked.into_iter().map(|p| idx[p]).collect()
                }
                None => {
                    let fp_shots: Vec<(usize, explore::RoundResult, Vec<Vec<f32>>)> = rendered.iter().map(|r| {
                        let round = explore::RoundResult { view: r.view.clone(), metrics: explore::Metrics::default(), score: r.score, round: 0 };
                        (r.seed_id, round, explore::fingerprint(&genome, &config, &r.view))
                    }).collect();
                    explore::select_diverse(&fp_shots, fp_shots.len())
                }
            };

            let mut results = Vec::with_capacity(keep.len());
            for &i in &keep {
                let r = &rendered[i];
                let novelty_score = novelty.as_mut().and_then(|n| n.score_blocking(r.png_path.clone()));
                log.log(&serde_json::json!({
                    "event": "result", "seed": r.seed_id,
                    "cx": r.view.cx, "cy": r.view.cy, "zoom": r.view.zoom, "score": r.score,
                    "aesthetic_ensemble": r.aesthetic,
                    "novelty": novelty_score, "png": r.png_path.display().to_string(), "nn": r.nn_path.display().to_string(),
                }));
                results.push(ExploreResult {
                    png_path: r.png_path.clone(), cx: r.view.cx, cy: r.view.cy, zoom: r.view.zoom,
                    score: r.score, novelty: novelty_score,
                });
            }
            // Not selected — delete rather than leave orphaned speculative
            // renders behind (same cleanup `cmd_pool` does for a candidate
            // that fails a later gate).
            for (i, r) in rendered.iter().enumerate() {
                if !keep.contains(&i) {
                    let _ = std::fs::remove_file(&r.png_path);
                    let _ = std::fs::remove_file(&r.nn_path);
                }
            }

            if results.is_empty() {
                let _ = tx.send(ExploreMsg::Failed(
                    "No genuinely diverse, good-quality spot found from the current view — try a different location or zoom level.".to_string(),
                ));
                return;
            }

            // Most unusual first — see this method's doc comment on why
            // novelty (vs. the archive) is a different axis than the
            // search's own quality/diversity score.
            results.sort_by(|a, b| b.novelty.unwrap_or(f32::MIN).partial_cmp(&a.novelty.unwrap_or(f32::MIN)).unwrap_or(std::cmp::Ordering::Equal));
            let _ = tx.send(ExploreMsg::Done { results, out_dir });
        });
        self.explore_rx = Some(rx);
    }

    /// Spacebar quick-save: default resolution (last used) into the default folder,
    /// no dialog. Persists nothing new — just uses the remembered defaults.
    fn quick_save(&mut self) {
        let sw = self.prefs.last_save_width.max(64);
        let sh = self.prefs.last_save_height.max(64);
        let out_dir = self.save_out_dir();
        self.prefs.save_dir = out_dir.to_string_lossy().into_owned();
        self.prefs.save(&self.prefs_path);
        self.spawn_save(sw, sh, out_dir);
    }

    fn show_save_window(&mut self, ctx: &egui::Context) {
        if !self.show_save { return; }

        let mut do_save = false;
        let mut do_close = false;
        egui::Window::new("Save Fractal Image")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Width:");
                    ui.add(egui::TextEdit::singleline(&mut self.save_w_str).desired_width(80.0));
                    ui.label("Height:");
                    ui.add(egui::TextEdit::singleline(&mut self.save_h_str).desired_width(80.0));
                });
                let sw: u32 = self.save_w_str.trim().parse().unwrap_or(1920);
                let sh: u32 = self.save_h_str.trim().parse().unwrap_or(1080);
                if sw > 0 && sh > 0 {
                    let r = sw as f64 / sh as f64;
                    ui.label(
                        egui::RichText::new(format!("→ ratio {sw}:{sh} = {r:.3}"))
                            .color(Color32::GRAY),
                    );
                }
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    ui.add(egui::TextEdit::singleline(&mut self.save_dir_str)
                        .desired_width(280.0)
                        .hint_text("output folder for saved PNGs"));
                });
                ui.label(
                    egui::RichText::new("Remembered as the default for next time.")
                        .color(Color32::GRAY),
                );
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() { do_save = true; }
                    if ui.button("Cancel").clicked() { do_close = true; }
                });
            });

        if do_save {
            let sw: u32 = self.save_w_str.trim().parse().unwrap_or(1920);
            let sh: u32 = self.save_h_str.trim().parse().unwrap_or(1080);
            if sw >= 64 && sh >= 64 {
                let out_dir = self.save_out_dir();
                self.prefs.last_save_width  = sw;
                self.prefs.last_save_height = sh;
                self.prefs.save_dir = out_dir.to_string_lossy().into_owned();
                self.prefs.save(&self.prefs_path);
                self.spawn_save(sw, sh, out_dir);
            }
            self.show_save = false;
        }
        if do_close {
            self.show_save = false;
        }
    }

    /// Repo root — thin wrapper kept so existing `Self::project_root()`
    /// call sites in this file didn't all need touching; the actual logic
    /// now lives once, shared with `queue.rs`/`video_export.rs`, in
    /// `nnfractals::project_root()` (this file's own copy was the SAME
    /// duplicated-path-resolution mistake this project has already been
    /// bitten by once — see that function's doc comment).
    fn project_root() -> PathBuf {
        nnfractals::project_root()
    }

    /// Runs `program args...` in the background, streaming every stdout/
    /// stderr line into `eo_log` live — same shape as `launcher.rs::
    /// spawn_tracked`, generalized here to cover both the `explorer`
    /// binary and `scripts/*.py` (this project's two kinds of exploration
    /// subprocess) so a long `vae-explore` run's `iteration N: ...`
    /// progress is visible without leaving the viewer.
    fn spawn_explore_stage(&mut self, stage: &str, program: PathBuf, args: Vec<String>, cwd: PathBuf) {
        if self.eo_busy {
            self.eo_message = "a pipeline stage is already running".to_string();
            return;
        }
        let mut cmd = Command::new(&program);
        cmd.args(&args).current_dir(&cwd).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                self.eo_message = format!("could not start {stage}: {e} (looked for {})", program.display());
                return;
            }
        };
        self.eo_child_pid = Some(child.id());
        self.eo_cancelling = false;
        let (tx, rx) = mpsc::channel();
        let ctx = self.egui_ctx.clone();
        if let Some(out) = child.stdout.take() {
            let tx = tx.clone();
            let ctx = ctx.clone();
            thread::spawn(move || {
                for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                    let _ = tx.send(ExploreOpsMsg::Line(line));
                    ctx.request_repaint();
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            let tx = tx.clone();
            let ctx = ctx.clone();
            thread::spawn(move || {
                for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                    let _ = tx.send(ExploreOpsMsg::Line(line));
                    ctx.request_repaint();
                }
            });
        }
        thread::spawn(move || {
            // Fallback terminal signal, same race caveat launcher.rs's own
            // waiter thread documents: the process exiting doesn't strictly
            // guarantee every already-buffered stdout/stderr line has been
            // drained by the reader threads yet, so Done/Failed can very
            // occasionally show up a frame before the last log line does.
            // Not worth synchronizing further for a status readout.
            let msg = match child.wait() {
                Ok(s) if s.success() => ExploreOpsMsg::Done,
                Ok(s) => ExploreOpsMsg::Failed(format!("exited with {s}")),
                Err(e) => ExploreOpsMsg::Failed(format!("wait failed: {e}")),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
        self.eo_busy = true;
        self.eo_stage = stage.to_string();
        self.eo_log.clear();
        self.eo_message.clear();
        self.eo_rx = Some(rx);
        self.eo_preview_texture = None;
        self.eo_preview_stem.clear();
        self.eo_preview_view = None;
        self.eo_scan_candidates.clear();
        self.eo_scan_seed_view = None;
        self.eo_vz_progress = None;
        self.eo_saliency_texture = None;
        self.eo_saliency_mtime = None;
    }

    /// Kills the currently-running stage's top-level process (same plain
    /// `kill <pid>` this project's launcher.rs stop button already uses).
    /// Doesn't flip `eo_busy` itself — the waiter thread in
    /// `spawn_explore_stage` will observe the process exit and send
    /// `Failed`, which the normal poll loop picks up (read as "cancelled"
    /// via `eo_cancelling`, set here first).
    fn cancel_explore_stage(&mut self) {
        let Some(pid) = self.eo_child_pid else { return };
        self.eo_cancelling = true;
        self.eo_message = format!("cancelling {}…", self.eo_stage);
        let _ = Command::new("kill").arg(pid.to_string()).status();
    }

    /// Applies one of `EO_PRESETS` to the growth fields wholesale.
    fn apply_eo_preset(&mut self, idx: usize) {
        let Some(&(_, _, iterations, n_seeds, recursion_depth, top_k, patience, min_improvement)) = EO_PRESETS.get(idx) else { return };
        self.eo_iterations = iterations.to_string();
        self.eo_n_seeds = n_seeds.to_string();
        self.eo_recursion_depth = recursion_depth.to_string();
        self.eo_top_k = top_k.to_string();
        self.eo_patience = patience.to_string();
        self.eo_min_improvement = min_improvement.to_string();
    }

    /// Live "what is actually happening" preview: while a stage runs,
    /// periodically (throttled — directory scans + PNG decode aren't
    /// free) checks `eo_out_dir_str` for the most-recently-modified
    /// saved zone PNG and loads it as a thumbnail texture, plus counts
    /// `.nn` files for a live zone count independent of log-text parsing.
    fn update_explore_preview(&mut self, ctx: &egui::Context) {
        if !self.eo_busy { return; }
        let now = std::time::Instant::now();
        if let Some(last) = self.eo_preview_last_scan
            && now.duration_since(last).as_millis() < 800 {
            return;
        }
        self.eo_preview_last_scan = Some(now);

        let dir = PathBuf::from(self.eo_out_dir_str.trim());
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
        let mut count = 0usize;
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            if !stem.starts_with("zone_") { continue; }
            match path.extension().and_then(|s| s.to_str()) {
                Some("nn") => count += 1,
                Some("png") if !stem.ends_with("_raw") => {
                    if let Ok(meta) = entry.metadata()
                        && let Ok(modified) = meta.modified()
                        && newest.as_ref().is_none_or(|(t, _)| modified > *t) {
                        newest = Some((modified, path));
                    }
                }
                _ => {}
            }
        }
        self.eo_zone_count = count;
        if let Some((_, path)) = newest {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if stem != self.eo_preview_stem {
                if let Ok(img) = image::open(&path) {
                    let rgb = img.to_rgb8();
                    let size = [rgb.width() as usize, rgb.height() as usize];
                    let color_image = ColorImage::from_rgb(size, rgb.as_raw());
                    self.eo_preview_texture = Some(ctx.load_texture(format!("eo_preview_{stem}"), color_image, TextureOptions::LINEAR));
                }
                self.eo_preview_view = load_genome(&path.with_extension("nn"))
                    .ok()
                    .map(|g| (g.view_cx as f64, g.view_cy as f64, g.view_zoom.max(0.1) as f64));
                self.eo_preview_stem = stem;
            }
        }

        // This level's full candidate set — see `eo_scan_candidates`'s doc
        // comment. Only `explorer vae-explore` (stage 1) writes
        // "level_scanning" lines, so this is a no-op during the other 3
        // stages. Re-scans the whole log each tick rather than tracking a
        // byte offset — simpler, and a JSONL log for one growth run is at
        // most a few MB, cheap to re-read every 800ms.
        if let Ok(text) = std::fs::read_to_string(dir.join("vae_explore_log.jsonl"))
            && let Some(line) = text.lines().rev().find(|l| l.contains("\"level_scanning\""))
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(arr) = v.get("candidates").and_then(|c| c.as_array()) {
                self.eo_scan_candidates = arr.iter().take(EO_SCAN_OVERLAY_MAX).filter_map(|c| {
                    Some((
                        c.get("cx")?.as_f64()?,
                        c.get("cy")?.as_f64()?,
                        c.get("zoom")?.as_f64()?,
                        c.get("gate_pass")?.as_bool()?,
                    ))
                }).collect();
            }
            let seed = (|| Some((
                v.get("seed_cx")?.as_f64()?,
                v.get("seed_cy")?.as_f64()?,
                v.get("seed_zoom")?.as_f64()?,
            )))();
            // Auto-navigate the main canvas to wherever the scan currently
            // is (Carl: log showing real progress "but nothing on the
            // screen" — without this, the overlay squares almost never
            // land anywhere near whatever the canvas happened to already
            // be showing). Only re-navigates/re-renders when the seed
            // actually changed, not every 800ms tick.
            if self.eo_follow_scan
                && let Some((sx, sy, szoom)) = seed
                && self.eo_scan_seed_view != Some((sx, sy, szoom)) {
                self.view.cx = sx; self.view.cx_lo = 0.0;
                self.view.cy = sy; self.view.cy_lo = 0.0;
                self.view.zoom = szoom.max(0.1);
                self.sync_xy = true;
                self.request_render(true);
            }
            self.eo_scan_seed_view = seed;
        }

        // video-zoom-explore's own live-progress signal — see
        // `video_zoom_explore::zoom_level`'s doc comment on `is_real_step`/
        // `"committed_move"`. That module never saves incremental zone_*
        // files (only a final winners manifest at the very end), so
        // without this, the zone-scan above always reports 0 zones / no
        // preview during such a run, and the ONLY visible feedback would
        // be the raw scan-square cloud above — including every rejected
        // lookahead probe, with nothing distinguishing real progress from
        // exploratory noise (this is what made a correctly-working search
        // look "stuck" — Carl's real report, 2026-08-13). Reuses
        // `eo_preview_view` (already drawn as a white+red square on the
        // main canvas) rather than adding a new overlay color — same
        // "here's the confirmed current position" meaning vae-explore's
        // own zone marker has, just fed by a different event. A separate
        // read (not merged with the level_scanning parse above) — simplest
        // given the two events are independent and this file is "at most a
        // few MB," per that block's own doc comment.
        if let Ok(text) = std::fs::read_to_string(dir.join("vae_explore_log.jsonl"))
            && let Some(line) = text.lines().rev().find(|l| l.contains("\"committed_move\""))
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            self.eo_preview_view = (|| Some((
                v.get("cx")?.as_f64()?,
                v.get("cy")?.as_f64()?,
                v.get("zoom")?.as_f64()?,
            )))();
        }

        // "N of BUDGET real moves committed" (Carl's request, 2026-08-13:
        // "give me an idea of the progress of the search") — find the most
        // recent `"seed_started"` line (logs `seed_id`/`budget` once per
        // seed, see `video_zoom_explore::explore_from_seed`) and count
        // `"committed_move"` lines AFTER it, so a multi-seed run
        // (`--n-seeds` > 1) resets the count per seed instead of counting
        // across all of them cumulatively.
        if let Ok(text) = std::fs::read_to_string(dir.join("vae_explore_log.jsonl")) {
            let lines: Vec<&str> = text.lines().collect();
            if let Some(start_idx) = lines.iter().rposition(|l| l.contains("\"seed_started\""))
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(lines[start_idx]) {
                let parsed = (|| Some((
                    v.get("seed_id")?.as_u64()? as usize,
                    v.get("budget")?.as_u64()? as usize,
                )))();
                if let Some((seed_id, budget)) = parsed {
                    let committed = lines[start_idx + 1..]
                        .iter()
                        .filter(|l| l.contains("\"committed_move\""))
                        .count();
                    self.eo_vz_progress = Some((seed_id, committed, budget));
                }
            }
        }

        // Saliency heatmap, if the run was started with a --saliency-model
        // (see start_vae_growth) — overwritten in place each level, so
        // reload only when its mtime actually changes.
        let heatmap_path = dir.join("saliency_heatmap_latest.png");
        if let Ok(meta) = std::fs::metadata(&heatmap_path)
            && let Ok(modified) = meta.modified()
            && self.eo_saliency_mtime != Some(modified)
            && let Ok(img) = image::open(&heatmap_path) {
            let rgb = img.to_rgb8();
            let size = [rgb.width() as usize, rgb.height() as usize];
            let color_image = ColorImage::from_rgb(size, rgb.as_raw());
            self.eo_saliency_texture = Some(ctx.load_texture("eo_saliency", color_image, TextureOptions::LINEAR));
            self.eo_saliency_mtime = Some(modified);
        }
    }

    /// Stage 1: `explorer vae-explore` from the CURRENT view — grows
    /// `eo_out_dir_str` (append-mode, safe to re-run) via the recursive
    /// VAE-driven drill. The current genome is written to `_seed_genome.nn`
    /// inside the out_dir first (a leading underscore, not `zone_`-prefixed,
    /// so it's invisible to vae-explore's own append-mode resume scan) since
    /// the viewer's loaded genome is usually a specific saved/GA-discovered
    /// one, not a named `known_formulas::LIBRARY` entry.
    fn start_vae_growth(&mut self) {
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        if out_dir.as_os_str().is_empty() {
            self.eo_message = "set an output directory first".to_string();
            return;
        }
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            self.eo_message = format!("cannot create {}: {e}", out_dir.display());
            return;
        }
        let seed_path = out_dir.join("_seed_genome.nn");
        if let Err(e) = save_genome(&self.genome, &seed_path) {
            self.eo_message = format!("cannot save seed genome: {e}");
            return;
        }
        let mut args = vec![
            "vae-explore".to_string(),
            seed_path.to_string_lossy().into_owned(),
            self.view.cx.to_string(), self.view.cy.to_string(), self.view.zoom.to_string(),
            out_dir.to_string_lossy().into_owned(),
            "--iterations".to_string(), self.eo_iterations.trim().to_string(),
            "--n-seeds".to_string(), self.eo_n_seeds.trim().to_string(),
            "--recursion-depth".to_string(), self.eo_recursion_depth.trim().to_string(),
            "--top-k".to_string(), self.eo_top_k.trim().to_string(),
            "--canvas-res".to_string(), self.eo_canvas_res.trim().to_string(),
            "--method".to_string(), EO_METHODS[self.eo_method_idx].to_string(),
            "--select-by".to_string(), EO_SELECT_BY[self.eo_select_by_idx].to_string(),
            "--patience".to_string(), self.eo_patience.trim().to_string(),
            "--min-improvement".to_string(), self.eo_min_improvement.trim().to_string(),
        ];
        if !self.eo_saliency_model_path.trim().is_empty() {
            args.push("--saliency-model".to_string());
            args.push(self.eo_saliency_model_path.trim().to_string());
        }
        let root = Self::project_root();
        self.spawn_explore_stage("Growing corpus", locate_sibling_bin("explorer"), args, root);
    }

    /// Runs `explorer video-zoom-explore` in the background from the
    /// CURRENT view, same subprocess/progress plumbing as `start_vae_growth`
    /// (`spawn_explore_stage`, shared `eo_out_dir_str`/`eo_busy`/scan
    /// overlay — this stage writes the same `vae_explore_log.jsonl`
    /// `"level_scanning"` events, so the existing overlay just works here
    /// too). `--final-width` reads `video_w_str` directly (the "Add to
    /// Queue" section's own field) rather than a new one of its own — the
    /// DD-boundary gate only means anything relative to the video that
    /// will actually get exported later.
    fn start_video_zoom_explore(&mut self) {
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        if out_dir.as_os_str().is_empty() {
            self.eo_message = "set an output directory first".to_string();
            return;
        }
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            self.eo_message = format!("cannot create {}: {e}", out_dir.display());
            return;
        }
        let seed_path = out_dir.join("_seed_genome.nn");
        if let Err(e) = save_genome(&self.genome, &seed_path) {
            self.eo_message = format!("cannot save seed genome: {e}");
            return;
        }
        let final_width = self.video_w_str.trim().parse::<u32>().unwrap_or(1280).max(64);
        let args = vec![
            "video-zoom-explore".to_string(),
            seed_path.to_string_lossy().into_owned(),
            self.view.cx.to_string(), self.view.cy.to_string(), self.view.zoom.to_string(),
            out_dir.to_string_lossy().into_owned(),
            "--depth".to_string(), self.eo_vz_depth.trim().to_string(),
            "--finalists".to_string(), self.eo_vz_finalists.trim().to_string(),
            "--lookahead-plies".to_string(), self.eo_vz_lookahead.trim().to_string(),
            "--method".to_string(), EO_METHODS[self.eo_vz_method_idx].to_string(),
            "--final-width".to_string(), final_width.to_string(),
            "--top-winners".to_string(), self.eo_vz_top_winners.trim().to_string(),
            "--n-seeds".to_string(), self.eo_vz_n_seeds.trim().to_string(),
            "--min-score".to_string(), self.eo_vz_min_score.trim().to_string(),
        ];
        self.eo_vz_winners.clear(); // don't show a stale gallery from a previous run while this one is in flight
        let root = Self::project_root();
        self.spawn_explore_stage("Video-zoom exploring", locate_sibling_bin("explorer"), args, root);
    }

    /// Reads `video_zoom_winners.jsonl` (written by
    /// `video_zoom_explore::write_winners_manifest`) into `eo_vz_winners`
    /// for the gallery — called once from `ExploreOpsMsg::Done` when
    /// `eo_stage` shows the video-zoom stage just finished, same
    /// best-effort/tolerant style as `update_explore_preview` (a missing or
    /// unparseable manifest just leaves the gallery empty, no error shown).
    fn load_video_zoom_winners(&mut self, ctx: &egui::Context) {
        self.eo_vz_winners.clear();
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        let Ok(text) = std::fs::read_to_string(out_dir.join("video_zoom_winners.jsonl")) else { return };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let Some(rank) = v.get("rank").and_then(|x| x.as_u64()) else { continue };
            let rank = rank as usize;
            let n_legs = v.get("n_legs").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let final_probe_ratio = v.get("final_probe_ratio").and_then(|x| x.as_f64());
            let ended_reason = v.get("ended_reason").and_then(|x| x.as_str()).unwrap_or("?").to_string();
            let preview_mp4 = out_dir.join(v.get("preview_mp4").and_then(|x| x.as_str()).unwrap_or(""));
            let thumb_png = out_dir.join(v.get("thumb_png").and_then(|x| x.as_str()).unwrap_or(""));
            let chain: Vec<CapturedView> = v.get("chain").and_then(|c| c.as_array()).map(|arr| {
                arr.iter().filter_map(|c| Some(CapturedView {
                    cx: c.get("cx")?.as_f64()?, cx_lo: c.get("cx_lo")?.as_f64()?,
                    cy: c.get("cy")?.as_f64()?, cy_lo: c.get("cy_lo")?.as_f64()?,
                    zoom: c.get("zoom")?.as_f64()?, aspect: c.get("aspect")?.as_f64()?,
                })).collect()
            }).unwrap_or_default();
            if chain.len() < 2 { continue; }

            let thumb = image::open(&thumb_png).ok().map(|img| {
                let rgb = img.to_rgb8();
                let size = [rgb.width() as usize, rgb.height() as usize];
                let color_image = ColorImage::from_rgb(size, rgb.as_raw());
                ctx.load_texture(format!("eo_vz_winner_{rank}"), color_image, TextureOptions::LINEAR)
            });

            self.eo_vz_winners.push(VideoZoomWinnerUi { rank, n_legs, final_probe_ratio, ended_reason, chain, thumb, preview_mp4 });
        }
    }

    /// Queues one gallery winner's full discovered chain for full-quality
    /// export — same copy-genome-into-queue-dir, `QueueItem`, `save_queue`,
    /// and `wake_or_launch_queue_window` sequence as `add_to_queue`, just
    /// sourcing `start`/`end`/`waypoints` from the winner's already-known
    /// chain instead of the live view's manually-set start/end. Reuses the
    /// same steps/fps/width/height fields as "Add to Queue" (no separate
    /// quality settings for this path) — one canonical set of export
    /// quality knobs regardless of how a queue entry was produced.
    fn queue_video_zoom_winner(&mut self, idx: usize) {
        let keyframe_stride = self.prefs.video_keyframe_stride.max(1);
        let Some(winner) = self.eo_vz_winners.get(idx) else { return };
        let chain = winner.chain.clone();
        let (Some(start), Some(end)) = (chain.first().copied(), chain.last().copied()) else { return };

        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{:x}", d.as_nanos()))
            .unwrap_or_else(|_| format!("{:016x}", self.genome.id));
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let genome_label = self.nn_path.file_stem().and_then(|s| s.to_str())
            .unwrap_or("fractal").to_string();

        let qdir = nnfractals::video_export::queue_dir();
        if let Err(e) = std::fs::create_dir_all(&qdir) {
            self.eo_message = format!("Queue winner FAILED: {e}");
            return;
        }
        let nn_filename = format!("{id}.nn");
        if let Err(e) = std::fs::copy(&self.nn_path, qdir.join(&nn_filename)) {
            self.eo_message = format!("Queue winner FAILED: {e}");
            return;
        }

        let w = self.video_w_str.trim().parse::<u32>().unwrap_or(1280).max(64);
        let h = self.video_h_str.trim().parse::<u32>().unwrap_or(720).max(64);
        let steps = self.video_steps_str.trim().parse::<u32>().unwrap_or(60).max(2);
        let fps = self.video_fps_str.trim().parse::<u32>().unwrap_or(30).max(1);

        let item = nnfractals::video_export::QueueItem {
            id, nn_filename, genome_label, start, end, steps, fps, width: w, height: h,
            invert_coords: self.prefs.video_invert_coords,
            invert_range: self.prefs.video_invert_range,
            colormap: self.config.rendering.colormap.clone(),
            angle_coloring: self.angle_coloring,
            output_dir: self.save_out_dir().to_string_lossy().into_owned(),
            status: nnfractals::video_export::QueueStatus::Pending,
            output_path: None,
            error: None,
            created_at,
            waypoints: chain,
            chain_label: Some("zoom-explore chain".to_string()),
            keyframe_stride,
        };
        let mut items = nnfractals::video_export::load_queue();
        items.push(item);
        nnfractals::video_export::save_queue(&items);

        wake_or_launch_queue_window();
        self.eo_message = format!("Winner #{idx} queued ✓");
    }

    /// Runs `explorer retrain-saliency` in the background — auto-discovers
    /// every `explorer_out/*_vae` pool plus `saliency_manual_marks/`
    /// (see `cmd_retrain_saliency` in `explorer.rs`), regenerates the
    /// dataset, retrains, and overwrites `SALIENCY_DEFAULT_MODEL_PATH`.
    /// No extra fields exposed here deliberately — every knob keeps its
    /// CLI default (out_dir, canvas-res, max-per-pool, epochs,
    /// `explorer_out/last_successful_vae.pt` for scoring marks) rather
    /// than growing the window further; re-run `retrain-saliency` from a
    /// terminal directly if a specific run needs different settings.
    fn start_retrain_saliency(&mut self) {
        let root = Self::project_root();
        self.spawn_explore_stage("Retraining saliency model", locate_sibling_bin("explorer"), vec!["retrain-saliency".to_string()], root);
    }

    /// Stage 2: `explorer complex-export` — re/im/mag/tensor channels for
    /// every zone in `eo_out_dir_str`, into `<out_dir>/complex_export/`
    /// (self-contained under the same out_dir rather than a separate field,
    /// matching this session's own `weekend_complex_corpus/<name>/`
    /// convention closely enough without adding another text box).
    fn start_complex_export(&mut self) {
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        if out_dir.as_os_str().is_empty() {
            self.eo_message = "set an output directory first".to_string();
            return;
        }
        let export_dir = out_dir.join("complex_export");
        let args = vec![
            "complex-export".to_string(),
            out_dir.to_string_lossy().into_owned(),
            export_dir.to_string_lossy().into_owned(),
            "--res".to_string(), self.eo_export_res.trim().to_string(),
            "--limit".to_string(), self.eo_export_limit.trim().to_string(),
        ];
        let root = Self::project_root();
        self.spawn_explore_stage("Complex-exporting", locate_sibling_bin("explorer"), args, root);
    }

    /// Stage 3: `scripts/select_diverse_latent.py` — farthest-point
    /// diversity selection over the complex VAE's latent space, into
    /// `<out_dir>/diverse_selection/`.
    fn start_diversity_selection(&mut self) {
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        if out_dir.as_os_str().is_empty() {
            self.eo_message = "set an output directory first".to_string();
            return;
        }
        let export_dir = out_dir.join("complex_export");
        let sel_out = out_dir.join("diverse_selection");
        let root = Self::project_root();
        let args = vec![
            "scripts/select_diverse_latent.py".to_string(),
            "--model-path".to_string(), self.eo_model_path.trim().to_string(),
            "--dirs".to_string(), export_dir.to_string_lossy().into_owned(),
            "--pool-dir".to_string(), out_dir.to_string_lossy().into_owned(),
            "--out-dir".to_string(), sel_out.to_string_lossy().into_owned(),
            "--top-n".to_string(), self.eo_top_n.trim().to_string(),
            "--res".to_string(), self.eo_export_res.trim().to_string(),
        ];
        self.spawn_explore_stage("Selecting diverse zones", nnfractals::python_bin(&root), args, root);
    }

    /// Stage 4: `scripts/cluster_latent.py` — HDBSCAN latent-space
    /// clustering (UMAP reduction + DBCV-validity-driven `min-cluster-size`
    /// sweep unless `eo_min_cluster_size` is set), full membership rendered
    /// per cluster, into `<out_dir>/clusters/`.
    fn start_clustering(&mut self) {
        let out_dir = PathBuf::from(self.eo_out_dir_str.trim());
        if out_dir.as_os_str().is_empty() {
            self.eo_message = "set an output directory first".to_string();
            return;
        }
        let export_dir = out_dir.join("complex_export");
        let cl_out = out_dir.join("clusters");
        let root = Self::project_root();
        let mut args = vec![
            "scripts/cluster_latent.py".to_string(),
            "--model-path".to_string(), self.eo_model_path.trim().to_string(),
            "--dirs".to_string(), export_dir.to_string_lossy().into_owned(),
            "--pool-dir".to_string(), out_dir.to_string_lossy().into_owned(),
            "--out-dir".to_string(), cl_out.to_string_lossy().into_owned(),
            "--res".to_string(), self.eo_export_res.trim().to_string(),
            "--reps-per-cluster".to_string(), self.eo_reps_per_cluster.trim().to_string(),
            "--noise-sample".to_string(), self.eo_noise_sample.trim().to_string(),
        ];
        let mcs = self.eo_min_cluster_size.trim();
        if !mcs.is_empty() {
            args.push("--min-cluster-size".to_string());
            args.push(mcs.to_string());
        }
        self.spawn_explore_stage("Clustering", nnfractals::python_bin(&root), args, root);
    }

    /// "Explore Options" window — opens on clicking "Explore" instead of
    /// immediately running the classic `explore_diverse_mixed` search
    /// (still reachable via the button inside, unchanged). Exposes the
    /// complex-VAE latent-space exploration pipeline built this session
    /// (`explorer vae-explore`/`complex-export` + `scripts/
    /// select_diverse_latent.py`/`cluster_latent.py`) as four sequential
    /// stages against one shared out_dir — Carl's request, 2026-08-09.
    fn show_explore_options_window(&mut self, ctx: &egui::Context) {
        if !self.show_explore_options { return; }
        let mut do_close = false;
        egui::Window::new("Explore Options")
            .collapsible(false)
            .resizable(true)
            .default_width(480.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Output dir:").on_hover_text(
                        "Everything below lives here: saved zones (zone_NNNN.nn/.png), the \
                         growing VAE checkpoint, complex-export channels, and the diversity/\
                         cluster results. One folder per genome by default."
                    );
                    ui.add(egui::TextEdit::singleline(&mut self.eo_out_dir_str).desired_width(320.0))
                        .on_hover_text("Where this genome's exploration pipeline reads/writes everything.");
                });
                ui.label(egui::RichText::new(
                    "All four stages below share this folder — grow, then export, then \
                     diversify/cluster. Re-running Grow into the same folder APPENDS \
                     (safe to resume)."
                ).color(Color32::GRAY).small());
                ui.separator();

                ui.add_enabled_ui(!self.eo_busy, |ui| {
                    if ui.button("Quick Explore (classic, from current view)")
                        .on_hover_text(
                            "The original one-click search: explore::explore_diverse_mixed — \
                             several seeds × rounds, ranked by novelty, unrelated to the complex-\
                             VAE pipeline below. Fast, in-process, no output-dir dependency."
                        )
                        .clicked()
                    {
                        self.start_explore();
                    }
                });
                ui.separator();

                egui::CollapsingHeader::new("1. Grow corpus (explorer vae-explore)").default_open(true).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Preset:").on_hover_text(
                            "Applies a whole known-good parameter set at once, found through real \
                             trial-and-error this project's exploration runs — pick one instead of \
                             hand-tuning every field below from scratch."
                        );
                        egui::ComboBox::from_id_salt("eo_preset")
                            .selected_text(EO_PRESETS[self.eo_preset_idx].0)
                            .show_ui(ui, |ui| {
                                for (i, p) in EO_PRESETS.iter().enumerate() {
                                    if ui.selectable_value(&mut self.eo_preset_idx, i, p.0).clicked() {
                                        self.apply_eo_preset(i);
                                    }
                                }
                            });
                    });
                    ui.label(egui::RichText::new(EO_PRESETS[self.eo_preset_idx].1).color(Color32::GRAY).small());
                    egui::Grid::new("eo_grow_grid").num_columns(4).show(ui, |ui| {
                        ui.label("Iterations").on_hover_text(
                            "How many outer select→train→rescore rounds to run. Each one grows \
                             the corpus, retrains the real-valued escape-time VAE on it, then \
                             rescoring drives the NEXT round's zone selection. Stops early on \
                             patience or --target-recon-mse."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_iterations).desired_width(60.0))
                            .on_hover_text("Outer select→train→rescore rounds.");
                        ui.label("Seeds").on_hover_text(
                            "How many starting points explore::pick_seeds picks (wide-radius \
                             sweep around the search anchor) at the start of each iteration. \
                             More seeds ≠ always more zones — pick_seeds' own region-dedup caps \
                             how many genuinely distinct positions exist near one anchor."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_n_seeds).desired_width(60.0))
                            .on_hover_text("Starting points swept per iteration.");
                        ui.end_row();
                        ui.label("Recursion depth").on_hover_text(
                            "How many levels deep each seed drills — render canvas, coarse-scan \
                             for candidates, save the good ones, zoom into the winner, repeat. \
                             Deeper = more zones per seed, at the cost of slower canvas renders \
                             (deep zoom can fall to the CPU/DD tier)."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_recursion_depth).desired_width(60.0))
                            .on_hover_text("Levels drilled deeper per seed.");
                        ui.label("Top-K per level").on_hover_text(
                            "How many coarse-scan candidates get precisely rendered+saved (if \
                             they pass the gate and aren't near-duplicates) at EACH recursion \
                             level. Raising this extracts more zones from the SAME seeds — the \
                             fix for a seed that's genuinely rich but capped by a low top-K."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_top_k).desired_width(60.0))
                            .on_hover_text("Candidates saved per recursion level.");
                        ui.end_row();
                        ui.label("Canvas res").on_hover_text(
                            "Resolution of the wide canvas each seed/level re-renders before \
                             coarse-scanning it for candidates. Higher = finer candidate \
                             detection but much slower. Keep at or below 4095 — 4096 and up \
                             exceeds the GPU's single-dispatch pixel limit, which forces EVERY \
                             canvas render onto the ~20-40s CPU path regardless of zoom depth, \
                             not just genuinely deep ones."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_canvas_res).desired_width(60.0))
                            .on_hover_text("Pixel width/height of the per-level scan canvas.");
                        ui.label("Patience").on_hover_text(
                            "Outer iterations allowed with no reconstruction-error improvement \
                             (≥2% better than the best seen) before stopping early. Separate \
                             from the growth-stall recenter mechanism below — this tracks model \
                             quality, recentering tracks zone-count growth."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_patience).desired_width(60.0))
                            .on_hover_text("Iterations without recon-error improvement before early stop.");
                        ui.end_row();
                        ui.label("Method").on_hover_text(
                            "Coarse-scan scoring method. 'mixed' cycles entropy/edge/gated-\
                             entropy/gated-edge across iterations — different methods surface \
                             different candidates from the same canvas, which is why growth \
                             often comes in bursts tied to which method is active."
                        );
                        egui::ComboBox::from_id_salt("eo_method").selected_text(EO_METHODS[self.eo_method_idx])
                            .show_ui(ui, |ui| {
                                for (i, m) in EO_METHODS.iter().enumerate() {
                                    ui.selectable_value(&mut self.eo_method_idx, i, *m);
                                }
                            })
                            .response.on_hover_text("Which richness metric ranks coarse-scan candidates.");
                        ui.label("Select by").on_hover_text(
                            "How each level picks its ONE winner (the seed for the next, deeper \
                             level) among that level's saved zones: max-error = the VAE's most \
                             novel/undermodeled pick (default, active-learning style), min-error \
                             = the most prototypical, random = no VAE bias."
                        );
                        egui::ComboBox::from_id_salt("eo_select_by").selected_text(EO_SELECT_BY[self.eo_select_by_idx])
                            .show_ui(ui, |ui| {
                                for (i, m) in EO_SELECT_BY.iter().enumerate() {
                                    ui.selectable_value(&mut self.eo_select_by_idx, i, *m);
                                }
                            })
                            .response.on_hover_text("How the next recursion level's seed is chosen.");
                        ui.end_row();
                        ui.label("Min improvement").on_hover_text(
                            "Relative reconstruction-error improvement (e.g. 0.02 = 2%) needed \
                             over the best-so-far to reset the patience counter. Smaller = more \
                             forgiving of a slow-improving model."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_min_improvement).desired_width(60.0))
                            .on_hover_text("Relative recon-error improvement needed to reset patience.");
                        ui.end_row();
                        ui.label("Saliency model").on_hover_text(
                            "A checkpoint from scripts/train_saliency.py (a small fully-convolutional \
                             net trained to predict, from the wide scan canvas, WHERE a VAE would find \
                             high reconstruction error — 'saliency' meaning roughly 'how much this spot \
                             stands out/would be interesting to zoom into'). On by default as of \
                             2026-08-10 if a checkpoint exists at this path — its predicted heatmap \
                             ADDS extra candidate positions each level ON TOP of the proven coarse-scan \
                             grid (it can't remove or override anything the grid already finds). Clear \
                             this field to fall back to the grid alone."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_saliency_model_path).desired_width(300.0))
                            .on_hover_text("Path to a saliency_model.pt checkpoint, or blank to disable.");
                        ui.end_row();
                    });
                    ui.label(egui::RichText::new(
                        "If growth stalls (a zero-new-zone iteration), the search anchor \
                         automatically jumps to a random already-saved zone instead of re-\
                         scanning the same exhausted neighborhood forever — no field for this, \
                         it just happens."
                    ).color(Color32::GRAY).small());
                    ui.horizontal(|ui| {
                        ui.add_enabled_ui(!self.eo_busy, |ui| {
                            if ui.button("Start Growth")
                                .on_hover_text("Runs `explorer vae-explore` in the background from the current view.")
                                .clicked()
                            { self.start_vae_growth(); }
                            if ui.button("Retrain saliency model")
                                .on_hover_text(
                                    "Runs `explorer retrain-saliency` in the background: regenerates \
                                     the training set from EVERY explorer_out/*_vae pool plus any \
                                     saved manual marks (explorer_out/saliency_manual_marks/), then \
                                     retrains and overwrites the default checkpoint \
                                     (explorer_out/saliency_model.pt) that every subsequent Grow-\
                                     corpus run picks up automatically. Takes a few minutes — mostly \
                                     the dataset regeneration, training itself is quick."
                                )
                                .clicked()
                            { self.start_retrain_saliency(); }
                        });
                    });
                });

                egui::CollapsingHeader::new("2. Complex-export").default_open(false).show(ui, |ui| {
                    egui::Grid::new("eo_export_grid").num_columns(4).show(ui, |ui| {
                        ui.label("Res").on_hover_text(
                            "Pixel width/height of the exported real/imaginary/magnitude/escape-\
                             time channel PNGs — the complex VAE's actual training/inference \
                             resolution is fixed at 512, so this should normally stay 512."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_export_res).desired_width(60.0))
                            .on_hover_text("Exported channel PNG resolution (leave at 512).");
                        ui.label("Limit").on_hover_text(
                            "Maximum number of zones to export, sorted by filename. Set this at \
                             least as high as the pool's total zone count or later ones get \
                             silently skipped."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_export_limit).desired_width(60.0))
                            .on_hover_text("Max zones to export — should be ≥ the pool's total zone count.");
                        ui.end_row();
                    });
                    ui.add_enabled_ui(!self.eo_busy, |ui| {
                        if ui.button("Start Complex-Export")
                            .on_hover_text(
                                "Runs `explorer complex-export`: renders the bailout z-value \
                                 (re/im/mag) and escape-time field for every zone, needed by \
                                 both stages below."
                            )
                            .clicked()
                        { self.start_complex_export(); }
                    });
                });

                egui::CollapsingHeader::new("3. Diversity selection").default_open(false).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Model:").on_hover_text(
                            "Path to a trained complex VAE checkpoint (--variant vae) — needs a \
                             genuine posterior mean, so a plain (non-variational) AE checkpoint \
                             won't work here."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_model_path).desired_width(300.0))
                            .on_hover_text("Complex VAE checkpoint used to embed every zone's latent vector.");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Top N:").on_hover_text(
                            "How many zones to select via greedy farthest-point sampling in \
                             latent space: start from the zone closest to the corpus centroid, \
                             repeatedly add whichever remaining zone is farthest from its \
                             nearest already-picked neighbor."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_top_n).desired_width(60.0))
                            .on_hover_text("Zones to pick — maximally different from each other, not just top-ranked.");
                    });
                    ui.add_enabled_ui(!self.eo_busy, |ui| {
                        if ui.button("Select Diverse Zones")
                            .on_hover_text("Runs scripts/select_diverse_latent.py.")
                            .clicked()
                        { self.start_diversity_selection(); }
                    });
                });

                egui::CollapsingHeader::new("4. Clustering").default_open(false).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Model:").on_hover_text(
                            "Same complex VAE checkpoint as diversity selection — shared field, \
                             both stages embed zones the same way."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_model_path).desired_width(300.0))
                            .on_hover_text("Complex VAE checkpoint used to embed every zone's latent vector.");
                    });
                    egui::Grid::new("eo_cluster_grid").num_columns(4).show(ui, |ui| {
                        ui.label("Min cluster size").on_hover_text(
                            "HDBSCAN's core density parameter. Leave blank to auto-sweep a range \
                             scaled to pool size and keep whichever scores highest on \
                             relative_validity_ (a DBCV approximation) — set a fixed value only \
                             to force a specific granularity (e.g. more, smaller clusters)."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_min_cluster_size)
                            .desired_width(60.0).hint_text("auto"))
                            .on_hover_text("Blank = auto-pick by validity score. Smaller value = more, finer clusters.");
                        ui.label("Reps/cluster").on_hover_text(
                            "How many representative zones to render per cluster, closest-to-\
                             centroid first. Set ≥ the largest expected cluster size to render \
                             every member instead of just a sample."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_reps_per_cluster).desired_width(60.0))
                            .on_hover_text("Representatives rendered per cluster (≥ largest cluster = full membership).");
                        ui.end_row();
                        ui.label("Noise sample").on_hover_text(
                            "How many HDBSCAN-labeled-noise zones (points that don't fit any \
                             dense cluster) to render for inspection — 0 to skip rendering them."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_noise_sample).desired_width(60.0))
                            .on_hover_text("Noise (unclustered) zones to render, 0 to skip.");
                        ui.end_row();
                    });
                    ui.label(egui::RichText::new(
                        "Leave min cluster size blank to auto-sweep and pick the best by \
                         relative_validity_ (DBCV). Reps/cluster ≥ largest cluster renders \
                         every member, not just a sample."
                    ).color(Color32::GRAY).small());
                    ui.add_enabled_ui(!self.eo_busy, |ui| {
                        if ui.button("Run Clustering")
                            .on_hover_text("Runs scripts/cluster_latent.py (UMAP reduction + HDBSCAN).")
                            .clicked()
                        { self.start_clustering(); }
                    });
                });

                egui::CollapsingHeader::new("5. Video-Zoom Explore").default_open(false).show(ui, |ui| {
                    ui.label(egui::RichText::new(
                        "Searches from the CURRENT view for a zoom path whose rendered, \
                         compressed preview video is as large/entropic as possible — a \
                         boring/smooth zoom compresses small, a visually rich one resists \
                         compression. Looks several moves ahead before committing to each \
                         step (like a chess engine), capped just before the DD precision \
                         wall at the export width set below in \"Add to Queue\"."
                    ).color(Color32::GRAY).small());
                    ui.horizontal(|ui| {
                        ui.label("Method:");
                        egui::ComboBox::from_id_salt("eo_vz_method")
                            .selected_text(EO_METHODS[self.eo_vz_method_idx])
                            .show_ui(ui, |ui| {
                                for (i, m) in EO_METHODS.iter().enumerate() {
                                    ui.selectable_value(&mut self.eo_vz_method_idx, i, *m);
                                }
                            });
                    });
                    egui::Grid::new("eo_vz_grid").num_columns(4).show(ui, |ui| {
                        ui.label("Depth").on_hover_text("Real committed zoom steps before stopping.");
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_depth).desired_width(40.0));
                        ui.label("Finalists/level").on_hover_text("Immediate branches evaluated at each real step.");
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_finalists).desired_width(40.0));
                        ui.end_row();
                        ui.label("Lookahead plies").on_hover_text(
                            "Moves looked ahead per branch before committing to the first move \
                             — the \"chess-like\" part. Cost scales roughly linearly with this."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_lookahead).desired_width(40.0));
                        ui.label("Top winners").on_hover_text("How many ranked results to keep at the end.");
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_top_winners).desired_width(40.0));
                        ui.end_row();
                        ui.label("Seeds").on_hover_text(
                            "1 = search only from the current view. >1 additionally wide-searches \
                             for that many starting regions instead of just the one."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_n_seeds).desired_width(40.0));
                        ui.label("Min score").on_hover_text(
                            "Absolute floor on a candidate's raw score — below this, a candidate \
                             is rejected outright rather than just ranked low, so a neighborhood \
                             where NOTHING clears the bar registers as a dead end and the search \
                             backtracks, instead of always drilling into whatever's \"best\" \
                             nearby even if everything nearby is bad. Raise if a run still ends up \
                             in near-flat territory; lower if real zones start getting rejected."
                        );
                        ui.add(egui::TextEdit::singleline(&mut self.eo_vz_min_score).desired_width(40.0));
                        ui.end_row();
                    });
                    ui.add_enabled_ui(!self.eo_busy, |ui| {
                        if ui.button("Start Video-Zoom Explore")
                            .on_hover_text("Runs `explorer video-zoom-explore` from the current view.")
                            .clicked()
                        { self.start_video_zoom_explore(); }
                    });
                    if self.eo_busy && let Some((seed_id, committed, budget)) = self.eo_vz_progress {
                        let frac = if budget > 0 { committed as f32 / budget as f32 } else { 0.0 };
                        ui.horizontal(|ui| {
                            ui.add(egui::ProgressBar::new(frac.min(1.0)).desired_width(160.0));
                            ui.label(format!("seed {seed_id}: {committed}/{budget} real moves committed"))
                                .on_hover_text(
                                    "Real search steps actually taken so far, out of this seed's \
                                     total backtracking budget (depth × 3) — NOT a smooth time \
                                     estimate, since depth-first backtracking isn't uniformly \
                                     paced, but an honest count of concrete progress. Distinct \
                                     from the orange/gray squares below, which include lookahead \
                                     probes that may never be taken."
                                );
                        });
                    }
                    if !self.eo_vz_winners.is_empty() {
                        ui.separator();
                        ui.label(format!("{} winner(s), ranked by compressed-video ratio:", self.eo_vz_winners.len()));
                        // `self.eo_vz_winners` stays borrowed for this whole
                        // closure; a "Queue this winner" click just records
                        // its index locally (`queue_idx`, not `self`) and
                        // the actual `&mut self` call happens once, after
                        // the closure (and that borrow) ends.
                        let mut queue_idx: Option<usize> = None;
                        egui::ScrollArea::horizontal().id_salt("eo_vz_gallery").show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for (i, w) in self.eo_vz_winners.iter().enumerate() {
                                    ui.vertical(|ui| {
                                        if let Some(tex) = &w.thumb {
                                            ui.add(egui::Image::new(tex).max_width(96.0).max_height(96.0));
                                        }
                                        ui.label(format!(
                                            "#{} · {:.4} · {} legs",
                                            w.rank, w.final_probe_ratio.unwrap_or(0.0), w.n_legs
                                        )).on_hover_text(format!(
                                            "Ended: {} · preview: {}", w.ended_reason, w.preview_mp4.display()
                                        ));
                                        if ui.button("Queue this winner")
                                            .on_hover_text("Adds this winner's full discovered path to the video export queue, using the Add-to-Queue settings below (steps/fps/width/height).")
                                            .clicked()
                                        { queue_idx = Some(i); }
                                    });
                                    ui.separator();
                                }
                            });
                        });
                        if let Some(i) = queue_idx {
                            self.queue_video_zoom_winner(i);
                        }
                    }
                });

                ui.separator();
                ui.checkbox(&mut self.eo_follow_scan, "Follow scan (auto-navigate main canvas)")
                    .on_hover_text(
                        "While Stage 1 runs, snaps the main fractal view to wherever the search \
                         currently is each time it moves to a new seed/recursion level, and \
                         re-renders there (a fast preview render, not a final-quality one) — \
                         without this, the scanning/saved squares almost never line up with \
                         whatever the canvas already happened to be showing. Turn off if you'd \
                         rather keep navigating manually while a stage runs in the background."
                    );
                if self.eo_busy {
                    ui.horizontal(|ui| {
                        ui.colored_label(Color32::YELLOW, format!("{}…", self.eo_stage));
                        let clicked = ui.add_enabled(!self.eo_cancelling, egui::Button::new("Cancel"))
                            .on_hover_text("Kills the running subprocess (`kill <pid>`).")
                            .clicked();
                        if clicked { self.cancel_explore_stage(); }
                    });
                } else if !self.eo_message.is_empty() {
                    let col = if self.eo_message.contains("FAILED") || self.eo_message.contains("cannot") {
                        Color32::from_rgb(255, 120, 120)
                    } else {
                        Color32::LIGHT_GREEN
                    };
                    ui.colored_label(col, &self.eo_message);
                }

                if self.eo_zone_count > 0 || self.eo_preview_texture.is_some() || self.eo_preview_view.is_some() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            // `eo_zone_count`/`eo_preview_texture` only ever
                            // populate for stages that save numbered
                            // zone_*.nn files (vae-explore) — video-zoom-
                            // explore has no such concept (nothing is saved
                            // until the run's very end), so it only ever
                            // drives `eo_preview_view` via `"committed_move"`
                            // events. Show each piece only when it's
                            // actually meaningful for whichever stage is
                            // running, rather than a misleading "0" or a
                            // blank image slot.
                            if self.eo_zone_count > 0 || self.eo_preview_texture.is_some() {
                                ui.label(format!("Zones saved so far: {}", self.eo_zone_count))
                                    .on_hover_text(
                                        "Live count of zone_*.nn files in the output directory. \
                                         Rescanned roughly twice a second while a stage is running."
                                    );
                            }
                            if let Some(tex) = &self.eo_preview_texture {
                                ui.add(egui::Image::new(tex).max_width(220.0).max_height(220.0))
                                    .on_hover_text(
                                        "Most recently saved zone's colormapped preview — a live \
                                         look at what the search is actually finding right now."
                                    );
                            } else if let Some((cx, cy, zoom)) = self.eo_preview_view {
                                ui.label(format!("Current position: ({cx:.6}, {cy:.6}) @ {zoom:.3e}×"))
                                    .on_hover_text(
                                        "This stage doesn't save incremental files — this is the \
                                         most recently COMMITTED real search step (also shown as \
                                         the red square on the main canvas), not just a candidate \
                                         still being evaluated."
                                    );
                            }
                            if !self.eo_scan_candidates.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.colored_label(Color32::from_rgb(255, 165, 0), "■").on_hover_text(
                                        "Squares drawn on the main fractal canvas (behind this \
                                         window) for this level's candidates, colored by status."
                                    );
                                    ui.label("scanning (passed the cheap gate, not yet rendered/saved)");
                                });
                                ui.horizontal(|ui| {
                                    ui.colored_label(Color32::from_gray(120), "■").on_hover_text(
                                        "Rejected on cheap coarse metrics (degenerate / too \
                                         intricate / not enough edge detail) — never gets a \
                                         precise render, never saved."
                                    );
                                    ui.label("rejected this level");
                                });
                                ui.horizontal(|ui| {
                                    ui.colored_label(Color32::RED, "■").on_hover_text(
                                        "The most recently saved zone_*.nn, or (video-zoom-explore) \
                                         the most recently committed real search step."
                                    );
                                    ui.label("validated / committed");
                                });
                            }
                        });
                        if let Some(tex) = &self.eo_saliency_texture {
                            ui.separator();
                            ui.vertical(|ui| {
                                ui.label("Saliency heatmap").on_hover_text(
                                    "Raw predicted heatmap from the saliency model for the CURRENT \
                                     level's canvas — brighter = the model predicts higher VAE \
                                     reconstruction error (more \"interesting\") there. Also drawn \
                                     as a translucent red tint directly over the main canvas at the \
                                     seed's frame extent."
                                );
                                ui.add(egui::Image::new(tex).max_width(160.0).max_height(160.0));
                            });
                        }
                    });
                }

                egui::ScrollArea::vertical().max_height(180.0).stick_to_bottom(true).show(ui, |ui| {
                    for line in &self.eo_log {
                        ui.label(egui::RichText::new(line).small().monospace());
                    }
                });

                ui.separator();
                if ui.button("Close").clicked() { do_close = true; }
            });
        if do_close {
            self.show_explore_options = false;
        }
    }
}

/// Milliseconds since the UI thread last began a frame, as a wall-clock
/// stamp — read by the watchdog thread in `main` to notice a frozen window.
/// See `spawn_ui_watchdog` for why this exists at all.
static UI_LAST_FRAME_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether an explore stage is running, as of the last frame the UI thread
/// managed to draw. The watchdog only judges stalls while this is true: an
/// idle egui window legitimately draws NOTHING for minutes (it repaints on
/// events, not on a clock), so an unconditional watchdog would report every
/// quiet period as a freeze. During a stage the viewer calls
/// `request_repaint()` on every child log line, so frames are guaranteed —
/// and their absence is real. If the UI thread freezes, this keeps whatever
/// value it had at that moment, which is exactly the state to judge by.
static UI_STAGE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Two relaxed stores per frame — see `UI_LAST_FRAME_MS`.
        UI_LAST_FRAME_MS.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
        UI_STAGE_ACTIVE.store(self.eo_busy, std::sync::atomic::Ordering::Relaxed);
        let ctx = ui.ctx().clone();

        // IPC: load new genome if another launch delegated to us
        while let Ok(path) = self.ipc_rx.try_recv() {
            self.load_new_genome(path);
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        // Hi-res save progress: update the status line as background saves report in.
        while let Ok(msg) = self.save_rx.try_recv() {
            match msg {
                SaveMsg::Started { w, h } => {
                    self.save_status = format!("Rendering {w}×{h} PNG…");
                }
                SaveMsg::Done(path) => {
                    self.saves_active = self.saves_active.saturating_sub(1);
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    self.save_status = format!("Saved → {name}");
                    eprintln!("Saved → {}", path.display());
                }
                SaveMsg::Failed(e) => {
                    self.saves_active = self.saves_active.saturating_sub(1);
                    self.save_status = format!("Save FAILED: {e}");
                    eprintln!("Save error: {e}");
                }
            }
        }
        // Drop handles for saves that have already completed (keeps the vec small;
        // any still-running ones remain to be joined on exit).
        self.save_jobs.retain(|h| !h.is_finished());

        // Auto-palette: apply result when background scoring finishes
        if let Some(ref rx) = self.auto_pal_rx {
            if let Ok(best) = rx.try_recv() {
                self.set_colormap(best);
                self.auto_pal_busy = false;
                self.auto_pal_rx = None;
            }
        }

        // Outer-limit search: when the background search finishes, zoom the
        // view out to the found limit — X/Y independently when both are
        // available (the true asymmetric bounding box), falling back to the
        // combined XY square for whichever axis didn't find one on its own.
        // Applied at the CENTER the search actually ran at (OuterLimitResult
        // carries it), not assumed to still match self.view — the user may
        // have panned away while the search was running in the background.
        if let Some(ref rx) = self.outer_limit_rx {
            if let Ok(result) = rx.try_recv() {
                self.outer_limit_busy = false;
                self.outer_limit_rx = None;
                let half_x = result.x.or(result.xy).map(|v| v as f64);
                let half_y = result.y.or(result.xy).map(|v| v as f64);
                if let (Some(hx), Some(hy)) = (half_x, half_y) {
                    self.push_view();
                    self.view.cx = result.cx;
                    self.view.cx_lo = 0.0;
                    self.view.cy = result.cy;
                    self.view.cy_lo = 0.0;
                    self.view.zoom = (2.0 / hy).clamp(MIN_ZOOM, MAX_ZOOM);
                    self.view.aspect = hx / hy;
                    self.sync_xy = true;
                    self.request_render(false);
                }
                self.outer_limit_result = Some(result);
            }
        }

        // Auto-select: poll the current round. A found square goes to
        // Previewing (held on screen, highlighted red, for
        // AUTO_SELECT_PREVIEW) before the zoom is actually applied; after
        // applying it, Waiting holds the newly-zoomed result on screen for
        // AUTO_SELECT_WAIT before the next queued round (if any) starts —
        // "waiting a few seconds after rendering so the user can check the
        // evolution of the zoom."
        match &self.auto_select_state {
            AutoSelectState::Searching(rx) => {
                if let Ok(result) = rx.try_recv() {
                    match result {
                        Some((dx, dy, zoom)) => {
                            // Only counts as genuine recovery from a
                            // failure chain once it clears the ceiling we
                            // most recently failed at — the search is fully
                            // deterministic, so merely "succeeding" isn't
                            // enough on its own: it can succeed its way
                            // right back into the same wall it just backed
                            // away from (confirmed empirically: without
                            // this check, a 1-level backtrack oscillates
                            // between the same two zoom levels forever).
                            if zoom > self.auto_select_stuck_ceiling {
                                self.auto_select_backtracks_left = AUTO_SELECT_MAX_BACKTRACK;
                                self.auto_select_backtrack_depth = 1;
                                self.auto_select_stuck_ceiling = f64::INFINITY;
                            }
                            self.auto_select_state = AutoSelectState::Previewing {
                                dx, dy, zoom, until: std::time::Instant::now() + AUTO_SELECT_PREVIEW,
                            };
                        }
                        None => {
                            // The current view's own fixed local grid found
                            // nothing (even after find_interesting_square's
                            // internal widened retry) — this can mean the
                            // LAST committed square was itself a mediocre
                            // pick that walked the search into a dead end,
                            // not that there's nothing left anywhere nearby.
                            // Step back and try again from there before
                            // giving up — backing up MORE levels each
                            // consecutive failure, since retrying from the
                            // immediate parent just re-finds the identical
                            // doomed candidate (the search is deterministic).
                            if self.auto_select_backtracks_left > 0 && !self.view_stack.is_empty() {
                                self.auto_select_backtracks_left -= 1;
                                if self.auto_select_stuck_ceiling.is_infinite() {
                                    self.auto_select_stuck_ceiling = self.view.zoom;
                                }
                                self.backtrack_view(self.auto_select_backtrack_depth);
                                self.auto_select_backtrack_depth += 1;
                                self.spawn_auto_select_search();
                            } else {
                                self.auto_select_queued = 0;
                                self.auto_select_message =
                                    "Nothing sufficiently interesting found nearby, even after backtracking — try panning/zooming manually.".to_string();
                                self.auto_select_state = AutoSelectState::Idle;
                            }
                        }
                    }
                }
            }
            AutoSelectState::Previewing { dx, dy, zoom, until } => {
                if std::time::Instant::now() >= *until {
                    let (dx, dy, zoom) = (*dx, *dy, *zoom);
                    self.push_view();
                    // dx/dy are offsets from the view's center at search time
                    // (see find_interesting_square); add them in DD so the
                    // new center stays precise past f64's ~10^11 zoom limit
                    // instead of collapsing back to plain-f64 precision.
                    let new_cx = self.view.cx_dd() + Dd::from_f64(dx);
                    let new_cy = self.view.cy_dd() + Dd::from_f64(dy);
                    self.view.set_cx_dd(new_cx);
                    self.view.set_cy_dd(new_cy);
                    self.view.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                    // aspect deliberately left as-is: the square found becomes
                    // the new frame's full height, width follows the current
                    // aspect setting, same as any other zoom action.
                    self.sync_xy = true;
                    self.request_render(false);
                    self.auto_select_state = AutoSelectState::Waiting(std::time::Instant::now() + AUTO_SELECT_WAIT);
                }
            }
            AutoSelectState::Waiting(deadline) => {
                if std::time::Instant::now() >= *deadline {
                    self.auto_select_state = AutoSelectState::Idle;
                }
            }
            AutoSelectState::Idle => {}
        }
        // Single consumption point for queued rounds — the only place that
        // ever starts one, so a click can never fire more rounds than
        // requested (previously, incrementing on click AND separately
        // consuming on the next Waiting→Searching transition double-counted
        // the very first click's round).
        if matches!(self.auto_select_state, AutoSelectState::Idle) && self.auto_select_queued > 0 {
            self.auto_select_queued -= 1;
            self.start_auto_select_round();
        }

        // Wormhole: poll the current search. Same preview-then-apply shape
        // as Auto-Select, no backtracking (see WormholeState's doc comment).
        match &self.wormhole_state {
            WormholeState::Searching(rx) => {
                if let Ok(result) = rx.try_recv() {
                    match result {
                        Some((dx, dy, zoom)) => {
                            self.wormhole_state = WormholeState::Previewing {
                                dx, dy, zoom, until: std::time::Instant::now() + AUTO_SELECT_PREVIEW,
                            };
                        }
                        None => {
                            self.wormhole_queued = 0;
                            self.wormhole_message =
                                "No confident self-similar copy found nearby — try a different view.".to_string();
                            self.wormhole_state = WormholeState::Idle;
                        }
                    }
                }
            }
            WormholeState::Previewing { dx, dy, zoom, until } => {
                if std::time::Instant::now() >= *until {
                    let (dx, dy, zoom) = (*dx, *dy, *zoom);
                    self.push_view();
                    let new_cx = self.view.cx_dd() + Dd::from_f64(dx);
                    let new_cy = self.view.cy_dd() + Dd::from_f64(dy);
                    self.view.set_cx_dd(new_cx);
                    self.view.set_cy_dd(new_cy);
                    self.view.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                    self.sync_xy = true;
                    self.request_render(false);
                    self.wormhole_state = WormholeState::Waiting(std::time::Instant::now() + AUTO_SELECT_WAIT);
                }
            }
            WormholeState::Waiting(deadline) => {
                if std::time::Instant::now() >= *deadline {
                    self.wormhole_state = WormholeState::Idle;
                }
            }
            WormholeState::Idle => {}
        }
        if matches!(self.wormhole_state, WormholeState::Idle) && self.wormhole_queued > 0 {
            self.wormhole_queued -= 1;
            self.spawn_wormhole_search();
        }

        if let Some(ref rx) = self.wormhole_video_rx {
            if let Ok(result) = rx.try_recv() {
                self.wormhole_video_busy = false;
                self.wormhole_video_rx = None;
                self.wormhole_video_message = match result {
                    Ok(()) => "Wormhole video queued ✓".to_string(),
                    Err(e) => e,
                };
            }
        }

        if let Some(ref rx) = self.explore_rx {
            if let Ok(msg) = rx.try_recv() {
                self.explore_busy = false;
                self.explore_rx = None;
                self.explore_message = match msg {
                    ExploreMsg::Done { results, out_dir } => {
                        // Sorted most-unusual-first by start_explore — [0] IS the answer to
                        // "find the most unusual", not just the count of what was found.
                        match results.first() {
                            Some(top) => format!(
                                "Found {} in {} — most unusual: novelty={} score={:.2} cx={:.6} cy={:.6} zoom={:.3e}  {}",
                                results.len(), out_dir.display(),
                                top.novelty.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
                                top.score, top.cx, top.cy, top.zoom, top.png_path.display(),
                            ),
                            None => format!("Found 0 in {}", out_dir.display()),
                        }
                    }
                    ExploreMsg::Failed(e) => e,
                };
            }
        }

        // Explore Options pipeline: drain ALL buffered log lines each frame
        // (not just one) so a fast-printing stage like complex-export
        // doesn't visibly lag behind what the subprocess already emitted.
        if let Some(ref rx) = self.eo_rx {
            let mut finished = false;
            // Set inside the loop (while `rx`, borrowed from `self.eo_rx`,
            // is still in use by the loop condition itself — calling a
            // `&mut self` method there would conflict with that borrow) and
            // acted on afterward, same deferral `finished` below already
            // relies on.
            let mut just_finished_video_zoom = false;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    ExploreOpsMsg::Line(l) => {
                        self.eo_log.push(l);
                        if self.eo_log.len() > 500 { self.eo_log.remove(0); }
                    }
                    ExploreOpsMsg::Done => {
                        self.eo_message = format!("{} finished ✓", self.eo_stage);
                        just_finished_video_zoom = self.eo_stage == "Video-zoom exploring";
                        finished = true;
                    }
                    ExploreOpsMsg::Failed(e) => {
                        self.eo_message = if self.eo_cancelling {
                            format!("{} cancelled", self.eo_stage)
                        } else {
                            format!("{} FAILED: {e}", self.eo_stage)
                        };
                        finished = true;
                    }
                }
            }
            if just_finished_video_zoom {
                self.load_video_zoom_winners(&ctx);
            }
            if finished {
                self.eo_busy = false;
                self.eo_rx = None;
                self.eo_child_pid = None;
                self.eo_cancelling = false;
            }
        }

        // Zoom animation: advance one step per frame, request preview render
        if self.zoom_anim {
            self.apply_zoom(true, 1.02);
            ctx.request_repaint();
        }

        self.poll_render(&ctx);
        self.handle_keyboard(&ctx);
        self.show_toolbar(ui);
        self.show_status_bar(ui);
        self.show_bottom_bar(ui);
        self.show_fractal_panel(ui);
        self.show_help_window(&ctx);
        self.show_save_window(&ctx);
        self.update_explore_preview(&ctx);
        self.show_explore_options_window(&ctx);

        // Auto-upgrade: if the settled render was a preview (user paused after panning/zooming),
        // kick off a full-quality render.  Using the is_preview flag instead of a size heuristic
        // avoids an infinite loop when the DD or f64 path caps resolution below display size.
        if self.render_complete && self.displayed_gen == self.render_gen && self.displayed_is_preview {
            self.request_render(false);
        }

        if !self.render_complete || self.displayed_gen < self.render_gen
            || self.saves_active > 0 || self.outer_limit_busy
            || !matches!(self.auto_select_state, AutoSelectState::Idle) {
            ctx.request_repaint();
        }
    }

    /// Don't lose in-flight hi-res saves when the window closes: block until every
    /// background save thread has finished writing its PNG. A deep-zoom save can take
    /// a while, so exit may pause briefly — but the file is guaranteed to land.
    fn on_exit(&mut self) {
        // Kill any running Explore Options subprocess (`explorer vae-explore`
        // / `video-zoom-explore` / a scripts/*.py stage — see
        // `spawn_explore_stage`) so closing the viewer doesn't leave it
        // running orphaned in the background — a real report (Carl,
        // 2026-08-13): closing the window left a video-zoom-explore search
        // still consuming CPU with no way back to its Cancel button. Same
        // `kill <pid>` `cancel_explore_stage` already uses. Deliberately
        // does NOT touch `nnfractals-queue` (via `wake_or_launch_queue_window`)
        // — that one is an intentionally independent background service,
        // meant to keep processing the export queue after the viewer closes.
        if let Some(pid) = self.eo_child_pid {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
        if !self.save_jobs.is_empty() {
            eprintln!("Waiting for {} pending save(s) to finish…", self.save_jobs.len());
        }
        for h in self.save_jobs.drain(..) {
            let _ = h.join();
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute a selection rectangle constrained to `aspect` (w/h), returning (rect, valid).
fn selection_rect(start: egui::Pos2, cur: egui::Pos2, aspect: f32) -> (egui::Rect, bool) {
    let dx = cur.x - start.x;
    let dy = cur.y - start.y;
    // Constrain so sel_w / sel_h == aspect
    let (sw, sh) = if dx.abs() / aspect.max(0.001) < dy.abs() {
        (dy.abs() * aspect, dy.abs())
    } else {
        (dx.abs(), dx.abs() / aspect.max(0.001))
    };
    if sw < MIN_SEL_PX || sh < MIN_SEL_PX {
        return (egui::Rect::NOTHING, false);
    }
    let x1 = if dx >= 0.0 { start.x } else { start.x - sw };
    let y1 = if dy >= 0.0 { start.y } else { start.y - sh };
    (egui::Rect::from_min_size(egui::Pos2::new(x1, y1), egui::Vec2::new(sw, sh)), true)
}

/// Gradient energy of an RGB image: sum of squared pixel differences along x and y.
/// Higher = more visual detail visible with this palette — used by auto-palette.
fn auto_palette_score(rgb: &[u8], w: usize, h: usize) -> f32 {
    let mut sum = 0.0f64;
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let i = (y * w + x) * 3;
            let r = (y * w + x + 1) * 3;  // right neighbour
            let d = ((y + 1) * w + x) * 3; // down neighbour
            for c in 0..3 {
                let dx = rgb[r + c] as f64 - rgb[i + c] as f64;
                let dy = rgb[d + c] as f64 - rgb[i + c] as f64;
                sum += dx * dx + dy * dy;
            }
        }
    }
    (sum / (w * h) as f64) as f32
}

/// Returns the WASD / arrow-key step multiplier for the given modifiers.
fn modifier_scale(mods: &egui::Modifiers) -> f64 {
    if mods.ctrl && mods.shift { 30.0 }
    else if mods.ctrl && mods.alt { 0.3 }
    else if mods.shift { 2.0 }
    else if mods.alt   { 0.5 }
    else               { 1.0 }
}

// ── Default config ────────────────────────────────────────────────────────────

fn default_config() -> Config {
    use nnfractals::config::{DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig, RenderingConfig};
    Config {
        dedup: DedupConfig::default(),
        mass_extinction: MassExtinctionConfig::default(),
        rendering: RenderingConfig {
            default_width: 800, default_height: 800,
            max_iter: 256, bailout: 4.0,
            colormap: "turbo".into(),
            view_x_min: -2.0, view_x_max: 2.0,
            view_y_min: -2.0, view_y_max: 2.0,
        },
        optimization: OptimizationConfig {
            population_size: 40, elitism_count: 6,
            mutation_rate: 0.20, mutation_scale: 0.08,
            eval_width: 64, eval_height: 64, eval_max_iter: 128,
            restart_after_gens: 30, novelty_weight: 0.45,
            novelty_k: 5, archive_size: 150,
            self_replication_weight: 0.35,
            fractal_recursion_weight: 0.35,
            recursion_pred_weight: 0.60,
            formula_diversity_weight: 0.30,
            clip_pred_weight: 0.50,
            formula_system: "dag".to_string(),
            max_nodes: 14, max_depth: 5,
            ood_weight: 0.0,
            pref_weight: 0.4,
            seed_pref_weight: 3.0,
            musiq_weight: 0.25,
            pref_elite_count: 4,
            archive_random_ratio: 0.30,
            duplicate_penalty_weight: 0.50,
            archive_seeding_enabled: false,
            angle_structure_weight: 0.0,
            img_novelty_weight: 0.0,
        },
        output: OutputConfig {
            save_dir: "./fractals".into(),
            population_dir: "./populations".into(),
            min_entropy_prefilter: 0.42, max_entropy_prefilter: 0.65,
            min_clip_score: 0.512, min_laion_score: 5.30,
            min_beauty: 0.35, min_save_distance: 0.04,
            min_ensemble: 4.6, min_musiq: 30.0, min_pref: 0.45,
        },
    }
}

// ── IPC — single-instance socket ─────────────────────────────────────────────

/// Cleans up the Unix socket file on drop (best-effort).
struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

fn socket_path() -> PathBuf {
    let tag = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    std::env::temp_dir().join(format!("nnfractals-viewer-{tag}.sock"))
}

/// Socket path for the queue window — same tag-by-user convention as
/// `socket_path()` above, distinct filename.
fn queue_socket_path() -> PathBuf {
    let tag = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    std::env::temp_dir().join(format!("nnfractals-queue-{tag}.sock"))
}

/// Locate a sibling binary next to this one — same lookup order
/// `browser.rs`/`launcher.rs` already use to find each other, and the same
/// release-first fix applied there (see `browser.rs::locate_bin`'s doc
/// comment for the real incident this fixes): target/release/<name> is
/// checked BEFORE this exe's own dir, so a debug build never silently
/// prefers a debug sibling just because it happens to sit next to one.
fn locate_sibling_bin(name: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(target) = dir.parent() {
                let c = target.join("release").join(name);
                if c.exists() { return c; }
            }
            let c = dir.join(name);
            if c.exists() { return c; }
            if let Some(target) = dir.parent() {
                let c = target.join("debug").join(name);
                if c.exists() { return c; }
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let c = PathBuf::from(home).join(".local/bin").join(name);
        if c.exists() { return c; }
    }
    PathBuf::from(name)
}

/// Wake an already-open queue window (its own IPC listener will notice the
/// connection, reload `queue.json`, and focus itself), or launch a fresh
/// one if none is running — the queue binary needs no arguments.
fn wake_or_launch_queue_window() {
    if UnixStream::connect(queue_socket_path()).is_ok() {
        return;
    }
    let _ = std::process::Command::new(locate_sibling_bin("nnfractals-queue")).spawn();
}

/// Try to connect to a running viewer and hand it the new path.
/// Returns true if delegated successfully (caller should exit).
fn try_delegate(sock: &Path, path: &Path) -> bool {
    match UnixStream::connect(sock) {
        Ok(mut s) => {
            let _ = s.write_all(path.to_string_lossy().as_bytes());
            true
        }
        Err(_) => false,
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Reports when the UI thread stops drawing frames.
///
/// The window freezes hard during explore stages — niri's close request goes
/// unanswered and only `killall` clears it, which destroys the evidence. An
/// external watcher can see a UI thread that is *blocked* (no CPU), but not
/// one that is *spinning*: a livelock burns CPU and looks healthy from
/// outside. Only the UI thread itself knows whether frames are still coming,
/// so it stamps `UI_LAST_FRAME_MS` and this thread notices the gap.
///
/// Costs one relaxed atomic store per frame and one wakeup per second.
const UI_STALL_MS: u64 = 3_000;

/// What the watchdog should say about one sample. Split out from the thread
/// so the state machine is testable without real time, real threads, or a
/// window — same reason `video_zoom_explore::usable_leg_span` is a free
/// function.
#[derive(Debug, PartialEq, Eq)]
enum StallVerdict {
    Quiet,
    Started(u64),
    Continuing(u64),
    Recovered,
}

/// `reported_at` is `Some(last_frame_stamp)` once an episode has been
/// announced, so a later sample can tell "still the same freeze" (the stamp
/// hasn't moved) from "a frame arrived" (it has).
fn stall_verdict(
    stage_active: bool, last_frame_ms: u64, now: u64, reported_at: Option<u64>,
) -> StallVerdict {
    // Not drawing while idle is normal — see `UI_STAGE_ACTIVE`. A frame
    // stamp of 0 means no frame has been drawn yet at all (startup).
    if !stage_active || last_frame_ms == 0 {
        return if reported_at.is_some() { StallVerdict::Recovered } else { StallVerdict::Quiet };
    }
    let age = now.saturating_sub(last_frame_ms);
    match reported_at {
        None if age >= UI_STALL_MS => StallVerdict::Started(age),
        // Same episode: the UI thread has not drawn since it was announced.
        // Re-report periodically rather than once, so the log shows how long
        // it lasted even when the process is killed before it recovers.
        Some(start) if start == last_frame_ms => {
            if age % 10_000 < 1_100 { StallVerdict::Continuing(age) } else { StallVerdict::Quiet }
        }
        Some(_) => StallVerdict::Recovered,
        None => StallVerdict::Quiet,
    }
}

fn spawn_ui_watchdog() {
    thread::spawn(move || {
        let mut reported_at: Option<u64> = None;
        loop {
            thread::sleep(std::time::Duration::from_secs(1));
            let last = UI_LAST_FRAME_MS.load(std::sync::atomic::Ordering::Relaxed);
            let active = UI_STAGE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
            match stall_verdict(active, last, now_ms(), reported_at) {
                StallVerdict::Started(age) => {
                    eprintln!("[viewer] UI THREAD STALLED — no frame for {:.1}s (stage: running)",
                              age as f64 / 1000.0);
                    reported_at = Some(last);
                }
                StallVerdict::Continuing(age) => {
                    eprintln!("[viewer] UI THREAD STILL STALLED — {:.0}s", age as f64 / 1000.0);
                }
                StallVerdict::Recovered => {
                    eprintln!("[viewer] UI thread recovered");
                    reported_at = None;
                }
                StallVerdict::Quiet => {}
            }
        }
    });
}

#[cfg(test)]
mod watchdog_tests {
    use super::*;

    #[test]
    fn an_idle_window_is_never_reported_as_stalled() {
        // The whole reason UI_STAGE_ACTIVE exists: egui draws nothing for
        // minutes when nothing is happening, which must not look like a hang.
        assert_eq!(stall_verdict(false, 1_000, 1_000_000, None), StallVerdict::Quiet);
    }

    #[test]
    fn startup_before_the_first_frame_is_not_a_stall() {
        assert_eq!(stall_verdict(true, 0, 60_000, None), StallVerdict::Quiet);
    }

    #[test]
    fn a_running_stage_that_stops_drawing_is_reported_once_then_periodically() {
        // Under the threshold: nothing.
        assert_eq!(stall_verdict(true, 10_000, 12_000, None), StallVerdict::Quiet);
        // Over it: announced, with the age.
        assert_eq!(stall_verdict(true, 10_000, 13_500, None), StallVerdict::Started(3_500));
        // Same episode (frame stamp unmoved), off the 10s beat: silent.
        assert_eq!(stall_verdict(true, 10_000, 15_000, Some(10_000)), StallVerdict::Quiet);
        // Same episode, on the beat: re-reported.
        assert_eq!(stall_verdict(true, 10_000, 20_000, Some(10_000)), StallVerdict::Continuing(10_000));
    }

    #[test]
    fn a_new_frame_ends_the_episode() {
        // The frame stamp moved, so the UI thread is drawing again.
        assert_eq!(stall_verdict(true, 21_000, 21_100, Some(10_000)), StallVerdict::Recovered);
        // A stage that finishes mid-freeze also closes the episode, rather
        // than leaving it open forever.
        assert_eq!(stall_verdict(false, 10_000, 30_000, Some(10_000)), StallVerdict::Recovered);
    }
}

fn main() -> anyhow::Result<()> {
    let nn_path = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("Usage: nnfractals-viewer <genome.nn>")
    })?;

    // Opt-in: allow an unprivileged watcher to attach and read stacks.
    // This window freezes hard enough during an explore stage that niri's
    // close request goes unanswered and only `killall` clears it — which
    // also destroys the evidence. Yama's ptrace_scope=1 (the setting on this
    // machine) restricts ptrace to direct ancestors, so a watcher running
    // ALONGSIDE the viewer cannot dump stacks mid-freeze without root. Gated
    // behind an env var so an ordinary run keeps the default restriction;
    // scripts/hang_watch.sh documents the workflow.
    #[cfg(target_os = "linux")]
    if std::env::var_os("NNFRACTALS_ALLOW_PTRACE").is_some() {
        // SAFETY: PR_SET_PTRACER passes no pointers and touches no memory —
        // its only effect is relaxing who may attach to THIS process.
        unsafe { libc::prctl(libc::PR_SET_PTRACER, libc::PR_SET_PTRACER_ANY); }
        eprintln!("[viewer] ptrace attach allowed (NNFRACTALS_ALLOW_PTRACE)");
    }

    spawn_ui_watchdog();

    // ── Single-instance IPC ───────────────────────────────────────────────────
    let sock_path = socket_path();
    if try_delegate(&sock_path, &nn_path) {
        eprintln!("[viewer] Delegated to running instance.");
        return Ok(());
    }
    // No existing instance — become the server.
    let _ = std::fs::remove_file(&sock_path); // remove any stale socket
    let (ipc_tx, ipc_rx) = mpsc::channel::<PathBuf>();
    let _sock_guard = match UnixListener::bind(&sock_path) {
        Ok(listener) => {
            let tx = ipc_tx;
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if let Ok(mut s) = stream {
                        let mut buf = String::new();
                        if s.read_to_string(&mut buf).is_ok() {
                            let p = PathBuf::from(buf.trim());
                            if p.exists() { let _ = tx.send(p); }
                        }
                    }
                }
            });
            Some(SocketGuard(sock_path))
        }
        Err(e) => { eprintln!("[viewer] IPC unavailable: {e}"); None }
    };

    // ── GPU init ──────────────────────────────────────────────────────────────
    #[cfg(feature = "wgpu-backend")]
    {
        render_gpu::init_gpu();
        eprintln!(
            "[viewer] Renderer: {}",
            if render_gpu::gpu_available() { "GPU (wgpu)" } else { "CPU (rayon fallback)" }
        );
    }

    let prefs_path = Path::new(&nn_path).parent().unwrap_or(Path::new("."))
        .join("viewer_prefs.toml");
    let prefs = ViewerPrefs::load(&prefs_path);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Viewer")
            .with_inner_size([prefs.window_width as f32, prefs.window_height as f32]),
        ..Default::default()
    };

    eframe::run_native(
        "NNFractals Viewer",
        options,
        Box::new(move |cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new(cc, nn_path, ipc_rx).expect("Failed to load genome")))
        }),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}

#[cfg(test)]
mod angle_coloring_tests {
    use super::*;
    use nnfractals::formula::{op, OpNode};
    use nnfractals::genome::{FormulaTerm, Genome};

    fn mandelbrot_dag_genome() -> Genome {
        let program = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        Genome { program, bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
    }

    // Manual smoke-test surrogate: since this is a native egui app (no
    // browser/click-automation tool available for it here), verify the
    // toggle's actual rendering effect directly rather than by driving the
    // UI — the checkbox itself just flips App.angle_coloring and requests a
    // re-render (see show_toolbar), so this exercises the real code path.
    #[test]
    fn angle_coloring_changes_output_for_dag_genome() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let view = View::new_square(0.0, 0.0, 1.0);
        let normal = render_cpu(&genome, &config, &view, 48, 48, 64, false, false, true);
        let angled = render_cpu(&genome, &config, &view, 48, 48, 64, false, true, true);
        assert_eq!(normal.len(), angled.len());
        assert_ne!(normal, angled,
            "angle-coloring toggle produced identical output to normal coloring");
    }

    #[test]
    fn angle_coloring_inert_for_legacy_genome() {
        // Legacy (non-DAG) genomes have no exit-angle data — the toggle must
        // be a silent no-op (render_cpu gates on genome.uses_program()).
        let terms = vec![
            FormulaTerm { basis: 0, re: 1.0, im: 0.0 }, // z²
            FormulaTerm { basis: 7, re: 1.0, im: 0.0 }, // c
        ];
        let genome = Genome { terms, view_zoom: 1.0, ..Default::default() };
        assert!(!genome.uses_program());
        let config = default_config();
        let view = View::new_square(0.0, 0.0, 1.0);
        let normal = render_cpu(&genome, &config, &view, 48, 48, 64, false, false, true);
        let angled = render_cpu(&genome, &config, &view, 48, 48, 64, false, true, true);
        assert_eq!(normal, angled, "angle_coloring must be inert for legacy genomes");
    }

    #[test]
    fn angle_coloring_inert_at_deep_zoom() {
        // want_angle requires !use_f64 — deep zoom must silently fall back
        // to normal coloring rather than crashing or threading angle
        // capture through the f64/DD paths (explicitly out of scope).
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let deep_view = View::new_square(0.0, 0.0, 1e15); // forces the f64 path
        let normal = render_cpu(&genome, &config, &deep_view, 32, 32, 64, true, false, true);
        let angled = render_cpu(&genome, &config, &deep_view, 32, 32, 64, true, true, true);
        assert_eq!(normal, angled, "angle_coloring must be inert at deep zoom (f64/DD paths)");
    }
}

#[cfg(test)]
mod bottom_bar_tests {
    use super::*;
    use nnfractals::formula::{op, OpNode};
    use nnfractals::genome::Genome;

    fn mandelbrot_dag_genome() -> Genome {
        let program = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        Genome { program, bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
    }

    // ── view_is_degenerate ──────────────────────────────────────────────

    #[test]
    fn degenerate_true_for_uniform_buffer() {
        let rgb = vec![10u8; 48 * 48 * 3]; // every pixel identical
        assert!(view_is_degenerate(&rgb));
    }

    #[test]
    fn degenerate_false_for_varied_buffer() {
        let mut rgb = vec![0u8; 48 * 48 * 3];
        for (i, px) in rgb.chunks_mut(3).enumerate() {
            let v = (i % 200) as u8; // no single color dominates >95%
            px[0] = v; px[1] = v.wrapping_add(40); px[2] = v.wrapping_add(80);
        }
        assert!(!view_is_degenerate(&rgb));
    }

    // ── rgb_compression_entropy ──────────────────────────────────────────

    #[test]
    fn compression_entropy_low_for_uniform_buffer() {
        let rgb = vec![10u8; 48 * 48 * 3]; // every pixel identical — compresses to almost nothing
        let e = rgb_compression_entropy(&rgb, 48, 48);
        assert!(e < 0.05, "expected near-zero entropy for a uniform image, got {e}");
    }

    #[test]
    fn compression_entropy_higher_for_varied_buffer() {
        let uniform = vec![10u8; 48 * 48 * 3];
        let mut varied = vec![0u8; 48 * 48 * 3];
        for (i, px) in varied.chunks_mut(3).enumerate() {
            // Pseudo-random, not just a smooth gradient — a real compressor
            // should struggle with this much more than a flat image.
            let v = ((i * 2654435761) % 256) as u8;
            px[0] = v; px[1] = v.wrapping_add(85); px[2] = v.wrapping_add(170);
        }
        let e_uniform = rgb_compression_entropy(&uniform, 48, 48);
        let e_varied = rgb_compression_entropy(&varied, 48, 48);
        assert!(e_varied > e_uniform, "expected varied content to compress worse (higher entropy): \
                                        uniform={e_uniform} varied={e_varied}");
    }

    // ── mean_std / pick_by_scale_normalized_entropy ──────────────────────

    #[test]
    fn mean_std_basic() {
        let (mean, std) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((mean - 5.0).abs() < 1e-4, "mean={mean}");
        assert!((std - 2.0).abs() < 1e-4, "std={std}"); // classic textbook example
    }

    #[test]
    fn mean_std_empty_is_zero() {
        assert_eq!(mean_std(&[]), (0.0, 0.0));
    }

    #[test]
    fn scale_normalized_pick_favors_standout_over_mediocre_wide() {
        // Directly replicates the pattern found in a real logged session:
        // scale 0 (wide) candidates all cluster around a middling entropy
        // with the wide scale's usual moderately-higher baseline; scale 3
        // (narrow) candidates are mostly low EXCEPT one that clearly stands
        // out above its own peers. Raw-max selection (the old behavior)
        // always picked scale 0 regardless — confirmed from 16/16 real
        // rounds. The z-score approach should recognize the scale-3 outlier
        // as the actual best pick since it's far above what's typical for
        // ITS class, while every scale-0 candidate is unremarkable within
        // its own class.
        let mut candidates: Vec<(usize, f64, f64, f64, f32)> = Vec::new();
        for i in 0..10 {
            candidates.push((0, i as f64, 0.0, 2.0, 0.50 + (i % 3) as f32 * 0.01)); // wide: ~0.50-0.52, unremarkable
        }
        for i in 0..10 {
            candidates.push((3, i as f64, 0.0, 16.0, 0.10 + (i % 3) as f32 * 0.01)); // narrow: mostly ~0.10-0.12
        }
        candidates.push((3, 99.0, 0.0, 16.0, 0.35)); // one narrow candidate way above its own peers

        let winner = pick_by_scale_normalized_entropy(&candidates, 4).unwrap();
        assert_eq!(winner.0, 3, "expected the scale-3 outlier to win, got scale {}", winner.0);
        assert!((winner.4 - 0.35).abs() < 1e-6);
    }

    #[test]
    fn scale_normalized_pick_still_excludes_uniformly_bad_scale() {
        // The z-score approach must NOT be fooled the other way either: a
        // scale group that's uniformly terrible shouldn't win just because
        // its least-bad member is a local outlier — AUTO_SELECT_MIN_ENTROPY_
        // FRACTION exists specifically to prevent this.
        let mut candidates: Vec<(usize, f64, f64, f64, f32)> = Vec::new();
        for i in 0..10 {
            candidates.push((0, i as f64, 0.0, 2.0, 0.50)); // wide: uniformly decent
        }
        for i in 0..9 {
            candidates.push((3, i as f64, 0.0, 16.0, 0.02)); // narrow: uniformly near-zero
        }
        candidates.push((3, 99.0, 0.0, 16.0, 0.05)); // still a local standout, but tiny in absolute terms

        let winner = pick_by_scale_normalized_entropy(&candidates, 4).unwrap();
        assert_eq!(winner.0, 0, "a uniformly-bad scale's local outlier must not beat a genuinely good scale");
    }

    // ── outer_limit_search ───────────────────────────────────────────────

    #[test]
    fn outer_limit_finds_xy_for_classic_mandelbrot() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        // Starting half-extent 2.0 (view spans roughly [-2,2]²) already frames
        // the classic Mandelbrot set reasonably — the search should find a
        // real (non-None) combined XY limit, not bail out immediately.
        let result = outer_limit_search(&genome, &config, 0.0, 0.0, 2.0);
        assert!(result.xy.is_some(), "expected a combined XY limit for a real fractal");
        assert!(result.x.is_some(), "expected an X limit for a real fractal");
        assert!(result.y.is_some(), "expected a Y limit for a real fractal");
    }

    #[test]
    fn outer_limit_none_when_starting_view_already_degenerate() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        // An absurdly large starting half-extent is already dominated by
        // uniform escaped background — the search must report "no limit
        // found" honestly rather than guessing.
        let result = outer_limit_search(&genome, &config, 0.0, 0.0, 1.0e8);
        assert!(result.xy.is_none());
    }

    // ── search-radius widening ────────────────────────────────────────
    //
    // Before this, every candidate square was confined STRICTLY inside the
    // frame that produced it — so once a round committed to a square, every
    // later round could only search deeper inside it. If the coarse grid
    // that picked that square was even slightly off-target (routine at
    // extreme zoom, where the true boundary gets thinner than a 5x5 grid
    // can resolve), there was no way back: every subsequent round just
    // searched closer to the same already-wrong spot, and a weak/empty
    // round could never recover. `sweep_candidates`'s `radius_mult` lets a
    // retry pass sample outside the naive frame instead.

    #[test]
    fn normal_sweep_stays_within_the_frame() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let view = View::new_square(-0.5, 0.0, 1.0);
        let cur_w = 4.0 / view.zoom * view.aspect;
        let cur_h = 4.0 / view.zoom;
        let half_x_frame = cur_w / 2.0;
        let half_y_frame = cur_h / 2.0;

        let (normal, _) = sweep_candidates(&genome, &config, &view, cur_w, cur_h, 1.0, true);
        assert!(!normal.is_empty());
        for &(_, dx, dy, _, _) in &normal {
            assert!(dx.abs() <= half_x_frame + 1e-9, "normal-radius candidate dx={dx} escaped the frame");
            assert!(dy.abs() <= half_y_frame + 1e-9, "normal-radius candidate dy={dy} escaped the frame");
        }
    }

    #[test]
    fn widened_sweep_can_reach_outside_the_normal_frame() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let view = View::new_square(-0.5, 0.0, 1.0);
        let cur_w = 4.0 / view.zoom * view.aspect;
        let cur_h = 4.0 / view.zoom;
        let half_x_frame = cur_w / 2.0;

        let (wide, _) = sweep_candidates(&genome, &config, &view, cur_w, cur_h, AUTO_SELECT_WIDEN_RADIUS, true);
        let wide_max_dx = wide.iter().map(|c| c.1.abs()).fold(0.0f64, f64::max);
        assert!(wide_max_dx > half_x_frame,
            "widened pass (radius {AUTO_SELECT_WIDEN_RADIUS}) never reached outside the normal frame — \
             it can't recover detail a mis-centered round stepped past");
    }

    // ── find_interesting_square ─────────────────────────────────────────

    #[test]
    fn auto_select_finds_a_square_for_classic_mandelbrot() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        // A view framing the classic Mandelbrot set's boundary detail should
        // have SOME non-degenerate candidate square to pick.
        let view = View::new_square(-0.5, 0.0, 1.0);
        let result = find_interesting_square(&genome, &config, &view, true);
        assert!(result.is_some(), "expected a candidate square for a real fractal");
        let (_, _, zoom) = result.unwrap();
        assert!(zoom > view.zoom, "auto-select must zoom IN, not out");
    }

    #[test]
    fn auto_select_none_when_view_already_degenerate() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        // Zoomed absurdly far out — every candidate square is uniform
        // escaped background. Must report "nothing found" honestly.
        let view = View::new_square(0.0, 0.0, 1.0e-8);
        let result = find_interesting_square(&genome, &config, &view, true);
        assert!(result.is_none());
    }

    #[test]
    fn auto_select_zoom_matches_one_of_the_documented_scales() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let view = View::new_square(-0.5, 0.0, 1.0);
        let (_, _, zoom) = find_interesting_square(&genome, &config, &view, true).unwrap();
        // For each candidate scale s: side = cur_h*s, zoom = 4.0/side. cur_h
        // at view.zoom=1.0 is 4.0, so zoom = 4.0/(4.0*s) = view.zoom/s —
        // the winner (whichever scale had the highest detail score) must
        // land on exactly one of these, not some other value.
        let matches_a_scale = AUTO_SELECT_SCALES.iter()
            .any(|&s| (zoom - view.zoom / s).abs() < 1e-9);
        assert!(matches_a_scale, "zoom {zoom} doesn't match any AUTO_SELECT_SCALES-derived value");
        assert!(zoom > view.zoom, "auto-select must zoom IN, not out");
    }

    // ── Double-double (DD) deep-zoom regression ─────────────────────────
    //
    // Root cause of "DD never worked": `View::bounds()` computes
    // `cx - half_x` in plain f64. Once `half_x` (the view's own
    // half-width) drops below ~0.5 ULP of `cx`, `cx - half_x` and
    // `cx + half_x` both round back to exactly `cx` — the window silently
    // collapses to zero width, precisely at the zoom depth DD exists to
    // handle. `find_interesting_square` built its whole search window
    // from `bounds()`, so every scale's `side <= 0.0` guard tripped and
    // it always returned `None` once a user's Auto-Select zoom crossed
    // the DD threshold — auto-select (the tool this session added
    // specifically to drive deep zoom exploration) could never actually
    // reach DD-precision territory. Fixed by computing the search window
    // from the zoom-derived span (`4.0/zoom`, never a small-minus-small
    // subtraction) and adding offsets onto the view's actual `cx_dd()` /
    // `cy_dd()` via double-double `Add`.

    #[test]
    fn bounds_collapses_to_zero_width_at_deep_zoom() {
        // Documents the actual root cause: reproduces it directly against
        // View::bounds() (still used as-is at shallow zoom) so a future
        // change can't silently reintroduce the same trap elsewhere.
        let mut view = View::new_square(-0.7436438870371587, 0.13182590420531198, 1.0);
        view.zoom = 2f64.powi(60);
        let (xmin, xmax, _, _) = view.bounds();
        assert_eq!(xmin, xmax, "expected bounds() to collapse at 2^60 zoom (documents the bug)");
    }

    // ── Manual DD gate ───────────────────────────────────────────────────
    //
    // Past a certain zoom, whether a direction still has anything left to
    // resolve depends on the formula's own escape-iteration count relative
    // to depth, not just on arithmetic precision — auto-escalating to DD
    // can render a perfectly flat frame that looks like a bug but isn't.
    // `allow_dd=false` caps rendering at the plain-f64 path unconditionally,
    // regardless of what `needs_dd` would otherwise say, so the user can
    // choose when DD is worth paying for instead of it kicking in as a
    // surprise.

    #[test]
    fn allow_dd_false_stays_on_f64_path_past_the_dd_threshold() {
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        // Drill to a genuinely deep, known-good point the same way
        // `auto_select_keeps_drilling_past_the_f64_precision_limit` does —
        // a hand-picked coordinate isn't reliable here (verified elsewhere
        // this session: an arbitrary "famous" deep-zoom coordinate quoted
        // to ~16 digits stops having resolvable detail past zoom ~2^30,
        // regardless of DD, simply for not being close enough to the
        // boundary — this needs a coordinate actually reached by drilling).
        let mut view = View::new_square(-0.75, 0.1, 4.0);
        for _ in 0..25 {
            let (dx, dy, zoom) = find_interesting_square(&genome, &config, &view, true).unwrap();
            let new_cx = view.cx_dd() + Dd::from_f64(dx);
            let new_cy = view.cy_dd() + Dd::from_f64(dy);
            view = View { cx: new_cx.hi, cx_lo: new_cx.lo, cy: new_cy.hi, cy_lo: new_cy.lo, zoom, aspect: 1.0 };
        }
        assert!(needs_dd(&view, AUTO_SELECT_RES), "test setup must actually reach the DD threshold");

        let iter = config.optimization.eval_max_iter;
        let rgb_allowed = render_cpu(&genome, &config, &view, AUTO_SELECT_RES, AUTO_SELECT_RES, iter, true, false, true);
        assert!(!view_is_degenerate(&rgb_allowed),
            "allow_dd=true should resolve real detail at this known-good drilled-to depth");

        // Same view, allow_dd=false: must NOT take the DD branch — forced
        // onto the f64 path, which at this depth is fed a collapsed
        // bounds() window and renders degenerate. Confirms the toggle
        // actually gates the code path, not just a cosmetic flag.
        let rgb_capped = render_cpu(&genome, &config, &view, AUTO_SELECT_RES, AUTO_SELECT_RES, iter, true, false, false);
        assert!(view_is_degenerate(&rgb_capped),
            "allow_dd=false must render via the f64 path (degenerate here) even though DD would resolve it");
    }

    #[test]
    fn auto_select_keeps_drilling_past_the_f64_precision_limit() {
        // A single guessed deep-zoom coordinate isn't a valid test — at
        // insane zoom nearly every point is solidly inside/outside the set
        // regardless of arithmetic correctness (confirmed empirically: a
        // real seahorse-valley coordinate good to ~16 digits renders as a
        // single flat color from zoom 2^30 on, DD or no DD, because it's
        // simply not close enough to the boundary to still be "in frame"
        // that deep). The only valid test is the actual workflow: repeatedly
        // call find_interesting_square + apply the winner (exactly what
        // clicking Auto-Select N times does), and confirm it can still
        // follow real detail after crossing into DD territory — instead of
        // getting stuck the moment bounds()-based candidates collapse.
        //
        // Empirically, the pre-fix version of this exact loop drilled fine
        // through round 30 (zoom 2^55.8), then got permanently stuck at
        // round 31 when view.bounds() returned cur_w=0 (verified via a
        // standalone probe against the old bounds()-based logic before this
        // fix landed) — every subsequent round returned None forever,
        // matching the reported "auto-select stops finding anything past a
        // certain zoom" symptom exactly.
        let genome = mandelbrot_dag_genome();
        let config = default_config();
        let mut view = View::new_square(-0.75, 0.1, 4.0);
        let mut crossed_dd_threshold = false;

        for round in 0..35 {
            let result = find_interesting_square(&genome, &config, &view, true);
            let (dx, dy, zoom) = result.unwrap_or_else(|| {
                panic!("auto-select got stuck at round {round} (zoom 2^{:.1}, dd={}) \
                        — found no candidate square at all",
                    view.zoom.log2(), needs_dd(&view, AUTO_SELECT_RES));
            });
            assert!(zoom > view.zoom, "round {round}: must zoom in, not out");

            let new_cx = view.cx_dd() + nnfractals::dd::Dd::from_f64(dx);
            let new_cy = view.cy_dd() + nnfractals::dd::Dd::from_f64(dy);
            view = View { cx: new_cx.hi, cx_lo: new_cx.lo, cy: new_cy.hi, cy_lo: new_cy.lo, zoom, aspect: 1.0 };
            if needs_dd(&view, AUTO_SELECT_RES) { crossed_dd_threshold = true; }
        }

        assert!(crossed_dd_threshold, "test didn't actually reach DD-precision territory — not a valid regression check");
        assert!(view.zoom.log2() > 60.0, "expected drilling to reach at least zoom 2^60, got 2^{:.1}", view.zoom.log2());
    }
}
