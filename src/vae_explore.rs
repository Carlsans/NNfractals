//! VAE-driven per-formula recursive exploration — see the project plan
//! ("VAE-driven per-formula recursive exploration") for the full design.
//!
//! A per-formula VAE (trained from scratch on RAW escape-time fields, not
//! colormapped RGB — see `scripts/autoencoder.py`) scores candidate zones
//! by reconstruction error; the recursive drill picks the highest-error
//! (most novel/undermodeled) zone each level, "recreates it in high def,"
//! and repeats. Falls back to random selection when no VAE exists yet
//! (`vae_scorer: None`), matching the outer training loop's own bootstrap
//! (`src/bin/explorer.rs::cmd_vae_explore`).
//!
//! Two-stage zone proposal, deliberately different costs: `coarse_scan`
//! pixel-crops the ALREADY-rendered canvas buffer (no new render — the
//! canvas is at a fixed zoom, so cropping loses no information a fresh
//! render at that same zoom would add) and scores via cheap raw-field
//! entropy/edge_density/intricacy; `render_and_save_zones` is the
//! genuinely new, expensive stage — each surviving candidate needs a
//! FRESH render at its own (deeper) zoom, since a pixel-crop of the
//! shallower canvas literally cannot contain detail that only resolves at
//! higher zoom (confirmed: no "render once, crop deeper" pattern exists
//! anywhere else in this codebase, for exactly this reason).

use crate::colormap::apply_colormap;
use crate::config::Config;
use crate::explore::{self, Candidate, Logger, Metrics, ScoreMethod};
use crate::fitness;
use crate::fractal::{edge_density, field_intricacy, local_edge_contrast, local_intricacy_contrast};
use crate::genome::Genome;
use crate::io;
use crate::render_gpu;
use crate::saliency::SaliencyScorer;
use crate::vae_score::VaeScorer;
use crate::video_export::{needs_f64, render_escape_times, View};
use rand::seq::SliceRandom;
use std::path::{Path, PathBuf};

/// 4095, not 4096: `4096² = 16,777,216` is ONE WORKGROUP over
/// `render_gpu`'s single-dispatch pixel limit (`WG_SIZE * MAX_WORKGROUPS_PER_DIM`
/// = `256 * 65535` = 16,776,960 — see `recursion_level`'s `exceeds_gpu_dispatch`
/// check), while `4095² = 16,769,025` fits comfortably. This was a REAL bug,
/// not just a theoretical one: at the old 4096 default, `exceeds_gpu_dispatch`
/// was true for every single canvas render regardless of zoom, so EVERY
/// level of EVERY seed fell to the ~20-40s CPU path unconditionally — Carl
/// reported the whole search as "still slow" after the backtracking fix,
/// and this off-by-one-workgroup default (already flagged in a comment
/// right next to the check, just never acted on) turned out to be the
/// actual dominant cost, not the search algorithm. Genuinely deep zooms
/// still correctly fall to the CPU/DD tier via `needs_f64` — that part is
/// real precision necessity, not this bug.
pub const CANVAS_RES: u32 = 4095;
pub const ZONE_RES: u32 = 512;
/// Default location for a trained `scripts/train_saliency.py` checkpoint —
/// mirrors `explorer_out/last_successful_vae.pt`'s convention (one flat,
/// dataset-independent file, not per-formula bookkeeping). Used as the
/// default `--saliency-model` value (CLI) and the viewer's default field
/// value — both still gate on the file actually existing before enabling
/// anything, so a fresh checkout with no trained model yet behaves exactly
/// like before Phase 22 (grid-only search, no saliency candidates).
pub const SALIENCY_DEFAULT_MODEL_PATH: &str = "explorer_out/saliency_model.pt";
/// Fractions of the canvas view's own extent — pixel-crop geometry, not a
/// render-cost concept, so this lives here rather than in `explore.rs`
/// alongside `SCALES`/`WIDE_SCALES`.
pub const CANVAS_SCAN_SCALES: &[f64] = &[0.5, 0.25, 0.125];
/// Every crop, regardless of its own pixel size, is strided-sampled down
/// to this fixed resolution before scoring — O(this²) work per candidate
/// regardless of scale, not O(crop area) (which would make the largest
/// scan scale dominate runtime).
pub const COARSE_SAMPLE_RES: usize = 128;

#[derive(Clone, Copy)]
pub struct ZoneGate {
    pub max_intricacy: f32,
    pub min_edge_density: f32,
}

/// Position tolerance for `is_near_duplicate`, relative to the view's OWN
/// extent at its zoom (2%: two centers within 2% of the current view's
/// width of each other, at essentially the same zoom, render near-
/// identical content). "Small hysteresis" per Carl's request (2026-08-09,
/// after a real bug was found: `pick_seeds` is a deterministic function of
/// `(genome, base_view, method)`, and vae-explore's outer loop calls it
/// every iteration with the SAME unchanging base_view — with only 4
/// methods cycling in `--method mixed`, the exact same seeds — and
/// therefore the exact same drilled candidates — got rediscovered and
/// RE-SAVED every 4th iteration, with no dedup check anywhere. Confirmed
/// in practice: 66-96% of zones across every existing vae-explore pool
/// this project had built were exact-coordinate duplicates of another
/// saved zone, some single locations saved 80+ times). Tight enough to
/// only catch genuine repeats — the real repeats found were near-EXACT
/// (same value to 6 decimal places), not marginal near-misses — not so
/// tight it lets real near-duplicates slip through.
const DEDUP_POS_TOL: f64 = 0.02;
/// Relative zoom tolerance for `is_near_duplicate` (5%).
const DEDUP_ZOOM_TOL: f64 = 0.05;

/// Whether `view` is a near-duplicate of anything already in `seen` — see
/// `DEDUP_POS_TOL`/`DEDUP_ZOOM_TOL`. `seen` stores `(cx, cy, zoom)` only,
/// not full genomes: within one vae-explore run every saved zone shares
/// the same genome, so comparing the view alone is sufficient.
fn is_near_duplicate(view: &View, seen: &[(f64, f64, f64)], pos_tol: f64, zoom_tol: f64) -> bool {
    let extent = 4.0 / view.zoom;
    let pos_thresh = pos_tol * extent;
    seen.iter().any(|&(cx, cy, zoom)| {
        let zoom_ratio = (view.zoom / zoom).max(zoom / view.zoom);
        zoom_ratio <= 1.0 + zoom_tol
            && (view.cx - cx).abs() < pos_thresh
            && (view.cy - cy).abs() < pos_thresh
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectBy { MaxError, MinError, Random }

pub struct SavedZone {
    pub stem: String,
    pub view: View,
    pub metrics: Metrics,
    pub raw_png_path: PathBuf,
}

/// Pixel-space crop rectangle for one candidate position within a
/// `canvas_w`x`canvas_h` buffer — pure geometry, no image data, factored
/// out so it's unit-testable without a `Genome`/`Config`/GPU. Returns
/// `(x0, y0, side)` (top-left corner + side length in pixels), or `None`
/// if the requested crop doesn't fit this canvas (too small to be
/// meaningful, or bigger than the canvas itself).
fn crop_rect_px(canvas_w: u32, canvas_h: u32, scale: f64, dx: f64, dy: f64, px_per_unit_x: f64, px_per_unit_y: f64) -> Option<(u32, u32, u32)> {
    let side = (canvas_w as f64 * scale).round() as i64;
    if side < 2 || side as u32 > canvas_w || side as u32 > canvas_h { return None; }
    let center_x = (canvas_w as f64 / 2.0 + dx * px_per_unit_x).round() as i64;
    let center_y = (canvas_h as f64 / 2.0 + dy * px_per_unit_y).round() as i64;
    let x0 = (center_x - side / 2).clamp(0, canvas_w as i64 - side) as u32;
    let y0 = (center_y - side / 2).clamp(0, canvas_h as i64 - side) as u32;
    Some((x0, y0, side as u32))
}

/// Nearest/strided-sample a `side`x`side` square crop (top-left at
/// `x0,y0`) of `field` (row-major, `canvas_w` wide) down to
/// `out_res`x`out_res`.
fn strided_sample(field: &[f32], canvas_w: u32, canvas_h: u32, x0: u32, y0: u32, side: u32, out_res: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_res * out_res);
    for oy in 0..out_res {
        let sy = (y0 + (oy as u32 * side) / out_res as u32).min(canvas_h - 1);
        for ox in 0..out_res {
            let sx = (x0 + (ox as u32 * side) / out_res as u32).min(canvas_w - 1);
            out.push(field[(sy * canvas_w + sx) as usize]);
        }
    }
    out
}

/// Crops+scores ONE position (grid-derived or, since Phase 21's saliency
/// integration, off-grid — see `saliency_candidates`) against an already-
/// rendered canvas: pixel-crop, strided-sample down to `COARSE_SAMPLE_RES`,
/// raw-field metrics (the escape-time equivalent of what
/// `explore::field_to_candidate` computes for colormapped candidates).
/// Factored out of `coarse_scan` so both the fixed grid AND saliency-picked
/// positions score through the exact same gate-relevant metrics — a
/// saliency candidate gets no special treatment, it just gets a fair shot
/// at the same `top_k` ranking everything else competes for.
#[allow(clippy::too_many_arguments)]
fn score_position(
    canvas_field: &[f32], canvas_w: u32, canvas_h: u32, px_per_unit_x: f64, px_per_unit_y: f64,
    config: &Config, method: ScoreMethod, scale_idx: usize, scale: f64, dx: f64, dy: f64, zoom: f64,
) -> Option<Candidate> {
    let (x0, y0, side) = crop_rect_px(canvas_w, canvas_h, scale, dx, dy, px_per_unit_x, px_per_unit_y)?;
    let sample = strided_sample(canvas_field, canvas_w, canvas_h, x0, y0, side, COARSE_SAMPLE_RES);
    let max_iter = config.rendering.max_iter;
    let (local_edge, edge_contrast) = local_edge_contrast(&sample, COARSE_SAMPLE_RES, COARSE_SAMPLE_RES, crate::fractal::LOCAL_TILE_GRID);
    let (local_intric, intric_contrast) = local_intricacy_contrast(&sample, COARSE_SAMPLE_RES, COARSE_SAMPLE_RES, crate::fractal::LOCAL_TILE_GRID);
    // A candidate that reads as "uniform" over the WHOLE patch (surviving
    // `is_degenerate`'s >95%-the-same check) can still contain a small,
    // genuinely interesting feature — a crisp circular boundary, say —
    // that a whole-patch statistic dilutes into invisibility (Carl,
    // 2026-08-11: confirmed real on a genome with multiple visually-
    // attractive circular structures the coarse scan never picked, even
    // though the VAE itself rated one meaningfully above-median
    // interesting once actually asked via a manual mark). Only treat it
    // as a genuine dead end if NEITHER the whole-patch check NOR any
    // single tile shows real content.
    let degenerate = fitness::is_degenerate(&sample) && local_edge < crate::fractal::LOCAL_EDGE_DEGENERATE_OVERRIDE;
    // `m` carries the TRUE whole-patch values, ALWAYS — this is what
    // `render_and_save_zones`'s `max_intricacy`/`min_edge_density` gate
    // (and `rank_unsaved`'s, and `level_scanning`'s logged `gate_pass`)
    // reads directly, and MUST stay the honest measurement: a real bug,
    // caught by a live re-test rather than shipped, was writing the
    // boosted values straight into `Metrics` here, which inflated
    // intricacy past `max_intricacy` for candidates that would otherwise
    // have passed, and dropped 17 of 20 seeds to zero saved zones in one
    // test run. The contrast-gated boost below affects ONLY the separate
    // numeric `score` used for RANKING candidates against each other —
    // never gating, which stays a pass/fail on the real content.
    let m = Metrics {
        entropy: if degenerate { 0.0 } else { fitness::entropy_score_fast(&sample, max_iter) },
        edge_density: edge_density(&sample, COARSE_SAMPLE_RES, COARSE_SAMPLE_RES),
        intricacy: field_intricacy(&sample, COARSE_SAMPLE_RES, COARSE_SAMPLE_RES),
        degenerate,
    };
    let mut score = explore::score(&m, method);
    // Rescuing the RANKING (not just the degenerate gate) is gated on
    // CONTRAST, not a bare local max — an earlier version blended raw
    // `local_edge_density` straight into the score and caused a REAL,
    // observed regression (Carl, 2026-08-11, watching a live run get
    // stuck reaching only 32x deeper than its base zoom after 500+ saved
    // zones): a wide/shallow crop has a much higher a-priori chance of
    // containing SOME locally-okay tile purely because it covers more
    // ground, so rewarding "best tile anywhere" made every level
    // systematically prefer staying shallow. `contrast` (max minus the
    // tiles' own mean) tells the two cases apart — see
    // `fractal::tile_stats`'s doc comment — and only fires for a genuine
    // standout, confirmed against a second real case (Carl, same day: a
    // large flat disc with a thin detailed ring at its edge, invisible to
    // every whole-patch metric AND to the plain degenerate-only rescue,
    // since the only candidates whose crop even reached it were wide
    // enough to dilute it back down to an uncompetitive score). Rescored
    // via a SEPARATE boosted Metrics, `.max()`'d against the real score —
    // that temporary struct never escapes this function, so it can't leak
    // into gating the way the earlier bug did.
    if edge_contrast >= crate::fractal::LOCAL_CONTRAST_MIN || intric_contrast >= crate::fractal::LOCAL_CONTRAST_MIN {
        let boosted = Metrics { edge_density: m.edge_density.max(local_edge), intricacy: m.intricacy.max(local_intric), ..m };
        score = score.max(explore::score(&boosted, method));
    }
    Some(Candidate { scale_idx, dx, dy, zoom, metrics: m, score })
}

/// Coarse, no-new-render zone proposal: lays out crop positions via the
/// EXISTING `sweep_positions` grid×scale geometry (unchanged — same
/// proven layout `sweep`/`sweep_cpu` use), scores each via `score_position`.
/// Returns `explore::Candidate` directly (not a separate type) so the
/// caller can reuse `explore::rank_by_zscore` unchanged.
pub(crate) fn coarse_scan(
    canvas_field: &[f32], canvas_w: u32, canvas_h: u32, canvas_view: &View,
    config: &Config, method: ScoreMethod,
) -> Vec<Candidate> {
    let cur_w = 4.0 / canvas_view.zoom * canvas_view.aspect;
    let cur_h = 4.0 / canvas_view.zoom;
    let positions = explore::sweep_positions(cur_w, cur_h, 1.0, f64::INFINITY, CANVAS_SCAN_SCALES);
    let px_per_unit_x = canvas_w as f64 / cur_w;
    let px_per_unit_y = canvas_h as f64 / cur_h;

    positions.into_iter()
        .filter_map(|p| score_position(
            canvas_field, canvas_w, canvas_h, px_per_unit_x, px_per_unit_y,
            config, method, p.scale_idx, CANVAS_SCAN_SCALES[p.scale_idx], p.dx, p.dy, p.zoom,
        ))
        .collect()
}

/// How many distinct heatmap peaks `saliency_candidates` turns into extra
/// candidates, each tried at every `CANVAS_SCAN_SCALES` tier (so 8 peaks =
/// up to 24 extra cheap crop-scores — the grid stays the primary, proven
/// candidate source; this only ADDS off-grid alternatives it might have
/// missed, gated through the exact same `score_position`/rank/gate
/// pipeline as everything else).
const SALIENCY_TOP_PEAKS: usize = 8;
/// Minimum separation between kept peaks, as a fraction of the heatmap's
/// own smaller dimension — cheap non-max suppression so 8 "peaks" aren't
/// all the same one blurry hot region.
const SALIENCY_PEAK_MIN_SEP_FRAC: f64 = 0.1;

/// Turns a predicted saliency heatmap into extra candidate positions:
/// simple greedy non-max suppression (sort cells by value, keep a cell
/// only if it's not within `SALIENCY_PEAK_MIN_SEP_FRAC` of an already-kept
/// one) to get `SALIENCY_TOP_PEAKS` distinct hot spots, each converted to
/// a (dx, dy) offset using the SAME row/col-to-coordinate convention
/// `render_escape_times`/training data generation (`cmd_saliency_data` in
/// `explorer.rs`) both use, then scored via `score_position` at every
/// `CANVAS_SCAN_SCALES` tier (the heatmap predicts WHERE, not what zoom —
/// trying every tier lets the gate/rank pick whichever depth actually
/// looks good there, same as a grid candidate could).
fn saliency_candidates(
    heatmap: &crate::saliency::Heatmap, canvas_field: &[f32], canvas_w: u32, canvas_h: u32,
    canvas_view: &View, config: &Config, method: ScoreMethod,
) -> Vec<Candidate> {
    let cur_w = 4.0 / canvas_view.zoom * canvas_view.aspect;
    let cur_h = 4.0 / canvas_view.zoom;
    let px_per_unit_x = canvas_w as f64 / cur_w;
    let px_per_unit_y = canvas_h as f64 / cur_h;

    // Excludes the outermost ring of cells: a real, consistently-observed
    // artifact in the trained net (verified 2026-08-10 against the actual
    // first checkpoint, not theoretical) — zero-padding in every stride-2
    // conv layer concentrates a spurious high-activation bias at the
    // literal image corners/edges, strong enough that cell (0,0) was the
    // single highest-value cell in EVERY sample checked, regardless of
    // actual canvas content. The interior cells' point-wise regression
    // was independently confirmed strong (0.979 Pearson correlation
    // against real held-out VAE reconstruction-error labels) — this is
    // purely a border effect, not evidence the whole model is unreliable.
    let mut cells: Vec<(usize, usize, f32)> = (0..heatmap.height)
        .flat_map(|row| (0..heatmap.width).map(move |col| (row, col)))
        .filter(|&(row, col)| row > 0 && col > 0 && row + 1 < heatmap.height && col + 1 < heatmap.width)
        .map(|(row, col)| (row, col, heatmap.at(row, col)))
        .collect();
    cells.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let min_sep = SALIENCY_PEAK_MIN_SEP_FRAC * heatmap.height.min(heatmap.width) as f64;
    let mut peaks: Vec<(usize, usize)> = Vec::new();
    for &(row, col, _) in &cells {
        if peaks.len() >= SALIENCY_TOP_PEAKS { break; }
        let too_close = peaks.iter().any(|&(pr, pc)| {
            let dr = row as f64 - pr as f64;
            let dc = col as f64 - pc as f64;
            (dr * dr + dc * dc).sqrt() < min_sep
        });
        if !too_close { peaks.push((row, col)); }
    }

    let mut out = Vec::with_capacity(peaks.len() * CANVAS_SCAN_SCALES.len());
    for (row, col) in peaks {
        // Cell-center fraction, same convention as `cmd_saliency_data`'s
        // px/py labels: col 0 = xmin, row 0 = ymin.
        let px = (col as f64 + 0.5) / heatmap.width as f64;
        let py = (row as f64 + 0.5) / heatmap.height as f64;
        let dx = (px - 0.5) * cur_w;
        let dy = (py - 0.5) * cur_h;
        for (scale_idx, &scale) in CANVAS_SCAN_SCALES.iter().enumerate() {
            let zoom = 4.0 / (cur_h * scale);
            if let Some(c) = score_position(
                canvas_field, canvas_w, canvas_h, px_per_unit_x, px_per_unit_y,
                config, method, scale_idx, scale, dx, dy, zoom,
            ) {
                out.push(c);
            }
        }
    }
    out
}

/// The genuinely new, expensive stage: fresh-renders each surviving
/// candidate at its own (deeper) view, batching every GPU/f32-eligible
/// candidate into ONE dispatch (mirrors `sweep()`'s own batching trick —
/// a 512x512 render is ~1MB, so up to ~120 fit under the 128MB Vulkan
/// single-dispatch limit `sweep()` is itself bounded by), falling back to
/// one-dispatch-per-candidate only for the ones that individually need
/// the CPU/DD tier. Gates on the CHEAP coarse metrics already computed by
/// `coarse_scan` — no second gate after the precise render, gating stays
/// single-pass. Saves three files per surviving zone with the same
/// append-mode numbering `cmd_pool` uses.
///
/// `seen` is the running dedup registry (see `is_near_duplicate`) — a
/// candidate that matches an entry already in it is dropped BEFORE the
/// expensive fresh-render step, not just before saving, so a rejected
/// duplicate costs nothing beyond the already-sunk coarse_scan pass.
/// Checked and updated cumulatively as candidates are accepted, so
/// near-duplicates WITHIN this same batch are caught too, not just
/// against zones saved in a prior call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_and_save_zones(
    genome: &Genome, config: &Config, canvas_view: &View, candidates: &[Candidate],
    gate: &ZoneGate, out_dir: &Path, next_stem: &mut usize, log: &mut Logger,
    seen: &mut Vec<(f64, f64, f64)>,
) -> Vec<SavedZone> {
    let gated: Vec<&Candidate> = candidates.iter()
        .filter(|c| !c.metrics.degenerate && c.metrics.intricacy <= gate.max_intricacy && c.metrics.edge_density >= gate.min_edge_density)
        .collect();
    if gated.is_empty() { return Vec::new(); }

    let mut passing: Vec<&Candidate> = Vec::new();
    let mut views: Vec<View> = Vec::new();
    for c in gated {
        let view = explore::apply_offset(canvas_view, c.dx, c.dy, c.zoom);
        if is_near_duplicate(&view, seen, DEDUP_POS_TOL, DEDUP_ZOOM_TOL) { continue; }
        seen.push((view.cx, view.cy, view.zoom));
        passing.push(c);
        views.push(view);
    }
    if passing.is_empty() { return Vec::new(); }

    let gpu_idx: Vec<usize> = (0..views.len()).filter(|&i| !needs_f64(&views[i], ZONE_RES)).collect();
    let cpu_idx: Vec<usize> = (0..views.len()).filter(|&i| needs_f64(&views[i], ZONE_RES)).collect();

    let mut fields: Vec<Option<Vec<f32>>> = (0..views.len()).map(|_| None).collect();
    if !gpu_idx.is_empty() {
        let items: Vec<render_gpu::DagItem> = gpu_idx.iter().map(|_| render_gpu::dag_item(genome)).collect();
        let bounds: Vec<(f32, f32, f32, f32)> = gpu_idx.iter().map(|&i| {
            let v = &views[i];
            let half = (2.0 / v.zoom) as f32;
            let cx = v.cx as f32;
            let cy = v.cy as f32;
            (cx - half, cx + half, cy - half, cy + half)
        }).collect();
        let batch = render_gpu::render_batch_dag(&items, &bounds, ZONE_RES, ZONE_RES, config.rendering.max_iter);
        for (k, &i) in gpu_idx.iter().enumerate() { fields[i] = Some(batch[k].clone()); }
    }
    for &i in &cpu_idx {
        let field = render_escape_times(genome, config, &views[i], ZONE_RES, ZONE_RES, config.rendering.max_iter, true, true);
        fields[i] = Some(field);
    }

    let mut saved = Vec::new();
    for (i, cand) in passing.iter().enumerate() {
        let Some(field) = fields[i].take() else { continue };
        let view = views[i].clone();
        let stem = format!("zone_{next_stem:04}");
        *next_stem += 1;

        let mut g = genome.clone();
        g.view_cx = view.cx as f32;
        g.view_cy = view.cy as f32;
        g.view_zoom = view.zoom as f32;
        let snapped = View::new_square(g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64);

        let nn_path = out_dir.join(format!("{stem}.nn"));
        let raw_png_path = out_dir.join(format!("{stem}_raw.png"));
        let preview_png_path = out_dir.join(format!("{stem}.png"));

        io::save_genome(&g, &nn_path).expect("save genome");
        io::save_raw_field(&field, ZONE_RES, ZONE_RES, config.rendering.max_iter, &raw_png_path).expect("save raw field");
        let rgb = apply_colormap(&field, config.rendering.max_iter, &config.rendering.colormap);
        io::save_png(&rgb, ZONE_RES, ZONE_RES, &preview_png_path).expect("save preview png");

        log.log(&serde_json::json!({
            "event": "zone_saved", "stem": stem, "cx": g.view_cx, "cy": g.view_cy, "zoom": g.view_zoom,
            "entropy": cand.metrics.entropy, "edge_density": cand.metrics.edge_density, "intricacy": cand.metrics.intricacy,
        }));

        saved.push(SavedZone { stem, view: snapped, metrics: cand.metrics, raw_png_path });
    }
    saved
}

/// Ranks a level's saved zones best-first: by VAE reconstruction error
/// (scorer present and `select_by` isn't `Random`), or shuffled otherwise —
/// preserving the old single-winner code's "or randomly if no VAE yet"
/// behavior, just exposing the WHOLE ranked list instead of only the
/// front. Zones the scorer individually failed to score (sidecar hiccup)
/// are silently dropped rather than kept at an arbitrary position — same
/// graceful-degrade spirit as the rest of this sidecar's error handling —
/// UNLESS every single one failed, in which case falling back to a
/// shuffled full list beats returning nothing.
fn rank_saved(vae_scorer: Option<&mut VaeScorer>, saved: &[SavedZone], select_by: SelectBy) -> Vec<View> {
    if let Some(scorer) = vae_scorer
        && select_by != SelectBy::Random {
        let mut scored: Vec<(&SavedZone, f32)> = saved.iter()
            .filter_map(|z| scorer.score_blocking(z.raw_png_path.clone()).map(|e| (z, e)))
            .collect();
        if !scored.is_empty() {
            match select_by {
                SelectBy::MaxError => scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
                _ => scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
            }
            return scored.into_iter().map(|(z, _)| z.view.clone()).collect();
        }
    }
    let mut out: Vec<View> = saved.iter().map(|z| z.view.clone()).collect();
    out.shuffle(&mut rand::rng());
    out
}

/// Ranks this level's gate-passing-but-UNSAVED candidates (rejected only by
/// dedup, since `render_and_save_zones` gates on the same predicate before
/// even attempting a render) best-first by their coarse score. This is what
/// `recursion_level` falls back to when nothing got saved but the level
/// wasn't a genuine dead end.
fn rank_unsaved(seed_view: &View, top: &[Candidate], gate: &ZoneGate) -> Vec<View> {
    let mut alts: Vec<&Candidate> = top.iter()
        .filter(|c| !c.metrics.degenerate && c.metrics.intricacy <= gate.max_intricacy && c.metrics.edge_density >= gate.min_edge_density)
        .collect();
    alts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    alts.into_iter().map(|c| explore::apply_offset(seed_view, c.dx, c.dy, c.zoom)).collect()
}

pub struct RecursionOpts {
    pub top_k: usize,
    pub canvas_res: u32,
    pub select_by: SelectBy,
}

/// Writes the predicted heatmap as a normalized (min-max stretched)
/// grayscale PNG so it's small/cheap to load — the viewer applies its own
/// color tint when overlaying it, this is deliberately not pre-colorized.
/// Overwrites the SAME filename every level (not accumulated per-level
/// files): this is a live "what is the model looking at right now" debug
/// view, not an archive.
fn save_saliency_heatmap_png(heatmap: &crate::saliency::Heatmap, path: &Path) {
    let (min, max) = heatmap.data.iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let range = (max - min).max(1e-6);
    let rgb: Vec<u8> = heatmap.data.iter()
        .flat_map(|&v| { let t = (((v - min) / range) * 255.0).clamp(0.0, 255.0) as u8; [t, t, t] })
        .collect();
    let _ = io::save_png(&rgb, heatmap.width as u32, heatmap.height as u32, path);
}

/// One recursion level: render the seed view fresh at `canvas_res` ("recreate
/// it in high def"), coarse-scan for candidate zones, precisely render+save
/// the top-K. Returns EVERY viable continuation for this level, ranked
/// best-first — not just a single winner: `recursive_drill` treats the rest
/// as a backtracking stack, so if descending into the best option dead-ends
/// several levels deeper, it "unzooms" back to the next-best sibling here
/// instead of abandoning the whole seed (Carl's request, 2026-08-10 — a
/// fixed single greedy path per seed had no way to recover from a bad
/// early choice). An empty return means a genuine dead end: nothing even
/// passed the coarse gate at this level.
///
/// `saliency`, when `Some`, ADDS extra off-grid candidates from a trained
/// `SaliencyNet`'s predicted heatmap (`saliency_candidates`) alongside the
/// fixed grid `coarse_scan` already produces — the grid stays the primary,
/// proven candidate source, this only gives the model's off-grid guesses a
/// fair shot at the same gate/rank everything else goes through. `None`
/// (no model trained yet, or `--saliency-model` not passed) behaves
/// EXACTLY as before this existed — the grid alone.
#[allow(clippy::too_many_arguments)]
pub fn recursion_level(
    genome: &Genome, config: &Config, seed_view: &View, vae_scorer: Option<&mut VaeScorer>,
    saliency: Option<&mut SaliencyScorer>,
    method: ScoreMethod, opts: &RecursionOpts, gate: &ZoneGate, out_dir: &Path,
    next_stem: &mut usize, log: &mut Logger, seen: &mut Vec<(f64, f64, f64)>,
) -> Vec<View> {
    // Two independent reasons the GPU/f32 batch path can't take this
    // render: `needs_f64` (precision — deep zoom exceeds f32's safe
    // range) and a dispatch-size ceiling `needs_f64` knows nothing about
    // (a single render's pixel count exceeds what one compute dispatch
    // can cover — see `render_gpu::MAX_WORKGROUPS_PER_DIM`'s doc comment
    // and `CANVAS_RES`'s own doc comment for why the default is 4095, not
    // 4096). Either forces the same CPU/f64 tier — slower, but with no
    // such ceiling (`render_escape_times`'s CPU path is plain rayon over
    // however many pixels are asked for).
    let pixel_count = opts.canvas_res as u64 * opts.canvas_res as u64;
    let gpu_dispatch_limit = render_gpu::WG_SIZE as u64 * render_gpu::MAX_WORKGROUPS_PER_DIM as u64;
    let exceeds_gpu_dispatch = pixel_count > gpu_dispatch_limit;
    let use_f64 = needs_f64(seed_view, opts.canvas_res) || exceeds_gpu_dispatch;
    if use_f64 {
        let why = if exceeds_gpu_dispatch { "exceeds the GPU's single-dispatch pixel limit" } else { "needs DD/f64 precision at this zoom" };
        eprintln!("vae-explore: canvas render at zoom={:.3e} {why} — this level will be slow (CPU tier)", seed_view.zoom);
    }
    let start = std::time::Instant::now();
    let canvas_field = render_escape_times(genome, config, seed_view, opts.canvas_res, opts.canvas_res, config.rendering.max_iter, use_f64, true);
    if start.elapsed().as_secs_f32() > 5.0 {
        eprintln!("vae-explore: canvas render took {:.1}s (zoom={:.3e}) — consider a smaller --canvas-res for faster iteration", start.elapsed().as_secs_f32(), seed_view.zoom);
    }

    let mut candidates = coarse_scan(&canvas_field, opts.canvas_res, opts.canvas_res, seed_view, config, method);
    if let Some(scorer) = saliency {
        let canvas_png = out_dir.join("_saliency_canvas.png");
        if io::save_raw_field(&canvas_field, opts.canvas_res, opts.canvas_res, config.rendering.max_iter, &canvas_png).is_ok()
            && let Some(heatmap) = scorer.score_blocking(canvas_png) {
            candidates.extend(saliency_candidates(&heatmap, &canvas_field, opts.canvas_res, opts.canvas_res, seed_view, config, method));
            save_saliency_heatmap_png(&heatmap, &out_dir.join("saliency_heatmap_latest.png"));
        }
    }
    let ranked = explore::rank_by_zscore(&candidates, 0.0);
    let top: Vec<Candidate> = ranked.into_iter().take(opts.top_k).collect();

    // Logged BEFORE the (possibly slow, tens-of-seconds-per-candidate on
    // the CPU/DD tier) precise render below, so a live log-tailer — the
    // viewer's Explore Options preview — can show this level's whole
    // candidate set the moment it's known, not only once zones are
    // actually saved to disk. `gate_pass` mirrors the exact predicate
    // `render_and_save_zones` gates on, MINUS the dedup check (`seen` is
    // only resolved during that render, and checking it here would mean
    // computing `apply_offset` twice) — so a candidate marked `gate_pass:
    // true` here can still end up not saved if it turns out to be a
    // near-duplicate. Close enough for a debug visualization, not a
    // promise of the exact outcome.
    log.log(&serde_json::json!({
        "event": "level_scanning",
        "seed_cx": seed_view.cx, "seed_cy": seed_view.cy, "seed_zoom": seed_view.zoom,
        "candidates": top.iter().map(|c| {
            let v = explore::apply_offset(seed_view, c.dx, c.dy, c.zoom);
            serde_json::json!({
                "cx": v.cx, "cy": v.cy, "zoom": v.zoom,
                "gate_pass": !c.metrics.degenerate
                    && c.metrics.intricacy <= gate.max_intricacy
                    && c.metrics.edge_density >= gate.min_edge_density,
            })
        }).collect::<Vec<_>>(),
    }));

    let saved = render_and_save_zones(genome, config, seed_view, &top, gate, out_dir, next_stem, log, seen);
    if !saved.is_empty() {
        return rank_saved(vae_scorer, &saved, opts.select_by);
    }

    // Every candidate this level was gate-rejected OR a near-duplicate of
    // already-saved content (`seen`) — don't let the whole recursion path
    // die here. On a resumed/append-mode run especially, a seed's shallow
    // FIRST level frequently re-lands on already-well-covered territory
    // (existing zones give decent shallow-zoom coverage); the level ONE
    // deeper from that same spot is a genuinely different, usually-fresh
    // position. Fall through to the gate-passing candidates (dedup or not)
    // ranked best-first, purely to keep drilling toward that deeper
    // territory — `apply_offset` is pure geometry, no render/save cost, so
    // this costs nothing beyond the coarse_scan pass already paid for.
    // An empty result (nothing even passed the gate) is a genuine dead end.
    rank_unsaved(seed_view, &top, gate)
}

/// Max total levels a single seed's drill will visit, as a multiple of
/// `depth`. A plain single-path descent costs exactly `depth` canvas
/// renders; letting a dead end backtrack to a sibling (see
/// `recursion_level`'s doc comment) can cost more, so this bounds the
/// worst case (every branch dead-ending immediately) instead of letting
/// one unlucky seed eat the whole iteration's time budget. In the common
/// case — the greedy best-first path just works, no backtracking needed —
/// actual cost stays exactly `depth`, same as before this existed.
const BACKTRACK_BUDGET_FACTOR: usize = 3;
/// How many ranked siblings from one level get kept as backtrack
/// candidates, capped well below `top_k` so a level with many gate-passing
/// candidates doesn't itself cause a combinatorial blow-up.
const MAX_ALTERNATES: usize = 3;

/// DFS with backtracking, up to `depth` levels deep per path: descends into
/// each level's best candidate (see `recursion_level`), but when a branch
/// dead-ends, "unzooms" back to the next-best untried sibling — possibly
/// several levels shallower — instead of abandoning the whole seed the way
/// a single greedy path did. Returns the number of zones saved across every
/// branch visited.
#[allow(clippy::too_many_arguments)]
pub fn recursive_drill(
    genome: &Genome, config: &Config, seed_view: View, depth: usize, mut vae_scorer: Option<&mut VaeScorer>,
    mut saliency: Option<&mut SaliencyScorer>,
    method: ScoreMethod, opts: &RecursionOpts, gate: &ZoneGate, out_dir: &Path,
    next_stem: &mut usize, log: &mut Logger, seen: &mut Vec<(f64, f64, f64)>,
) -> usize {
    let start_stem = *next_stem;
    // (levels still allowed to descend from this view, the view itself).
    // LIFO: a sibling is pushed at the SAME remaining budget as the winner
    // it lost to (trying it is going sideways, not deeper), then the
    // winner is pushed last so it's always explored before any sibling —
    // best-first, with backtracking only kicking in once a branch is
    // actually exhausted.
    let mut stack: Vec<(usize, View)> = vec![(depth, seed_view)];
    let budget = depth.saturating_mul(BACKTRACK_BUDGET_FACTOR).max(depth);
    let mut visited = 0usize;
    while let Some((remaining, view)) = stack.pop() {
        if remaining == 0 || visited >= budget { continue; }
        visited += 1;
        let mut candidates = recursion_level(
            genome, config, &view, vae_scorer.as_deref_mut(), saliency.as_deref_mut(),
            method, opts, gate, out_dir, next_stem, log, seen,
        );
        if candidates.is_empty() { continue; } // genuine dead end — the stack backtracks on its own
        let winner = candidates.remove(0);
        for alt in candidates.into_iter().take(MAX_ALTERNATES).rev() {
            stack.push((remaining, alt));
        }
        stack.push((remaining - 1, winner));
    }
    *next_stem - start_stem
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig, RenderingConfig};

    fn test_config() -> Config {
        Config {
            dedup: DedupConfig::default(), mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig { default_width: 512, default_height: 512, max_iter: 100, bailout: 4.0,
                colormap: "turbo".into(), view_x_min: -2.0, view_x_max: 2.0, view_y_min: -2.0, view_y_max: 2.0 },
            optimization: OptimizationConfig { population_size: 40, elitism_count: 6, mutation_rate: 0.20, mutation_scale: 0.08,
                eval_width: 64, eval_height: 64, eval_max_iter: 128, restart_after_gens: 30, novelty_weight: 0.45,
                novelty_k: 5, archive_size: 150, self_replication_weight: 0.35, fractal_recursion_weight: 0.35,
                recursion_pred_weight: 0.60, formula_diversity_weight: 0.30, clip_pred_weight: 0.50,
                formula_system: "dag".to_string(), max_nodes: 14, max_depth: 5, ood_weight: 0.0, pref_weight: 0.4,
                seed_pref_weight: 3.0, musiq_weight: 0.25, pref_elite_count: 4, archive_random_ratio: 0.30,
                duplicate_penalty_weight: 0.50, archive_seeding_enabled: false, angle_structure_weight: 0.0, img_novelty_weight: 0.0 },
            output: OutputConfig { save_dir: "./fractals".into(), population_dir: "./populations".into(),
                min_entropy_prefilter: 0.42, max_entropy_prefilter: 0.65, min_clip_score: 0.512, min_laion_score: 5.30,
                min_beauty: 0.35, min_save_distance: 0.04, min_ensemble: 4.6, min_musiq: 30.0, min_pref: 0.45 },
        }
    }

    #[test]
    fn crop_rect_stays_in_bounds_at_every_scale() {
        let canvas = 4096u32;
        let px_per_unit = canvas as f64 / 4.0; // cur_w = cur_h = 4.0 at zoom=1
        for &scale in CANVAS_SCAN_SCALES {
            let (x0, y0, side) = crop_rect_px(canvas, canvas, scale, 0.0, 0.0, px_per_unit, px_per_unit)
                .unwrap_or_else(|| panic!("center crop at scale {scale} should be valid"));
            assert!(x0 + side <= canvas, "crop must not exceed canvas width");
            assert!(y0 + side <= canvas, "crop must not exceed canvas height");
            assert_eq!(side, (canvas as f64 * scale).round() as u32);

            // Near an edge — must clamp into bounds, not go negative/overflow.
            let (x0e, y0e, sidee) = crop_rect_px(canvas, canvas, scale, 1.99, 1.99, px_per_unit, px_per_unit)
                .expect("near-edge crop should still be valid (clamped)");
            assert!(x0e + sidee <= canvas);
            assert!(y0e + sidee <= canvas);
        }
    }

    #[test]
    fn crop_rect_rejects_degenerate_sizes() {
        assert!(crop_rect_px(4096, 4096, 0.0001, 0.0, 0.0, 1024.0, 1024.0).is_none(), "a sub-2px crop must be rejected");
    }

    #[test]
    fn coarse_scan_distinguishes_flat_from_structured() {
        let canvas = 256u32;
        let flat: Vec<f32> = vec![50.0; (canvas * canvas) as usize];
        let mut checker = vec![0.0f32; (canvas * canvas) as usize];
        for y in 0..canvas {
            for x in 0..canvas {
                let block = (x / 4 + y / 4) % 2;
                checker[(y * canvas + x) as usize] = if block == 0 { 0.0 } else { 90.0 };
            }
        }
        let view = View::new_square(0.0, 0.0, 1.0);
        let config = test_config();

        let flat_candidates = coarse_scan(&flat, canvas, canvas, &view, &config, ScoreMethod::EdgeDensity);
        let checker_candidates = coarse_scan(&checker, canvas, canvas, &view, &config, ScoreMethod::EdgeDensity);

        assert!(!flat_candidates.is_empty() && !checker_candidates.is_empty());
        assert!(flat_candidates.iter().all(|c| c.metrics.degenerate), "a perfectly flat field must score every candidate degenerate");

        let mean = |cs: &[Candidate]| cs.iter().map(|c| c.metrics.edge_density).sum::<f32>() / cs.len() as f32;
        let (flat_edge, checker_edge) = (mean(&flat_candidates), mean(&checker_candidates));
        assert!(checker_edge > flat_edge, "checkerboard ({checker_edge}) should score higher edge_density than a flat field ({flat_edge})");
    }
}
