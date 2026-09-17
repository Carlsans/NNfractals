//! Fitness/quality metrics for ray-marched quaternion DAG fractals — the
//! 3D-surface analog of `fitness.rs`'s 2D escape-time `beauty_score_full`.
//!
//! Grounded directly against real examples Carl pointed at, across two
//! rounds — the first round's metric shipped, got used to pick a batch,
//! and Carl caught a real gap in it by eye that this module now corrects.
//!
//! **Round 1** — Mandelbulb (the reference for "interesting") vs. genome
//! `60e94f4f394bbdb2` ("not interesting"):
//!
//! | metric                          | Bulb  | 60e94 |
//! |----------------------------------|-------|-------|
//! | hit-fraction (frame coverage)    | 0.943 | 0.255 |
//! | interior-fraction of hit pixels  | 0.948 | 0.114 |
//! | naive gradient/edge density      | 3.52  | 10.67 (higher!) |
//!
//! `solidity` (interior-fraction) was the clean discriminator; naive
//! edge/gradient density was a TRAP (higher for the boring genome, since
//! thin threads create lots of edge-against-background contrast per
//! pixel) — any richness metric here stays restricted to interior pixels
//! so it can't reward stringy noise.
//!
//! **Round 2** — after a batch picked by round-1 scoring, Carl reported:
//! *"all the fractal mostly have a spherical shape, never letting see
//! their inside... compared to the mandelbulb, they are chaotic and not
//! that interesting."* Direct comparison of two genomes that scored
//! ALMOST IDENTICALLY under round-1 metrics (`a6598cb3b21c491b` ≈ 0.93,
//! `8e24fe3bf318e6e3` ≈ 0.93), rendered with the SAME properly-framed
//! camera (this round also fixed a camera-framing bug — see
//! `explorer.rs`'s `score_probe_cameras`), showed why: `a6598cb3` is a
//! genuinely organic asymmetric form with real cavities/notches, while
//! `8e24fe3b` is a near-perfect smooth SPHERE with only surface-texture
//! noise. Round 1 had nothing that could tell these apart — `solidity`,
//! `coverage`, `shading_richness` are all about whether the SURFACE looks
//! rich, not whether the overall FORM (silhouette) is a sphere or
//! something structured. `silhouette_irregularity` (new) fixes this:
//! coefficient of variation of the silhouette's radius as a function of
//! angle around its centroid — near 0 for a circle/sphere, large for a
//! silhouette with real protrusions/notches. Measured on exactly this
//! pair (properly framed): confirms the split (see this module's own
//! tests for the calibrated values).
//!
//! `silhouette_irregularity` itself needed two passes to get right.
//! First cut used the raw hit-mask — and genome `60e94f4f394bbdb2`
//! (round 1's own "not interesting" example, a sparse cluster of thin
//! comb-like teeth) scored the HIGHEST irregularity of every genome
//! checked, because one stray thread reaching out in some direction
//! inflates that bucket's radius just as much as a real solid wing would.
//! Restricting to INTERIOR pixels (hit AND all 8 neighbors hit) helped
//! but wasn't sufficient on its own: each comb tooth is wide enough to
//! have its own interior pixels, so a genome made of several disconnected
//! blobs at different distances from their shared centroid still read as
//! "high variance" — scattered clutter, not one irregular boundary. The
//! fix that actually worked: restrict to the LARGEST CONNECTED component
//! of interior pixels (`largest_connected_component`, 4-connected flood
//! fill) before measuring radius-vs-angle — see that function's docs and
//! this module's tests for the before/after on `60e94` specifically.
//!
//! `anisotropy_score` (`quat_dag.rs`) is the cheapest component (no
//! rendering at all), reusing this project's own earlier discovery that
//! several hand-built formulas (Mandelbrot/Tricorn/Cubic/Quartic/Celtic)
//! are exactly spherically symmetric under `TimeAxis::R` — a genome whose
//! DAG only ever rescales its vector part's existing direction inherits
//! the same degeneracy. Note this is a NECESSARY but not SUFFICIENT
//! condition for looking spherical — `8e24fe3b` has real (if textured)
//! direction-dependence yet still renders as a sphere, which is exactly
//! why `silhouette_irregularity` had to be added as its own component
//! rather than assumed to follow from anisotropy.
//!
//! **Calibration is a first pass**, same as `fitness.rs`'s own documented
//! recalibration history — this only needs to RANK genomes well enough to
//! beat random selection, not hit an absolute target. Revisit weights and
//! targets in `QuatFitnessBreakdown::total` once more genomes have been
//! scored and eyeballed against their scores (as happened here once
//! already).

use crate::formula::OpNode;

/// Per-component breakdown, mirroring `fitness::BeautyBreakdown`'s shape.
/// All fields are in `[0, 1]` (higher is always better).
#[derive(Clone, Copy, Debug, Default)]
pub struct QuatFitnessBreakdown {
    /// From `quat_dag::anisotropy_score` — 0 for a formula that's
    /// mathematically guaranteed to look the same from every angle.
    /// Necessary but NOT sufficient for a non-spherical silhouette — see
    /// `silhouette_irregularity`.
    pub anisotropy: f32,
    /// Fraction of probe-view pixels that hit the surface, one component
    /// of `view_metrics`'s output, averaged across views. Scored against
    /// a target rather than used raw — both near-empty (nothing to see)
    /// and 100%-filled-with-no-silhouette views are worse than a healthy
    /// partial fill.
    pub coverage: f32,
    /// What fraction of hit pixels are themselves surrounded by hit
    /// pixels (not near-background) — distinguishes a solid volumetric
    /// object from scattered thin filaments. See module docs for the
    /// measured Bulb-vs-60e94 gap.
    pub solidity: f32,
    /// Shading (surface-normal/lighting) variation, restricted to
    /// interior (solid) pixels only — rich curvature detail on an
    /// actually-solid surface, not edge contrast from thin threads.
    pub shading_richness: f32,
    /// Histogram entropy of escape-time (color) values over hit pixels —
    /// same construction as `fitness::beauty_score_full`'s `color_entropy`
    /// (32-bin Shannon entropy, normalized by log2(bins)), applied to the
    /// ray-marched color-probe channel instead of a 2D escape-time grid.
    pub color_entropy: f32,
    /// How much the silhouette's outer radius varies with angle around
    /// its own centroid — near 0 for a smooth sphere (bad — "never
    /// letting see the inside", Carl's own words), large for a form with
    /// real protrusions, notches, and visible concavities (Mandelbulb-
    /// like). See module docs for the `a6598cb3` (organic) vs `8e24fe3b`
    /// (sphere) pair this was calibrated against.
    pub silhouette_irregularity: f32,
}

impl QuatFitnessBreakdown {
    /// Weights, revised after round 2: `silhouette_irregularity` now
    /// carries the single biggest weight — it's the ONLY component that
    /// caught the sphere-vs-organic gap that prompted this revision, so
    /// it needs to dominate, not just contribute. `solidity` and
    /// `anisotropy` are still real preconditions (a solid, direction-
    /// sensitive object is necessary for a structured silhouette to even
    /// be possible) but no longer the largest terms. Sums to 1.0.
    pub fn total(&self) -> f32 {
        0.30 * self.silhouette_irregularity
            + 0.20 * self.solidity
            + 0.15 * self.anisotropy
            + 0.15 * self.coverage
            + 0.10 * self.shading_richness
            + 0.10 * self.color_entropy
    }
}

/// Coverage target: reward hit-fraction roughly linearly up to 50% frame
/// fill, saturating at 1.0 beyond that — both examples so far (Bulb 0.94,
/// 60e94 0.26) are well inside a "more is better up to a point" regime,
/// not near some upper falloff; revisit if a genome that fills 100% of
/// frame with a flat, silhouette-less wall turns up and scores too well.
fn coverage_score(hit_frac: f32) -> f32 {
    (hit_frac / 0.5).min(1.0)
}

/// Finds connected components among `interior_xy` (4-connected flood
/// fill over a `width x height` grid) and returns the LARGEST one's
/// pixel positions.
///
/// **Why this is needed, not optional**: a first version of
/// `silhouette_irregularity` measured radius-vs-angle over ALL interior
/// pixels regardless of whether they formed one connected mass — and
/// genome `60e94f4f394bbdb2` (Carl's original "not interesting" example:
/// a sparse cluster of separate comb-like teeth, each individually a few
/// pixels wide) still maxed out the score. Each tooth is wide enough to
/// have its own interior pixels, but the teeth are SEPARATE blobs at
/// different distances from their shared centroid — high radius-vs-angle
/// variance, but from scattered clutter, not one irregular boundary.
/// Restricting to the single largest connected mass is what actually
/// distinguishes "one object with real protrusions" (Mandelbulb,
/// `a6598cb3`) from "several disconnected chunks" (`60e94`) — verified in
/// this module's own tests against both.
fn largest_connected_component(interior_xy: &[(usize, usize)], width: usize, height: usize) -> Vec<(usize, usize)> {
    if interior_xy.is_empty() {
        return Vec::new();
    }
    let mut grid = vec![false; width * height];
    for &(x, y) in interior_xy {
        grid[y * width + x] = true;
    }
    let mut visited = vec![false; width * height];
    let mut best: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &(sx, sy) in interior_xy {
        let idx0 = sy * width + sx;
        if visited[idx0] {
            continue;
        }
        let mut component = Vec::new();
        stack.push((sx, sy));
        visited[idx0] = true;
        while let Some((x, y)) = stack.pop() {
            component.push((x, y));
            let neighbors = [
                (x.wrapping_sub(1), y), (x + 1, y),
                (x, y.wrapping_sub(1)), (x, y + 1),
            ];
            for (nx, ny) in neighbors {
                if nx < width && ny < height {
                    let nidx = ny * width + nx;
                    if grid[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
        }
        if component.len() > best.len() {
            best = component;
        }
    }
    best
}

/// Coefficient-of-variation of the silhouette's outer radius as a
/// function of angle around its own centroid, in `[0, 1]`. `N_BINS`
/// angular buckets; for each pixel of the LARGEST CONNECTED interior
/// component (see `largest_connected_component` — hit, surrounded by hit
/// pixels, AND part of one coherent solid mass rather than a scattered
/// blob), tracks the MAX distance from centroid seen in its bucket — a
/// circle/sphere silhouette has nearly constant per-bucket radius (score
/// near 0), a silhouette with real protrusions and notches (wings,
/// cavities reaching the edge, asymmetric limbs) has large swings between
/// buckets.
fn silhouette_irregularity(interior_xy: &[(usize, usize)], width: usize, height: usize) -> f32 {
    const N_BINS: usize = 36;
    let component = largest_connected_component(interior_xy, width, height);
    let n = component.len() as f64;
    if n < 8.0 {
        return 0.0;
    }
    let (sx, sy) = component.iter().fold((0.0f64, 0.0f64), |(sx, sy), &(x, y)| (sx + x as f64, sy + y as f64));
    let (cx, cy) = (sx / n, sy / n);

    let mut bin_max_r = [0.0f64; N_BINS];
    let mut bin_has_sample = [false; N_BINS];
    for &(x, y) in &component {
        let (dx, dy) = (x as f64 - cx, y as f64 - cy);
        let r = (dx * dx + dy * dy).sqrt();
        if r < 1e-6 {
            continue;
        }
        let theta = dy.atan2(dx) + std::f64::consts::PI; // [0, 2π)
        let bin = ((theta / (2.0 * std::f64::consts::PI)) * N_BINS as f64) as usize;
        let bin = bin.min(N_BINS - 1);
        bin_has_sample[bin] = true;
        if r > bin_max_r[bin] {
            bin_max_r[bin] = r;
        }
    }
    let radii: Vec<f64> = (0..N_BINS).filter(|&b| bin_has_sample[b]).map(|b| bin_max_r[b]).collect();
    if radii.len() < N_BINS / 2 {
        // Fewer than half the angular bins had a pixel from the largest
        // component — that mass doesn't surround its own centroid, not a
        // genuinely irregular closed shape; don't reward that.
        return 0.0;
    }
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    if mean < 1e-6 {
        return 0.0;
    }
    let var = radii.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64;
    let cv = var.sqrt() / mean;
    // Empirically (largest-connected-component, 256x256 probes, properly
    // framed — see explorer.rs's score_probes): a near-perfect sphere
    // (`8e24fe3bf318e6e3`) measures CV≈0.003-0.005; a genuinely organic
    // silhouette with notches/protrusions (`a6598cb3b21c491b`) measures
    // CV≈0.06-0.09. Both real fractal geometry, not synthetic extremes —
    // the gap is much narrower in absolute terms than it looks by eye,
    // but still a clear >10x ratio. Normalize so the organic example
    // lands comfortably above the middle, leaving room above it, with a
    // small floor so sphere-level noise reads as ~0.
    ((cv - 0.004) / 0.08).clamp(0.0, 1.0) as f32
}

/// Analyzes one rendered probe view's `shading`/`color_t` buffers
/// (`quat_dag::render_raymarch_dag_frame`'s or the GPU codegen path's
/// output shape — background pixels have `shading <= 0.0`) and returns
/// `(coverage_score, solidity, shading_richness, color_entropy,
/// silhouette_irregularity)` for that view. Callers average this across a
/// few camera angles before folding in `anisotropy` (see module docs on
/// why multi-angle matters).
pub fn view_metrics(shading: &[f32], color_t: &[f32], width: u32, height: u32) -> (f32, f32, f32, f32, f32) {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || shading.len() != w * h || color_t.len() != shading.len() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let hit = |x: usize, y: usize| shading[y * w + x] > 0.0;

    let mut hit_count = 0usize;
    let mut interior_count = 0usize;
    let mut interior_shading = Vec::new();
    let mut interior_color = Vec::new();
    let mut interior_xy = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !hit(x, y) {
                continue;
            }
            hit_count += 1;
            // "Interior" = itself hit AND every one of its 8 neighbors
            // hit (frame-edge pixels can't be interior — treated as
            // non-interior rather than specially, a negligible effect
            // except on tiny probe resolutions).
            let interior = x > 0
                && y > 0
                && x + 1 < w
                && y + 1 < h
                && hit(x - 1, y - 1) && hit(x, y - 1) && hit(x + 1, y - 1)
                && hit(x - 1, y) && hit(x + 1, y)
                && hit(x - 1, y + 1) && hit(x, y + 1) && hit(x + 1, y + 1);
            if interior {
                interior_count += 1;
                interior_shading.push(shading[y * w + x]);
                interior_color.push(color_t[y * w + x]);
                interior_xy.push((x, y));
            }
        }
    }
    let hit_frac = hit_count as f32 / (w * h) as f32;
    let coverage = coverage_score(hit_frac);
    let solidity = if hit_count == 0 { 0.0 } else { interior_count as f32 / hit_count as f32 };
    let irregularity = silhouette_irregularity(&interior_xy, w, h);

    let shading_richness = if interior_shading.len() < 4 {
        0.0
    } else {
        let mean = interior_shading.iter().sum::<f32>() / interior_shading.len() as f32;
        let var = interior_shading.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / interior_shading.len() as f32;
        // Shading lives in [0.15, 1.0] (ambient floor to full lit) — max
        // possible std-dev for a bimodal split of that range is ~0.425;
        // normalize against that so a genuinely full-range surface scores
        // near 1.0.
        (var.sqrt() / 0.425).min(1.0)
    };

    const BINS: usize = 32;
    let color_entropy = if interior_color.len() < 4 {
        0.0
    } else {
        let lo = interior_color.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = interior_color.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let span = (hi - lo).max(1e-6);
        let mut hist = [0u32; BINS];
        for &c in &interior_color {
            let b = (((c - lo) / span) * (BINS as f32 - 1.0)) as usize;
            hist[b.min(BINS - 1)] += 1;
        }
        let n = interior_color.len() as f32;
        hist.iter().filter(|&&c| c > 0).map(|&c| { let p = c as f32 / n; -p * p.log2() }).sum::<f32>() / (BINS as f32).log2()
    };

    (coverage, solidity, shading_richness, color_entropy, irregularity)
}

/// Folds `anisotropy` (computed once, view-independent) together with
/// per-view `(coverage, solidity, shading_richness, color_entropy,
/// silhouette_irregularity)` tuples (from `view_metrics`, one per camera
/// angle probed) into a single breakdown, averaging across views.
pub fn combine_views(anisotropy: f32, per_view: &[(f32, f32, f32, f32, f32)]) -> QuatFitnessBreakdown {
    if per_view.is_empty() {
        return QuatFitnessBreakdown { anisotropy, ..Default::default() };
    }
    let n = per_view.len() as f32;
    let (mut coverage, mut solidity, mut shading_richness, mut color_entropy, mut irregularity) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(c, s, r, e, i) in per_view {
        coverage += c;
        solidity += s;
        shading_richness += r;
        color_entropy += e;
        irregularity += i;
    }
    QuatFitnessBreakdown {
        anisotropy,
        coverage: coverage / n,
        solidity: solidity / n,
        shading_richness: shading_richness / n,
        color_entropy: color_entropy / n,
        silhouette_irregularity: irregularity / n,
    }
}

// ============================================================================
// Extended metrics — deliberately diverse, NOT folded into `total()`
// ============================================================================
//
// Carl's own framing: "I won't know if a metric is good until I've tested
// it... find the most different metrics you can." Everything below is
// therefore written in its plain, natural mathematical sense (a
// textbook circularity/symmetry/dimension measure), NOT pre-biased
// toward "higher = more interesting" the way `silhouette_irregularity`
// deliberately is — baking in an assumed-good direction for a metric
// whose whole point is to be tested empirically would defeat the
// purpose. Each function's doc comment says plainly what a high vs. low
// value means structurally; none of this changes `QuatFitnessBreakdown`
// or `.total()` — these are additional, independently-persisted fields
// for sorting/inspecting in the gallery, not new selection pressure.
// See `cmd_quat_dag_evolve`'s `--fitness-metric` flag (explorer.rs) for
// how a metric here actually gets tested as a selection criterion, one
// at a time, deliberately — not by guessing weights up front.
//
// All single-view metrics reuse the exact same `shading`/`color_t`
// probe buffers `view_metrics` already computes from (no shader/GPU
// codegen changes — see the module-level rationale for why that's
// deliberately out of scope here).

/// Per-component breakdown of the 15 new metrics computable from ONE
/// rendered view. Averaged across probe views the same way
/// `QuatFitnessBreakdown` is (see `combine_extended_views`). All fields
/// [0,1] except `bilateral_symmetry`/`color_shading_corr`/
/// `color_band_autocorr`, which are correlations remapped from [-1,1]
/// (0.5 = no relationship either way) — documented per-field.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuatExtendedMetrics {
    // --- silhouette / topology (from the hit-mask) ---
    /// Box-counting fractal dimension of the silhouette BOUNDARY,
    /// remapped from its natural ~[1,2] range (1 = smooth curve, 2 =
    /// space-filling) to [0,1]. Different axis from
    /// `silhouette_irregularity`: a boundary can be rough/self-similar
    /// at every scale (high here) while still having a fairly round
    /// overall outline (low irregularity), or vice versa.
    pub box_dim: f32,
    /// Gliding-box lacunarity of the hit-mask, squashed to [0,1] via
    /// `x/(1+x)`. Low = uniform/dense texture, high = "gappy"/porous.
    /// Two silhouettes with identical `box_dim` can still differ a lot
    /// here.
    pub lacunarity: f32,
    /// hit-mask area ÷ convex-hull area of the largest connected
    /// component, in [0,1]. 1.0 = perfectly convex (no
    /// notches/concavities reach in from the boundary), lower = real
    /// concave structure. Different from `silhouette_irregularity`
    /// (angular-radius variance): a crescent has deep concavity here
    /// but only moderate radius variance if the "bite" doesn't reach the
    /// outer edge.
    pub convexity: f32,
    /// `1 - circularity` (`circularity = 4π·area/perimeter²`, computed
    /// from actual boundary-pixel counting), in [0,1] (clamped — pixel
    /// perimeter can slightly overestimate a true smooth boundary's
    /// length). 0 = as round as a disc of the same area, higher = more
    /// elongated/convoluted boundary relative to its enclosed area. A
    /// different construction from both `convexity` and
    /// `silhouette_irregularity`.
    pub isoperimetric: f32,
    /// Mean of horizontal- and vertical-mirror IoU against the hit-mask
    /// itself, in [0,1]. 1.0 = perfectly bilaterally symmetric about
    /// its own centroid on both axes. A genuinely new axis: WITHIN-view
    /// orientation structure, vs. `anisotropy`'s ACROSS-view symmetry.
    pub bilateral_symmetry: f32,
    /// Distance between the largest component's mass centroid and its
    /// own bounding-box center, normalized by the bounding box's half-
    /// diagonal, in [0,1]. 0 = perfectly centered mass, higher = the
    /// silhouette's "visual weight" sits off to one side.
    pub centroid_offset: f32,
    /// Largest connected component's pixel count ÷ total hit pixel
    /// count, in [0,1]. 1.0 = one coherent mass with no separate
    /// fragments. Different from `solidity` (neighbor-fill within a
    /// mass, not fragment count).
    pub largest_component_frac: f32,

    // --- surface / shading (interior pixels only) ---
    /// Mean local gradient magnitude of `shading` over interior pixels,
    /// normalized/clamped to [0,1]. Local roughness/curvature DENSITY —
    /// distinct from `shading_richness`'s GLOBAL variance (two flat
    /// regions of different brightness have high variance but zero
    /// local gradient).
    pub shading_gradient: f32,
    /// Third standardized moment (skewness) of interior `shading`
    /// values, remapped from ~[-2,2] to [0,1] (0.5 = symmetric
    /// distribution). Low = mostly bright with dark crevices, high =
    /// mostly dark with bright highlights. Orthogonal to
    /// `shading_richness` (2nd moment).
    pub shading_skewness: f32,
    /// Fraction of interior pixels near the fully-lit ceiling
    /// (`shading > 0.85`), in [0,1].
    pub specular_fraction: f32,
    /// Fraction of interior pixels near the ambient floor
    /// (`shading < 0.25`), in [0,1]. Together with `specular_fraction`,
    /// characterizes light/dark balance beyond raw spread.
    pub crevice_fraction: f32,

    // --- color (from color_t, locally normalized to its own [lo,hi]) ---
    /// Mean local gradient magnitude of (locally-normalized) `color_t`
    /// over interior pixels, in [0,1]. Spatial color complexity —
    /// distinct from `color_entropy`'s histogram (entropy is blind to
    /// spatial arrangement: a checkerboard and a smooth gradient with
    /// the same histogram have identical entropy but very different
    /// gradient magnitude).
    pub color_gradient: f32,
    /// Pearson correlation between `shading` and `color_t` over
    /// interior pixels, remapped from [-1,1] to [0,1] (0.5 =
    /// uncorrelated). A genuinely new JOINT statistic — are color bands
    /// aligned with surface curvature, or decoupled from it?
    pub color_shading_corr: f32,
    /// Cheap proxy for color-banding periodicity: mean Pearson
    /// autocorrelation of (locally-normalized) `color_t` at horizontal
    /// pixel-offset lags of 2/4/8, remapped from [-1,1] to [0,1] (a
    /// full FFT would be the "real" version of this; this is a few
    /// lag-correlation samples instead, deliberately cheap). High =
    /// regular fine banding, low/near-0.5 = noise-like or very smooth.
    pub color_band_autocorr: f32,
    /// (max−min) of `color_t` over interior pixels, relative to
    /// `max_iter`, in [0,1]. A narrow-but-evenly-spread range scores
    /// high on `color_entropy` but low here — a genuinely different
    /// statistic.
    pub color_range_utilization: f32,
}

impl QuatExtendedMetrics {
    /// Replaces any non-finite (NaN/±inf) field with 0.0. Belt-and-
    /// suspenders on top of `view_extended_metrics`'s finite-shading
    /// guard: a `f32::NAN`/`INFINITY` reaching `Genome` gets serialized
    /// as JSON `null` (serde_json's documented behavior for non-finite
    /// floats) — and a `null` where an `f32` is expected fails to
    /// DESERIALIZE, corrupting the genome file's next load. Every path
    /// that can produce one of these fields calls this before returning,
    /// so that failure mode is structurally impossible here regardless
    /// of which specific computation an unstable evolved formula
    /// happens to break.
    fn sanitized(self) -> Self {
        let f = |v: f32| if v.is_finite() { v } else { 0.0 };
        QuatExtendedMetrics {
            box_dim: f(self.box_dim),
            lacunarity: f(self.lacunarity),
            convexity: f(self.convexity),
            isoperimetric: f(self.isoperimetric),
            bilateral_symmetry: f(self.bilateral_symmetry),
            centroid_offset: f(self.centroid_offset),
            largest_component_frac: f(self.largest_component_frac),
            shading_gradient: f(self.shading_gradient),
            shading_skewness: f(self.shading_skewness),
            specular_fraction: f(self.specular_fraction),
            crevice_fraction: f(self.crevice_fraction),
            color_gradient: f(self.color_gradient),
            color_shading_corr: f(self.color_shading_corr),
            color_band_autocorr: f(self.color_band_autocorr),
            color_range_utilization: f(self.color_range_utilization),
        }
    }
}

fn mean_std(vals: &[f32]) -> (f32, f32) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    (mean, var.sqrt())
}

/// Pearson correlation of two equal-length slices, in [-1,1]; 0.0 if
/// either has (near-)zero variance.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let (mean_a, std_a) = mean_std(a);
    let (mean_b, std_b) = mean_std(b);
    if std_a < 1e-6 || std_b < 1e-6 {
        return 0.0;
    }
    let cov = a.iter().zip(b).map(|(&x, &y)| (x - mean_a) * (y - mean_b)).sum::<f32>() / a.len() as f32;
    (cov / (std_a * std_b)).clamp(-1.0, 1.0)
}

/// Andrew's monotone-chain convex hull over integer pixel positions;
/// returns the hull polygon's area via the shoelace formula. `points`
/// need not be sorted or deduplicated.
fn convex_hull_area(points: &[(usize, usize)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut pts: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }
    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * pts.len());
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    if hull.len() < 3 {
        return 0.0;
    }
    let mut area2 = 0.0;
    for i in 0..hull.len() {
        let (x1, y1) = hull[i];
        let (x2, y2) = hull[(i + 1) % hull.len()];
        area2 += x1 * y2 - x2 * y1;
    }
    (area2 / 2.0).abs()
}

/// Box-counting dimension of the hit-mask's boundary, remapped from its
/// natural ~[1,2] range to [0,1]. Boundary = hit pixels with at least
/// one non-hit 4-neighbor (or touching the frame edge). Counts boxes of
/// several sizes that contain a boundary pixel and fits the log-log
/// slope by least squares.
fn box_counting_dimension(hit: &dyn Fn(usize, usize) -> bool, width: usize, height: usize) -> f32 {
    let mut boundary = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if !hit(x, y) {
                continue;
            }
            let edge_touch = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            let neighbor_bg = edge_touch
                || !hit(x - 1, y) || !hit(x + 1, y) || !hit(x, y - 1) || !hit(x, y + 1);
            if neighbor_bg {
                boundary.push((x, y));
            }
        }
    }
    if boundary.len() < 8 {
        return 0.0;
    }
    let min_dim = width.min(height).max(4);
    let sizes: Vec<usize> = [2usize, 4, 8, 16, 32].into_iter().filter(|&s| s < min_dim).collect();
    if sizes.len() < 2 {
        return 0.0;
    }
    let mut xs = Vec::with_capacity(sizes.len());
    let mut ys = Vec::with_capacity(sizes.len());
    for &s in &sizes {
        let mut boxes = std::collections::HashSet::new();
        for &(x, y) in &boundary {
            boxes.insert((x / s, y / s));
        }
        if boxes.is_empty() {
            continue;
        }
        xs.push((1.0 / s as f64).ln());
        ys.push((boxes.len() as f64).ln());
    }
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..xs.len() {
        num += (xs[i] - mean_x) * (ys[i] - mean_y);
        den += (xs[i] - mean_x).powi(2);
    }
    if den < 1e-9 {
        return 0.0;
    }
    let slope = num / den; // ~dimension for a boundary curve
    ((slope - 1.0) / 1.0).clamp(0.0, 1.0) as f32
}

/// Gliding-box lacunarity over the hit-mask (stride = box size / 2, for
/// speed over an exhaustive slide), squashed to [0,1] via `x/(1+x)`.
fn lacunarity(hit: &dyn Fn(usize, usize) -> bool, width: usize, height: usize) -> f32 {
    let box_size = (width.min(height) / 8).max(2);
    let stride = (box_size / 2).max(1);
    let mut counts = Vec::new();
    let mut y = 0;
    while y + box_size <= height {
        let mut x = 0;
        while x + box_size <= width {
            let mut c = 0u32;
            for by in y..y + box_size {
                for bx in x..x + box_size {
                    if hit(bx, by) {
                        c += 1;
                    }
                }
            }
            counts.push(c as f64);
            x += stride;
        }
        y += stride;
    }
    if counts.len() < 4 {
        return 0.0;
    }
    let mean = counts.iter().sum::<f64>() / counts.len() as f64;
    if mean < 1e-6 {
        return 0.0;
    }
    let var = counts.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / counts.len() as f64;
    let lac = var / (mean * mean);
    (lac / (1.0 + lac)) as f32
}

/// Computes all 15 single-view extended metrics from one probe's
/// rendered buffers. `max_iter` is needed only for
/// `color_range_utilization`'s normalization.
pub fn view_extended_metrics(shading: &[f32], color_t: &[f32], width: u32, height: u32, max_iter: f32) -> QuatExtendedMetrics {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || shading.len() != w * h || color_t.len() != shading.len() {
        return QuatExtendedMetrics::default();
    }
    // `> 0.0` alone lets a +infinity shading value through as "hit" (a
    // real, observed failure mode: an unstable evolved formula can push
    // the raymarcher's shading computation to infinity/NaN at some
    // pixels). `view_metrics` above doesn't hit this in practice because
    // it only ever takes means/variances of the resulting values, which
    // happen to stay finite for the genomes checked so far — but this
    // function's gradient math (differencing ADJACENT pixels) amplifies
    // exactly this case (inf - inf = NaN), and a NaN/inf field would
    // otherwise get persisted into `Genome` as JSON `null`, which then
    // fails to deserialize as `f32` on the next load — a real genome
    // file was corrupted this way while testing this function. Treat
    // non-finite shading as background, same as any other non-hit pixel.
    let hit = |x: usize, y: usize| shading[y * w + x].is_finite() && shading[y * w + x] > 0.0;
    let hit_xy: Vec<(usize, usize)> = (0..h).flat_map(|y| (0..w).filter(move |&x| hit(x, y)).map(move |x| (x, y))).collect();
    if hit_xy.is_empty() {
        return QuatExtendedMetrics::default();
    }

    let box_dim = box_counting_dimension(&hit, w, h);
    let lac = lacunarity(&hit, w, h);

    let largest = largest_connected_component(&hit_xy, w, h);
    let (convexity, isoperimetric, centroid_offset) = if largest.len() < 8 {
        (0.0, 0.0, 0.0)
    } else {
        let area = largest.len() as f64;
        let hull_area = convex_hull_area(&largest).max(area); // hull can't be smaller than the mask itself
        let convexity = (area / hull_area) as f32;

        let mut perimeter = 0usize;
        let mut min_x = usize::MAX;
        let mut max_x = 0usize;
        let mut min_y = usize::MAX;
        let mut max_y = 0usize;
        let set: std::collections::HashSet<(usize, usize)> = largest.iter().copied().collect();
        for &(x, y) in &largest {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            let edge_touch = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            let neighbor_bg = edge_touch
                || !set.contains(&(x - 1, y)) || !set.contains(&(x + 1, y))
                || !set.contains(&(x, y - 1)) || !set.contains(&(x, y + 1));
            if neighbor_bg {
                perimeter += 1;
            }
        }
        let circularity = if perimeter == 0 { 1.0 } else { (4.0 * std::f64::consts::PI * area) / (perimeter as f64 * perimeter as f64) };
        let isoperimetric = (1.0 - circularity).clamp(0.0, 1.0) as f32;

        let (sx, sy) = largest.iter().fold((0.0f64, 0.0f64), |(sx, sy), &(x, y)| (sx + x as f64, sy + y as f64));
        let (cx, cy) = (sx / area, sy / area);
        let (bcx, bcy) = ((min_x + max_x) as f64 / 2.0, (min_y + max_y) as f64 / 2.0);
        let half_diag = (((max_x - min_x) as f64).powi(2) + ((max_y - min_y) as f64).powi(2)).sqrt() / 2.0;
        let centroid_offset = if half_diag < 1e-6 { 0.0 } else { (((cx - bcx).powi(2) + (cy - bcy).powi(2)).sqrt() / half_diag).clamp(0.0, 1.0) as f32 };

        (convexity.clamp(0.0, 1.0), isoperimetric, centroid_offset)
    };

    // Bilateral symmetry: mirror the hit-mask about its own centroid on
    // each axis and measure IoU against the original.
    let (sx, sy) = hit_xy.iter().fold((0.0f64, 0.0f64), |(sx, sy), &(x, y)| (sx + x as f64, sy + y as f64));
    let (ccx, ccy) = (sx / hit_xy.len() as f64, sy / hit_xy.len() as f64);
    let mut h_agree = 0u32;
    let mut v_agree = 0u32;
    let mut total = 0u32;
    for y in 0..h {
        for x in 0..w {
            let is_hit = hit(x, y);
            let mx = (2.0 * ccx - x as f64).round() as isize;
            let my = (2.0 * ccy - y as f64).round() as isize;
            let mirror_h = mx >= 0 && (mx as usize) < w && hit(mx as usize, y);
            let mirror_v = my >= 0 && (my as usize) < h && hit(x, my as usize);
            if is_hit == mirror_h {
                h_agree += 1;
            }
            if is_hit == mirror_v {
                v_agree += 1;
            }
            total += 1;
        }
    }
    let bilateral_symmetry = if total == 0 { 0.0 } else { 0.5 * (h_agree as f32 + v_agree as f32) / total as f32 };

    let largest_component_frac = largest.len() as f32 / hit_xy.len().max(1) as f32;

    // Interior-restricted surface/color metrics, matching view_metrics's
    // own interior convention exactly.
    let interior = |x: usize, y: usize| {
        x > 0 && y > 0 && x + 1 < w && y + 1 < h
            && hit(x - 1, y - 1) && hit(x, y - 1) && hit(x + 1, y - 1)
            && hit(x - 1, y) && hit(x + 1, y)
            && hit(x - 1, y + 1) && hit(x, y + 1) && hit(x + 1, y + 1)
    };
    let mut interior_shading = Vec::new();
    let mut interior_color = Vec::new();
    let mut interior_xy = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if hit(x, y) && interior(x, y) {
                interior_shading.push(shading[y * w + x]);
                interior_color.push(color_t[y * w + x]);
                interior_xy.push((x, y));
            }
        }
    }

    if interior_xy.len() < 9 {
        return QuatExtendedMetrics {
            box_dim, lacunarity: lac, convexity, isoperimetric, bilateral_symmetry, centroid_offset, largest_component_frac,
            ..Default::default()
        };
    }

    // Local [0,1] normalization of color for scale-invariant gradient/
    // correlation/autocorrelation math (mirrors color_entropy's own
    // lo/hi span technique).
    let lo = interior_color.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = interior_color.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-6);
    let color_range_utilization = ((hi - lo) / max_iter.max(1.0)).clamp(0.0, 1.0);

    let mut shading_grad_sum = 0.0f32;
    let mut color_grad_sum = 0.0f32;
    let mut grad_count = 0u32;
    for &(x, y) in &interior_xy {
        if x + 1 < w && interior(x + 1, y) {
            shading_grad_sum += (shading[y * w + x + 1] - shading[y * w + x]).abs();
            let cn = (color_t[y * w + x + 1] - lo) / span;
            let cc = (color_t[y * w + x] - lo) / span;
            color_grad_sum += (cn - cc).abs();
            grad_count += 1;
        }
        if y + 1 < h && interior(x, y + 1) {
            shading_grad_sum += (shading[(y + 1) * w + x] - shading[y * w + x]).abs();
            let cn = (color_t[(y + 1) * w + x] - lo) / span;
            let cc = (color_t[y * w + x] - lo) / span;
            color_grad_sum += (cn - cc).abs();
            grad_count += 1;
        }
    }
    let shading_gradient = if grad_count == 0 { 0.0 } else { (shading_grad_sum / grad_count as f32 / 0.2).clamp(0.0, 1.0) };
    let color_gradient = if grad_count == 0 { 0.0 } else { (color_grad_sum / grad_count as f32 / 0.2).clamp(0.0, 1.0) };

    let (mean_sh, std_sh) = mean_std(&interior_shading);
    let shading_skewness = if std_sh < 1e-6 {
        0.5
    } else {
        let skew = interior_shading.iter().map(|&v| ((v - mean_sh) / std_sh).powi(3)).sum::<f32>() / interior_shading.len() as f32;
        (skew.clamp(-2.0, 2.0) + 2.0) / 4.0
    };
    let specular_fraction = interior_shading.iter().filter(|&&v| v > 0.85).count() as f32 / interior_shading.len() as f32;
    let crevice_fraction = interior_shading.iter().filter(|&&v| v < 0.25).count() as f32 / interior_shading.len() as f32;

    let color_shading_corr = (pearson(&interior_shading, &interior_color) + 1.0) / 2.0;

    let color_norm: Vec<f32> = interior_color.iter().map(|&c| (c - lo) / span).collect();
    // O(1)-lookup grid of normalized color at interior pixels (a linear
    // `.position()` scan per pixel here would be O(n^2) over probe-sized
    // buffers — this runs once per genome score, every generation).
    let color_grid: std::collections::HashMap<(usize, usize), f32> = interior_xy
        .iter()
        .zip(color_norm.iter())
        .map(|(&p, &c)| (p, c))
        .collect();
    let mut autocorr_sum = 0.0f32;
    let mut autocorr_n = 0u32;
    for &lag in &[2usize, 4, 8] {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for y in 0..h {
            for x in 0..w.saturating_sub(lag) {
                if let (Some(&ca), Some(&cb)) = (color_grid.get(&(x, y)), color_grid.get(&(x + lag, y))) {
                    a.push(ca);
                    b.push(cb);
                }
            }
        }
        if a.len() >= 8 {
            autocorr_sum += (pearson(&a, &b) + 1.0) / 2.0;
            autocorr_n += 1;
        }
    }
    let color_band_autocorr = if autocorr_n == 0 { 0.5 } else { autocorr_sum / autocorr_n as f32 };

    QuatExtendedMetrics {
        box_dim, lacunarity: lac, convexity, isoperimetric, bilateral_symmetry, centroid_offset, largest_component_frac,
        shading_gradient, shading_skewness, specular_fraction, crevice_fraction,
        color_gradient, color_shading_corr, color_band_autocorr, color_range_utilization,
    }
    .sanitized()
}

/// Field-wise average across probe views, mirroring `combine_views`.
pub fn combine_extended_views(per_view: &[QuatExtendedMetrics]) -> QuatExtendedMetrics {
    if per_view.is_empty() {
        return QuatExtendedMetrics::default();
    }
    let n = per_view.len() as f32;
    let mut acc = QuatExtendedMetrics::default();
    for v in per_view {
        acc.box_dim += v.box_dim;
        acc.lacunarity += v.lacunarity;
        acc.convexity += v.convexity;
        acc.isoperimetric += v.isoperimetric;
        acc.bilateral_symmetry += v.bilateral_symmetry;
        acc.centroid_offset += v.centroid_offset;
        acc.largest_component_frac += v.largest_component_frac;
        acc.shading_gradient += v.shading_gradient;
        acc.shading_skewness += v.shading_skewness;
        acc.specular_fraction += v.specular_fraction;
        acc.crevice_fraction += v.crevice_fraction;
        acc.color_gradient += v.color_gradient;
        acc.color_shading_corr += v.color_shading_corr;
        acc.color_band_autocorr += v.color_band_autocorr;
        acc.color_range_utilization += v.color_range_utilization;
    }
    QuatExtendedMetrics {
        box_dim: acc.box_dim / n,
        lacunarity: acc.lacunarity / n,
        convexity: acc.convexity / n,
        isoperimetric: acc.isoperimetric / n,
        bilateral_symmetry: acc.bilateral_symmetry / n,
        centroid_offset: acc.centroid_offset / n,
        largest_component_frac: acc.largest_component_frac / n,
        shading_gradient: acc.shading_gradient / n,
        shading_skewness: acc.shading_skewness / n,
        specular_fraction: acc.specular_fraction / n,
        crevice_fraction: acc.crevice_fraction / n,
        color_gradient: acc.color_gradient / n,
        color_shading_corr: acc.color_shading_corr / n,
        color_band_autocorr: acc.color_band_autocorr / n,
        color_range_utilization: acc.color_range_utilization / n,
    }
}

/// Intersection-over-union of two same-shaped hit-masks (`shading > 0`).
/// 1.0 = identical silhouettes, 0.0 = no overlap at all. Used both for
/// cross-view silhouette comparison (empirical companion to the purely
/// analytic `anisotropy_score` — they can disagree, which is itself
/// informative) and for C-sensitivity (same view, two different C
/// values).
pub fn hitmask_iou(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut inter = 0u32;
    let mut union = 0u32;
    for i in 0..a.len() {
        let ha = a[i] > 0.0;
        let hb = b[i] > 0.0;
        if ha || hb {
            union += 1;
        }
        if ha && hb {
            inter += 1;
        }
    }
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

/// `|hit_frac(a) - hit_frac(b)|` — cheap companion to `hitmask_iou`: a
/// coarser but even cheaper signal for "does the apparent size change a
/// lot" (directional protrusions across views, or shape "breathing"
/// across C).
pub fn coverage_delta(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let frac = |s: &[f32]| s.iter().filter(|&&v| v > 0.0).count() as f32 / s.len() as f32;
    (frac(a) - frac(b)).abs()
}

/// Structural/analytic metrics computed directly from a DAG program —
/// no rendering at all, same cost class as `anisotropy_score`. Returns
/// `(node_count_norm, opcode_diversity, max_depth_norm)`, all [0,1].
/// `node_count_norm`/`max_depth_norm` are normalized against
/// `formula::N_SLOTS` (24), the hard register-file ceiling every DAG
/// program is capped at regardless of `config.optimization.max_nodes`.
pub fn structural_metrics(program: &[OpNode]) -> (f32, f32, f32) {
    if program.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    const N_SLOTS: f32 = 24.0;
    let node_count_norm = (program.len() as f32 / N_SLOTS).clamp(0.0, 1.0);

    let distinct: std::collections::HashSet<u8> = program.iter().map(|n| n.op).collect();
    let opcode_diversity = distinct.len() as f32 / program.len() as f32;

    // The program's LIVE (root-reaching) depth, not the max over every
    // node including dead introns — a dead subtree's depth doesn't
    // affect what the formula actually computes, so it shouldn't count
    // as "how deep is this formula". Shared with quat_genome_ops's
    // depth-aware mutation guard — see that function's docs.
    let max_depth_norm = (crate::formula::program_depth(program) as f32 / N_SLOTS).clamp(0.0, 1.0);

    (node_count_norm, opcode_diversity, max_depth_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::op;

    fn make_solid_disc(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let (wf, hf) = (w as f32, h as f32);
        let (cx, cy) = (wf / 2.0, hf / 2.0);
        let r = wf.min(hf) * 0.4;
        let mut shading = vec![0.0f32; (w * h) as usize];
        let mut color = vec![60.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let d = (dx * dx + dy * dy).sqrt();
                if d < r {
                    let idx = (y * w + x) as usize;
                    // vary shading/color smoothly across the disc so
                    // richness/entropy have something real to measure
                    shading[idx] = 0.15 + 0.85 * (1.0 - d / r);
                    color[idx] = 10.0 + 40.0 * (d / r);
                }
            }
        }
        (shading, color)
    }

    fn make_thin_lines(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let mut shading = vec![0.0f32; (w * h) as usize];
        let mut color = vec![60.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if x % 6 == 0 {
                    let idx = (y * w + x) as usize;
                    shading[idx] = 0.4;
                    color[idx] = 20.0;
                }
            }
        }
        (shading, color)
    }

    /// A silhouette with real angular protrusions (a 5-pointed star,
    /// roughly — the shape this metric is meant to reward) versus a
    /// smooth disc of the same rough size (Carl's actual complaint: real
    /// genomes rendering as smooth spheres). Synthetic, but the same
    /// qualitative gap measured on the real `a6598cb3` (organic) vs
    /// `8e24fe3b` (sphere) pair — see module docs.
    fn make_star(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let (wf, hf) = (w as f32, h as f32);
        let (cx, cy) = (wf / 2.0, hf / 2.0);
        let base_r = wf.min(hf) * 0.15;
        let spike_r = wf.min(hf) * 0.45;
        let mut shading = vec![0.0f32; (w * h) as usize];
        let color = vec![30.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let d = (dx * dx + dy * dy).sqrt();
                let theta = dy.atan2(dx);
                // 5 lobes: radius oscillates between base_r and spike_r
                let lobe = (0.5 + 0.5 * (theta * 5.0).cos()).powf(1.5);
                let r_at_theta = base_r + (spike_r - base_r) * lobe;
                if d < r_at_theta {
                    shading[(y * w + x) as usize] = 0.5;
                }
            }
        }
        (shading, color)
    }

    #[test]
    fn a_solid_disc_scores_much_higher_solidity_than_thin_scattered_lines() {
        let (w, h) = (64, 64);
        let (disc_shading, disc_color) = make_solid_disc(w, h);
        let (lines_shading, lines_color) = make_thin_lines(w, h);
        let (_, disc_solidity, _, _, _) = view_metrics(&disc_shading, &disc_color, w, h);
        let (_, lines_solidity, _, _, _) = view_metrics(&lines_shading, &lines_color, w, h);
        assert!(disc_solidity > 0.8, "solid disc should have high solidity, got {disc_solidity}");
        assert!(lines_solidity < 0.1, "1px-wide lines should have near-zero solidity, got {lines_solidity}");
        assert!(disc_solidity > lines_solidity * 5.0, "disc={disc_solidity} lines={lines_solidity}");
    }

    #[test]
    fn all_background_gives_zero_everything() {
        let (w, h) = (32, 32);
        let shading = vec![0.0f32; (w * h) as usize];
        let color = vec![60.0f32; (w * h) as usize];
        let (coverage, solidity, richness, entropy, irregularity) = view_metrics(&shading, &color, w, h);
        assert_eq!((coverage, solidity, richness, entropy, irregularity), (0.0, 0.0, 0.0, 0.0, 0.0));
    }

    /// The load-bearing test for this whole revision: a smooth disc
    /// (Carl's "spherical, never letting see the inside" complaint) must
    /// score much lower on `silhouette_irregularity` than a star-shaped
    /// silhouette with real protrusions, even though BOTH are equally
    /// solid — this is exactly the gap round-1 metrics missed.
    #[test]
    fn a_star_silhouette_scores_far_more_irregular_than_a_smooth_disc() {
        let (w, h) = (128, 128);
        let (disc_shading, disc_color) = make_solid_disc(w, h);
        let (star_shading, star_color) = make_star(w, h);
        let (_, _, _, _, disc_irr) = view_metrics(&disc_shading, &disc_color, w, h);
        let (_, _, _, _, star_irr) = view_metrics(&star_shading, &star_color, w, h);
        assert!(disc_irr < 0.15, "smooth disc should score low irregularity, got {disc_irr}");
        assert!(star_irr > 0.6, "5-lobed star should score high irregularity, got {star_irr}");
    }

    /// Several separate solid blobs at different distances from their
    /// shared centroid (each individually solid enough to have interior
    /// pixels, like real comb "teeth") — this is the exact shape of the
    /// bug found against genome `60e94f4f394bbdb2`: high radius-vs-angle
    /// variance from SCATTERED clutter, not from one irregular boundary.
    /// Must score LOW despite that raw variance, once restricted to the
    /// largest connected component.
    fn make_scattered_blobs(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
        let mut shading = vec![0.0f32; (w * h) as usize];
        let color = vec![30.0f32; (w * h) as usize];
        // 5 discs (radius 15px, well separated — corners + center of the
        // frame, pairwise center distance > 60px, more than double the
        // 30px sum-of-radii touching threshold) mimicking separate comb
        // teeth. Sized large enough that pixelation of each disc's own
        // boundary isn't itself a confound (r=6 in an earlier version of
        // this test produced a spuriously high CV purely from
        // discretization noise on a tiny circle, not genuine irregularity
        // — and an earlier center layout at r=18 had two blobs only
        // 34.5px apart, close enough to actually touch/merge, which is
        // exactly the failure mode this test exists to rule out).
        let centers: [(f32, f32, f32); 5] = [
            (0.15, 0.15, 15.0),
            (0.85, 0.15, 15.0),
            (0.15, 0.85, 15.0),
            (0.85, 0.85, 15.0),
            (0.5, 0.5, 15.0),
        ];
        for &(fx, fy, r) in &centers {
            let (bx, by) = (fx * w as f32, fy * h as f32);
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 - bx, y as f32 - by);
                    if (dx * dx + dy * dy).sqrt() < r {
                        shading[(y * w + x) as usize] = 0.5;
                    }
                }
            }
        }
        (shading, color)
    }

    #[test]
    fn scattered_disconnected_blobs_score_low_irregularity_despite_high_raw_variance() {
        let (w, h) = (128, 128);
        let (blobs_shading, blobs_color) = make_scattered_blobs(w, h);
        let (_, _, _, _, blobs_irr) = view_metrics(&blobs_shading, &blobs_color, w, h);
        assert!(blobs_irr < 0.15, "scattered disconnected blobs (comb-tooth shape) should score low irregularity once restricted to the largest connected component, got {blobs_irr}");
    }

    #[test]
    fn combine_views_averages_and_carries_anisotropy_through() {
        let bd = combine_views(0.7, &[(0.2, 0.8, 0.5, 0.6, 0.9), (0.4, 0.6, 0.3, 0.4, 0.5)]);
        assert_eq!(bd.anisotropy, 0.7);
        assert!((bd.coverage - 0.3).abs() < 1e-6);
        assert!((bd.solidity - 0.7).abs() < 1e-6);
        assert!((bd.shading_richness - 0.4).abs() < 1e-6);
        assert!((bd.color_entropy - 0.5).abs() < 1e-6);
        assert!((bd.silhouette_irregularity - 0.7).abs() < 1e-6);
    }

    #[test]
    fn total_is_a_weighted_sum_in_zero_one() {
        let bd = QuatFitnessBreakdown { anisotropy: 1.0, coverage: 1.0, solidity: 1.0, shading_richness: 1.0, color_entropy: 1.0, silhouette_irregularity: 1.0 };
        assert!((bd.total() - 1.0).abs() < 1e-5);
        let zero = QuatFitnessBreakdown::default();
        assert_eq!(zero.total(), 0.0);
    }

    // ---- extended metrics ----

    #[test]
    fn extended_metrics_on_all_background_give_defaults() {
        let (w, h) = (32, 32);
        let shading = vec![0.0f32; (w * h) as usize];
        let color = vec![60.0f32; (w * h) as usize];
        let m = view_extended_metrics(&shading, &color, w, h, 60.0);
        assert_eq!(m.box_dim, 0.0);
        assert_eq!(m.convexity, 0.0);
        assert_eq!(m.largest_component_frac, 0.0);
    }

    #[test]
    fn convexity_and_isoperimetric_distinguish_disc_from_star() {
        let (w, h) = (128, 128);
        let (disc_shading, disc_color) = make_solid_disc(w, h);
        let (star_shading, star_color) = make_star(w, h);
        let disc = view_extended_metrics(&disc_shading, &disc_color, w, h, 60.0);
        let star = view_extended_metrics(&star_shading, &star_color, w, h, 60.0);
        assert!(disc.convexity > 0.9, "a disc is already convex, got {}", disc.convexity);
        assert!(star.convexity < disc.convexity, "a 5-lobed star should be measurably less convex than a disc: star={} disc={}", star.convexity, disc.convexity);
        assert!(disc.isoperimetric < 0.1, "a disc is as round as it gets, got {}", disc.isoperimetric);
        assert!(star.isoperimetric > disc.isoperimetric, "a star's boundary should be less circular than a disc's: star={} disc={}", star.isoperimetric, disc.isoperimetric);
    }

    /// Broad safety net: every field stays finite and in its documented
    /// range across every synthetic shape this module already has on
    /// hand — this is intentionally not a precise per-field assertion
    /// (several of these metrics don't have an obvious expected ordering
    /// on these particular synthetic shapes), it's a guard against NaN/
    /// out-of-range regressions in a genuinely complex block of math.
    #[test]
    fn extended_metrics_stay_finite_and_in_unit_range() {
        let (w, h) = (96, 96);
        for (shading, color) in [make_solid_disc(w, h), make_star(w, h), make_scattered_blobs(w, h), make_thin_lines(w, h)] {
            let m = view_extended_metrics(&shading, &color, w, h, 60.0);
            for (name, v) in [
                ("box_dim", m.box_dim), ("lacunarity", m.lacunarity), ("convexity", m.convexity),
                ("isoperimetric", m.isoperimetric), ("bilateral_symmetry", m.bilateral_symmetry),
                ("centroid_offset", m.centroid_offset), ("largest_component_frac", m.largest_component_frac),
                ("shading_gradient", m.shading_gradient), ("shading_skewness", m.shading_skewness),
                ("specular_fraction", m.specular_fraction), ("crevice_fraction", m.crevice_fraction),
                ("color_gradient", m.color_gradient), ("color_shading_corr", m.color_shading_corr),
                ("color_band_autocorr", m.color_band_autocorr), ("color_range_utilization", m.color_range_utilization),
            ] {
                assert!(v.is_finite(), "{name} was not finite: {v}");
                assert!((0.0..=1.0).contains(&v), "{name} out of [0,1]: {v}");
            }
        }
    }

    #[test]
    fn combine_extended_views_averages_field_wise() {
        let a = QuatExtendedMetrics { box_dim: 0.2, convexity: 0.8, ..Default::default() };
        let b = QuatExtendedMetrics { box_dim: 0.6, convexity: 0.4, ..Default::default() };
        let combined = combine_extended_views(&[a, b]);
        assert!((combined.box_dim - 0.4).abs() < 1e-6);
        assert!((combined.convexity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn hitmask_iou_identical_is_one_disjoint_is_zero() {
        let (w, h) = (16, 16);
        let mut left_half = vec![0.0f32; w * h];
        let mut right_half = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if x < w / 2 {
                    left_half[y * w + x] = 0.5;
                } else {
                    right_half[y * w + x] = 0.5;
                }
            }
        }
        assert!((hitmask_iou(&left_half, &left_half) - 1.0).abs() < 1e-6);
        assert_eq!(hitmask_iou(&left_half, &right_half), 0.0);
    }

    #[test]
    fn coverage_delta_matches_hit_fraction_difference() {
        let a = vec![0.5f32, 0.5, 0.0, 0.0]; // 50% hit
        let b = vec![0.5f32, 0.0, 0.0, 0.0]; // 25% hit
        assert!((coverage_delta(&a, &b) - 0.25).abs() < 1e-6);
        assert_eq!(coverage_delta(&a, &a), 0.0);
    }

    #[test]
    fn structural_metrics_on_a_hand_built_program() {
        // 0: Z (leaf)  1: C (leaf)  2: SIN(Z)  3: ADD(SIN(Z), C)
        let prog = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let (node_count_norm, opcode_diversity, max_depth_norm) = structural_metrics(&prog);
        assert!((node_count_norm - 4.0 / 24.0).abs() < 1e-6);
        assert!((opcode_diversity - 1.0).abs() < 1e-6, "4 distinct opcodes across 4 nodes should be 1.0, got {opcode_diversity}");
        assert!((max_depth_norm - 2.0 / 24.0).abs() < 1e-6, "Z->SIN->ADD is a depth-2 chain, got norm {max_depth_norm}");
    }

    #[test]
    fn structural_metrics_empty_program_is_all_zero() {
        assert_eq!(structural_metrics(&[]), (0.0, 0.0, 0.0));
    }
}
