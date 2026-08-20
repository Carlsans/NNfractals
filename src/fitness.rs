use image;

/// Fast single-component score for GA selection: normalised Shannon entropy of escape times.
/// Range [0,1]; 0 = all pixels same escape time (degenerate), 1 = perfectly uniform histogram.
/// Used as Stage-1 prefilter before the CLIP aesthetic scorer.
pub fn entropy_score_fast(escape_times: &[f32], max_iter: u32) -> f32 {
    const BINS: usize = 32;
    let n = escape_times.len();
    if n == 0 { return 0.0; }
    let max = max_iter as f32;
    let mut hist = [0u32; BINS];
    for &t in escape_times {
        let b = ((t / max) * (BINS as f32 - 1.0)) as usize;
        hist[b.min(BINS - 1)] += 1;
    }
    let n_f = n as f32;
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n_f; -p * p.log2() })
        .sum::<f32>() / (BINS as f32).log2()
}

/// Edge density: fraction of adjacent pixel pairs with a large escape-time jump.
/// Targets fractals with rich boundary structure (alternating inside/outside pixels).
/// Returns [0, 1]; score 1.0 ≈ 20% of pixel pairs are edges.
pub fn edge_density_fast(escape_times: &[f32], width: usize, max_iter: u32) -> f32 {
    if escape_times.len() < 4 { return 0.0; }
    let height = escape_times.len() / width;
    let max_val = escape_times.iter().cloned().fold(0.0_f32, f32::max);
    let threshold = (max_val * 0.008).max(0.5);  // same threshold as beauty_score_full

    let mut edge_pairs = 0u32;
    let mut total = 0u32;

    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let a = escape_times[y * width + x];
            let b = escape_times[y * width + x + 1];
            if (a - b).abs() > threshold { edge_pairs += 1; }
            total += 1;
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let a = escape_times[y * width + x];
            let b = escape_times[(y + 1) * width + x];
            if (a - b).abs() > threshold { edge_pairs += 1; }
            total += 1;
        }
    }
    // 20% edge pairs → score 1.0; linear below, clamped above
    let frac = edge_pairs as f32 / total.max(1) as f32;
    (frac / 0.20).min(1.0)
}

/// Structural richness of the bailout exit-angle field (arg(z) at escape,
/// DAG genomes only — see fractal::dag_escape_pixel). Mirrors
/// edge_density_fast's shape, but uses CIRCULAR distance since angle wraps
/// at ±π (a plain difference would falsely treat +π and -π, right next to
/// each other on the circle, as maximally different). Returns [0,1]; 0 when
/// the angle buffer wasn't captured (all-zero — angle_structure_weight=0
/// skips computing it entirely, see evaluate_fitness_full/_batch).
pub fn angle_structure_score(angles: &[f32], width: usize) -> f32 {
    if angles.len() < 4 || width == 0 { return 0.0; }
    let height = angles.len() / width;
    const TWO_PI: f32 = std::f32::consts::TAU;
    let circ_dist = |a: f32, b: f32| { let d = (a - b).rem_euclid(TWO_PI); d.min(TWO_PI - d) };
    let threshold = std::f32::consts::FRAC_PI_4; // 45°

    let mut edge_pairs = 0u32;
    let mut total = 0u32;

    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let a = angles[y * width + x];
            let b = angles[y * width + x + 1];
            if circ_dist(a, b) > threshold { edge_pairs += 1; }
            total += 1;
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let a = angles[y * width + x];
            let b = angles[(y + 1) * width + x];
            if circ_dist(a, b) > threshold { edge_pairs += 1; }
            total += 1;
        }
    }
    // Same 20%→1.0 saturation convention as edge_density_fast.
    let frac = edge_pairs as f32 / total.max(1) as f32;
    (frac / 0.20).min(1.0)
}

#[cfg(test)]
mod angle_structure_tests {
    use super::angle_structure_score;

    #[test]
    fn empty_or_undersized_is_zero() {
        assert_eq!(angle_structure_score(&[], 4), 0.0);
        assert_eq!(angle_structure_score(&[0.0, 0.1, 0.2], 4), 0.0); // len < 4
        assert_eq!(angle_structure_score(&[0.0, 0.1, 0.2, 0.3], 0), 0.0); // width 0
    }

    #[test]
    fn wrap_around_near_pi_is_not_an_edge() {
        // A checkerboard alternating +π-ε / -π+ε would score maximal edge
        // density under a NAIVE linear |a-b| comparison (diff ≈ 2π), but
        // these two angles are actually adjacent on the circle (diff ≈ 2ε) —
        // the circular-distance implementation must treat this as smooth,
        // not structured.
        use std::f32::consts::PI;
        let near_pos = PI - 0.01;
        let near_neg = -PI + 0.01;
        let w = 4;
        let angles = vec![near_pos, near_neg, near_pos, near_neg,
                           near_neg, near_pos, near_neg, near_pos,
                           near_pos, near_neg, near_pos, near_neg,
                           near_neg, near_pos, near_neg, near_pos];
        let score = angle_structure_score(&angles, w);
        assert!(score < 0.05, "wrap-around angles falsely scored as structured: {score}");
    }

    #[test]
    fn genuinely_alternating_angles_score_high() {
        // 0 vs π (a true 180° jump every step) should saturate the score.
        let w = 4;
        let angles = vec![0.0, std::f32::consts::PI, 0.0, std::f32::consts::PI,
                           std::f32::consts::PI, 0.0, std::f32::consts::PI, 0.0,
                           0.0, std::f32::consts::PI, 0.0, std::f32::consts::PI,
                           std::f32::consts::PI, 0.0, std::f32::consts::PI, 0.0];
        let score = angle_structure_score(&angles, w);
        assert!(score > 0.95, "genuinely alternating angles scored too low: {score}");
    }
}

/// PNG compression entropy: render fractal → apply colormap → encode PNG in memory.
/// Returns bytes_per_pixel of the compressed PNG (higher = harder to compress = more visual detail).
/// This is the primary fitness metric — it directly measures structural complexity as perceived
/// by lossless compression, which correlates with what makes a fractal visually interesting.
pub fn png_compression_entropy(
    escape_times: &[f32],
    width: u32,
    height: u32,
    max_iter: u32,
    colormap: &str,
) -> f32 {
    let rgb = crate::colormap::apply_colormap(escape_times, max_iter, colormap);
    let mut buf = std::io::Cursor::new(Vec::with_capacity(8192));
    image::write_buffer_with_format(
        &mut buf,
        &rgb,
        width,
        height,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .unwrap_or(());
    let png_bytes = buf.into_inner().len() as f32;
    let raw_bytes = (width * height * 3) as f32;
    png_bytes / raw_bytes  // 0..1+ (>1 theoretically impossible; ~0.3 boring, ~0.9+ rich)
}

/// Lag-1 horizontal Pearson correlation of an escape-time field — how much a
/// pixel predicts its right-hand neighbour. **The noise detector.**
///
/// Structure is spatially coherent (neighbouring pixels are related), random
/// dither is not, and the separation is enormous rather than marginal.
/// Measured on real rendered frames (2026-08-15):
///
/// ```text
///   pure-noise frames:  0.027, 0.016, 0.002
///   good-structure:     0.333, 0.297, 0.262
///   rich start frame:   0.928
/// ```
///
/// This exists because BOTH of this project's existing complexity metrics are
/// blind to speckle noise, in a way that actively rewards it:
///   * `png_compression_entropy` is MAXIMISED by noise — noise is
///     incompressible, so garbage scores like rich structure.
///   * `multiscale_entropy` was supposed to fix that by collapsing the coarse
///     term, but dense speckle SURVIVES 4x average-pooling, so it doesn't.
/// A `video-zoom-explore` search ranked a pure noise field as its #1 winner
/// because of this (Carl's first batch run). Correlation catches it because
/// it measures a different property entirely — spatial coherence, not
/// information content — and noise cannot fake coherence.
///
/// Returns 0.0 for a degenerate (constant) field, which is correct here: a
/// flat frame has no structure to preserve either.
pub fn spatial_coherence(field: &[f32], width: u32, height: u32) -> f32 {
    let (w, h) = (width as usize, height as usize);
    if w < 2 || h < 1 || field.len() < w * h { return 0.0; }
    let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        let row = y * w;
        for x in 0..w - 1 {
            let a = field[row + x] as f64;
            let b = field[row + x + 1] as f64;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; n += 1.0;
        }
    }
    if n < 2.0 { return 0.0; }
    let cov = sab / n - (sa / n) * (sb / n);
    let va = (saa / n - (sa / n).powi(2)).max(0.0);
    let vb = (sbb / n - (sb / n).powi(2)).max(0.0);
    let denom = (va * vb).sqrt();
    if denom <= 1e-12 { return 0.0; }
    (cov / denom).clamp(-1.0, 1.0) as f32
}

/// Below this, a REGION is speckle noise rather than structure. Sits in the
/// wide empty gap between the two measured regimes — deliberately not a
/// knife-edge.
pub const MIN_SPATIAL_COHERENCE: f32 = 0.10;

/// A tile with luminance std below this is FLAT, not noisy. Flat regions
/// (solid interior, smooth basin) legitimately have zero correlation, so
/// without this guard they'd be indistinguishable from dither and every
/// frame containing a solid area would be rejected.
const TILE_TEXTURE_STD: f32 = 6.0;

/// Fraction of a frame's TEXTURED tiles that are speckle noise, on a 4x4
/// grid. **Use this, not whole-frame [`spatial_coherence`], to judge a
/// frame.**
///
/// Whole-frame correlation has a real blind spot: a frame that is half
/// smooth gradient and half dither averages to a healthy-looking score. A
/// real one measured 0.564 whole-frame — comfortably "coherent" — while
/// being visibly half garbage. Tiling localises the judgement, and skipping
/// flat tiles keeps legitimately-empty regions from counting against it.
///
/// Measured separation (2026-08-15), which is why the 0.25 threshold is
/// safe rather than tuned:
///
/// ```text
///   good frames (3 sampled):        0.00, 0.00, 0.00
///   half-noise frame:               0.56
///   pure-noise frame:               0.94
/// ```
///
/// Input is expected to be COLORMAPPED luminance on a 0-255 scale — the
/// thing that actually ships — not the raw escape-time field.
pub fn noise_tile_fraction(lum: &[f32], width: u32, height: u32) -> f32 {
    const N: usize = 4;
    let (w, h) = (width as usize, height as usize);
    if w < N * 2 || h < N * 2 || lum.len() < w * h { return 0.0; }
    let (tw, th) = (w / N, h / N);
    let (mut textured, mut noisy) = (0u32, 0u32);
    let mut tile = Vec::with_capacity(tw * th);
    for ty in 0..N {
        for tx in 0..N {
            tile.clear();
            for y in ty * th..(ty + 1) * th {
                for x in tx * tw..(tx + 1) * tw {
                    tile.push(lum[y * w + x]);
                }
            }
            let n = tile.len() as f32;
            let mean = tile.iter().sum::<f32>() / n;
            let var = tile.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            if var.sqrt() < TILE_TEXTURE_STD { continue; } // flat, not noisy
            textured += 1;
            if spatial_coherence(&tile, tw as u32, th as u32) < MIN_SPATIAL_COHERENCE {
                noisy += 1;
            }
        }
    }
    if textured == 0 { return 0.0; }
    noisy as f32 / textured as f32
}

/// Above this fraction of noisy textured tiles, a frame is garbage.
pub const MAX_NOISE_TILE_FRACTION: f32 = 0.25;

#[cfg(test)]
mod coherence_tests {
    use super::*;

    #[test]
    fn spatial_coherence_separates_noise_from_structure() {
        let (w, h) = (64u32, 64u32);
        // Deterministic pseudo-random speckle — no spatial relationship
        // between neighbours, which is the defining property of the frames
        // this exists to reject.
        let mut seed = 12345u32;
        let noise: Vec<f32> = (0..w * h).map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as f32
        }).collect();
        // A smooth gradient — maximally coherent.
        let smooth: Vec<f32> = (0..w * h).map(|i| (i % w) as f32).collect();

        let cn = spatial_coherence(&noise, w, h);
        let cs = spatial_coherence(&smooth, w, h);
        assert!(cn < MIN_SPATIAL_COHERENCE, "noise must fail the gate, got {cn}");
        assert!(cs > MIN_SPATIAL_COHERENCE, "structure must pass the gate, got {cs}");
        assert!(cs > cn, "structure must score above noise ({cs} vs {cn})");
    }

    #[test]
    fn noise_tile_fraction_ignores_flat_tiles_but_catches_localised_dither() {
        let (w, h) = (64u32, 64u32);
        let mut seed = 99u32;
        let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); ((seed >> 16) & 0xFF) as f32 };

        // Half smooth gradient, half dither — the case whole-frame
        // correlation misses, because the smooth half carries the average.
        let mut half = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                half[y * w as usize + x] = if x < 32 { (x * 4) as f32 } else { rnd() };
            }
        }
        assert!(noise_tile_fraction(&half, w, h) > MAX_NOISE_TILE_FRACTION,
            "half-dither frame must be rejected, got {}", noise_tile_fraction(&half, w, h));

        // Structure with a large FLAT region — must NOT be called noise.
        let mut with_flat = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                with_flat[y * w as usize + x] = if x < 32 { 40.0 } else { (x * 4) as f32 };
            }
        }
        assert_eq!(noise_tile_fraction(&with_flat, w, h), 0.0,
            "flat regions are empty, not noisy — must not trigger the gate");
    }

    #[test]
    fn spatial_coherence_of_a_flat_field_is_zero_not_nan() {
        // Zero variance would divide by zero; a flat frame is also worthless,
        // so 0.0 is both safe and semantically right.
        let flat = vec![7.0f32; 32 * 32];
        let c = spatial_coherence(&flat, 32, 32);
        assert_eq!(c, 0.0);
        assert!(c.is_finite());
    }

    #[test]
    fn spatial_coherence_handles_degenerate_dimensions() {
        assert_eq!(spatial_coherence(&[], 0, 0), 0.0);
        assert_eq!(spatial_coherence(&[1.0], 1, 1), 0.0);
        // Field shorter than w*h must not panic.
        assert_eq!(spatial_coherence(&[1.0, 2.0], 64, 64), 0.0);
    }
}

/// Multiscale structured entropy: geometric mean of fine-scale (full res) and
/// coarse-scale (4× average-pool) PNG compression entropy.
///
/// Key property: noise averages to near-uniform at coarse scale → coarse PNG
/// entropy collapses → geometric mean collapses. Structured fractals stay complex
/// at every scale → both terms remain high → product stays high. This directly
/// penalises granular noise while preserving reward for genuine visual complexity.
pub fn multiscale_entropy(
    escape_times: &[f32], width: u32, height: u32, max_iter: u32, colormap: &str,
) -> f32 {
    let w = width as usize;
    let h = height as usize;

    // Fine-scale: existing PNG metric
    let fine = png_compression_entropy(escape_times, width, height, max_iter, colormap);

    // Coarse-scale: 4× average-pool (64px → 16px)
    const FACTOR: usize = 4;
    let cw = (w / FACTOR).max(1);
    let ch = (h / FACTOR).max(1);
    let mut coarse = vec![0.0f32; cw * ch];
    for ty in 0..ch {
        for tx in 0..cw {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for dy in 0..FACTOR {
                for dx in 0..FACTOR {
                    let py = (ty * FACTOR + dy).min(h.saturating_sub(1));
                    let px = (tx * FACTOR + dx).min(w.saturating_sub(1));
                    sum += escape_times[py * w + px];
                    count += 1;
                }
            }
            coarse[ty * cw + tx] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }
    let coarse_ent = png_compression_entropy(
        &coarse, cw as u32, ch as u32, max_iter, colormap,
    );

    // Geometric mean: 0 if either scale is near-uniform, high only when both are rich
    (fine * coarse_ent).sqrt()
}

/// Shannon entropy of escape-time values.
pub fn entropy_from_escape_times(escape_times: &[f32], max_iter: u32) -> f32 {
    let mut bins = vec![0u32; (max_iter + 1) as usize];
    for &t in escape_times {
        let bin = (t as usize).min(max_iter as usize);
        bins[bin] += 1;
    }
    let n = escape_times.len() as f32;
    bins.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n; -p * p.log2() })
        .sum()
}

/// Normalized 32-bin histogram of escape times — behavioral descriptor for novelty search.
/// Each entry is a frequency in [0,1], summing to 1.
pub fn behavior_descriptor(escape_times: &[f32], max_iter: u32) -> Vec<f32> {
    const N_BINS: usize = 32;
    let mut bins = [0u32; N_BINS];
    let scale = N_BINS as f32 / (max_iter as f32 + 1.0);
    for &t in escape_times {
        let bin = ((t * scale) as usize).min(N_BINS - 1);
        bins[bin] += 1;
    }
    let n = escape_times.len() as f32;
    bins.iter().map(|&c| c as f32 / n).collect()
}

/// Average L2 distance to k nearest neighbors in the archive.
/// Returns a value in roughly [0, 1] (histogram L2 distance between two distributions).
pub fn novelty_score(descriptor: &[f32], archive: &[Vec<f32>], k: usize) -> f32 {
    if archive.is_empty() {
        return 1.0;
    }
    let mut dists: Vec<f32> = archive.iter()
        .map(|d| {
            descriptor.iter().zip(d.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(dists.len());
    dists[..k].iter().sum::<f32>() / k as f32
}

/// Returns true when >95% of pixels have the same escape time — degenerate/boring.
pub fn is_degenerate(escape_times: &[f32]) -> bool {
    if escape_times.is_empty() { return true; }
    let first = escape_times[0];
    let same = escape_times.iter().filter(|&&t| (t - first).abs() < 0.5).count();
    same as f32 / escape_times.len() as f32 > 0.95
}

/// Per-component breakdown of the beauty score.
#[derive(Clone, Debug, Default)]
pub struct BeautyBreakdown {
    pub boundary:  f32,
    pub edge:      f32,
    pub entropy:   f32,
    pub self_sim:  f32,
    pub cool_zone: f32,
}

impl BeautyBreakdown {
    pub fn total(&self) -> f32 {
        0.20 * self.boundary + 0.25 * self.edge + 0.20 * self.entropy
            + 0.15 * self.self_sim + 0.20 * self.cool_zone
    }
}

/// Full beauty score returning both composite and per-component breakdown.
pub fn beauty_score_full(escape_times: &[f32], width: usize, max_iter: u32) -> (f32, BeautyBreakdown) {
    let n = escape_times.len();
    let height = n / width.max(1);
    if n == 0 || height == 0 { return (0.0, BeautyBreakdown::default()); }
    let max = max_iter as f32;

    let boundary_frac = escape_times.iter()
        .filter(|&&t| t > max * 0.05 && t < max * 0.90)
        .count() as f32 / n as f32;
    // Holomorphic formulas produce thin high-contrast boundaries (~10-25% of pixels).
    // Recalibrated target: 0.20 (was 0.55 which fit the old per-pixel NN architecture).
    let boundary_score = (1.0 - ((boundary_frac - 0.20) * 1.5_f32).abs()).max(0.0);

    // Smooth-coloring produces fractional escape times; adjacent-pixel differences
    // are typically 0.1–0.5 near the boundary. Old threshold (0.03*max=1.44) was
    // too coarse — captured only ~1.7% of pairs. 0.008*max=0.38 catches fine detail.
    let edge_thresh = max * 0.008;
    let mut edge_count = 0u32;
    let mut total_pairs = 0u32;
    for y in 0..height {
        for x in 0..width {
            let t = escape_times[y * width + x];
            if x + 1 < width {
                if (t - escape_times[y * width + x + 1]).abs() > edge_thresh { edge_count += 1; }
                total_pairs += 1;
            }
            if y + 1 < height {
                if (t - escape_times[(y + 1) * width + x]).abs() > edge_thresh { edge_count += 1; }
                total_pairs += 1;
            }
        }
    }
    let edge_density = edge_count as f32 / total_pairs.max(1) as f32;
    let edge_score   = (edge_density * 4.0).min(1.0);

    const BINS: usize = 32;
    let mut hist = [0u32; BINS];
    for &t in escape_times {
        let b = ((t / max) * (BINS as f32 - 1.0)) as usize;
        hist[b.min(BINS - 1)] += 1;
    }
    let n_f = n as f32;
    let color_entropy: f32 = hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n_f; -p * p.log2() })
        .sum::<f32>() / (BINS as f32).log2();

    let self_sim = {
        let w4 = (width / 4).max(1);
        let h4 = (height / 4).max(1);
        if w4 < 2 || h4 < 2 {
            0.5
        } else {
            let small: Vec<f32> = (0..h4)
                .flat_map(|y| (0..w4).map(move |x| escape_times[y * 4 * width + x * 4]))
                .collect();
            let full_ent  = entropy_from_escape_times(escape_times, max_iter);
            let small_ent = entropy_from_escape_times(&small, max_iter);
            if full_ent > 0.5 { (small_ent / full_ent).min(1.0).max(0.0) } else { 0.0 }
        }
    };

    let cool_frac = escape_times.iter()
        .filter(|&&t| t > max * 0.05 && t < max * 0.40)
        .count() as f32 / n as f32;
    // Recalibrated target: 0.12 (holomorphic formulas produce ~8-15% in cool band).
    let cool_zone_score = (1.0 - ((cool_frac - 0.12) * 3.0).abs()).max(0.0);

    let bd = BeautyBreakdown {
        boundary: boundary_score,
        edge:     edge_score,
        entropy:  color_entropy,
        self_sim,
        cool_zone: cool_zone_score,
    };
    (bd.total(), bd)
}

/// Fractal beauty score in [0, 1].
/// Tuned to correlate with CLIP aesthetic perception: edge density and color entropy
/// are the strongest predictors of perceived visual quality.
pub fn beauty_score(escape_times: &[f32], width: usize, max_iter: u32) -> f32 {
    let n = escape_times.len();
    let height = n / width.max(1);
    if n == 0 || height == 0 { return 0.0; }
    let max = max_iter as f32;

    // 1. Boundary zone fraction: pixels in the detail-rich 5–90% band.
    //    Target ~55%: produces a vivid image with both structure and open space.
    let boundary_frac = escape_times.iter()
        .filter(|&&t| t > max * 0.05 && t < max * 0.90)
        .count() as f32 / n as f32;
    // Holomorphic formulas produce thin high-contrast boundaries (~10-25% of pixels).
    // Recalibrated target: 0.20 (was 0.55 which fit the old per-pixel NN architecture).
    let boundary_score = (1.0 - ((boundary_frac - 0.20) * 1.5_f32).abs()).max(0.0);

    // 2. Edge density: fraction of adjacent pixel pairs with a notable gradient.
    //    This is the #1 predictor of CLIP aesthetic score for fractals.
    //    Rich structure = many local transitions across the image.
    // Smooth-coloring produces fractional escape times; adjacent-pixel differences
    // are typically 0.1–0.5 near the boundary. Old threshold (0.03*max=1.44) was
    // too coarse — captured only ~1.7% of pairs. 0.008*max=0.38 catches fine detail.
    let edge_thresh = max * 0.008;
    let mut edge_count = 0u32;
    let mut total_pairs = 0u32;
    for y in 0..height {
        for x in 0..width {
            let t = escape_times[y * width + x];
            if x + 1 < width {
                if (t - escape_times[y * width + x + 1]).abs() > edge_thresh { edge_count += 1; }
                total_pairs += 1;
            }
            if y + 1 < height {
                if (t - escape_times[(y + 1) * width + x]).abs() > edge_thresh { edge_count += 1; }
                total_pairs += 1;
            }
        }
    }
    let edge_density = edge_count as f32 / total_pairs.max(1) as f32;
    let edge_score   = (edge_density * 4.0).min(1.0); // saturates at 25% edge pairs

    // 3. Color entropy: distribution of escape times across 32 bins.
    //    Entropy-based measure captures true richness (not just bin occupancy).
    const BINS: usize = 32;
    let mut hist = [0u32; BINS];
    for &t in escape_times {
        let b = ((t / max) * (BINS as f32 - 1.0)) as usize;
        hist[b.min(BINS - 1)] += 1;
    }
    let n_f = n as f32;
    let color_entropy: f32 = hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n_f; -p * p.log2() })
        .sum::<f32>() / (BINS as f32).log2();

    // 4. Multi-scale self-similarity: true fractals look complex at every scale.
    let self_sim = {
        let w4 = (width / 4).max(1);
        let h4 = (height / 4).max(1);
        if w4 < 2 || h4 < 2 {
            0.5
        } else {
            let small: Vec<f32> = (0..h4)
                .flat_map(|y| (0..w4).map(move |x| escape_times[y * 4 * width + x * 4]))
                .collect();
            let full_ent  = entropy_from_escape_times(escape_times, max_iter);
            let small_ent = entropy_from_escape_times(&small, max_iter);
            if full_ent > 0.5 { (small_ent / full_ent).min(1.0).max(0.0) } else { 0.0 }
        }
    };

    // 5. Cool-zone score: fraction of pixels in the 5–40% escape range.
    //    With turbo colormap this range maps to blue/cyan — CLIP-preferred aesthetic.
    //    Target ~30% of pixels; penalises both all-interior (boring) and all-exterior
    //    (washed-out) images, rewarding a vivid cool palette.
    let cool_frac = escape_times.iter()
        .filter(|&&t| t > max * 0.05 && t < max * 0.40)
        .count() as f32 / n as f32;
    // Recalibrated target: 0.12 (holomorphic formulas produce ~8-15% in cool band).
    let cool_zone_score = (1.0 - ((cool_frac - 0.12) * 3.0).abs()).max(0.0);

    0.20 * boundary_score + 0.25 * edge_score + 0.20 * color_entropy + 0.15 * self_sim + 0.20 * cool_zone_score
}
