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
