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

/// Fraction of adjacent pixel pairs with similar escape times (< 2-iteration difference).
/// High score = smooth fractal boundaries; low score = granular/chaotic noise.
/// A smooth Mandelbrot set scores ~0.85; a random noise image scores ~0.4.
pub fn smoothness_score(escape_times: &[f32], width: usize) -> f32 {
    let n = escape_times.len();
    if n == 0 { return 0.0; }
    let height = n / width;
    let mut smooth = 0u32;
    let mut total = 0u32;
    let threshold = 3.0_f32; // smooth coloring gives fractional values, threshold in iterations

    for y in 0..height {
        for x in 0..width {
            let t = escape_times[y * width + x];
            if x + 1 < width {
                if (t - escape_times[y * width + x + 1]).abs() < threshold { smooth += 1; }
                total += 1;
            }
            if y + 1 < height {
                if (t - escape_times[(y + 1) * width + x]).abs() < threshold { smooth += 1; }
                total += 1;
            }
        }
    }
    smooth as f32 / total.max(1) as f32
}

/// Returns true when >95% of pixels have the same escape time — degenerate/boring.
pub fn is_degenerate(escape_times: &[f32]) -> bool {
    if escape_times.is_empty() { return true; }
    let first = escape_times[0];
    let same = escape_times.iter().filter(|&&t| (t - first).abs() < 0.5).count();
    same as f32 / escape_times.len() as f32 > 0.95
}

/// Fractal beauty score in [0, 1].
/// Combines boundary richness, gradient coherence, multi-scale self-similarity,
/// and color spread — all computable from the escape-time array alone.
pub fn beauty_score(escape_times: &[f32], width: usize, max_iter: u32) -> f32 {
    let n = escape_times.len();
    let height = n / width.max(1);
    if n == 0 || height == 0 { return 0.0; }
    let max = max_iter as f32;

    // 1. Boundary richness: pixels in the "interesting" fractal zone (5-95% of max_iter).
    //    These are the pixels near the escape boundary where fractal detail lives.
    let boundary_frac = escape_times.iter()
        .filter(|&&t| t > max * 0.05 && t < max * 0.95)
        .count() as f32 / n as f32;
    // Peak score at ~40%; falls off toward 0% (all interior) or 100% (all escaped)
    let boundary_score = (1.0 - ((boundary_frac - 0.40) * 2.5_f32).abs()).max(0.0);

    // 2. Gradient coherence: ratio of mean gradient to max gradient.
    //    High ratio = smooth, structured boundaries; low ratio = spiky noise.
    let mut total_grad = 0.0f32;
    let mut max_grad = 0.0f32;
    let mut count = 0u32;
    for y in 0..height {
        for x in 0..width {
            let t = escape_times[y * width + x];
            if x + 1 < width {
                let g = (t - escape_times[y * width + x + 1]).abs();
                total_grad += g;
                if g > max_grad { max_grad = g; }
                count += 1;
            }
            if y + 1 < height {
                let g = (t - escape_times[(y + 1) * width + x]).abs();
                total_grad += g;
                if g > max_grad { max_grad = g; }
                count += 1;
            }
        }
    }
    let mean_grad = total_grad / count.max(1) as f32;
    let coherence = if max_grad > 0.0 { (mean_grad / max_grad).min(1.0) } else { 0.0 };
    let activity  = (mean_grad / (max * 0.005)).min(1.0); // non-zero gradient = not flat
    let grad_score = 0.6 * coherence + 0.4 * activity;

    // 3. Multi-scale self-similarity: entropy at 1/4 resolution vs full resolution.
    //    True fractals look complex at every scale, so these should be similar.
    let self_sim = {
        let w4 = width / 4;
        let h4 = height / 4;
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

    // 4. Color spread: fraction of the escape-time range actually used.
    //    Beautiful fractals paint with the full palette, not a few bins.
    const BINS: usize = 32;
    let mut used_bins = [false; BINS];
    for &t in escape_times {
        let b = ((t / max) * (BINS as f32 - 1.0)) as usize;
        used_bins[b.min(BINS - 1)] = true;
    }
    let spread = used_bins.iter().filter(|&&b| b).count() as f32 / BINS as f32;

    0.30 * boundary_score + 0.25 * grad_score + 0.25 * self_sim + 0.20 * spread
}

/// Shannon entropy of |z|² magnitudes (legacy, unused in current fitness path).
pub fn entropy_from_magnitudes(magnitudes: &[f32], eval_clamp: f32) -> f32 {
    const N_BINS: usize = 64;
    let max_val = (eval_clamp * eval_clamp).max(f32::EPSILON);
    let mut bins = [0u32; N_BINS];
    for &m in magnitudes {
        let t = (m / max_val).clamp(0.0, 1.0);
        let bin = ((t * (N_BINS - 1) as f32) as usize).min(N_BINS - 1);
        bins[bin] += 1;
    }
    let n = magnitudes.len() as f32;
    bins.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n; -p * p.log2() })
        .sum()
}
