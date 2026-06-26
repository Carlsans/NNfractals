use rayon::prelude::*;
use crate::config::Config;
use crate::genome::Genome;
use crate::formula::apply_formula;
#[cfg(feature = "wgpu-backend")]
use crate::render_gpu;

/// CPU fitness: (png_compression_entropy, behavioral_descriptor).
/// Fitness = PNG compressed size / raw size at eval resolution.
/// A fractal with rich visual structure compresses poorly → high score.
/// This directly measures what makes fractals aesthetically interesting.
pub fn evaluate_fitness_full(genome: &Genome, config: &Config) -> (f32, Vec<f32>) {
    let ew = config.optimization.eval_width;
    let eh = config.optimization.eval_height;
    let escape_times = render_cpu_iter(genome, config, ew, eh, config.optimization.eval_max_iter);
    let fitness = crate::fitness::png_compression_entropy(
        &escape_times, ew, eh,
        config.optimization.eval_max_iter,
        &config.rendering.colormap,
    );
    let descriptor = crate::fitness::behavior_descriptor(&escape_times, config.optimization.eval_max_iter);
    (fitness, descriptor)
}

/// CPU rendering — returns smooth escape times [H*W].
pub fn render_cpu(genome: &Genome, config: &Config, width: u32, height: u32) -> Vec<f32> {
    render_cpu_iter(genome, config, width, height, config.rendering.max_iter)
}

pub fn render_cpu_iter(
    genome: &Genome, config: &Config, width: u32, height: u32, max_iter: u32,
) -> Vec<f32> {
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;
    let (xmin, xmax, ymin, ymax) = genome.view_bounds();
    let fw = genome.formula_weights();

    // Try GPU first (batch API, but single-genome path).
    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        return render_gpu::render_fractal(
            &fw, width, height, max_iter,
            xmin, xmax, ymin, ymax, bailout_sq,
        );
    }

    // CPU fallback: parallel over pixels via Rayon.
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n  = (width * height) as usize;
    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = apply_formula(&fw, zx, zy, cx, cy);
            zx = nzx; zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let log2_mod = mod_sq.log2() * 0.5;
                let nu = log2_mod.log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
            if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
        }
        max_iter as f32
    }).collect()
}

/// Render explicit fractal-plane bounds (GPU when available, else CPU) → escape times.
pub fn render_bounds(
    fw: &[(f32, f32)], config: &Config, width: u32, height: u32, max_iter: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
) -> Vec<f32> {
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;

    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        return render_gpu::render_fractal(fw, width, height, max_iter, xmin, xmax, ymin, ymax, bailout_sq);
    }

    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n  = (width * height) as usize;
    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = apply_formula(fw, zx, zy, cx, cy);
            zx = nzx; zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let nu = (mod_sq.log2() * 0.5).log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
            if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
        }
        max_iter as f32
    }).collect()
}

/// Down-pool an escape-time field to a `PS×PS` contrast-normalised (z-scored) vector.
/// Z-scoring makes it invariant to the overall escape-time offset, which rises with
/// zoom depth — what survives is the *shape* of the structure.
fn structure_vec(field: &[f32], w: usize, h: usize, ps: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; ps * ps];
    for py in 0..ps {
        for px in 0..ps {
            let x0 = px * w / ps;
            let x1 = ((px + 1) * w / ps).max(x0 + 1).min(w);
            let y0 = py * h / ps;
            let y1 = ((py + 1) * h / ps).max(y0 + 1).min(h);
            let mut sum = 0.0f32;
            let mut cnt = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += field[y * w + x];
                    cnt += 1;
                }
            }
            pooled[py * ps + px] = sum / cnt.max(1) as f32;
        }
    }
    let n    = pooled.len() as f32;
    let mean = pooled.iter().sum::<f32>() / n;
    let var  = pooled.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std  = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; ps * ps]; // degenerate / flat → zero vector → zero correlation
    }
    pooled.iter().map(|v| (v - mean) / std).collect()
}

/// Pearson correlation of two equal-length z-scored vectors → [-1, 1].
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let n = a.len() as f32;
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / n
}

/// Fraction of adjacent pixel pairs whose escape time jumps by a notable amount —
/// a scale-free measure of how much boundary/structure is present in a frame.
fn edge_density(field: &[f32], w: usize, h: usize) -> f32 {
    if field.len() < 4 { return 0.0; }
    let maxv = field.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
    let thr  = maxv * 0.01;
    let mut edges = 0u32;
    let mut total = 0u32;
    for y in 0..h {
        for x in 0..w {
            let t = field[y * w + x];
            if x + 1 < w { if (t - field[y * w + x + 1]).abs() > thr { edges += 1; } total += 1; }
            if y + 1 < h { if (t - field[(y + 1) * w + x]).abs() > thr { edges += 1; } total += 1; }
        }
    }
    edges as f32 / total.max(1) as f32
}

/// Richest interior boundary point: the highest-gradient pixel within the central
/// 70% of the frame (so the next zoom has room around it). Returns pixel coords.
fn richest_boundary_point(field: &[f32], w: usize, h: usize) -> Option<(usize, usize)> {
    let maxv = field.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
    let (mx0, mx1) = (w * 15 / 100, w * 85 / 100);
    let (my0, my1) = (h * 15 / 100, h * 85 / 100);
    let mut best = (-1.0f32, 0usize, 0usize);
    for py in my0.max(1)..my1.min(h - 1) {
        for px in mx0.max(1)..mx1.min(w - 1) {
            let t = field[py * w + px];
            if t < maxv * 0.03 || t > maxv * 0.98 { continue; }
            let gx = (field[py * w + px + 1] - field[py * w + px - 1]).abs();
            let gy = (field[(py + 1) * w + px] - field[(py - 1) * w + px]).abs();
            let g  = gx + gy;
            if g > best.0 { best = (g, px, py); }
        }
    }
    if best.0 < 0.0 { None } else { Some((best.1, best.2)) }
}

/// Zoom self-replication score in [0, 1].
///
/// Does the fractal keep reproducing rich structure as you zoom into its
/// boundary? (The defining Mandelbrot property — infinite self-similar detail.)
///
/// Method: render the base view, then repeatedly zoom in — at each level
/// re-centring on the richest boundary point and rendering deeper. We track:
///  1. **Boundary persistence** — does edge density stay high at depth, or
///     smooth away (non-fractal)?
///  2. **Cross-scale shape correlation** — does each zoom level structurally
///     resemble the previous (contrast-normalised), i.e. self-similar?
/// The score combines the two. A smooth/degenerate map collapses to ≈0; a
/// fractal that stays complex and self-similar across scales approaches 1.
pub fn self_replication_score(genome: &Genome, config: &Config) -> f32 {
    const RES: u32   = 96;
    const PS:  usize = 32;
    const LEVELS: usize = 5;     // base + 4 deeper zooms
    const ZOOM_STEP: f32 = 5.0;  // 5⁴ ≈ 625× total depth

    let fw = genome.formula_weights();
    let mi = config.rendering.max_iter;
    let (w, h) = (RES as usize, RES as usize);

    let (x0, x1, y0, y1) = genome.view_bounds();
    let mut cx   = (x0 + x1) * 0.5;
    let mut cy   = (y0 + y1) * 0.5;
    let mut half = (x1 - x0) * 0.5;

    let mut edge_dens:  Vec<f32>      = Vec::with_capacity(LEVELS);
    let mut structs:    Vec<Vec<f32>> = Vec::with_capacity(LEVELS);

    for level in 0..LEVELS {
        let field = render_bounds(&fw, config, RES, RES, mi,
                                  cx - half, cx + half, cy - half, cy + half);
        edge_dens.push(edge_density(&field, w, h));
        structs.push(structure_vec(&field, w, h, PS));

        // Re-centre on the richest boundary point for the next (deeper) zoom.
        if level + 1 < LEVELS {
            match richest_boundary_point(&field, w, h) {
                Some((px, py)) => {
                    let wf = (w - 1).max(1) as f32;
                    let hf = (h - 1).max(1) as f32;
                    cx = (cx - half) + (px as f32 / wf) * (2.0 * half);
                    cy = (cy - half) + (py as f32 / hf) * (2.0 * half);
                    half /= ZOOM_STEP;
                }
                // No boundary left to follow → structure smoothed out; stop early.
                None => break,
            }
        }
    }

    if structs.is_empty() { return 0.0; }
    let base_ed = edge_dens[0].max(1e-4);
    if base_ed < 0.01 { return 0.0; } // base has essentially no structure

    // 1. Persistence: average retained edge density at each deeper level vs base.
    //    Reaching fewer than LEVELS levels (broke early) counts the missing levels
    //    as zero — a fractal that smooths out is correctly penalised.
    let mut persist = 0.0f32;
    for i in 1..LEVELS {
        let ratio = edge_dens.get(i).copied().unwrap_or(0.0) / base_ed;
        persist += ratio.min(1.0);
    }
    persist /= (LEVELS - 1) as f32;

    // 2. Self-similarity: mean positive correlation between consecutive scales.
    let mut corr = 0.0f32;
    let mut pairs = 0u32;
    for i in 0..structs.len().saturating_sub(1) {
        corr += correlation(&structs[i], &structs[i + 1]).max(0.0);
        pairs += 1;
    }
    if pairs > 0 { corr /= pairs as f32; }

    // Persistence is the dominant signal (it directly measures "stays complex at
    // depth"); shape correlation refines it. Weighted blend, clamped to [0,1].
    (0.65 * persist + 0.35 * corr).clamp(0.0, 1.0)
}

/// Batch-evaluate all genomes in ONE GPU dispatch with per-genome view bounds.
#[cfg(feature = "wgpu-backend")]
pub fn evaluate_fitness_batch(
    genomes: &[crate::genome::Genome],
    config:  &Config,
) -> Vec<(f32, Vec<f32>)> {
    let ew  = config.optimization.eval_width;
    let eh  = config.optimization.eval_height;
    let emi = config.optimization.eval_max_iter;
    let bsq = config.rendering.bailout * config.rendering.bailout;

    let fw_vecs: Vec<Vec<(f32,f32)>> = genomes.iter()
        .map(|g| g.formula_weights())
        .collect();
    let views: Vec<(f32,f32,f32,f32)> = genomes.iter()
        .map(|g| { let (a,b,c,d) = g.view_bounds(); (a,b,c,d) })
        .collect();

    let fw_refs: Vec<&[(f32,f32)]> = fw_vecs.iter().map(|v| v.as_slice()).collect();
    let escape_batch = render_gpu::render_batch(&fw_refs, &views, ew, eh, emi, bsq);

    // Parallelize PNG encoding across all CPU cores while GPU is idle post-dispatch
    escape_batch.into_par_iter().map(|et| {
        let fitness = crate::fitness::png_compression_entropy(
            &et, ew, eh, emi, &config.rendering.colormap,
        );
        let desc = crate::fitness::behavior_descriptor(&et, emi);
        (fitness, desc)
    }).collect()
}

