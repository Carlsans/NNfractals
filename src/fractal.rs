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

