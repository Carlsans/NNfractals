use burn::tensor::{Tensor, TensorData, backend::Backend};
use rayon::prelude::*;
use crate::config::Config;
use crate::genome::Genome;
use crate::formula::{apply_formula, apply_formula_tensor};
use crate::transformer::transformer_forward_latent;

/// Build a [H*W, 2] tensor of (cx, cy) pixel coordinates using a genome's evolved view.
pub fn pixel_coords<B: Backend>(
    width: u32, height: u32, genome: &Genome, device: &B::Device,
) -> Tensor<B, 2> {
    let (xmin, xmax, ymin, ymax) = genome.view_bounds();
    pixel_coords_raw::<B>(width, height, xmin, xmax, ymin, ymax, device)
}

fn pixel_coords_raw<B: Backend>(
    width: u32, height: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let n  = (width * height) as usize;
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let mut data = Vec::with_capacity(n * 2);
    for py in 0..height {
        for px in 0..width {
            data.push(xmin + (px as f32 / wf) * (xmax - xmin));
            data.push(ymin + (py as f32 / hf) * (ymax - ymin));
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [n, 2]), device)
}

/// CPU fitness: (beauty, behavioral_descriptor).
pub fn evaluate_fitness_full(genome: &Genome, config: &Config) -> (f32, Vec<f32>) {
    let w = config.optimization.eval_width as usize;
    let escape_times = render_cpu_iter(
        genome, config,
        config.optimization.eval_width,
        config.optimization.eval_height,
        config.optimization.eval_max_iter,
    );
    let beauty     = crate::fitness::beauty_score(&escape_times, w, config.optimization.eval_max_iter);
    let descriptor = crate::fitness::behavior_descriptor(&escape_times, config.optimization.eval_max_iter);
    (beauty, descriptor)
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
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n = (width * height) as usize;

    // Run transformer ONCE per genome to get formula weights
    let fw = transformer_forward_latent(&genome.transformer, &genome.latent);

    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);

        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = apply_formula(&fw, zx, zy, cx, cy);
            zx = nzx;
            zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let log2_mod = mod_sq.log2() * 0.5;
                let nu = log2_mod.log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
            // Prevent NaN propagation
            if !zx.is_finite() || !zy.is_finite() {
                return iter as f32;
            }
        }
        max_iter as f32
    }).collect()
}

/// Tensor formula step for backprop.
/// fw_re/fw_im: [1, N_BASIS]; z/c: [N, 2] → returns [N, 2].
pub fn formula_step_tensor<B: Backend>(
    fw_re: &Tensor<B, 2>,
    fw_im: &Tensor<B, 2>,
    z: &Tensor<B, 2>,
    c: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    apply_formula_tensor(fw_re, fw_im, z, c)
}
