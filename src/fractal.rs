use burn::tensor::{Tensor, TensorData, backend::Backend};
use rayon::prelude::*;
use crate::config::Config;
use crate::genome::Genome;
use crate::formula::{eval_formula, eval_formula_tensor};
use crate::transformer::transformer_forward_scalar;

/// Build a [H*W, 2] tensor of (cx, cy) pixel coordinates.
pub fn pixel_coords<B: Backend>(
    width: u32, height: u32, config: &Config, device: &B::Device,
) -> Tensor<B, 2> {
    let n = (width * height) as usize;
    let xmin = config.rendering.view_x_min;
    let xmax = config.rendering.view_x_max;
    let ymin = config.rendering.view_y_min;
    let ymax = config.rendering.view_y_max;
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
    let xmin = config.rendering.view_x_min;
    let xmax = config.rendering.view_x_max;
    let ymin = config.rendering.view_y_min;
    let ymax = config.rendering.view_y_max;
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n = (width * height) as usize;

    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);

        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = forward_pixel(genome, zx, zy, cx, cy);
            zx = nzx;
            zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let log2_mod = mod_sq.log2() * 0.5;
                let nu = log2_mod.log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
        }
        max_iter as f32
    }).collect()
}

/// One fractal iteration:
///   z_new = formula(z, c) + nn_blend * tanh_per_component(Transformer(z, c))
fn forward_pixel(genome: &Genome, zx: f32, zy: f32, cx: f32, cy: f32) -> (f32, f32) {
    let (fx, fy) = eval_formula(&genome.formula, zx, zy, cx, cy);
    let (nx, ny) = transformer_forward_scalar(&genome.transformer, zx, zy, cx, cy);
    (fx + genome.nn_blend * nx.tanh(), fy + genome.nn_blend * ny.tanh())
}

/// Tensor formula step for backprop.
pub fn formula_step_tensor<B: Backend>(
    formula: &[crate::formula::ComplexOp],
    z: &Tensor<B, 2>,
    c: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let zx = z.clone().narrow(1, 0, 1);
    let zy = z.clone().narrow(1, 1, 1);
    let cx = c.clone().narrow(1, 0, 1);
    let cy = c.clone().narrow(1, 1, 1);
    let (fx, fy) = eval_formula_tensor(formula, zx, zy, &cx, &cy);
    Tensor::cat(vec![fx, fy], 1)
}
