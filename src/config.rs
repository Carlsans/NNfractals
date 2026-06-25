use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub rendering: RenderingConfig,
    pub network: NetworkConfig,
    pub optimization: OptimizationConfig,
    pub output: OutputConfig,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RenderingConfig {
    pub default_width: u32,
    pub default_height: u32,
    pub max_iter: u32,
    pub bailout: f32,
    pub colormap: String,
    pub view_x_min: f32,
    pub view_x_max: f32,
    pub view_y_min: f32,
    pub view_y_max: f32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct NetworkConfig {
    /// Transformer embedding / hidden width (complex-valued neurons).
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    /// Feed-forward multiplier: d_ff = d_model * ff_mult.
    #[serde(default = "default_ff_mult")]
    pub ff_mult: usize,
    pub activation_pool: Vec<String>,
    // kept for config-file compat; value is ignored now that we use the transformer
    #[serde(default)]
    pub hidden_layers: Vec<usize>,
}

fn default_d_model() -> usize { 8 }
fn default_ff_mult()  -> usize { 2 }

#[derive(Deserialize, Clone, Debug)]
pub struct OptimizationConfig {
    pub population_size: usize,
    pub elitism_count: usize,
    pub mutation_rate: f32,
    pub mutation_scale: f32,
    pub activation_mutation_prob: f32,
    pub backprop_steps: u32,
    pub learning_rate: f32,
    pub eval_width: u32,
    pub eval_height: u32,
    pub eval_max_iter: u32,
    /// Max iterations used during backprop (shorter = faster graph, weaker gradient)
    #[serde(default = "default_backprop_max_iter")]
    pub backprop_max_iter: u32,
    pub eval_clamp: f32,
    pub restart_after_gens: u64,
    pub novelty_weight: f32,
    pub novelty_k: usize,
    pub archive_size: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct OutputConfig {
    pub save_dir: PathBuf,
    pub population_dir: PathBuf,
    /// Minimum beauty score [0, 1] for a fractal to be saved.
    #[serde(default = "default_min_beauty")]
    pub min_beauty: f32,
    /// Minimum L2 distance between saved-image behavioral descriptors.
    #[serde(default = "default_min_save_distance")]
    pub min_save_distance: f32,
    // Legacy fields — kept for compat, ignored
    #[serde(default)]
    pub min_entropy: f32,
    #[serde(default)]
    pub min_png_size_kb: u64,
    #[serde(default)]
    pub min_smoothness: f32,
}

fn default_backprop_max_iter() -> u32 { 6 }
fn default_min_beauty()        -> f32 { 0.45 }
fn default_min_save_distance() -> f32 { 0.10 }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn d_ff(&self) -> usize {
        self.network.d_model * self.network.ff_mult
    }
}
