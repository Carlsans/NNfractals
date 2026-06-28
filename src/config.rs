use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub rendering: RenderingConfig,
    pub optimization: OptimizationConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub dedup: DedupConfig,
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
pub struct OptimizationConfig {
    pub population_size: usize,
    pub elitism_count: usize,
    pub mutation_rate: f32,
    pub mutation_scale: f32,
    pub eval_width: u32,
    pub eval_height: u32,
    pub eval_max_iter: u32,
    pub restart_after_gens: u64,
    pub novelty_weight: f32,
    pub novelty_k: usize,
    pub archive_size: usize,
    /// How strongly to favour zoom-self-replicating fractals when ranking archive
    /// seeds for the next epoch. seed_rank = beauty + weight · self_replication.
    #[serde(default = "default_self_replication_weight")]
    pub self_replication_weight: f32,
    /// How strongly to favour fractals with embedded miniature copies of the whole
    /// set (baby-Mandelbrots) when ranking archive seeds for the next epoch.
    /// seed_rank += weight · fractal_recursion.
    #[serde(default = "default_fractal_recursion_weight")]
    pub fractal_recursion_weight: f32,
    /// Major per-generation selection weight on the formula-only predicted
    /// recursion (RecursionModel). fitness = png_entropy + novelty·nw +
    /// recursion_pred_weight · predicted_recursion. 0 disables.
    #[serde(default = "default_recursion_pred_weight")]
    pub recursion_pred_weight: f32,
    /// Per-generation bonus for genomes whose formula is structurally distant
    /// from the recent archive (k-NN in normalised 58-dim basis-weight space).
    /// fitness += formula_diversity_weight · formula_diversity.
    /// 0 disables. Tuned across diversity loop iterations.
    #[serde(default = "default_formula_diversity_weight")]
    pub formula_diversity_weight: f32,
    /// Per-generation bonus for genomes whose formula is predicted (by a
    /// lightweight linear model trained on the archive) to produce high CLIP
    /// aesthetic score. fitness += clip_pred_weight · pred_clip.
    /// 0 disables. Set to 0 if clip_model.json is absent (criterion inert).
    #[serde(default = "default_clip_pred_weight")]
    pub clip_pred_weight: f32,
    /// Formula representation: "legacy" (flat 58-basis weighted sum) or "dag"
    /// (evolvable expression-DAG / genetic programming). Phase-1 rollout flag.
    #[serde(default = "default_formula_system")]
    pub formula_system: String,
    /// Max nodes in an evolved DAG program (≤ N_SLOTS=16).
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    /// Max depth of an evolved DAG program (tameness vs wildness dial).
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
}

fn default_self_replication_weight()    -> f32 { 0.35 }
fn default_fractal_recursion_weight()   -> f32 { 0.35 }
fn default_recursion_pred_weight()      -> f32 { 0.60 }
fn default_formula_diversity_weight()   -> f32 { 0.30 }
fn default_clip_pred_weight()           -> f32 { 0.50 }
fn default_formula_system()             -> String { "legacy".to_string() }
fn default_max_nodes()                  -> usize { 14 }
fn default_max_depth()                  -> usize { 5 }

#[derive(Deserialize, Clone, Debug)]
pub struct OutputConfig {
    pub save_dir: PathBuf,
    pub population_dir: PathBuf,
    /// Minimum entropy score [0,1] for a genome to pass the fast prefilter (Stage 1).
    #[serde(default = "default_min_entropy_prefilter")]
    pub min_entropy_prefilter: f32,
    /// Maximum PNG entropy [0,1+]. Above this a fractal is salt-and-pepper noise
    /// (high compression entropy but visually incoherent → poor CLIP). Rejected.
    #[serde(default = "default_max_entropy_prefilter")]
    pub max_entropy_prefilter: f32,
    /// Minimum CLIP zero-shot score [0,1] (Stage 2). Both clip AND laion must pass.
    #[serde(default = "default_min_clip_score")]
    pub min_clip_score: f32,
    /// Minimum LAION MLP aesthetic score [0,10] (Stage 2).
    #[serde(default = "default_min_laion_score")]
    pub min_laion_score: f32,
    /// Minimum beauty score [0, 1] for a fractal to be saved (fallback when CLIP unavailable).
    #[serde(default = "default_min_beauty")]
    pub min_beauty: f32,
    /// Minimum L2 distance between saved-image behavioral descriptors.
    #[serde(default = "default_min_save_distance")]
    pub min_save_distance: f32,
}

/// Periodic near-duplicate cleanup run by the evolution loop.
#[derive(Deserialize, Clone, Debug)]
pub struct DedupConfig {
    /// Multi-scale DCT similarity cutoff [0,1]; pairs above this are deduplicated.
    #[serde(default = "default_dedup_threshold")]
    pub similarity_threshold: f32,
    /// Hours between automatic cleanup passes during evolution. 0 disables it.
    #[serde(default = "default_dedup_interval_hours")]
    pub interval_hours: f32,
}

impl Default for DedupConfig {
    fn default() -> Self {
        DedupConfig {
            similarity_threshold: default_dedup_threshold(),
            interval_hours:       default_dedup_interval_hours(),
        }
    }
}

fn default_dedup_threshold()        -> f32 { 0.97 }
fn default_dedup_interval_hours()   -> f32 { 2.0 }
fn default_min_beauty()             -> f32 { 0.45 }
fn default_min_save_distance()      -> f32 { 0.10 }
fn default_min_entropy_prefilter()  -> f32 { 0.20 }
fn default_max_entropy_prefilter()  -> f32 { 0.65 }
fn default_min_clip_score()         -> f32 { 0.49 }
fn default_min_laion_score()        -> f32 { 5.15 }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}
