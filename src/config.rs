use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub rendering: RenderingConfig,
    pub optimization: OptimizationConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub dedup: DedupConfig,
    #[serde(default)]
    pub mass_extinction: MassExtinctionConfig,
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
    /// seeds for the next epoch: `seed_rank += weight · self_replication`.
    /// Only has any effect when `archive_seeding_enabled = true`.
    #[serde(default = "default_self_replication_weight")]
    pub self_replication_weight: f32,
    /// How strongly to favour fractals with embedded miniature copies of the whole
    /// set (baby-Mandelbrots) when ranking archive seeds for the next epoch:
    /// `seed_rank += weight · fractal_recursion`.
    /// Only has any effect when `archive_seeding_enabled = true`.
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
    /// Out-of-distribution novelty weight: bonus for genomes whose rendered
    /// behavior is far from ALL already-saved genomes (min-distance), pushing
    /// evolution toward fractals unlike anything in the collection. 0 disables.
    #[serde(default = "default_ood_weight")]
    pub ood_weight: f32,
    /// Weight on the human-preference model (pref_score ∈ [0,1], trained via the
    /// browser's ⚖ Rate mode + scripts/train_pref.py). Blended into the saved
    /// fitness and archive-seed ranking: score += pref_weight · pref_score.
    /// 0 disables (inert until a pref model exists and the sidecar emits it).
    #[serde(default = "default_pref_weight")]
    pub pref_weight: f32,
    /// Weight on pref_score specifically when ranking archive genomes to SEED the
    /// initial/restart population. Larger than pref_weight so the individuals that
    /// *start* the group are the highest-pref (best-by-your-taste) genomes. 0 = use
    /// the generic aesthetic ranking for seeding.
    #[serde(default = "default_seed_pref_weight")]
    pub seed_pref_weight: f32,
    /// Weight on MUSIQ technical quality (∈ ~[30,80], normalized to [0,1] as
    /// (musiq-30)/50). Blended positively into saved fitness and seed ranking:
    /// score += musiq_weight · musiq_norm. Kept below pref_weight so pref stays
    /// dominant while high-musiq fractals are actively rewarded (Carl's taste).
    /// 0 disables.
    #[serde(default = "default_musiq_weight")]
    pub musiq_weight: f32,
    /// ITER3: number of high measured-pref recently-saved genomes re-injected into the
    /// breeding pool each generation, so real human-preference steers the search (cheap
    /// geometric fitness is ⊥ taste). 0 disables (pure fitness-elite breeding).
    #[serde(default = "default_pref_elite_count")]
    pub pref_elite_count: u32,
    /// Fraction of a freshly (re)seeded population that's pure-random genomes; the
    /// rest is derived from archive seeds via crossover/mutation (never a plain
    /// clone of a saved genome — see Optimizer::seed_population). Applies at
    /// startup and every stagnation restart. 0 = archive-only fill, 1 = pure random.
    #[serde(default = "default_archive_random_ratio")]
    pub archive_random_ratio: f32,
    /// Weight on the formula-duplication penalty: fitness -= duplicate_penalty_weight
    /// · (uses_in_archive / archive_total)². `uses_in_archive` counts saved genomes
    /// sharing the candidate's structural formula signature (formula_ops_label), so
    /// formula families that already dominate the gallery pay a quadratic price for
    /// more of the same. 0 disables. See FormulaUsageTracker.
    #[serde(default = "default_duplicate_penalty_weight")]
    pub duplicate_penalty_weight: f32,
    /// Whether startup warm-start and stagnation-restart reseed from the
    /// on-disk archive (fractals_N/*.nn, via load_archive_seeds) or build
    /// purely from random/exotic genomes. Off by default: repeated
    /// archive-seeded restarts were found to reinforce whatever formula
    /// family the archive had already converged toward, rather than
    /// escaping it. Set true to restore the original archive-seeded
    /// behavior (kept intact, not removed, for experimentation).
    #[serde(default = "default_archive_seeding_enabled")]
    pub archive_seeding_enabled: bool,
    /// Per-generation bonus for genomes whose bailout exit-angle field
    /// (arg z at escape, DAG genomes only — see fitness::angle_structure_score)
    /// has rich angular structure. fitness += angle_structure_weight ·
    /// angle_structure. 0 disables (default) — the angle buffer is then
    /// skipped entirely on both GPU and CPU (see
    /// render_gpu::render_batch_dag_angle's "free when disabled" design).
    #[serde(default = "default_angle_structure_weight")]
    pub angle_structure_weight: f32,
    /// Weight on image-based novelty (novelty_score ∈ [0,∞): avg L2 distance
    /// to the k nearest archive genomes in a learned DINOv2+VICReg embedding
    /// space — see novelty_scorer.py / scripts/train_novelty.py). Blended
    /// into saved fitness AND archive-seed ranking, alongside
    /// self_replication/fractal_recursion/pref_score: score += weight ·
    /// novelty_score. 0 disables (default, and inert automatically if no
    /// trained head/novelty_scorer.py is present). Distinct from
    /// `novelty_weight` above, which is a different (behavioral-descriptor,
    /// purely synchronous) novelty signal already live in step()'s fitness.
    #[serde(default = "default_img_novelty_weight")]
    pub img_novelty_weight: f32,
    /// Weight on the base complexity term of the per-generation fitness —
    /// `fitness::multiscale_entropy`, the geometric mean of fine (full-res) and
    /// coarse (4x-pooled) PNG-compression entropy. This term used to carry an
    /// implicit, unconfigurable weight of 1.0; it is the anchor every other
    /// weight is implicitly scaled against, so 1.0 remains the default and
    /// changing it rescales the meaning of all of them at once. 0 disables the
    /// only signal that punishes noise — see `multiscale_entropy`'s doc comment.
    #[serde(default = "default_entropy_weight")]
    pub entropy_weight: f32,
    /// Anti-bloat penalty per DAG node: `fitness -= complexity_penalty *
    /// program.len()`, so the GA prefers compact expressions over ones padded
    /// out to noise. Legacy (non-DAG) genomes have an empty program and pay
    /// nothing. Previously a hardcoded `COMPLEXITY_PENALTY` in optimizer.rs.
    #[serde(default = "default_complexity_penalty")]
    pub complexity_penalty: f32,
    /// Weight on the aesthetic score when ranking archive genomes to SEED a
    /// population (`ensemble/10`, else `clip_score`, else `beauty`).
    /// Previously hardcoded to 1.0 in `load_archive_seeds`.
    /// Only has any effect when `archive_seeding_enabled = true`.
    #[serde(default = "default_seed_aesthetic_weight")]
    pub seed_aesthetic_weight: f32,
    /// Weight on normalised LAION (`laion_score/10`) in that same seed ranking.
    /// Previously hardcoded to 0.15. Note LAION is no longer produced by the
    /// scorer sidecar (it returns 0.0), so this is inert for anything saved
    /// after the CLIP/LAION removal and only affects older archive entries.
    #[serde(default = "default_seed_laion_weight")]
    pub seed_laion_weight: f32,
}

impl Default for OptimizationConfig {
    /// NOT used for deserialization — the fields above without a
    /// `#[serde(default)]` stay required, so a config file that omits
    /// `novelty_weight` still fails loudly instead of silently inheriting a
    /// value. This exists purely so the handful of hardcoded `Config` literals
    /// in the codebase can say `..Default::default()` instead of enumerating
    /// every field and drifting out of sync (which `viewer.rs::default_config`
    /// had already done, by up to 0.6 on some weights).
    fn default() -> Self {
        OptimizationConfig {
            population_size: 100,
            elitism_count: 6,
            mutation_rate: 0.20,
            mutation_scale: 0.08,
            eval_width: 64,
            eval_height: 64,
            eval_max_iter: 128,
            restart_after_gens: 20,
            novelty_weight: 0.60,
            novelty_k: 5,
            archive_size: 150,
            self_replication_weight:  default_self_replication_weight(),
            fractal_recursion_weight: default_fractal_recursion_weight(),
            recursion_pred_weight:    default_recursion_pred_weight(),
            formula_diversity_weight: default_formula_diversity_weight(),
            clip_pred_weight:         default_clip_pred_weight(),
            formula_system:           default_formula_system(),
            max_nodes:                default_max_nodes(),
            max_depth:                default_max_depth(),
            ood_weight:               default_ood_weight(),
            pref_weight:              default_pref_weight(),
            seed_pref_weight:         default_seed_pref_weight(),
            musiq_weight:             default_musiq_weight(),
            pref_elite_count:         default_pref_elite_count(),
            archive_random_ratio:     default_archive_random_ratio(),
            duplicate_penalty_weight: default_duplicate_penalty_weight(),
            archive_seeding_enabled:  default_archive_seeding_enabled(),
            angle_structure_weight:   default_angle_structure_weight(),
            img_novelty_weight:       default_img_novelty_weight(),
            entropy_weight:           default_entropy_weight(),
            complexity_penalty:       default_complexity_penalty(),
            seed_aesthetic_weight:    default_seed_aesthetic_weight(),
            seed_laion_weight:        default_seed_laion_weight(),
        }
    }
}

// 0.20, not the 0.35 these keys used to default to: until this commit both were
// parsed and then IGNORED, with `load_archive_seeds` using a hardcoded 0.20 for
// each. 0.20 is therefore what "unchanged behaviour" means for a config that
// omits them. A config that sets them explicitly (all four shipped ones say
// 0.35) now gets the value it asked for.
fn default_self_replication_weight()    -> f32 { 0.20 }
fn default_fractal_recursion_weight()   -> f32 { 0.20 }
fn default_recursion_pred_weight()      -> f32 { 0.60 }
fn default_formula_diversity_weight()   -> f32 { 0.30 }
fn default_clip_pred_weight()           -> f32 { 0.50 }
fn default_formula_system()             -> String { "dag".to_string() }
fn default_max_nodes()                  -> usize { 14 }
fn default_max_depth()                  -> usize { 5 }
fn default_ood_weight()                 -> f32 { 0.0 }
fn default_pref_weight()                -> f32 { 0.4 }
fn default_seed_pref_weight()           -> f32 { 3.0 }
fn default_musiq_weight()               -> f32 { 0.25 }
fn default_pref_elite_count()           -> u32 { 4 }
fn default_archive_random_ratio()       -> f32 { 0.30 }
fn default_duplicate_penalty_weight()   -> f32 { 0.50 }
fn default_archive_seeding_enabled()    -> bool { false }
fn default_angle_structure_weight()     -> f32 { 0.0 }
fn default_img_novelty_weight()         -> f32 { 0.0 }
fn default_entropy_weight()             -> f32 { 1.0 }
fn default_complexity_penalty()         -> f32 { 0.012 }
fn default_seed_aesthetic_weight()      -> f32 { 1.0 }
fn default_seed_laion_weight()          -> f32 { 0.15 }

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
    /// Minimum CLIP zero-shot score [0,1] — legacy Stage-2 fallback gate.
    /// UNREACHABLE IN PRACTICE: the scorer sidecar hardcodes `clip, laion =
    /// 0.0, 0.0` (aesthetic_scorer.py, "CLIP/LAION removed — kept as 0.0 for
    /// protocol stability"), and this branch only runs when the ensemble is
    /// absent or `min_ensemble = 0`. Kept so old configs still parse.
    #[serde(default = "default_min_clip_score")]
    pub min_clip_score: f32,
    /// Minimum LAION MLP aesthetic score [0,10] — same legacy fallback gate,
    /// same caveat as `min_clip_score` above.
    #[serde(default = "default_min_laion_score")]
    pub min_laion_score: f32,
    /// Minimum beauty score [0, 1] for a fractal to be saved (fallback when CLIP unavailable).
    #[serde(default = "default_min_beauty")]
    pub min_beauty: f32,
    /// Minimum L2 distance between saved-image behavioral descriptors.
    #[serde(default = "default_min_save_distance")]
    pub min_save_distance: f32,
    /// Minimum ensemble aesthetic score = mean(nima, topiq_iaa, ap25) [~1,10] for a
    /// fractal to pass the Stage-2 gate. These fractal-tuned models discriminate far
    /// better than CLIP/LAION. Applied only when the multi-model sidecar provides them;
    /// otherwise the old clip/laion gate is used. 0 disables (falls back to clip/laion).
    #[serde(default = "default_min_ensemble")]
    pub min_ensemble: f32,
    /// Minimum MUSIQ technical-quality score [0,100]; rejects blurry/degenerate saves.
    /// Applied only when musiq is available. 0 disables.
    #[serde(default = "default_min_musiq")]
    pub min_musiq: f32,
    /// Minimum human-preference score [0,1] (pref_score) for a fractal to pass the
    /// gate. Cheap geometric fitness is orthogonal to taste (measured corr≈0.05), so
    /// pref only enters at the gate + seeding — this floor keeps low-taste fractals
    /// out of the gallery. Applied only when pref is scored (>0). 0 disables.
    #[serde(default = "default_min_pref")]
    pub min_pref: f32,
}

impl Default for OutputConfig {
    /// See `OptimizationConfig::default` — for `..Default::default()` in
    /// hardcoded `Config` literals only, not for deserialization. `save_dir`
    /// and `population_dir` stay required in a real config file.
    fn default() -> Self {
        OutputConfig {
            save_dir:              PathBuf::from("./fractals_1"),
            population_dir:        PathBuf::from("./populations_1"),
            min_entropy_prefilter: default_min_entropy_prefilter(),
            max_entropy_prefilter: default_max_entropy_prefilter(),
            min_clip_score:        default_min_clip_score(),
            min_laion_score:       default_min_laion_score(),
            min_beauty:            default_min_beauty(),
            min_save_distance:     default_min_save_distance(),
            min_ensemble:          default_min_ensemble(),
            min_musiq:             default_min_musiq(),
            min_pref:              default_min_pref(),
        }
    }
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

/// Rare, randomly-triggered population wipe — independent of, and coexists
/// with, the stagnation-restart mechanism. Modeled as a Poisson process: each
/// generation fires with probability derived from `events_per_day`, so the
/// real-world rate stays roughly constant regardless of how fast/slow
/// generations run. When it fires: force-saves the current best-ever genome
/// if not already saved (file-level safety net only), keeps a
/// uniformly-random, fitness-BLIND survivor fraction of the population, and
/// refills every other slot with pure random/exotic genomes (no breeding, no
/// archive-seeding, regardless of archive_seeding_enabled). Resets
/// stagnant_gens afterward.
#[derive(Deserialize, Clone, Debug)]
pub struct MassExtinctionConfig {
    /// Expected number of mass-extinction events per real-world day, averaged
    /// over the run's lifetime. 0 disables the feature entirely.
    #[serde(default = "default_mass_extinction_events_per_day")]
    pub events_per_day: f32,
    /// Minimum fraction of the population kept (uniformly at random,
    /// independent of fitness) when a mass extinction fires.
    #[serde(default = "default_mass_extinction_min_survivor_frac")]
    pub min_survivor_frac: f32,
    /// Maximum fraction kept. Actual kept fraction is resampled uniformly in
    /// [min_survivor_frac, max_survivor_frac] fresh on every event.
    #[serde(default = "default_mass_extinction_max_survivor_frac")]
    pub max_survivor_frac: f32,
}

impl Default for MassExtinctionConfig {
    fn default() -> Self {
        MassExtinctionConfig {
            events_per_day:    default_mass_extinction_events_per_day(),
            min_survivor_frac: default_mass_extinction_min_survivor_frac(),
            max_survivor_frac: default_mass_extinction_max_survivor_frac(),
        }
    }
}

fn default_dedup_threshold()        -> f32 { 0.97 }
fn default_dedup_interval_hours()   -> f32 { 2.0 }
fn default_mass_extinction_events_per_day()    -> f32 { 3.0 }
fn default_mass_extinction_min_survivor_frac() -> f32 { 0.0 }
fn default_mass_extinction_max_survivor_frac() -> f32 { 0.03 }
fn default_min_beauty()             -> f32 { 0.45 }
fn default_min_save_distance()      -> f32 { 0.10 }
fn default_min_entropy_prefilter()  -> f32 { 0.20 }
fn default_max_entropy_prefilter()  -> f32 { 0.65 }
fn default_min_clip_score()         -> f32 { 0.49 }
fn default_min_laion_score()        -> f32 { 5.15 }
fn default_min_ensemble()           -> f32 { 4.6 }
fn default_min_musiq()              -> f32 { 30.0 }
fn default_min_pref()               -> f32 { 0.45 }

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}
