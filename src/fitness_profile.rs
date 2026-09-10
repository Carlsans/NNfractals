//! Named, user-editable mixes of the evolution fitness weights.
//!
//! A profile is a **sparse overlay**, not a second `Config`: every weight is an
//! `Option<f32>`, `None` means "leave whatever the config file says alone", and
//! only the keys a profile actually cares about are written back out. That is
//! deliberate — the four shipped `config*.toml` files carry a hand-written
//! tuning audit trail (`CYCLE11: population_size=130 RULED OUT PERMANENTLY…`,
//! `ITER4: raised 0.50→0.55…`) on nearly every line, and round-tripping the
//! whole `Config` through `toml::to_string_pretty` would erase all of it. So
//! `Config` deliberately stays `Deserialize`-only and profiles live in their own
//! small files under `profiles/`, applied on top at startup via `--profile`.
//!
//! The field table below is the single source of truth for what a profile can
//! contain: the struct, the TOML schema, the `get`/`set` accessors and the GUI's
//! field list are all generated from it, so adding a knob is one line.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::Config;

/// Which section of the fitness pipeline a knob belongs to. Drives the GUI's
/// grouping and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Per-generation selection fitness — `Optimizer::step`.
    Selection,
    /// Saved-genome fitness and archive-seed ranking.
    Blend,
    /// Save-gate thresholds — pass/fail, not weights.
    Gate,
}

/// How a knob participates in the fitness sum. `Positive` and `Penalty` are the
/// ones that share a scale and can meaningfully be shown as percentages of the
/// mix; `Threshold` values are in their own units and never normalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Positive,
    Penalty,
    Threshold,
}

/// Static metadata for one profile knob.
pub struct FieldSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub group: Group,
    pub sign: Sign,
    /// The value in effect when neither the profile nor the config sets it —
    /// i.e. `config.rs`'s `default_*()` for this key.
    pub default: f32,
    pub help: &'static str,
}

macro_rules! profile_fields {
    ($( $key:ident, $group:expr, $sign:expr, $default:expr, $label:expr, $help:expr );* $(;)?) => {
        /// A named fitness mix. Every weight is optional; absent = inherit the
        /// config file's value.
        #[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
        pub struct FitnessProfile {
            #[serde(default)] pub name: String,
            #[serde(default)] pub notes: String,
            $(
                #[serde(default, skip_serializing_if = "Option::is_none")]
                pub $key: Option<f32>,
            )*
        }

        /// Every knob a profile can carry, in display order.
        pub const FIELDS: &[FieldSpec] = &[
            $( FieldSpec {
                key: stringify!($key), label: $label, group: $group,
                sign: $sign, default: $default, help: $help,
            } ),*
        ];

        impl FitnessProfile {
            /// The profile's own override for `key`, if it sets one.
            pub fn get(&self, key: &str) -> Option<f32> {
                match key { $( stringify!($key) => self.$key, )* _ => None }
            }

            /// Set (or with `None`, clear) the override for `key`. Unknown keys
            /// are ignored — callers iterate [`FIELDS`], so this cannot happen
            /// by accident.
            pub fn set(&mut self, key: &str, v: Option<f32>) {
                match key { $( stringify!($key) => self.$key = v, )* _ => {} }
            }
        }
    };
}

profile_fields! {
    // ── Per-generation selection fitness ───────────────────────────────────
    entropy_weight, Group::Selection, Sign::Positive, 1.0,
        "Entropy (multiscale)",
        "The backbone of selection: geometric mean of fine and 4x-pooled PNG-compression \
         entropy. It is the term every other weight is implicitly scaled against, so 1.0 \
         is the natural reference — changing it rescales the meaning of the whole mix at \
         once. It is also the only term that punishes noise, because noise averages to \
         near-uniform when downsampled and collapses the coarse half.";
    novelty_weight, Group::Selection, Sign::Positive, 0.60,
        "Behavioural novelty",
        "Average L2 distance from this genome's 32-bin escape-time histogram to its k \
         nearest neighbours in a rolling archive of recent genomes. Rewards fractals that \
         BEHAVE unlike what the run has just seen. Raise to fight convergence on one \
         visual family.";
    formula_diversity_weight, Group::Selection, Sign::Positive, 0.30,
        "Formula diversity",
        "The same k-NN novelty measure, but in formula-structure space rather than \
         rendering space. Two different formulas can render to similar histograms, so \
         this rewards genuinely different formula families even when they look alike.";
    ood_weight, Group::Selection, Sign::Positive, 0.0,
        "Out-of-distribution",
        "Distance to the NEAREST already-saved genome (minimum, not mean). Where \
         behavioural novelty asks 'unlike the last 150 evaluated', this asks 'unlike \
         anything in the collection'. Computes to 0 until the run has saved something.";
    angle_structure_weight, Group::Selection, Sign::Positive, 0.0,
        "Angle structure",
        "Richness of the bailout exit-angle field, arg(z) at escape. DAG genomes only. \
         At 0 the angle buffer is skipped entirely on both GPU and CPU, so this knob is \
         genuinely free when off and roughly doubles evaluation cost when on.";
    recursion_pred_weight, Group::Selection, Sign::Positive, 0.60,
        "Predicted recursion  [inert]",
        "Formula-only linear prediction of the baby-Mandelbrot recursion score. INERT for \
         DAG genomes: recursion_model.json was trained on the old 58-basis representation \
         and returns a near-constant (~0.682), which is why every shipped config sets this \
         to 0. Turning it on re-activates a known-broken signal — retrain the model first.";
    clip_pred_weight, Group::Selection, Sign::Positive, 0.50,
        "Predicted CLIP  [inert]",
        "Formula-only linear prediction of the CLIP aesthetic score. INERT for the same \
         reason as predicted recursion (returns ~0.5197), and CLIP was found to collapse \
         on fractals anyway — the sidecar no longer computes it at all.";
    complexity_penalty, Group::Selection, Sign::Penalty, 0.012,
        "Complexity penalty / node",
        "Subtracted per DAG node, so the GA prefers compact expressions over ones padded \
         out until they look busy. Multiscale entropy is the other backstop against that. \
         Legacy (non-DAG) genomes have no program and pay nothing.";
    duplicate_penalty_weight, Group::Selection, Sign::Penalty, 0.50,
        "Duplicate-family penalty",
        "Quadratic in the fraction of the SAVED archive already using this genome's \
         formula signature, so a family that comes to dominate the gallery is actively \
         pushed away from. Distinct from formula diversity, which only sees the recent \
         per-generation window rather than everything on disk.";

    // ── Saved fitness + archive-seed ranking ───────────────────────────────
    pref_weight, Group::Blend, Sign::Positive, 0.4,
        "Human preference",
        "Weight on the trained taste model (browser Rate mode + train_pref.py) in the \
         fitness stored on a saved genome. Cheap geometric fitness is close to orthogonal \
         to taste, so preference enters at the gate and at seeding rather than per \
         generation.";
    musiq_weight, Group::Blend, Sign::Positive, 0.25,
        "MUSIQ technical quality",
        "Normalised as (musiq-30)/50. Rewards sharp, well-formed renders. Kept below \
         preference so taste stays dominant.";
    img_novelty_weight, Group::Blend, Sign::Positive, 0.0,
        "Image-embedding novelty",
        "Distance to the k nearest saved images in a learned DINOv2+VICReg embedding. \
         Needs a trained head and a per-pool .novelty_cache.npz; inert automatically \
         without them. Currently trained for fractals_1 only.";
    seed_pref_weight, Group::Blend, Sign::Positive, 3.0,
        "Seed: preference",
        "Preference weight when ranking archive genomes to SEED a population. \
         Deliberately much larger than the per-save preference weight so the individuals \
         that START a run are the best-by-your-taste ones.";
    seed_aesthetic_weight, Group::Blend, Sign::Positive, 1.0,
        "Seed: aesthetic",
        "Weight on the ensemble aesthetic score (nima/topiq/ap25 mean, /10) in seed \
         ranking. Was hardcoded to 1.0 before this knob existed.";
    seed_laion_weight, Group::Blend, Sign::Positive, 0.15,
        "Seed: LAION  [legacy]",
        "Weight on normalised LAION in seed ranking. Was hardcoded to 0.15. The scorer \
         sidecar no longer produces LAION (returns 0.0), so this only affects genomes \
         saved before that change.";
    self_replication_weight, Group::Blend, Sign::Positive, 0.20,
        "Seed: self-replication",
        "Weight on the measured zoom self-replication score in seed ranking — does the \
         fractal keep producing structure as you zoom its boundary. Until recently this \
         config key was parsed and then ignored, with 0.20 hardcoded in its place.";
    fractal_recursion_weight, Group::Blend, Sign::Positive, 0.20,
        "Seed: fractal recursion",
        "Weight on the measured baby-Mandelbrot recursion score in seed ranking. Same \
         history as self-replication above. Note this measurement is reported to be \
         always 0 for DAG genomes, so verify before relying on it.";

    // ── Save gate ──────────────────────────────────────────────────────────
    min_ensemble, Group::Gate, Sign::Threshold, 4.6,
        "Min ensemble aesthetic",
        "The single most consequential save gate: mean of nima, topiq_iaa and ap25, on \
         roughly a 1-10 scale. Absent from every shipped config, so it has been silently \
         running at its 4.6 code default — state it explicitly here. 0 falls back to the \
         legacy CLIP/LAION gate, which no longer works.";
    min_musiq, Group::Gate, Sign::Threshold, 30.0,
        "Min MUSIQ",
        "Technical-quality floor [0,100]; rejects blurry or degenerate saves. Only \
         applied when MUSIQ is actually scored.";
    min_pref, Group::Gate, Sign::Threshold, 0.45,
        "Min preference",
        "Taste floor [0,1] at the save gate. Only applied once a preference model exists \
         and the sidecar emits a non-zero score.";
    min_entropy_prefilter, Group::Gate, Sign::Threshold, 0.20,
        "Min PNG entropy",
        "Lower edge of the cheap Stage-1 band, on RAW fine-scale PNG entropy (not \
         multiscale). Below this a render is too uniform to be worth scoring.";
    max_entropy_prefilter, Group::Gate, Sign::Threshold, 0.65,
        "Max PNG entropy",
        "Upper edge of the same band. Above this a render is salt-and-pepper noise, which \
         compresses badly for the wrong reason.";
    min_save_distance, Group::Gate, Sign::Threshold, 0.10,
        "Min save distance",
        "Minimum L2 distance from this genome's behavioural descriptor to every \
         previously-saved one. The near-duplicate guard at save time; the periodic DCT \
         dedup pass cleans up whatever slips through.";
    min_beauty, Group::Gate, Sign::Threshold, 0.45,
        "Min beauty  [fallback]",
        "Geometric beauty floor, used ONLY when the aesthetic sidecar is unavailable. \
         Under normal operation this never fires.";
}

impl FitnessProfile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut p: FitnessProfile = toml::from_str(&text)?;
        if p.name.is_empty() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                p.name = stem.to_string();
            }
        }
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Overlay this profile onto `config`. Only keys the profile actually sets
    /// are touched; everything else keeps the config file's value.
    pub fn apply(&self, config: &mut Config) {
        macro_rules! opt { ($f:ident) => { if let Some(v) = self.$f { config.optimization.$f = v; } } }
        macro_rules! out { ($f:ident) => { if let Some(v) = self.$f { config.output.$f = v; } } }

        opt!(entropy_weight);
        opt!(novelty_weight);
        opt!(formula_diversity_weight);
        opt!(ood_weight);
        opt!(angle_structure_weight);
        opt!(recursion_pred_weight);
        opt!(clip_pred_weight);
        opt!(complexity_penalty);
        opt!(duplicate_penalty_weight);
        opt!(pref_weight);
        opt!(musiq_weight);
        opt!(img_novelty_weight);
        opt!(seed_pref_weight);
        opt!(seed_aesthetic_weight);
        opt!(seed_laion_weight);
        opt!(self_replication_weight);
        opt!(fractal_recursion_weight);

        out!(min_ensemble);
        out!(min_musiq);
        out!(min_pref);
        out!(min_entropy_prefilter);
        out!(max_entropy_prefilter);
        out!(min_save_distance);
        out!(min_beauty);
    }

    /// Snapshot every knob's current value out of a `Config` — the "Save As"
    /// path, so a new profile starts from what the run is actually doing rather
    /// than from a pile of blanks.
    pub fn from_config(name: &str, config: &Config) -> Self {
        let mut p = FitnessProfile { name: name.to_string(), ..Default::default() };
        for f in FIELDS {
            p.set(f.key, Some(config_value(config, f.key)));
        }
        p
    }

    /// The value that will actually be in force for `key`: this profile's
    /// override if it has one, otherwise the config's.
    pub fn effective(&self, key: &str, config: &Config) -> f32 {
        self.get(key).unwrap_or_else(|| config_value(config, key))
    }

    /// Each positive selection term's share of the mix, as a percentage. This
    /// is the "proportions" view: the weights stay the raw numbers the code
    /// uses, and the share is derived for display. Penalties and thresholds are
    /// excluded — a penalty has no share of a positive total, and thresholds
    /// aren't on the same scale at all.
    pub fn selection_shares(&self, config: &Config) -> Vec<(&'static str, f32, f32)> {
        let terms: Vec<(&'static str, f32)> = FIELDS.iter()
            .filter(|f| f.group == Group::Selection && f.sign == Sign::Positive)
            .map(|f| (f.key, self.effective(f.key, config)))
            .collect();
        let total: f32 = terms.iter().map(|(_, v)| v.max(0.0)).sum();
        terms.into_iter()
            .map(|(k, v)| {
                let share = if total > 0.0 { v.max(0.0) / total * 100.0 } else { 0.0 };
                (k, v, share)
            })
            .collect()
    }

    /// One-line rendering of the resulting per-generation fitness expression,
    /// for the GUI footer and the evolution binary's startup line. Terms at
    /// exactly 0 are omitted — they contribute nothing and only add noise.
    pub fn fitness_expression(&self, config: &Config) -> String {
        let mut parts: Vec<String> = Vec::new();
        for f in FIELDS.iter().filter(|f| f.group == Group::Selection) {
            let v = self.effective(f.key, config);
            if v == 0.0 { continue; }
            let short = short_name(f.key);
            match f.sign {
                Sign::Penalty => parts.push(format!("− {v:.3}·{short}")),
                _ => parts.push(format!("{}{v:.2}·{short}", if parts.is_empty() { "" } else { "+ " })),
            }
        }
        if parts.is_empty() { return "fitness = 0 (every selection term is zero)".to_string(); }
        format!("fitness = {}", parts.join(" "))
    }
}

/// Read one knob out of a `Config` by key. Kept next to `apply` so the two
/// can't drift.
fn config_value(config: &Config, key: &str) -> f32 {
    let o = &config.optimization;
    let g = &config.output;
    match key {
        "entropy_weight"           => o.entropy_weight,
        "novelty_weight"           => o.novelty_weight,
        "formula_diversity_weight" => o.formula_diversity_weight,
        "ood_weight"               => o.ood_weight,
        "angle_structure_weight"   => o.angle_structure_weight,
        "recursion_pred_weight"    => o.recursion_pred_weight,
        "clip_pred_weight"         => o.clip_pred_weight,
        "complexity_penalty"       => o.complexity_penalty,
        "duplicate_penalty_weight" => o.duplicate_penalty_weight,
        "pref_weight"              => o.pref_weight,
        "musiq_weight"             => o.musiq_weight,
        "img_novelty_weight"       => o.img_novelty_weight,
        "seed_pref_weight"         => o.seed_pref_weight,
        "seed_aesthetic_weight"    => o.seed_aesthetic_weight,
        "seed_laion_weight"        => o.seed_laion_weight,
        "self_replication_weight"  => o.self_replication_weight,
        "fractal_recursion_weight" => o.fractal_recursion_weight,
        "min_ensemble"             => g.min_ensemble,
        "min_musiq"                => g.min_musiq,
        "min_pref"                 => g.min_pref,
        "min_entropy_prefilter"    => g.min_entropy_prefilter,
        "max_entropy_prefilter"    => g.max_entropy_prefilter,
        "min_save_distance"        => g.min_save_distance,
        "min_beauty"               => g.min_beauty,
        _ => 0.0,
    }
}

fn short_name(key: &str) -> &'static str {
    match key {
        "entropy_weight"           => "entropy",
        "novelty_weight"           => "novelty",
        "formula_diversity_weight" => "fdiv",
        "ood_weight"               => "ood",
        "angle_structure_weight"   => "angle",
        "recursion_pred_weight"    => "pred_rec",
        "clip_pred_weight"         => "pred_clip",
        "complexity_penalty"       => "nodes",
        "duplicate_penalty_weight" => "dup²",
        _ => "?",
    }
}

/// `<root>/profiles` — where profile files live.
pub fn profiles_dir(root: &Path) -> PathBuf {
    root.join("profiles")
}

/// Every `*.toml` under `profiles/`, as (display name, path), sorted by name.
/// Missing directory is not an error — it just means no profiles yet.
pub fn list_profiles(root: &Path) -> Vec<(String, PathBuf)> {
    let dir = profiles_dir(root);
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<(String, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("toml"))
        .filter_map(|p| {
            p.file_stem().and_then(|s| s.to_str()).map(|s| (s.to_string(), p.clone()))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Resolve a `--profile` argument: either a bare name looked up under
/// `profiles/`, or a literal path to a `.toml` file.
pub fn resolve(root: &Path, arg: &str) -> PathBuf {
    if arg.ends_with(".toml") {
        let direct = PathBuf::from(arg);
        if direct.exists() { return direct; }
    }
    profiles_dir(root).join(format!("{}.toml", arg.trim_end_matches(".toml")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OptimizationConfig, OutputConfig, RenderingConfig, DedupConfig, MassExtinctionConfig};

    fn base_config() -> Config {
        Config {
            dedup: DedupConfig::default(),
            mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig {
                default_width: 512, default_height: 512, max_iter: 192, bailout: 4.0,
                colormap: "turbo".into(), view_x_min: -2.0, view_x_max: 2.0,
                view_y_min: -2.0, view_y_max: 2.0,
            },
            optimization: OptimizationConfig::default(),
            output: OutputConfig::default(),
        }
    }

    #[test]
    fn empty_profile_changes_nothing() {
        let mut c = base_config();
        let before = c.clone();
        FitnessProfile::default().apply(&mut c);
        for f in FIELDS {
            assert_eq!(
                config_value(&c, f.key), config_value(&before, f.key),
                "empty profile must not touch {}", f.key
            );
        }
    }

    #[test]
    fn a_single_override_leaves_every_other_key_alone() {
        let mut c = base_config();
        let before = c.clone();
        let p = FitnessProfile { novelty_weight: Some(0.9), ..Default::default() };
        p.apply(&mut c);
        assert_eq!(c.optimization.novelty_weight, 0.9);
        for f in FIELDS.iter().filter(|f| f.key != "novelty_weight") {
            assert_eq!(
                config_value(&c, f.key), config_value(&before, f.key),
                "{} must be untouched", f.key
            );
        }
    }

    #[test]
    fn every_field_applies_to_the_right_config_slot() {
        // Catches a copy-paste in `apply`'s opt!/out! lists: set each key to a
        // unique sentinel and check it lands where `config_value` reads it.
        for (i, f) in FIELDS.iter().enumerate() {
            let mut c = base_config();
            let sentinel = 100.0 + i as f32;
            let mut p = FitnessProfile::default();
            p.set(f.key, Some(sentinel));
            p.apply(&mut c);
            assert_eq!(config_value(&c, f.key), sentinel, "{} did not apply", f.key);
        }
    }

    #[test]
    fn get_set_round_trip_for_every_field() {
        let mut p = FitnessProfile::default();
        for f in FIELDS {
            assert_eq!(p.get(f.key), None, "{} should start unset", f.key);
            p.set(f.key, Some(1.25));
            assert_eq!(p.get(f.key), Some(1.25), "{} did not round-trip", f.key);
            p.set(f.key, None);
            assert_eq!(p.get(f.key), None, "{} did not clear", f.key);
        }
    }

    #[test]
    fn toml_round_trip_preserves_only_what_was_set() {
        let p = FitnessProfile {
            name: "taste-led".into(),
            notes: "pref dominant".into(),
            pref_weight: Some(0.8),
            novelty_weight: Some(0.35),
            ..Default::default()
        };
        let text = toml::to_string_pretty(&p).unwrap();
        // Sparse: unset keys must not appear at all, or loading would pin them.
        assert!(!text.contains("ood_weight"), "unset keys must be omitted:\n{text}");
        let back: FitnessProfile = toml::from_str(&text).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn unknown_keys_in_a_profile_file_do_not_break_loading() {
        // Forward compatibility: a profile written by a newer build must still
        // load in an older one, minus the knob it doesn't know.
        let text = "name = \"future\"\npref_weight = 0.5\nsome_unknown_future_knob = 12.0\n";
        let p: FitnessProfile = toml::from_str(text).unwrap();
        assert_eq!(p.pref_weight, Some(0.5));
    }

    #[test]
    fn from_config_then_apply_is_the_identity() {
        let c = base_config();
        let mut c2 = base_config();
        FitnessProfile::from_config("snapshot", &c).apply(&mut c2);
        for f in FIELDS {
            assert_eq!(config_value(&c2, f.key), config_value(&c, f.key), "{}", f.key);
        }
    }

    #[test]
    fn defaults_in_the_field_table_match_config_rs() {
        // The table's `default` is what the GUI shows as "inherited". If it
        // drifts from config.rs the GUI silently lies, which is exactly the bug
        // viewer.rs::default_config had.
        let c = base_config();
        for f in FIELDS {
            assert_eq!(
                config_value(&c, f.key), f.default,
                "FIELDS default for {} disagrees with config.rs", f.key
            );
        }
    }

    #[test]
    fn selection_shares_sum_to_100_and_track_the_override() {
        let c = base_config();
        let p = FitnessProfile {
            entropy_weight: Some(1.0),
            novelty_weight: Some(1.0),
            formula_diversity_weight: Some(0.0),
            ood_weight: Some(0.0),
            angle_structure_weight: Some(0.0),
            recursion_pred_weight: Some(0.0),
            clip_pred_weight: Some(0.0),
            ..Default::default()
        };
        let shares = p.selection_shares(&c);
        let total: f32 = shares.iter().map(|(_, _, s)| s).sum();
        assert!((total - 100.0).abs() < 1e-3, "shares must total 100%, got {total}");
        let entropy = shares.iter().find(|(k, _, _)| *k == "entropy_weight").unwrap();
        assert!((entropy.2 - 50.0).abs() < 1e-3, "two equal terms should be 50% each, got {}", entropy.2);
    }

    #[test]
    fn selection_shares_are_all_zero_when_everything_is_off() {
        let c = base_config();
        let mut p = FitnessProfile::default();
        for f in FIELDS.iter().filter(|f| f.group == Group::Selection) {
            p.set(f.key, Some(0.0));
        }
        for (k, _, share) in p.selection_shares(&c) {
            assert_eq!(share, 0.0, "{k} must be 0% not NaN when the mix is empty");
        }
    }

    #[test]
    fn fitness_expression_omits_zero_terms() {
        let c = base_config();
        let p = FitnessProfile {
            entropy_weight: Some(1.0),
            novelty_weight: Some(0.6),
            ood_weight: Some(0.0),
            recursion_pred_weight: Some(0.0),
            clip_pred_weight: Some(0.0),
            angle_structure_weight: Some(0.0),
            formula_diversity_weight: Some(0.0),
            ..Default::default()
        };
        let s = p.fitness_expression(&c);
        assert!(s.contains("entropy"), "{s}");
        assert!(s.contains("novelty"), "{s}");
        assert!(!s.contains("ood"), "zero terms must be omitted: {s}");
    }

    #[test]
    fn resolve_accepts_a_bare_name_and_a_path() {
        let root = Path::new("/tmp/nnf_profiles_test");
        assert_eq!(resolve(root, "taste-led"), root.join("profiles/taste-led.toml"));
        // A name that already carries the extension must not double it up.
        assert_eq!(resolve(root, "taste-led.toml"), root.join("profiles/taste-led.toml"));
    }

    #[test]
    fn list_profiles_on_a_missing_dir_is_empty_not_an_error() {
        assert!(list_profiles(Path::new("/nonexistent/nnfractals/root")).is_empty());
    }

    /// The shipped profiles are the ones Carl will actually load, so they get
    /// checked against the real config files rather than a synthetic Config.
    /// `repo` is the crate root; these tests are inert if the files are absent.
    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn every_shipped_profile_parses() {
        let found = list_profiles(&repo());
        assert!(!found.is_empty(), "no profiles/*.toml found — did they get deleted?");
        for (name, path) in found {
            let p = FitnessProfile::load(&path)
                .unwrap_or_else(|e| panic!("profiles/{name}.toml failed to load: {e}"));
            assert!(!p.notes.trim().is_empty(), "profiles/{name}.toml has no notes");
            assert_eq!(p.name, name, "profiles/{name}.toml declares a mismatched name");
        }
    }

    /// The whole point of the "current*" profiles is that loading one changes
    /// nothing. If this ever fails, either a config file was retuned without
    /// updating its profile or a `default_*()` moved — both worth knowing.
    #[test]
    fn current_profiles_are_no_ops_on_the_configs_they_mirror() {
        for (profile, config) in [
            ("current",               "config.toml"),
            ("current-no-imgnovelty", "config2.toml"),
            ("current-relaxed",       "config3.toml"),
        ] {
            let cfg_path = repo().join(config);
            let prof_path = repo().join(format!("profiles/{profile}.toml"));
            if !cfg_path.exists() || !prof_path.exists() { continue; }

            let before = Config::load(&cfg_path)
                .unwrap_or_else(|e| panic!("{config} failed to parse: {e}"));
            let mut after = before.clone();
            FitnessProfile::load(&prof_path).unwrap().apply(&mut after);

            for f in FIELDS {
                assert_eq!(
                    config_value(&after, f.key), config_value(&before, f.key),
                    "profiles/{profile}.toml changes {} on {config} \
                     ({} -> {}) — it is supposed to be a no-op",
                    f.key, config_value(&before, f.key), config_value(&after, f.key)
                );
            }
        }
    }

    #[test]
    fn load_falls_back_to_the_filename_for_an_unnamed_profile() {
        let dir = std::env::temp_dir().join(format!("nnf_prof_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("explore-wide.toml");
        std::fs::write(&path, "ood_weight = 0.5\n").unwrap();
        let p = FitnessProfile::load(&path).unwrap();
        assert_eq!(p.name, "explore-wide");
        assert_eq!(p.ood_weight, Some(0.5));
        std::fs::remove_dir_all(&dir).ok();
    }
}
