//! Applies Carl's own trained quaternion-fractal preference model
//! natively in Rust — no Python round-trip, no image render.
//!
//! `scripts/train_pref_quat.py` fits a linear Bradley-Terry model over
//! the ~30 `quat_*` metric fields (`Genome`, computed by
//! `quat_dag_fitness.rs`) from pairwise "⚖ Rate" comparisons made in the
//! browser, and (since the same session that added this module) writes
//! a plain-JSON twin of its `.npz` right alongside it. This exists so
//! `quat-dag-evolve` can select on that model instead of the generic,
//! not-fractal-tuned NIMA/TOPIQ/AP25 ensemble (`aesthetic::
//! AestheticScorer`) that used to be the only "aesthetic" signal
//! available — Carl's own words: "discontinue the aesthetic scorer
//! unless it is one I trained myself."
//!
//! Scoring is exactly `train_pref_quat.py --score-only`'s formula
//! (`clip((dot(features, w) - lo) / (hi - lo), 0, 1)`), reproduced here
//! so evolution sees the identical number a post-hoc gallery rescore
//! would compute for the same genome.

use std::path::Path;

pub struct QuatPrefModel {
    fields: Vec<String>,
    w: Vec<f32>,
    lo: f32,
    hi: f32,
}

impl QuatPrefModel {
    /// Loads the JSON model file (same shape `train_pref_quat.py` writes:
    /// `{"fields": [...], "w": [...], "lo": f64, "hi": f64}`). Returns
    /// `None` if it doesn't exist or fails to parse — callers treat both
    /// the same way: fall back to geometric-only fitness, never silently
    /// substitute a different (generic) scorer.
    pub fn load(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        let fields: Vec<String> = v.get("fields")?.as_array()?.iter().filter_map(|x| x.as_str().map(String::from)).collect();
        let w: Vec<f32> = v.get("w")?.as_array()?.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
        let lo = v.get("lo")?.as_f64()? as f32;
        let hi = v.get("hi")?.as_f64()? as f32;
        if fields.is_empty() || fields.len() != w.len() {
            return None;
        }
        Some(QuatPrefModel { fields, w, lo, hi })
    }

    /// Number of comparisons the trained weights don't record directly —
    /// just the feature count, for a quick startup sanity print.
    pub fn feature_count(&self) -> usize {
        self.fields.len()
    }

    /// Score in `[0,1]`; higher = more like what Carl rated as preferred.
    /// `feature(name)` maps a `quat_*` field name to its value — a
    /// closure rather than a fixed struct so this works against any
    /// source of those fields (currently `QuatFullMetrics`, see
    /// `explorer.rs`'s `quat_full_metrics_feature`).
    pub fn score(&self, feature: impl Fn(&str) -> f32) -> f32 {
        let raw: f32 = self.fields.iter().zip(&self.w).map(|(name, wi)| feature(name) * wi).sum();
        let rng = if self.hi > self.lo { self.hi - self.lo } else { 1.0 };
        ((raw - self.lo) / rng).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_model(dir: &Path, fields: &[&str], w: &[f32], lo: f32, hi: f32) -> std::path::PathBuf {
        let path = dir.join(format!("model_{}.json", rand_suffix()));
        let json = serde_json::json!({ "fields": fields, "w": w, "lo": lo, "hi": hi });
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
        path
    }

    fn rand_suffix() -> String {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
        format!("{}_{nanos}", std::process::id())
    }

    #[test]
    fn load_missing_file_returns_none() {
        assert!(QuatPrefModel::load(Path::new("/nonexistent/pref_model_quat.json")).is_none());
    }

    #[test]
    fn score_matches_hand_computed_dot_product() {
        let dir = std::env::temp_dir();
        let path = write_model(&dir, &["quat_coverage", "quat_convexity"], &[2.0, -1.0], 0.0, 10.0);
        let model = QuatPrefModel::load(&path).expect("model should load");
        let _ = std::fs::remove_file(&path);
        assert_eq!(model.feature_count(), 2);
        // raw = 2.0*3.0 + (-1.0)*4.0 = 2.0; (2.0 - 0.0) / (10.0 - 0.0) = 0.2
        let score = model.score(|name| match name {
            "quat_coverage" => 3.0,
            "quat_convexity" => 4.0,
            _ => panic!("unexpected field {name}"),
        });
        assert!((score - 0.2).abs() < 1e-5, "expected 0.2, got {score}");
    }

    #[test]
    fn score_clamps_to_unit_range() {
        let dir = std::env::temp_dir();
        let path = write_model(&dir, &["quat_coverage"], &[1.0], 0.0, 1.0);
        let model = QuatPrefModel::load(&path).expect("model should load");
        let _ = std::fs::remove_file(&path);
        assert_eq!(model.score(|_| 100.0), 1.0, "raw score far above hi must clamp to 1.0");
        assert_eq!(model.score(|_| -100.0), 0.0, "raw score far below lo must clamp to 0.0");
    }
}
