//! Formula-only recursion predictor: a cheap linear model that estimates a
//! genome's `fractal_recursion` from its formula features alone (no rendering),
//! so evolution can favour self-replicating formula families every generation.
//!
//! Trained offline by `scripts/fit_recursion_model.py` on the archive of measured
//! genomes and written to `recursion_model.json`. The feature vector here MUST
//! match `Genome::recursion_features()` byte-for-byte with the Python builder.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Clone, Debug)]
pub struct RecursionModel {
    pub mean:   Vec<f32>,
    pub std:    Vec<f32>,
    pub weight: Vec<f32>,
    pub bias:   f32,
    #[serde(default)] pub cv_pearson: f32,
    #[serde(default)] pub n_samples:  usize,
}

impl RecursionModel {
    /// Load from JSON; returns None if missing or structurally invalid (the GA
    /// then falls back to predicted recursion 0, i.e. the criterion is inert).
    pub fn load(path: &Path) -> Option<Self> {
        let s = std::fs::read_to_string(path).ok()?;
        let m: RecursionModel = serde_json::from_str(&s).ok()?;
        let d = m.weight.len();
        if d > 0 && m.mean.len() == d && m.std.len() == d { Some(m) } else { None }
    }

    /// Predicted recursion in [0,1] from a standardized linear model.
    pub fn predict(&self, feats: &[f32]) -> f32 {
        if feats.len() != self.weight.len() { return 0.0; }
        let mut acc = self.bias;
        for i in 0..feats.len() {
            let s = if self.std[i].abs() < 1e-9 { 1.0 } else { self.std[i] };
            acc += self.weight[i] * (feats[i] - self.mean[i]) / s;
        }
        acc.clamp(0.0, 1.0)
    }
}
