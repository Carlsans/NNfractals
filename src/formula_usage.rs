//! Tracks how many saved genomes already use each exact formula, so the GA can
//! charge a quadratic fitness penalty for piling more individuals onto an
//! already-crowded formula — a persistent, archive-wide diversity pressure
//! distinct from the per-generation k-NN `formula_diversity` novelty bonus
//! (which only sees the last `archive_size` generations).
//!
//! Key = `Genome::formula_expr()` (the full readable expression — coefficients,
//! wiring, warp/julia/bailout included), NOT the coarser `formula_ops_label()`
//! structural "shape". A full-expression key only penalizes true near-exact
//! duplicates; genomes that share an op-set family but differ in coefficients
//! or wiring are free to coexist and explore that family.

use std::collections::HashMap;
use std::path::Path;

use crate::genome::Genome;

pub struct FormulaUsageTracker {
    counts: HashMap<String, u32>,
    total: u32,
    gens_since_rescan: u32,
}

impl FormulaUsageTracker {
    /// Build the initial counts from whatever is already on disk (empty for a
    /// fresh save_dir).
    pub fn new(save_dir: &Path) -> Self {
        let mut t = Self { counts: HashMap::new(), total: 0, gens_since_rescan: 0 };
        t.rescan(save_dir);
        t
    }

    /// Incremental update for one newly-saved individual.
    pub fn record(&mut self, genome: &Genome) {
        *self.counts.entry(genome.formula_expr()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Full rebuild from the on-disk archive. Corrects drift the incremental
    /// `record()` path can't see: dedup.py deletions, browser deletions/moves,
    /// or genomes saved by another process instance.
    pub fn rescan(&mut self, save_dir: &Path) {
        let mut counts: HashMap<String, u32> = HashMap::new();
        let mut total = 0u32;
        if let Ok(entries) = std::fs::read_dir(save_dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("nn") { continue; }
                if let Ok(g) = crate::io::load_genome(&path) {
                    *counts.entry(g.formula_expr()).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
        self.counts = counts;
        self.total = total;
        self.gens_since_rescan = 0;
    }

    /// Call once per generation; triggers a full `rescan` every `every_gens`
    /// generations so the incrementally-updated counts don't drift forever.
    pub fn maybe_periodic_rescan(&mut self, save_dir: &Path, every_gens: u32) {
        self.gens_since_rescan += 1;
        if self.gens_since_rescan >= every_gens {
            self.rescan(save_dir);
        }
    }

    /// Quadratic penalty for a candidate genome's formula family, normalized by
    /// archive size so it stays bounded in [0, weight] regardless of how large
    /// the archive grows: weight · (uses_in_archive / archive_total)².
    pub fn penalty(&self, genome: &Genome, weight: f32) -> f32 {
        if weight == 0.0 || self.total == 0 { return 0.0; }
        let count = self.counts.get(&genome.formula_expr()).copied().unwrap_or(0) as f32;
        let frac = count / self.total as f32;
        weight * frac * frac
    }
}
