//! MAP-Elites archive for `quat-dag-evolve --map-elites`.
//!
//! Every prior scalar-fitness run of the quaternion GA (6 geometric
//! metrics, 23 extended, 5 organization metrics, compression_capped,
//! sphericity) converged onto one lineage — Carl's own read: "the top
//! metrics always get chaotic... the middle individuals tend to be
//! better." That is the classic symptom of optimizing a single scalar: it
//! always has SOME exploit, and elitist selection chases it to the
//! exclusion of everything else. MAP-Elites sidesteps this structurally
//! rather than by tuning yet another penalty term: individuals are binned
//! into a grid of "behaviour descriptor" niches (axes that are NOT
//! optimized, just measured), and each niche independently keeps only its
//! own best-quality occupant. A collapsed lineage can dominate at most one
//! niche.
//!
//! Three axes (`descriptor_cell`'s argument order): sphericity,
//! hit_rate/solidity (both already bounded to `[0,1]`, so evenly-spaced
//! bins), and `organization_ordinal` (unbounded in practice, so
//! quantile-calibrated bins — see `ORDINAL_QUANTILE_EDGES`). 5 bins each
//! = 125 cells. See the approved plan
//! (`~/.claude/plans/majestic-sauteeing-nova.md`, Phase 2a) and memory
//! `project-taste-driven-quat-ga` for the full design rationale. The
//! optional 4th (role-model-distance) axis and `--me-axes`/`--me-bins`
//! configurability are deliberately NOT implemented yet (Phase 2c, "after
//! 2b works") — this module is intentionally the fixed-3-axis MVP.

use std::collections::{HashMap, VecDeque};

pub const N_BINS: usize = 5;

/// Quantile edges (20/40/60/80th percentile) for `organization_ordinal`'s
/// 5 bins, calibrated 2026-09-17 from 297 rescored genomes across
/// `fractals_dag_quat/` and `fractals_dag_quat_night_subtree/` (the two
/// archives the approved plan names as the calibration source). Unlike
/// sphericity/hit_rate, `organization_ordinal` has no natural `[0,1]`
/// range, so even bins would leave most of the range empty — these edges
/// keep the 5 bins roughly equally populated for real evolved genomes.
const ORDINAL_QUANTILE_EDGES: [f64; N_BINS - 1] = [0.075024046, 0.08777134, 0.15998082, 0.2537924];

/// Even-spaced edges for axes already bounded to `[0,1]` (sphericity,
/// hit_rate).
const EVEN_UNIT_EDGES: [f64; N_BINS - 1] = [0.2, 0.4, 0.6, 0.8];

fn bin_index(value: f64, edges: &[f64; N_BINS - 1]) -> u8 {
    edges.iter().filter(|&&e| value >= e).count() as u8
}

/// Maps raw (sphericity, hit_rate, organization_ordinal) values to a
/// 3-axis cell index — each component in `0..N_BINS`.
pub fn descriptor_cell(sphericity: f64, hit_rate: f64, ordinal: f64) -> [u8; 3] {
    [
        bin_index(sphericity.clamp(0.0, 1.0), &EVEN_UNIT_EDGES),
        bin_index(hit_rate.clamp(0.0, 1.0), &EVEN_UNIT_EDGES),
        bin_index(ordinal.max(0.0), &ORDINAL_QUANTILE_EDGES),
    ]
}

/// Human-readable cell name for filenames/logs, e.g. `"sph2_sol4_ord1"`.
pub fn cell_name(cell: [u8; 3]) -> String {
    format!("sph{}_sol{}_ord{}", cell[0], cell[1], cell[2])
}

/// One archive cell's occupant — the highest-quality individual seen for
/// this niche so far.
#[derive(Clone)]
pub struct Elite<T> {
    pub individual: T,
    pub quality: f64,
    pub gen_added: usize,
}

pub enum InsertOutcome {
    /// This cell had no occupant before.
    NewCell,
    /// This cell had a worse occupant, now replaced.
    Improved,
    /// This cell's existing occupant already scores at least as well.
    Rejected,
}

/// How many recent (cell, gen) improvements `sample_parent`'s
/// recency-bias draw considers — a bounded ring buffer, not the whole
/// history, so "recent" stays meaningful across a long run.
const RECENT_IMPROVEMENTS_CAP: usize = 500;

pub struct Archive<T> {
    cells: HashMap<[u8; 3], Elite<T>>,
    recent_improvements: VecDeque<([u8; 3], usize)>,
}

impl<T: Clone> Default for Archive<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Clone> Archive<T> {
    pub fn new() -> Self {
        Archive { cells: HashMap::new(), recent_improvements: VecDeque::new() }
    }

    /// Inserts `individual` at `cell` if it beats (or fills) that cell's
    /// current occupant. Ties favor the incumbent (strictly `>` to
    /// replace) so a re-evaluation of the same elite at unchanged quality
    /// doesn't spuriously reset `gen_added` / spam `recent_improvements`.
    pub fn insert(&mut self, cell: [u8; 3], individual: T, quality: f64, generation: usize) -> InsertOutcome {
        let outcome = match self.cells.get(&cell) {
            Some(existing) if quality <= existing.quality => return InsertOutcome::Rejected,
            Some(_) => InsertOutcome::Improved,
            None => InsertOutcome::NewCell,
        };
        self.cells.insert(cell, Elite { individual, quality, gen_added: generation });
        self.recent_improvements.push_back((cell, generation));
        while self.recent_improvements.len() > RECENT_IMPROVEMENTS_CAP {
            self.recent_improvements.pop_front();
        }
        outcome
    }

    pub fn coverage(&self) -> f64 {
        self.cells.len() as f64 / (N_BINS * N_BINS * N_BINS) as f64
    }

    pub fn mean_quality(&self) -> f64 {
        if self.cells.is_empty() { return 0.0; }
        self.cells.values().map(|e| e.quality).sum::<f64>() / self.cells.len() as f64
    }

    pub fn max_quality(&self) -> f64 {
        self.cells.values().map(|e| e.quality).fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn len(&self) -> usize { self.cells.len() }
    pub fn is_empty(&self) -> bool { self.cells.is_empty() }

    /// Uniform-over-occupied-cells parent sampling, with a 20% chance of
    /// instead drawing from a cell improved within the last
    /// `recent_window` generations (falls back to the uniform draw if no
    /// such improvement exists yet, e.g. very early in a run) — a mild
    /// bias toward niches that are actively getting better, without ever
    /// letting one lineage dominate parent selection the way elitist
    /// truncation does in the scalar path.
    pub fn sample_parent(&self, rng: &mut impl rand::Rng, current_gen: usize, recent_window: usize) -> Option<&T> {
        if self.cells.is_empty() { return None; }
        if rng.random_bool(0.2) {
            let recent: Vec<&[u8; 3]> = self.recent_improvements.iter()
                .filter(|(_, g)| current_gen.saturating_sub(*g) <= recent_window)
                .map(|(c, _)| c)
                .collect();
            if !recent.is_empty() {
                let cell = recent[rng.random_range(0..recent.len())];
                if let Some(e) = self.cells.get(cell) {
                    return Some(&e.individual);
                }
            }
        }
        let idx = rng.random_range(0..self.cells.len());
        self.cells.values().nth(idx).map(|e| &e.individual)
    }

    pub fn elites(&self) -> impl Iterator<Item = (&[u8; 3], &Elite<T>)> {
        self.cells.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_index_covers_all_5_bins_across_the_unit_range() {
        assert_eq!(bin_index(0.0, &EVEN_UNIT_EDGES), 0);
        assert_eq!(bin_index(0.19, &EVEN_UNIT_EDGES), 0);
        assert_eq!(bin_index(0.2, &EVEN_UNIT_EDGES), 1);
        assert_eq!(bin_index(0.5, &EVEN_UNIT_EDGES), 2);
        assert_eq!(bin_index(0.81, &EVEN_UNIT_EDGES), 4);
        assert_eq!(bin_index(1.0, &EVEN_UNIT_EDGES), 4);
    }

    #[test]
    fn descriptor_cell_clamps_out_of_range_inputs() {
        // Negative/over-1 sphericity or hit_rate (shouldn't happen, but a
        // clamp is cheap insurance) still lands in a valid bin, not a
        // panic or an out-of-range index.
        let cell = descriptor_cell(-5.0, 5.0, 0.0);
        assert_eq!(cell[0], 0);
        assert_eq!(cell[1], 4);
    }

    #[test]
    fn cell_name_round_trips_distinct_cells_to_distinct_names() {
        let a = cell_name([0, 1, 2]);
        let b = cell_name([0, 1, 3]);
        assert_ne!(a, b);
        assert_eq!(cell_name([2, 4, 0]), "sph2_sol4_ord0");
    }

    #[test]
    fn insert_fills_empty_cell_as_new() {
        let mut archive: Archive<u32> = Archive::new();
        let outcome = archive.insert([0, 0, 0], 42, 0.5, 0);
        assert!(matches!(outcome, InsertOutcome::NewCell));
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn insert_only_replaces_a_strictly_better_occupant() {
        let mut archive: Archive<u32> = Archive::new();
        archive.insert([0, 0, 0], 1, 0.5, 0);
        // Worse quality: rejected, occupant unchanged.
        let outcome = archive.insert([0, 0, 0], 2, 0.3, 1);
        assert!(matches!(outcome, InsertOutcome::Rejected));
        assert_eq!(archive.elites().next().unwrap().1.individual, 1);
        // Equal quality: also rejected (ties favor the incumbent).
        let outcome = archive.insert([0, 0, 0], 3, 0.5, 2);
        assert!(matches!(outcome, InsertOutcome::Rejected));
        // Strictly better: improved.
        let outcome = archive.insert([0, 0, 0], 4, 0.9, 3);
        assert!(matches!(outcome, InsertOutcome::Improved));
        assert_eq!(archive.elites().next().unwrap().1.individual, 4);
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn coverage_and_mean_quality_track_inserted_cells() {
        let mut archive: Archive<u32> = Archive::new();
        assert_eq!(archive.coverage(), 0.0);
        assert_eq!(archive.mean_quality(), 0.0);
        archive.insert([0, 0, 0], 1, 0.4, 0);
        archive.insert([1, 0, 0], 2, 0.8, 0);
        assert_eq!(archive.len(), 2);
        assert!((archive.coverage() - 2.0 / 125.0).abs() < 1e-9);
        assert!((archive.mean_quality() - 0.6).abs() < 1e-9);
        assert!((archive.max_quality() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn sample_parent_returns_none_when_empty_and_some_when_occupied() {
        use rand::SeedableRng;
        let empty: Archive<u32> = Archive::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert!(empty.sample_parent(&mut rng, 0, 5).is_none());

        let mut archive: Archive<u32> = Archive::new();
        archive.insert([0, 0, 0], 7, 0.5, 0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        assert_eq!(archive.sample_parent(&mut rng, 0, 5), Some(&7));
    }
}
