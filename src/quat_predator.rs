//! Predator-prey coevolution for the quaternion GA's lineage-capture
//! problem — Carl's own idea, worked out in conversation after the
//! `silhouette_irregularity+shading_gradient+color_shading_corr+solidity+
//! coverage` combo run plateaued at best total=3.200 from generation 139
//! onward (126+ generations, zero improvement), with 62/200 (31%) of the
//! final population sharing that exact fitness value — one lineage had
//! captured the survivor pool and neither the subtree-crossover operator
//! nor the existing genotype-novelty bonus broke it loose.
//!
//! The mechanism: a second, much cheaper population of `Predator`s, each
//! just a direction (unit vector) in the SAME metric-space the active
//! `--fitness-metric` spec scores prey on — no genome, no DAG program, no
//! rendering. Each generation, predators are scored on how well their
//! direction predicts which prey are currently overrepresented
//! (`commonness`, a k-NN density measure over prey metric vectors), and
//! prey fitness gets a MULTIPLICATIVE discount proportional to how well
//! the best predator "catches" it. This is deliberately an arms race, not
//! a fixed penalty: as one lineage comes to dominate, predators evolve
//! toward whatever metric-space direction best explains that dominance,
//! pushing prey away from it — then predators have to re-specialize once
//! a different cluster rises. A fixed diversity bonus (already tried,
//! `DIVERSITY_BONUS_WEIGHT` in `explorer.rs`) can't adapt like that; it
//! rewards the same kind of difference every generation, which a
//! dominant lineage can eventually satisfy without actually diversifying.
//!
//! Deliberately NOT a full second GA over genome/DAG representations —
//! predators only need to be "good at explaining overrepresentation,"
//! which is pure vector math over metrics prey already compute. That
//! keeps this genuinely cheap (no extra renders, no extra aesthetic-
//! scorer round trips) and keeps the two populations decoupled from
//! `QuatIndividual`/`Genome` entirely, so this module is plain,
//! independently unit-testable numeric code — `explorer.rs`'s
//! `cmd_quat_dag_evolve` is the only caller and owns all the
//! genome-specific glue (extracting `metric_vec` from each individual,
//! writing the discounted total back).

use rand::Rng;

/// One predator: a direction in metric-space it's betting explains which
/// prey are currently overrepresented. `weights` is a unit vector,
/// term-for-term aligned with whatever `--fitness-metric` spec is active
/// (same order as each prey's `metric_vec`) — predator STRENGTH comes
/// from how well the direction explains overrepresentation, not from
/// arbitrarily large weight magnitudes, which is why `weights` is
/// re-normalized after every mutation.
#[derive(Clone, Debug)]
pub struct Predator {
    pub weights: Vec<f64>,
    /// Correlation with prey `commonness` this generation — set by
    /// `evaluate_predators`, read by `evolve_predators`'s selection.
    pub fitness: f64,
}

impl Predator {
    fn random(dims: usize, rng: &mut impl Rng) -> Self {
        let weights: Vec<f64> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        Predator { weights: normalize(&weights), fitness: 0.0 }
    }

    /// Dot product against a prey's metric vector.
    pub fn score(&self, metric_vec: &[f64]) -> f64 {
        self.weights.iter().zip(metric_vec).map(|(w, v)| w * v).sum()
    }
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-9 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// Fresh predator population of unit-vector random directions.
pub fn spawn_predators(count: usize, dims: usize, rng: &mut impl Rng) -> Vec<Predator> {
    (0..count).map(|_| Predator::random(dims, rng)).collect()
}

/// "Commonness" of each prey in the population — k-nearest-neighbor
/// density over metric vectors (mean distance to the `k` nearest OTHER
/// individuals, inverted so a genome with many near-identical neighbors —
/// the 62-clone situation this module exists to fix — reads as HIGH
/// commonness, not low). O(n^2) in population size, same cost class as
/// the existing `knn_novelty` genotype/phenotype novelty in `explorer.rs`
/// and fine at the population sizes (~100-300) this GA actually runs at.
pub fn commonness(metric_vecs: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = metric_vecs.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut dists: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| euclidean(&metric_vecs[i], &metric_vecs[j])).collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kk = k.min(dists.len()).max(1);
        let mean_knn = dists[..kk].iter().sum::<f64>() / kk as f64;
        out[i] = 1.0 / (1.0 + mean_knn);
    }
    out
}

/// Pearson correlation between two equal-length series.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a < 1e-12 || var_b < 1e-12 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Scores every predator against the current prey population: `.fitness`
/// becomes its correlation with `density` — how well THIS predator's
/// metric-space direction explains overrepresentation this generation.
pub fn evaluate_predators(predators: &mut [Predator], metric_vecs: &[Vec<f64>], density: &[f64]) {
    for p in predators.iter_mut() {
        let scores: Vec<f64> = metric_vecs.iter().map(|mv| p.score(mv)).collect();
        p.fitness = pearson(&scores, density);
    }
}

/// Elitist selection + Gaussian-mutation refill for the predator
/// population — same shape as the prey GA's breeding loop but far
/// cheaper (pure vector math, no rendering). A small immigrant share of
/// brand-new random directions keeps predators from collapsing onto one
/// exploit and missing a newly-dominant prey cluster once the current one
/// scatters.
pub fn evolve_predators(predators: &mut Vec<Predator>, rng: &mut impl Rng, dims: usize) {
    const SURVIVOR_FRAC: f64 = 0.4;
    const IMMIGRANT_FRAC: f64 = 0.15;
    const MUTATION_SIGMA: f64 = 0.25;
    predators.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    let n = predators.len();
    let survivor_count = (((n as f64) * SURVIVOR_FRAC).ceil() as usize).clamp(1, n);
    let survivors: Vec<Predator> = predators[..survivor_count].to_vec();
    let mut next: Vec<Predator> = survivors.clone();
    while next.len() < n {
        if rng.random::<f64>() < IMMIGRANT_FRAC {
            next.push(Predator::random(dims, rng));
        } else {
            let parent = &survivors[rng.random_range(0..survivors.len())];
            let mutated: Vec<f64> = parent.weights.iter().map(|w| w + rng.random_range(-1.0..1.0) * MUTATION_SIGMA).collect();
            next.push(Predator { weights: normalize(&mutated), fitness: 0.0 });
        }
    }
    *predators = next;
}

/// Default fraction of a caught prey's fitness that predators can shave
/// off. Deliberately MULTIPLICATIVE and capped well under 1.0: a
/// low-quality genome can never be pushed ABOVE a high-quality one just
/// because predators haven't learned to hunt it yet — this is the same
/// degenerate-shape trap this project hit once already with naive
/// edge/gradient metrics rewarding thin, boring "blade" shapes. At 0.4,
/// even a fully-caught genome keeps 60% of its earned fitness.
pub const DEFAULT_PENALTY_WEIGHT: f64 = 0.4;

/// Applies one generation's predator pressure to a scored prey
/// population's fitness totals IN PLACE, then evolves the predator
/// population for the next round. `metric_vecs` and `totals` must be the
/// same length and order (one entry per prey individual); `predators`'
/// dimensionality must match `metric_vecs[i].len()`. Returns the best
/// predator's fitness this generation (purely for logging — lets
/// `cmd_quat_dag_evolve` print how well predators are currently tracking
/// overrepresentation, the same way it already logs mean/best/worst prey
/// fitness).
pub fn apply_predator_pressure(metric_vecs: &[Vec<f64>], totals: &mut [f64], predators: &mut Vec<Predator>, penalty_weight: f64, rng: &mut impl Rng) -> f64 {
    if metric_vecs.is_empty() || metric_vecs[0].is_empty() {
        return 0.0;
    }
    let dims = metric_vecs[0].len();
    let density = commonness(metric_vecs, 5);
    evaluate_predators(predators, metric_vecs, &density);
    let best_predator_fitness = predators.iter().map(|p| p.fitness).fold(f64::NEG_INFINITY, f64::max).max(0.0);

    // Each prey's "caught" score = the BEST (max) score any single
    // predator gives it — min-max normalized across the population so
    // the discount always spans a comparable [0,1] range regardless of
    // the raw dot-product scale, which drifts as predator directions
    // change generation to generation.
    let raw: Vec<f64> = metric_vecs.iter().map(|mv| predators.iter().map(|p| p.score(mv)).fold(f64::NEG_INFINITY, f64::max)).collect();
    let lo = raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-9);
    for (total, r) in totals.iter_mut().zip(&raw) {
        let caught = ((r - lo) / span).clamp(0.0, 1.0);
        *total *= 1.0 - penalty_weight * caught;
    }

    evolve_predators(predators, rng, dims);
    best_predator_fitness
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn commonness_is_higher_for_a_dense_cluster_than_an_outlier() {
        // 8 near-identical points clustered near the origin, 1 outlier far
        // away — the cluster members should read as far more "common"
        // than the outlier.
        let mut metric_vecs: Vec<Vec<f64>> = (0..8).map(|i| vec![0.01 * i as f64, 0.01 * i as f64]).collect();
        metric_vecs.push(vec![10.0, 10.0]);
        let density = commonness(&metric_vecs, 3);
        let cluster_mean: f64 = density[..8].iter().sum::<f64>() / 8.0;
        let outlier = density[8];
        assert!(cluster_mean > outlier * 2.0, "cluster density {cluster_mean} should be well above outlier density {outlier}");
    }

    #[test]
    fn predators_learn_to_catch_a_dominant_cluster() {
        // 2 metric dims. A dense cluster around (0.9, 0.1) dominates the
        // population; a handful of scattered points elsewhere don't. After
        // enough generations of evolve_predators, the best predator
        // should score the dominant cluster noticeably higher than the
        // scattered points — i.e. predators actually learn to hunt
        // whatever's overrepresented, not just react to it once.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut metric_vecs: Vec<Vec<f64>> = (0..40).map(|i| vec![0.9 + 0.005 * (i % 5) as f64, 0.1 + 0.005 * (i % 5) as f64]).collect();
        for i in 0..10 {
            metric_vecs.push(vec![0.1 * i as f64, 0.9 - 0.05 * i as f64]);
        }
        let mut predators = spawn_predators(20, 2, &mut rng);
        let density = commonness(&metric_vecs, 5);
        for _ in 0..30 {
            evaluate_predators(&mut predators, &metric_vecs, &density);
            evolve_predators(&mut predators, &mut rng, 2);
        }
        evaluate_predators(&mut predators, &metric_vecs, &density);
        let best = predators.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
        let cluster_score: f64 = (0..40).map(|i| best.score(&metric_vecs[i])).sum::<f64>() / 40.0;
        let scatter_score: f64 = (40..50).map(|i| best.score(&metric_vecs[i])).sum::<f64>() / 10.0;
        assert!(cluster_score > scatter_score, "best predator (fitness={}) should score the dominant cluster ({cluster_score}) above the scattered points ({scatter_score})", best.fitness);
    }

    #[test]
    fn predator_pressure_never_flips_a_low_quality_prey_above_a_high_quality_one() {
        // The degenerate-shape trap this is guarding against: even a
        // fully-caught high-quality genome must never end up scored below
        // an uncaught low-quality one, since the penalty is multiplicative
        // and bounded well under 1.0.
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let metric_vecs = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![0.01, 0.01]];
        let mut totals = vec![3.0, 3.0, 0.1];
        let mut predators = spawn_predators(10, 2, &mut rng);
        // Bias predators hard toward catching the dominant (1.0,1.0)
        // cluster so this is a genuine worst-case, not a lucky draw.
        for p in predators.iter_mut() {
            p.weights = vec![1.0, 1.0];
        }
        apply_predator_pressure(&metric_vecs, &mut totals, &mut predators, DEFAULT_PENALTY_WEIGHT, &mut rng);
        assert!(totals[0] > totals[2], "caught high-quality prey ({}) must stay above uncaught low-quality prey ({})", totals[0], totals[2]);
        assert!(totals[0] >= 3.0 * (1.0 - DEFAULT_PENALTY_WEIGHT) - 1e-9, "penalty must never exceed DEFAULT_PENALTY_WEIGHT: got {}", totals[0]);
    }
}
