use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use crate::formula::{N_BASIS, basis_name};

/// Bounds on the number of active terms in a genome's formula.
pub const MIN_TERMS: usize = 2;
pub const MAX_TERMS: usize = 8;
/// Per-mutation structural probabilities.
const BASIS_SWAP_PROB: f32 = 0.18;
const TERM_ADD_PROB:   f32 = 0.20;
const TERM_DROP_PROB:  f32 = 0.15;

/// One term of the iterated map: coeff · φ_basis(z, c), coeff = re + i·im.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FormulaTerm {
    pub basis: u8,   // index into the 0..N_BASIS basis functions
    pub re: f32,
    pub im: f32,
}

impl FormulaTerm {
    fn random(rng: &mut impl Rng) -> Self {
        FormulaTerm {
            basis: rng.random_range(0..N_BASIS as u8),
            re: rng.random::<f32>() * 2.0 - 1.0,
            im: rng.random::<f32>() * 2.0 - 1.0,
        }
    }

    fn random_exotic(rng: &mut impl Rng) -> Self {
        // Bases proven in top-CLIP genomes from archive analysis:
        //   0=z²  18=sin  30=tan  46=|BS|(burning-ship)  51=z/(z²+1)  52=(z²-1)/(z²+1)  54=1/(z²+c)
        // The latest record (0.5294) used [cosh, 52, 46, 52, (z-c)²] — |BS| is load-bearing.
        const EXOTIC: &[u8] = &[0, 18, 30, 46, 51, 52, 54];
        FormulaTerm {
            basis: EXOTIC[rng.random_range(0..EXOTIC.len())],
            re: rng.random::<f32>() * 2.0 - 1.0,
            im: rng.random::<f32>() * 2.0 - 1.0,
        }
    }
}

/// A fractal genome: the formula is evolved DIRECTLY as a sparse set of weighted
/// basis terms (no transformer/latent indirection). z_new = Σ coeffᵢ · φ_basisᵢ(z, c).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub terms: Vec<FormulaTerm>,
    pub fitness: f32,
    /// Raw beauty score (no novelty inflation) at time of save; 0.0 if not yet saved.
    #[serde(default)]
    pub beauty: f32,
    #[serde(default)] pub beauty_boundary:  f32,
    #[serde(default)] pub beauty_edge:      f32,
    #[serde(default)] pub beauty_entropy:   f32,
    #[serde(default)] pub beauty_self_sim:  f32,
    #[serde(default)] pub beauty_cool_zone: f32,
    /// CLIP zero-shot aesthetic score [0,1] at save time; 0.0 if unavailable.
    #[serde(default)] pub clip_score:  f32,
    /// LAION MLP aesthetic score [0,10] at save time; 0.0 if unavailable.
    #[serde(default)] pub laion_score: f32,
    pub id: u64,
    #[serde(default)]
    pub view_cx: f32,
    #[serde(default)]
    pub view_cy: f32,
    #[serde(default = "default_view_zoom")]
    pub view_zoom: f32,
}

fn default_view_zoom() -> f32 { 1.0 }

impl Genome {
    pub fn view_bounds(&self) -> (f32, f32, f32, f32) {
        let half = 2.0 / self.view_zoom;
        (self.view_cx - half, self.view_cx + half, self.view_cy - half, self.view_cy + half)
    }

    /// Expand the sparse term set into the dense [N_BASIS] complex weight vector
    /// consumed by `apply_formula` and the GPU shader. Terms sharing a basis sum.
    pub fn formula_weights(&self) -> Vec<(f32, f32)> {
        let mut w = vec![(0.0f32, 0.0f32); N_BASIS];
        for t in &self.terms {
            let i = (t.basis as usize).min(N_BASIS - 1);
            w[i].0 += t.re;
            w[i].1 += t.im;
        }
        w
    }

    fn random_view(rng: &mut impl Rng) -> (f32, f32, f32) {
        if rng.random::<f32>() < 0.70 {
            (0.0, 0.0, 1.0)
        } else {
            let z = 0.8 + rng.random::<f32>() * 2.2;
            let pan = 0.8 / z;
            (
                (rng.random::<f32>() * 2.0 - 1.0) * pan,
                (rng.random::<f32>() * 2.0 - 1.0) * pan,
                z,
            )
        }
    }

    fn new(terms: Vec<FormulaTerm>, view: (f32, f32, f32), rng: &mut impl Rng) -> Self {
        Genome {
            terms,
            fitness: 0.0,
            beauty: 0.0,
            beauty_boundary: 0.0, beauty_edge: 0.0, beauty_entropy: 0.0,
            beauty_self_sim: 0.0, beauty_cool_zone: 0.0, clip_score: 0.0, laion_score: 0.0,
            id: rng.random(),
            view_cx: view.0,
            view_cy: view.1,
            view_zoom: view.2,
        }
    }

    pub fn random(_config: &Config, rng: &mut impl Rng) -> Self {
        let n = rng.random_range(MIN_TERMS..=MAX_TERMS);
        let terms = (0..n).map(|_| FormulaTerm::random(rng)).collect();
        let view = Self::random_view(rng);
        Self::new(terms, view, rng)
    }

    /// Like random(), but guarantees at least one term from the exotic basis range
    /// (burning-ship, essential singularity, rational, conjugate). Forces the GA to
    /// keep exploring visual regions it won't discover via uniform sampling.
    pub fn random_exotic(config: &Config, rng: &mut impl Rng) -> Self {
        let n = rng.random_range(MIN_TERMS..=MAX_TERMS);
        let mut terms: Vec<FormulaTerm> = (0..n).map(|_| FormulaTerm::random(rng)).collect();
        // Replace the first term with an exotic one.
        if let Some(t) = terms.first_mut() {
            *t = FormulaTerm::random_exotic(rng);
        }
        let view = Self::random_view(rng);
        Self::new(terms, view, rng)
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        // Union of parents' terms, each kept with prob 0.5, clamped to [MIN, MAX].
        let mut terms: Vec<FormulaTerm> = Vec::new();
        for t in a.terms.iter().chain(b.terms.iter()) {
            if rng.random_bool(0.5) { terms.push(*t); }
            if terms.len() >= MAX_TERMS { break; }
        }
        while terms.len() < MIN_TERMS {
            // Pull a guaranteed term from a parent (or random) to stay above the floor.
            let src = if rng.random_bool(0.5) { &a.terms } else { &b.terms };
            if let Some(t) = src.get(rng.random_range(0..src.len().max(1))) {
                terms.push(*t);
            } else {
                terms.push(FormulaTerm::random(rng));
            }
        }
        let view = (
            (a.view_cx + b.view_cx) * 0.5,
            (a.view_cy + b.view_cy) * 0.5,
            (a.view_zoom * b.view_zoom).sqrt(),
        );
        Self::new(terms, view, rng)
    }

    pub fn mutate(&self, config: &Config, rng: &mut impl Rng) -> Self {
        let mr = config.optimization.mutation_rate;
        let ms = config.optimization.mutation_scale;

        let mut child = self.clone();
        child.id      = rng.random();
        child.fitness = 0.0;

        // Per-term: perturb coefficient and occasionally swap which basis function it uses.
        for t in child.terms.iter_mut() {
            if rng.random::<f32>() < mr { t.re += rng.random::<f32>() * 2.0 * ms - ms; }
            if rng.random::<f32>() < mr { t.im += rng.random::<f32>() * 2.0 * ms - ms; }
            if rng.random::<f32>() < BASIS_SWAP_PROB {
                t.basis = rng.random_range(0..N_BASIS as u8);
            }
        }

        // Structural: grow / shrink the term set within bounds.
        if child.terms.len() < MAX_TERMS && rng.random::<f32>() < TERM_ADD_PROB {
            child.terms.push(FormulaTerm::random(rng));
        }
        if child.terms.len() > MIN_TERMS && rng.random::<f32>() < TERM_DROP_PROB {
            let idx = rng.random_range(0..child.terms.len());
            child.terms.remove(idx);
        }

        // View mutations (unchanged).
        if rng.random::<f32>() < 0.30 {
            let zoom_delta = if rng.random::<f32>() < 0.65 {
                1.0 + rng.random::<f32>() * 1.0
            } else {
                1.0 / (1.0 + rng.random::<f32>() * 0.3)
            };
            child.view_zoom = (child.view_zoom * zoom_delta).clamp(0.5, 25.0);
        }
        if rng.random::<f32>() < 0.30 {
            let pan = 0.5 / child.view_zoom;
            child.view_cx = (child.view_cx + (rng.random::<f32>() * 2.0 - 1.0) * pan).clamp(-2.5, 2.5);
            child.view_cy = (child.view_cy + (rng.random::<f32>() * 2.0 - 1.0) * pan).clamp(-2.5, 2.5);
        }

        child
    }

    /// Top-3 active basis functions by |coeff|, plus zoom.
    pub fn formula_label(&self) -> String {
        let top = self.top_basis(3);
        format!("{}  z={:.1}x", top.join("+"), self.view_zoom)
    }

    /// Short identifier (top-2 basis functions) for formula diversity tracking.
    pub fn formula_ops_label(&self) -> String {
        self.top_basis(2).join("+")
    }

    fn top_basis(&self, k: usize) -> Vec<String> {
        let mut indexed: Vec<(f32, u8)> = self.terms.iter()
            .map(|t| (t.re * t.re + t.im * t.im, t.basis))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k)
            .map(|(_, b)| basis_name(*b as usize).to_string())
            .collect()
    }
}
