use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use crate::formula::{N_BASIS, basis_name, op, OpNode, N_SLOTS};

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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Genome {
    /// Human-readable rendering of the formula — a comment for people reading the
    /// .nn JSON (JSON has no real comments). Set at save time; ignored on load.
    #[serde(default)] pub formula_readable: String,
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
    /// Zoom self-replication score [0,1]: how much the fractal reproduces its
    /// whole-set structure under deep zoom (Mandelbrot-like). 0.0 if not measured.
    #[serde(default)] pub self_replication: f32,
    /// Fractal-recursion score [0,1]: how strongly a *complete miniature copy of
    /// the whole set* (a baby-Mandelbrot) reappears embedded inside it. 0.0 if not
    /// measured. Distinct from `self_replication` (boundary-detail persistence).
    #[serde(default)] pub fractal_recursion: f32,
    /// Formula-only predicted recursion [0,1] from `RecursionModel` at eval time;
    /// drives selection. Tracked so retraining can compare predicted vs measured.
    #[serde(default)] pub pred_recursion: f32,
    /// Formula-only predicted CLIP aesthetic score [0,1] from a linear model
    /// trained on the saved archive. Drives selection toward beautiful formula families.
    #[serde(default)] pub pred_clip: f32,
    /// Formula-space novelty [0,∞]: average L2 distance to k nearest archive
    /// genomes in normalised 58-dim basis-weight space. Higher = structurally
    /// distinct formula family. Drives selection when formula_diversity_weight > 0.
    #[serde(default)] pub formula_diversity: f32,
    /// Expression-DAG program (Phase-1 formula system). When non-empty, this
    /// replaces the flat `terms` basis-sum: z_{n+1} = eval_program(program,z,c).
    /// Empty ⇒ legacy genome that still evaluates via `terms`/`formula_weights`.
    #[serde(default)] pub program: Vec<crate::formula::OpNode>,
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

    /// Normalised 58-dim basis-weight vector for formula-diversity k-NN scoring.
    /// Each element is Σ|coeff| on that basis, then divided by the L2 norm so
    /// the metric is direction-sensitive (which bases dominate) not scale-sensitive.
    pub fn formula_basis_normalized(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; N_BASIS];
        for t in &self.terms {
            let b = (t.basis as usize).min(N_BASIS - 1);
            v[b] += (t.re * t.re + t.im * t.im).sqrt();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// Formula-only feature vector for the recursion predictor. MUST stay
    /// byte-for-byte aligned with `features()` in scripts/fit_recursion_model.py:
    ///   [0..N_BASIS)  Σ |coeff| of terms on each basis
    ///   [N_BASIS]     num_terms
    ///   [N_BASIS+1]   total |coeff|
    ///   [N_BASIS+2]   c · z-power interaction (basis7 · Σ basis0..3)
    pub fn recursion_features(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; N_BASIS + 3];
        for t in &self.terms {
            let b = (t.basis as usize).min(N_BASIS - 1);
            f[b] += (t.re * t.re + t.im * t.im).sqrt();
        }
        f[N_BASIS]     = self.terms.len() as f32;
        f[N_BASIS + 1] = f[..N_BASIS].iter().sum();
        f[N_BASIS + 2] = f[7] * (f[0] + f[1] + f[2] + f[3]);
        f
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

    /// True when this genome uses the expression-DAG formula system.
    pub fn uses_program(&self) -> bool { !self.program.is_empty() }

    /// Human-readable formula string. DAG genomes render as an infix expression
    /// from the root (z_{n+1} = …); legacy genomes as the weighted basis sum.
    pub fn formula_expr(&self) -> String {
        if self.uses_program() {
            if self.program.is_empty() { return "z".into(); }
            format!("z_next = {}", render_node(&self.program, self.program.len() - 1, 0))
        } else {
            let parts: Vec<String> = self.terms.iter()
                .map(|t| format!("{}·{}", fmt_c(t.re, t.im), basis_name(t.basis as usize)))
                .collect();
            let body = if parts.is_empty() { "0".into() } else { parts.join(" + ") };
            format!("z_next = {body}")
        }
    }

    /// Representation-aware formula descriptor for k-NN formula-diversity scoring:
    /// normalized opcode histogram (DAG) or normalized basis-weight vector (legacy).
    /// Two genomes of the same representation are directly comparable; transitional
    /// mixed populations compare on the shorter common prefix (acceptable).
    pub fn formula_descriptor(&self) -> Vec<f32> {
        if self.uses_program() {
            let mut v = vec![0.0f32; op::N_OPS];
            for n in &self.program {
                v[(n.op as usize).min(op::N_OPS - 1)] += 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        } else {
            self.formula_basis_normalized()
        }
    }

    /// Convert the legacy flat basis-sum into an equivalent expression-DAG
    /// program: z_new = Σ coeffᵢ·basisᵢ(z,c) → a DAG of the new primitives.
    /// Returns None if the result would exceed N_SLOTS or uses a basis the new
    /// op set can't represent exactly (then the genome stays on the legacy path).
    /// Used for archive migration and as a cross-check oracle against `apply_formula`.
    pub fn legacy_to_program(&self) -> Option<Vec<OpNode>> {
        let mut b = ProgramBuilder::new();
        let z = b.push(op::Z, 0, 0, 0.0, 0.0)?;
        let c = b.push(op::C, 0, 0, 0.0, 0.0)?;
        let mut term_roots: Vec<u8> = Vec::new();
        for t in &self.terms {
            let base = build_basis(&mut b, t.basis, z, c)?;
            let k = b.push(op::CONST, 0, 0, t.re, t.im)?;
            let scaled = b.push(op::MUL, k, base, 0.0, 0.0)?;
            term_roots.push(scaled);
        }
        if term_roots.is_empty() { return None; }
        let mut acc = term_roots[0];
        for &r in &term_roots[1..] {
            acc = b.push(op::ADD, acc, r, 0.0, 0.0)?;
        }
        let _ = acc;
        Some(b.into_nodes())
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
            formula_readable: String::new(),
            terms,
            fitness: 0.0,
            beauty: 0.0,
            beauty_boundary: 0.0, beauty_edge: 0.0, beauty_entropy: 0.0,
            beauty_self_sim: 0.0, beauty_cool_zone: 0.0, clip_score: 0.0, laion_score: 0.0,
            self_replication: 0.0,
            fractal_recursion: 0.0,
            pred_recursion: 0.0,
            pred_clip: 0.0,
            formula_diversity: 0.0,
            program: Vec::new(),
            id: rng.random(),
            view_cx: view.0,
            view_cy: view.1,
            view_zoom: view.2,
        }
    }

    /// True when the GA should produce expression-DAG genomes.
    fn dag_mode(config: &Config) -> bool {
        config.optimization.formula_system == "dag"
    }

    pub fn random(config: &Config, rng: &mut impl Rng) -> Self {
        if Self::dag_mode(config) {
            let prog = random_program(rng, config.optimization.max_nodes, config.optimization.max_depth, false);
            let view = Self::random_view(rng);
            let mut g = Self::new(Vec::new(), view, rng);
            g.program = prog;
            return g;
        }
        let n = rng.random_range(MIN_TERMS..=MAX_TERMS);
        let terms = (0..n).map(|_| FormulaTerm::random(rng)).collect();
        let view = Self::random_view(rng);
        Self::new(terms, view, rng)
    }

    /// Like random(), but biases in rare/exotic primitives (DAG: transcendental
    /// ops; legacy: exotic bases). Forces the GA to keep exploring visual regions
    /// it won't discover via uniform sampling.
    pub fn random_exotic(config: &Config, rng: &mut impl Rng) -> Self {
        if Self::dag_mode(config) {
            let prog = random_program(rng, config.optimization.max_nodes, config.optimization.max_depth, true);
            let view = Self::random_view(rng);
            let mut g = Self::new(Vec::new(), view, rng);
            g.program = prog;
            return g;
        }
        let n = rng.random_range(MIN_TERMS..=MAX_TERMS);
        let mut terms: Vec<FormulaTerm> = (0..n).map(|_| FormulaTerm::random(rng)).collect();
        // Replace the first term with an exotic one.
        if let Some(t) = terms.first_mut() {
            *t = FormulaTerm::random_exotic(rng);
        }
        let view = Self::random_view(rng);
        Self::new(terms, view, rng)
    }

    pub fn crossover(a: &Self, b: &Self, config: &Config, rng: &mut impl Rng) -> Self {
        // DAG crossover when both parents carry programs (avoids mixed-rep children).
        if a.uses_program() && b.uses_program() {
            let prog = crossover_program(&a.program, &b.program, rng, config.optimization.max_nodes);
            let view = (
                (a.view_cx + b.view_cx) * 0.5,
                (a.view_cy + b.view_cy) * 0.5,
                (a.view_zoom * b.view_zoom).sqrt(),
            );
            let mut g = Self::new(Vec::new(), view, rng);
            g.program = prog;
            return g;
        }
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

        // DAG genomes mutate their program; legacy genomes mutate their terms.
        if self.uses_program() {
            child.program = mutate_program(
                &self.program, rng,
                config.optimization.max_nodes, config.optimization.max_depth,
            );
            child.mutate_view(rng);
            return child;
        }

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

        child.mutate_view(rng);
        child
    }

    /// In-place stochastic zoom/pan mutation, shared by legacy and DAG paths.
    fn mutate_view(&mut self, rng: &mut impl Rng) {
        if rng.random::<f32>() < 0.30 {
            let zoom_delta = if rng.random::<f32>() < 0.65 {
                1.0 + rng.random::<f32>() * 1.0
            } else {
                1.0 / (1.0 + rng.random::<f32>() * 0.3)
            };
            self.view_zoom = (self.view_zoom * zoom_delta).clamp(0.5, 25.0);
        }
        if rng.random::<f32>() < 0.30 {
            let pan = 0.5 / self.view_zoom;
            self.view_cx = (self.view_cx + (rng.random::<f32>() * 2.0 - 1.0) * pan).clamp(-2.5, 2.5);
            self.view_cy = (self.view_cy + (rng.random::<f32>() * 2.0 - 1.0) * pan).clamp(-2.5, 2.5);
        }
    }

    /// Human-readable formula label (basis names or DAG ops), plus zoom.
    pub fn formula_label(&self) -> String {
        if self.uses_program() {
            return format!("{}  z={:.1}x", self.program_ops_label(), self.view_zoom);
        }
        let top = self.top_basis(3);
        format!("{}  z={:.1}x", top.join("+"), self.view_zoom)
    }

    /// Short structural identifier for formula-diversity tracking. For DAG genomes
    /// this is the sorted set of non-leaf ops; for legacy, the top-2 bases.
    pub fn formula_ops_label(&self) -> String {
        if self.uses_program() {
            return self.program_ops_label();
        }
        self.top_basis(2).join("+")
    }

    /// Sorted distinct non-leaf op names of a DAG program (its "shape").
    fn program_ops_label(&self) -> String {
        let mut ops: Vec<&str> = self.program.iter()
            .map(|n| op::name(n.op))
            .filter(|s| !matches!(*s, "z" | "c" | "k"))
            .collect();
        ops.sort_unstable();
        ops.dedup();
        ops.join("+")
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

// ════════════════════════════════════════════════════════════════════════════
// Expression-DAG construction helpers (legacy conversion + program building).
// ════════════════════════════════════════════════════════════════════════════

/// Accumulates OpNodes in topological order, returning the index of each pushed
/// node. Refuses to grow past N_SLOTS (returns None), so callers can bail out of
/// conversions that wouldn't fit the register file.
pub struct ProgramBuilder { nodes: Vec<OpNode> }

impl ProgramBuilder {
    pub fn new() -> Self { ProgramBuilder { nodes: Vec::new() } }

    /// Push a node; returns its index (u8) or None if the program is full.
    pub fn push(&mut self, op: u8, a: u8, b: u8, kre: f32, kim: f32) -> Option<u8> {
        if self.nodes.len() >= N_SLOTS { return None; }
        let idx = self.nodes.len() as u8;
        self.nodes.push(OpNode { op, a, b, kre, kim });
        Some(idx)
    }

    pub fn into_nodes(self) -> Vec<OpNode> { self.nodes }
    pub fn len(&self) -> usize { self.nodes.len() }
}

#[cfg(test)]
mod gp_op_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn is_valid(prog: &[OpNode]) -> bool {
        if prog.is_empty() || prog.len() > N_SLOTS { return false; }
        for (i, n) in prog.iter().enumerate() {
            if (n.op as usize) >= op::N_OPS { return false; }
            let ar = op::arity(n.op);
            if ar >= 1 && (n.a as usize) >= i { return false; } // input must precede
            if ar >= 2 && (n.b as usize) >= i { return false; }
        }
        true
    }

    // random/mutate/crossover always yield valid topological DAGs within caps,
    // and eval_program stays finite over a range of inputs.
    #[test]
    fn gp_operators_preserve_validity() {
        let mut rng = StdRng::seed_from_u64(42);
        let (max_nodes, max_depth) = (14usize, 5usize);
        for _ in 0..500 {
            let exotic = rng.random_bool(0.5);
            let a = random_program(&mut rng, max_nodes, max_depth, exotic);
            let b = random_program(&mut rng, max_nodes, max_depth, false);
            assert!(is_valid(&a), "random invalid: {a:?}");
            assert!(is_valid(&b));
            let m = mutate_program(&a, &mut rng, max_nodes, max_depth);
            assert!(is_valid(&m), "mutate invalid: {m:?}");
            let x = crossover_program(&a, &b, &mut rng, max_nodes);
            assert!(is_valid(&x), "crossover invalid: {x:?}");
            // eval stays finite (NaN/inf is allowed to be produced but must not panic;
            // here we just confirm it returns without UB and is usually finite).
            for &(zx, zy, cx, cy) in &[(0.3f32,0.4,-0.5,0.6),(1.2,-0.7,0.1,0.2)] {
                let _ = crate::formula::eval_program(&x, zx, zy, cx, cy);
            }
        }
    }
}

#[cfg(test)]
mod legacy_conv_tests {
    use super::*;
    use crate::formula::{eval_basis, eval_program};

    // Every basis that build_basis claims to cover must, once wrapped in a
    // program, evaluate identically to the trusted eval_basis across sample points.
    #[test]
    fn each_basis_subtree_matches_eval_basis() {
        let pts = [(0.3f32,0.4,-0.5,0.6),(1.1,-0.7,0.2,0.3),(-0.8,0.25,0.5,-0.4),(0.05,-0.9,-0.3,0.7)];
        let mut covered = 0;
        for i in 0..N_BASIS as u8 {
            let mut b = ProgramBuilder::new();
            let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
            let c = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
            let root = match build_basis(&mut b, i, z, c) {
                Some(r) => r,
                None => continue, // uncovered (31 sinh, 32 cosh, 49 z|z|) — fine
            };
            // build_basis returns the root index; ensure it's the last node so
            // eval_program returns it. If not, append an identity ADD(root,const0).
            let mut prog = b.into_nodes();
            if root as usize != prog.len() - 1 {
                // wrap: tie root to the tail via a no-op MUL by 1
                let one = prog.len() as u8;
                prog.push(OpNode { op: op::CONST, a: 0, b: 0, kre: 1.0, kim: 0.0 });
                prog.push(OpNode { op: op::MUL, a: root, b: one, kre: 0.0, kim: 0.0 });
            }
            for &(zx, zy, cx, cy) in &pts {
                let (px, py) = eval_program(&prog, zx, zy, cx, cy);
                let (ex, ey) = eval_basis(i as usize, zx, zy, cx, cy);
                assert!((px - ex).abs() < 1e-4 && (py - ey).abs() < 1e-4,
                    "basis {i} mismatch at ({zx},{zy},{cx},{cy}): dag=({px},{py}) eval_basis=({ex},{ey})");
            }
            covered += 1;
        }
        assert!(covered >= 50, "expected ≥50 bases covered, got {covered}");
    }

    // A full legacy genome converts to a program that renders identically.
    #[test]
    fn legacy_genome_to_program_matches() {
        let g = Genome {
            terms: vec![
                FormulaTerm { basis: 0,  re: 1.0,  im: 0.0 },  // z²
                FormulaTerm { basis: 7,  re: 1.0,  im: 0.0 },  // c
                FormulaTerm { basis: 52, re: 0.3, im: -0.2 },  // (z²−1)/(z²+1)
            ],
            ..Default::default()
        };
        let prog = g.legacy_to_program().expect("should fit");
        let fw = g.formula_weights();
        for &(zx, zy, cx, cy) in &[(0.4f32,0.3,-0.2,0.5),(0.9,-0.6,0.1,0.4)] {
            let (px, py) = eval_program(&prog, zx, zy, cx, cy);
            let (lx, ly) = crate::formula::apply_formula(&fw, zx, zy, cx, cy);
            assert!((px - lx).abs() < 1e-4 && (py - ly).abs() < 1e-4,
                "genome mismatch: dag=({px},{py}) legacy=({lx},{ly})");
        }
    }
}

/// Format a complex coefficient compactly for the readable formula string.
fn fmt_c(re: f32, im: f32) -> String {
    if im.abs() < 1e-4 { format!("{:.2}", re) }
    else if re.abs() < 1e-4 { format!("{:.2}i", im) }
    else if im < 0.0 { format!("({:.2}-{:.2}i)", re, -im) }
    else { format!("({:.2}+{:.2}i)", re, im) }
}

/// Render a DAG node as an infix/functional expression string. Shared nodes are
/// expanded (the program is a DAG; the readable form is a tree). Depth-guarded.
fn render_node(prog: &[OpNode], i: usize, depth: usize) -> String {
    if depth > 14 || i >= prog.len() { return "…".into(); }
    let n = prog[i];
    let a = (n.a as usize).min(i.saturating_sub(1));
    let b = (n.b as usize).min(i.saturating_sub(1));
    let ra = || render_node(prog, a, depth + 1);
    let rb = || render_node(prog, b, depth + 1);
    match n.op {
        op::Z       => "z".into(),
        op::C       => "c".into(),
        op::CONST   => fmt_c(n.kre, n.kim),
        op::SQR     => format!("({})²", ra()),
        op::CUBE    => format!("({})³", ra()),
        op::QUART   => format!("({})⁴", ra()),
        op::RECIP   => format!("1/({})", ra()),
        op::SIN     => format!("sin({})", ra()),
        op::COS     => format!("cos({})", ra()),
        op::EXP     => format!("exp({})", ra()),
        op::LOG     => format!("log({})", ra()),
        op::TANH    => format!("tanh({})", ra()),
        op::CONJ    => format!("conj({})", ra()),
        op::ABSFOLD => format!("|{}|ʙs", ra()),
        op::ABSRE   => format!("absRe({})", ra()),
        op::ABSIM   => format!("absIm({})", ra()),
        op::NORMZ   => format!("({})/|·|", ra()),
        op::ADD     => format!("({} + {})", ra(), rb()),
        op::SUB     => format!("({} - {})", ra(), rb()),
        op::MUL     => format!("{}·{}", ra(), rb()),
        op::DIV     => format!("({} / {})", ra(), rb()),
        _           => "?".into(),
    }
}

// ── Genetic-programming operators on flat topological DAG arrays ────────────────

const UNARY_OPS: [u8; 14] = [
    op::SQR, op::CUBE, op::QUART, op::RECIP, op::SIN, op::COS, op::EXP, op::LOG,
    op::TANH, op::CONJ, op::ABSFOLD, op::ABSRE, op::ABSIM, op::NORMZ,
];
const BINARY_OPS: [u8; 4] = [op::ADD, op::SUB, op::MUL, op::DIV];
// Rare/transcendental ops that produce unusual fractals — biased in via "exotic".
const EXOTIC_OPS: [u8; 6] = [op::SIN, op::EXP, op::RECIP, op::NORMZ, op::ABSFOLD, op::LOG];

fn rand_const(rng: &mut impl Rng) -> OpNode {
    OpNode { op: op::CONST, a: 0, b: 0, kre: re_k(rng), kim: im_k(rng) }
}
fn re_k(rng: &mut impl Rng) -> f32 { rng.random::<f32>() * 2.0 - 1.0 }
fn im_k(rng: &mut impl Rng) -> f32 { rng.random::<f32>() * 2.0 - 1.0 }

/// Grow a random valid topological DAG: leaves first, then unary/binary nodes
/// referencing earlier nodes, respecting depth/node caps. The root (last node)
/// is forced non-trivial. `exotic` biases in a rare transcendental op.
pub fn random_program(rng: &mut impl Rng, max_nodes: usize, max_depth: usize, exotic: bool) -> Vec<OpNode> {
    let cap = max_nodes.clamp(4, N_SLOTS);
    let mut nodes: Vec<OpNode> = Vec::new();
    let mut depth: Vec<u8> = Vec::new();
    let mut push = |nodes: &mut Vec<OpNode>, depth: &mut Vec<u8>, n: OpNode, d: u8| {
        nodes.push(n); depth.push(d);
    };
    push(&mut nodes, &mut depth, OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }, 0);
    push(&mut nodes, &mut depth, OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 }, 0);
    if rng.random_bool(0.5) {
        let k = rand_const(rng);
        push(&mut nodes, &mut depth, k, 0);
    }

    let target = rng.random_range(4..=cap);
    let mut tries = 0usize;
    while nodes.len() < target && tries < target * 6 {
        tries += 1;
        let r = rng.random::<f32>();
        if r < 0.60 {
            let a = rng.random_range(0..nodes.len());
            let b = rng.random_range(0..nodes.len());
            let d = 1 + depth[a].max(depth[b]);
            if d as usize > max_depth { continue; }
            let opc = BINARY_OPS[rng.random_range(0..BINARY_OPS.len())];
            push(&mut nodes, &mut depth, OpNode { op: opc, a: a as u8, b: b as u8, kre: 0.0, kim: 0.0 }, d);
        } else if r < 0.95 {
            let a = rng.random_range(0..nodes.len());
            let d = 1 + depth[a];
            if d as usize > max_depth { continue; }
            let opc = if exotic && rng.random_bool(0.5) {
                EXOTIC_OPS[rng.random_range(0..EXOTIC_OPS.len())]
            } else {
                UNARY_OPS[rng.random_range(0..UNARY_OPS.len())]
            };
            push(&mut nodes, &mut depth, OpNode { op: opc, a: a as u8, b: 0, kre: 0.0, kim: 0.0 }, d);
        } else if nodes.len() < cap {
            let k = rand_const(rng);
            push(&mut nodes, &mut depth, k, 0);
        }
    }

    // Force a non-leaf root that mixes two earlier nodes (so the map isn't trivial).
    if op::arity(nodes[nodes.len() - 1].op) == 0 && nodes.len() < N_SLOTS {
        let a = (nodes.len() - 1) as u8;
        let b = if nodes.len() >= 2 { (nodes.len() - 2) as u8 } else { 0 };
        let opc = BINARY_OPS[rng.random_range(0..BINARY_OPS.len())];
        nodes.push(OpNode { op: opc, a, b, kre: 0.0, kim: 0.0 });
    }
    nodes
}

/// Mutate a DAG program: perturb a constant, swap an op (arity-fixing inputs),
/// rewire an input, grow a node on top, or drop the root — all topology-safe.
pub fn mutate_program(prog: &[OpNode], rng: &mut impl Rng, max_nodes: usize, max_depth: usize) -> Vec<OpNode> {
    let mut p = prog.to_vec();
    if p.len() < 2 { return random_program(rng, max_nodes, max_depth, false); }
    let cap = max_nodes.clamp(4, N_SLOTS);

    // 1–2 edits per mutation.
    let edits = rng.random_range(1..=2);
    for _ in 0..edits {
        match rng.random_range(0..5) {
            0 => { // perturb a constant (or convert a leaf to const)
                let i = rng.random_range(0..p.len());
                if p[i].op == op::CONST {
                    p[i].kre += re_k(rng) * 0.3;
                    p[i].kim += im_k(rng) * 0.3;
                } else if op::arity(p[i].op) == 0 {
                    p[i] = rand_const(rng);
                }
            }
            1 => { // swap op, fixing inputs to satisfy new arity
                let i = rng.random_range(1..p.len());
                let new_op = if rng.random_bool(0.5) {
                    UNARY_OPS[rng.random_range(0..UNARY_OPS.len())]
                } else {
                    BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
                };
                let ar = op::arity(new_op);
                p[i].op = new_op;
                if ar >= 1 { p[i].a = rng.random_range(0..i) as u8; }
                if ar >= 2 { p[i].b = rng.random_range(0..i) as u8; }
            }
            2 => { // rewire an input to another earlier node
                let i = rng.random_range(1..p.len());
                if op::arity(p[i].op) >= 1 { p[i].a = rng.random_range(0..i) as u8; }
                if op::arity(p[i].op) >= 2 { p[i].b = rng.random_range(0..i) as u8; }
            }
            3 => { // grow: new root combining old root with a random earlier node
                if p.len() < cap {
                    let root = (p.len() - 1) as u8;
                    let other = rng.random_range(0..p.len()) as u8;
                    let opc = if rng.random_bool(0.5) {
                        BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
                    } else { op::ADD };
                    p.push(OpNode { op: opc, a: root, b: other, kre: 0.0, kim: 0.0 });
                }
            }
            _ => { // prune the root (shrink) if it leaves a usable program
                if p.len() > 3 { p.pop(); }
            }
        }
    }
    let _ = max_depth;
    p
}

/// Crossover two DAG programs: child = combine(rootA, rootB) by concatenating
/// B's nodes after A's (index-remapped) and adding a binary combiner as the new
/// root. Topology-safe. Falls back to mutating the smaller parent if over cap.
pub fn crossover_program(a: &[OpNode], b: &[OpNode], rng: &mut impl Rng, max_nodes: usize) -> Vec<OpNode> {
    let cap = max_nodes.clamp(4, N_SLOTS);
    if a.is_empty() { return b.to_vec(); }
    if b.is_empty() { return a.to_vec(); }
    if a.len() + b.len() + 1 > cap {
        let (small, _) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        return mutate_program(small, rng, max_nodes, 5);
    }
    let mut child = a.to_vec();
    let off = child.len() as u8;
    let root_a = (child.len() - 1) as u8;
    for n in b {
        let mut m = *n;
        if op::arity(m.op) >= 1 { m.a = m.a.saturating_add(off); }
        if op::arity(m.op) >= 2 { m.b = m.b.saturating_add(off); }
        child.push(m);
    }
    let root_b = (child.len() - 1) as u8;
    let opc = BINARY_OPS[rng.random_range(0..BINARY_OPS.len())];
    child.push(OpNode { op: opc, a: root_a, b: root_b, kre: 0.0, kim: 0.0 });
    child
}

/// Build a subtree computing legacy basis `i` over leaf nodes `z` and `c`,
/// returning the root index. None if basis `i` isn't exactly representable in
/// the new op set or the program overflows. Mirrors `eval_basis` in formula.rs.
fn build_basis(b: &mut ProgramBuilder, i: u8, z: u8, c: u8) -> Option<u8> {
    use op::*;
    const PI: f32 = std::f32::consts::PI;
    let one = |b: &mut ProgramBuilder| b.push(CONST, 0, 0, 1.0, 0.0);
    let negz = |b: &mut ProgramBuilder, z: u8| -> Option<u8> {
        let m = b.push(CONST, 0, 0, -1.0, 0.0)?;
        b.push(MUL, m, z, 0.0, 0.0)
    };
    match i {
        0  => b.push(SQR, z, 0, 0.0, 0.0),                                  // z²
        1  => b.push(CUBE, z, 0, 0.0, 0.0),                                 // z³
        2  => b.push(QUART, z, 0, 0.0, 0.0),                                // z⁴
        3  => { let q = b.push(QUART, z, 0, 0.0, 0.0)?; b.push(MUL, q, z, 0.0, 0.0) }, // z⁵
        4  => Some(z),                                                       // z
        5  => b.push(RECIP, z, 0, 0.0, 0.0),                               // 1/z
        6  => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; b.push(RECIP, z2, 0, 0.0, 0.0) }, // 1/z²
        7  => Some(c),                                                       // c
        8  => b.push(SQR, c, 0, 0.0, 0.0),                                  // c²
        9  => b.push(CUBE, c, 0, 0.0, 0.0),                                 // c³
        10 => b.push(MUL, z, c, 0.0, 0.0),                                  // zc
        11 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; b.push(MUL, z2, c, 0.0, 0.0) }, // z²c
        12 => { let c2 = b.push(SQR, c, 0, 0.0, 0.0)?; b.push(MUL, z, c2, 0.0, 0.0) }, // zc²
        13 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let c2 = b.push(SQR, c, 0, 0.0, 0.0)?; b.push(MUL, z2, c2, 0.0, 0.0) }, // z²c²
        14 => b.push(DIV, c, z, 0.0, 0.0),                                  // c/z
        15 => { let s = b.push(ADD, z, c, 0.0, 0.0)?; b.push(SQR, s, 0, 0.0, 0.0) }, // (z+c)²
        16 => { let s = b.push(SUB, z, c, 0.0, 0.0)?; b.push(SQR, s, 0, 0.0, 0.0) }, // (z−c)²
        17 => { let s = b.push(MUL, z, c, 0.0, 0.0)?; b.push(SQR, s, 0, 0.0, 0.0) }, // (zc)²
        18 => b.push(SIN, z, 0, 0.0, 0.0),                                  // sin(z)
        19 => b.push(COS, z, 0, 0.0, 0.0),                                  // cos(z)
        20 => { let p = b.push(CONST, 0, 0, PI, 0.0)?; let pz = b.push(MUL, p, z, 0.0, 0.0)?; b.push(SIN, pz, 0, 0.0, 0.0) }, // sin(πz)
        21 => { let p = b.push(CONST, 0, 0, PI, 0.0)?; let pz = b.push(MUL, p, z, 0.0, 0.0)?; b.push(COS, pz, 0, 0.0, 0.0) }, // cos(πz)
        22 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; b.push(SIN, z2, 0, 0.0, 0.0) }, // sin(z²)
        23 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; b.push(COS, z2, 0, 0.0, 0.0) }, // cos(z²)
        24 => { let s = b.push(ADD, z, c, 0.0, 0.0)?; b.push(SIN, s, 0, 0.0, 0.0) }, // sin(z+c)
        25 => { let s = b.push(ADD, z, c, 0.0, 0.0)?; b.push(COS, s, 0, 0.0, 0.0) }, // cos(z+c)
        26 => { let s = b.push(MUL, z, c, 0.0, 0.0)?; b.push(SIN, s, 0, 0.0, 0.0) }, // sin(zc)
        27 => { let s = b.push(MUL, z, c, 0.0, 0.0)?; b.push(COS, s, 0, 0.0, 0.0) }, // cos(zc)
        28 => { let s = b.push(SIN, z, 0, 0.0, 0.0)?; b.push(MUL, z, s, 0.0, 0.0) }, // z·sin(z)
        29 => { let s = b.push(COS, z, 0, 0.0, 0.0)?; b.push(MUL, z, s, 0.0, 0.0) }, // z·cos(z)
        30 => { let s = b.push(SIN, z, 0, 0.0, 0.0)?; let cs = b.push(COS, z, 0, 0.0, 0.0)?; b.push(DIV, s, cs, 0.0, 0.0) }, // tan(z)
        33 => b.push(TANH, z, 0, 0.0, 0.0),                                 // tanh(z)
        34 => b.push(EXP, z, 0, 0.0, 0.0),                                  // exp(z)
        35 => { let nz = negz(b, z)?; b.push(EXP, nz, 0, 0.0, 0.0) },        // exp(−z)
        36 => { let s = b.push(MUL, z, c, 0.0, 0.0)?; b.push(EXP, s, 0, 0.0, 0.0) }, // exp(zc)
        37 => { let e = b.push(EXP, z, 0, 0.0, 0.0)?; b.push(MUL, z, e, 0.0, 0.0) }, // z·exp(z)
        38 => { let e = b.push(EXP, z, 0, 0.0, 0.0)?; b.push(MUL, e, c, 0.0, 0.0) }, // exp(z)·c
        39 => { let o = one(b)?; let s = b.push(ADD, z, o, 0.0, 0.0)?; b.push(LOG, s, 0, 0.0, 0.0) }, // log(z+1)
        40 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let o = one(b)?; let s = b.push(ADD, z2, o, 0.0, 0.0)?; b.push(LOG, s, 0, 0.0, 0.0) }, // log(z²+1)
        41 => { let o = one(b)?; let s = b.push(ADD, z, o, 0.0, 0.0)?; let l = b.push(LOG, s, 0, 0.0, 0.0)?; b.push(MUL, z, l, 0.0, 0.0) }, // z·log(z+1)
        42 => { let r = b.push(RECIP, z, 0, 0.0, 0.0)?; b.push(SIN, r, 0, 0.0, 0.0) }, // sin(1/z)
        43 => { let r = b.push(RECIP, z, 0, 0.0, 0.0)?; b.push(EXP, r, 0, 0.0, 0.0) }, // exp(1/z)
        44 => b.push(ABSRE, z, 0, 0.0, 0.0),                               // |Re|+iIm
        45 => b.push(ABSIM, z, 0, 0.0, 0.0),                               // Re+i|Im|
        46 => b.push(ABSFOLD, z, 0, 0.0, 0.0),                             // |BS|
        47 => b.push(CONJ, z, 0, 0.0, 0.0),                               // conj
        48 => { let cj = b.push(CONJ, z, 0, 0.0, 0.0)?; b.push(SQR, cj, 0, 0.0, 0.0) }, // conj(z)²
        50 => b.push(NORMZ, z, 0, 0.0, 0.0),                              // z/|z|
        51 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let o = one(b)?; let d = b.push(ADD, z2, o, 0.0, 0.0)?; b.push(DIV, z, d, 0.0, 0.0) }, // z/(z²+1)
        52 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let o1 = one(b)?; let nr = b.push(SUB, z2, o1, 0.0, 0.0)?; let o2 = one(b)?; let dr = b.push(ADD, z2, o2, 0.0, 0.0)?; b.push(DIV, nr, dr, 0.0, 0.0) }, // (z²−1)/(z²+1)
        53 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let o = one(b)?; let d = b.push(SUB, z, o, 0.0, 0.0)?; b.push(DIV, z2, d, 0.0, 0.0) }, // z²/(z−1)
        54 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let d = b.push(ADD, z2, c, 0.0, 0.0)?; b.push(RECIP, d, 0, 0.0, 0.0) }, // 1/(z²+c)
        55 => { let z2 = b.push(SQR, z, 0, 0.0, 0.0)?; let z2c = b.push(MUL, z2, c, 0.0, 0.0)?; let d = b.push(ADD, z, c, 0.0, 0.0)?; b.push(DIV, z2c, d, 0.0, 0.0) }, // z²c/(z+c)
        56 => b.push(CONST, 0, 0, 1.0, 0.0),                              // 1
        57 => b.push(CONST, 0, 0, 0.0, 1.0),                              // i
        // 31 sinh, 32 cosh, 49 z·|z| — no exact single-op equivalent → bail.
        _  => None,
    }
}
