use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use crate::cppn::ActivationType;
use crate::formula::{ComplexOp, template_formulas, mutate_formula, crossover_formulas};

/// Single weight matrix stored in row-major order.
/// weights[o * in_size + i] = weight from input i to output o.
/// Complex NN: weights_im holds the imaginary part; empty ⟹ zeros (old-file compat).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerData {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub in_size: usize,
    pub out_size: usize,
    #[serde(default)]
    pub weights_im: Vec<f32>,
    #[serde(default)]
    pub biases_im: Vec<f32>,
}

impl LayerData {
    pub fn random(in_size: usize, out_size: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0_f32 / (in_size + out_size) as f32).sqrt() * std::f32::consts::SQRT_2;
        let n = in_size * out_size;
        let mut rand_vec = |n: usize| -> Vec<f32> {
            (0..n).map(|_| rng.random::<f32>() * 2.0 * scale - scale).collect()
        };
        LayerData {
            weights:    rand_vec(n),
            weights_im: rand_vec(n),
            biases:     vec![0.0f32; out_size],
            biases_im:  vec![0.0f32; out_size],
            in_size,
            out_size,
        }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        let mut c = a.clone();
        for (cv, &bv) in c.weights.iter_mut().zip(b.weights.iter()) {
            if rng.random_bool(0.5) { *cv = bv; }
        }
        for (cv, &bv) in c.weights_im.iter_mut().zip(b.weights_im.iter()) {
            if rng.random_bool(0.5) { *cv = bv; }
        }
        for (cv, &bv) in c.biases.iter_mut().zip(b.biases.iter()) {
            if rng.random_bool(0.5) { *cv = bv; }
        }
        for (cv, &bv) in c.biases_im.iter_mut().zip(b.biases_im.iter()) {
            if rng.random_bool(0.5) { *cv = bv; }
        }
        c
    }

    pub fn mutate(&mut self, rate: f32, scale: f32, rng: &mut impl Rng) {
        for w in self.weights.iter_mut() {
            if rng.random::<f32>() < rate { *w += rng.random::<f32>() * 2.0 * scale - scale; }
        }
        for w in self.weights_im.iter_mut() {
            if rng.random::<f32>() < rate { *w += rng.random::<f32>() * 2.0 * scale - scale; }
        }
        for b in self.biases.iter_mut() {
            if rng.random::<f32>() < rate { *b += rng.random::<f32>() * 2.0 * scale - scale; }
        }
        for b in self.biases_im.iter_mut() {
            if rng.random::<f32>() < rate { *b += rng.random::<f32>() * 2.0 * scale - scale; }
        }
    }
}

/// All weights for one transformer block:
///   2 tokens [z, c] → self-attention → CPPN feed-forward → 1 complex output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerWeights {
    /// Input embedding: complex scalar → d_model complex vector (in=1, out=d_model)
    pub embed_z: LayerData,
    pub embed_c: LayerData,
    /// Shared attention projections (in=d_model, out=d_model)
    pub w_q: LayerData,
    pub w_k: LayerData,
    pub w_v: LayerData,
    /// Attention output projection (in=d_model, out=d_model)
    pub w_o: LayerData,
    /// CPPN feed-forward layer 1 (in=d_model, out=d_ff)
    pub ff1: LayerData,
    /// Feed-forward layer 2 (in=d_ff, out=d_model)
    pub ff2: LayerData,
    /// Output projection: d_model → 1 complex scalar
    pub output: LayerData,
    /// Per-neuron evolved activation for ff1 (length = d_ff)
    pub ff_acts: Vec<ActivationType>,
}

impl TransformerWeights {
    pub fn random(d_model: usize, d_ff: usize, pool: &[ActivationType], rng: &mut impl Rng) -> Self {
        TransformerWeights {
            embed_z: LayerData::random(1,       d_model, rng),
            embed_c: LayerData::random(1,       d_model, rng),
            w_q:     LayerData::random(d_model, d_model, rng),
            w_k:     LayerData::random(d_model, d_model, rng),
            w_v:     LayerData::random(d_model, d_model, rng),
            w_o:     LayerData::random(d_model, d_model, rng),
            ff1:     LayerData::random(d_model, d_ff,    rng),
            ff2:     LayerData::random(d_ff,    d_model, rng),
            output:  LayerData::random(d_model, 1,       rng),
            ff_acts: (0..d_ff).map(|_| {
                if pool.is_empty() { ActivationType::Tanh }
                else { pool[rng.random_range(0..pool.len())].clone() }
            }).collect(),
        }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        let mut c = a.clone();
        c.embed_z = LayerData::crossover(&a.embed_z, &b.embed_z, rng);
        c.embed_c = LayerData::crossover(&a.embed_c, &b.embed_c, rng);
        c.w_q     = LayerData::crossover(&a.w_q,     &b.w_q,     rng);
        c.w_k     = LayerData::crossover(&a.w_k,     &b.w_k,     rng);
        c.w_v     = LayerData::crossover(&a.w_v,     &b.w_v,     rng);
        c.w_o     = LayerData::crossover(&a.w_o,     &b.w_o,     rng);
        c.ff1     = LayerData::crossover(&a.ff1,     &b.ff1,     rng);
        c.ff2     = LayerData::crossover(&a.ff2,     &b.ff2,     rng);
        c.output  = LayerData::crossover(&a.output,  &b.output,  rng);
        for (ca, ba) in c.ff_acts.iter_mut().zip(b.ff_acts.iter()) {
            if rng.random_bool(0.5) { *ca = ba.clone(); }
        }
        c
    }

    pub fn mutate(&mut self, rate: f32, scale: f32, act_prob: f32, pool: &[ActivationType], rng: &mut impl Rng) {
        self.embed_z.mutate(rate, scale, rng);
        self.embed_c.mutate(rate, scale, rng);
        self.w_q.mutate(rate, scale, rng);
        self.w_k.mutate(rate, scale, rng);
        self.w_v.mutate(rate, scale, rng);
        self.w_o.mutate(rate, scale, rng);
        self.ff1.mutate(rate, scale, rng);
        self.ff2.mutate(rate, scale, rng);
        self.output.mutate(rate, scale, rng);
        if !pool.is_empty() {
            for act in self.ff_acts.iter_mut() {
                if rng.random::<f32>() < act_prob {
                    *act = pool[rng.random_range(0..pool.len())].clone();
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub transformer: TransformerWeights,
    pub fitness: f32,
    pub id: u64,
    #[serde(default = "default_formula")]
    pub formula: Vec<ComplexOp>,
    #[serde(default = "default_nn_blend")]
    pub nn_blend: f32,
    /// View center: evolves to discover interesting zoom regions.
    #[serde(default)]
    pub view_cx: f32,
    #[serde(default)]
    pub view_cy: f32,
    /// Zoom level: 1.0 = full [-2,2] view, 4.0 = 4× zoom, etc.
    #[serde(default = "default_view_zoom")]
    pub view_zoom: f32,
}

fn default_formula()   -> Vec<ComplexOp> { vec![ComplexOp::Square, ComplexOp::AddC] }
fn default_nn_blend()  -> f32 { 1.0 }
fn default_view_zoom() -> f32 { 1.0 }

impl Genome {
    /// Compute the (xmin, xmax, ymin, ymax) view for this genome.
    /// Base half-range is 2.0 (covers [-2,2] at zoom=1).
    pub fn view_bounds(&self) -> (f32, f32, f32, f32) {
        let half = 2.0 / self.view_zoom;
        (
            self.view_cx - half,
            self.view_cx + half,
            self.view_cy - half,
            self.view_cy + half,
        )
    }

    pub fn random(config: &Config, rng: &mut impl Rng) -> Self {
        let d_model = config.network.d_model;
        let d_ff    = config.d_ff();
        let pool    = ActivationType::all_from_config(&config.network.activation_pool);

        let templates = template_formulas();
        let formula = if rng.random::<f32>() < 0.6 {
            templates[rng.random_range(0..templates.len())].clone()
        } else {
            let len = rng.random_range(2usize..=5);
            (0..len).map(|_| ComplexOp::random(rng)).collect()
        };

        let log_min = 0.05_f32.ln();
        let log_max = 3.0_f32.ln();
        let nn_blend = (log_min + rng.random::<f32>() * (log_max - log_min)).exp();

        // 70% start centered, 30% with a slight random pan/zoom
        let (view_cx, view_cy, view_zoom) = if rng.random::<f32>() < 0.70 {
            (0.0, 0.0, 1.0)
        } else {
            let z = 0.8 + rng.random::<f32>() * 2.2;
            let pan = 0.8 / z;
            (
                (rng.random::<f32>() * 2.0 - 1.0) * pan,
                (rng.random::<f32>() * 2.0 - 1.0) * pan,
                z,
            )
        };

        Genome {
            transformer: TransformerWeights::random(d_model, d_ff, &pool, rng),
            fitness: 0.0,
            id: rng.random(),
            formula,
            nn_blend,
            view_cx,
            view_cy,
            view_zoom,
        }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        Genome {
            transformer: TransformerWeights::crossover(&a.transformer, &b.transformer, rng),
            fitness:     0.0,
            id:          rng.random(),
            formula:     crossover_formulas(&a.formula, &b.formula, rng),
            nn_blend:    if rng.random_bool(0.5) { a.nn_blend } else { b.nn_blend },
            // Interpolate view: arithmetic mean for position, geometric mean for zoom
            view_cx:   (a.view_cx + b.view_cx) * 0.5,
            view_cy:   (a.view_cy + b.view_cy) * 0.5,
            view_zoom: (a.view_zoom * b.view_zoom).sqrt(),
        }
    }

    pub fn mutate(&self, config: &Config, rng: &mut impl Rng) -> Self {
        let pool = ActivationType::all_from_config(&config.network.activation_pool);
        let mr   = config.optimization.mutation_rate;
        let ms   = config.optimization.mutation_scale;
        let ap   = config.optimization.activation_mutation_prob;

        let mut child = self.clone();
        child.id      = rng.random();
        child.fitness = 0.0;
        child.transformer.mutate(mr, ms, ap, &pool, rng);

        // 5% chance: completely replace formula — escapes local optima in formula space
        if rng.random::<f32>() < 0.05 {
            let len = rng.random_range(2usize..=6);
            child.formula = (0..len).map(|_| ComplexOp::random(rng)).collect();
        } else if rng.random::<f32>() < 0.45 {
            mutate_formula(&mut child.formula, rng);
        }

        if rng.random::<f32>() < 0.12 {
            child.nn_blend *= 1.0 + rng.random::<f32>() * 0.6 - 0.3;
            child.nn_blend = child.nn_blend.clamp(0.02, 5.0);
        }

        // View mutations: biased toward zooming IN (deeper = richer detail).
        if rng.random::<f32>() < 0.30 {
            let zoom_delta = if rng.random::<f32>() < 0.65 {
                1.0 + rng.random::<f32>() * 1.0   // zoom in: ×1.0 → ×2.0
            } else {
                1.0 / (1.0 + rng.random::<f32>() * 0.3)  // zoom out: ÷1.0 → ÷1.3
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

    pub fn formula_label(&self) -> String {
        let ops: Vec<&str> = self.formula.iter().map(|op| op.label()).collect();
        format!("{}  b={:.2}  z={:.1}x", ops.join("→"), self.nn_blend, self.view_zoom)
    }

    pub fn formula_ops_label(&self) -> String {
        self.formula.iter().map(|op| op.label()).collect::<Vec<_>>().join("→")
    }
}
