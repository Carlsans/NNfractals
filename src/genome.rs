use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use crate::cppn::ActivationType;
use crate::formula::{N_BASIS, basis_name};
use crate::transformer::transformer_forward_latent;

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

/// All weights for one transformer block.
/// embed_z and embed_c now map d_model → d_model (project full latent → two tokens).
/// output maps d_model → N_BASIS (formula weights).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerWeights {
    /// Latent projection: d_model → d_model (was 1 → d_model)
    pub embed_z: LayerData,
    pub embed_c: LayerData,
    /// Attention projections: d_model → d_model
    pub w_q: LayerData,
    pub w_k: LayerData,
    pub w_v: LayerData,
    pub w_o: LayerData,
    /// Feed-forward: d_model → d_ff → d_model
    pub ff1: LayerData,
    pub ff2: LayerData,
    /// Formula weight output: d_model → N_BASIS
    pub output: LayerData,
    pub ff_acts: Vec<ActivationType>,
}

impl TransformerWeights {
    pub fn random(d_model: usize, d_ff: usize, pool: &[ActivationType], rng: &mut impl Rng) -> Self {
        TransformerWeights {
            embed_z: LayerData::random(d_model, d_model, rng),
            embed_c: LayerData::random(d_model, d_model, rng),
            w_q:     LayerData::random(d_model, d_model, rng),
            w_k:     LayerData::random(d_model, d_model, rng),
            w_v:     LayerData::random(d_model, d_model, rng),
            w_o:     LayerData::random(d_model, d_model, rng),
            ff1:     LayerData::random(d_model, d_ff,    rng),
            ff2:     LayerData::random(d_ff,    d_model, rng),
            output:  LayerData::random(d_model, N_BASIS, rng),
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
    /// Latent code: d_model complex values evolved by GA.
    /// Passed to the transformer to synthesize N_BASIS formula weights.
    pub latent: Vec<(f32, f32)>,
    pub fitness: f32,
    /// Raw beauty score (no novelty inflation) at time of save; 0.0 if not yet saved.
    #[serde(default)]
    pub beauty: f32,
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

    pub fn random(config: &Config, rng: &mut impl Rng) -> Self {
        let d_model = config.network.d_model;
        let d_ff    = config.d_ff();
        let pool    = ActivationType::all_from_config(&config.network.activation_pool);

        // Latent: d_model complex values, small Gaussian
        let latent: Vec<(f32, f32)> = (0..d_model)
            .map(|_| {
                let r = rng.random::<f32>() * 2.0 - 1.0;
                let i = rng.random::<f32>() * 2.0 - 1.0;
                (r * 0.5, i * 0.5)
            })
            .collect();

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
            latent,
            fitness: 0.0,
            beauty: 0.0,
            id: rng.random(),
            view_cx,
            view_cy,
            view_zoom,
        }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        let latent = a.latent.iter().zip(b.latent.iter())
            .map(|(&(ar, ai), &(br, bi))| {
                let r = if rng.random_bool(0.5) { ar } else { br };
                let i = if rng.random_bool(0.5) { ai } else { bi };
                (r, i)
            })
            .collect();

        Genome {
            transformer: TransformerWeights::crossover(&a.transformer, &b.transformer, rng),
            latent,
            fitness: 0.0,
            beauty: 0.0,
            id: rng.random(),
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

        // Gaussian mutation on latent (same rate/scale as transformer)
        for (r, i) in child.latent.iter_mut() {
            if rng.random::<f32>() < mr { *r += rng.random::<f32>() * 2.0 * ms - ms; }
            if rng.random::<f32>() < mr { *i += rng.random::<f32>() * 2.0 * ms - ms; }
        }

        // View mutations
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

    /// Label showing top-3 active basis functions by |weight|.
    pub fn formula_label(&self) -> String {
        let fw = transformer_forward_latent(&self.transformer, &self.latent);
        let mut indexed: Vec<(f32, usize)> = fw.iter()
            .enumerate()
            .map(|(i, &(r, im))| (r*r + im*im, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top3: Vec<String> = indexed.iter().take(3)
            .map(|(_, i)| basis_name(*i).to_string())
            .collect();
        format!("{}  z={:.1}x", top3.join("+"), self.view_zoom)
    }

    /// Short identifier for formula diversity tracking.
    pub fn formula_ops_label(&self) -> String {
        let fw = transformer_forward_latent(&self.transformer, &self.latent);
        let mut indexed: Vec<(f32, usize)> = fw.iter()
            .enumerate()
            .map(|(i, &(r, im))| (r*r + im*im, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(2)
            .map(|(_, i)| basis_name(*i).to_string())
            .collect::<Vec<_>>()
            .join("+")
    }
}
