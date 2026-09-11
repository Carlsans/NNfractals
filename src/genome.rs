use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::config::Config;
use crate::formula::{N_BASIS, basis_name, op, OpNode, N_SLOTS, TimeMod, ModTarget, mod_offset};

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
    /// NIMA (AVA) aesthetic score [~1,10]; discriminates fractals far better than
    /// LAION/CLIP. 0.0 if unavailable.
    #[serde(default)] pub nima: f32,
    /// TOPIQ-IAA aesthetic score [~1,10]; high spread on fractals. 0.0 if unavailable.
    #[serde(default)] pub topiq_iaa: f32,
    /// Aesthetic Predictor v2.5 (SigLIP) score [~1,10]. 0.0 if unavailable.
    #[serde(default)] pub ap25_score: f32,
    /// MUSIQ technical-quality score [0,100] (sharpness/structure). 0.0 if unavailable.
    #[serde(default)] pub musiq: f32,
    /// Ensemble aesthetic = mean(nima, topiq_iaa, ap25_score) at save time. This is
    /// the fractal-tuned "beauty by human standard" signal the GA now selects on.
    #[serde(default)] pub aesthetic_ensemble: f32,
    /// Human-preference score [0,1] from the trained taste model (pref_model.npz,
    /// SigLIP + Bradley-Terry). 0.0 if no model / not scored. Blended into selection
    /// via optimization.pref_weight.
    #[serde(default)] pub pref_score: f32,
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
    /// Structural richness [0,1] of the bailout exit-angle field (DAG genomes
    /// only — see fitness::angle_structure_score). 0.0 if not measured
    /// (angle_structure_weight = 0). Drives selection when > 0.
    #[serde(default)] pub angle_structure: f32,
    /// Closest named reference formula (Mandelbrot, Tricorn, Burning Ship, …)
    /// to this genome's base program, by behavioral correlation — see
    /// fractal::known_formula_match. Empty = no match above threshold.
    /// DISCOVERY/CURIOSITY ONLY: never read by fitness, selection, or seeding.
    #[serde(default)] pub known_formula_match: String,
    /// Correlation score backing known_formula_match; 0.0 when empty.
    #[serde(default)] pub known_formula_score: f32,
    /// Image-space novelty [0,∞]: average L2 distance to k nearest archive
    /// genomes' learned visual embeddings (frozen DINOv2 + small VICReg-trained
    /// head, trained on this pool's own images — see scripts/train_novelty.py).
    /// Higher = visually/structurally distinct from the rest of the pool.
    /// 0.0 if not yet scored. Computed live at save time by novelty_scorer.py
    /// (src/novelty.rs) and blended into saved fitness + archive-seed ranking
    /// via optimization.img_novelty_weight (0 = inert); also backfillable in
    /// bulk via scripts/train_novelty.py --score-only.
    #[serde(default)] pub novelty_score: f32,
    /// Numeric k-means cluster id over the same learned embedding space
    /// (scripts/train_novelty.py --score-only). -1 if not yet clustered. No
    /// semantic label yet — inspect clusters via the browser's thumbnails.
    #[serde(default)] pub novelty_cluster: i32,
    /// Wormhole match quality [0,1]: how strongly a smaller embedded copy
    /// of this genome's own (saved) view recurs somewhere inside it — see
    /// fractal::wormhole_search. 0.0 if not measured or no confident match
    /// found. DISCOVERY/NAVIGATION ONLY: never read by fitness, selection,
    /// or seeding — distinct from `fractal_recursion`, which is a presence/
    /// absence signal used in seed ranking (and, unlike this field, was
    /// found to silently always score 0.0 for DAG genomes).
    #[serde(default)] pub wormhole_score: f32,
    /// Offset (fractal-plane units, relative to view_cx/view_cy) of the
    /// best wormhole match. Meaningless when wormhole_score == 0.0.
    #[serde(default)] pub wormhole_dx: f32,
    #[serde(default)] pub wormhole_dy: f32,
    /// Zoom of the matched window (same convention as view_zoom: half_y =
    /// 2.0/wormhole_zoom). Meaningless when wormhole_score == 0.0.
    #[serde(default)] pub wormhole_zoom: f32,
    /// Expression-DAG program (Phase-1 formula system). When non-empty, this
    /// replaces the flat `terms` basis-sum: z_{n+1} = eval_program(program,z,c).
    /// Empty ⇒ legacy genome that still evaluates via `terms`/`formula_weights`.
    #[serde(default)] pub program: Vec<crate::formula::OpNode>,
    // ── Phase-3 iteration dynamics (DAG genomes only) ───────────────────────────
    /// Julia mode: the pixel coordinate becomes the initial iterate z₀ and `c`
    /// is held at the constant (julia_cre, julia_cim) — yields Julia-set families
    /// instead of Mandelbrot-style parameter-plane fractals.
    #[serde(default)] pub julia_mode: bool,
    #[serde(default)] pub julia_cre: f32,
    #[serde(default)] pub julia_cim: f32,
    /// Phoenix memory: z_{n+1} = program(z,c) + p·z_{n-1}. p=(0,0) disables it.
    #[serde(default)] pub phoenix_re: f32,
    #[serde(default)] pub phoenix_im: f32,
    /// Per-genome escape radius (evolved). Affects boundary texture/detail.
    #[serde(default = "default_bailout_radius")] pub bailout_radius: f32,
    // ── Phase-4 coordinate warp (DAG genomes only) ──────────────────────────────
    /// Optional warp program applied once to the pixel coordinate before iterating
    /// (c ← warp(pixel)). Empty = identity. Bends the plane (spirals/folds).
    #[serde(default)] pub warp: Vec<crate::formula::OpNode>,
    /// Optional time modulation — the third dimension. Empty means a static
    /// fractal, which is every `.nn` file written before this existed, so old
    /// archives load and render bit-identically. Never consumed by the renderer
    /// directly: `at_time` bakes it into a per-frame `Genome` clone instead, so
    /// the CPU/f64/DD kernels and the WGSL shader are all untouched.
    #[serde(default)] pub time_mod: Vec<TimeMod>,
    /// Evolved time formulas — the same expression-DAG system as `program`,
    /// read as a function of `t` instead of of the iterate (see
    /// `crate::time_program`). Applied by `at_time` exactly like `time_mod`,
    /// and additive with it: a genome may carry both.
    ///
    /// Kept as its own list rather than folded into `TimeMod` because the two
    /// are different objects — five scalars against a graph — and because
    /// `TimeMod` is `Copy`, which a `Vec<OpNode>` would take away from every
    /// site that holds one.
    #[serde(default)] pub time_prog: Vec<crate::time_program::TimeProgram>,
    pub id: u64,
    #[serde(default)]
    pub view_cx: f32,
    #[serde(default)]
    pub view_cy: f32,
    #[serde(default = "default_view_zoom")]
    pub view_zoom: f32,
}

fn default_view_zoom() -> f32 { 1.0 }
fn default_bailout_radius() -> f32 { 4.0 }

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
            let mut s = format!("z_next = {}", render_node(&self.program, self.program.len() - 1, 0));
            if self.phoenix_re != 0.0 || self.phoenix_im != 0.0 {
                s.push_str(&format!(" + {}·z_prev", fmt_c(self.phoenix_re, self.phoenix_im)));
            }
            if !self.warp.is_empty() {
                s.push_str(&format!("   [warp: c={}]", render_node(&self.warp, self.warp.len() - 1, 0)));
            }
            if self.julia_mode {
                s.push_str(&format!("   [julia c={}, z0=pixel]", fmt_c(self.julia_cre, self.julia_cim)));
            }
            s.push_str(&format!("   [bailout r={:.1}]", self.bailout_radius));
            for m in &self.time_mod {
                s.push_str(&format!(
                    "   [anim: {} {} amp={:.3} freq={:.2}]",
                    m.target.label(), m.shape.label(), m.amp, m.freq
                ));
            }
            for tp in &self.time_prog {
                s.push_str(&format!(
                    "   [tprog: {} amp={:.3} freq={:.2} {}]",
                    tp.target.label(), tp.amp, tp.freq, tp.expr()
                ));
            }
            s
        } else {
            let parts: Vec<String> = self.terms.iter()
                .map(|t| format!("{}·{}", fmt_c(t.re, t.im), basis_name(t.basis as usize)))
                .collect();
            let body = if parts.is_empty() { "0".into() } else { parts.join(" + ") };
            format!("z_next = {body}")
        }
    }

    /// A clone of this genome with every [`TimeMod`] evaluated at `t ∈ [0,1)`.
    ///
    /// This is the entire rendering change for the time axis. Because every
    /// modulated scalar is a plain genome field that the renderer re-reads on
    /// each call — and `render_gpu::dag_item` repacks all of them into the
    /// per-genome upload block on every dispatch — the returned `Genome` renders
    /// correctly through the f32 GPU path, the f64 CPU path and the DD path with
    /// no change to any of them, and none to `fractal.wgsl`.
    ///
    /// Modulations are OFFSETS from the genome's own values, so `amp = 0` is
    /// always exactly the original fractal. Two modulations aimed at the same
    /// scalar sum, which is the useful behaviour (a slow drift plus a fast
    /// wobble). An empty `time_mod` returns a plain clone, so a static genome is
    /// bit-identical to not calling this at all.
    pub fn at_time(&self, t: f32) -> Genome {
        self.at_time_in_view(t, f32::INFINITY)
    }

    /// [`Genome::at_time`], with every offset capped at
    /// [`crate::time_program::MAX_OFFSET_FRACTION`] of the rendered view's
    /// half-extent.
    ///
    /// This is what the exporters call, and it is what keeps a zoom-plus-time
    /// video watchable all the way down. A modulation sized to morph the whole
    /// set at the establishing shot is, at zoom 1e9, a jump of tens of thousands
    /// of screen-widths — the same offset, an entirely different picture. Past
    /// `time_program::animatable_zoom_limit()` no representable offset is small
    /// enough to be an animation at all, so the cap drives the modulation below
    /// the f32 resolution of the scalar it drives and the formula simply holds
    /// still while the camera keeps going. The alternative is a second half that
    /// flashes.
    ///
    /// `half_extent = INFINITY` disables the cap, which is what bare `at_time`
    /// uses: a caller with no view has no frame to be measured against.
    pub fn at_time_in_view(&self, t: f32, half_extent: f32) -> Genome {
        if self.time_mod.is_empty() && self.time_prog.is_empty() {
            return self.clone();
        }
        let cap = if half_extent.is_finite() {
            crate::time_program::MAX_OFFSET_FRACTION * half_extent.abs()
        } else {
            f32::INFINITY
        };
        let mut g = self.clone();
        for m in &self.time_mod {
            let (dre, dim) = mod_offset(m, t);
            let (dre, dim) = crate::time_program::clamp_offset(dre, dim, cap);
            apply_offset(&mut g, m.target, dre, dim);
        }
        for tp in &self.time_prog {
            let (dre, dim) = tp.eval(t);
            let (dre, dim) = crate::time_program::clamp_offset(dre, dim, cap);
            apply_offset(&mut g, tp.target, dre, dim);
        }
        g
    }

    /// Whether anything about this genome's formula changes over `t`.
    pub fn animates(&self) -> bool {
        !self.time_mod.is_empty() || !self.time_prog.is_empty()
    }

    /// Why `self` and `other` cannot be blended, or `Ok(())` if they can.
    ///
    /// Two things genuinely do not interpolate. `julia_mode` is a discrete
    /// switch — there is no half-Julia — so both sides must agree. And the two
    /// programs have to fit end to end in the register file with room for the
    /// four nodes that join them; dead-code stripping is what usually makes
    /// that possible (evolved DAGs are about half intron, so live length runs
    /// roughly half of raw).
    pub fn blend_compatibility(&self, other: &Genome) -> Result<(), String> {
        if self.program.is_empty() || other.program.is_empty() {
            return Err("blending needs two DAG genomes; one of these has no program".into());
        }
        if self.julia_mode != other.julia_mode {
            return Err(format!(
                "julia mode differs ({} vs {}) — it is a discrete switch, not something                  that can be half-applied",
                self.julia_mode, other.julia_mode
            ));
        }
        let (a, b) = (live_len(&self.program), live_len(&other.program));
        if a + b + BLEND_GLUE_NODES > N_SLOTS {
            return Err(format!(
                "too big: {a} + {b} live nodes + {BLEND_GLUE_NODES} to join them exceeds the                  {N_SLOTS}-slot register file"
            ));
        }
        Ok(())
    }

    /// A genome whose iteration is `f_self + s·(f_other − f_self)`.
    ///
    /// This blends the FORMULAS, not the images. At every intermediate `s` the
    /// result is a genuine iterated map, so each frame is a real fractal with
    /// real boundary structure — not a cross-fade, where the in-between frames
    /// would be pictures of nothing.
    ///
    /// `s = 0` reproduces `self` exactly and `s = 1` reproduces `other`'s
    /// iteration exactly, so a morph can be dialled from nothing the same way a
    /// [`TimeMod`] can.
    ///
    /// Returns `None` when [`blend_compatibility`](Self::blend_compatibility)
    /// says no, so a caller can fall back to rendering `self` unchanged rather
    /// than produce an invalid program.
    pub fn blend_with(&self, other: &Genome, s: f32) -> Option<Genome> {
        if self.blend_compatibility(other).is_err() {
            return None;
        }
        let a = strip_dead(&self.program);
        let b = strip_dead(&other.program);
        let off = a.len() as u8;

        let mut prog = a.clone();
        for nd in &b {
            let mut m = *nd;
            let ar = op::arity(m.op);
            if ar >= 1 { m.a += off; }
            if ar >= 2 { m.b += off; }
            prog.push(m);
        }
        let root_a = (a.len() - 1) as u8;
        let root_b = (prog.len() - 1) as u8;

        // f_self + s·(f_other − f_self): four nodes, and one fewer than the
        // (1−s)·A + s·B form needs.
        let i_sub = prog.len() as u8;
        prog.push(OpNode { op: op::SUB, a: root_b, b: root_a, kre: 0.0, kim: 0.0 });
        let i_k = prog.len() as u8;
        prog.push(OpNode { op: op::CONST, a: 0, b: 0, kre: s, kim: 0.0 });
        let i_mul = prog.len() as u8;
        prog.push(OpNode { op: op::MUL, a: i_k, b: i_sub, kre: 0.0, kim: 0.0 });
        prog.push(OpNode { op: op::ADD, a: root_a, b: i_mul, kre: 0.0, kim: 0.0 });

        let mut g = self.clone();
        g.program = prog;
        g.terms = Vec::new();
        let lerp = |x: f32, y: f32| x + (y - x) * s;
        g.bailout_radius = lerp(self.bailout_radius, other.bailout_radius).max(MIN_BAILOUT_RADIUS);
        g.phoenix_re = lerp(self.phoenix_re, other.phoenix_re);
        g.phoenix_im = lerp(self.phoenix_im, other.phoenix_im);
        g.julia_cre = lerp(self.julia_cre, other.julia_cre);
        g.julia_cim = lerp(self.julia_cim, other.julia_cim);
        // The warp runs once on the pixel coordinate before iterating, so it is
        // not part of the blended map. Crossing from one warp to another is a
        // separate axis; keep this side's and say so rather than silently
        // dropping it.
        g.warp = self.warp.clone();
        Some(g)
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
            nima: 0.0, topiq_iaa: 0.0, ap25_score: 0.0, musiq: 0.0, aesthetic_ensemble: 0.0,
            pref_score: 0.0,
            self_replication: 0.0,
            fractal_recursion: 0.0,
            pred_recursion: 0.0,
            pred_clip: 0.0,
            formula_diversity: 0.0,
            angle_structure: 0.0,
            known_formula_match: String::new(),
            known_formula_score: 0.0,
            novelty_score: 0.0,
            novelty_cluster: -1,
            wormhole_score: 0.0,
            wormhole_dx: 0.0,
            wormhole_dy: 0.0,
            wormhole_zoom: 0.0,
            program: Vec::new(),
            julia_mode: false,
            julia_cre: 0.0,
            julia_cim: 0.0,
            phoenix_re: 0.0,
            phoenix_im: 0.0,
            bailout_radius: default_bailout_radius(),
            warp: Vec::new(),
            time_mod: Vec::new(),
            time_prog: Vec::new(),
            id: rng.random(),
            view_cx: view.0,
            view_cy: view.1,
            view_zoom: view.2,
        }
    }

    /// Randomize Phase-3/4 iteration dynamics for a fresh DAG genome. Most
    /// genomes stay standard (Mandelbrot-style, no phoenix, no warp); a minority
    /// get Julia mode / phoenix memory / a coordinate warp to seed unusual families.
    fn randomize_dynamics(&mut self, rng: &mut impl Rng) {
        if rng.random_bool(0.30) {
            self.julia_mode = true;
            self.julia_cre = re_k(rng) * 0.9;
            self.julia_cim = im_k(rng) * 0.9;
        }
        if rng.random_bool(0.25) {
            self.phoenix_re = re_k(rng) * 0.5;
            self.phoenix_im = im_k(rng) * 0.5;
        }
        self.bailout_radius = 2.0 + rng.random::<f32>() * 8.0; // [2, 10]
        if rng.random_bool(0.30) {
            // a small warp program bends the input plane
            let wexotic = rng.random_bool(0.5);
            self.warp = random_program(rng, 8, 4, wexotic);
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
            g.randomize_dynamics(rng);
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
            g.randomize_dynamics(rng);
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
            g.inherit_dynamics(a, b, rng);
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
            child.mutate_dynamics(rng);
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

    /// In-place mutation of Phase-3/4 iteration dynamics (DAG genomes).
    fn mutate_dynamics(&mut self, rng: &mut impl Rng) {
        if rng.random_bool(0.10) {
            self.julia_mode = !self.julia_mode;
            if self.julia_mode && self.julia_cre == 0.0 && self.julia_cim == 0.0 {
                self.julia_cre = re_k(rng) * 0.9;
                self.julia_cim = im_k(rng) * 0.9;
            }
        }
        if self.julia_mode && rng.random_bool(0.40) {
            self.julia_cre += re_k(rng) * 0.2;
            self.julia_cim += im_k(rng) * 0.2;
        }
        if rng.random_bool(0.20) {
            self.phoenix_re += re_k(rng) * 0.2;
            self.phoenix_im += im_k(rng) * 0.2;
        }
        if rng.random_bool(0.15) { self.phoenix_re = 0.0; self.phoenix_im = 0.0; }
        if rng.random_bool(0.30) {
            let f = 0.8 + rng.random::<f32>() * 0.5;
            self.bailout_radius = (self.bailout_radius * f).clamp(2.0, 16.0);
        }
        if rng.random_bool(0.15) {
            if self.warp.is_empty() {
                self.warp = random_program(rng, 8, 4, false);
            } else if rng.random_bool(0.30) {
                self.warp.clear();
            } else {
                self.warp = mutate_program(&self.warp, rng, 8, 4);
            }
        }
    }

    /// Inherit dynamics from one of two parents per field (crossover helper).
    fn inherit_dynamics(&mut self, a: &Self, b: &Self, rng: &mut impl Rng) {
        let pick = |rng: &mut dyn rand::RngCore| rng.random_bool(0.5);
        if pick(rng) { self.julia_mode = a.julia_mode; self.julia_cre = a.julia_cre; self.julia_cim = a.julia_cim; }
        else         { self.julia_mode = b.julia_mode; self.julia_cre = b.julia_cre; self.julia_cim = b.julia_cim; }
        if pick(rng) { self.phoenix_re = a.phoenix_re; self.phoenix_im = a.phoenix_im; }
        else         { self.phoenix_re = b.phoenix_re; self.phoenix_im = b.phoenix_im; }
        self.bailout_radius = (a.bailout_radius + b.bailout_radius) * 0.5;
        self.warp = if pick(rng) { a.warp.clone() } else { b.warp.clone() };
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


/// Smallest escape radius a `Bailout` modulation may drive to. Below roughly
/// this every orbit escapes immediately and the frame goes uniformly flat.
pub const MIN_BAILOUT_RADIUS: f32 = 0.25;

/// Add `(dre, dim)` to node `idx`'s constant, if that node exists and is
/// actually a CONST. Silently does nothing otherwise — a stale node index in a
/// hand-edited `.nn` should render a static fractal, not panic.
/// Add a complex offset to whichever scalar `target` names.
///
/// The one place the two animation systems meet: a fixed-shape `TimeMod` and an
/// evolved `TimeProgram` differ entirely in how they produce `(dre, dim)` and
/// not at all in what they do with it, so `at_time` computes the offset two ways
/// and lands here either way.
fn apply_offset(g: &mut Genome, target: ModTarget, dre: f32, dim: f32) {
    match target {
        ModTarget::JuliaC => {
            g.julia_cre += dre;
            g.julia_cim += dim;
        }
        ModTarget::Phoenix => {
            g.phoenix_re += dre;
            g.phoenix_im += dim;
        }
        ModTarget::Bailout => {
            // A non-positive escape radius makes every orbit escape on
            // iteration 0 and the frame goes flat, so floor it rather
            // than let a large amplitude destroy the clip.
            g.bailout_radius = (g.bailout_radius + dre).max(MIN_BAILOUT_RADIUS);
        }
        ModTarget::ProgConst { node } => {
            offset_const(&mut g.program, node, dre, dim);
        }
        ModTarget::WarpConst { node } => {
            offset_const(&mut g.warp, node, dre, dim);
        }
        ModTarget::ProgScale { node } => {
            // k = 1 + offset, NOT the offset itself: a multiplier of 0
            // would annihilate the subtree, and `amp = 0` has to stay
            // the identity like every other target.
            if let Some(next) = splice_scale(&g.program, node, 1.0 + dre, dim) {
                g.program = next;
            }
        }
    }
}

fn offset_const(prog: &mut [OpNode], idx: u8, dre: f32, dim: f32) {
    if let Some(n) = prog.get_mut(idx as usize) {
        if n.op == op::CONST {
            n.kre += dre;
            n.kim += dim;
        }
    }
}

/// Splice `MUL(CONST(kre,kim), prog[node])` into `prog` so the subtree rooted at
/// `node` is scaled by a complex constant.
///
/// The two new nodes go directly AFTER `node` rather than at the end, and every
/// later operand index is remapped (+2, and any reference to `node` itself
/// repointed at the new MUL). Appending at the end instead would make the MUL
/// the program's root, which silently replaces the whole formula with a scaled
/// copy of one subtree — right only when `node` happens to be the root already.
///
/// Returns `None` if there is no room within [`N_SLOTS`], leaving the caller to
/// render the genome unmodulated rather than produce an invalid program.
fn splice_scale(prog: &[OpNode], node: u8, kre: f32, kim: f32) -> Option<Vec<OpNode>> {
    let n = node as usize;
    if n >= prog.len() || prog.len() + 2 > N_SLOTS {
        return None;
    }
    let k_idx = (n + 1) as u8;
    let mul_idx = (n + 2) as u8;

    let mut out: Vec<OpNode> = Vec::with_capacity(prog.len() + 2);
    out.extend_from_slice(&prog[..=n]);
    out.push(OpNode { op: op::CONST, a: 0, b: 0, kre, kim });
    out.push(OpNode { op: op::MUL, a: k_idx, b: node, kre: 0.0, kim: 0.0 });

    // Everything after the splice point shifts by 2; references to `node` now
    // mean "the scaled version of node".
    let remap = |i: u8| -> u8 {
        if i as usize == n { mul_idx } else if (i as usize) > n { i + 2 } else { i }
    };
    for src in &prog[n + 1..] {
        let mut nd = *src;
        let arity = op::arity(nd.op);
        if arity >= 1 { nd.a = remap(nd.a); }
        if arity >= 2 { nd.b = remap(nd.b); }
        out.push(nd);
    }
    Some(out)
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
    render_node_with(prog, i, depth, "z", "c")
}

/// `render_node` with the two leaves named by the caller.
///
/// The DAG is the same graph whether it is iterating a fractal or driving the
/// time axis, but the leaves are not the same thing: `crate::time_program`
/// reads `op::C` as a phasor and `op::Z` as a ramp, and printing those as
/// `c` and `z` would describe a formula nobody wrote.
pub(crate) fn render_node_with(
    prog: &[OpNode], i: usize, depth: usize, z_label: &str, c_label: &str,
) -> String {
    if depth > 14 || i >= prog.len() { return "…".into(); }
    let n = prog[i];
    let a = (n.a as usize).min(i.saturating_sub(1));
    let b = (n.b as usize).min(i.saturating_sub(1));
    let ra = || render_node_with(prog, a, depth + 1, z_label, c_label);
    let rb = || render_node_with(prog, b, depth + 1, z_label, c_label);
    match n.op {
        op::Z       => z_label.into(),
        op::C       => c_label.into(),
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


/// Nodes needed to join two programs into a blended one: SUB, CONST, MUL, ADD.
pub const BLEND_GLUE_NODES: usize = 4;

/// `prog` with every node that cannot reach the root removed, renumbered.
///
/// Evolved DAGs accumulate introns — measured across a 400-genome sample, live
/// length is a median of 6 against a raw median of 11. For blending that is the
/// difference between fitting the register file 53% of the time and 95% of it.
pub fn strip_dead(prog: &[OpNode]) -> Vec<OpNode> {
    let live = crate::formula::reachable_from_root(prog);
    let mut remap = vec![0u8; prog.len()];
    let mut out: Vec<OpNode> = Vec::with_capacity(prog.len());
    for (i, nd) in prog.iter().enumerate() {
        if !live[i] { continue; }
        remap[i] = out.len() as u8;
        let mut m = *nd;
        let ar = op::arity(m.op);
        if ar >= 1 { m.a = remap[m.a as usize]; }
        if ar >= 2 { m.b = remap[m.b as usize]; }
        out.push(m);
    }
    out
}

/// How many of `prog`'s nodes actually contribute to its result.
pub fn live_len(prog: &[OpNode]) -> usize {
    crate::formula::reachable_from_root(prog).iter().filter(|v| **v).count()
}

#[cfg(test)]
mod at_time_tests {
    use super::*;
    use crate::formula::{ModShape, ModTarget, TimeMod, eval_program};

    /// Same validity rule the GP operators are held to: every operand must name
    /// a strictly-earlier node, and the program must fit the register file.
    fn is_valid_dag(prog: &[OpNode]) -> bool {
        prog.len() <= N_SLOTS
            && prog.iter().enumerate().all(|(i, n)| {
                let arity = op::arity(n.op);
                (arity < 1 || (n.a as usize) < i) && (arity < 2 || (n.b as usize) < i)
            })
    }

    fn mandelbrot() -> Genome {
        let mut g = Genome::default();
        g.program = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        g.bailout_radius = 4.0;
        g
    }

    // ── Evolved time formulas (crate::time_program) ──────────────────────────

    /// The phasor leaf alone: `f(t) = e^{2πit}`, the canonical looping program.
    fn phasor_prog(target: ModTarget, amp: f32) -> crate::time_program::TimeProgram {
        crate::time_program::TimeProgram::new(
            target,
            vec![OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 }],
            amp,
        )
    }

    #[test]
    fn an_empty_time_prog_is_a_plain_clone_at_every_t() {
        // Same guarantee as the `time_mod` version: adding the field must not
        // change how a single already-archived genome renders.
        let g = mandelbrot();
        let base = serde_json::to_string(&g).unwrap();
        assert!(g.time_prog.is_empty());
        for i in 0..25 {
            let t = i as f32 / 25.0;
            assert_eq!(serde_json::to_string(&g.at_time(t)).unwrap(), base, "t={t}");
        }
    }

    #[test]
    fn a_time_program_drives_julia_c_around_its_stored_value() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = -0.4;
        g.julia_cim = 0.6;
        g.time_prog = vec![phasor_prog(ModTarget::JuliaC, 0.1)];

        // t=0 puts the phasor at (1,0), so c moves along +re by exactly amp.
        let a = g.at_time(0.0);
        assert!((a.julia_cre - (-0.4 + 0.1)).abs() < 1e-5, "got {}", a.julia_cre);
        assert!((a.julia_cim - 0.6).abs() < 1e-5, "got {}", a.julia_cim);

        // A quarter turn later it is on +im, still at radius amp from centre.
        let b = g.at_time(0.25);
        assert!((b.julia_cre - (-0.4)).abs() < 1e-5, "got {}", b.julia_cre);
        assert!((b.julia_cim - (0.6 + 0.1)).abs() < 1e-5, "got {}", b.julia_cim);
    }

    #[test]
    fn a_time_program_with_zero_amp_leaves_the_genome_untouched() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = -0.4;
        g.julia_cim = 0.6;
        let base = serde_json::to_string(&g).unwrap();
        g.time_prog = vec![phasor_prog(ModTarget::JuliaC, 0.0)];

        let mut expected: Genome = serde_json::from_str(&base).unwrap();
        expected.time_prog = g.time_prog.clone();
        let want = serde_json::to_string(&expected).unwrap();
        for i in 0..10 {
            assert_eq!(serde_json::to_string(&g.at_time(i as f32 / 10.0)).unwrap(), want,
                       "amp=0 must be the identity, exactly as it is for TimeMod");
        }
    }

    #[test]
    fn time_mod_and_time_prog_stack_additively() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = 0.0;
        g.julia_cim = 0.0;
        // Cosine at t=0 is +1 → +0.2 on re; phasor at t=0 is (1,0) → +0.05 on re.
        g.time_mod = vec![TimeMod::new(ModTarget::JuliaC, ModShape::Cosine, 0.2)];
        g.time_prog = vec![phasor_prog(ModTarget::JuliaC, 0.05)];
        let a = g.at_time(0.0);
        assert!((a.julia_cre - 0.25).abs() < 1e-5,
                "both channels must apply, got {}", a.julia_cre);
    }

    #[test]
    fn a_rejected_time_program_renders_the_fractal_unchanged() {
        // A constant f(t) is not an animation; it must not corrupt the genome.
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = -0.4;
        g.time_prog = vec![crate::time_program::TimeProgram::new(
            ModTarget::JuliaC,
            vec![OpNode { op: op::CONST, a: 0, b: 0, kre: 9.0, kim: 9.0 }],
            1.0,
        )];
        assert!(!g.time_prog[0].profile().passed());
        assert!((g.at_time(0.7).julia_cre - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn animates_reports_either_channel() {
        let mut g = mandelbrot();
        assert!(!g.animates());
        g.time_prog = vec![phasor_prog(ModTarget::Bailout, 1.0)];
        assert!(g.animates(), "an evolved time formula animates the formula too");
        g.time_prog.clear();
        g.time_mod = vec![TimeMod::new(ModTarget::Bailout, ModShape::Sine, 1.0)];
        assert!(g.animates());
    }

    #[test]
    fn a_genome_without_time_prog_still_loads() {
        // Backward compatibility, against the shape every archived .nn has.
        let json = r#"{"terms":[],"fitness":0.0,"program":[{"op":0,"a":0,"b":0}],"id":7}"#;
        let g: Genome = serde_json::from_str(json).unwrap();
        assert!(g.time_prog.is_empty());
        assert!(g.time_mod.is_empty());
        assert!(!g.animates());
    }

    #[test]
    fn a_genome_with_time_prog_round_trips() {
        let mut g = mandelbrot();
        g.time_prog = vec![crate::time_program::TimeProgram {
            target: ModTarget::ProgScale { node: 2 },
            prog: vec![
                OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
                OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            ],
            amp: 0.3,
            freq: 2.0,
            phase: 0.125,
        }];
        let s = serde_json::to_string(&g).unwrap();
        let back: Genome = serde_json::from_str(&s).unwrap();
        assert_eq!(back.time_prog, g.time_prog);
        // And the saved formula is self-describing.
        assert!(g.formula_expr().contains("[tprog:"), "{}", g.formula_expr());
    }

    #[test]
    fn a_time_program_that_scales_a_subtree_keeps_the_dag_valid() {
        // ProgScale splices two nodes at render time; the result still has to
        // be a legal program for every kernel that will evaluate it.
        let mut g = mandelbrot();
        g.time_prog = vec![phasor_prog(ModTarget::ProgScale { node: 2 }, 0.5)];
        for i in 0..12 {
            let a = g.at_time(i as f32 / 12.0);
            assert!(is_valid_dag(&a.program), "t={i}/12 produced an invalid DAG: {:?}", a.program);
        }
    }

    #[test]
    fn an_empty_time_mod_is_a_plain_clone_at_every_t() {
        // The guarantee that nothing about existing rendering changed.
        let g = mandelbrot();
        let base = serde_json::to_string(&g).unwrap();
        for i in 0..25 {
            let t = i as f32 / 25.0;
            assert_eq!(serde_json::to_string(&g.at_time(t)).unwrap(), base, "t={t}");
        }
    }

    #[test]
    fn zero_amplitude_leaves_the_genome_untouched() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = -0.4;
        g.julia_cim = 0.6;
        let base = serde_json::to_string(&g).unwrap();
        g.time_mod = vec![TimeMod::new(ModTarget::JuliaC, ModShape::Orbit, 0.0)];

        let mut expected: Genome = serde_json::from_str(&base).unwrap();
        expected.time_mod = g.time_mod.clone();
        let want = serde_json::to_string(&expected).unwrap();
        for i in 0..10 {
            assert_eq!(serde_json::to_string(&g.at_time(i as f32 / 10.0)).unwrap(), want);
        }
    }

    #[test]
    fn julia_orbit_moves_c_on_a_circle_around_its_stored_value() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.julia_cre = -0.4;
        g.julia_cim = 0.6;
        g.time_mod = vec![TimeMod::new(ModTarget::JuliaC, ModShape::Orbit, 0.1)];
        for i in 0..16 {
            let t = i as f32 / 16.0;
            let a = g.at_time(t);
            let r = ((a.julia_cre + 0.4).powi(2) + (a.julia_cim - 0.6).powi(2)).sqrt();
            assert!((r - 0.1).abs() < 1e-5, "radius {r} at t={t}");
        }
    }

    #[test]
    fn bailout_never_drops_to_a_degenerate_radius() {
        // A huge amplitude must clamp, not produce a radius that makes every
        // orbit escape on iteration 0 and the frame go flat.
        let mut g = mandelbrot();
        g.time_mod = vec![TimeMod::new(ModTarget::Bailout, ModShape::Sine, 50.0)];
        for i in 0..40 {
            let a = g.at_time(i as f32 / 40.0);
            assert!(a.bailout_radius >= MIN_BAILOUT_RADIUS,
                "radius {} below floor", a.bailout_radius);
        }
    }

    #[test]
    fn prog_scale_produces_a_valid_dag_and_actually_scales() {
        let mut g = mandelbrot();
        // Scale node 2 (the SQR), which is NOT the root — the case where naive
        // append-at-the-end would silently replace the whole formula.
        g.time_mod = vec![TimeMod::new(ModTarget::ProgScale { node: 2 }, ModShape::Cosine, 0.5)];
        let a = g.at_time(0.0); // cos(0) = 1 → k = 1.5
        assert_eq!(a.program.len(), 6, "two nodes spliced in");
        assert!(is_valid_dag(&a.program), "invalid DAG: {:?}", a.program);

        // The root must still be the ADD, with its SQR operand repointed at the
        // new MUL — i.e. z' = 1.5·z² + c, not 1.5·z².
        let root = a.program.last().unwrap();
        assert_eq!(root.op, op::ADD, "root should still be the ADD");

        let (zx, zy, cx, cy) = (0.3f32, 0.2f32, -0.1f32, 0.4f32);
        let (got_x, got_y) = eval_program(&a.program, zx, zy, cx, cy);
        // 1.5·(z²) + c, computed independently.
        let (sx, sy) = (zx * zx - zy * zy, 2.0 * zx * zy);
        let (want_x, want_y) = (1.5 * sx + cx, 1.5 * sy + cy);
        assert!((got_x - want_x).abs() < 1e-5 && (got_y - want_y).abs() < 1e-5,
            "got ({got_x},{got_y}) want ({want_x},{want_y})");
    }

    #[test]
    fn prog_scale_at_zero_amplitude_evaluates_identically() {
        // The spliced program has two extra nodes, so it is not byte-identical —
        // but multiplying by k = 1 must leave the VALUE untouched.
        let mut g = mandelbrot();
        g.time_mod = vec![TimeMod::new(ModTarget::ProgScale { node: 2 }, ModShape::Sine, 0.0)];
        let a = g.at_time(0.25);
        for (zx, zy, cx, cy) in [(0.3f32, 0.2f32, -0.1f32, 0.4f32), (-1.2, 0.7, 0.05, -0.3)] {
            let plain = eval_program(&g.program, zx, zy, cx, cy);
            let scaled = eval_program(&a.program, zx, zy, cx, cy);
            assert!((plain.0 - scaled.0).abs() < 1e-6 && (plain.1 - scaled.1).abs() < 1e-6,
                "k=1 changed the value: {plain:?} vs {scaled:?}");
        }
    }

    #[test]
    fn prog_scale_is_refused_when_the_register_file_is_full() {
        let mut g = mandelbrot();
        // Pad to N_SLOTS-1 so a +2 splice cannot fit.
        while g.program.len() < N_SLOTS - 1 {
            let i = g.program.len() as u8;
            g.program.push(OpNode { op: op::ADD, a: i - 1, b: i - 2, kre: 0.0, kim: 0.0 });
        }
        let before = g.program.clone();
        g.time_mod = vec![TimeMod::new(ModTarget::ProgScale { node: 2 }, ModShape::Sine, 0.5)];
        let a = g.at_time(0.3);
        assert_eq!(a.program, before, "must render unmodulated rather than overflow");
        assert!(is_valid_dag(&a.program));
    }

    #[test]
    fn prog_scale_at_the_root_scales_the_whole_formula() {
        let mut g = mandelbrot();
        let root = (g.program.len() - 1) as u8;
        g.time_mod = vec![TimeMod::new(ModTarget::ProgScale { node: root }, ModShape::Cosine, 1.0)];
        let a = g.at_time(0.0); // k = 2
        assert!(is_valid_dag(&a.program));
        let (zx, zy, cx, cy) = (0.3f32, 0.2f32, -0.1f32, 0.4f32);
        let plain = eval_program(&g.program, zx, zy, cx, cy);
        let scaled = eval_program(&a.program, zx, zy, cx, cy);
        assert!((scaled.0 - 2.0 * plain.0).abs() < 1e-5
             && (scaled.1 - 2.0 * plain.1).abs() < 1e-5,
            "root scale should double: {plain:?} -> {scaled:?}");
    }

    #[test]
    fn a_stale_node_index_is_ignored_rather_than_panicking() {
        let mut g = mandelbrot();
        g.time_mod = vec![
            TimeMod::new(ModTarget::ProgConst { node: 99 }, ModShape::Sine, 0.5),
            TimeMod::new(ModTarget::ProgScale { node: 99 }, ModShape::Sine, 0.5),
            TimeMod::new(ModTarget::WarpConst { node: 3 }, ModShape::Sine, 0.5),
        ];
        let a = g.at_time(0.4);
        assert_eq!(a.program, g.program);
        assert!(a.warp.is_empty());
    }

    #[test]
    fn two_modulations_on_one_scalar_sum() {
        let mut g = mandelbrot();
        g.julia_mode = true;
        g.time_mod = vec![
            TimeMod { target: ModTarget::JuliaC, shape: ModShape::Cosine, amp: 0.1, freq: 1.0, phase: 0.0 },
            TimeMod { target: ModTarget::JuliaC, shape: ModShape::Cosine, amp: 0.2, freq: 1.0, phase: 0.0 },
        ];
        // cos(0) = 1 for both → +0.3 total.
        assert!((g.at_time(0.0).julia_cre - 0.3).abs() < 1e-6);
    }

    #[test]
    fn time_mod_survives_a_genome_round_trip_and_old_files_default_to_empty() {
        let mut g = mandelbrot();
        g.time_mod = vec![TimeMod::new(ModTarget::JuliaC, ModShape::Orbit, 0.2)];
        let json = serde_json::to_string(&g).unwrap();
        let back: Genome = serde_json::from_str(&json).unwrap();
        assert_eq!(back.time_mod, g.time_mod);

        // A real archive file, written long before this field existed.
        let archive = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fractals_1");
        if let Ok(entries) = std::fs::read_dir(&archive) {
            if let Some(nn) = entries.flatten()
                .map(|e| e.path())
                .find(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
            {
                let old: Genome = serde_json::from_str(&std::fs::read_to_string(&nn).unwrap())
                    .unwrap_or_else(|e| panic!("{} failed to load: {e}", nn.display()));
                assert!(old.time_mod.is_empty(),
                    "an archived genome must load as static");
            }
        }
    }
}

#[cfg(test)]
mod blend_tests {
    use super::*;
    use crate::formula::eval_program;

    fn n(o: u8, a: u8, b: u8) -> OpNode { OpNode { op: o, a, b, kre: 0.0, kim: 0.0 } }

    fn mandel() -> Genome {
        let mut g = Genome::default();
        g.program = vec![n(op::Z,0,0), n(op::C,0,0), n(op::SQR,0,0), n(op::ADD,2,1)];
        g.bailout_radius = 4.0;
        g
    }
    /// z' = z³ + c
    fn cubic() -> Genome {
        let mut g = Genome::default();
        g.program = vec![n(op::Z,0,0), n(op::C,0,0), n(op::CUBE,0,0), n(op::ADD,2,1)];
        g.bailout_radius = 8.0;
        g
    }

    fn is_valid(prog: &[OpNode]) -> bool {
        prog.len() <= N_SLOTS && prog.iter().enumerate().all(|(i, nd)| {
            let ar = op::arity(nd.op);
            (ar < 1 || (nd.a as usize) < i) && (ar < 2 || (nd.b as usize) < i)
        })
    }

    #[test]
    fn the_endpoints_reproduce_each_side_exactly() {
        // The invariant the whole feature rests on: a morph must be dialable
        // from nothing, like every TimeMod.
        let (a, b) = (mandel(), cubic());
        for (zx, zy, cx, cy) in [(0.3f32,0.2f32,-0.1f32,0.4f32), (-1.1,0.6,0.05,-0.3)] {
            let at0 = a.blend_with(&b, 0.0).unwrap();
            let want_a = eval_program(&a.program, zx, zy, cx, cy);
            let got_a = eval_program(&at0.program, zx, zy, cx, cy);
            assert!((got_a.0-want_a.0).abs() < 1e-5 && (got_a.1-want_a.1).abs() < 1e-5,
                "s=0 must be A: {got_a:?} vs {want_a:?}");

            let at1 = a.blend_with(&b, 1.0).unwrap();
            let want_b = eval_program(&b.program, zx, zy, cx, cy);
            let got_b = eval_program(&at1.program, zx, zy, cx, cy);
            assert!((got_b.0-want_b.0).abs() < 1e-5 && (got_b.1-want_b.1).abs() < 1e-5,
                "s=1 must be B: {got_b:?} vs {want_b:?}");
        }
    }

    #[test]
    fn the_midpoint_is_the_average_of_the_two_maps() {
        let (a, b) = (mandel(), cubic());
        let mid = a.blend_with(&b, 0.5).unwrap();
        let (zx, zy, cx, cy) = (0.3f32, 0.2f32, -0.1f32, 0.4f32);
        let fa = eval_program(&a.program, zx, zy, cx, cy);
        let fb = eval_program(&b.program, zx, zy, cx, cy);
        let got = eval_program(&mid.program, zx, zy, cx, cy);
        assert!((got.0 - 0.5*(fa.0+fb.0)).abs() < 1e-5
             && (got.1 - 0.5*(fa.1+fb.1)).abs() < 1e-5,
            "midpoint {got:?} should be the mean of {fa:?} and {fb:?}");
    }

    #[test]
    fn a_blend_is_always_a_valid_dag() {
        let (a, b) = (mandel(), cubic());
        for i in 0..=10 {
            let g = a.blend_with(&b, i as f32 / 10.0).unwrap();
            assert!(is_valid(&g.program), "invalid at s={i}: {:?}", g.program);
        }
    }

    #[test]
    fn scalars_lerp_and_bailout_stays_usable() {
        let (a, b) = (mandel(), cubic());   // bailout 4 and 8
        let mid = a.blend_with(&b, 0.5).unwrap();
        assert!((mid.bailout_radius - 6.0).abs() < 1e-5);
        assert!(mid.bailout_radius >= MIN_BAILOUT_RADIUS);
    }

    #[test]
    fn julia_mode_mismatch_is_refused_with_a_reason() {
        let a = mandel();
        let mut b = cubic();
        b.julia_mode = true;
        let err = a.blend_compatibility(&b).unwrap_err();
        assert!(err.contains("julia"), "{err}");
        assert!(a.blend_with(&b, 0.5).is_none());
    }

    #[test]
    fn an_oversized_pair_is_refused_rather_than_truncated() {
        let mut a = mandel();
        let mut b = cubic();
        // Push both to 11 live nodes: 11 + 11 + 4 = 26 > 24.
        for g in [&mut a, &mut b] {
            while live_len(&g.program) < 11 {
                let i = (g.program.len() - 1) as u8;
                g.program.push(OpNode { op: op::ADD, a: i, b: i.saturating_sub(1), kre: 0.0, kim: 0.0 });
            }
        }
        let err = a.blend_compatibility(&b).unwrap_err();
        assert!(err.contains("register file"), "{err}");
        assert!(a.blend_with(&b, 0.5).is_none());
    }

    #[test]
    fn dead_nodes_are_stripped_so_more_pairs_fit() {
        // The measured enabler: raw length would blow the budget, live length
        // does not.
        let mut a = mandel();
        // 16 nodes of intron nothing reads.
        for _ in 0..12 {
            a.program.insert(3, OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 });
        }
        let last = a.program.len() - 1;
        a.program[last] = OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 };
        assert_eq!(a.program.len(), 16);
        assert_eq!(live_len(&a.program), 4, "only z, c, sqr, add are live");
        let b = cubic();
        assert!(a.blend_compatibility(&b).is_ok(), "should fit once introns are ignored");
        let g = a.blend_with(&b, 0.5).unwrap();
        assert_eq!(g.program.len(), 4 + 4 + BLEND_GLUE_NODES);
        assert!(is_valid(&g.program));
    }

    #[test]
    fn strip_dead_preserves_the_computed_value() {
        let mut a = mandel();
        for _ in 0..5 {
            a.program.insert(2, OpNode { op: op::EXP, a: 0, b: 0, kre: 0.0, kim: 0.0 });
        }
        let last = a.program.len() - 1;
        a.program[last] = OpNode { op: op::ADD, a: (last - 1) as u8, b: 1, kre: 0.0, kim: 0.0 };
        let stripped = strip_dead(&a.program);
        let (zx, zy, cx, cy) = (0.3f32, 0.2f32, -0.1f32, 0.4f32);
        let before = eval_program(&a.program, zx, zy, cx, cy);
        let after = eval_program(&stripped, zx, zy, cx, cy);
        assert!((before.0-after.0).abs() < 1e-6 && (before.1-after.1).abs() < 1e-6,
            "stripping changed the value: {before:?} -> {after:?}");
    }
}
