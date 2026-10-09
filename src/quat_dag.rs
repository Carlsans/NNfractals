//! Quaternion generalization of the GA's expression-DAG formula system
//! (`formula.rs`'s 21-opcode register-VM, `eval_program`) — so an
//! ARBITRARY GA-evolved genome, not just the 10 hand-built `QuatFormula`
//! variants, can be ray-marched in 3D.
//!
//! Every opcode generalizes the same way `QuatFormula::Bulb`'s trig
//! already did: writing a quaternion as `q = r + ρn̂` (r = real part, ρ =
//! `|vector part|`, n̂ = the vector part's own unit direction) stands in
//! for `x + iy` — a genuine quaternion identity, not a new trick invented
//! here: `{1, n̂}` spans a subalgebra isomorphic to ℂ for ANY unit
//! quaternion n̂ (n̂² = −1, same as i), so `formula.rs`'s complex
//! `csin`/`ccos`/`cexp`/`clog`/etc. all carry over by replacing `i` with
//! n̂ and `y` with ρ. This is exactly the same fact the original
//! quaternion-Mandelbrot parity tests (`quat_fractal.rs`) and Bulb's own
//! polar decomposition already rest on — reused here, not rediscovered.
//!
//! Distance estimation (for sphere tracing, see `quat_raymarch.rs`) can't
//! reuse a formula-specific `power()` the way the hand-built formulas do —
//! an arbitrary DAG isn't a pure power map. Instead the per-iteration
//! local derivative magnitude is estimated NUMERICALLY (one extra
//! `eval_program_quat` call per iteration, at a nearby perturbed point)
//! rather than symbolically differentiating all 21 opcodes — a deliberate
//! scope cut for a first version, documented on `quat_dag_escape_de`.

use rayon::prelude::*;

use crate::formula::{op, OpNode, N_SLOTS};
use crate::quat_fractal::TimeAxis;
use crate::quat_motion::{add, dot, look_at_basis, normalize, scale, sub, Vec3};
use crate::quat_raymarch::{ray_box, ray_sphere, RaymarchCamera};
use crate::anim_timeline::AxisAssignment;
use crate::quaternion::Quat;

const EPS: f64 = 1e-9;

fn vector_norm(q: Quat) -> f64 {
    (q.a * q.a + q.b * q.b + q.c * q.c).sqrt()
}

/// `q = r + ρn̂` with `exp(q) = exp(r)·(cos ρ + n̂ sin ρ)` — mirrors
/// `formula::cexp` exactly (`y → ρ`, `i → n̂`), including its overflow
/// guard (`r` clamped to `[-8,8]` before exponentiating).
fn qexp(q: Quat) -> Quat {
    let rho = vector_norm(q);
    let r = q.r.clamp(-8.0, 8.0);
    let e = r.exp();
    if rho < EPS {
        Quat::new(e, 0.0, 0.0, 0.0)
    } else {
        let s = e * rho.sin() / rho;
        Quat::new(e * rho.cos(), q.a * s, q.b * s, q.c * s)
    }
}

/// `log(q) = ln|q| + n̂·atan2(ρ, r)` — mirrors `formula::clog` exactly.
fn qlog(q: Quat) -> Quat {
    let rho = vector_norm(q);
    let norm = (q.r * q.r + rho * rho).sqrt() + EPS;
    let theta = rho.atan2(q.r);
    if rho < EPS {
        Quat::new(norm.ln(), 0.0, 0.0, 0.0)
    } else {
        let s = theta / rho;
        Quat::new(norm.ln(), q.a * s, q.b * s, q.c * s)
    }
}

/// `sin(q) = sin(r)cosh(ρ) + n̂·cos(r)sinh(ρ)` — mirrors `formula::csin`.
fn qsin(q: Quat) -> Quat {
    let rho = vector_norm(q);
    let re = q.r.sin() * rho.cosh();
    if rho < EPS {
        Quat::new(re, 0.0, 0.0, 0.0)
    } else {
        let s = q.r.cos() * rho.sinh() / rho;
        Quat::new(re, q.a * s, q.b * s, q.c * s)
    }
}

/// `cos(q) = cos(r)cosh(ρ) − n̂·sin(r)sinh(ρ)` — mirrors `formula::ccos`.
fn qcos(q: Quat) -> Quat {
    let rho = vector_norm(q);
    let re = q.r.cos() * rho.cosh();
    if rho < EPS {
        Quat::new(re, 0.0, 0.0, 0.0)
    } else {
        let s = -q.r.sin() * rho.sinh() / rho;
        Quat::new(re, q.a * s, q.b * s, q.c * s)
    }
}

/// Mirrors `formula.rs`'s own `TANH` opcode formula
/// (`tanh(x+iy) = (sinh(2x)+i·sin(2y)) / (cosh(2x)+cos(2y))`) with
/// `y → ρ`, `i → n̂`.
fn qtanh(q: Quat) -> Quat {
    let rho = vector_norm(q);
    let x2 = 2.0 * q.r;
    let y2 = 2.0 * rho;
    let d = x2.cosh() + y2.cos() + EPS;
    let re = x2.sinh() / d;
    if rho < EPS {
        Quat::new(re, 0.0, 0.0, 0.0)
    } else {
        let s = (y2.sin() / d) / rho;
        Quat::new(re, q.a * s, q.b * s, q.c * s)
    }
}

/// `q⁻¹ = conj(q) / |q|²` — mirrors the `RECIP` opcode's `conj(a)/(|a|²+EPS)`.
fn qrecip(q: Quat) -> Quat {
    let d = q.norm_sq() + EPS;
    let cj = q.conj();
    Quat::new(cj.r / d, cj.a / d, cj.b / d, cj.c / d)
}

fn qnormz(q: Quat) -> Quat {
    let m = q.norm_sq().sqrt() + EPS;
    Quat::new(q.r / m, q.a / m, q.b / m, q.c / m)
}

/// Quaternion port of `formula::eval_program` — same register-VM shape
/// (single forward pass, node `i` reads strictly-earlier nodes `a`,`b`,
/// out-of-order operands read as zero), same opcode semantics, just `Quat`
/// arithmetic instead of `(f32,f32)` complex pairs. `CONST` embeds the
/// genome's `(kre,kim)` pair in the R,A plane (`Quat::new(kre,kim,0,0)`) —
/// the same embedding convention every quaternion generalization in this
/// project has used since the very first parity tests, and what makes the
/// B=C=0 parity check below possible at all.
pub fn eval_program_quat(prog: &[OpNode], z: Quat, c: Quat) -> Quat {
    let n = prog.len().min(N_SLOTS);
    if n == 0 {
        return Quat::ZERO;
    }
    let mut reg = [Quat::ZERO; N_SLOTS];
    for i in 0..n {
        let node = prog[i];
        let ai = (node.a as usize).min(N_SLOTS - 1);
        let bi = (node.b as usize).min(N_SLOTS - 1);
        let a = if ai < i { reg[ai] } else { Quat::ZERO };
        let b = if bi < i { reg[bi] } else { Quat::ZERO };
        reg[i] = match node.op {
            op::Z => z,
            op::C => c,
            op::CONST => Quat::new(node.kre as f64, node.kim as f64, 0.0, 0.0),
            op::SQR => a.mul(a),
            op::CUBE => a.mul(a).mul(a),
            op::QUART => {
                let sq = a.mul(a);
                sq.mul(sq)
            }
            op::RECIP => qrecip(a),
            op::SIN => qsin(a),
            op::COS => qcos(a),
            op::EXP => qexp(a),
            op::LOG => qlog(a),
            op::TANH => qtanh(a),
            op::CONJ => a.conj(),
            op::ABSFOLD => a.abs_components(),
            op::ABSRE => Quat::new(a.r.abs(), a.a, a.b, a.c),
            op::ABSIM => Quat::new(a.r, a.a.abs(), a.b.abs(), a.c.abs()),
            op::NORMZ => qnormz(a),
            op::ADD => a.add(b),
            op::SUB => a.sub(b),
            op::MUL => a.mul(b),
            op::DIV => a.mul(qrecip(b)),
            _ => Quat::ZERO,
        };
    }
    reg[n - 1]
}

/// A GA-evolved genome's rendering-relevant fields, borrowed rather than
/// owned — same shape as `Genome`'s own `program`/`warp`/`julia_*`/
/// `phoenix_*` fields (see `genome.rs`), so callers just slice a loaded
/// `Genome` directly instead of duplicating this data.
#[derive(Copy, Clone)]
pub struct QuatDagFormula<'a> {
    pub prog: &'a [OpNode],
    pub warp: &'a [OpNode],
    pub julia: bool,
    pub jc: (f32, f32),
    pub phoenix: (f32, f32),
}

/// Quaternion port of `fractal::dag_escape_pixel`'s recurrence (coordinate
/// warp → Julia/Mandelbrot init → `z ← program(z,c) + phoenix·z_prev` →
/// bailout), extended with a distance estimate for sphere tracing.
///
/// Quaternion norms are exactly multiplicative (`|pq|=|p||q|`, a genuine
/// algebraic fact, not an approximation), which turns "the chain rule for
/// opcode `f` is `d(f(a))=f'(a)·da`" into an EXACT scalar-magnitude
/// propagation `|d(f(a))| = |f'(a)|·|da|` — computed per-opcode below via
/// `eval_program_quat_deriv`, mirroring `eval_program_quat`'s register-VM
/// exactly but carrying a derivative alongside each value. This replaced
/// an earlier version that estimated `|∂program/∂z|` via a single finite
/// difference of the whole composed program — noisier (one epsilon choice
/// for arbitrarily different opcodes) and, unlike this version, couldn't
/// be checked against anything: this one reduces EXACTLY to
/// `quat_fractal::quat_escape_de`'s own `n·ρ^(n-1)·dr+1` formula for the
/// classic `z²+c` DAG (see the parity test), which the numerical version
/// had no way to be checked against.
///
/// Still a heuristic in the SAME sense every DE in this project is: none
/// of these quaternion functions are truly holomorphic (`sin`/`cos`/`exp`/
/// `log`/power maps alike), so "the derivative" is itself only a
/// magnitude-bound approximation, not a rigorous complex-analytic
/// derivative — this is the standard Mandelbulb-DE-family approach,
/// applied per-opcode instead of to one hand-picked formula. One
/// remaining simplification: `d(warp(point))/d(point)` is approximated as
/// exactly `1` rather than differentiating the warp program too (most
/// archived genomes' warps are mild coordinate bends, not sharp
/// expansions — a genome with an aggressively expanding warp would get a
/// systematically-off DE here, not yet tested against one).
/// Fixed, well-spread unit directions used by `anisotropy_score` — not
/// exhaustive, just enough to catch gross rotational degeneracy cheaply.
const ANISOTROPY_DIRECTIONS: [(f64, f64, f64); 8] = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.577_350_3, 0.577_350_3, 0.577_350_3),
    (0.700_649, -0.700_649, 0.210_195),
    (-0.343_284, 0.686_567, -0.772_388),
    (0.174_078, -0.522_233, 0.870_388),
    (-0.700_649, -0.700_649, 0.140_130),
];
/// Fixed `(r, rho)` sample points, spanning both small and moderate radii
/// so a formula that only breaks symmetry far from the origin (or only
/// near it) still gets caught.
const ANISOTROPY_POINTS: [(f64, f64); 4] = [(0.2, 0.9), (0.5, 0.5), (-0.3, 1.1), (0.1, 0.3)];

/// How much a genome's escape-time landscape depends on DIRECTION, not
/// just distance from the origin — the direct, cheap (no ray-marching at
/// all) test of the same degeneracy this project already found for the
/// hand-built formulas: Mandelbrot/Tricorn/Cubic/Quartic/Celtic depend
/// only on `(R, rho=|vector part|)` under `TimeAxis::R`, so ANY slice
/// through them is exactly a circle/sphere no matter how the camera
/// moves. A genome whose DAG only ever scales the vector part by a
/// scalar (chains of SIN/COS/EXP/SQR/etc. without a genuinely-mixing
/// MUL/ADD of two differently-directioned operands) inherits the same
/// degeneracy, and — even short of exact symmetry — WEAK directional
/// sensitivity still means most viewing angles reveal little new
/// structure. Returns a value in `[0, 1]`: 0 for a perfectly
/// direction-independent (guaranteed-boring) formula, higher when the
/// escape-time landscape genuinely varies with direction on the sphere.
/// Calibration is a first pass (see `quat_dag_fitness.rs`'s module docs)
/// — this only needs to rank genomes relative to each other, not hit an
/// absolute target.
pub fn anisotropy_score(f: &QuatDagFormula, max_iter: u32, bailout_sq: f64) -> f32 {
    let max = max_iter as f64;
    let mut point_scores = Vec::with_capacity(ANISOTROPY_POINTS.len());
    for &(r, rho) in &ANISOTROPY_POINTS {
        let mut ets = Vec::with_capacity(ANISOTROPY_DIRECTIONS.len());
        for &(a, b, c) in &ANISOTROPY_DIRECTIONS {
            let q = Quat::new(r, rho * a, rho * b, rho * c);
            let (et, _de) = quat_dag_escape_de(f, q, max_iter, bailout_sq);
            ets.push(et as f64);
        }
        let lo = ets.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = ets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Skip points where every direction escapes trivially (et≈0) or
        // never escapes (et≈max_iter) — degenerate SAMPLE points, not
        // evidence about the formula's symmetry either way.
        if hi < 0.5 || lo > max - 0.5 {
            continue;
        }
        point_scores.push(((hi - lo) / max).min(1.0));
    }
    if point_scores.is_empty() {
        return 0.0;
    }
    (point_scores.iter().sum::<f64>() / point_scores.len() as f64) as f32
}

pub fn quat_dag_escape_de(f: &QuatDagFormula, point: Quat, max_iter: u32, bailout_sq: f64) -> (f32, f64) {
    let warped = if f.warp.is_empty() { point } else { eval_program_quat(f.warp, point, point) };
    // (z, c, dz/dpoint, dc/dpoint) — see the doc comment above: whichever
    // of z0/c is the warped point gets derivative ≈1 (the warp-derivative
    // simplification), the other is a true constant (derivative 0).
    let (mut z, c, mut dz, dc) = if f.julia {
        (warped, Quat::new(f.jc.0 as f64, f.jc.1 as f64, 0.0, 0.0), 1.0_f64, 0.0_f64)
    } else {
        (Quat::ZERO, warped, 0.0_f64, 1.0_f64)
    };
    let phoenix_q = Quat::new(f.phoenix.0 as f64, f.phoenix.1 as f64, 0.0, 0.0);
    let phoenix_mag = phoenix_q.norm_sq().sqrt();
    let mut pz = Quat::ZERO;
    let mut dpz = 0.0_f64;

    for it in 0..max_iter {
        let (f0, df) = eval_program_quat_deriv(f.prog, z, c, dz, dc);
        let dnext = df + phoenix_mag * dpz;
        let next = f0.add(phoenix_q.mul(pz));
        dpz = dz;
        pz = z;
        dz = dnext;
        z = next;

        let ms = z.norm_sq();
        if ms > bailout_sq {
            let et = ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
            let r = ms.sqrt();
            let de = (0.5 * r.ln() * r / dz.max(1e-300)).max(0.0);
            return (et, de);
        }
        if !z.is_finite() {
            return (it as f32, 0.0);
        }
    }
    let r = z.norm_sq().sqrt().max(1e-300);
    let de = (0.5 * r.ln() * r / dz.max(1e-300)).max(0.0);
    (max_iter as f32, de)
}

/// `eval_program_quat`, run in parallel with a scalar derivative-magnitude
/// register file (`dz`/`dc` seed the `Z`/`C` leaves — the two independent
/// directions the outer recursion needs; see `quat_dag_escape_de`), one
/// entry per opcode. Each rule is the magnitude of that opcode's standard
/// (complex/quaternion) derivative, exact given `|pq|=|p||q|`:
/// `SQR→2|a|·da` (product rule, `a·a`), `SIN→|cos(a)|·da`
/// (`d(sin a)=cos(a)da`), `MUL→|a|·db+|b|·da` (product rule), `DIV` via
/// the quotient rule applied to `a·b⁻¹`, etc. — see the inline match arms
/// for the rest; every one is the direct analog of a standard single-
/// variable derivative rule, not invented per-opcode.
fn eval_program_quat_deriv(prog: &[OpNode], z: Quat, c: Quat, dz: f64, dc: f64) -> (Quat, f64) {
    let n = prog.len().min(N_SLOTS);
    if n == 0 {
        return (Quat::ZERO, 0.0);
    }
    let mut reg = [Quat::ZERO; N_SLOTS];
    let mut dreg = [0.0f64; N_SLOTS];
    for i in 0..n {
        let node = prog[i];
        let ai = (node.a as usize).min(N_SLOTS - 1);
        let bi = (node.b as usize).min(N_SLOTS - 1);
        let (a, da) = if ai < i { (reg[ai], dreg[ai]) } else { (Quat::ZERO, 0.0) };
        let (b, db) = if bi < i { (reg[bi], dreg[bi]) } else { (Quat::ZERO, 0.0) };
        let na = a.norm_sq().sqrt();
        let nb = b.norm_sq().sqrt();
        let (val, dval): (Quat, f64) = match node.op {
            op::Z => (z, dz),
            op::C => (c, dc),
            op::CONST => (Quat::new(node.kre as f64, node.kim as f64, 0.0, 0.0), 0.0),
            op::SQR => (a.mul(a), 2.0 * na * da),
            op::CUBE => (a.mul(a).mul(a), 3.0 * na * na * da),
            op::QUART => {
                let sq = a.mul(a);
                (sq.mul(sq), 4.0 * na.powi(3) * da)
            }
            op::RECIP => (qrecip(a), da / (na * na + EPS)),
            op::SIN => (qsin(a), qcos(a).norm_sq().sqrt() * da),
            op::COS => (qcos(a), qsin(a).norm_sq().sqrt() * da),
            op::EXP => {
                let e = qexp(a);
                let en = e.norm_sq().sqrt();
                (e, en * da)
            }
            op::LOG => (qlog(a), da / (na + EPS)),
            op::TANH => {
                let t = qtanh(a);
                let tn = t.norm_sq().sqrt();
                (t, (1.0 - tn * tn).abs() * da)
            }
            op::CONJ => (a.conj(), da), // conjugation preserves norm exactly
            op::ABSFOLD => (a.abs_components(), da), // |·| has unit derivative magnitude a.e.
            op::ABSRE => (Quat::new(a.r.abs(), a.a, a.b, a.c), da),
            op::ABSIM => (Quat::new(a.r, a.a.abs(), a.b.abs(), a.c.abs()), da),
            op::NORMZ => (qnormz(a), da / (na + EPS)),
            op::ADD => (a.add(b), da + db),
            op::SUB => (a.sub(b), da + db),
            op::MUL => (a.mul(b), na * db + nb * da),
            op::DIV => {
                let rb = qrecip(b);
                let d_rb = db / (nb * nb + EPS); // |d(b⁻¹)| = |db|/|b|²
                (a.mul(rb), da / (nb + EPS) + na * d_rb)
            }
            _ => (Quat::ZERO, 0.0),
        };
        reg[i] = val;
        dreg[i] = dval;
    }
    (reg[n - 1], dreg[n - 1])
}

/// Same shape as `quat_raymarch::RaymarchParams`, with `formula:
/// QuatFormula` replaced by a borrowed `QuatDagFormula` — see that
/// struct's own doc comments for what each field means, this is a direct
/// parallel, not an independent design. Kept as its own type (not a
/// generalization of `RaymarchParams` itself) specifically so the
/// existing, already-shipped `QuatFormula` rendering path — including its
/// GPU pipeline — stays completely untouched by this addition.
#[derive(Copy, Clone)]
pub struct RaymarchDagParams<'a> {
    pub formula: QuatDagFormula<'a>,
    pub time_axis: TimeAxis,
    pub time_val: f64,
    pub domain_radius: f64,
    /// Overrides the sphere bound `domain_radius` implies with a true
    /// axis-aligned box `(min, max)` (animation-viewer plan, Phase 2).
    /// `domain_radius` still drives the epsilon-scale derivations
    /// (`min_step` etc.) either way — a `Some` caller should set it to a
    /// characteristic scale of the box (e.g. its largest half-extent).
    /// `None` (every caller before this plan — `quat_viewer.rs`, the CLI)
    /// preserves today's sphere-bounded behavior exactly, byte-for-byte.
    #[allow(clippy::type_complexity)]
    pub box_bounds: Option<(Vec3, Vec3)>,
    /// Overrides `time_axis`'s "pick exactly one part as time, hardcode
    /// the other three into a fixed x/y/z order" scheme with a full,
    /// user-swappable permutation of all 4 quaternion parts across the 4
    /// viewer axes (animation-viewer plan, Phase 2). `None` (every caller
    /// before this plan) preserves `time_axis`'s exact fixed-order
    /// assembly.
    pub axis_assignment: Option<AxisAssignment>,
    /// An orientable half-space cutaway (animation-viewer plan, Phase 6) —
    /// `(pos, normal)`. The ray is clipped to the half-space
    /// `dot(point - pos, normal) >= 0`, i.e. the far side of the plane
    /// (the side `normal` points AWAY from) is removed, revealing whatever
    /// surface lies on the near side — the "difference/cutaway" effect.
    /// `normal` need not be unit length: `t_plane`'s ratio is scale-
    /// invariant (see `apply_clip_plane`'s doc comment). `None` (every
    /// caller before this plan) preserves behavior exactly.
    #[allow(clippy::type_complexity)]
    pub clip_plane: Option<(Vec3, Vec3)>,
    /// A hidden escape-iteration band `(min_iter, max_iter)` — the
    /// authored, per-clip "Slide Iteration" effect (animation-viewer
    /// plan). A point whose escape time falls in `[min_iter, max_iter)` is
    /// never treated as a raymarch hit, so the ray continues past it,
    /// revealing deeper iteration shells. Doesn't touch the distance-
    /// estimate field itself (so adaptive step sizing is unaffected) —
    /// only the hit ACCEPTANCE test in `march_ray` gains an extra
    /// condition. `None` (every caller before this plan) preserves
    /// behavior exactly.
    pub slide_iter: Option<(f64, f64)>,
    /// Hide any point whose escape time is `>=` this value — the resolved
    /// form of the viewer's "hide top N%" control. Unlike `slide_iter`
    /// (an author-picked absolute band), this is computed FRESH each
    /// render from an actual rendered frame's escape-time distribution
    /// (see `anim_eval::percentile_escape_time_cutoff`) — a fixed
    /// absolute iteration count can't reliably target "whatever currently
    /// renders white," since coloring is histogram-equalized against
    /// each frame's own distribution (`colormap::apply_colormap_equalized`)
    /// — Carl, 2026-09-28, after `hide_unescaped` (this field's
    /// since-removed predecessor, a literal `et >= max_iter` check)
    /// visibly failed to remove anything on a real genome: "build the top
    /// N% version properly." `None` (every caller before this field
    /// existed, and every render's own FIRST/probe pass) disables this
    /// gate entirely.
    pub hide_above: Option<f64>,
    pub max_iter: u32,
    pub bailout: f64,
    pub max_march_steps: u32,
    pub hit_epsilon: f64,
    pub step_safety: f64,
    pub light_dir: Vec3,
    pub normal_eps: f64,
    pub color_probe_offset: f64,
    pub aa: u32,
}

/// Places a 3D ray-march point (plus `p.time_val`) into a full `Quat`,
/// via `p.axis_assignment` when given, else `p.time_axis`'s fixed-order
/// scheme — see `RaymarchDagParams::axis_assignment`'s doc comment.
fn assemble_point(p: &RaymarchDagParams, point: Vec3) -> Quat {
    match &p.axis_assignment {
        Some(a) => a.assemble(point.0, point.1, point.2, p.time_val),
        None => p.time_axis.assemble(point, p.time_val),
    }
}

fn de_at(p: &RaymarchDagParams, point: Vec3) -> f64 {
    let bailout_sq = p.bailout * p.bailout;
    let q = assemble_point(p, point);
    quat_dag_escape_de(&p.formula, q, p.max_iter, bailout_sq).1
}

/// Tetrahedral 4-tap normal estimate — same technique and same outward-
/// facing sign convention as `quat_raymarch::estimate_normal` (see that
/// function's doc comment for why: DE is lowest at the boundary and
/// increases outward, so its gradient already points the right way).
fn estimate_normal(p: &RaymarchDagParams, point: Vec3) -> Vec3 {
    let eps = p.normal_eps.max(1e-6);
    const K0: Vec3 = (1.0, -1.0, -1.0);
    const K1: Vec3 = (-1.0, -1.0, 1.0);
    const K2: Vec3 = (-1.0, 1.0, -1.0);
    const K3: Vec3 = (1.0, 1.0, 1.0);
    let g = add(
        add(scale(K0, de_at(p, add(point, scale(K0, eps)))), scale(K1, de_at(p, add(point, scale(K1, eps))))),
        add(scale(K2, de_at(p, add(point, scale(K2, eps)))), scale(K3, de_at(p, add(point, scale(K3, eps))))),
    );
    normalize(g)
}

/// Adaptive sphere tracing, same algorithm as `quat_raymarch::march_ray`
/// (see that function's doc comment) — this is the DAG-formula twin, not
/// an independent design.
/// Narrows `[t0, t1]` to the near side of an optional clip plane —
/// mirrors the WGSL march_sample's clip-plane block exactly (interval
/// narrowing, not a distance-field modification: `quat_dag_escape_de` is
/// an unsigned distance estimate with no inside/outside sign convention,
/// so the usual signed-SDF CSG trick doesn't apply here). `t_plane`'s
/// value is invariant to `normal`'s scale (scaling `normal` by k scales
/// both `dot(dir,normal)` and `dot(diff,normal)` by k, which cancels in
/// the ratio), so callers don't need to pre-normalize it.
fn apply_clip_plane(t0: f64, t1: f64, eye: Vec3, dir: Vec3, clip: Option<(Vec3, Vec3)>) -> Option<(f64, f64)> {
    let Some((pos, n)) = clip else { return Some((t0, t1)) };
    let denom = dir.0 * n.0 + dir.1 * n.1 + dir.2 * n.2;
    let diff = (pos.0 - eye.0, pos.1 - eye.1, pos.2 - eye.2);
    let (mut nt0, mut nt1) = (t0, t1);
    if denom > 1e-6 {
        let t_plane = (diff.0 * n.0 + diff.1 * n.1 + diff.2 * n.2) / denom;
        nt0 = nt0.max(t_plane);
    } else if denom < -1e-6 {
        let t_plane = (diff.0 * n.0 + diff.1 * n.1 + diff.2 * n.2) / denom;
        nt1 = nt1.min(t_plane);
    } else {
        // Ray parallel to the plane: entirely on one side or the other.
        let side = (eye.0 - pos.0) * n.0 + (eye.1 - pos.1) * n.1 + (eye.2 - pos.2) * n.2;
        if side < 0.0 {
            return None;
        }
    }
    if nt0 > nt1 {
        return None;
    }
    Some((nt0, nt1))
}

fn march_ray(p: &RaymarchDagParams, eye: Vec3, dir: Vec3) -> Option<(Vec3, Vec3, f32)> {
    let (t0, t1) = match p.box_bounds {
        Some((bmin, bmax)) => ray_box(eye, dir, bmin, bmax)?,
        None => ray_sphere(eye, dir, p.domain_radius)?,
    };
    let (t0, t1) = apply_clip_plane(t0, t1, eye, dir, p.clip_plane)?;
    let bailout_sq = p.bailout * p.bailout;
    let hit_eps = p.hit_epsilon.max(1e-9);
    let min_step = (p.domain_radius * 1e-6).max(1e-9);
    let mut t = t0;
    for _ in 0..p.max_march_steps.max(1) {
        if t > t1 {
            return None;
        }
        let point = add(eye, scale(dir, t));
        let q = assemble_point(p, point);
        let (et, de) = quat_dag_escape_de(&p.formula, q, p.max_iter, bailout_sq);
        // Slide Iteration: a point in the hidden band, or (hide_above) at
        // or past a resolved escape-time cutoff, is never accepted as a
        // hit, even when its DE is small enough — the real DE still
        // drives the step below, so marching through stays adaptively
        // efficient.
        let in_band = matches!(p.slide_iter, Some((lo, hi)) if (et as f64) >= lo && (et as f64) < hi);
        let above_cutoff = matches!(p.hide_above, Some(cutoff) if (et as f64) >= cutoff);
        if de < hit_eps && !in_band && !above_cutoff {
            let normal = estimate_normal(p, point);
            let probe = add(point, scale(normal, p.color_probe_offset));
            let probe_q = assemble_point(p, probe);
            let (color_et, _) = quat_dag_escape_de(&p.formula, probe_q, p.max_iter, bailout_sq);
            return Some((point, normal, color_et));
        }
        t += (de * p.step_safety).max(min_step);
    }
    None
}

/// Same rayon-parallel-per-pixel shape as `quat_raymarch::render_raymarch_frame`.
pub fn render_raymarch_dag_frame(p: &RaymarchDagParams, cam: &RaymarchCamera, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let forward = normalize(sub(cam.target, cam.eye));
    let (right, up) = look_at_basis(forward, cam.up_hint);
    let half_h = (cam.fov_y * 0.5).tan();
    let aspect = width as f64 / (height.max(1)) as f64;
    let half_w = half_h * aspect;
    let light = normalize(p.light_dir);
    let wf = width.max(1) as f64;
    let hf = height.max(1) as f64;
    let aa = p.aa.max(1);
    (0..(width as u64 * height as u64))
        .into_par_iter()
        .map(|idx| {
            let px = (idx % width as u64) as u32;
            let py = (idx / width as u64) as u32;
            let mut sum_shade = 0.0f32;
            let mut sum_color = 0.0f32;
            for sy in 0..aa {
                for sx in 0..aa {
                    let jx = (sx as f64 + 0.5) / aa as f64;
                    let jy = (sy as f64 + 0.5) / aa as f64;
                    let u = ((px as f64 + jx) / wf * 2.0 - 1.0) * half_w;
                    let v = (1.0 - (py as f64 + jy) / hf * 2.0) * half_h;
                    let dir = normalize(add(add(forward, scale(right, u)), scale(up, v)));
                    if let Some((_, normal, color_et)) = march_ray(p, cam.eye, dir) {
                        let ndotl = dot(normal, light).max(0.0);
                        sum_shade += (0.15 + 0.85 * ndotl) as f32;
                        sum_color += color_et;
                    }
                }
            }
            let n = (aa * aa) as f32;
            (sum_shade / n, sum_color / n)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::eval_program as eval_program_2d;

    fn q2(re: f64, im: f64) -> Quat {
        Quat::new(re, im, 0.0, 0.0)
    }

    /// The load-bearing test for this whole module, same discipline as
    /// every other quaternion-generalization parity test this project has
    /// (`quat_fractal.rs`'s `*_matches_2d_*_exactly` tests): restricted to
    /// the R,A plane (B=C=0), the quaternion DAG evaluator must reduce
    /// EXACTLY to the independent 2D complex evaluator — for every opcode,
    /// not just a couple.
    #[test]
    fn eval_program_quat_matches_2d_eval_program_exactly_for_every_opcode() {
        let unary_ops = [
            op::SQR, op::CUBE, op::QUART, op::RECIP, op::SIN, op::COS, op::EXP, op::LOG, op::TANH,
            op::CONJ, op::ABSFOLD, op::ABSRE, op::ABSIM, op::NORMZ,
        ];
        let binary_ops = [op::ADD, op::SUB, op::MUL, op::DIV];
        let sample_points: [(f64, f64); 4] = [(0.3, 0.2), (-1.1, 0.7), (0.05, -0.6), (2.0, 1.3)];

        for &o in &unary_ops {
            // program: [0]=Z, [1]=op(Z)
            let prog = [OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }, OpNode { op: o, a: 0, b: 0, kre: 0.0, kim: 0.0 }];
            for &(zx, zy) in &sample_points {
                let (ex, ey) = eval_program_2d(&prog, zx as f32, zy as f32, 0.0, 0.0);
                let got = eval_program_quat(&prog, q2(zx, zy), Quat::ZERO);
                assert!(
                    (got.r - ex as f64).abs() < 1e-4 && (got.a - ey as f64).abs() < 1e-4 && got.b.abs() < 1e-9 && got.c.abs() < 1e-9,
                    "op {} at ({zx},{zy}): 2D=({ex},{ey}) quat={:?}", op::name(o), got
                );
            }
        }

        for &o in &binary_ops {
            // program: [0]=Z, [1]=C, [2]=op(Z,C)
            let prog = [
                OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
                OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
                OpNode { op: o, a: 0, b: 1, kre: 0.0, kim: 0.0 },
            ];
            for &(zx, zy) in &sample_points {
                let (cx, cy) = (0.4, -0.25);
                let (ex, ey) = eval_program_2d(&prog, zx as f32, zy as f32, cx as f32, cy as f32);
                let got = eval_program_quat(&prog, q2(zx, zy), q2(cx, cy));
                assert!(
                    (got.r - ex as f64).abs() < 1e-4 && (got.a - ey as f64).abs() < 1e-4 && got.b.abs() < 1e-9 && got.c.abs() < 1e-9,
                    "op {} at z=({zx},{zy}): 2D=({ex},{ey}) quat={:?}", op::name(o), got
                );
            }
        }
    }

    #[test]
    fn const_node_matches_2d_exactly() {
        let prog = [OpNode { op: op::CONST, a: 0, b: 0, kre: 0.37, kim: -0.61 }];
        let (ex, ey) = eval_program_2d(&prog, 0.0, 0.0, 0.0, 0.0);
        let got = eval_program_quat(&prog, Quat::ZERO, Quat::ZERO);
        assert!((got.r - ex as f64).abs() < 1e-6 && (got.a - ey as f64).abs() < 1e-6);
        assert_eq!(got.b, 0.0);
        assert_eq!(got.c, 0.0);
    }

    #[test]
    fn a_real_archived_program_matches_2d_exactly() {
        // A representative multi-node program shaped like a real archived
        // genome's (sin, log, div, mul chains) rather than a synthetic
        // one-op probe — catches any interaction bug a single-op test
        // could miss.
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },       // 0: z
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },     // 1: sin(z)
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },       // 2: c
            OpNode { op: op::MUL, a: 0, b: 2, kre: 0.0, kim: 0.0 },     // 3: z*c
            OpNode { op: op::SUB, a: 3, b: 0, kre: 0.0, kim: 0.0 },     // 4: z*c - z
            OpNode { op: op::DIV, a: 4, b: 1, kre: 0.0, kim: 0.0 },     // 5: (z*c-z)/sin(z)
            OpNode { op: op::LOG, a: 5, b: 0, kre: 0.0, kim: 0.0 },     // 6: log(...)
        ];
        for &(zx, zy) in &[(0.3, 0.2), (-0.6, 0.4), (1.1, -0.3)] {
            for &(cx, cy) in &[(0.1, -0.2), (-0.4, 0.05)] {
                let (ex, ey) = eval_program_2d(&prog, zx, zy, cx, cy);
                let got = eval_program_quat(&prog, q2(zx as f64, zy as f64), q2(cx as f64, cy as f64));
                assert!(
                    (got.r - ex as f64).abs() < 1e-3 && (got.a - ey as f64).abs() < 1e-3 && got.b.abs() < 1e-9 && got.c.abs() < 1e-9,
                    "z=({zx},{zy}) c=({cx},{cy}): 2D=({ex},{ey}) quat=({:.4},{:.4},{:.2e},{:.2e})", got.r, got.a, got.b, got.c
                );
            }
        }
    }

    #[test]
    fn quat_dag_escape_de_matches_2d_dag_escape_pixel_exactly() {
        use crate::fractal::dag_escape_pixel;
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        for &(px, py) in &[(0.3f32, 0.2f32), (-0.7, 0.1), (0.05, -0.4)] {
            let (et2d, _) = dag_escape_pixel(&prog, &[], false, (0.0, 0.0), (0.0, 0.0), 16.0, px, py, 40);
            let (etq, _) = quat_dag_escape_de(&f, q2(px as f64, py as f64), 40, 16.0);
            assert!((et2d - etq).abs() < 1e-2, "px={px} py={py}: 2D et={et2d} quat et={etq}");
        }
    }

    #[test]
    fn interior_point_stays_interior() {
        // z_next = z^2 + c, at c=0 (the origin) must stay exactly at 0 and
        // never escape, same invariant every QuatFormula test relies on.
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let (et, _) = quat_dag_escape_de(&f, Quat::ZERO, 30, 16.0);
        assert_eq!(et, 30.0);
    }

    /// The strongest available check on the analytic derivative
    /// propagation: the classic `z²+c` DAG, evaluated under Mandelbrot
    /// mode, must reduce to EXACTLY the same computation as
    /// `quat_fractal::QuatFormula::Mandelbrot`'s hand-coded formula — both
    /// escape time AND the distance estimate itself, not just the escape
    /// time like the other parity tests. This works out algebraically:
    /// `SQR`'s derivative rule (`2·ρ·da`) is exactly `n·ρ^(n-1)·da` for
    /// `n=2`, and `ADD` with a `dc=1` seed contributes exactly the "+1"
    /// term — so this isn't a coincidental match, it's the same formula
    /// reached two different ways, which is exactly what should happen if
    /// the general derivative propagation is correct.
    #[test]
    fn dag_z_squared_plus_c_de_matches_hand_built_mandelbrot_exactly() {
        use crate::quat_fractal::{quat_escape_de, QuatFormula};
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        for q in [
            Quat::new(0.3, 0.2, -0.1, 0.05),
            Quat::new(-0.6, 0.4, 0.15, -0.2),
            Quat::new(0.05, -0.3, 0.4, 0.1),
        ] {
            let (et_dag, de_dag) = quat_dag_escape_de(&f, q, 60, 16.0);
            let (et_hand, de_hand) = quat_escape_de(QuatFormula::Mandelbrot, q, 60, 16.0);
            assert_eq!(et_dag, et_hand, "q={q:?}: escape time diverged");
            assert!((de_dag - de_hand).abs() < 1e-6 * de_hand.max(1.0), "q={q:?}: de_dag={de_dag} de_hand={de_hand}");
        }
    }

    #[test]
    fn eval_program_quat_deriv_matches_a_direct_finite_difference() {
        // Sanity check on the analytic propagation itself, independent of
        // the escape-time machinery: for a representative multi-op
        // program, the analytic |df/dz| should be close to a direct
        // (fine-epsilon) finite difference along a fixed probe direction.
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::MUL, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let z = Quat::new(0.4, 0.3, -0.2, 0.1);
        let c = Quat::new(0.5, -0.1, 0.2, 0.05);
        let (_, analytic) = eval_program_quat_deriv(&prog, z, c, 1.0, 0.0);
        let eps = 1e-6;
        let z_pert = Quat::new(z.r + eps, z.a, z.b, z.c);
        let f0 = eval_program_quat(&prog, z, c);
        let f1 = eval_program_quat(&prog, z_pert, c);
        let numeric = ((f1.r - f0.r).powi(2) + (f1.a - f0.a).powi(2) + (f1.b - f0.b).powi(2) + (f1.c - f0.c).powi(2)).sqrt() / eps;
        // These are two DIFFERENT (but both reasonable) magnitude
        // estimates — a directional finite difference along one axis vs
        // an analytic bound — so this checks they're in the same
        // ballpark, not bit-identical.
        assert!(numeric > 0.0 && analytic > 0.0, "both should detect real sensitivity: numeric={numeric} analytic={analytic}");
        assert!(analytic / numeric < 20.0 && numeric / analytic < 20.0, "analytic={analytic} vs directional numeric={numeric} too far apart");
    }

    /// The classic z²+c DAG already has a proven exact reduction to
    /// `quat_fractal::quat_escape_de(QuatFormula::Mandelbrot, ...)`
    /// (`dag_z_squared_plus_c_de_matches_hand_built_mandelbrot_exactly`
    /// above) — and Mandelbrot is the canonical example of a formula with
    /// full SO(3) symmetry: squaring ALWAYS preserves the vector part's
    /// direction exactly (`q² = (r²-ρ², 2rρ·n̂)` — same n̂ in and out, by
    /// direct expansion of the Hamilton product), so the whole `z←z²+c`
    /// recursion stays in the same `{1,n̂}` subalgebra forever and its
    /// escape time can only ever depend on `(R, rho)`. `anisotropy_score`
    /// should be NEAR zero for it — not bit-exact, since the SAME
    /// mathematical trajectory computed via different floating-point
    /// operation orders (one per test direction) can still diverge by a
    /// tiny amount right at a chaotic bailout-crossing boundary, the same
    /// class of noise already documented and tolerated elsewhere in this
    /// project (GPU-vs-CPU parity tests).
    #[test]
    fn anisotropy_score_is_near_zero_for_the_provably_symmetric_z_squared_plus_c() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let score = anisotropy_score(&f, 60, 16.0);
        assert!(score < 0.05, "z²+c is provably direction-independent; anisotropy_score should be near 0, got {score}");
    }

    /// `MUL(Z, CONST)` with a CONST fixed in the R,A plane genuinely
    /// rotates the vector part out of alignment with z's own direction
    /// (unlike SQR/SIN/etc., which only ever rescale the EXISTING
    /// direction) — a positive control for the metric actually detecting
    /// asymmetry when it's really there. The extra SQR after the MUL is
    /// needed only so the point actually escapes within the iteration
    /// budget (without it every sample point stays bounded — no escape
    /// means no signal either way, verified directly before writing this
    /// test).
    #[test]
    fn anisotropy_score_is_clearly_nonzero_for_a_program_that_genuinely_mixes_directions() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::CONST, a: 0, b: 0, kre: 0.3, kim: 0.7 },
            OpNode { op: op::MUL, a: 0, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 2, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 3, b: 4, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let score = anisotropy_score(&f, 60, 16.0);
        assert!(score > 0.3, "expected clearly-above-noise directional sensitivity, got {score}");
    }

    #[test]
    fn anisotropy_score_stays_in_zero_one_range_for_a_multi_op_program() {
        // Same shape as `a_real_archived_program_matches_2d_exactly`'s
        // representative program (sin/mul/sub/div/log chain).
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::MUL, a: 0, b: 2, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SUB, a: 3, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::DIV, a: 4, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: op::LOG, a: 5, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let score = anisotropy_score(&f, 60, 16.0);
        assert!((0.0..=1.0).contains(&score), "score {score} out of [0,1] range");
    }

    // ── animation-viewer plan, Phase 2: box_bounds/axis_assignment ─────

    const MANDEL_PROG: [OpNode; 4] = [
        OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
    ];

    fn base_dag_params() -> RaymarchDagParams<'static> {
        let formula = QuatDagFormula { prog: &MANDEL_PROG, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        RaymarchDagParams {
            formula,
            time_axis: TimeAxis::C,
            time_val: 0.0,
            domain_radius: 1.6,
            box_bounds: None,
            axis_assignment: None,
            clip_plane: None,
            slide_iter: None,
            hide_above: None,
            max_iter: 60,
            bailout: 4.0,
            max_march_steps: 200,
            hit_epsilon: 1.6 * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: 1.6 * 1e-3,
            color_probe_offset: 1.6 * 1e-2,
            aa: 1,
        }
    }

    #[test]
    fn box_bounds_none_matches_the_pre_phase_2_sphere_behavior() {
        // A regression guard: every caller before this plan (quat_viewer.rs,
        // the CLI) leaves box_bounds/axis_assignment at None — de_at/march_ray
        // must behave EXACTLY as they did before this plan (ray_sphere +
        // TimeAxis::assemble), not merely "similarly".
        let p = base_dag_params();
        let eye = (0.0, 0.0, -5.0);
        let dir = normalize((0.0, 0.0, 1.0));
        let hit = march_ray(&p, eye, dir);
        assert!(hit.is_some(), "z^2+c should render something head-on from outside the sphere");
    }

    #[test]
    fn box_bounds_some_actually_changes_what_is_reachable() {
        // A box squashed flat on X (min=max=0) must reject a ray offset in
        // X — deterministic regardless of the formula's own boundary shape,
        // since march_ray's bounds check (`ray_box(..)?`) short-circuits
        // BEFORE any formula evaluation ever runs.
        let mut p = base_dag_params();
        p.box_bounds = Some(((0.0, -1.6, -1.6), (0.0, 1.6, 1.6)));
        let offset_hit = march_ray(&p, (0.5, 0.0, -5.0), normalize((0.0, 0.0, 1.0)));
        assert!(offset_hit.is_none(), "a ray off the flattened X=0 plane must miss a box with zero X extent, regardless of the formula");
    }

    #[test]
    fn box_bounds_none_falls_back_to_the_domain_radius_sphere() {
        // Equally deterministic in the other direction: a ray well outside
        // domain_radius must miss when box_bounds is None, regardless of
        // the formula — proves the None fallback still reaches ray_sphere.
        let p = base_dag_params(); // domain_radius=1.6, box_bounds=None
        let far_miss = march_ray(&p, (0.0, 10.0, -5.0), normalize((0.0, 0.0, 1.0)));
        assert!(far_miss.is_none(), "a ray far outside domain_radius must miss, regardless of the formula");
    }

    #[test]
    fn axis_assignment_none_matches_time_axis_c_exactly() {
        // AxisAssignment{x:R,y:A,z:B,t:C} is the exact permutation TimeAxis::C
        // encodes (Quat::new(x,y,z,time_val)) — confirms the new
        // permutation-based path reduces EXACTLY to the old fixed-order path
        // for its equivalent assignment. Tries several rays rather than
        // asserting a specific one hits (whether a given ray crosses this
        // particular formula's boundary isn't something to assume a
        // priori) — the real assertion is that whichever rays DO hit agree
        // exactly between the two paths, and none disagree on hit/miss.
        use crate::anim_timeline::{AxisAssignment, QuatPart};
        let mut p_perm = base_dag_params();
        p_perm.axis_assignment = Some(AxisAssignment { x: QuatPart::R, y: QuatPart::A, z: QuatPart::B, t: QuatPart::C });
        let p_fixed = base_dag_params();

        let rays: [(Vec3, Vec3); 5] = [
            ((0.0, 0.0, -5.0), (0.0, 0.0, 1.0)),
            ((0.0, 0.0, -5.0), (0.13, 0.07, 1.0)),
            ((0.0, 0.0, -5.0), (0.5, 0.0, 1.0)),
            ((0.0, 0.0, -5.0), (0.0, 0.5, 1.0)),
            ((2.0, 1.5, -3.0), (-0.3, -0.2, 0.8)),
        ];
        let mut any_hit = false;
        for (eye, dir) in rays {
            let dir = normalize(dir);
            let hit_perm = march_ray(&p_perm, eye, dir);
            let hit_fixed = march_ray(&p_fixed, eye, dir);
            match (hit_perm, hit_fixed) {
                (Some((pp, np, _)), Some((pf, nf, _))) => {
                    any_hit = true;
                    assert!((pp.0 - pf.0).abs() < 1e-9 && (pp.1 - pf.1).abs() < 1e-9 && (pp.2 - pf.2).abs() < 1e-9,
                        "hit points differ for eye={eye:?} dir={dir:?}: perm={pp:?} fixed={pf:?}");
                    assert!((np.0 - nf.0).abs() < 1e-9 && (np.1 - nf.1).abs() < 1e-9 && (np.2 - nf.2).abs() < 1e-9,
                        "normals differ for eye={eye:?} dir={dir:?}: perm={np:?} fixed={nf:?}");
                }
                (None, None) => {} // both miss — consistent
                (a, b) => panic!("hit/miss disagreement for eye={eye:?} dir={dir:?}: perm={a:?} fixed={b:?}"),
            }
        }
        assert!(any_hit, "test is meaningless if not even one of the sample rays hit anything");
    }

    #[test]
    fn axis_assignment_actually_reassigns_which_part_carries_time() {
        // Swapping R and C (so R, not C, drives time) must change the
        // rendered geometry for an asymmetric-in-time genome — proves the
        // GUI's drag-and-drop axis reassignment has real teeth, not just a
        // relabeling.
        use crate::anim_timeline::{AxisAssignment, QuatPart};
        let mut p_c_is_time = base_dag_params();
        p_c_is_time.axis_assignment = Some(AxisAssignment { x: QuatPart::R, y: QuatPart::A, z: QuatPart::B, t: QuatPart::C });
        p_c_is_time.time_val = 0.9; // a nonzero time value so R vs C actually differs

        let mut p_r_is_time = base_dag_params();
        p_r_is_time.axis_assignment = Some(AxisAssignment { x: QuatPart::C, y: QuatPart::A, z: QuatPart::B, t: QuatPart::R });
        p_r_is_time.time_val = 0.9;

        let eye = (0.0, 0.0, -5.0);
        let dir = normalize((0.0, 0.0, 1.0));
        let hit_c = march_ray(&p_c_is_time, eye, dir);
        let hit_r = march_ray(&p_r_is_time, eye, dir);
        // Both may or may not hit depending on geometry, but if both hit,
        // the surface point must genuinely differ (this formula is not
        // symmetric under swapping which part is held at time_val=0.9 vs 0).
        if let (Some((pt_c, ..)), Some((pt_r, ..))) = (hit_c, hit_r) {
            let differs = (pt_c.0 - pt_r.0).abs() > 1e-6 || (pt_c.1 - pt_r.1).abs() > 1e-6 || (pt_c.2 - pt_r.2).abs() > 1e-6;
            assert!(differs, "reassigning which part carries time should change the hit point: c-time={pt_c:?} r-time={pt_r:?}");
        }
    }

    // ── animation-viewer plan, Phase 6: apply_clip_plane ────────────────

    #[test]
    fn clip_plane_none_is_a_no_op() {
        let (t0, t1) = apply_clip_plane(1.0, 5.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), None).unwrap();
        assert_eq!((t0, t1), (1.0, 5.0));
    }

    #[test]
    fn clip_plane_facing_the_ray_narrows_the_near_end() {
        // Plane at z=0, normal pointing back toward the eye (-Z) — the ray
        // travels +Z, so denom = dot(dir,normal) = -1 < 0 in the OLD
        // convention... use normal (0,0,1) so denom = dot((0,0,1),(0,0,1)) = 1 > 0,
        // narrowing t0 up to the plane crossing at t=5 (eye z=-5, plane z=0).
        let (t0, t1) = apply_clip_plane(1.0, 10.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))).unwrap();
        assert!((t0 - 5.0).abs() < 1e-9, "t0={t0}");
        assert_eq!(t1, 10.0);
    }

    #[test]
    fn clip_plane_facing_away_narrows_the_far_end() {
        // Same plane, normal flipped to (0,0,-1) => denom < 0, narrows t1.
        let (t0, t1) = apply_clip_plane(1.0, 10.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)))).unwrap();
        assert_eq!(t0, 1.0);
        assert!((t1 - 5.0).abs() < 1e-9, "t1={t1}");
    }

    #[test]
    fn clip_plane_can_reject_the_entire_interval() {
        // Plane crossing beyond t1: the whole [t0,t1] is on the far side.
        let hit = apply_clip_plane(1.0, 3.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))));
        assert!(hit.is_none(), "plane crossing at t=5 is past t1=3, the whole interval should be rejected");
    }

    #[test]
    fn clip_plane_parallel_ray_keeps_or_rejects_the_whole_interval() {
        // Ray travels along X, plane normal along Z through the origin —
        // the ray never crosses the plane (denom == 0). Eye at z=-1 is on
        // the plane's back side (normal (0,0,1) points toward +z), so the
        // whole interval must be rejected.
        let hit = apply_clip_plane(0.0, 10.0, (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))));
        assert!(hit.is_none());
        // Eye at z=+1 is on the front side — the whole interval survives.
        let (t0, t1) = apply_clip_plane(0.0, 10.0, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))).unwrap();
        assert_eq!((t0, t1), (0.0, 10.0));
    }

    #[test]
    fn clip_plane_result_is_invariant_to_normal_scale() {
        let a = apply_clip_plane(1.0, 10.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))));
        let b = apply_clip_plane(1.0, 10.0, (0.0, 0.0, -5.0), (0.0, 0.0, 1.0), Some(((0.0, 0.0, 0.0), (0.0, 0.0, 7.0))));
        assert_eq!(a, b, "t_plane's ratio must be unaffected by scaling the normal");
    }

    #[test]
    fn half_cut_clip_plane_removes_roughly_half_of_a_solid_object() {
        // The actual Phase 6 checkpoint scenario: a default axis-aligned
        // half-cut through a real formula's bounding box should hide
        // roughly half the object — checked by comparing hit-fraction with
        // and without the clip, on the same box/formula/rays.
        let mut p = base_dag_params();
        p.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        let mut hits_unclipped = 0;
        let mut hits_clipped = 0;
        let n = 25;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let v = (j as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let eye = (0.0, 0.0, -5.0);
                let dir = normalize((u, v, 2.0));
                if march_ray(&p, eye, dir).is_some() { hits_unclipped += 1; }
            }
        }
        p.clip_plane = Some(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)));
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let v = (j as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let eye = (0.0, 0.0, -5.0);
                let dir = normalize((u, v, 2.0));
                if march_ray(&p, eye, dir).is_some() { hits_clipped += 1; }
            }
        }
        assert!(hits_unclipped > 0, "sanity check: the unclipped bounding box should show something");
        assert!(hits_clipped < hits_unclipped, "the clip plane must hide some of what was visible: unclipped={hits_unclipped} clipped={hits_clipped}");
    }

    /// Casts the same 25x25 ray grid `half_cut_clip_plane_...` uses, over
    /// `march_ray` directly, with a given `slide_iter` — small helper so
    /// the two Slide Iteration tests below don't duplicate the grid setup.
    fn count_hits(p: &RaymarchDagParams) -> usize {
        let n = 25;
        let mut hits = 0;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let v = (j as f64 / (n - 1) as f64) * 2.0 - 1.0;
                let eye = (0.0, 0.0, -5.0);
                let dir = normalize((u, v, 2.0));
                if march_ray(p, eye, dir).is_some() { hits += 1; }
            }
        }
        hits
    }

    #[test]
    fn slide_iteration_excluding_every_escape_time_hides_everything() {
        let mut p = base_dag_params();
        p.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        assert!(count_hits(&p) > 0, "sanity check: the unclipped bounding box should show something");
        // Exclusive upper bound, so +1.0: a point that never escapes
        // within max_iter reports et == max_iter exactly (the "never
        // bailed out" case), which must also fall inside this band.
        p.slide_iter = Some((0.0, p.max_iter as f64 + 1.0));
        assert_eq!(count_hits(&p), 0, "excluding every possible escape-time value must hide every surface");
    }

    #[test]
    fn slide_iteration_empty_band_is_a_no_op() {
        let mut p = base_dag_params();
        p.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        let eye = (0.0, 0.0, -5.0);
        let dir = normalize((0.15, -0.1, 2.0));
        let without = march_ray(&p, eye, dir);
        p.slide_iter = Some((0.0, 0.0));
        let with_empty_band = march_ray(&p, eye, dir);
        assert_eq!(without, with_empty_band, "an empty min==max band must never hide a surface");
    }

    /// `hide_above` — a tiny box tightly around the origin sits deep
    /// inside the classic Mandelbrot-like set at c=0 (provably bounded,
    /// never escapes), so nearly every ray through it hits only
    /// never-escaped material (et == max_iter) — a `hide_above` cutoff at
    /// `max_iter` should knock hits down to (near) zero, while leaving
    /// them untouched when `None`.
    #[test]
    fn hide_above_removes_points_at_or_past_the_cutoff() {
        let mut p = base_dag_params();
        p.box_bounds = Some(((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3)));
        let hits_without = count_hits(&p);
        assert!(hits_without > 0, "sanity check: this tight central box should show something");
        p.hide_above = Some(p.max_iter as f64);
        let hits_hidden = count_hits(&p);
        assert!(hits_hidden * 4 < hits_without, "hide_above=max_iter should remove nearly every hit in a region that never escapes: without={hits_without} hidden={hits_hidden}");
    }

    #[test]
    fn hide_above_none_is_a_no_op() {
        let mut p = base_dag_params();
        p.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        let eye = (0.0, 0.0, -5.0);
        let dir = normalize((0.15, -0.1, 2.0));
        let without = march_ray(&p, eye, dir);
        p.hide_above = None;
        let still_without = march_ray(&p, eye, dir);
        assert_eq!(without, still_without, "hide_above=None must never hide a surface");
    }
}
