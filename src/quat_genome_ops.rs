//! Quaternion-only genetic-programming operator variants — a real
//! subtree crossover and a tunable-strength mutation, used ONLY by
//! `quat-dag-evolve` (`explorer.rs`, via `--crossover-mode`/
//! `--mutation-strength`). Deliberately NOT modifying `genome.rs`'s
//! `crossover_program`/`mutate_program` in place: those are shared with
//! the 2D formula-DAG GA (`Genome::mutate`'s DAG branch calls the exact
//! same functions), which is already tuned and working — Carl's own
//! call, confirmed directly rather than assumed, was to keep this fix
//! isolated to quaternion evolution for now rather than risk the 2D
//! system on an unproven change. Adopting these for 2D later, if the
//! single-metric experiment runs (see explorer.rs's `--fitness-metric`)
//! show they're actually better, is a separate, deliberate decision.
//!
//! # What was actually wrong with the existing operators
//!
//! Read in full (not guessed) before writing anything here:
//! `crossover_program` (`genome.rs`) isn't subtree crossover — it grafts
//! parent A's ENTIRE program plus parent B's ENTIRE program under one
//! new random combiner root. No crossover point is ever chosen inside
//! either parent, so for any fixed pair of parents there are only 4
//! possible children (the choice of ADD/SUB/MUL/DIV at the top), and a
//! successful crossover always produces a child of size
//! `len(a)+len(b)+1` — never smaller. When that exceeds the node cap
//! (common: `random_program` seeds programs averaging ~9 nodes against a
//! default 14-node cap), it silently discards the LARGER parent
//! entirely and falls back to mutating only the smaller one — real
//! recombination only happens between already-small programs. Overnight
//! evidence (`fractals_dag_quat/OVERNIGHT_LOG.md`) already showed the
//! GA needed heavy immigration (fresh unrelated genomes each
//! generation) to stay diverse — Carl's own read: that's the GA's own
//! crossover/mutation not doing real evolutionary work, immigration
//! compensating for weak operators rather than just adding healthy
//! diversity.
//!
//! # The fix
//!
//! `crossover_program_subtree` picks an actual point INSIDE each
//! parent, extracts that point's dependency closure as a genuine
//! subtree (reusing `formula::reachable_from_root` — treating a program
//! SLICE `&prog[0..=k]` as its own root-k program is exactly what that
//! function already does, no new traversal needed), and splices B's
//! extracted subtree into A in place of A's chosen subtree. Structural
//! validity (every operand still points to a strictly-earlier node) is
//! maintained by construction, the same way the existing operators do
//! it — not by a post-hoc repair pass.

use crate::formula::{op, OpNode, N_SLOTS};
use crate::genome::{strip_dead, UNARY_OPS, BINARY_OPS, re_k, im_k};
use rand::Rng;

/// Tunable knobs for `mutate_program_tuned`. `Default` matches
/// `genome.rs::mutate_program`'s current hardcoded behavior exactly (1-2
/// edits, ±0.3 constant perturbation) — this is a strict superset of the
/// existing behavior, not a silent change, so the default single-metric
/// run and the existing one are directly comparable.
#[derive(Clone, Copy, Debug)]
pub struct MutationStrength {
    /// Number of edit passes applied per call is drawn uniformly from
    /// `min_edits..=max_edits`.
    pub min_edits: u32,
    pub max_edits: u32,
    /// Half-width of the uniform perturbation applied to a CONST node's
    /// (kre, kim) on a "perturb constant" edit.
    pub const_perturb_scale: f32,
    /// Depth-aware guard on the "grow" edit kind (the only one that can
    /// increase depth) — skip growing past this, computed fresh via
    /// `formula::program_depth` since no per-node depth is stored. The
    /// existing `mutate_program` accepts a `max_depth` parameter and
    /// silently discards it (`let _ = max_depth;`); this is the
    /// replacement that actually enforces it.
    pub max_depth: usize,
}

impl Default for MutationStrength {
    fn default() -> Self {
        MutationStrength { min_edits: 1, max_edits: 2, const_perturb_scale: 0.3, max_depth: 12 }
    }
}

/// Real subtree crossover for flat topological-order DAG programs.
///
/// Picks a random node in each parent as a crossover point (retried a
/// few times to prefer a non-leaf point when possible — crossing over at
/// a bare `Z`/`C`/`CONST` leaf is a valid but low-value swap), extracts
/// B's chosen point's full dependency closure via `strip_dead(&b[0..=
/// cut_b])` (reindexed to a contiguous 0-based block — this is exactly
/// what `strip_dead` already does, just given a slice ending at the
/// chosen point instead of the whole program), and splices it into A in
/// place of A's chosen point: everything in A before the cut point is
/// kept as-is, B's extracted subtree is inserted right after (its
/// internal operand indices offset the same way the existing whole-
/// program crossover already offsets B's indices), and everything in A
/// after the cut point is kept but remapped — any reference to the OLD
/// cut point now points at B's spliced-in subtree's new root, and
/// references to anything else shift by the size difference.
///
/// Retries with a smaller extracted subtree (biasing `cut_b` toward
/// earlier/smaller nodes) if the result would exceed `max_nodes`; if
/// that still doesn't fit after a few tries, strips dead code from both
/// parents first (many "too big" pairs are only too big because of
/// uncollected introns — `strip_dead` already exists and is already used
/// by `blend_with` for exactly this reason) and retries once more before
/// finally falling back to mutating the smaller (stripped) parent, the
/// same last resort the existing operator uses — now genuinely rare
/// rather than the common case.
pub fn crossover_program_subtree(a: &[OpNode], b: &[OpNode], rng: &mut impl Rng, max_nodes: usize) -> Vec<OpNode> {
    let cap = max_nodes.clamp(4, N_SLOTS);
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let try_splice = |a: &[OpNode], b: &[OpNode], rng: &mut dyn rand::RngCore, cut_a: usize| -> Option<Vec<OpNode>> {
        // Bias cut_b toward the front of b (smaller closures) on later
        // attempts by capping the range it's drawn from — kept simple:
        // caller controls this by passing a truncated `b`.
        let cut_b = rng.random_range(0..b.len());
        let b_subtree = strip_dead(&b[0..=cut_b]);
        let new_len = cut_a + b_subtree.len() + (a.len() - cut_a - 1);
        if new_len > cap || new_len == 0 {
            return None;
        }
        let mut child: Vec<OpNode> = Vec::with_capacity(new_len);
        // Prefix: a[0..cut_a], unchanged, same positions.
        child.extend_from_slice(&a[0..cut_a]);
        // B's extracted subtree, operand indices offset by cut_a (same
        // technique the existing whole-program crossover already uses).
        let off = cut_a as u8;
        for n in &b_subtree {
            let mut m = *n;
            let ar = op::arity(m.op);
            if ar >= 1 {
                m.a = m.a.saturating_add(off);
            }
            if ar >= 2 {
                m.b = m.b.saturating_add(off);
            }
            child.push(m);
        }
        let new_root_for_old_cut_a = (cut_a + b_subtree.len() - 1) as u8;
        // Suffix: a[cut_a+1..], remapped — old index < cut_a stays put,
        // old index == cut_a now points at the spliced-in root, old
        // index > cut_a shifts by however much the splice changed the
        // node count before it.
        let shift = b_subtree.len() as isize - 1; // net nodes inserted in place of the 1 old cut_a node
        for n in &a[cut_a + 1..] {
            let mut m = *n;
            let ar = op::arity(m.op);
            let remap = |old: u8| -> u8 {
                let oi = old as usize;
                if oi < cut_a {
                    old
                } else if oi == cut_a {
                    new_root_for_old_cut_a
                } else {
                    (oi as isize + shift) as u8
                }
            };
            if ar >= 1 {
                m.a = remap(m.a);
            }
            if ar >= 2 {
                m.b = remap(m.b);
            }
            child.push(m);
        }
        Some(child)
    };

    let gp = gen_params();
    let (pref_attempts, leaf_retries) = (gp.cut_pref_attempts.round().max(0.0) as usize, gp.cut_leaf_retries.round().max(0.0) as usize);
    for attempt in 0..6 {
        // Prefer a non-leaf cut point on early attempts (more interesting
        // swaps); any attempt beyond the 3rd just takes whatever's drawn,
        // so this can't loop forever on an all-leaf program.
        let mut cut_a = rng.random_range(0..a.len());
        if attempt < pref_attempts {
            for _ in 0..leaf_retries {
                if op::arity(a[cut_a].op) > 0 {
                    break;
                }
                cut_a = rng.random_range(0..a.len());
            }
        }
        if let Some(child) = try_splice(a, b, rng, cut_a) {
            return child;
        }
    }

    // Everything tried was too big — strip dead code from both parents
    // (frequently the actual reason two "big" programs don't fit) and
    // try once more before giving up on real recombination.
    let a_live = strip_dead(a);
    let b_live = strip_dead(b);
    if a_live.len() + 1 <= cap {
        let cut_a = rng.random_range(0..a_live.len());
        if let Some(child) = try_splice(&a_live, &b_live, rng, cut_a) {
            return child;
        }
    }
    let (small, _) = if a_live.len() <= b_live.len() { (&a_live, &b_live) } else { (&b_live, &a_live) };
    mutate_program_tuned(small, rng, max_nodes, MutationStrength::default().max_depth, &MutationStrength::default())
}

/// Every number the program generator and mutation used to hard-code, so the tuning can
/// search them (`quat-dag-evolve --gen-params name:value,...`; unlisted = the defaults here,
/// which reproduce the old behaviour).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenParams {
    /// A new `bulb` node's power is drawn uniformly from `bulb_pow_min..=bulb_pow_max`.
    pub bulb_pow_min: f64,
    pub bulb_pow_max: f64,
    /// Chance that a new bulb's angular power is drawn independently of its radial power
    /// (0 = the same power for both, the classic Mandelbulb).
    pub bulb_twist: f64,
    /// A new `kbc` constant is `±1 * kbc_scale` in each of its B and C parts.
    pub kbc_scale: f64,
    /// Fresh random constants are uniform in `±const_range` (real and imaginary part).
    pub const_range: f64,
    /// Chance a new random program starts with a constant node.
    pub const_start: f64,
    /// Per-step chance the generator adds a constant node instead of an op.
    pub const_node: f64,
    /// Node cap for mutation, crossover and random immigrants (the format maximum is 24).
    pub max_nodes: f64,
    /// Dynamics of random immigrants only: Julia mode / phoenix terms / warp program chances,
    /// their constant sizes, and the bailout radius range.
    pub julia_prob: f64,
    pub julia_scale: f64,
    pub phoenix_prob: f64,
    pub phoenix_scale: f64,
    pub warp_prob: f64,
    pub bailout_min: f64,
    pub bailout_max: f64,
    /// Subtree crossover: the first `cut_pref_attempts` of its 6 tries redraw a leaf cut point up
    /// to `cut_leaf_retries` times, looking for an op node.
    pub cut_pref_attempts: f64,
    pub cut_leaf_retries: f64,
    /// The prune edit never shrinks a program below this many nodes.
    pub prune_min: f64,
    /// Legacy crossover: relative chance of each combiner on top of the two parents.
    pub comb_add: f64,
    pub comb_sub: f64,
    pub comb_mul: f64,
    pub comb_div: f64,
}

const GEN_DEFAULT: GenParams = GenParams {
    bulb_pow_min: 2.0, bulb_pow_max: 9.0, bulb_twist: 0.0, kbc_scale: 0.5, const_range: 1.0,
    const_start: 0.5, const_node: 0.05, max_nodes: 24.0,
    julia_prob: 0.30, julia_scale: 0.9, phoenix_prob: 0.25, phoenix_scale: 0.5, warp_prob: 0.30,
    bailout_min: 2.0, bailout_max: 10.0,
    cut_pref_attempts: 3.0, cut_leaf_retries: 3.0, prune_min: 3.0,
    comb_add: 1.0, comb_sub: 1.0, comb_mul: 1.0, comb_div: 1.0,
};

impl Default for GenParams {
    fn default() -> Self { GEN_DEFAULT }
}

pub static GEN_PARAMS: std::sync::Mutex<GenParams> = std::sync::Mutex::new(GEN_DEFAULT);

pub fn gen_params() -> GenParams { *GEN_PARAMS.lock().unwrap() }

/// Install `g` for this process, including the legacy crossover's combiner weights (which live
/// in `genome.rs`, shared with the 2D GA).
pub fn set_gen_params(g: GenParams) {
    *GEN_PARAMS.lock().unwrap() = g;
    *crate::genome::COMBINER_WEIGHTS.lock().unwrap() = [g.comb_add, g.comb_sub, g.comb_mul, g.comb_div];
}

/// Node cap used by the quat GA's breeding (`GenParams::max_nodes`, 4..=24).
pub fn max_nodes() -> usize { (gen_params().max_nodes.round() as usize).clamp(4, N_SLOTS) }

/// Parse `name:value,...` into `GenParams` (unlisted names keep the default; unknown names are an error).
pub fn parse_gen_params(s: &str) -> Result<GenParams, String> {
    let mut g = GenParams::default();
    for part in s.split(',').filter(|p| !p.trim().is_empty()) {
        let (k, v) = part.split_once(':').ok_or_else(|| format!("bad entry {part:?}"))?;
        let v: f64 = v.trim().parse().map_err(|_| format!("bad number in {part:?}"))?;
        match k.trim() {
            "bulb_pow_min" => g.bulb_pow_min = v, "bulb_pow_max" => g.bulb_pow_max = v, "bulb_twist" => g.bulb_twist = v,
            "kbc_scale" => g.kbc_scale = v, "const_range" => g.const_range = v, "const_start" => g.const_start = v,
            "const_node" => g.const_node = v, "max_nodes" => g.max_nodes = v,
            "julia_prob" => g.julia_prob = v, "julia_scale" => g.julia_scale = v,
            "phoenix_prob" => g.phoenix_prob = v, "phoenix_scale" => g.phoenix_scale = v,
            "warp_prob" => g.warp_prob = v, "bailout_min" => g.bailout_min = v, "bailout_max" => g.bailout_max = v,
            "cut_pref_attempts" => g.cut_pref_attempts = v, "cut_leaf_retries" => g.cut_leaf_retries = v, "prune_min" => g.prune_min = v,
            "comb_add" => g.comb_add = v, "comb_sub" => g.comb_sub = v, "comb_mul" => g.comb_mul = v, "comb_div" => g.comb_div = v,
            other => return Err(format!("unknown gen param {other:?}")),
        }
    }
    Ok(g)
}

impl GenParams {
    /// (radial, angular) power for a new bulb.
    fn bulb_powers(&self, rng: &mut impl Rng) -> (f32, f32) {
        let (lo, hi) = (self.bulb_pow_min.min(self.bulb_pow_max), self.bulb_pow_min.max(self.bulb_pow_max));
        let draw = |rng: &mut dyn rand::RngCore| if hi > lo { rng.random_range(lo..hi) as f32 } else { lo as f32 };
        let n = draw(rng);
        let a = if self.bulb_twist > 0.0 && rng.random_bool(self.bulb_twist.clamp(0.0, 1.0)) { draw(rng) } else { n };
        (n, a)
    }
    /// Fresh constant node, uniform in `±const_range`.
    fn fresh_const(&self, rng: &mut impl Rng) -> OpNode {
        let r = self.const_range as f32;
        OpNode { op: op::CONST, a: 0, b: 0, kre: re_k(rng) * r, kim: im_k(rng) * r }
    }
    fn kbc_parts(&self, rng: &mut impl Rng) -> (f32, f32) {
        let k = self.kbc_scale as f32;
        (re_k(rng) * k, im_k(rng) * k)
    }
}

/// `Genome::randomize_dynamics` with the chances and sizes taken from `GenParams`.
pub fn randomize_dynamics_tuned(g: &mut crate::genome::Genome, rng: &mut impl Rng) {
    let gp = gen_params();
    if rng.random_bool(gp.julia_prob.clamp(0.0, 1.0)) {
        g.julia_mode = true;
        g.julia_cre = re_k(rng) * gp.julia_scale as f32;
        g.julia_cim = im_k(rng) * gp.julia_scale as f32;
    }
    if rng.random_bool(gp.phoenix_prob.clamp(0.0, 1.0)) {
        g.phoenix_re = re_k(rng) * gp.phoenix_scale as f32;
        g.phoenix_im = im_k(rng) * gp.phoenix_scale as f32;
    }
    let (lo, hi) = (gp.bailout_min.min(gp.bailout_max), gp.bailout_min.max(gp.bailout_max));
    g.bailout_radius = (lo + rng.random::<f64>() * (hi - lo)) as f32;
    if rng.random_bool(gp.warp_prob.clamp(0.0, 1.0)) {
        let wexotic = rng.random_bool(0.5);
        g.warp = crate::genome::random_program(rng, 8, 4, wexotic);
    }
}

/// Off with `quat-dag-evolve --no-bulb` (for with/without comparisons).
pub static BULB_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Chance per `mutate_program_tuned` call of splicing in a BULB node
/// (f64 bits; `quat-dag-evolve --bulb-prob`).
pub static BULB_SPLICE_PROB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x3FB999999999999A); // 0.1
/// Relative weights of the 5 edit kinds (perturb constant, swap op, rewire, grow,
/// prune); `quat-dag-evolve --edit-weights a,b,c,d,e`. Default: equal.
pub static EDIT_WEIGHTS: std::sync::Mutex<[f64; 5]> = std::sync::Mutex::new([1.0; 5]);
/// Index of the splice-only pseudo-op `kbcmul` in the op-weight table: `x × kbc` (a MUL node
/// whose other input is a fresh CONSTBC leaf, in a random order since quaternion products do
/// not commute). It is not a real opcode, so programs and every evaluator are unchanged.
pub const KBC_MUL: u8 = op::N_OPS as u8;
/// Size of the op-weight table: every opcode plus `KBC_MUL`.
pub const N_WEIGHTS: usize = op::N_OPS + 1;

/// Per-op selection weights for the mutation's "swap op" and "grow" edits
/// (`quat-dag-evolve --op-weights sqr:0.07,mul:0.25,bulb:0.15,...`). `None` = the legacy
/// behaviour (uniform unary/binary choice, bulb/kbc spliced at their own probabilities).
/// When set it covers EVERY op, `bulb` and `kbc` included: picking one of those splices it in
/// (`splice_bulb`/`splice_kbc`) instead of replacing a node, since both need a companion
/// constant, and the independent bulb/kbc splice chances are switched off.
pub static OP_WEIGHTS: std::sync::Mutex<Option<[f64; N_WEIGHTS]>> = std::sync::Mutex::new(None);

/// Weighted pick from `ops` (weight 0 = never). `--no-bulb`/`--no-kbc` zero those ops.
fn pick_weighted(rng: &mut impl Rng, ops: &[u8], w: &[f64; N_WEIGHTS]) -> Option<u8> {
    let wt = |o: u8| -> f64 {
        let off = (o == op::BULB && !BULB_ENABLED.load(std::sync::atomic::Ordering::Relaxed))
            || ((o == op::CONSTBC || o == KBC_MUL) && !KBC_ENABLED.load(std::sync::atomic::Ordering::Relaxed));
        if off { 0.0 } else { w[o as usize].max(0.0) }
    };
    let total: f64 = ops.iter().map(|&o| wt(o)).sum();
    if total <= 0.0 { return None; }
    let mut r = rng.random::<f64>() * total;
    for &o in ops {
        let x = wt(o);
        if r < x { return Some(o); }
        r -= x;
    }
    ops.last().copied()
}

/// Splice `kind` (BULB or CONSTBC) after node `i` with fresh random constants. Same depth
/// and size guards as the legacy splices; None when it does not fit.
fn splice_op(rng: &mut impl Rng, p: &[OpNode], i: usize, kind: u8, cap: usize, max_depth: usize) -> Option<Vec<OpNode>> {
    if p.len() + 2 > cap || crate::formula::program_depth(p) >= max_depth { return None; }
    if kind == op::BULB {
        if p[i].op == op::CONST { return None; }
        let (n, a) = gen_params().bulb_powers(rng);
        splice_bulb(p, i as u8, n, a)
    } else if kind == KBC_MUL {
        let (kb, kc) = gen_params().kbc_parts(rng);
        splice_kbc_with(p, i as u8, kb, kc, op::MUL, rng.random_bool(0.5))
    } else {
        let (kb, kc) = gen_params().kbc_parts(rng);
        splice_kbc(p, i as u8, kb, kc)
    }
}

/// Share of immigrants that are freshly generated from `OP_WEIGHTS` instead of loaded from
/// the pool (f64 bits; `quat-dag-evolve --random-immigrant-frac`).
pub static RANDOM_IMMIGRANT_FRAC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Largest program (node count) a random immigrant may have; its length is drawn uniformly
/// from 3..=this (z, c and one op is the smallest program) (`quat-dag-evolve --random-immigrant-nodes`).
pub static RANDOM_IMMIGRANT_NODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(24);

/// A fresh random program whose op mix follows `w` (the tuning's op weights): same shape as
/// `genome::random_program` (leaves z, c and sometimes a constant, then nodes wired to earlier
/// ones within the depth/size caps), but every op is drawn from the weight table. `bulb` and
/// `kbc` bring their companion constant node (`bulb(x, k)`, `x + kbc`). All-zero weights
/// fall back to `random_program`.
pub fn random_program_weighted(rng: &mut impl Rng, max_nodes: usize, max_depth: usize, w: &[f64; N_WEIGHTS]) -> Vec<OpNode> {
    let mut cand: Vec<u8> = UNARY_OPS.to_vec();
    cand.extend_from_slice(&BINARY_OPS);
    cand.push(op::BULB);
    cand.push(op::CONSTBC);
    cand.push(KBC_MUL);
    if pick_weighted(rng, &cand, w).is_none() {
        return crate::genome::random_program(rng, max_nodes, max_depth, false);
    }
    let gp = gen_params();
    let cap = max_nodes.clamp(3, N_SLOTS);
    let leaf = |o: u8| OpNode { op: o, a: 0, b: 0, kre: 0.0, kim: 0.0 };
    let mut nodes = vec![leaf(op::Z), leaf(op::C)];
    let mut depth = vec![0usize, 0];
    if rng.random_bool(gp.const_start.clamp(0.0, 1.0)) {
        nodes.push(gp.fresh_const(rng));
        depth.push(0);
    }
    let target = rng.random_range(3..=cap);
    let mut tries = 0usize;
    while nodes.len() < target && tries < target * 8 {
        tries += 1;
        if rng.random_bool(gp.const_node.clamp(0.0, 1.0)) {
            nodes.push(gp.fresh_const(rng));
            depth.push(0);
            continue;
        }
        let Some(o) = pick_weighted(rng, &cand, w) else { break };
        let a = rng.random_range(0..nodes.len());
        match o {
            op::BULB | op::CONSTBC | KBC_MUL => {
                if nodes.len() + 2 > cap { continue; }
                let d = 1 + depth[a];
                if d > max_depth { continue; }
                let k = nodes.len() as u8;
                if o == op::BULB {
                    let (n, an) = gp.bulb_powers(rng);
                    nodes.push(OpNode { op: op::CONST, a: 0, b: 0, kre: n, kim: an });
                } else {
                    let (kb, kc) = gp.kbc_parts(rng);
                    nodes.push(OpNode { op: op::CONSTBC, a: 0, b: 0, kre: kb, kim: kc });
                }
                depth.push(0);
                let (ia, ib) = if o == KBC_MUL && rng.random_bool(0.5) { (k, a as u8) } else { (a as u8, k) };
                nodes.push(OpNode { op: match o { op::BULB => op::BULB, KBC_MUL => op::MUL, _ => op::ADD }, a: ia, b: ib, kre: 0.0, kim: 0.0 });
                depth.push(d);
            }
            _ if op::arity(o) == 2 => {
                let b = rng.random_range(0..nodes.len());
                let d = 1 + depth[a].max(depth[b]);
                if d > max_depth { continue; }
                nodes.push(OpNode { op: o, a: a as u8, b: b as u8, kre: 0.0, kim: 0.0 });
                depth.push(d);
            }
            _ => {
                let d = 1 + depth[a];
                if d > max_depth { continue; }
                nodes.push(OpNode { op: o, a: a as u8, b: 0, kre: 0.0, kim: 0.0 });
                depth.push(d);
            }
        }
    }
    // A non-leaf root that mixes two earlier nodes, as in random_program.
    if op::arity(nodes[nodes.len() - 1].op) == 0 && nodes.len() < N_SLOTS {
        let a = (nodes.len() - 1) as u8;
        let b = if nodes.len() >= 2 { (nodes.len() - 2) as u8 } else { 0 };
        let bins = pick_weighted(rng, &BINARY_OPS, w).unwrap_or(op::ADD);
        nodes.push(OpNode { op: bins, a, b, kre: 0.0, kim: 0.0 });
    }
    nodes
}

fn prob(p: &std::sync::atomic::AtomicU64) -> f64 { f64::from_bits(p.load(std::sync::atomic::Ordering::Relaxed)).clamp(0.0, 1.0) }

/// Off with `quat-dag-evolve --no-kbc`.
pub static KBC_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Chance per `mutate_program_tuned` call of splicing in a B/C-part constant
/// (f64 bits; `quat-dag-evolve --kbc-prob`).
pub static KBC_SPLICE_PROB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x3FB999999999999A); // 0.1

/// Splice `ADD(prog[node], CONSTBC(kb, kc))` directly after `node` and repoint later
/// references to it (same remapping as `splice_bulb`). None if it would not fit.
pub fn splice_kbc(prog: &[OpNode], node: u8, kb: f32, kc: f32) -> Option<Vec<OpNode>> {
    splice_kbc_with(prog, node, kb, kc, op::ADD, false)
}

/// `splice_kbc` with a chosen combiner (`ADD` or `MUL`) and operand order (`kbc_first` puts the
/// constant on the left, which matters for the non-commutative quaternion product).
pub fn splice_kbc_with(prog: &[OpNode], node: u8, kb: f32, kc: f32, comb: u8, kbc_first: bool) -> Option<Vec<OpNode>> {
    let n = node as usize;
    if n >= prog.len() || prog.len() + 2 > N_SLOTS {
        return None;
    }
    let (k_idx, s_idx) = ((n + 1) as u8, (n + 2) as u8);
    let mut out: Vec<OpNode> = Vec::with_capacity(prog.len() + 2);
    out.extend_from_slice(&prog[..=n]);
    out.push(OpNode { op: op::CONSTBC, a: 0, b: 0, kre: kb, kim: kc });
    let (ia, ib) = if kbc_first { (k_idx, node) } else { (node, k_idx) };
    out.push(OpNode { op: comb, a: ia, b: ib, kre: 0.0, kim: 0.0 });
    let remap = |i: u8| -> u8 { if i as usize == n { s_idx } else if (i as usize) > n { i + 2 } else { i } };
    for src in &prog[n + 1..] {
        let mut nd = *src;
        let ar = op::arity(nd.op);
        if ar >= 1 { nd.a = remap(nd.a); }
        if ar >= 2 { nd.b = remap(nd.b); }
        out.push(nd);
    }
    Some(out)
}

/// Splice `BULB(prog[node], CONST(n_r, n_a))` directly after `node` and repoint
/// every later reference to `node` at it (same index-remapping as
/// `genome.rs::splice_scale`). None if it would not fit in `N_SLOTS`.
pub fn splice_bulb(prog: &[OpNode], node: u8, n_r: f32, n_a: f32) -> Option<Vec<OpNode>> {
    let n = node as usize;
    if n >= prog.len() || prog.len() + 2 > N_SLOTS {
        return None;
    }
    let (k_idx, b_idx) = ((n + 1) as u8, (n + 2) as u8);
    let mut out: Vec<OpNode> = Vec::with_capacity(prog.len() + 2);
    out.extend_from_slice(&prog[..=n]);
    out.push(OpNode { op: op::CONST, a: 0, b: 0, kre: n_r, kim: n_a });
    out.push(OpNode { op: op::BULB, a: node, b: k_idx, kre: 0.0, kim: 0.0 });
    let remap = |i: u8| -> u8 { if i as usize == n { b_idx } else if (i as usize) > n { i + 2 } else { i } };
    for src in &prog[n + 1..] {
        let mut nd = *src;
        let ar = op::arity(nd.op);
        if ar >= 1 { nd.a = remap(nd.a); }
        if ar >= 2 { nd.b = remap(nd.b); }
        out.push(nd);
    }
    Some(out)
}

/// Same 5 edit kinds as `genome.rs::mutate_program` (perturb a constant,
/// swap an op, rewire an input, grow, prune), but with real parameters
/// instead of hardcoded constants — see `MutationStrength`'s docs — and
/// an actual depth-aware guard on the "grow" edit kind, computed via
/// `formula::program_depth` (no per-node depth is stored, so this is a
/// fresh O(n) pass; skipped for the other 4 edit kinds, none of which
/// can increase depth).
pub fn mutate_program_tuned(prog: &[OpNode], rng: &mut impl Rng, max_nodes: usize, max_depth: usize, strength: &MutationStrength) -> Vec<OpNode> {
    let mut p = prog.to_vec();
    if p.len() < 2 {
        return crate::genome::random_program(rng, max_nodes, max_depth, false);
    }
    let cap = max_nodes.clamp(4, N_SLOTS);
    // Bulbify: the only way BULB enters a population (seed pools and immigrants
    // predate it). Later edits (constant perturbation, rewiring b to a c-dependent
    // node) give it evolving / time-driven powers and rotations. Same depth guard
    // as the grow edit: the splice adds one level on every path through `node`.
    // Counts as one of this call's edits.
    let op_table = *OP_WEIGHTS.lock().unwrap();
    let mut spliced = false;
    if op_table.is_none() && BULB_ENABLED.load(std::sync::atomic::Ordering::Relaxed) && p.len() + 2 <= cap && rng.random_bool(prob(&BULB_SPLICE_PROB))
        && crate::formula::program_depth(&p) < strength.max_depth
    {
        let i = rng.random_range(0..p.len());
        if p[i].op != op::CONST {
            let (n, a) = gen_params().bulb_powers(rng);
            if let Some(q) = splice_bulb(&p, i as u8, n, a) { p = q; spliced = true; }
        }
    }

    let edits = if strength.max_edits <= strength.min_edits {
        strength.min_edits
    } else {
        rng.random_range(strength.min_edits..=strength.max_edits)
    };
    // KBC splice: x → x + kbc(kb, kc), the only way a B/C-part constant enters (seed
    // pools predate it). Breaks the B↔C interchange symmetry behind the ring balls.
    let mut spliced = spliced;
    if op_table.is_none() && !spliced && KBC_ENABLED.load(std::sync::atomic::Ordering::Relaxed) && p.len() + 2 <= cap
        && rng.random_bool(prob(&KBC_SPLICE_PROB)) && crate::formula::program_depth(&p) < strength.max_depth
    {
        let i = rng.random_range(0..p.len());
        if op::arity(p[i].op) > 0 || p[i].op == op::Z || p[i].op == op::C {
            let (kb, kc) = gen_params().kbc_parts(rng);
            if let Some(q) = splice_kbc(&p, i as u8, kb, kc) { p = q; spliced = true; }
        }
    }
    let edits = if spliced { edits.saturating_sub(1) } else { edits };
    let weights = *EDIT_WEIGHTS.lock().unwrap();
    let total: f64 = weights.iter().map(|w| w.max(0.0)).sum();
    for _ in 0..edits {
        let kind = if total <= 0.0 { rng.random_range(0..5) } else {
            let mut r = rng.random::<f64>() * total;
            let mut k = 4;
            for (i, w) in weights.iter().enumerate() { let w = w.max(0.0); if r < w { k = i; break; } r -= w; }
            k
        };
        match kind {
            0 => {
                let i = rng.random_range(0..p.len());
                if p[i].op == op::CONST || p[i].op == op::CONSTBC {
                    p[i].kre += re_k(rng) * strength.const_perturb_scale;
                    p[i].kim += im_k(rng) * strength.const_perturb_scale;
                } else if op::arity(p[i].op) == 0 {
                    p[i] = gen_params().fresh_const(rng);
                }
            }
            1 => {
                let i = rng.random_range(1..p.len());
                let new_op = match &op_table {
                    Some(w) => {
                        let mut cand: Vec<u8> = UNARY_OPS.to_vec();
                        cand.extend_from_slice(&BINARY_OPS);
                        cand.push(op::BULB);
                        cand.push(op::CONSTBC);
                        cand.push(KBC_MUL);
                        let Some(o) = pick_weighted(rng, &cand, w) else { continue };
                        if o == op::BULB || o == op::CONSTBC || o == KBC_MUL {
                            // Needs a companion constant: splice it in at a random node.
                            let at = rng.random_range(0..p.len());
                            if let Some(q) = splice_op(rng, &p, at, o, cap, strength.max_depth) { p = q; }
                            continue;
                        }
                        o
                    }
                    None => if rng.random_bool(0.5) {
                        UNARY_OPS[rng.random_range(0..UNARY_OPS.len())]
                    } else {
                        BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
                    },
                };
                let ar = op::arity(new_op);
                p[i].op = new_op;
                if ar >= 1 {
                    p[i].a = rng.random_range(0..i) as u8;
                }
                if ar >= 2 {
                    p[i].b = rng.random_range(0..i) as u8;
                }
            }
            2 => {
                let i = rng.random_range(1..p.len());
                if op::arity(p[i].op) >= 1 {
                    p[i].a = rng.random_range(0..i) as u8;
                }
                if op::arity(p[i].op) >= 2 {
                    p[i].b = rng.random_range(0..i) as u8;
                }
            }
            3 => {
                if p.len() < cap && crate::formula::program_depth(&p) < strength.max_depth {
                    let root = (p.len() - 1) as u8;
                    let other = rng.random_range(0..p.len()) as u8;
                    let opc = match &op_table {
                        Some(w) => {
                            let mut cand: Vec<u8> = BINARY_OPS.to_vec();
                            cand.push(op::BULB);
                            let Some(o) = pick_weighted(rng, &cand, w) else { continue };
                            if o == op::BULB {
                                if let Some(q) = splice_op(rng, &p, p.len() - 1, o, cap, strength.max_depth) { p = q; }
                                continue;
                            }
                            o
                        }
                        None => if rng.random_bool(0.5) {
                            BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
                        } else {
                            op::ADD
                        },
                    };
                    p.push(OpNode { op: opc, a: root, b: other, kre: 0.0, kim: 0.0 });
                    // `other` may be a deeper (dead) node than the root, so check the
                    // new root's actual depth, not just the old root's.
                    if crate::formula::program_depth(&p) > strength.max_depth {
                        p.pop();
                    }
                }
            }
            _ => {
                if p.len() > (gen_params().prune_min.round() as usize).max(2) {
                    p.pop();
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests that set the process-global `OP_WEIGHTS`.
    static OPW_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    use rand::SeedableRng;

    /// Run `mutate_program_tuned` many times with `w` and count op occurrences in the results.
    fn op_counts_after_mutation(w: [f64; N_WEIGHTS]) -> [usize; op::N_OPS] {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let strength = MutationStrength { min_edits: 2, max_edits: 3, ..MutationStrength::default() };
        let base = crate::genome::random_program(&mut rng, 12, 6, false);
        *OP_WEIGHTS.lock().unwrap() = Some(w);
        let mut counts = [0usize; op::N_OPS];
        for _ in 0..400 {
            for n in mutate_program_tuned(&base, &mut rng, 24, 12, &strength) { counts[n.op as usize] += 1; }
        }
        *OP_WEIGHTS.lock().unwrap() = None;
        counts
    }

    #[test]
    fn weighted_random_programs_are_valid_and_follow_the_weights() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut w = [0.0; N_WEIGHTS];
        w[op::SIN as usize] = 1.0; w[op::MUL as usize] = 1.0; w[op::BULB as usize] = 1.0;
        let (mut sin, mut bulb, mut other) = (0, 0, 0);
        for _ in 0..300 {
            let p = random_program_weighted(&mut rng, 14, 6, &w);
            assert!(p.len() <= N_SLOTS);
            for (i, n) in p.iter().enumerate() {
                let ar = op::arity(n.op);
                if ar >= 1 { assert!((n.a as usize) < i, "node {i} reads a later node"); }
                if ar >= 2 { assert!((n.b as usize) < i); }
                match n.op {
                    op::SIN => sin += 1,
                    op::BULB => bulb += 1,
                    op::MUL | op::Z | op::C | op::CONST => {}
                    _ => other += 1,
                }
            }
            assert!(crate::formula::program_depth(&p) <= 7);
        }
        assert!(sin > 100 && bulb > 100, "sin={sin} bulb={bulb}");
        assert!(other <= 300, "only the forced root may be another op (add/sub/div): {other}");
    }

    #[test]
    fn op_weights_steer_which_ops_appear() {
        let _g = OPW_LOCK.lock().unwrap();
        let mut only_sin = [0.0; N_WEIGHTS];
        only_sin[op::SIN as usize] = 1.0;
        let c = op_counts_after_mutation(only_sin);
        let base_sin = c[op::SIN as usize];
        let mut only_bulb = [0.0; N_WEIGHTS];
        only_bulb[op::BULB as usize] = 1.0;
        let b = op_counts_after_mutation(only_bulb);
        assert!(b[op::BULB as usize] > 100, "bulb weight 1, everything else 0 must splice bulbs in: {}", b[op::BULB as usize]);
        assert!(base_sin > b[op::SIN as usize], "sin weighted ops produce more sin than a bulb-only table");
        let mut no_bulb = [1.0; N_WEIGHTS];
        no_bulb[op::BULB as usize] = 0.0;
        no_bulb[op::CONSTBC as usize] = 0.0;
        no_bulb[KBC_MUL as usize] = 0.0; // also splices a CONSTBC leaf
        let n = op_counts_after_mutation(no_bulb);
        assert_eq!(n[op::BULB as usize] + n[op::CONSTBC as usize], 0, "zero weight means the op never appears");
    }

    #[test]
    fn kbcmul_splices_a_multiply_by_a_bc_constant() {
        let _g = OPW_LOCK.lock().unwrap();
        let mut only = [0.0; N_WEIGHTS];
        only[KBC_MUL as usize] = 1.0;
        let c = op_counts_after_mutation(only);
        assert!(c[op::CONSTBC as usize] > 100, "kbcmul must bring CONSTBC leaves: {}", c[op::CONSTBC as usize]);
        let base = [OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }, OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 }];
        let q = splice_kbc_with(&base, 1, 0.3, -0.2, op::MUL, true).unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!((q[2].op, q[3].op), (op::CONSTBC, op::MUL));
        assert_eq!((q[3].a, q[3].b), (2, 1), "constant on the left when kbc_first");
        let q = splice_kbc_with(&base, 1, 0.3, -0.2, op::MUL, false).unwrap();
        assert_eq!((q[3].a, q[3].b), (1, 2));
    }

    #[test]
    fn splice_bulb_wraps_the_node_and_keeps_the_program_valid() {
        // z² + c  →  bulb(z, 8+8i)² + c
        let prog = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let out = splice_bulb(&prog, 0, 8.0, 8.0).unwrap();
        assert!(is_valid(&out));
        assert_eq!(out[2].op, op::BULB);
        assert_eq!((out[2].a, out[2].b), (0, 1));
        assert_eq!(out[4].op, op::SQR);
        assert_eq!(out[4].a, 2); // z² now squares bulb(z)
        assert_eq!((out[5].a, out[5].b), (4, 3));
    }

    fn is_valid(prog: &[OpNode]) -> bool {
        if prog.is_empty() || prog.len() > N_SLOTS {
            return false;
        }
        for (i, n) in prog.iter().enumerate() {
            if n.op as usize >= op::N_OPS {
                return false;
            }
            let ar = op::arity(n.op);
            if ar >= 1 && n.a as usize >= i {
                return false;
            }
            if ar >= 2 && n.b as usize >= i {
                return false;
            }
        }
        true
    }

    /// Same fuzz discipline `genome.rs`'s own
    /// `gp_operators_preserve_validity` test uses for the existing
    /// operators — the load-bearing check for a hand-rolled index-
    /// splicing operator like `crossover_program_subtree`.
    #[test]
    fn subtree_crossover_and_tuned_mutation_preserve_validity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
        let mut pool: Vec<Vec<OpNode>> = (0..12).map(|_| crate::genome::random_program(&mut rng, 14, 5, false)).collect();
        for p in &pool {
            assert!(is_valid(p), "seed program itself invalid: {p:?}");
        }
        let strength = MutationStrength::default();
        for _ in 0..500 {
            let i = rng.random_range(0..pool.len());
            let j = rng.random_range(0..pool.len());
            let child = if rng.random_bool(0.5) {
                crossover_program_subtree(&pool[i], &pool[j], &mut rng, 14)
            } else {
                mutate_program_tuned(&pool[i], &mut rng, 14, 5, &strength)
            };
            assert!(is_valid(&child), "invalid child from parents {:?} / {:?}: {child:?}", pool[i], pool[j]);
            let _ = crate::formula::eval_program(&child, 0.1, 0.2, 0.3, -0.1); // must not panic
            let replace_idx = rng.random_range(0..pool.len());
            pool[replace_idx] = child;
        }
    }

    #[test]
    fn subtree_crossover_child_is_not_always_the_sum_of_both_parent_sizes() {
        // The old whole-program grafting operator ALWAYS produces
        // len(a)+len(b)+1 on success — this is the property that made it
        // not real subtree crossover. Over enough draws, the subtree
        // version should produce a genuine spread of child sizes,
        // including some strictly smaller than that sum.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let a = crate::genome::random_program(&mut rng, 14, 5, false);
        let b = crate::genome::random_program(&mut rng, 14, 5, false);
        let whole_program_size = a.len() + b.len() + 1;
        let mut saw_smaller = false;
        let mut sizes = std::collections::HashSet::new();
        for _ in 0..40 {
            let child = crossover_program_subtree(&a, &b, &mut rng, 14);
            sizes.insert(child.len());
            if child.len() < whole_program_size {
                saw_smaller = true;
            }
        }
        assert!(saw_smaller, "never produced a child smaller than the old whole-program-grafting size ({whole_program_size}) across 40 draws");
        assert!(sizes.len() > 1, "child size never varied across 40 draws — got {sizes:?}");
    }

    #[test]
    fn mutation_strength_default_matches_legacy_hardcoded_behavior() {
        let s = MutationStrength::default();
        assert_eq!(s.min_edits, 1);
        assert_eq!(s.max_edits, 2);
        assert!((s.const_perturb_scale - 0.3).abs() < 1e-6);
    }

    /// Only the "grow" edit kind (the one that appends a node and can
    /// therefore increase depth by pushing a new root on top) is
    /// depth-guarded here — "swap op"/"rewire input" can also
    /// technically deepen a live path by rewiring toward a deeper
    /// earlier node, which this does NOT guard against (a real, accepted
    /// gap, not a mistake — see the module doc's "what was actually
    /// wrong" section for the specific thing this targets). A grow edit
    /// is the only one that changes `p.len()`, so "did the program get
    /// longer" isolates exactly the guarded mechanism without being
    /// confused by the other 4 edit kinds' own effects on depth.
    #[test]
    fn tuned_mutation_grow_edit_respects_the_depth_guard() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        // A deliberately deep, narrow chain: SIN(SIN(SIN(...Z...))).
        let mut prog = vec![OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }];
        for _ in 0..8 {
            let prev = (prog.len() - 1) as u8;
            prog.push(OpNode { op: op::SIN, a: prev, b: 0, kre: 0.0, kim: 0.0 });
        }
        let cap_depth = crate::formula::program_depth(&prog);
        let strength = MutationStrength { min_edits: 1, max_edits: 1, const_perturb_scale: 0.3, max_depth: cap_depth };
        for _ in 0..2000 {
            let before_len = prog.len();
            let next = mutate_program_tuned(&prog, &mut rng, 24, 24, &strength);
            if next.len() > before_len {
                // A grow edit fired — the guard must have required
                // program_depth(&prog) < cap_depth beforehand, so the
                // resulting depth can be at most cap_depth.
                assert!(
                    crate::formula::program_depth(&next) <= cap_depth,
                    "grow edit pushed depth to {} past the {cap_depth} guard",
                    crate::formula::program_depth(&next)
                );
            }
            prog = next;
        }
    }
}
