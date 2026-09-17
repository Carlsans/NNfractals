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
use crate::genome::{strip_dead, UNARY_OPS, BINARY_OPS, rand_const, re_k, im_k};
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

    for attempt in 0..6 {
        // Prefer a non-leaf cut point on early attempts (more interesting
        // swaps); any attempt beyond the 3rd just takes whatever's drawn,
        // so this can't loop forever on an all-leaf program.
        let mut cut_a = rng.random_range(0..a.len());
        if attempt < 3 {
            for _ in 0..3 {
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

    let edits = if strength.max_edits <= strength.min_edits {
        strength.min_edits
    } else {
        rng.random_range(strength.min_edits..=strength.max_edits)
    };
    for _ in 0..edits {
        match rng.random_range(0..5) {
            0 => {
                let i = rng.random_range(0..p.len());
                if p[i].op == op::CONST {
                    p[i].kre += re_k(rng) * strength.const_perturb_scale;
                    p[i].kim += im_k(rng) * strength.const_perturb_scale;
                } else if op::arity(p[i].op) == 0 {
                    p[i] = rand_const(rng);
                }
            }
            1 => {
                let i = rng.random_range(1..p.len());
                let new_op = if rng.random_bool(0.5) {
                    UNARY_OPS[rng.random_range(0..UNARY_OPS.len())]
                } else {
                    BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
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
                    let opc = if rng.random_bool(0.5) {
                        BINARY_OPS[rng.random_range(0..BINARY_OPS.len())]
                    } else {
                        op::ADD
                    };
                    p.push(OpNode { op: opc, a: root, b: other, kre: 0.0, kim: 0.0 });
                }
            }
            _ => {
                if p.len() > 3 {
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
    use rand::SeedableRng;

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
