//! Removes the parts of a quaternion DAG formula that change nothing.
//!
//! Carl (2026-10-14): "if removing part of the formula change nothing,
//! remove that part" — so the GA pool never carries useless elements.
//!
//! Two kinds of removal:
//! 1. Exact, no checking needed: dead nodes (introns) are dropped and
//!    identical subexpressions are merged, via the lossless text round trip
//!    `parse_program(program_to_text(p))` (the corpus test in
//!    `formula_text` proves it evaluates identically).
//! 2. Candidate edits, each kept only if the fractal is unchanged:
//!    bypass a node (replace it by one of its operands — `sin(x)`→`x`,
//!    `x+t`→`x`), replace a node by the constant 0 or 1, fold a subtree
//!    with no `z`/`c` in it into one constant, or drop the warp entirely.
//!    "Unchanged" means the edited formula gives the same value (within
//!    `TOL`, float noise) on every input the fractal actually feeds it:
//!    the warp at `N_POINTS` fixed 4D points of the radius-2 ball, the
//!    program at the `(z, c)` states the ORIGINAL orbits visit from them.
//!    Always against the original, never the previous step, so small
//!    differences cannot accumulate. Only edits that make the program
//!    strictly smaller are tried, largest removal first.
//!
//! Why values and not iterated orbits: comparing escape times let `log(z)`
//! pass as `z` (both stay inside), and comparing where orbits end rejected
//! folding a constant on a chaotic formula — a 1e-8 rounding moved 39% of
//! the orbit ends. The formula's value is what changes or not; iteration
//! only amplifies noise. The distance estimate is deliberately not
//! compared: it is a derivative bound, so value-identical forms (`z - z`
//! vs `0`) give different bounds even though they draw the same set.

use rayon::prelude::*;

use crate::formula::{op, OpNode};
use crate::formula_text::{parse_program, program_to_text};
use crate::quat_dag::{eval_program_quat, QuatDagFormula};
use crate::quaternion::Quat;

const N_POINTS: usize = 2048;
const SAMPLE_RADIUS: f64 = 2.0;
/// Orbit length whose states are recorded — well above the GA's and the
/// viewer's default 60, so states only reached deep in an orbit count too.
const MAX_ITER: u32 = 200;
const TOL: f64 = 1e-6;

pub struct Simplified {
    pub program: Vec<OpNode>,
    pub warp: Vec<OpNode>,
    /// `program.len() + warp.len()` before and after.
    pub nodes_before: usize,
    pub nodes_after: usize,
}

/// Exact canonical form: introns dropped, shared subexpressions merged,
/// root last. Falls back to the input if it cannot be re-parsed.
fn canonical(p: &[OpNode]) -> Vec<OpNode> {
    if p.is_empty() {
        return Vec::new();
    }
    parse_program(&program_to_text(p)).unwrap_or_else(|_| p.to_vec())
}

/// Decides "changes nothing" on the inputs the fractal actually feeds the
/// formula: the warp at every sample point, and the program at every
/// `(z, c)` the original orbits visit.
struct Oracle {
    points: Vec<Quat>,
    warp_ref: Vec<Quat>,
    /// `(z, c, program(z, c))` along the original orbits.
    states: Vec<(Quat, Quat, Quat)>,
}

impl Oracle {
    fn new(f: &QuatDagFormula, bailout_radius: f32) -> Self {
        Self::with(f, bailout_radius, 0x9E37_79B9_7F4A_7C15, N_POINTS, MAX_ITER)
    }

    fn with(f: &QuatDagFormula, bailout_radius: f32, seed: u64, n_points: usize, max_iter: u32) -> Self {
        // Fixed xorshift stream: the same formula always simplifies the same way.
        let mut s: u64 = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut points = Vec::with_capacity(n_points);
        while points.len() < n_points {
            let q = Quat::new(next(), next(), next(), next());
            if q.norm_sq() <= 1.0 {
                points.push(Quat::new(q.r * SAMPLE_RADIUS, q.a * SAMPLE_RADIUS, q.b * SAMPLE_RADIUS, q.c * SAMPLE_RADIUS));
            }
        }
        let warp_ref: Vec<Quat> = points.iter().map(|&p| warp_at(f.warp, p)).collect();
        let bailout_sq = (bailout_radius as f64) * (bailout_radius as f64);
        let states = warp_ref
            .par_iter()
            .flat_map_iter(|&warped| orbit_states(f, warped, max_iter, bailout_sq))
            .collect();
        Oracle { points, warp_ref, states }
    }

    fn warp_matches(&self, warp: &[OpNode]) -> bool {
        self.points.par_iter().zip(self.warp_ref.par_iter()).all(|(&p, &r)| close(warp_at(warp, p), r))
    }

    fn program_matches(&self, prog: &[OpNode]) -> bool {
        self.states.par_iter().all(|&(z, c, r)| close(eval_program_quat(prog, z, c), r))
    }

    fn matches(&self, prog: &[OpNode], warp: &[OpNode]) -> bool {
        self.warp_matches(warp) && self.program_matches(prog)
    }

    /// Fraction of recorded states the program disagrees on (test diagnostics).
    #[cfg(test)]
    fn mismatch_fraction(&self, prog: &[OpNode], warp: &[OpNode]) -> f64 {
        if !self.warp_matches(warp) {
            return 1.0;
        }
        let bad = self.states.iter().filter(|&&(z, c, r)| !close(eval_program_quat(prog, z, c), r)).count();
        bad as f64 / self.states.len().max(1) as f64
    }
}

fn warp_at(warp: &[OpNode], p: Quat) -> Quat {
    if warp.is_empty() { p } else { eval_program_quat(warp, p, p) }
}

/// The `(z, c, program(z,c))` states of one orbit, following the
/// recurrence of `quat_dag::quat_dag_escape_de` (Julia or Mandelbrot
/// start, `z ← program(z,c) + phoenix·z_prev`, bailout). Every state of
/// the first 8 iterations, then every 16th, to bound the count.
fn orbit_states(f: &QuatDagFormula, warped: Quat, max_iter: u32, bailout_sq: f64) -> Vec<(Quat, Quat, Quat)> {
    let (mut z, c) = if f.julia {
        (warped, Quat::new(f.jc.0 as f64, f.jc.1 as f64, 0.0, 0.0))
    } else {
        (Quat::ZERO, warped)
    };
    let phoenix = Quat::new(f.phoenix.0 as f64, f.phoenix.1 as f64, 0.0, 0.0);
    let mut pz = Quat::ZERO;
    let mut out = Vec::new();
    for it in 0..max_iter {
        let v = eval_program_quat(f.prog, z, c);
        if it < 8 || it % 16 == 0 {
            out.push((z, c, v));
        }
        let next = v.add(phoenix.mul(pz));
        pz = z;
        z = next;
        if z.norm_sq() > bailout_sq || !z.is_finite() {
            break;
        }
    }
    out
}

/// Equal up to float noise. Exact rewrites (`x*1`, `conj(conj(x))`) match
/// bit for bit; the tolerance admits rounding — reassociated arithmetic, a
/// constant folded to f32 — far below what the f32 GPU renderer resolves.
fn close(a: Quat, b: Quat) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return a.is_finite() == b.is_finite();
    }
    a.sub(b).norm_sq().sqrt() <= TOL * b.norm_sq().sqrt().max(1.0)
}

fn depends_on_zc(p: &[OpNode], i: usize) -> bool {
    let n = p[i];
    match n.op {
        op::Z | op::C => true,
        _ => {
            let ar = op::arity(n.op);
            (ar >= 1 && (n.a as usize) < i && depends_on_zc(p, n.a as usize))
                || (ar >= 2 && (n.b as usize) < i && depends_on_zc(p, n.b as usize))
        }
    }
}

/// Indices of the nodes strictly beneath `i`.
fn descendants(p: &[OpNode], i: usize) -> Vec<usize> {
    let mut seen = vec![false; i];
    let mut stack = vec![i];
    while let Some(k) = stack.pop() {
        let n = p[k];
        let ar = op::arity(n.op);
        for (want, x) in [(1, n.a as usize), (2, n.b as usize)] {
            if ar >= want && x < k && !seen[x] {
                seen[x] = true;
                stack.push(x);
            }
        }
    }
    (0..i).filter(|&j| seen[j]).collect()
}

fn konst(re: f32, im: f32) -> OpNode {
    OpNode { op: op::CONST, a: 0, b: 0, kre: re, kim: im }
}

/// Every smaller program reachable from `p` (already canonical) by one
/// candidate edit, smallest first, without duplicates.
fn candidates(p: &[OpNode]) -> Vec<Vec<OpNode>> {
    let mut out: Vec<Vec<OpNode>> = Vec::new();
    let mut push = |q: Vec<OpNode>| {
        let q = canonical(&q);
        if q.len() < p.len() && !out.contains(&q) {
            out.push(q);
        }
    };
    let replaced = |i: usize, node: OpNode| {
        let mut q = p.to_vec();
        q[i] = node;
        q
    };
    for i in (0..p.len()).rev() {
        let n = p[i];
        let ar = op::arity(n.op);
        if ar == 0 {
            continue;
        }
        // Bypass: the node becomes a copy of any node beneath it (whose
        // own operands are earlier still, so the order stays valid).
        // Any descendant, not just a direct operand, so pairs that only
        // cancel together (`conj(conj(x))`) can go in one step.
        for j in descendants(p, i) {
            push(replaced(i, p[j]));
        }
        if !depends_on_zc(p, i) {
            let v = eval_program_quat(&p[..=i], Quat::ZERO, Quat::ZERO);
            if v.b == 0.0 && v.c == 0.0 && v.r.is_finite() && v.a.is_finite() {
                push(replaced(i, konst(v.r as f32, v.a as f32)));
            }
        }
        push(replaced(i, konst(0.0, 0.0)));
        push(replaced(i, konst(1.0, 0.0)));
    }
    out.sort_by_key(|q| q.len());
    out
}

/// Simplifies `f.prog` and `f.warp` (see the module docs). The result
/// renders the same fractal; `bailout_radius` is the genome's own.
pub fn simplify_formula(f: &QuatDagFormula, bailout_radius: f32) -> Simplified {
    let nodes_before = f.prog.len() + f.warp.len();
    let unchanged = || Simplified { program: f.prog.to_vec(), warp: f.warp.to_vec(), nodes_before, nodes_after: nodes_before };
    if f.prog.is_empty() {
        return unchanged();
    }
    let oracle = Oracle::new(f, bailout_radius);
    let mut prog = canonical(f.prog);
    let mut warp = canonical(f.warp);
    // The canonical form is exact by construction; this guards against a
    // printer/parser bug ever reaching the gene pool.
    if !oracle.matches(&prog, &warp) {
        return unchanged();
    }
    loop {
        let mut changed = false;
        if let Some(q) = candidates(&prog).into_iter().find(|q| oracle.matches(q, &warp)) {
            prog = q;
            changed = true;
        }
        if !warp.is_empty() {
            let mut warp_cands = vec![Vec::new()];
            warp_cands.extend(candidates(&warp));
            if let Some(w) = warp_cands.into_iter().find(|w| oracle.matches(&prog, w)) {
                warp = w;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let nodes_after = prog.len() + warp.len();
    Simplified { program: prog, warp, nodes_before, nodes_after }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simplify(text: &str) -> String {
        let prog = parse_program(text).unwrap();
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        program_to_text(&simplify_formula(&f, 2.0).program)
    }

    #[test]
    fn useless_parts_are_removed() {
        assert_eq!(simplify("z^2 + c + 0"), "z^2 + c");
        assert_eq!(simplify("(z^2 + c) * 1"), "z^2 + c");
        assert_eq!(simplify("conj(conj(z^2)) + c"), "z^2 + c");
        assert_eq!(simplify("z^2 + c + (z - z)"), "z^2 + c");
        assert_eq!(simplify("z^2 + c * cos(0)"), "z^2 + c");
    }

    #[test]
    fn meaningful_parts_are_kept() {
        for text in ["z^2 + c", "z^3 + c", "sin(z)^2 + c", "conj(z)^2 + c", "z^2 + c + 0.1", "log(z)"] {
            assert_eq!(simplify(text), text);
        }
    }

    #[test]
    fn introns_are_dropped() {
        let mut prog = parse_program("z^2 + c").unwrap();
        let root = prog.pop().unwrap();
        prog.push(OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 }); // dead
        prog.push(root);
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let s = simplify_formula(&f, 2.0);
        assert_eq!(s.nodes_before, 5);
        assert_eq!(s.nodes_after, 4);
        assert_eq!(program_to_text(&s.program), "z^2 + c");
    }

    #[test]
    fn identity_warp_is_dropped() {
        let prog = parse_program("z^2 + c").unwrap();
        let warp = parse_program("z * 1").unwrap();
        let f = QuatDagFormula { prog: &prog, warp: &warp, julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        assert!(simplify_formula(&f, 2.0).warp.is_empty());
    }

    /// Never changes what a real corpus genome draws, and reports how much
    /// it removes: `cargo test --release corpus_simplification -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn corpus_simplification() {
        let mut files: Vec<_> = std::fs::read_dir("train_corpus_quat")
            .expect("train_corpus_quat")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "nn"))
            .collect();
        files.sort();
        files.truncate(200);
        let (mut before, mut exact, mut after, mut touched) = (0, 0, 0, 0);
        let mut worst = (0.0f64, String::new());
        let mut mismatched = 0;
        let mut secs = 0.0;
        for path in &files {
            let Ok(g) = crate::io::load_genome(path) else { continue };
            if g.program.is_empty() {
                continue;
            }
            let f = QuatDagFormula { prog: &g.program, warp: &g.warp, julia: g.julia_mode, jc: (g.julia_cre, g.julia_cim), phoenix: (g.phoenix_re, g.phoenix_im) };
            let t0 = std::time::Instant::now();
            let s = simplify_formula(&f, g.bailout_radius);
            secs += t0.elapsed().as_secs_f64();
            // Independent check: other points, more iterations than the
            // decision used, so a sampling blind spot would show up here.
            let check = Oracle::with(&f, g.bailout_radius, 0xD1B5_4A32_D192_ED03, 8192, 400);
            let m = check.mismatch_fraction(&s.program, &s.warp);
            if m > 0.0 {
                mismatched += 1;
                let canon_only = check.mismatch_fraction(&canonical(&g.program), &canonical(&g.warp));
                eprintln!("  {path:?} {:.4}% (exact rewrite alone {:.4}%) warp `{}` -> `{}`", m * 100.0, canon_only * 100.0, program_to_text(&g.warp), program_to_text(&s.warp));
            }
            assert!(m < 0.005, "{path:?} changed on {:.2}% of states", m * 100.0);
            if m > worst.0 {
                worst = (m, format!("{path:?}: {} -> {}", program_to_text(&g.program), program_to_text(&s.program)));
            }
            exact += canonical(&g.program).len() + canonical(&g.warp).len();
            before += s.nodes_before;
            after += s.nodes_after;
            if s.nodes_after < s.nodes_before {
                touched += 1;
            }
        }
        eprintln!("{} genomes: {touched} simplified, nodes {before} -> {exact} (dead/duplicate nodes only) -> {after}", files.len());
        eprintln!("simplify: {:.1} ms per genome", secs * 1000.0 / files.len() as f64);
        eprintln!("{mismatched} genomes differ anywhere on the independent check");
        eprintln!("worst independent-check mismatch {:.4}% {}", worst.0 * 100.0, worst.1);
    }

}
