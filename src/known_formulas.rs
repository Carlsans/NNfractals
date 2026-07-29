// A small, hand-curated catalog of named iterated-complex-map fractal
// formulas, expressed in this project's own DAG opcode vocabulary
// (`formula::op`), used by `fractal::known_formula_match` to give evolved
// DAG genomes a best-effort "closest known formula" discovery label.
//
// Deliberately excludes Newton fractals and the Magnet fractal: both need a
// generic polynomial + derivative construct this DSL doesn't have, and
// Newton fractals specifically use a converge-to-root rendering paradigm
// fundamentally different from this project's escape-time bailout loop —
// not just a missing formula, a mismatched rendering model.
//
// "Physics" correspondence is real but narrow. The one strong, citable tie
// here is Lambda (the logistic map): population dynamics, period-doubling
// route to chaos, Lyapunov exponents. The rest are primarily mathematical
// variations on Mandelbrot, not literal physics equations — this catalog is
// honestly a math-formula-family matcher with occasional physics context,
// not a general physics classifier.

use crate::formula::{op, OpNode};
use crate::genome::ProgramBuilder;

pub struct KnownFormula {
    pub name:     &'static str,
    pub category: &'static str,
    pub context:  &'static str,
    pub build:    fn() -> Vec<OpNode>,
}

pub const LIBRARY: &[KnownFormula] = &[
    KnownFormula {
        name: "Mandelbrot", category: "classic power map",
        context: "z² + c — Mandelbrot, 1980; the archetypal escape-time fractal.",
        build: build_mandelbrot,
    },
    KnownFormula {
        name: "Tricorn (Mandelbar)", category: "classic power map",
        context: "conj(z)² + c — Milnor's antiholomorphic analogue of the Mandelbrot set.",
        build: build_tricorn,
    },
    KnownFormula {
        name: "Burning Ship", category: "non-holomorphic fold",
        context: "(|Re z|+i|Im z|)² + c — Michelitsch & Rössler, 1992; folds the orbit onto the axes.",
        build: build_burning_ship,
    },
    KnownFormula {
        name: "Burning Ship (cubic)", category: "non-holomorphic fold",
        context: "(|Re z|+i|Im z|)³ + c — cubic variant of the Burning Ship fold.",
        build: build_burning_ship_cubic,
    },
    KnownFormula {
        name: "Perpendicular Burning Ship", category: "non-holomorphic fold",
        context: "(Re z − i|Im z|)² + c — Burning-Ship-family fold, imaginary axis only.",
        build: build_perpendicular_burning_ship,
    },
    KnownFormula {
        name: "Celtic Mandelbrot", category: "non-holomorphic fold",
        context: "|Re(z²)| + i·Im(z²) + c — Burning-Ship-family variant, folds after squaring.",
        build: build_celtic,
    },
    KnownFormula {
        name: "Perpendicular Mandelbrot", category: "non-holomorphic fold",
        context: "(Re z + i|Im z|)² + c — Burning-Ship-family fold, imaginary axis only.",
        build: build_perpendicular_mandelbrot,
    },
    KnownFormula {
        name: "Buffalo", category: "non-holomorphic fold",
        context: "|z|² − |z| + c — Burning-Ship-family variant.",
        build: build_buffalo,
    },
    KnownFormula {
        name: "Multibrot (d=3)", category: "power map",
        context: "z³ + c — generalized Mandelbrot family, degree 3.",
        build: build_multibrot3,
    },
    KnownFormula {
        name: "Multibrot (d=4)", category: "power map",
        context: "z⁴ + c — generalized Mandelbrot family, degree 4.",
        build: build_multibrot4,
    },
    KnownFormula {
        name: "Multibrot (d=5)", category: "power map",
        context: "z⁵ + c — generalized Mandelbrot family, degree 5.",
        build: build_multibrot5,
    },
    KnownFormula {
        name: "Lambda (logistic map)", category: "real dynamics",
        context: "c·z·(1−z) — the logistic map; population dynamics, period-doubling \
                   route to chaos, Lyapunov exponents. The strongest real physics/biology \
                   tie-in in this catalog.",
        build: build_lambda,
    },
    KnownFormula {
        name: "Sine Mandelbrot", category: "transcendental",
        context: "sin(z) + c — transcendental escape-time map.",
        build: build_sine,
    },
    KnownFormula {
        name: "Cosine Mandelbrot", category: "transcendental",
        context: "cos(z) + c — transcendental escape-time map.",
        build: build_cosine,
    },
    KnownFormula {
        name: "Exponential Mandelbrot", category: "transcendental",
        context: "exp(z) + c — transcendental escape-time map.",
        build: build_exponential,
    },
    KnownFormula {
        name: "Tanh Mandelbrot", category: "transcendental",
        context: "tanh(z) + c — transcendental escape-time map.",
        build: build_tanh,
    },
];

fn build_mandelbrot() -> Vec<OpNode> {          // z² + c
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let z2 = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, z2, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_tricorn() -> Vec<OpNode> {              // conj(z)² + c
    let mut b = ProgramBuilder::new();
    let z   = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c   = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let cj  = b.push(op::CONJ, z, 0, 0.0, 0.0).unwrap();
    let cj2 = b.push(op::SQR, cj, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, cj2, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_burning_ship() -> Vec<OpNode> {         // (|Re z|+i|Im z|)² + c
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let f  = b.push(op::ABSFOLD, z, 0, 0.0, 0.0).unwrap();
    let f2 = b.push(op::SQR, f, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, f2, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_burning_ship_cubic() -> Vec<OpNode> {   // (|Re z|+i|Im z|)³ + c
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let f  = b.push(op::ABSFOLD, z, 0, 0.0, 0.0).unwrap();
    let f3 = b.push(op::CUBE, f, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, f3, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_perpendicular_burning_ship() -> Vec<OpNode> {  // (Re z − i|Im z|)² + c
    let mut b = ProgramBuilder::new();
    let z    = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c    = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let ai   = b.push(op::ABSIM, z, 0, 0.0, 0.0).unwrap();   // (Re z, |Im z|)
    let cj   = b.push(op::CONJ, ai, 0, 0.0, 0.0).unwrap();   // (Re z, -|Im z|)
    let cj2  = b.push(op::SQR, cj, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, cj2, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_celtic() -> Vec<OpNode> {               // |Re(z²)| + i·Im(z²) + c
    let mut b = ProgramBuilder::new();
    let z   = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c   = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let z2  = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
    let ar  = b.push(op::ABSRE, z2, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, ar, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_perpendicular_mandelbrot() -> Vec<OpNode> {   // (Re z + i|Im z|)² + c
    let mut b = ProgramBuilder::new();
    let z   = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c   = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let ai  = b.push(op::ABSIM, z, 0, 0.0, 0.0).unwrap();
    let ai2 = b.push(op::SQR, ai, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, ai2, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_buffalo() -> Vec<OpNode> {              // |z|² − |z| + c
    let mut b = ProgramBuilder::new();
    let z   = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c   = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let f   = b.push(op::ABSFOLD, z, 0, 0.0, 0.0).unwrap();
    let f2  = b.push(op::SQR, f, 0, 0.0, 0.0).unwrap();
    let sub = b.push(op::SUB, f2, f, 0.0, 0.0).unwrap();
    b.push(op::ADD, sub, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_multibrot3() -> Vec<OpNode> {           // z³ + c
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let z3 = b.push(op::CUBE, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, z3, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_multibrot4() -> Vec<OpNode> {           // z⁴ + c
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let z4 = b.push(op::QUART, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, z4, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_multibrot5() -> Vec<OpNode> {           // z⁵ + c  (= z⁴·z + c)
    let mut b = ProgramBuilder::new();
    let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let z4 = b.push(op::QUART, z, 0, 0.0, 0.0).unwrap();
    let z5 = b.push(op::MUL, z4, z, 0.0, 0.0).unwrap();
    b.push(op::ADD, z5, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_lambda() -> Vec<OpNode> {               // c·z·(1−z) — logistic map
    let mut b = ProgramBuilder::new();
    let z    = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c    = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let one  = b.push(op::CONST, 0, 0, 1.0, 0.0).unwrap();
    let omz  = b.push(op::SUB, one, z, 0.0, 0.0).unwrap();   // 1 − z
    let zz1  = b.push(op::MUL, z, omz, 0.0, 0.0).unwrap();   // z(1−z)
    b.push(op::MUL, c, zz1, 0.0, 0.0).unwrap();              // c·z(1−z)
    b.into_nodes()
}

fn build_sine() -> Vec<OpNode> {                 // sin(z) + c
    let mut b = ProgramBuilder::new();
    let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let s = b.push(op::SIN, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, s, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_cosine() -> Vec<OpNode> {               // cos(z) + c
    let mut b = ProgramBuilder::new();
    let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let s = b.push(op::COS, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, s, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_exponential() -> Vec<OpNode> {          // exp(z) + c
    let mut b = ProgramBuilder::new();
    let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let e = b.push(op::EXP, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, e, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

fn build_tanh() -> Vec<OpNode> {                 // tanh(z) + c
    let mut b = ProgramBuilder::new();
    let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
    let c = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
    let t = b.push(op::TANH, z, 0, 0.0, 0.0).unwrap();
    b.push(op::ADD, t, c, 0.0, 0.0).unwrap();
    b.into_nodes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::eval_program;

    #[test]
    fn every_entry_builds_a_nonempty_finite_program() {
        for kf in LIBRARY {
            let prog = (kf.build)();
            assert!(!prog.is_empty(), "{} produced an empty program", kf.name);
            assert!(prog.len() <= crate::formula::N_SLOTS,
                "{} exceeds N_SLOTS: {} nodes", kf.name, prog.len());
            // Evaluate a handful of sample (z=0,c) points, one iteration —
            // must stay finite (no NaN/inf from a malformed builder).
            for &(cx, cy) in &[(0.0, 0.0), (0.3, 0.3), (-1.0, 0.0), (0.0, 1.0), (-0.5, 0.5)] {
                let (rx, ry) = eval_program(&prog, 0.0, 0.0, cx, cy);
                assert!(rx.is_finite() && ry.is_finite(),
                    "{} produced non-finite output at c=({cx},{cy}): ({rx},{ry})", kf.name);
            }
        }
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<&str> = LIBRARY.iter().map(|kf| kf.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate name in LIBRARY");
    }
}
