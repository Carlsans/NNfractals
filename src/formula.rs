use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single complex-number transformation applied to z given context c.
/// Formulas are sequences of these ops, evolved by the GA.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ComplexOp {
    Square,           // z → z²                   (2-fold conj. symmetry)
    Cube,             // z → z³                   (3-fold rotational)
    Quart,            // z → z⁴                   (4-fold rotational)
    AddC,             // z → z + c                (Mandelbrot shift)
    SubC,             // z → z − c
    MulC,             // z → z × c                (complex product)
    ConjSquare,       // z → conj(z)²             (Tricorn/Mandelbar)
    AbsBoth,          // z → (|re|, |im|)          (Burning Ship fold)
    AbsRe,            // z → (|re|, im)
    AbsIm,            // z → (re, |im|)
    Recip,            // z → z̄ / (|z|² + ε)       (inversion map)
    Neg,              // z → −z
    Scale(f32),       // z → k·z
    Shift(f32, f32),  // z → z + (a + bi)
}

impl ComplexOp {
    fn discrete_pool() -> Vec<Self> {
        vec![
            Self::Square, Self::Cube, Self::Quart,
            Self::AddC, Self::SubC, Self::MulC,
            Self::ConjSquare, Self::AbsBoth, Self::AbsRe, Self::AbsIm,
            Self::Recip, Self::Neg,
        ]
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        let r: f32 = rng.random();
        if r < 0.80 {
            let pool = Self::discrete_pool();
            pool[rng.random_range(0..pool.len())].clone()
        } else if r < 0.90 {
            Self::Scale(rng.random::<f32>() * 2.0 - 1.0)
        } else {
            Self::Shift(rng.random::<f32>() * 1.0 - 0.5, rng.random::<f32>() * 1.0 - 0.5)
        }
    }

    /// Apply to scalar (f32) state.
    pub fn apply_scalar(&self, zx: f32, zy: f32, cx: f32, cy: f32) -> (f32, f32) {
        match self {
            Self::Square => (zx * zx - zy * zy, 2.0 * zx * zy),
            Self::Cube => {
                let (a, b) = (zx * zx, zy * zy);
                (zx * (a - 3.0 * b), zy * (3.0 * a - b))
            }
            Self::Quart => {
                let (a, b) = (zx * zx, zy * zy);
                (a * a - 6.0 * a * b + b * b, 4.0 * zx * zy * (a - b))
            }
            Self::AddC => (zx + cx, zy + cy),
            Self::SubC => (zx - cx, zy - cy),
            Self::MulC => (zx * cx - zy * cy, zx * cy + zy * cx),
            Self::ConjSquare => (zx * zx - zy * zy, -2.0 * zx * zy),
            Self::AbsBoth => (zx.abs(), zy.abs()),
            Self::AbsRe => (zx.abs(), zy),
            Self::AbsIm => (zx, zy.abs()),
            Self::Recip => { let m = zx * zx + zy * zy + 1e-8; (zx / m, -zy / m) }
            Self::Neg => (-zx, -zy),
            Self::Scale(k) => (k * zx, k * zy),
            Self::Shift(a, b) => (zx + a, zy + b),
        }
    }

    /// Apply to burn tensors [N, 1] for autodiff backprop.
    pub fn apply_tensor<B: Backend>(
        &self,
        zx: Tensor<B, 2>,
        zy: Tensor<B, 2>,
        cx: &Tensor<B, 2>,
        cy: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Self::Square => (
                zx.clone() * zx.clone() - zy.clone() * zy.clone(),
                zx * zy * 2.0,
            ),
            Self::Cube => {
                let a = zx.clone() * zx.clone();
                let b = zy.clone() * zy.clone();
                (zx.clone() * (a.clone() - b.clone() * 3.0), zy * (a * 3.0 - b))
            }
            Self::Quart => {
                let a = zx.clone() * zx.clone();
                let b = zy.clone() * zy.clone();
                (a.clone() * a.clone() - a.clone() * b.clone() * 6.0 + b.clone() * b.clone(),
                 zx * zy * (a - b) * 4.0)
            }
            Self::AddC => (zx + cx.clone(), zy + cy.clone()),
            Self::SubC => (zx - cx.clone(), zy - cy.clone()),
            Self::MulC => (
                zx.clone() * cx.clone() - zy.clone() * cy.clone(),
                zx * cy.clone() + zy * cx.clone(),
            ),
            Self::ConjSquare => (
                zx.clone() * zx.clone() - zy.clone() * zy.clone(),
                zx * zy * (-2.0),
            ),
            Self::AbsBoth => (zx.abs(), zy.abs()),
            Self::AbsRe => (zx.abs(), zy),
            Self::AbsIm => (zx, zy.abs()),
            Self::Recip => {
                let m = zx.clone() * zx.clone() + zy.clone() * zy.clone() + 1e-8_f32;
                (zx / m.clone(), zy.neg() / m)
            }
            Self::Neg => (zx.neg(), zy.neg()),
            Self::Scale(k) => (zx * *k, zy * *k),
            Self::Shift(a, b) => (zx + *a, zy + *b),
        }
    }

    /// Short label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Square     => "z²",
            Self::Cube       => "z³",
            Self::Quart      => "z⁴",
            Self::AddC       => "+c",
            Self::SubC       => "-c",
            Self::MulC       => "×c",
            Self::ConjSquare => "conj²",
            Self::AbsBoth    => "|z|",
            Self::AbsRe      => "|re|",
            Self::AbsIm      => "|im|",
            Self::Recip      => "1/z",
            Self::Neg        => "-z",
            Self::Scale(_)   => "scale",
            Self::Shift(..)  => "shift",
        }
    }
}

/// Apply a formula sequence to scalar (zx, zy).
pub fn eval_formula(formula: &[ComplexOp], zx: f32, zy: f32, cx: f32, cy: f32) -> (f32, f32) {
    let (mut rx, mut ry) = (zx, zy);
    for op in formula {
        (rx, ry) = op.apply_scalar(rx, ry, cx, cy);
    }
    (rx, ry)
}

/// Apply formula to burn tensors — used in backprop.
pub fn eval_formula_tensor<B: Backend>(
    formula: &[ComplexOp],
    zx: Tensor<B, 2>,
    zy: Tensor<B, 2>,
    cx: &Tensor<B, 2>,
    cy: &Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (mut rx, mut ry) = (zx, zy);
    for op in formula {
        (rx, ry) = op.apply_tensor(rx, ry, cx, cy);
    }
    (rx, ry)
}

/// Known-good starting formulas — biologically bootstraps the first generation
/// with formulas that are mathematically known to produce interesting fractals.
pub fn template_formulas() -> Vec<Vec<ComplexOp>> {
    vec![
        vec![ComplexOp::Square, ComplexOp::AddC],                                   // Mandelbrot
        vec![ComplexOp::Cube, ComplexOp::AddC],                                     // Cubic Mandelbrot
        vec![ComplexOp::Quart, ComplexOp::AddC],                                    // Quartic
        vec![ComplexOp::ConjSquare, ComplexOp::AddC],                               // Tricorn/Mandelbar
        vec![ComplexOp::AbsBoth, ComplexOp::Square, ComplexOp::AddC],               // Burning Ship
        vec![ComplexOp::AbsRe, ComplexOp::Square, ComplexOp::AddC],                 // Burning Ship variant
        vec![ComplexOp::AbsIm, ComplexOp::Square, ComplexOp::AddC],
        vec![ComplexOp::Square, ComplexOp::AddC, ComplexOp::Square, ComplexOp::AddC], // 2-step Mandelbrot
        vec![ComplexOp::Square, ComplexOp::MulC],                                   // z²c
        vec![ComplexOp::Square, ComplexOp::SubC],                                   // z²-c
        vec![ComplexOp::ConjSquare, ComplexOp::AbsBoth, ComplexOp::AddC],           // Tricorn+abs
        vec![ComplexOp::Cube, ComplexOp::SubC],                                     // z³-c (3-fold)
        vec![ComplexOp::Square, ComplexOp::Neg, ComplexOp::AddC],                   // -z²+c
        vec![ComplexOp::AbsBoth, ComplexOp::Cube, ComplexOp::AddC],                 // Cubic Burning Ship
        vec![ComplexOp::Square, ComplexOp::AddC, ComplexOp::AbsBoth],              // |z²+c|
        vec![ComplexOp::Square, ComplexOp::AddC, ComplexOp::MulC],                  // (z²+c)*c
    ]
}

/// Mutate a formula: point, insertion, deletion, or float nudge.
pub fn mutate_formula(formula: &mut Vec<ComplexOp>, rng: &mut impl Rng) {
    const MIN: usize = 2;
    const MAX: usize = 8;

    let r: f32 = rng.random();
    if r < 0.55 && !formula.is_empty() {
        // Point mutation
        let i = rng.random_range(0..formula.len());
        formula[i] = ComplexOp::random(rng);
    } else if r < 0.75 && formula.len() < MAX {
        // Insert a new op at a random position
        let i = rng.random_range(0..=formula.len());
        formula.insert(i, ComplexOp::random(rng));
    } else if formula.len() > MIN {
        // Delete
        let i = rng.random_range(0..formula.len());
        formula.remove(i);
    } else {
        // Fallback: mutate last op
        let last = formula.len().saturating_sub(1);
        formula[last] = ComplexOp::random(rng);
    }

    // Nudge float params
    for op in formula.iter_mut() {
        match op {
            ComplexOp::Scale(k) if rng.random::<f32>() < 0.25 => {
                *k = (*k + rng.random::<f32>() * 0.4 - 0.2).clamp(-2.0, 2.0);
            }
            ComplexOp::Shift(a, b) if rng.random::<f32>() < 0.25 => {
                *a += rng.random::<f32>() * 0.3 - 0.15;
                *b += rng.random::<f32>() * 0.3 - 0.15;
            }
            _ => {}
        }
    }
}

/// Single-point crossover of two formula sequences.
pub fn crossover_formulas(a: &[ComplexOp], b: &[ComplexOp], rng: &mut impl Rng) -> Vec<ComplexOp> {
    const MAX: usize = 8;
    if a.is_empty() { return b.to_vec(); }
    if b.is_empty() { return a.to_vec(); }
    let cut_a = rng.random_range(0..=a.len());
    let cut_b = rng.random_range(0..=b.len());
    let mut result: Vec<ComplexOp> =
        a[..cut_a].iter().chain(b[cut_b..].iter()).cloned().collect();
    result.truncate(MAX);
    if result.is_empty() { result = a.to_vec(); }
    result
}
