use burn::tensor::{Tensor, activation, backend::Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    Sin, Cos, Tanh, Square, Identity, Sigmoid,
    Gauss, Atan, Fract, Swish, Sinc, Tent,
    // Complex-native
    Conj,    // (re, −im) — conjugate
    AbsRe,   // (|re|, im) — Burning Ship fold on real part
    AbsIm,   // (re, |im|) — Burning Ship fold on imag part
    AbsBoth, // (|re|, |im|) — full Burning Ship fold
    Abs,     // (√(re²+im²), 0) — modulus, collapses to real axis
}

impl ActivationType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "sin"      => Some(Self::Sin),
            "cos"      => Some(Self::Cos),
            "tanh"     => Some(Self::Tanh),
            "square"   => Some(Self::Square),
            "identity" => Some(Self::Identity),
            "sigmoid"  => Some(Self::Sigmoid),
            "gauss"    => Some(Self::Gauss),
            "atan"     => Some(Self::Atan),
            "fract"    => Some(Self::Fract),
            "swish"    => Some(Self::Swish),
            "sinc"     => Some(Self::Sinc),
            "tent"     => Some(Self::Tent),
            "conj"     => Some(Self::Conj),
            "abs_re"   => Some(Self::AbsRe),
            "abs_im"   => Some(Self::AbsIm),
            "abs_both" => Some(Self::AbsBoth),
            "abs"      => Some(Self::Abs),
            _          => None,
        }
    }

    pub fn all_from_config(pool: &[String]) -> Vec<Self> {
        pool.iter().filter_map(|s| Self::from_str(s)).collect()
    }

    /// Real-valued scalar (legacy fallback).
    pub fn apply_scalar(&self, v: f32) -> f32 {
        match self {
            Self::Sin                                         => v.sin(),
            Self::Cos                                         => v.cos(),
            Self::Tanh                                        => v.tanh(),
            Self::Abs | Self::AbsRe | Self::AbsIm |
            Self::AbsBoth                                     => v.abs(),
            Self::Square                                      => v * v,
            Self::Identity | Self::Conj                       => v,
            Self::Sigmoid                                     => 1.0 / (1.0 + (-v).exp()),
            Self::Gauss                                       => (-v * v).exp(),
            Self::Atan                                        => v.atan() * std::f32::consts::FRAC_2_PI,
            Self::Fract                                       => v - v.floor(),
            Self::Swish                                       => v / (1.0 + (-v).exp()),
            Self::Sinc                                        => {
                let px = std::f32::consts::PI * v;
                if px.abs() < 1e-6 { 1.0 } else { px.sin() / px }
            }
            Self::Tent                                        => {
                let f = v - v.floor();
                1.0 - (2.0 * f - 1.0).abs()
            }
        }
    }

    /// Complex-valued activation: (re, im) → (re′, im′).
    pub fn apply_complex_scalar(&self, re: f32, im: f32) -> (f32, f32) {
        match self {
            // True complex operations
            Self::Square  => (re * re - im * im, 2.0 * re * im),
            Self::Conj    => (re, -im),
            Self::AbsRe   => (re.abs(), im),
            Self::AbsIm   => (re, im.abs()),
            Self::AbsBoth => (re.abs(), im.abs()),
            Self::Abs     => ((re * re + im * im).sqrt(), 0.0),
            Self::Gauss   => ((-re * re - im * im).exp(), 0.0),
            // Component-wise: apply scalar function to each part independently
            _ => (self.apply_scalar(re), self.apply_scalar(im)),
        }
    }
}

// ── Tensor helpers ───────────────────────────────────────────────────────────

/// Component-wise scalar activation on a [N, 1] tensor.
fn cw_tensor<B: Backend>(act: &ActivationType, col: Tensor<B, 2>) -> Tensor<B, 2> {
    match act {
        ActivationType::Sin      => col.sin(),
        ActivationType::Cos      => col.cos(),
        ActivationType::Tanh     => col.tanh(),
        ActivationType::Identity | ActivationType::Conj => col,
        ActivationType::Sigmoid  => activation::sigmoid(col),
        ActivationType::Gauss    => (col.clone() * col).neg().exp(),
        ActivationType::Atan     => col.atan() * std::f32::consts::FRAC_2_PI,
        ActivationType::Fract    => col.clone() - col.floor(),
        ActivationType::Swish    => col.clone() * activation::sigmoid(col),
        ActivationType::Sinc     => {
            let px = col.clone() * std::f32::consts::PI;
            let sinpx = px.clone().sin();
            let safe = px.abs().clamp(1e-6, f32::MAX);
            sinpx / safe
        }
        ActivationType::Tent     => {
            let f = col.clone() - col.floor();
            (f * 2.0 - 1.0).abs().neg() + 1.0
        }
        _ => col, // Square / Abs variants handled by caller
    }
}

/// Complex activation on single-neuron [N, 1] column tensors.
fn complex_col<B: Backend>(
    act: &ActivationType,
    re: Tensor<B, 2>,
    im: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    match act {
        ActivationType::Square  => (
            re.clone() * re.clone() - im.clone() * im.clone(),
            re * im * 2.0,
        ),
        ActivationType::Conj    => (re, im.neg()),
        ActivationType::AbsRe   => (re.abs(), im),
        ActivationType::AbsIm   => (re, im.abs()),
        ActivationType::AbsBoth => (re.abs(), im.abs()),
        ActivationType::Abs     => {
            let mag = (re.clone() * re + im.clone() * im.clone()).sqrt();
            (mag, im * 0.0_f32)
        }
        ActivationType::Gauss   => {
            let mag2 = re.clone() * re + im.clone() * im.clone();
            (mag2.neg().exp(), im * 0.0_f32)
        }
        _ => (cw_tensor(act, re), cw_tensor(act, im)),
    }
}

/// Apply per-neuron complex activations to full [N, neurons] re/im tensors.
pub fn apply_complex_activations<B: Backend>(
    x_re: Tensor<B, 2>,
    x_im: Tensor<B, 2>,
    activations: &[ActivationType],
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut re_cols = Vec::with_capacity(activations.len());
    let mut im_cols = Vec::with_capacity(activations.len());
    for (i, act) in activations.iter().enumerate() {
        let (or, oi) = complex_col(act, x_re.clone().narrow(1, i, 1), x_im.clone().narrow(1, i, 1));
        re_cols.push(or);
        im_cols.push(oi);
    }
    (Tensor::cat(re_cols, 1), Tensor::cat(im_cols, 1))
}

/// Autodiff-capable complex CPPN forward for backprop.
/// layers: Vec of (w_re [out,in], w_im [out,in], b_re [out], b_im [out])
/// Input: z = (z_re[N,1], z_im[N,1]), c = (c_re[N,1], c_im[N,1])
/// Output: (nn_re[N,1], nn_im[N,1])
pub fn cppn_forward_complex_tensor<B: Backend>(
    layers: &[(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>)],
    activations: &[Vec<ActivationType>],
    z_re: Tensor<B, 2>,
    z_im: Tensor<B, 2>,
    c_re: &Tensor<B, 2>,
    c_im: &Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // 2 complex inputs: [z, c] → [N, 2] re/im
    let mut x_re = Tensor::cat(vec![z_re, c_re.clone()], 1);
    let mut x_im = Tensor::cat(vec![z_im, c_im.clone()], 1);

    for ((w_re, w_im, b_re, b_im), acts) in layers.iter().zip(activations.iter()) {
        let batch = x_re.dims()[0];
        let out = acts.len();
        // Complex matmul: (W_re + i·W_im)(x_re + i·x_im) = W_re·x_re - W_im·x_im + i(W_re·x_im + W_im·x_re)
        let pre_re = x_re.clone().matmul(w_re.clone().transpose())
            - x_im.clone().matmul(w_im.clone().transpose())
            + b_re.clone().unsqueeze::<2>().expand([batch, out]);
        let pre_im = x_re.matmul(w_im.clone().transpose())
            + x_im.matmul(w_re.clone().transpose())
            + b_im.clone().unsqueeze::<2>().expand([batch, out]);
        (x_re, x_im) = apply_complex_activations(pre_re, pre_im, acts);
    }
    (x_re, x_im)  // [N, 1]
}
