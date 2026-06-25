use burn::tensor::{Tensor, activation, backend::Backend};
use crate::genome::{TransformerWeights, LayerData};
use crate::cppn::{ActivationType, apply_complex_activations};

// ── Types ────────────────────────────────────────────────────────────────────

/// (w_re [out,in], w_im [out,in], b_re [out], b_im [out])
pub type CLayer<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>);

/// Autodiff tensor representation of TransformerWeights.
/// All tensors are built with require_grad() for backprop.
pub struct TransformerTensors<B: Backend> {
    pub embed_z: CLayer<B>,
    pub embed_c: CLayer<B>,
    pub w_q:     CLayer<B>,
    pub w_k:     CLayer<B>,
    pub w_v:     CLayer<B>,
    pub w_o:     CLayer<B>,
    pub ff1:     CLayer<B>,
    pub ff2:     CLayer<B>,
    pub output:  CLayer<B>,
}

// ── Scalar forward (CPU rendering, per pixel) ────────────────────────────────

/// One transformer forward pass on a single (z, c) pair.
/// Returns the NN perturbation as a complex scalar (re, im).
pub fn transformer_forward_scalar(
    tw: &TransformerWeights,
    zx: f32, zy: f32, cx: f32, cy: f32,
) -> (f32, f32) {
    let d = tw.embed_z.out_size;

    // 1. Embed z and c: complex scalar → d-dimensional complex vector
    let (z_re, z_im) = embed_scalar(&tw.embed_z, zx, zy);
    let (c_re, c_im) = embed_scalar(&tw.embed_c, cx, cy);

    // 2. Compute Q, K, V for both tokens (shared projection weights)
    // Only the z-query is used for attention (we only need the z-position output)
    let (q_z_re, q_z_im) = complex_matvec(&tw.w_q, &z_re, &z_im);
    let _ = complex_matvec(&tw.w_q, &c_re, &c_im); // c query unused; weight used for K/V symmetry
    let (k_z_re, k_z_im) = complex_matvec(&tw.w_k, &z_re, &z_im);
    let (k_c_re, k_c_im) = complex_matvec(&tw.w_k, &c_re, &c_im);
    let (v_z_re, v_z_im) = complex_matvec(&tw.w_v, &z_re, &z_im);
    let (v_c_re, v_c_im) = complex_matvec(&tw.w_v, &c_re, &c_im);

    // Attention logits: Re(Q · K*) / sqrt(d)
    //   Re(Q · K*) = Σ_k (Q_re[k]*K_re[k] + Q_im[k]*K_im[k])
    let scale = 1.0 / (d as f32).sqrt();
    let logit_zz = (dot(&q_z_re, &k_z_re) + dot(&q_z_im, &k_z_im)) * scale;
    let logit_zc = (dot(&q_z_re, &k_c_re) + dot(&q_z_im, &k_c_im)) * scale;
    let (attn_zz, attn_zc) = softmax2(logit_zz, logit_zc);

    // Context for z token: weighted sum of V vectors
    let ctx_z_re: Vec<f32> = (0..d).map(|k| attn_zz * v_z_re[k] + attn_zc * v_c_re[k]).collect();
    let ctx_z_im: Vec<f32> = (0..d).map(|k| attn_zz * v_z_im[k] + attn_zc * v_c_im[k]).collect();

    // Output projection + residual
    let (ao_re, ao_im) = complex_matvec(&tw.w_o, &ctx_z_re, &ctx_z_im);
    let x_re: Vec<f32> = (0..d).map(|k| z_re[k] + ao_re[k]).collect();
    let x_im: Vec<f32> = (0..d).map(|k| z_im[k] + ao_im[k]).collect();

    // 3. CPPN feed-forward (z token only — output comes from the z position)
    let (pre_re, pre_im) = complex_matvec(&tw.ff1, &x_re, &x_im);
    let d_ff = tw.ff1.out_size;
    let mut act_re = vec![0.0f32; d_ff];
    let mut act_im = vec![0.0f32; d_ff];
    for k in 0..d_ff {
        let (ar, ai) = tw.ff_acts[k].apply_complex_scalar(pre_re[k], pre_im[k]);
        act_re[k] = ar;
        act_im[k] = ai;
    }
    let (ff_out_re, ff_out_im) = complex_matvec(&tw.ff2, &act_re, &act_im);
    let x_re: Vec<f32> = (0..d).map(|k| x_re[k] + ff_out_re[k]).collect();
    let x_im: Vec<f32> = (0..d).map(|k| x_im[k] + ff_out_im[k]).collect();

    // 4. Output projection: d_model → 1 complex scalar
    let (out_re, out_im) = complex_matvec(&tw.output, &x_re, &x_im);
    (out_re[0], out_im[0])
}

// ── Tensor forward (autodiff backprop) ──────────────────────────────────────

/// Differentiable transformer forward.
/// z_re, z_im: [N, 1]; c_re, c_im: [N, 1]
/// Returns (nn_re, nn_im): [N, 1] — the complex NN perturbation.
pub fn transformer_forward_tensor<B: Backend>(
    tt: &TransformerTensors<B>,
    ff_acts: &[ActivationType],
    z_re: Tensor<B, 2>,
    z_im: Tensor<B, 2>,
    c_re: &Tensor<B, 2>,
    c_im: &Tensor<B, 2>,
    d_model: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // 1. Embed: [N, 1] → [N, d_model]
    let (ze_re, ze_im) = complex_linear(&z_re,        &z_im,        &tt.embed_z);
    let (ce_re, ce_im) = complex_linear(c_re,          c_im,          &tt.embed_c);

    // 2. Q, K, V projections for both tokens: [N, d_model]
    let (q_z_re, q_z_im) = complex_linear(&ze_re, &ze_im, &tt.w_q);
    let (_q_c_re, _q_c_im) = complex_linear(&ce_re, &ce_im, &tt.w_q);
    let (k_z_re, k_z_im) = complex_linear(&ze_re, &ze_im, &tt.w_k);
    let (k_c_re, k_c_im) = complex_linear(&ce_re, &ce_im, &tt.w_k);
    let (v_z_re, v_z_im) = complex_linear(&ze_re, &ze_im, &tt.w_v);
    let (v_c_re, v_c_im) = complex_linear(&ce_re, &ce_im, &tt.w_v);

    // Attention logits: Re(Q_z · K*) / sqrt(d_model) → [N, 1]
    let scale = 1.0 / (d_model as f32).sqrt();
    let logit_zz = (q_z_re.clone() * k_z_re.clone() + q_z_im.clone() * k_z_im.clone())
        .sum_dim(1) * scale;
    let logit_zc = (q_z_re * k_c_re + q_z_im * k_c_im)
        .sum_dim(1) * scale;

    // Softmax over the 2-token sequence for the z query: [N, 2] → [N, 2]
    let logits_z = Tensor::cat(vec![logit_zz, logit_zc], 1);
    let attn_z   = activation::softmax(logits_z, 1);
    let attn_zz  = attn_z.clone().narrow(1, 0, 1); // [N, 1]
    let attn_zc  = attn_z.narrow(1, 1, 1);         // [N, 1]

    // Context = weighted sum of V: [N, d_model]
    let ctx_z_re = attn_zz.clone() * v_z_re + attn_zc.clone() * v_c_re;
    let ctx_z_im = attn_zz          * v_z_im + attn_zc          * v_c_im;

    // Output projection + residual
    let (ao_re, ao_im) = complex_linear(&ctx_z_re, &ctx_z_im, &tt.w_o);
    let x_re = ze_re.clone() + ao_re;
    let x_im = ze_im.clone() + ao_im;

    // 3. CPPN feed-forward (z token)
    let (pre_re, pre_im) = complex_linear(&x_re, &x_im, &tt.ff1);
    let (act_re, act_im) = apply_complex_activations(pre_re, pre_im, ff_acts);
    let (ffo_re, ffo_im) = complex_linear(&act_re, &act_im, &tt.ff2);
    let x_re = x_re + ffo_re;
    let x_im = x_im + ffo_im;

    // 4. Output projection: d_model → 1
    complex_linear(&x_re, &x_im, &tt.output)
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Embed a single complex scalar (x + iy) into a d-dimensional complex vector.
/// Uses LayerData with in_size=1: weights[k] = W[k,0].
fn embed_scalar(layer: &LayerData, x: f32, y: f32) -> (Vec<f32>, Vec<f32>) {
    let d = layer.out_size;
    let mut re = vec![0.0f32; d];
    let mut im = vec![0.0f32; d];
    for k in 0..d {
        let wr = layer.weights[k];
        let wi = layer.weights_im.get(k).copied().unwrap_or(0.0);
        re[k] = wr * x - wi * y + layer.biases[k];
        im[k] = wr * y + wi * x + layer.biases_im.get(k).copied().unwrap_or(0.0);
    }
    (re, im)
}

/// Complex matrix-vector product: (W_re + i·W_im)(x_re + i·x_im) + b.
fn complex_matvec(layer: &LayerData, x_re: &[f32], x_im: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (d_out, d_in) = (layer.out_size, layer.in_size);
    let mut re = vec![0.0f32; d_out];
    let mut im = vec![0.0f32; d_out];
    for o in 0..d_out {
        let mut sr = layer.biases[o];
        let mut si = layer.biases_im.get(o).copied().unwrap_or(0.0);
        for i in 0..d_in {
            let wr = layer.weights[o * d_in + i];
            let wi = layer.weights_im.get(o * d_in + i).copied().unwrap_or(0.0);
            sr += wr * x_re[i] - wi * x_im[i];
            si += wr * x_im[i] + wi * x_re[i];
        }
        re[o] = sr;
        im[o] = si;
    }
    (re, im)
}

/// Batch complex linear: x_re/x_im [N, in] → out_re/out_im [N, out].
fn complex_linear<B: Backend>(
    x_re: &Tensor<B, 2>,
    x_im: &Tensor<B, 2>,
    (w_re, w_im, b_re, b_im): &CLayer<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let batch = x_re.dims()[0];
    let out   = b_re.dims()[0];
    let out_re = x_re.clone().matmul(w_re.clone().transpose())
        - x_im.clone().matmul(w_im.clone().transpose())
        + b_re.clone().unsqueeze::<2>().expand([batch, out]);
    let out_im = x_re.clone().matmul(w_im.clone().transpose())
        + x_im.clone().matmul(w_re.clone().transpose())
        + b_im.clone().unsqueeze::<2>().expand([batch, out]);
    (out_re, out_im)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m  = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let s  = ea + eb;
    (ea / s, eb / s)
}
