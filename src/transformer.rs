use burn::tensor::{Tensor, activation, backend::Backend};
use crate::genome::{TransformerWeights, LayerData};
use crate::cppn::{ActivationType, apply_complex_activations};
use crate::formula::N_BASIS;

// ── Types ────────────────────────────────────────────────────────────────────

/// (w_re [out,in], w_im [out,in], b_re [out], b_im [out])
pub type CLayer<B> = (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>);

/// Autodiff tensor representation of TransformerWeights.
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

// ── Scalar forward (per-genome) ───────────────────────────────────────────────

/// Run the transformer on the genome's latent vector → N_BASIS complex formula weights.
/// Called ONCE per genome, not per pixel.
/// Returns a Vec of N_BASIS (re, im) pairs — the formula weight for each basis function.
pub fn transformer_forward_latent(
    tw: &TransformerWeights,
    latent: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    let d = tw.w_q.in_size;

    // Project latent (d_model complex values) → two d_model tokens.
    // embed_z and embed_c are now d_model→d_model (different learned projections).
    let lat_re: Vec<f32> = latent.iter().map(|x| x.0).collect();
    let lat_im: Vec<f32> = latent.iter().map(|x| x.1).collect();
    let (z_re, z_im) = complex_matvec(&tw.embed_z, &lat_re, &lat_im);
    let (c_re, c_im) = complex_matvec(&tw.embed_c, &lat_re, &lat_im);

    // Q, K, V projections
    let (q_z_re, q_z_im) = complex_matvec(&tw.w_q, &z_re, &z_im);
    let (k_z_re, k_z_im) = complex_matvec(&tw.w_k, &z_re, &z_im);
    let (k_c_re, k_c_im) = complex_matvec(&tw.w_k, &c_re, &c_im);
    let (v_z_re, v_z_im) = complex_matvec(&tw.w_v, &z_re, &z_im);
    let (v_c_re, v_c_im) = complex_matvec(&tw.w_v, &c_re, &c_im);

    // Hermitian attention: Re(Q · K*) / sqrt(d)
    let scale = 1.0 / (d as f32).sqrt();
    let logit_zz = (dot(&q_z_re, &k_z_re) + dot(&q_z_im, &k_z_im)) * scale;
    let logit_zc = (dot(&q_z_re, &k_c_re) + dot(&q_z_im, &k_c_im)) * scale;
    let (attn_zz, attn_zc) = softmax2(logit_zz, logit_zc);

    let ctx_re: Vec<f32> = (0..d).map(|k| attn_zz * v_z_re[k] + attn_zc * v_c_re[k]).collect();
    let ctx_im: Vec<f32> = (0..d).map(|k| attn_zz * v_z_im[k] + attn_zc * v_c_im[k]).collect();

    // Output projection + residual
    let (ao_re, ao_im) = complex_matvec(&tw.w_o, &ctx_re, &ctx_im);
    let x_re: Vec<f32> = (0..d).map(|k| z_re[k] + ao_re[k]).collect();
    let x_im: Vec<f32> = (0..d).map(|k| z_im[k] + ao_im[k]).collect();

    // CPPN feed-forward
    let (pre_re, pre_im) = complex_matvec(&tw.ff1, &x_re, &x_im);
    let d_ff = tw.ff1.out_size;
    let mut act_re = vec![0.0f32; d_ff];
    let mut act_im = vec![0.0f32; d_ff];
    for k in 0..d_ff {
        let (ar, ai) = tw.ff_acts[k].apply_complex_scalar(pre_re[k], pre_im[k]);
        act_re[k] = ar;
        act_im[k] = ai;
    }
    let (ffo_re, ffo_im) = complex_matvec(&tw.ff2, &act_re, &act_im);
    let x_re: Vec<f32> = (0..d).map(|k| x_re[k] + ffo_re[k]).collect();
    let x_im: Vec<f32> = (0..d).map(|k| x_im[k] + ffo_im[k]).collect();

    // Output: d_model → N_BASIS complex weights, tanh-bounded to (-1,1) per component.
    // Prevents the weighted basis sum from immediately escaping the bailout radius.
    let (out_re, out_im) = complex_matvec(&tw.output, &x_re, &x_im);
    out_re.into_iter().zip(out_im).map(|(r, i)| (r.tanh(), i.tanh())).collect()
}

// ── Tensor forward (autodiff backprop) ───────────────────────────────────────

/// Differentiable transformer: latent tensors → formula weight tensors.
/// latent_re/latent_im: [1, d_model].
/// Returns (fw_re, fw_im): [1, N_BASIS].
pub fn transformer_forward_latent_tensor<B: Backend>(
    tt: &TransformerTensors<B>,
    ff_acts: &[ActivationType],
    latent_re: &Tensor<B, 2>,
    latent_im: &Tensor<B, 2>,
    d_model: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Project latent → two d_model tokens
    let (z_re, z_im) = complex_linear(latent_re, latent_im, &tt.embed_z);
    let (c_re, c_im) = complex_linear(latent_re, latent_im, &tt.embed_c);

    // Q, K, V
    let (q_z_re, q_z_im) = complex_linear(&z_re, &z_im, &tt.w_q);
    let (k_z_re, k_z_im) = complex_linear(&z_re, &z_im, &tt.w_k);
    let (k_c_re, k_c_im) = complex_linear(&c_re, &c_im, &tt.w_k);
    let (v_z_re, v_z_im) = complex_linear(&z_re, &z_im, &tt.w_v);
    let (v_c_re, v_c_im) = complex_linear(&c_re, &c_im, &tt.w_v);

    // Attention logits: Re(Q_z · K*) / sqrt(d) → [1, 1]
    let scale = 1.0 / (d_model as f32).sqrt();
    let logit_zz = (q_z_re.clone() * k_z_re + q_z_im.clone() * k_z_im)
        .sum_dim(1) * scale;
    let logit_zc = (q_z_re * k_c_re + q_z_im * k_c_im)
        .sum_dim(1) * scale;

    let logits  = Tensor::cat(vec![logit_zz, logit_zc], 1);
    let attn    = activation::softmax(logits, 1);
    let attn_zz = attn.clone().narrow(1, 0, 1);
    let attn_zc = attn.narrow(1, 1, 1);

    let ctx_re = attn_zz.clone() * v_z_re + attn_zc.clone() * v_c_re;
    let ctx_im = attn_zz * v_z_im + attn_zc * v_c_im;

    let (ao_re, ao_im) = complex_linear(&ctx_re, &ctx_im, &tt.w_o);
    let x_re = z_re.clone() + ao_re;
    let x_im = z_im.clone() + ao_im;

    let (pre_re, pre_im) = complex_linear(&x_re, &x_im, &tt.ff1);
    let (act_re, act_im) = apply_complex_activations(pre_re, pre_im, ff_acts);
    let (ffo_re, ffo_im) = complex_linear(&act_re, &act_im, &tt.ff2);
    let x_re = x_re + ffo_re;
    let x_im = x_im + ffo_im;

    // Output: d_model → N_BASIS, tanh-bounded to keep formula weights in (-1, 1).
    let (fw_re, fw_im) = complex_linear(&x_re, &x_im, &tt.output);
    (fw_re.tanh(), fw_im.tanh())
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Complex matrix-vector product: (W_re + i·W_im)(x_re + i·x_im) + b.
pub fn complex_matvec(layer: &LayerData, x_re: &[f32], x_im: &[f32]) -> (Vec<f32>, Vec<f32>) {
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
