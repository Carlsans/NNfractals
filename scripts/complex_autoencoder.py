#!/usr/bin/env python3
"""Complex-valued convolutional autoencoder for the bailout z value
(real + imaginary part at the moment of escape — see
`src/fractal.rs::dag_escape_pixel_z` / `explorer complex-export`), NOT the
escape-time tensor `autoencoder.py`'s AutoEncoder trains on. Carl's own
feasibility check (visual inspection of real/imag/magnitude channels
across several genomes, both Mandelbrot-family and GA-discovered) found
genuine, formula-independent structure the escape-time signal alone
discards — this is the follow-up: an encoder/decoder that treats z as an
actual complex number instead of two unrelated real channels.

A complex number isn't just "two floats" to a network — a real Conv2d
run separately on `zx`/`zy` (or concatenated as 2 channels) can learn
ANY relationship between them, including none; a genuinely complex layer
is constrained to the specific linear structure complex multiplication
imposes (rotation + scaling), and non-holomorphic activations (CReLU here)
are needed because Liouville's theorem rules out any bounded holomorphic
one. Whether that constraint helps HERE is exactly the open question this
file exists to test — this is deliberately kept to the plain, well-
established building blocks (see e.g. "Deep Complex Networks", Trabelsi
et al. 2018) rather than anything exotic.

Represents a complex tensor as a `(real, imag)` pair of ordinary real
tensors throughout (matching the common `complexPyTorch`-style convention)
rather than PyTorch's native `complex64` dtype — native complex support is
solid for basic ops but batch norm / some interpolation modes are not
guaranteed available across versions, and the explicit-pair form makes
every operation's real/imaginary bookkeeping visible rather than implicit.

Plain (non-variational) autoencoder only, v1: a complex-domain
reparameterization trick (what does "sample a complex Gaussian" mean for
THIS bottleneck, and how does the KL term generalize?) is a real, separate
design question — not something to improvise by analogy with the real
VAE. `scripts/autoencoder.py`'s VAE stays the reference implementation for
now; this file can grow a variational bottleneck later if the plain AE
proves the representation is worth it.

Mirrors `Encoder512`/`Decoder512`'s exact 6-stage schedule (512->8,
channels 1->256) for direct comparability against the real-valued
baseline trained on the same corpus's escape-time tensor — same depth,
same channel counts, only the arithmetic (and therefore roughly 2x the
real parameters, one set for the real part of each weight and one for the
imaginary part) differs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 512
FEAT_GRID = 8
FEAT_CH = 256
# 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8
_CHANNELS = [1, 8, 16, 32, 64, 128, 256]


class ComplexConv2d(nn.Module):
    """`(W_r + iW_i)(x_r + ix_i) = (W_r x_r - W_i x_i) + i(W_r x_i + W_i x_r)`
    — two independent real convolutions, combined per the standard complex
    multiplication identity. Bias folded into each real conv (a complex
    bias is just `b_r + i b_i`, one real bias per part)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        re, im = x
        return (self.conv_re(re) - self.conv_im(im), self.conv_re(im) + self.conv_im(re))


class ComplexConvTranspose2d(nn.Module):
    """Decoder-side sibling of `ComplexConv2d` — same identity, transposed conv."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_re = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        re, im = x
        return (self.conv_re(re) - self.conv_im(im), self.conv_re(im) + self.conv_im(re))


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_re = nn.Linear(in_features, out_features)
        self.lin_im = nn.Linear(in_features, out_features)

    def forward(self, x):
        re, im = x
        return (self.lin_re(re) - self.lin_im(im), self.lin_re(im) + self.lin_im(re))


class ComplexBatchNorm2d(nn.Module):
    """NAIVE complex batch norm: independent real `BatchNorm2d` on the real
    and imaginary parts. The "correct" version (Trabelsi et al.) whitens
    re/im JOINTLY via the 2x2 covariance matrix `[[Vrr,Vri],[Vri,Vii]]`'s
    inverse square root — deliberately not implemented here (real added
    complexity, and it's a refinement, not a correctness requirement: this
    naive version still normalizes each part to zero mean/unit variance,
    it just doesn't decorrelate re from im). A reasonable place to upgrade
    from if this model's results justify the extra engineering.

    SUPERSEDED as the default norm layer (see `ComplexGroupNorm`) — kept
    for comparison. `BatchNorm2d`'s statistics come from the CURRENT
    batch during training but the ACCUMULATED running mean/var at eval
    time; with a small batch (4-16, all this project's per-genome corpora
    can support) those per-batch statistics are noisy and the running
    average takes many steps to settle, which plausibly explains why the
    first real training runs converged in loss but still reconstructed
    close to flat/blank on held-out data — a train/eval statistics
    mismatch, not a fundamental problem with the complex arithmetic."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features)
        self.bn_im = nn.BatchNorm2d(num_features)

    def forward(self, x):
        re, im = x
        return (self.bn_re(re), self.bn_im(im))


class ComplexGroupNorm(nn.Module):
    """Independent real `GroupNorm` on each part — like `ComplexBatchNorm2d`
    but normalizes within each SAMPLE (across a group of channels), not
    across the batch, so it has no batch-size dependence and no train/eval
    statistics mismatch at all (there's no running average to accumulate
    or get out of sync). This is now the DEFAULT norm layer in
    `_complex_down`/`_complex_up` — swapped in after the naive
    `ComplexBatchNorm2d` version showed real signs of a small-batch
    instability problem (loss converged but reconstructions stayed close
    to flat/blank even on TRAINING data seen 150 times, which a
    correctly-tuned autoencoder should be able to memorize)."""
    def __init__(self, num_features, num_groups=4):
        super().__init__()
        self.gn_re = nn.GroupNorm(num_groups, num_features)
        self.gn_im = nn.GroupNorm(num_groups, num_features)

    def forward(self, x):
        re, im = x
        return (self.gn_re(re), self.gn_im(im))


class ComplexBatchNormWhitened(nn.Module):
    """The PROPER complex batch norm (Trabelsi et al. 2018, "Deep Complex
    Networks") — whitens `(re, im)` JOINTLY via the inverse square root of
    their 2x2 covariance matrix `[[Vrr,Vri],[Vri,Vii]]`, so it actually
    DECORRELATES re from im (unlike `ComplexGroupNorm`'s independent
    per-part normalization, which zero-means/unit-variances each part but
    leaves any correlation between them untouched). Statistics reduced
    over batch+spatial dims per channel, with a running-average buffer for
    eval mode — same convention `nn.BatchNorm2d` uses, so it inherits that
    version's small-batch/train-eval-mismatch risk in exchange for the
    theoretically more correct whitening; worth comparing against
    `ComplexGroupNorm` empirically rather than assuming "more correct on
    paper" wins in practice at this project's batch sizes.

    Learnable affine step is a full 2x2 matrix (`gamma_rr/ri/ii`) plus a
    per-part shift (`beta_r/i`), not just two independent scalars — the
    literature's standard choice, since a diagonal-only affine after joint
    whitening would partially undo the decorrelation."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # gamma init: Vrr=Vii=1/sqrt(2), Vri=0 reproduces the standard
        # "unit variance circular complex" initialization from the paper.
        self.gamma_rr = nn.Parameter(torch.full((num_features,), 1.0 / 2 ** 0.5))
        self.gamma_ii = nn.Parameter(torch.full((num_features,), 1.0 / 2 ** 0.5))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean_r", torch.zeros(num_features))
        self.register_buffer("running_mean_i", torch.zeros(num_features))
        self.register_buffer("running_vrr", torch.full((num_features,), 0.5))
        self.register_buffer("running_vii", torch.full((num_features,), 0.5))
        self.register_buffer("running_vri", torch.zeros(num_features))

    def forward(self, x):
        re, im = x
        if self.training:
            mean_r = re.mean(dim=(0, 2, 3))
            mean_i = im.mean(dim=(0, 2, 3))
            with torch.no_grad():
                self.running_mean_r.lerp_(mean_r, self.momentum)
                self.running_mean_i.lerp_(mean_i, self.momentum)
        else:
            mean_r, mean_i = self.running_mean_r, self.running_mean_i

        cr = re - mean_r.view(1, -1, 1, 1)
        ci = im - mean_i.view(1, -1, 1, 1)

        if self.training:
            vrr = (cr * cr).mean(dim=(0, 2, 3)) + self.eps
            vii = (ci * ci).mean(dim=(0, 2, 3)) + self.eps
            vri = (cr * ci).mean(dim=(0, 2, 3))
            with torch.no_grad():
                self.running_vrr.lerp_(vrr, self.momentum)
                self.running_vii.lerp_(vii, self.momentum)
                self.running_vri.lerp_(vri, self.momentum)
        else:
            vrr, vii, vri = self.running_vrr, self.running_vii, self.running_vri

        # Inverse square root of [[vrr,vri],[vri,vii]] (closed form for a
        # 2x2 symmetric positive-definite matrix — Trabelsi et al. eq. 9).
        delta = (vrr * vii - vri * vri).clamp(min=self.eps)
        s = delta.sqrt()
        t = (vrr + vii + 2 * s).clamp(min=self.eps).sqrt()
        rst = 1.0 / (s * t)
        wrr = ((vii + s) * rst).view(1, -1, 1, 1)
        wii = ((vrr + s) * rst).view(1, -1, 1, 1)
        wri = ((-vri) * rst).view(1, -1, 1, 1)

        xr = wrr * cr + wri * ci
        xi = wri * cr + wii * ci

        gamma_rr = self.gamma_rr.view(1, -1, 1, 1)
        gamma_ii = self.gamma_ii.view(1, -1, 1, 1)
        gamma_ri = self.gamma_ri.view(1, -1, 1, 1)
        out_r = gamma_rr * xr + gamma_ri * xi + self.beta_r.view(1, -1, 1, 1)
        out_i = gamma_ri * xr + gamma_ii * xi + self.beta_i.view(1, -1, 1, 1)
        return (out_r, out_i)


class CReLU(nn.Module):
    """ReLU applied independently to real and imaginary parts. The simplest
    valid non-holomorphic complex activation (Liouville's theorem rules out
    any bounded holomorphic one).

    SUPERSEDED as the default activation (see `ModReLU`) — kept for
    comparison. A control experiment (the SAME data, at the SAME depth,
    trained with `autoencoder.py`'s proven real-valued layers instead of
    this file's complex ones) reconstructed real structure fine; the
    complex model collapsed to near-flat even after `ComplexGroupNorm`
    ruled out batch-statistics instability. CReLU zeroes re and im
    INDEPENDENTLY — whenever exactly one part is negative, the surviving
    part's phase relationship to the (now-zeroed) other part is destroyed,
    not just its magnitude reduced. Compounded across 6 layers this can
    plausibly wreck the phase information the whole point of a COMPLEX
    (not just 2-channel-real) network is to preserve."""
    def forward(self, x):
        re, im = x
        return (F.relu(re), F.relu(im))


class ModReLU(nn.Module):
    """`ReLU(|z| + b) * z/|z|` — thresholds MAGNITUDE only, via a learnable
    per-channel bias `b`, and rescales `(re, im)` by the same factor,
    preserving their ratio (phase) exactly. Now the default activation:
    unlike `CReLU`, it can never zero one part while leaving the other
    part's magnitude untouched, so it can't independently corrupt the
    re/im relationship the way `CReLU` can."""
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        re, im = x
        mag = torch.sqrt(re * re + im * im + 1e-8)
        scale = F.relu(mag + self.bias) / (mag + 1e-8)
        return (re * scale, im * scale)


class ComplexSigmoid(nn.Module):
    """Output-layer squash, independent per part — mirrors `Decoder512`'s
    real `nn.Sigmoid()`, matching this project's `[0,1]`-normalized PNG
    convention for both the real and imaginary channels (see
    `io::save_complex_channels`'s `norm_signed`/`norm_mag`)."""
    def forward(self, x):
        re, im = x
        return (torch.sigmoid(re), torch.sigmoid(im))


_NORM_LAYERS = {"groupnorm": ComplexGroupNorm, "whitened": ComplexBatchNormWhitened}


def _complex_down(in_ch, out_ch, norm="groupnorm"):
    return nn.Sequential(
        ComplexConv2d(in_ch, out_ch, 4, 2, 1), _NORM_LAYERS[norm](out_ch), ModReLU(out_ch),
    )


def _complex_up(in_ch, out_ch, norm="groupnorm"):
    return nn.Sequential(
        ComplexConvTranspose2d(in_ch, out_ch, 4, 2, 1), _NORM_LAYERS[norm](out_ch), ModReLU(out_ch),
    )


class ComplexResBlock(nn.Module):
    """Mirrors `autoencoder.py`'s `ResBlock` exactly (2x(conv3x3+norm)+skip,
    activation after the skip-add, kept small on purpose) — complex-valued
    throughout. One placed before each downsample/upsample stage, same
    "isolate whether residual depth helps independent of the bottleneck
    choice" motivation as the real `ResEncoder`."""
    def __init__(self, ch, norm="groupnorm"):
        super().__init__()
        self.conv1 = ComplexConv2d(ch, ch, 3, 1, 1)
        self.norm1 = _NORM_LAYERS[norm](ch)
        self.conv2 = ComplexConv2d(ch, ch, 3, 1, 1)
        self.norm2 = _NORM_LAYERS[norm](ch)
        self.act = ModReLU(ch)

    def forward(self, x):
        re, im = x
        h = self.act(self.norm1(self.conv1((re, im))))
        h_re, h_im = self.norm2(self.conv2(h))
        return self.act((re + h_re, im + h_im))


class ComplexEncoder(nn.Module):
    """`variational=True`: the FC head outputs a `2*latent_dim`-wide
    complex vector `(re, im)`, each half chunked into `(mu, logvar)` —
    i.e. `mu_re/logvar_re` from the real half, `mu_im/logvar_im` from the
    imaginary half. This treats the complex latent as two INDEPENDENT
    real Gaussian latents (one per part), not a joint complex-Gaussian
    with off-diagonal covariance — a deliberate simplification (see
    `ComplexAutoEncoder`'s docstring) that reduces to exactly the same,
    already-proven real VAE reparameterization/KL math applied twice,
    rather than inventing new complex-Gaussian sampling theory."""
    def __init__(self, latent_dim, in_ch=1, norm="groupnorm", residual=False, variational=False):
        super().__init__()
        self.variational = variational
        chs = [in_ch] + _CHANNELS[1:]
        stages = []
        for i in range(len(chs) - 1):
            if residual and i > 0:  # skip before the very first (in_ch=1) stage — too few channels for a useful residual block
                stages.append(ComplexResBlock(chs[i], norm))
            stages.append(_complex_down(chs[i], chs[i + 1], norm))
        self.conv = nn.Sequential(*stages)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = ComplexLinear(FEAT_CH * FEAT_GRID * FEAT_GRID, out_dim)

    def features(self, x):
        return self.conv(x)

    def forward(self, x):
        re, im = self.features(x)
        re, im = re.flatten(1), im.flatten(1)
        h_re, h_im = self.fc((re, im))
        if self.variational:
            mu_re, logvar_re = h_re.chunk(2, dim=-1)
            mu_im, logvar_im = h_im.chunk(2, dim=-1)
            return mu_re, logvar_re, mu_im, logvar_im
        return h_re, h_im


class ComplexDecoder(nn.Module):
    def __init__(self, latent_dim, out_ch=1, norm="groupnorm", residual=False):
        super().__init__()
        self.fc = ComplexLinear(latent_dim, FEAT_CH * FEAT_GRID * FEAT_GRID)
        chs = list(reversed(_CHANNELS[1:])) + [out_ch]
        stages = []
        for i in range(len(chs) - 2):
            if residual:
                stages.append(ComplexResBlock(chs[i], norm))
            stages.append(_complex_up(chs[i], chs[i + 1], norm))
        self.deconv = nn.Sequential(*stages)
        self.out = nn.Sequential(ComplexConvTranspose2d(chs[-2], chs[-1], 4, 2, 1), ComplexSigmoid())

    def forward(self, z):
        re, im = self.fc(z)
        re = re.view(-1, FEAT_CH, FEAT_GRID, FEAT_GRID)
        im = im.view(-1, FEAT_CH, FEAT_GRID, FEAT_GRID)
        return self.out(self.deconv((re, im)))


class ComplexAutoEncoder(nn.Module):
    """`in_ch`/`out_ch` are complex channel counts (1 for the bailout z
    value alone). `norm="groupnorm"` (default, the proven-working choice
    from Phase 2 — see project-complex-autoencoder memory) or `"whitened"`
    (the proper covariance-whitening `ComplexBatchNormWhitened`, an open
    comparison, see project-complex-nn-weekend-research memory item #3).
    `residual=True` adds a `ComplexResBlock` before each down/up-sample
    stage (item #5). `variational=True` (item #6) — see `ComplexEncoder`'s
    docstring for the reparameterization design: two INDEPENDENT real
    Gaussian latents (real part, imaginary part), each using the exact
    same reparameterization/KL math `autoencoder.py`'s own real VAE
    already uses, deliberately NOT a joint complex-Gaussian with
    off-diagonal covariance — a conservative, well-understood reduction
    rather than new probability theory."""
    def __init__(self, latent_dim=256, in_ch=1, norm="groupnorm", residual=False, variational=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_ch = in_ch
        self.res = RES
        self.norm = norm
        self.residual = residual
        self.variational = variational
        self.encoder = ComplexEncoder(latent_dim, in_ch=in_ch, norm=norm, residual=residual, variational=variational)
        self.decoder = ComplexDecoder(latent_dim, out_ch=in_ch, norm=norm, residual=residual)

    def encode(self, x):
        """The embedding used downstream: `(mu_re, mu_im)` for a VAE (the
        distribution's mean, not a noisy sample — mirrors `autoencoder.py`'s
        own `AutoEncoder.encode` convention exactly), the bottleneck
        vector directly for a plain AE."""
        if self.variational:
            mu_re, _, mu_im, _ = self.encoder(x)
            return mu_re, mu_im
        return self.encoder(x)

    def reconstruct_deterministic(self, x):
        """Mirrors `autoencoder.py`'s method of the same name — bypasses
        the stochastic reparameterization sample so a reconstruction-error
        RANKING (e.g. `score_complex_corpus.py`) doesn't flip between
        identical calls due to sampling noise."""
        return self.decoder(self.encode(x))

    def forward(self, x):
        if self.variational:
            mu_re, logvar_re, mu_im, logvar_im = self.encoder(x)
            # Same clamp-before-exp() stabilization as autoencoder.py's
            # real VAE, for the same reason (an unclamped logvar can blow
            # up within a handful of steps) — applied to each part.
            logvar_re = logvar_re.clamp(-10.0, 10.0)
            logvar_im = logvar_im.clamp(-10.0, 10.0)
            std_re = torch.exp(0.5 * logvar_re)
            std_im = torch.exp(0.5 * logvar_im)
            z_re = mu_re + std_re * torch.randn_like(std_re)
            z_im = mu_im + std_im * torch.randn_like(std_im)
            recon = self.decoder((z_re, z_im))
            return recon, mu_re, logvar_re, mu_im, logvar_im
        z = self.encoder(x)
        return self.decoder(z), None, None, None, None


def complex_mse_loss(recon, target):
    """Mean squared MAGNITUDE of the complex error — `|recon - target|^2`,
    the standard convention for a complex reconstruction loss (see the
    Anthropic /btw research this session pulled: MRI/radar complex
    autoencoders use this or the equivalent split-MSE form, which this is
    algebraically identical to)."""
    recon_re, recon_im = recon
    target_re, target_im = target
    return F.mse_loss(recon_re, target_re) + F.mse_loss(recon_im, target_im)


def complex_kl_loss(mu_re, logvar_re, mu_im, logvar_im):
    """KL divergence for the two-independent-real-Gaussians complex
    latent (see `ComplexEncoder`'s docstring) — exactly `autoencoder.py`'s
    own real-VAE KL formula, applied once per part and summed. Not a
    joint complex-Gaussian KL (which would need an off-diagonal
    covariance term) — a deliberate, conservative simplification."""
    kl_re = -0.5 * torch.mean(1 + logvar_re - mu_re.pow(2) - logvar_re.exp())
    kl_im = -0.5 * torch.mean(1 + logvar_im - mu_im.pow(2) - logvar_im.exp())
    return kl_re + kl_im


def complex_triplet_loss(anchor_mu_re, anchor_mu_im, positive_mu_re, positive_mu_im, margin=1.0):
    """Triplet-margin loss over the concatenated `(mu_re, mu_im)` embedding
    — the same latent representation `select_diverse_latent.py`/
    `cluster_latent.py` already use downstream, so this directly optimizes
    for what THOSE tools need: parameter-adjacent zones (the free,
    self-supervised positive-pair signal — see `compute_positive_pairs` in
    `train_complex_autoencoder.py`, 2026-08-08) pulled together, everything
    else pushed apart. Negative is `torch.roll(anchor, 1)` — a different
    anchor from the SAME batch, cheap in-batch negative sampling with no
    explicit mining step, relying on the DataLoader's per-epoch shuffle to
    vary which anchor lands next to which.

    L2-normalizes each embedding to the unit hypersphere before computing
    distance (standard practice, e.g. FaceNet's original triplet loss) —
    WITHOUT this, a first attempt (2026-08-08) blew up: on raw/unnormalized
    embeddings the margin objective has a degenerate solution (scale every
    embedding's magnitude up — this drives `d_pos - d_neg` toward the
    already-favored sign without learning any real structure), and
    reconstruction quality visibly degraded as the encoder chased it.
    Normalized, squared distance is bounded to `[0,4]`, so `margin=1.0` is
    a sane default and the scale-up escape hatch no longer exists."""
    anchor = F.normalize(torch.cat([anchor_mu_re, anchor_mu_im], dim=1), dim=1)
    positive = F.normalize(torch.cat([positive_mu_re, positive_mu_im], dim=1), dim=1)
    negative = torch.roll(anchor, shifts=1, dims=0)
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()


def multiscale_complex_mse_loss(recon, target, scales=(1, 2, 4, 8)):
    """Plain per-pixel MSE is well known to bias toward blur — averaging
    over many equally-plausible high-frequency completions minimizes
    pixel-wise error better than committing to any one of them sharply.
    This sums `complex_mse_loss` at the full resolution AND at several
    average-pooled (downsampled) resolutions, so the loss also rewards
    getting COARSE structure/shape right, not just fine pixel values —
    a standard, simple multi-scale trick (no pretrained perceptual model
    needed, so it applies to a complex-valued signal exactly as easily as
    a real one — perceptual losses via ImageNet-pretrained features don't
    have an obvious complex-domain analog, avoided for that reason).
    Every scale weighted equally; `scales=(1,2,4,8)` costs relatively
    little extra compute since each downsample shrinks the tensor 4x
    fewer elements than the last."""
    recon_re, recon_im = recon
    target_re, target_im = target
    total = 0.0
    for s in scales:
        if s == 1:
            r_re, r_im, t_re, t_im = recon_re, recon_im, target_re, target_im
        else:
            r_re = F.avg_pool2d(recon_re, s)
            r_im = F.avg_pool2d(recon_im, s)
            t_re = F.avg_pool2d(target_re, s)
            t_im = F.avg_pool2d(target_im, s)
        total = total + F.mse_loss(r_re, t_re) + F.mse_loss(r_im, t_im)
    return total / len(scales)
