#!/usr/bin/env python3
"""Simple convolutional AE / VAE for fractal images — a from-scratch,
domain-specific alternative to the frozen DINOv2/SigLIP backbone
(`train_pref.py::load_backbone`) used everywhere else in this project.
DINOv2 is trained on natural photos; fractal images have very different
statistics (self-similar texture, no objects/faces/scenes) — the same kind
of mismatch already confirmed to hurt CLIP/LAION as an aesthetic signal
(see the aesthetic-scoring project memory). This tests whether a small
encoder trained directly on this project's own fractal renders gives
train_navigate.py's regression head a more useful embedding than
repurposing a general-purpose backbone does.

AE and VAE share one encoder/decoder trunk; only the bottleneck differs
(VAE adds the reparameterization trick + a KL term pulling the latent
toward a standard normal). Trained by train_autoencoder.py; loaded here via
`load_ae_backbone`, which returns an `embed(pils) -> tensor` function with
the EXACT same interface as `train_pref.py::load_backbone`, so any script
that already accepts a pluggable backbone (train_navigate.py) can use
either interchangeably with zero changes to what's downstream of it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 128  # fixed input resolution. Small enough to train fast FROM SCRATCH
           # on this project's real corpus (tens of thousands of images) —
           # unlike DINOv2/SigLIP, there's no pretrained checkpoint doing
           # the heavy lifting here, so keeping this modest matters.

RES_512 = 512  # for vae_explore's per-formula VAEs — trained on raw
               # escape-time fields (1 channel), not colormapped RGB, so
               # more spatial resolution matters more than it did for the
               # 128px/3-channel nav-imitation backbone comparison RES was
               # originally introduced for.


class Encoder(nn.Module):
    # Spatial size after `features()` — every encoder variant (conv/resnet/
    # inception) shares this same 128->8 schedule, so `SpatialHead` (see
    # train_navigate_spatial.py) doesn't need to know which trunk it's on.
    FEAT_GRID = 8
    FEAT_CH = 256

    def __init__(self, latent_dim, variational):
        super().__init__()
        self.variational = variational
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),     # 128 -> 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),    # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), # 16 -> 8
        )
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        """Pre-flatten spatial feature map (B, 256, 8, 8) — what
        `SpatialHead` operates on directly, bypassing the global-pool step
        below entirely (see train_navigate_spatial.py's module docstring
        for why: global pooling is a prime suspect for why u/v prediction
        got NO signal across 16 embedding-based backbones)."""
        return self.conv(x)

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 8 -> 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 16 -> 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # 32 -> 64
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),                                   # 64 -> 128
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 8, 8))


class ResBlock(nn.Module):
    """Plain pre-activation-free residual block, 2x(conv3x3+BN)+skip — kept
    deliberately small (this project's whole point is testing "barebone"
    architectures, not a real ResNet-50)."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResEncoder(nn.Module):
    """Same 128->8 downsampling schedule as `Encoder`, but with one
    `ResBlock` before each downsample — isolates whether residual depth
    helps independent of the AE/VAE bottleneck choice."""
    def __init__(self, latent_dim, variational):
        super().__init__()
        self.variational = variational
        self.stem = _down(3, 32)                                  # 128 -> 64
        self.stage1 = nn.Sequential(ResBlock(32), _down(32, 64))  # 64 -> 32
        self.stage2 = nn.Sequential(ResBlock(64), _down(64, 128)) # 32 -> 16
        self.stage3 = nn.Sequential(ResBlock(128), _down(128, 256)) # 16 -> 8
        self.res_out = ResBlock(256)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        return self.res_out(self.stage3(self.stage2(self.stage1(self.stem(x)))))

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class ResDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.res_in = ResBlock(256)
        self.stage1 = nn.Sequential(_up(256, 128), ResBlock(128)) # 8 -> 16
        self.stage2 = nn.Sequential(_up(128, 64), ResBlock(64))   # 16 -> 32
        self.stage3 = nn.Sequential(_up(64, 32), ResBlock(32))    # 32 -> 64
        self.out = nn.Sequential(nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()) # 64 -> 128

    def forward(self, z):
        h = self.res_in(self.fc(z).view(-1, 256, 8, 8))
        return self.out(self.stage3(self.stage2(self.stage1(h))))


class InceptionBlock(nn.Module):
    """Several kernel sizes IN PARALLEL on the SAME input, concatenated —
    not one fixed receptive field per layer like every other encoder here.
    Fractals are self-similar (real structure at many scales at once), so
    forcing one kernel size per layer may be a genuinely bad fit, not just
    an arbitrary simplification — this is the classic GoogLeNet/Inception
    answer to that exact problem, kept small ("barebone": 4 branches, no
    5x5-as-two-3x3 factorization, no auxiliary heads)."""
    def __init__(self, cin, cout):
        super().__init__()
        b = cout // 4
        self.b1 = nn.Conv2d(cin, b, 1)
        self.b3 = nn.Conv2d(cin, b, 3, padding=1)
        self.b5 = nn.Conv2d(cin, b, 5, padding=2)
        self.bpool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(cin, cout - 3 * b, 1))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bpool(x)], dim=1)
        return self.act(self.bn(h))


class InceptionEncoder(nn.Module):
    """Same 128->8 downsampling schedule as `Encoder`/`ResEncoder`, but an
    `InceptionBlock` (parallel 1x1/3x3/5x5/pool branches) before each
    downsample instead of a plain conv or a residual block — isolates
    whether multi-kernel-size-in-parallel helps, independent of AE/VAE."""
    def __init__(self, latent_dim, variational):
        super().__init__()
        self.variational = variational
        self.stem = _down(3, 32)                                           # 128 -> 64
        self.stage1 = nn.Sequential(InceptionBlock(32, 32), _down(32, 64))   # 64 -> 32
        self.stage2 = nn.Sequential(InceptionBlock(64, 64), _down(64, 128))  # 32 -> 16
        self.stage3 = nn.Sequential(InceptionBlock(128, 128), _down(128, 256)) # 16 -> 8
        self.inc_out = InceptionBlock(256, 256)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        return self.inc_out(self.stage3(self.stage2(self.stage1(self.stem(x)))))

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class Encoder512(nn.Module):
    """512px entry point, `in_ch` channels (1 for vae_explore's raw
    escape-time fields). 6 downsampling stages instead of `Encoder`'s 4
    (512/2^6=8, landing on the same FEAT_GRID=8/FEAT_CH=256 endpoint
    everything else already assumes) — channels ramp 1->8->16->...->256
    rather than jumping straight to 32, since the extra two stages are
    working at much higher spatial resolution than the 128px encoders ever
    see."""
    FEAT_GRID = 8
    FEAT_CH = 256

    def __init__(self, latent_dim, variational, in_ch=1):
        super().__init__()
        self.variational = variational
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, 4, 2, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),    # 512 -> 256
            nn.Conv2d(8, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),      # 256 -> 128
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),     # 128 -> 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),     # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),   # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 16 -> 8
        )
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        return self.conv(x)

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class Decoder512(nn.Module):
    def __init__(self, latent_dim, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 8 -> 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 16 -> 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # 32 -> 64
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),    # 64 -> 128
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),      # 128 -> 256
            nn.ConvTranspose2d(8, out_ch, 4, 2, 1), nn.Sigmoid(),                               # 256 -> 512
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 8, 8))


class ResEncoder512(nn.Module):
    """2 new plain (no ResBlock) low-channel entry stages — 512->256->128,
    1->8->16 channels — feeding an otherwise ARCHITECTURALLY-UNCHANGED copy
    of `ResEncoder`'s own 4-stage 32->64->128->256 trunk (same `_down`/
    `ResBlock` building blocks, just reused via composition rather than
    duplicated), so it lands on the identical FEAT_GRID=8/FEAT_CH=256
    endpoint."""
    def __init__(self, latent_dim, variational, in_ch=1):
        super().__init__()
        self.variational = variational
        self.entry = nn.Sequential(_down(in_ch, 8), _down(8, 16))          # 512 -> 256 -> 128
        self.stem = _down(16, 32)                                          # 128 -> 64
        self.stage1 = nn.Sequential(ResBlock(32), _down(32, 64))           # 64 -> 32
        self.stage2 = nn.Sequential(ResBlock(64), _down(64, 128))          # 32 -> 16
        self.stage3 = nn.Sequential(ResBlock(128), _down(128, 256))        # 16 -> 8
        self.res_out = ResBlock(256)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        h = self.entry(x)
        return self.res_out(self.stage3(self.stage2(self.stage1(self.stem(h)))))

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class ResDecoder512(nn.Module):
    def __init__(self, latent_dim, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.res_in = ResBlock(256)
        self.stage1 = nn.Sequential(_up(256, 128), ResBlock(128))  # 8 -> 16
        self.stage2 = nn.Sequential(_up(128, 64), ResBlock(64))    # 16 -> 32
        self.stage3 = nn.Sequential(_up(64, 32), ResBlock(32))     # 32 -> 64
        self.exit = nn.Sequential(_up(32, 16), _up(16, 8))         # 64 -> 128 -> 256
        self.out = nn.Sequential(nn.ConvTranspose2d(8, out_ch, 4, 2, 1), nn.Sigmoid())  # 256 -> 512

    def forward(self, z):
        h = self.res_in(self.fc(z).view(-1, 256, 8, 8))
        h = self.exit(self.stage3(self.stage2(self.stage1(h))))
        return self.out(h)


class InceptionEncoder512(nn.Module):
    """Same entry-stage-prepended-ahead-of-an-unchanged-trunk pattern as
    `ResEncoder512`, applied to `InceptionEncoder`'s trunk instead."""
    def __init__(self, latent_dim, variational, in_ch=1):
        super().__init__()
        self.variational = variational
        self.entry = nn.Sequential(_down(in_ch, 8), _down(8, 16))              # 512 -> 256 -> 128
        self.stem = _down(16, 32)                                              # 128 -> 64
        self.stage1 = nn.Sequential(InceptionBlock(32, 32), _down(32, 64))     # 64 -> 32
        self.stage2 = nn.Sequential(InceptionBlock(64, 64), _down(64, 128))    # 32 -> 16
        self.stage3 = nn.Sequential(InceptionBlock(128, 128), _down(128, 256)) # 16 -> 8
        self.inc_out = InceptionBlock(256, 256)
        out_dim = latent_dim * 2 if variational else latent_dim
        self.fc = nn.Linear(256 * 8 * 8, out_dim)

    def features(self, x):
        h = self.entry(x)
        return self.inc_out(self.stage3(self.stage2(self.stage1(self.stem(h)))))

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        if self.variational:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h


class AutoEncoder(nn.Module):
    """`variational=False` -> plain AE (deterministic bottleneck, MSE-only
    loss). `variational=True` -> VAE (mu/logvar bottleneck, reparameterize,
    MSE + KL loss). Same trunk either way — only `Encoder.forward`'s return
    shape and the loss differ, so both variants stay directly comparable.
    `arch="resnet"` swaps in `ResEncoder`/`ResDecoder` (residual blocks),
    `arch="inception"` swaps in `InceptionEncoder` (parallel multi-kernel
    branches, plain `Decoder` — Inception is an encoder-side idea, the
    decoder stays simple) — both orthogonal to `variational`. A denoising
    AE/VAE (see train_autoencoder.py's `--denoise-std`/`--cutout`) reuses
    this SAME class too, for any arch, since denoising only changes what's
    fed in during training, not the architecture.

    `res`/`in_ch` default to the original 128px/3-channel (RGB) shape —
    fully backward compatible, existing checkpoints load unchanged. `res=512`
    (`vae_explore`'s per-formula VAEs, trained on raw single-channel
    escape-time fields — see `train_autoencoder.py --res 512 --channels 1`)
    dispatches to the `*512` sibling classes instead, same `arch`/
    `variational` semantics."""
    def __init__(self, latent_dim=256, variational=False, arch="conv", res=128, in_ch=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.variational = variational
        self.arch = arch
        self.res = res
        self.in_ch = in_ch
        if res == 512:
            if arch == "resnet":
                self.encoder = ResEncoder512(latent_dim, variational, in_ch=in_ch)
                self.decoder = ResDecoder512(latent_dim, out_ch=in_ch)
            elif arch == "inception":
                self.encoder = InceptionEncoder512(latent_dim, variational, in_ch=in_ch)
                self.decoder = Decoder512(latent_dim, out_ch=in_ch)
            else:
                self.encoder = Encoder512(latent_dim, variational, in_ch=in_ch)
                self.decoder = Decoder512(latent_dim, out_ch=in_ch)
        elif arch == "resnet":
            self.encoder = ResEncoder(latent_dim, variational)
            self.decoder = ResDecoder(latent_dim)
        elif arch == "inception":
            self.encoder = InceptionEncoder(latent_dim, variational)
            self.decoder = Decoder(latent_dim)
        else:
            self.encoder = Encoder(latent_dim, variational)
            self.decoder = Decoder(latent_dim)

    def encode(self, x):
        """The embedding used downstream: `mu` for a VAE (the
        distribution's mean is the standard deterministic choice at
        inference time — sampling would inject noise a downstream
        regression head doesn't want), the bottleneck vector directly for
        a plain AE."""
        if self.variational:
            mu, _ = self.encoder(x)
            return mu
        return self.encoder(x)

    def encode_spatial(self, x):
        """Pre-pool spatial feature map (B, 256, 8, 8), any arch — what
        `SpatialHead` (train_navigate_spatial.py) attends over instead of
        a collapsed global vector."""
        return self.encoder.features(x)

    def reconstruct_deterministic(self, x):
        """`forward()`'s VAE branch always samples `z = mu + std*randn_like`
        — correct for training (the KL term needs it) but wrong for
        SCORING: a ranking built on reconstruction error must not flip
        between two identical calls just because of sampling noise. Used by
        `score_vae_corpus.py`/`vae_scorer_sidecar.py`, never by the
        training loop."""
        return self.decoder(self.encode(x))

    def forward(self, x):
        if self.variational:
            mu, logvar = self.encoder(x)
            # Clamp BEFORE exp() / the KL term — an unclamped logvar can
            # blow up within a handful of steps (confirmed in practice on
            # the resnet arch: logvar ~= +40 after one epoch, exp(logvar)
            # ~1e17, which is exactly where a KL term reading ~4e16 comes
            # from). Standard VAE stabilization; inert in the normal
            # operating range so it doesn't change the plain-conv VAE's
            # already-good behavior.
            logvar = logvar.clamp(-10.0, 10.0)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decoder(z), mu, logvar
        z = self.encoder(x)
        return self.decoder(z), None, None


def loss_fn(recon, x, mu, logvar, kl_weight):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    if mu is None:
        return recon_loss, recon_loss, torch.tensor(0.0)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl, recon_loss, kl


def load_ae_model(weights_path, device):
    """Loads a trained AutoEncoder (any arch/variational combo, read from
    the checkpoint's own metadata) and returns the live `nn.Module` —
    shared by `load_ae_backbone` (wraps it as an `embed(pils)->tensor`
    closure) and train_navigate_spatial.py (needs the module itself, for
    `.encode_spatial()` and, when fine-tuning, its parameters)."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = AutoEncoder(latent_dim=ckpt["latent_dim"], variational=ckpt["variational"],
                         arch=ckpt.get("arch", "conv"), res=ckpt.get("res", 128),
                         in_ch=ckpt.get("in_ch", 3)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_ae_backbone(weights_path, device):
    """Mirrors `train_pref.py::load_backbone`'s interface exactly: returns
    an `embed(pils) -> normalized (B, D) tensor` function, so
    train_navigate.py can use a trained AE/VAE encoder as a drop-in
    replacement for the frozen DINOv2/SigLIP backbone."""
    model = load_ae_model(weights_path, device)

    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((model.res, model.res)), transforms.ToTensor()])
    mode = "L" if model.in_ch == 1 else "RGB"

    def embed(pils):
        batch = torch.stack([tf(im.convert(mode)) for im in pils]).to(device)
        with torch.no_grad():
            e = model.encode(batch)
        return F.normalize(e.float(), dim=-1)

    return embed
