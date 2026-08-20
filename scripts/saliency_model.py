#!/usr/bin/env python3
"""Fully-convolutional saliency net — predicts, from a wide fractal canvas,
a spatial heatmap of where a per-formula VAE (scripts/autoencoder.py) would
have high reconstruction error if drilled into. This is the "distill the
VAE's reconstruction-error signal into a fast spatial predictor" model
discussed with Carl (2026-08-10): the live exploration loop
(src/vae_explore.rs's coarse_scan) currently scores a fixed, sparse grid of
crops with cheap hand-written heuristics (entropy/edge/intricacy); this net
is meant to eventually replace that with one forward pass over the whole
canvas, at higher effective resolution and using a richer, VAE-derived
notion of "interesting."

Deliberately NO fully-connected/dense layer anywhere ("no directly
connected end layer" — Carl's literal phrasing) — every layer is a conv,
so the output stays spatial (a heatmap, not a single vector) and the same
trained weights work at any input resolution divisible by the total
downsampling stride, not just the exact training resolution.

Trained by train_saliency.py on data from `explorer saliency-data`
(src/bin/explorer.rs) — synthetic (canvas, sparse labeled point) pairs
built from already VAE-scored zones in existing vae-explore pools, not a
fresh expensive data-collection pass.
"""
import torch
import torch.nn as nn

# Single-channel (raw escape-time field, grayscale — matches
# io::save_raw_field's L8 output, same convention vae_scorer_sidecar.py's
# `mode = "L"` already uses for in_ch=1 models).
IN_CH = 1
# 4 stride-2 conv blocks: 256 -> 128 -> 64 -> 32 -> 16. Matches
# SALIENCY_CANVAS_RES=256 (src/bin/explorer.rs) as the intended TRAINING
# resolution, but nothing here hardcodes that — a fully-conv net run on a
# real exploration canvas (e.g. 4095px) just produces a proportionally
# larger heatmap (4095/16 ≈ 256x256 output cells) with the exact same
# weights.
DOWNSAMPLE_STRIDE = 16


def conv_block(in_ch, out_ch, groups=8):
    # GroupNorm not BatchNorm: inference happens one canvas at a time
    # (the sidecar's request/response protocol, mirroring vae_scorer_
    # sidecar.py), where BatchNorm's running stats would be noisy/wrong at
    # batch size 1 — same reasoning this project's complex-VAE work
    # (ComplexGroupNorm) already established for the same failure mode.
    g = min(groups, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(g, out_ch),
        nn.ReLU(inplace=True),
    )


class SaliencyNet(nn.Module):
    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.base_ch = base_ch
        self.trunk = nn.Sequential(
            conv_block(IN_CH, base_ch),        # 256 -> 128
            conv_block(base_ch, base_ch * 2),  # 128 -> 64
            conv_block(base_ch * 2, base_ch * 4),  # 64 -> 32
            conv_block(base_ch * 4, base_ch * 4),  # 32 -> 16
        )
        # 1x1 conv, not a Linear layer: stays fully convolutional. No
        # activation — the training target is a log1p-scaled
        # reconstruction-error regression, unbounded, so a raw linear
        # readout per spatial cell is the right head.
        self.head = nn.Conv2d(base_ch * 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> heatmap (B, 1, H/16, W/16), raw (unscaled) score."""
        x = self._normalize(x)
        return self.head(self.trunk(x))

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Per-image normalization (subtract each canvas's OWN mean,
        divide by its OWN std) — added 2026-08-10 after Carl noticed the
        net leaning on "higher iteration count = more interesting," a
        real but imperfect rule. Root cause: the raw escape-time field IS
        essentially a quantized iteration-count map — the model's ONLY
        input channel — so absolute brightness is the cheapest signal
        available and gradient descent will happily exploit it. This
        pushes the model toward RELATIVE local structure (this pixel vs.
        its own canvas's typical value) instead of absolute intensity.
        Lives inside the model itself, not the training data pipeline, so
        training and `saliency_sidecar.py` inference can't drift out of
        sync — both call the same `forward`. `std` floored well above
        zero so a near-flat canvas (a real, not rare, case — most of any
        fractal's exterior/interior IS uniform) doesn't blow up."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        return (x - mean) / std


def save_saliency_model(model: SaliencyNet, path):
    torch.save({"base_ch": model.base_ch, "state_dict": model.state_dict()}, path)


def load_saliency_model(path, device) -> SaliencyNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SaliencyNet(base_ch=ckpt.get("base_ch", 16))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
