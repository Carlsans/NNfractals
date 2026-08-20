#!/usr/bin/env python3
"""Train a simple convolutional AE or VAE on this project's own fractal
renders — see autoencoder.py's module docstring for why (a domain-specific
alternative to repurposing DINOv2/SigLIP as train_navigate.py's backbone).

Self-supervised (reconstruction loss only, no labels needed), so unlike
train_navigate.py it isn't limited to the ~300 labeled navigation examples
— it trains on the full GA-archive corpus (tens of thousands of renders
across fractals/, fractals_dag/, etc.), which is the whole point of
pretraining a domain-specific encoder this way.

Usage:
  python3 scripts/train_autoencoder.py --variant ae
  python3 scripts/train_autoencoder.py --variant vae --kl-weight 1e-3
"""
import argparse
import glob
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from autoencoder import AutoEncoder, RES, loss_fn


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    """Machine-readable progress line, same convention as the other train_*.py scripts."""
    print(f"PROGRESS {phase} {done} {total}", flush=True)


class ImageFolderFlat(Dataset):
    def __init__(self, paths, tf, res=RES, channels=3):
        self.paths = paths
        self.tf = tf
        self.res = res
        self.channels = channels
        self.mode = "L" if channels == 1 else "RGB"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert(self.mode)
            return self.tf(im)
        except Exception:
            return torch.zeros(self.channels, self.res, self.res)


def gather_paths(dirs, max_images, seed):
    paths = []
    for d in dirs:
        paths += glob.glob(f"{d}/*.png")
    random.Random(seed).shuffle(paths)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def corrupt(x, noise_std, cutout_frac):
    """Denoising-AE input corruption — loss is still computed against the
    CLEAN `x` (see the training loop), so this only ever affects what the
    encoder sees during TRAINING. Never applied at eval/inference: the real
    downstream use (train_navigate.py embedding a real screenshot) always
    hands the encoder a clean image, so corrupting validation/inference
    would test something other than what actually matters."""
    if noise_std > 0:
        x = (x + torch.randn_like(x) * noise_std).clamp(0.0, 1.0)
    if cutout_frac > 0:
        x = x.clone()
        b, _, h, w = x.shape
        ch, cw = max(1, int(h * cutout_frac)), max(1, int(w * cutout_frac))
        for i in range(b):
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)
            x[i, :, y0:y0 + ch, x0:x0 + cw] = 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[
        "fractals", "fractals_dag", "fractals_1", "train_corpus", "viewer_output", "nav_train_cache",
    ])
    ap.add_argument("--variant", choices=["ae", "vae"], required=True)
    ap.add_argument("--arch", choices=["conv", "resnet", "inception"], default="conv",
                     help="plain conv stack, residual blocks (ResEncoder), or parallel "
                          "multi-kernel-size blocks (InceptionEncoder)")
    ap.add_argument("--res", type=int, default=RES,
                     help="input resolution — 128 (default, RGB gallery renders) or 512 "
                          "(vae_explore's raw escape-time fields, use with --channels 1)")
    ap.add_argument("--channels", type=int, default=3, choices=[1, 3],
                     help="1 for raw single-channel escape-time fields, 3 for colormapped RGB (default)")
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=20000)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-weight", type=float, default=1e-3, help="VAE only; ignored for --variant ae")
    ap.add_argument("--denoise-std", type=float, default=0.0,
                     help="gaussian noise std added to the TRAINING input only (encoder still "
                          "must reconstruct the clean target) — 0 disables denoising")
    ap.add_argument("--cutout", type=float, default=0.0,
                     help="fraction of height/width randomly zeroed on the TRAINING input only "
                          "— 0 disables; combine with --denoise-std for a standard DAE recipe")
    ap.add_argument("--min-images", type=int, default=200,
                     help="hard floor on corpus size before refusing to train. Default (200) is "
                          "sized for the big RGB gallery corpus (train_navigate's backbone "
                          "comparisons) — vae_explore's per-formula corpora are much smaller "
                          "(one iteration realistically produces tens to low hundreds of zones), "
                          "so cmd_vae_explore passes a lower floor (20, matching train_novelty.py's "
                          "own established floor for an equally self-supervised loss).")
    ap.add_argument("--min-val", type=int, default=64,
                     help="minimum held-out validation images — same rationale as --min-images: "
                          "64 is sized for large corpora, cmd_vae_explore passes a smaller value "
                          "so a tiny early-iteration corpus doesn't lose most of its images to "
                          "validation.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                     help="warm-start from a previous checkpoint's weights instead of random "
                          "init, when its architecture matches (arch/latent_dim/res/in_ch/"
                          "variational all equal) — falls back to random init with a logged "
                          "warning on any mismatch or load failure, so this is always safe to "
                          "pass speculatively (e.g. a checkpoint from a different formula/config).")
    ap.add_argument("--out", default=None, help="default: <variant>_model.pt")
    ap.add_argument("--contact-sheet", default=None, help="default: <variant>_recon.png")
    args = ap.parse_args()

    variational = args.variant == "vae"
    out_path = args.out or f"{args.variant}_model.pt"
    contact_path = args.contact_sheet or f"{args.variant}_recon.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"gathering image corpus from {args.dirs}…")
    paths = gather_paths(args.dirs, args.max_images, args.seed)
    if args.channels == 1:
        # vae_explore.rs saves BOTH `zone_NNNN_raw.png` (the actual VAE
        # input — raw single-channel escape-time) and `zone_NNNN.png`
        # (colormapped human-browsable preview) per zone. gather_paths
        # globs ALL *.png unfiltered, so without this the RGB preview was
        # silently trained on too — doubling the corpus with a mismatched
        # distribution (colormap luminance != raw t/max_iter). Mirrors
        # score_vae_corpus.py's identical filter for the same file layout.
        paths = [p for p in paths if p.endswith("_raw.png")]
    if len(paths) < args.min_images:
        raise SystemExit(f"only {len(paths)} images found across {args.dirs} — need at least {args.min_images} (--min-images) to train an encoder.")
    n_val = max(args.min_val, int(len(paths) * args.val_fraction))
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    # Clamp to the actual train-set size, not just whatever --batch-size
    # asked for: with drop_last=True (needed so BatchNorm never sees a
    # ragged final batch), a batch_size bigger than the whole train set
    # produces a DataLoader with ZERO batches — silently, until the
    # epoch-summary log line's division by that zero blows up. The default
    # (128) is sized for the big RGB gallery corpus; vae_explore's
    # per-formula corpora are realistically far smaller, especially in
    # early iterations, so this clamp matters in practice, not just in
    # theory.
    batch_size = max(1, min(args.batch_size, len(train_paths)))
    if batch_size != args.batch_size:
        log(f"--batch-size {args.batch_size} > {len(train_paths)} train images — clamped to {batch_size}")
    log(f"{len(train_paths)} train / {len(val_paths)} val images (res {args.res}, channels {args.channels}, "
        f"latent_dim {args.latent_dim}, variant {args.variant}, arch {args.arch}, "
        f"denoise_std {args.denoise_std}, cutout {args.cutout})")

    tf = transforms.Compose([transforms.Resize((args.res, args.res)), transforms.ToTensor()])
    train_dl = DataLoader(ImageFolderFlat(train_paths, tf, res=args.res, channels=args.channels), batch_size=batch_size,
                           shuffle=True, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(ImageFolderFlat(val_paths, tf, res=args.res, channels=args.channels), batch_size=batch_size,
                         shuffle=False, num_workers=args.workers)

    model = AutoEncoder(latent_dim=args.latent_dim, variational=variational, arch=args.arch,
                         res=args.res, in_ch=args.channels).to(device)
    if args.init_from:
        try:
            ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
            compatible = (ckpt.get("arch") == args.arch and ckpt.get("latent_dim") == args.latent_dim
                          and ckpt.get("res", 128) == args.res and ckpt.get("in_ch", 3) == args.channels
                          and ckpt.get("variational") == variational)
            if compatible:
                model.load_state_dict(ckpt["state_dict"])
                log(f"warm-started from {args.init_from}")
            else:
                log(f"--init-from {args.init_from}: architecture mismatch, ignoring (random init)")
        except Exception as e:
            log(f"--init-from {args.init_from}: failed to load ({e}), ignoring (random init)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_dl)
    for epoch in range(args.epochs):
        model.train()
        running, running_recon, running_kl = 0.0, 0.0, 0.0
        for i, x in enumerate(train_dl):
            x = x.to(device)
            x_in = corrupt(x, args.denoise_std, args.cutout) if (args.denoise_std > 0 or args.cutout > 0) else x
            recon, mu, logvar = model(x_in)
            loss, recon_l, kl_l = loss_fn(recon, x, mu, logvar, args.kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item(); running_recon += recon_l.item(); running_kl += kl_l.item()
            if (i + 1) % 20 == 0:
                progress(f"train_epoch{epoch}", i + 1, steps_per_epoch)
        n = len(train_dl)
        log(f"epoch {epoch+1}/{args.epochs}  loss={running/n:.5f}  recon={running_recon/n:.5f}  kl={running_kl/n:.5f}")

    # ── Held-out reconstruction error — never trained on, the real check ──
    model.eval()
    val_recon_total, val_n = 0.0, 0
    with torch.no_grad():
        for x in val_dl:
            x = x.to(device)
            recon, _, _ = model(x)
            val_recon_total += F.mse_loss(recon, x, reduction="sum").item()
            val_n += x.numel()
    val_recon_mse = val_recon_total / val_n
    log(f"held-out val reconstruction MSE (per-pixel): {val_recon_mse:.6f}")
    # Diagnostic secondary number (train/val split only) — same convention
    # family as the PROGRESS lines. The AUTHORITATIVE per-iteration mean
    # `vae-explore`'s outer loop compares run-to-run comes from Rust parsing
    # score_vae_corpus.py's full-corpus manifest, not this line.
    print(f"RECON_MSE {val_recon_mse:.6f}", flush=True)

    # ── Contact sheet: original vs reconstruction, a real look not just a loss number ──
    with torch.no_grad():
        sample = next(iter(val_dl))[:12].to(device)
        recon, _, _ = model(sample)
        save_image(torch.cat([sample, recon], dim=0), contact_path, nrow=12)
    log(f"wrote reconstruction contact sheet -> {contact_path} (top row original, bottom row reconstructed)")

    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "variational": variational,
        "arch": args.arch,
        "res": args.res,
        "in_ch": args.channels,
    }, out_path)
    log(f"saved model -> {out_path}")


if __name__ == "__main__":
    main()
