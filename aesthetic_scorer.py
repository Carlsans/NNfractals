#!/usr/bin/env python3
"""Aesthetic scorer for NNfractals (multi-model).

Runs several image-only aesthetic/quality models and returns their scores per
image. CLIP + LAION are the originals; NIMA, TOPIQ-IAA, AP v2.5 (SigLIP) and
MUSIQ were added because CLIP/LAION barely vary across fractals — the new
models discriminate ~10x better (see scripts/compare_scorers.py).

Protocol (stdin/stdout):
  startup  -> prints "READY\n" once models are loaded
  request  -> one image file path per line on stdin
  response -> "clip laion nima topiq_iaa ap25 musiq\n" (space-separated floats)
              or "ERROR: <msg>\n" on failure
  Any model that fails to load contributes 0.0 (the sidecar still runs).

Scores:
  clip       [0,1]    zero-shot CLIP cosine vs good/bad prompts
  laion      [0,10]   LAION MLP (sac+logos+ava1-l14-linearMSE, ViT-L/14)
  nima       [~1,10]  NIMA aesthetic (AVA)                     — pyiqa
  topiq_iaa  [~1,10]  TOPIQ image-aesthetic                    — pyiqa
  ap25       [~1,10]  Aesthetic Predictor v2.5 (SigLIP)
  musiq      [0,100]  MUSIQ technical quality                  — pyiqa
"""

import sys
import torch
import torch.nn as nn
import open_clip
from PIL import Image
from huggingface_hub import hf_hub_download

# ── CLIP zero-shot prompts ────────────────────────────────────────────────────
GOOD_PROMPTS = [
    "a beautiful fractal with intricate self-similar patterns",
    "stunning abstract fractal art with vibrant colors and rich detail",
    "an aesthetically pleasing mathematical artwork with complex structure",
    "a highly detailed fractal image with beautiful color gradients",
    "award-winning generative art with deep fractal complexity",
]
BAD_PROMPTS = [
    "a boring uniform black image with no detail",
    "an ugly noisy pattern with no structure",
    "a plain solid color image",
    "a degenerate fractal that looks like random noise",
    "a featureless completely dark image",
]


class AestheticMLP(nn.Module):
    """LAION MLP head on CLIP ViT-L/14 embeddings."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(),
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.layers(x)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_models(device):
    m = {"device": device}

    # ── CLIP ViT-L/14 + LAION MLP (required baseline) ──
    log("Loading CLIP ViT-L/14...")
    clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device)
    clip.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        tok = tokenizer(GOOD_PROMPTS + BAD_PROMPTS).to(device)
        tf = clip.encode_text(tok).float()
        tf /= tf.norm(dim=-1, keepdim=True)
        good = tf[:len(GOOD_PROMPTS)].mean(0, keepdim=True)
        bad = tf[len(GOOD_PROMPTS):].mean(0, keepdim=True)
        good /= good.norm(dim=-1, keepdim=True)
        bad /= bad.norm(dim=-1, keepdim=True)
    m.update(clip=clip, preprocess=preprocess, good=good, bad=bad)

    log("Loading LAION aesthetic MLP...")
    wpath = hf_hub_download("camenduru/improved-aesthetic-predictor",
                            "sac+logos+ava1-l14-linearMSE.pth")
    mlp = AestheticMLP().to(device)
    mlp.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
    mlp.eval()
    m["mlp"] = mlp

    # ── pyiqa metrics (optional) ──
    m["pyiqa"] = {}
    try:
        import pyiqa
        for name in ("nima", "topiq_iaa", "musiq"):
            try:
                log(f"Loading pyiqa:{name}...")
                m["pyiqa"][name] = pyiqa.create_metric(name, device=device)
            except Exception as e:
                log(f"  pyiqa:{name} unavailable: {e}")
    except Exception as e:
        log(f"pyiqa unavailable: {e}")

    # ── Aesthetic Predictor v2.5 / SigLIP (optional) ──
    m["ap25"] = None
    try:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        log("Loading Aesthetic Predictor v2.5 (SigLIP)...")
        ap_model, ap_prep = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True, trust_remote_code=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        ap_model = ap_model.to(dtype).to(device).eval()
        m["ap25"] = (ap_model, ap_prep, dtype)
    except Exception as e:
        log(f"AP v2.5 unavailable: {e}")

    # ── Human-preference model: SigLIP base + trained linear head (pref_model.npz) ──
    m["pref"] = None
    import os
    if os.path.exists("pref_model.npz"):
        try:
            import numpy as np
            from transformers import AutoModel, AutoProcessor
            data = np.load("pref_model.npz")
            w = torch.tensor(np.asarray(data["w"]), dtype=torch.float32, device=device)
            lo = float(data["lo"]) if "lo" in data else 0.0
            hi = float(data["hi"]) if "hi" in data else 1.0
            rng = (hi - lo) or 1.0
            repo = "google/siglip-base-patch16-224"
            proc = AutoProcessor.from_pretrained(repo)
            sig = AutoModel.from_pretrained(repo).to(device).eval()
            log("Loading preference model (pref_model.npz) + SigLIP base...")
            m["pref"] = (proc, sig, w, lo, rng)
        except Exception as e:
            log(f"preference model unavailable: {e}")

    return m


def score_image(m, path):
    device = m["device"]
    img = Image.open(path).convert("RGB")

    # CLIP + LAION share one image embedding.
    with torch.no_grad():
        x = m["preprocess"](img).unsqueeze(0).to(device)
        feat = m["clip"].encode_image(x).float()
        fn = feat / feat.norm(dim=-1, keepdim=True)
        clip = ((fn @ m["good"].T).item() - (fn @ m["bad"].T).item() + 1.0) / 2.0
        laion = m["mlp"](fn).item()

    def pyiqa_score(name):
        met = m["pyiqa"].get(name)
        if met is None:
            return 0.0
        try:
            with torch.no_grad():
                return float(met(str(path)).item())
        except Exception:
            return 0.0

    nima = pyiqa_score("nima")
    topiq = pyiqa_score("topiq_iaa")
    musiq = pyiqa_score("musiq")

    ap25 = 0.0
    if m["ap25"] is not None:
        try:
            ap_model, ap_prep, dtype = m["ap25"]
            pv = ap_prep(images=img, return_tensors="pt").pixel_values.to(dtype).to(device)
            with torch.inference_mode():
                ap25 = float(ap_model(pv).logits.squeeze().float().cpu().item())
        except Exception:
            ap25 = 0.0

    pref = 0.0
    if m["pref"] is not None:
        try:
            proc, sig, w, lo, rng = m["pref"]
            inp = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                vision = getattr(sig, "vision_model", sig)
                out = vision(pixel_values=inp["pixel_values"])
                e = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                    else out.last_hidden_state.mean(1)
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                raw = float((e @ w).item())
            pref = min(1.0, max(0.0, (raw - lo) / rng))
        except Exception:
            pref = 0.0

    return clip, laion, nima, topiq, ap25, musiq, pref


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        m = load_models(device)
    except Exception as e:
        print(f"ERROR: failed to load models: {e}", flush=True)
        sys.exit(1)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            clip, laion, nima, topiq, ap25, musiq, pref = score_image(m, path)
            print(f"{clip:.4f} {laion:.4f} {nima:.4f} {topiq:.4f} {ap25:.4f} {musiq:.4f} {pref:.4f}",
                  flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
