#!/usr/bin/env python3
"""
Dual aesthetic scorer for NNfractals.
Runs CLIP zero-shot AND LAION MLP sequentially, sharing the CLIP image embedding.

Protocol (stdin/stdout):
  startup  → prints "READY\n" once both models are loaded
  request  → one image file path per line on stdin
  response → "clip_score laion_score\n" (two floats, space-separated)
             or "ERROR: <msg>\n" on failure

Scores:
  clip_score  : [0, 1]  zero-shot CLIP cosine similarity vs good/bad prompts
  laion_score : [0, 10] LAION MLP trained on human aesthetic ratings
                        (sac+logos+ava1-l14-linearMSE, ViT-L/14 backbone)
"""

import sys
import torch
import torch.nn as nn
import open_clip
from pathlib import Path
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

# ── LAION aesthetic MLP ───────────────────────────────────────────────────────
class AestheticMLP(nn.Module):
    """MLP head trained on top of CLIP ViT-L/14 embeddings."""
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


def load_models(device):
    # CLIP ViT-L/14
    print("Loading CLIP ViT-L/14...", file=sys.stderr, flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    # Pre-encode CLIP text anchors
    all_text = GOOD_PROMPTS + BAD_PROMPTS
    with torch.no_grad():
        tokens    = tokenizer(all_text).to(device)
        text_feat = model.encode_text(tokens).float()
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        good_feat = text_feat[:len(GOOD_PROMPTS)].mean(0, keepdim=True)
        bad_feat  = text_feat[len(GOOD_PROMPTS):].mean(0, keepdim=True)
        good_feat /= good_feat.norm(dim=-1, keepdim=True)
        bad_feat  /= bad_feat.norm(dim=-1, keepdim=True)

    # LAION MLP
    print("Loading LAION aesthetic MLP...", file=sys.stderr, flush=True)
    weights_path = hf_hub_download(
        "camenduru/improved-aesthetic-predictor",
        "sac+logos+ava1-l14-linearMSE.pth",
    )
    mlp = AestheticMLP().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()

    return model, preprocess, good_feat, bad_feat, mlp, device


def score_image(model, preprocess, good_feat, bad_feat, mlp, device, path: str):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img).float()
        feat_norm = feat / feat.norm(dim=-1, keepdim=True)

        # CLIP zero-shot score → [0, 1]
        good_sim  = (feat_norm @ good_feat.T).item()
        bad_sim   = (feat_norm @ bad_feat.T).item()
        clip_score = (good_sim - bad_sim + 1.0) / 2.0

        # LAION MLP score → [0, 10] (raw output, no clamp)
        laion_score = mlp(feat_norm).item()

    return clip_score, laion_score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess, good_feat, bad_feat, mlp, device = load_models(device)
    except Exception as e:
        print(f"ERROR: failed to load models: {e}", flush=True)
        sys.exit(1)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            clip_score, laion_score = score_image(
                model, preprocess, good_feat, bad_feat, mlp, device, path
            )
            print(f"{clip_score:.4f} {laion_score:.4f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
