#!/usr/bin/env python3
"""
Aesthetic scorer sidecar for NNfractals.
Uses LAION Improved Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP).

Protocol (stdin/stdout):
  startup  → prints "READY\n" once model is loaded
  request  → one image file path per line on stdin
  response → one float score (1-10 scale) per line on stdout
             or "ERROR: <message>\n" on failure
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path


# Weights are cached in ~/.cache/nnfractals/
CACHE_DIR = Path.home() / ".cache" / "nnfractals"
WEIGHTS_FILE = CACHE_DIR / "sa_0_4_vit_l_14_linear.pth"

# Download URL for LAION improved aesthetic predictor v2 (linear, ViT-L/14)
WEIGHTS_URL = (
    "https://github.com/christos-c/improved-aesthetic-predictor"
    "/releases/download/v0.1/sa_0_4_vit_l_14_linear.pth"
)


class AestheticMLP(nn.Module):
    """5-layer MLP matching the LAION improved aesthetic predictor v2 architecture."""
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128),        nn.Dropout(0.2),
            nn.Linear(128, 64),          nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def download_weights() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace hub first (may be cached from previous runs)
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(
            repo_id="christos-c/improved-aesthetic-predictor",
            filename="sa_0_4_vit_l_14_linear.pth",
        )
        import shutil
        shutil.copy(src, WEIGHTS_FILE)
        return
    except Exception:
        pass

    # Fall back to direct GitHub release download
    print("Downloading aesthetic predictor weights (~18 MB)…", file=sys.stderr, flush=True)
    import urllib.request
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)


def load_models():
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP ViT-L/14 with OpenAI pretrained weights
    print("Loading CLIP ViT-L/14…", file=sys.stderr, flush=True)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    clip_model.eval()

    # Aesthetic MLP
    if not WEIGHTS_FILE.exists():
        download_weights()

    print("Loading aesthetic predictor…", file=sys.stderr, flush=True)
    mlp = AestheticMLP(768).to(device)
    state = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()

    return clip_model, preprocess, mlp, device


def score_image(clip_model, preprocess, mlp, device, path: str) -> float:
    from PIL import Image
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img).float()
        features /= features.norm(dim=-1, keepdim=True)
        score = mlp(features).item()
    return score


def main() -> None:
    try:
        clip_model, preprocess, mlp, device = load_models()
    except Exception as e:
        print(f"ERROR: failed to load models: {e}", flush=True)
        sys.exit(1)

    # Signal readiness to the Rust process
    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            s = score_image(clip_model, preprocess, mlp, device, path)
            print(f"{s:.4f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
