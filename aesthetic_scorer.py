#!/usr/bin/env python3
"""
Aesthetic scorer sidecar for NNfractals.
Uses CLIP ViT-L/14 zero-shot scoring against aesthetic text prompts.
No external weight downloads needed — works purely from the pretrained CLIP model.

Protocol (stdin/stdout):
  startup  → prints "READY\n" once model is loaded
  request  → one image file path per line on stdin
  response → one float score (approx 0–1) per line, or "ERROR: <msg>\n"
"""

import sys
import torch
import open_clip
from pathlib import Path
from PIL import Image

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


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-L/14 on {device}...", file=sys.stderr, flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    all_text = GOOD_PROMPTS + BAD_PROMPTS
    with torch.no_grad():
        tokens    = tokenizer(all_text).to(device)
        text_feat = model.encode_text(tokens).float()
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        good_feat = text_feat[:len(GOOD_PROMPTS)].mean(0, keepdim=True)
        bad_feat  = text_feat[len(GOOD_PROMPTS):].mean(0, keepdim=True)
        good_feat /= good_feat.norm(dim=-1, keepdim=True)
        bad_feat  /= bad_feat.norm(dim=-1, keepdim=True)

    return model, preprocess, good_feat, bad_feat, device


def score_image(model, preprocess, good_feat, bad_feat, device, path: str) -> float:
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img).float()
        feat /= feat.norm(dim=-1, keepdim=True)
        good_sim = (feat @ good_feat.T).item()
        bad_sim  = (feat @ bad_feat.T).item()
    # Map to [0, 1]: 1.0 = perfectly matches "beautiful fractal", 0.0 = matches "ugly noise"
    return (good_sim - bad_sim + 1.0) / 2.0


def main() -> None:
    try:
        model, preprocess, good_feat, bad_feat, device = load_models()
    except Exception as e:
        print(f"ERROR: failed to load CLIP: {e}", flush=True)
        sys.exit(1)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            score = score_image(model, preprocess, good_feat, bad_feat, device, path)
            print(f"{score:.4f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
