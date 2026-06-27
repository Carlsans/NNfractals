#!/usr/bin/env python3
"""
Generate a bar chart comparing CLIP and LAION scores across colormaps.
Reads colormap_loop_state.json history and saves colormap_scores.png.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def main(state_path="colormap_loop_state.json", out="colormap_scores.png"):
    state = json.load(open(state_path))
    history = state.get("history", [])
    if not history:
        print("No history yet.")
        return

    # Sort by laion_mean descending for display
    history = sorted(history, key=lambda h: h.get("laion_mean", 0), reverse=True)

    names     = [h["colormap"] for h in history]
    laion_m   = [h.get("laion_mean", 0) for h in history]
    laion_max = [h.get("laion_max",  0) for h in history]
    clip_m    = [h.get("clip_mean",  0) for h in history]
    saves     = [h.get("saves", 0)      for h in history]

    n = len(names)
    x = np.arange(n)
    w = 0.28

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.9), 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w,     laion_m,   w, label="LAION mean",  color="#2196F3", alpha=0.85)
    bars2 = ax1.bar(x,         laion_max, w, label="LAION max",   color="#03A9F4", alpha=0.60)
    bars3 = ax2.bar(x + w,     clip_m,    w, label="CLIP mean",   color="#FF5722", alpha=0.85)

    # Annotate saves count above each group
    for i, (s, lm) in enumerate(zip(saves, laion_m)):
        ax1.text(x[i], lm + 0.005, f"n={s}", ha="center", va="bottom", fontsize=7, color="#555")

    ax1.set_ylabel("LAION aesthetic score", color="#2196F3")
    ax2.set_ylabel("CLIP score", color="#FF5722")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax1.set_title("Colormap comparison — CLIP & LAION scores\n(sorted by LAION mean, n=saves per 30-min window)", fontsize=12)

    # Unified legend
    handles = [
        mpatches.Patch(color="#2196F3", alpha=0.85, label="LAION mean"),
        mpatches.Patch(color="#03A9F4", alpha=0.60, label="LAION max"),
        mpatches.Patch(color="#FF5722", alpha=0.85, label="CLIP mean"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=9)

    # Mark winner
    winner = names[0]
    ax1.axvspan(-0.5, 0.5, color="gold", alpha=0.12, zorder=0)
    ax1.text(0, ax1.get_ylim()[1] * 0.98, "★ winner", ha="center", va="top",
             fontsize=9, color="goldenrod", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  (winner: {winner})")

    # Also print table
    print(f"\n{'Colormap':<12} {'LAION mean':>10} {'LAION max':>9} {'CLIP mean':>9} {'saves':>6}")
    print("-" * 52)
    for h in history:
        print(f"{h['colormap']:<12} {h.get('laion_mean',0):>10.4f} {h.get('laion_max',0):>9.4f} "
              f"{h.get('clip_mean',0):>9.4f} {h.get('saves',0):>6}")

if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "colormap_loop_state.json"
    op = sys.argv[2] if len(sys.argv) > 2 else "colormap_scores.png"
    main(sp, op)
