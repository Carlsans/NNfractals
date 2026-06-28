# NNfractals — Complete Principles & Design Reference

*Last updated: 2026-06-28. Audience: developers and AI agents resuming work on this codebase.*

This document is the authoritative explanation of **every principle in action** in NNfractals — what each mechanism does, the exact numbers it uses, *why* it exists, and what was tried and rejected. It is written to be read end-to-end by a human or loaded as ground truth by an AI agent. Line references point into `src/`.

---

## Table of Contents

1. [What This Program Does](#1-what-this-program-does)
2. [The Formula Representation](#2-the-formula-representation)
3. [The 58 Basis Functions](#3-the-58-basis-functions)
4. [Rendering: Escape-Time Iteration](#4-rendering-escape-time-iteration)
5. [Colormaps and Why Turbo Won](#5-colormaps-and-why-turbo-won)
6. [The Genetic Algorithm](#6-the-genetic-algorithm)
7. [The Per-Generation Fitness Function](#7-the-per-generation-fitness-function)
8. [The Aesthetic Scorer (CLIP + LAION)](#8-the-aesthetic-scorer-clip--laion)
9. [The Save Gate](#9-the-save-gate)
10. [Measured-at-Save Scores: Recursion & Self-Replication](#10-measured-at-save-scores-recursion--self-replication)
11. [The Formula-Only Proxy Models](#11-the-formula-only-proxy-models)
12. [Epoch Restart & Archive Seeding](#12-epoch-restart--archive-seeding)
13. [Deduplication](#13-deduplication)
14. [Dual Instances & The Daemon](#14-dual-instances--the-daemon)
15. [Configuration Reference](#15-configuration-reference)
16. [What Was Tried and Removed](#16-what-was-tried-and-removed)
17. [File Map](#17-file-map)
18. [Key Invariants](#18-key-invariants)
19. [Operations Playbook](#19-operations-playbook)
20. [What to Try Next](#20-what-to-try-next)

---

## 1. What This Program Does

NNfractals is a **genetic algorithm that autonomously evolves mathematical fractal formulas** and saves only the most aesthetically beautiful results, judged by neural aesthetic models (CLIP + LAION). It runs indefinitely as a background daemon.

The core loop, repeated every generation:

```
evolve formulas → render images on GPU → score complexity/novelty/aesthetics
   → rank → save the beautiful ones → mutate/crossover the best → repeat
```

Two independent daemon instances run in parallel, both feeding a shared pool of saved fractals (`fractals/`). A periodic dedup pass keeps the pool free of near-duplicates.

The program is **fully self-contained and unsupervised** once started: it warm-starts from previously saved fractals, evolves continuously, and self-prunes. Human/agent intervention is limited to tuning weights and retraining the proxy models.

---

## 2. The Formula Representation

Every fractal is defined by a **sparse iterated complex map**:

```
z₀ = 0
z_{n+1} = Σ_i coeff_i · φ_{basis_i}(z_n, c)
```

where:
- `z` is the current complex iterate, `c` is the per-pixel parameter (the complex-plane coordinate of that pixel)
- `φ_{basis_i}` is one of **58 holomorphic basis functions** (`src/formula.rs`)
- `coeff_i = re + i·im` is a complex coefficient
- A genome carries **2 to 8 terms** (`MIN_TERMS=2`, `MAX_TERMS=8` in `src/genome.rs:7-8`), each `(basis_index, re, im)`

The genome is **sparse**: of 58 possible bases, only 2–8 are active in any fractal. Terms sharing a basis are summed when expanded to the dense weight vector (`Genome::formula_weights()`, `src/genome.rs:130`).

### Genome Struct (`src/genome.rs:46-86`)

| Field | Type | Meaning |
|-------|------|---------|
| `terms` | `Vec<FormulaTerm>` | The sparse formula (2–8 terms) |
| `fitness` | f32 | GA selection score (composite, recomputed each gen) |
| `beauty` | f32 | Stored final score at save (LAION/10 or fallback) |
| `beauty_boundary/edge/entropy/self_sim/cool_zone` | f32 | 5-component beauty breakdown |
| `clip_score` | f32 [0,1] | CLIP aesthetic, measured at save |
| `laion_score` | f32 [0,10] | LAION aesthetic, measured at save |
| `self_replication` | f32 [0,1] | Zoom self-similarity, measured at save |
| `fractal_recursion` | f32 [0,1] | Embedded baby-copies, measured at save |
| `pred_recursion` | f32 [0,1] | Formula-predicted recursion, every gen |
| `pred_clip` | f32 [0,1] | Formula-predicted CLIP, every gen |
| `formula_diversity` | f32 | k-NN distance in basis-space, every gen |
| `id` | u64 | Random unique id (hex filename) |
| `view_cx/cy/zoom` | f32 | Per-genome viewport (center + zoom) |

**Why sparse direct evolution?** An earlier architecture used a transformer with a latent code and backpropagation. It fought the GA and drove elites toward noise (see §16). Direct sparse evolution naturally produces *distinct formula families* — Mandelbrot variants, Julia-like sets, rational maps, transcendental maps — because different basis combinations yield visually different fractals.

### Per-Genome Viewport (`src/genome.rs:91-94`)

Each genome carries its own view window: `view_bounds()` returns `(cx − 2/zoom, cx + 2/zoom, cy − 2/zoom, cy + 2/zoom)`. So `zoom=1` shows the standard `[-2,2]²` window; higher zoom shows a tighter region. View params are evolved (mutated/crossed) alongside the formula. The `[rendering] view_*` fields in config.toml apply only to the standalone viewer/CLI render, not the GA.

---

## 3. The 58 Basis Functions

Full list from `basis_name()` (`src/formula.rs:227-248`). These are the holomorphic (and a few non-holomorphic, e.g. conjugate / burning-ship) building blocks the GA composes:

| Idx | φ | Idx | φ | Idx | φ | Idx | φ |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 0 | z² | 15 | (z+c)² | 30 | tan | 45 | Re+\|Im\| |
| 1 | z³ | 16 | (z−c)² | 31 | sinh | 46 | \|BS\| (burning-ship) |
| 2 | z⁴ | 17 | (zc)² | 32 | cosh | 47 | conj |
| 3 | z⁵ | 18 | sin | 33 | tanh | 48 | conj² |
| 4 | z | 19 | cos | 34 | exp | 49 | z\|z\| |
| 5 | 1/z | 20 | sin(π) | 35 | exp(−z) | 50 | z/\|z\| |
| 6 | 1/z² | 21 | cos(π) | 36 | exp(zc) | 51 | z/(z²+1) |
| 7 | c | 22 | sin(z²) | 37 | z·exp | 52 | (z²−1)/(z²+1) |
| 8 | c² | 23 | cos(z²) | 38 | exp·c | 53 | z²/(z−1) |
| 9 | c³ | 24 | sin(z+c) | 39 | log(z+1) | 54 | 1/(z²+c) |
| 10 | zc | 25 | cos(z+c) | 40 | log(z²+1) | 55 | z²c/(z+c) |
| 11 | z²c | 26 | sin(zc) | 41 | z·log | 56 | 1 (constant) |
| 12 | zc² | 27 | cos(zc) | 42 | sin(1/z) | 57 | i (constant) |
| 13 | z²c² | 28 | z·sin | 43 | exp(1/z) | | |
| 14 | c/z | 29 | z·cos | 44 | \|Re\|+Im | | |

**Basis index 7 (`c`) and indices 0–3 (`z²…z⁵`) are special**: their interaction `f[7]·(f[0]+f[1]+f[2]+f[3])` is feature 61 in the recursion predictor — a Mandelbrot-like map needs both a `c` term and a `z`-power term to produce self-replication (see §11).

### Exotic Bases (`src/genome.rs:31-41`)

A curated subset proven to score well on CLIP, force-injected during restarts:
```
EXOTIC = [0:z², 18:sin, 30:tan, 46:|BS|, 51:z/(z²+1), 52:(z²−1)/(z²+1), 54:1/(z²+c)]
```
The current record genome (CLIP 0.5294) used `[cosh, (z²−1)/(z²+1), |BS|, (z²−1)/(z²+1), (z−c)²]` — the burning-ship absolute-value basis is load-bearing for top scores.

---

## 4. Rendering: Escape-Time Iteration

**Implementation:** `render_cpu_iter()` (`src/fractal.rs:67`), GPU path in `src/render_gpu.rs`.

For each pixel, `c` = its complex coordinate, `z` starts at 0. Iterate `z ← Σ coeff·φ(z,c)` up to `max_iter` times. If `|z|² > bailout²` (bailout=4.0 → bailout²=16), the pixel "escapes". The escape time uses **smooth (continuous) coloring**:

```
escape_time = iter + 1 − log₂(log₂|z|)     (src/fractal.rs:99-101)
```

This fractional value eliminates the banding that integer iteration counts produce, giving smooth gradients essential for the aesthetic scorers. If `z` goes non-finite, the raw iteration count is returned. Pixels that never escape return `max_iter` (interior points).

**Two resolutions:**
- **Eval** (64×64, `eval_max_iter=128`): fast, used every generation for fitness — 4096 pixels/genome × 40 genomes/gen
- **Full** (512×512, `max_iter=192`): used only when saving a genome — richer detail in the final PNG

**GPU batching:** All 40 genomes per generation are rendered in a single GPU dispatch via wgpu/Vulkan (`evaluate_fitness_batch`). If no GPU, Rayon parallelizes CPU rendering across all cores. The same `apply_formula` math runs in both the WGSL shader and the Rust CPU path — they must stay in sync.

---

## 5. Colormaps and Why Turbo Won

**Implementation:** `apply_colormap()` (`src/colormap.rs:3`). Escape time is normalized to `[0,1]` via `t/max_iter`, then mapped to RGB.

Available colormaps: built-in gradients (turbo, viridis, plasma, magma, inferno via the `colorous` crate) plus 9 hand-written ones (earth, bone, neon, lava, aurora, galaxy, sunset, arctic, ember).

### The Colormap Experiment (4h, 13 colormaps, 30-min windows)

**Turbo dominated decisively:**

| Colormap | Saves/window | CLIP mean | LAION mean |
|----------|-------------|-----------|------------|
| **turbo** | **382** | **0.5227** | **5.332** |
| lava | 8 | 0.5189 | 5.301 |
| arctic | 4 | 0.5131 | 5.299 |
| neon | 9 | 0.5085 | 5.265 |
| aurora | 2 | 0.5018 | 5.247 |
| ember | 2 | 0.5040 | 5.244 |
| sunset | 1 | 0.4961 | 5.204 |
| earth | 3 | 0.4875 | 5.170 |
| galaxy | 2 | 0.4839 | 5.156 |

**Why turbo wins:** Turbo maps low escape times (the majority of pixels in most fractals) to **blue/cyan**. CLIP ViT-L/14 was trained on photographs and associates cool blue tones with high-quality imagery (ocean, sky, professional photography). Warm palettes (lava, sunset, ember) read as "graphic art" and drop CLIP below the 0.520 save threshold. All 9 custom colormaps failed below 0.520. **Turbo is now permanently fixed in config.** Chart: `colormap_scores.png`.

This is also why `beauty_score`'s "cool-zone" component (`src/fitness.rs:351-359`) rewards pixels in the 5–40% escape band — that band is blue/cyan under turbo.

---

## 6. The Genetic Algorithm

**Implementation:** `Optimizer` (`src/optimizer.rs`).

### Population
- **40 individuals**, evolved in discrete generations (`population_size=40`)
- Each generation: evaluate all 40 → rank by fitness → attempt saves → evolve

### Selection & Reproduction (`evolve()`, `src/optimizer.rs:547`)

1. **Formula-diverse elitism**: instead of taking the top-6 by raw fitness, take one representative per *unique formula signature* (top-2 basis label, `formula_ops_label()`). This prevents the elite pool from collapsing into 6 near-identical formulas. Fills to `elitism_count=6` with remaining best if fewer than 6 unique types exist.
2. **Continuous random injection**: 3 fresh random genomes every generation (`N_RANDOM_PER_GEN=3`). This sustained trickle — not just the restart bursts — is what broke the historical CLIP plateau (best rose past 0.50).
3. **Crossover + mutation**: fill the rest of the population. 50% chance of crossover between two *different-formula* elites then mutate; 50% chance of pure mutation of one elite.

### Crossover (`Genome::crossover`, `src/genome.rs:194`)
Union of both parents' terms, each kept with probability 0.5, clamped to `[2,8]` terms. View = average of centers, **geometric mean** of zooms (`sqrt(zoom_a · zoom_b)`).

### Mutation (`Genome::mutate`, `src/genome.rs:218`)
- **Coefficient perturbation**: each term's `re`/`im` perturbed by ±`mutation_scale` (0.08) with probability `mutation_rate` (0.20)
- **Basis swap**: each term swaps to a random basis with probability `BASIS_SWAP_PROB=0.18`
- **Term grow/shrink**: add a random term (prob `TERM_ADD_PROB=0.20`, if < 8 terms); drop a random term (prob `TERM_DROP_PROB=0.15`, if > 2 terms)
- **View mutation**: 30% chance to multiply zoom by a delta (mostly zoom-in, clamped `[0.5, 25]`); 30% chance to pan (clamped `[-2.5, 2.5]`)

### Best-Ever Tracking & Stagnation (`src/optimizer.rs:237-247`)
The single best genome by **raw beauty** (re-evaluated without novelty inflation) is tracked. If it doesn't improve by > 0.005 for `restart_after_gens=30` generations, an epoch restart fires (see §12).

---

## 7. The Per-Generation Fitness Function

This is the heart of the system. Every generation, all 40 genomes get a composite fitness (`src/optimizer.rs:225-226`):

```
fitness = multiscale_entropy
        + 0.45 · visual_novelty       (novelty_weight)
        + 0.80 · pred_recursion       (recursion_pred_weight)
        + 0.30 · formula_diversity    (formula_diversity_weight)
        + 0.50 · pred_clip            (clip_pred_weight)
```

The five terms operate in **different spaces** — image complexity, rendering-behavior space, formula-structure space, and learned-aesthetic space — so they don't redundantly reward the same thing.

### Term 1 — Multiscale Entropy (implicit weight 1.0)

**`multiscale_entropy()` (`src/fitness.rs:90`).** The primary complexity signal. Geometric mean of PNG-compression entropy at two scales:

- **Fine** (64×64): `png_compression_entropy()` = `compressed_PNG_bytes / raw_bytes`. Higher = harder to compress = more structural detail. (~0.3 boring, ~0.9+ rich.)
- **Coarse** (16×16): the 64×64 escape field 4×-average-pooled (`FACTOR=4`), then the same PNG metric.

```
multiscale_entropy = sqrt(fine_entropy × coarse_entropy)
```

**Why geometric mean of two scales?** This is the **anti-noise mechanism**. Pure fine-scale entropy rewards salt-and-pepper noise just as much as genuine structure (noise compresses poorly → high entropy). But noise averages to near-uniform when downsampled 4× → coarse entropy collapses toward 0 → the geometric mean collapses. A genuinely structured fractal stays complex at *both* scales, so the product stays high. This single change is what stops the GA from drifting toward granular noise.

### Term 2 — Visual Novelty (weight 0.45)

**`novelty_score()` over `behavior_descriptor()` (`src/fitness.rs:143,157`).** Each genome's escape times are binned into a **32-bin normalized histogram** (its "behavioral descriptor" — the distribution of escape depths). Novelty = average L2 distance to the **k=5 nearest neighbors** in a rolling archive of the 150 most recently evaluated descriptors (`archive_size=150`, `novelty_k=5`).

**Why:** Without novelty pressure the GA converges to one visual family. Novelty rewards genomes whose escape-time *behavior* differs from what's been seen recently — pushing exploration of distinct fractal appearances (spirals, webs, bubbles, dendrites).

The archive is **primed on init** with the starting population's own descriptors (`src/optimizer.rs:81-95`) to avoid a generation-1 novelty spike (empty archive → everything scores ≈1.0).

### Term 3 — Predicted Recursion (weight 0.80, the largest)

**Linear model `recursion_model.json` via `RecursionModel::predict()` (`src/recursion_model.rs:33`).** A formula-only estimate of `fractal_recursion` (how strongly the set contains embedded baby-copies of itself). Computed in microseconds from formula structure — no rendering. See §11 for the model. Largest weight because recursion (baby-Mandelbrots) is the single most prized visual property.

### Term 4 — Formula Diversity (weight 0.30)

**`novelty_score()` over `formula_basis_normalized()` (`src/genome.rs:99`).** Same k-NN novelty mechanism as Term 2, but in **formula-structure space** instead of rendering space. Each genome maps to a 58-dim vector where element `i` = Σ|coeff| on basis `i`, L2-normalized (direction-sensitive: *which* bases dominate, not how large the coefficients are). Novelty = avg L2 distance to k=5 nearest in a rolling 150-entry formula archive.

**Why separate from visual novelty?** Two formulas with different basis mixes can render to similar escape-time histograms (Term 2 wouldn't distinguish them), and vice versa. Term 4 explicitly rewards genuinely different *formula families*, preventing the population from converging on one basis combination even if it explores view/coefficient variations.

**Validated by a 3×2h tuning loop:** formula diversity rose 0.70 → 0.80 → 0.87 across iterations with 96–97% unique formula signatures per window, while LAION held ~5.31. Weight 0.30 was stable throughout — no auto-tuning needed.

### Term 5 — Predicted CLIP (weight 0.50)

**Linear model `clip_model.json`, same `RecursionModel` struct.** A formula-only estimate of the CLIP aesthetic score. This is the **aesthetic feedback loop**: before this term, CLIP/LAION were only a binary save gate, so the GA had no aesthetic signal in its fitness and scores were stationary. Term 5 lets the GA actively steer toward formula families that historically produce CLIP-beautiful images. See §11.

---

## 8. The Aesthetic Scorer (CLIP + LAION)

**Rust side:** `AestheticScorer` (`src/aesthetic.rs`). **Python side:** `aesthetic_scorer.py`.

A persistent Python sidecar process is spawned once at startup. It loads two models (~30–60s on first run) then prints `READY`. The Rust side communicates over stdin/stdout: send one image file path per line, receive `"clip_score laion_score"` back.

### CLIP Zero-Shot Score → [0,1] (`aesthetic_scorer.py:96-99`)

CLIP **ViT-L/14** encodes the image. The score is a contrast between two prompt sets:
```
clip_score = (good_similarity − bad_similarity + 1.0) / 2.0
```
- **GOOD prompts**: "a beautiful fractal with intricate self-similar patterns", "stunning abstract fractal art with vibrant colors and rich detail", "award-winning generative art with deep fractal complexity", etc.
- **BAD prompts**: "a boring uniform black image", "an ugly noisy pattern with no structure", "a degenerate fractal that looks like random noise", etc.

So CLIP measures: *does this look more like beautiful fractal art than like noise/emptiness, per a vision-language model trained on millions of captioned images?*

### LAION Aesthetic Score → [0,10] (`aesthetic_scorer.py:42-58`)

An MLP head (768→1024→128→64→16→1) trained on human aesthetic ratings (the `sac+logos+ava1-l14-linearMSE` LAION aesthetic predictor), fed the normalized CLIP ViT-L/14 embedding. This is a direct learned predictor of *human-rated visual appeal*, sharing CLIP's image embedding so both scores come from one encode.

**Cost:** ~0.5–2s/image on GPU, with a 15s blocking timeout. Far too slow to run on all 40 genomes/generation — hence it's used only at the save gate (a handful of candidates per generation) and as an occasional async "probe" every 3 generations for the live display.

---

## 9. The Save Gate

**`try_save()` (`src/optimizer.rs:446`).** A genome is persisted (`{id}.png` + `{id}.nn`) only after clearing four stages, in increasing cost order. Each generation, candidates are tried in descending raw-PNG-entropy order until `elitism_count=6` *real* attempts are spent ("exists" hits are free).

### Stage 1 — PNG Entropy Prefilter (fast, Rust) — `src/optimizer.rs:454-469`
- `is_degenerate()` rejects if > 95% of pixels share an escape time
- `min_entropy_prefilter=0.38 ≤ raw_fine_PNG_entropy ≤ max_entropy_prefilter=0.70`
- Below 0.38 → "low_png" (too uniform/boring); above 0.70 → "noise" (salt-and-pepper)
- **Note:** this uses **raw fine-scale** entropy, *not* multiscale. Multiscale is for selection ranking; the raw band is the save filter.

### Stage 2 — Diversity Gate (fast, Rust) — `src/optimizer.rs:471-476`
Min L2 distance from this genome's behavioral descriptor to **all** previously saved descriptors must be ≥ `min_save_distance=0.05`. Else "low_diversity". The 300 most-recent save descriptors persist across epoch restarts so the same view isn't re-saved each epoch.

### Stage 3 — Best-View Search (fast, Rust) — `best_entropy_view()` (`src/fractal.rs:43-65`)
Before the expensive full render, search a **3×3 grid of view offsets** (±`0.30/zoom` pan in each axis, 9 candidates) at 64×64, scoring each by multiscale entropy. Render the *winning* view at full 512×512 for CLIP. **The genome's stored view params are unchanged** — only the render uses the optimized composition. This recovers CLIP that would be lost because the GA evolves view params for fitness, not for CLIP framing.

### Stage 4 — Aesthetic OR Gate (expensive, Python) — `src/optimizer.rs:503-512`
Render 512×512 → save temp PNG → CLIP+LAION score. Pass if:
```
clip_score ≥ 0.522   OR   laion_score ≥ 5.30
```

**Why OR, not AND?** OR privileges fractals that are **excellent on at least one** aesthetic axis. A fractal with CLIP 0.525 but mediocre LAION 5.22 saves (CLIP excels); one with CLIP 0.515 but LAION 5.35 also saves (LAION excels). The previous AND gate blocked both. (History: AND at 0.520/5.30 was the long-standing default; a brief relaxed-AND experiment at 0.512/5.20 over-saturated the pool with ~1000 saves/2h; the OR gate at 0.522/5.30 is the current best balance of quality and throughput.)

If the scorer is unavailable, fall back to Rust `beauty_score ≥ min_beauty=0.35`.

### After Passing
The permanent PNG is written, then the genome's `self_replication` and `fractal_recursion` are measured (§10) and stored in the `.nn` JSON along with all scores. Stored `beauty`/`fitness` = `laion_score/10`.

---

## 10. Measured-at-Save Scores: Recursion & Self-Replication

These two render-based measurements are too expensive for the per-generation loop, so they're computed only for the handful of genomes that pass the save gate. They travel with the `.nn` file and feed archive seed ranking (§12) and proxy-model training (§11).

### Self-Replication Score (`self_replication_score`, `src/fractal.rs:278`)

*"Does the fractal keep reproducing rich structure as you zoom into its boundary?"* — the defining Mandelbrot property.

Method (5 zoom levels, 5× each → ~625× total depth, 96×96 each):
1. Render the base view; record edge density and a 32×32 contrast-normalized (z-scored) **structure vector**
2. Re-center on the **richest boundary point** (highest-gradient pixel in central 70%) and zoom in 5×; repeat
3. **Persistence** = average retained edge density at each deeper level vs base (does structure survive zoom or smooth away?)
4. **Self-similarity** = mean positive correlation between consecutive levels' structure vectors
5. Score = `0.65·persistence + 0.35·self_similarity`, clamped [0,1]

Z-scoring the structure vector makes it invariant to the escape-time offset that rises with zoom depth — what survives is the *shape*. A smooth/degenerate map collapses to ≈0; a true fractal approaches 1.

### Fractal Recursion Score (`fractal_recursion_score`, ~`src/fractal.rs:396+`)

*"Does a complete miniature copy of the whole set (a baby-Mandelbrot) appear embedded inside it?"* — distinct from self-replication (which is boundary-detail persistence).

Method: grid the frame, rank cells by `local_edge_density · (1 + 2·island_bonus)` where the island bonus rewards a cell holding *some but not all* interior (a contained body wrapped in boundary). Descend into top candidate cells and **template-match** the local structure against the whole-set template across all **8 dihedral orientations** (`dihedral_variants`, `src/fractal.rs:350`) — so rotated/mirrored copies still match. Gated by `field_intricacy()` (density of gradient sign-flips along scanlines) to reject smooth radial ramps and require genuine non-monotone structure. (Pure noise also scores high on intricacy but fails the copy-match, so the two together single out true recursion.)

---

## 11. The Formula-Only Proxy Models

Both `pred_recursion` and `pred_clip` use the **identical** `RecursionModel` struct (`src/recursion_model.rs`) and the **identical** 61-dim feature vector — only the training target differs. This is a deliberate reuse: the linear model is generic, "RecursionModel" is just the historical name.

### The Feature Vector (`recursion_features()`, `src/genome.rs:116`)

61 dimensions, which **must stay byte-for-byte aligned** with the `features()` function in both Python training scripts:
- `f[0..58)`: Σ|coeff| per basis (which bases drive the formula, and how hard)
- `f[58]`: number of terms
- `f[59]`: total |coeff| across all terms
- `f[60]`: `f[7] · (f[0]+f[1]+f[2]+f[3])` = `c × z-power` interaction (a Mandelbrot-like map needs both `c` and a z-power present)

### The Model (`RecursionModel::predict`, `src/recursion_model.rs:33`)

Standardized linear regression:
```
predict(feats) = clamp( bias + Σ_i weight_i · (feats_i − mean_i) / std_i , 0, 1)
```
Trained offline by Ridge regression with 5-fold cross-validation. If `clip_model.json` / `recursion_model.json` is missing or malformed, `load()` returns `None` and that fitness term silently becomes 0 (criterion inert) — the GA still runs.

### Current Trained Models

| Model | Script | Target | n | cv_pearson | λ | Weight |
|-------|--------|--------|---|-----------|---|--------|
| `recursion_model.json` | `fit_recursion_model.py` | `fractal_recursion` [0,1] | 2088 | 0.513 | 5.0 | 0.80 |
| `clip_model.json` | `fit_clip_model.py` | `clip_score` [0,1] | 2985 | 0.452 | 10.0 | 0.50 |

CLIP uses stronger regularization (λ=10) because its target range is narrow ([0.47, 0.53]) and easy to overfit. cv_pearson ≈ 0.45 is a weak *individual* predictor but a meaningful *population-level* bias — exactly how the recursion predictor (0.51) proved effective. Both should be retrained as the archive grows (see §19).

**Why formula-only?** Rendering + CLIP-scoring all 40 genomes/gen would be ~40× slower. The linear model approximates the expensive label from formula structure alone in microseconds, so the bias can be applied every generation to every genome.

---

## 12. Epoch Restart & Archive Seeding

### Warm-Start at Launch (`Optimizer::new`, `src/optimizer.rs:40`)
On startup, load up to **12 archive seeds** (best saved genomes) + 8 fresh randoms, fill the rest with seed mutations. So a fresh process resumes from the best prior fractals, not from scratch.

### Stagnation Restart (`restart_population`, `src/optimizer.rs:378`)
After `restart_after_gens=30` generations without best-ever improvement:
1. Force-save the current best-ever if not already saved
2. Reload **6 archive seeds** (fewer than warm-start — keep top genomes in play without over-constraining)
3. Inject **18 random genomes**, of which the first **8 are "exotic"** (forced to use a proven-good rare basis from the EXOTIC set)
4. Fill the rest with seed mutations
5. Retain the 300 most-recent save descriptors so the diversity gate keeps blocking already-saved regions

### Archive Seed Ranking (`load_archive_seeds`, `src/optimizer.rs:144`)
Seeds are ranked by an **aesthetic-first** score:
```
seed_score = 2.0·clip_score + 0.15·(laion/10) + 0.20·self_replication + 0.20·fractal_recursion
```
(Falls back to `beauty` if `clip_score` is 0.) **CLIP is weighted 2×** so epoch restarts inherit the most aesthetically successful ancestors. The earlier formula (`clip + 0.35·self_rep + 0.35·fractal_rec`) let recursion dominate and deprioritized beautiful-but-non-recursive genomes — changed to fix exactly that.

---

## 13. Deduplication

**`scripts/dedup.py`**, auto-triggered every `interval_hours=2.0` from within the evolution process (`spawn_dedup_cleaner` in `src/main.rs`). Similarity = geometric mean of DCT-based perceptual similarity at three scales (16×16, 32×32, 64×64 grayscale). Pairs above `similarity_threshold=0.97` are deduplicated (the weaker-LAION copy is deleted).

Dedup is what lets the save gate be *relaxed* (`min_save_distance=0.05` instead of 0.08): the per-run diversity gate is fast/approximate, dedup is the thorough cross-run cleanup. Both daemon instances trigger it independently.

---

## 14. Dual Instances & The Daemon

**`scripts/evo_daemon.sh`** manages two parallel evolution processes:

| Instance | Config | Population dir | PID file |
|----------|--------|---------------|----------|
| 1 | `config.toml` | `populations/` | `evolution.pid` |
| 2 | `config2.toml` | `populations2/` | `evolution2.pid` |

Both write saves to the **shared** `fractals/` directory. The two populations evolve independently (different RNG, different stagnation timing), exploring different formula families simultaneously, roughly doubling throughput. Dedup reconciles any cross-instance near-duplicates. The configs are identical except `population_dir`.

Both are launched with `setsid` so they survive terminal/agent disconnection (resilient to usage-limit interruptions).

```bash
bash scripts/evo_daemon.sh status     # pids for both instances
bash scripts/evo_daemon.sh ensure     # start any instance not running
bash scripts/evo_daemon.sh restart    # stop+start both (after a rebuild)
bash scripts/evo_daemon.sh stop       # stop both
```

---

## 15. Configuration Reference

From `config.toml` (config2.toml is identical except `population_dir`).

### `[rendering]`
| Key | Value | Meaning |
|-----|-------|---------|
| `default_width/height` | 512 | Full-render (save) resolution |
| `max_iter` | 192 | Full-render iteration depth |
| `bailout` | 4.0 | Escape radius (\|z\|>4 → escaped) |
| `colormap` | "turbo" | Fixed winner; maps low escapes to blue/cyan |
| `view_*` | ±2.0 | Viewer/CLI window only (not GA) |

### `[optimization]`
| Key | Value | Meaning |
|-----|-------|---------|
| `population_size` | 40 | Genomes per generation |
| `elitism_count` | 6 | Survivors + save-scan budget |
| `mutation_rate` | 0.20 | Per-coefficient perturbation prob |
| `mutation_scale` | 0.08 | Perturbation magnitude |
| `eval_width/height` | 64 | Fitness-eval resolution |
| `eval_max_iter` | 128 | Fitness-eval iteration depth |
| `restart_after_gens` | 30 | Stagnation → epoch restart |
| `novelty_weight` | 0.45 | Visual-novelty fitness weight |
| `novelty_k` | 5 | k-NN neighbors for novelty |
| `archive_size` | 150 | Rolling novelty/formula archive size |
| `self_replication_weight` | 0.35 | (legacy seed-rank knob) |
| `fractal_recursion_weight` | 0.35 | (legacy seed-rank knob) |
| `recursion_pred_weight` | 0.80 | **pred_recursion fitness weight** |
| `formula_diversity_weight` | 0.30 | **formula_diversity fitness weight** |
| `clip_pred_weight` | 0.50 | **pred_clip fitness weight** |

### `[output]`
| Key | Value | Meaning |
|-----|-------|---------|
| `min/max_entropy_prefilter` | 0.38 / 0.70 | Stage-1 raw PNG entropy band |
| `min_clip_score` | 0.522 | OR-gate CLIP threshold |
| `min_laion_score` | 5.30 | OR-gate LAION threshold |
| `min_beauty` | 0.35 | Fallback gate if scorer down |
| `min_save_distance` | 0.05 | Stage-2 diversity gate |

### `[dedup]`
| Key | Value | Meaning |
|-----|-------|---------|
| `similarity_threshold` | 0.97 | DCT similarity cutoff |
| `interval_hours` | 2.0 | Auto-dedup cadence |

---

## 16. What Was Tried and Removed

| Approach | Why removed |
|----------|-------------|
| **Transformer + backprop** | A transformer mapped a latent code → formula, trained by backprop on an escape-variance objective. It fought the GA, drove elites toward noise (the objective was misaligned with beauty), and was ~7× slower. Removing it for direct sparse GA evolution was a major simplification. |
| **Single-scale PNG entropy as fitness** | Rewards salt-and-pepper noise as strongly as structure. Replaced with multiscale entropy (geometric mean fine×coarse), which collapses on noise. |
| **AND gate (clip AND laion)** | Too restrictive — a CLIP-excellent fractal was blocked by borderline LAION and vice versa. Changed to OR. |
| **Relaxed AND gate (0.512 / 5.20)** | Too permissive — ~1000 saves/2h diluted quality. Replaced with OR at 0.522/5.30. |
| **Recursion-heavy seed ranking** (`clip + 0.35·self_rep + 0.35·rec`) | Deprioritized beautiful non-recursive genomes as epoch seeds. Changed to `2·clip + 0.15·laion + 0.20·self_rep + 0.20·rec`. |
| **Near-max PNG entropy save criterion** | A single noisy genome set `max` very high, blocking all structured-but-compressible beautiful fractals. Replaced with a flat band [0.38, 0.70]. |
| **Old boundary/cool-zone calibration targets (0.55)** | Fit the old per-pixel NN architecture; holomorphic formulas produce thin boundaries (~10–25%). Recalibrated to 0.20 / 0.12. |

---

## 17. File Map

```
src/
  main.rs              CLI entry (--config, --render); spawns dedup cleaner + run_evolution
  optimizer.rs         GA core: step(), evolve(), try_save(), restart_population(),
                       load_archive_seeds(), force_save(). The Optimizer struct + all weights.
  genome.rs            Genome + FormulaTerm; mutate(), crossover(), random_exotic();
                       feature vectors: formula_basis_normalized(), recursion_features(),
                       formula_weights(); view_bounds(); basis labels.
  fractal.rs           render_cpu_iter()/render_bounds() (escape-time iteration, GPU+CPU);
                       evaluate_fitness_full()/_batch(); best_entropy_view();
                       self_replication_score(); fractal_recursion_score(); field_intricacy();
                       dihedral_variants(); structure_vec(); recursion_candidates().
  fitness.rs           png_compression_entropy(); multiscale_entropy(); behavior_descriptor();
                       novelty_score(); is_degenerate(); beauty_score_full()/beauty_score()
                       (5-component: boundary/edge/entropy/self_sim/cool_zone); edge_density_fast().
  formula.rs           apply_formula() (the 58-basis evaluator, macro for f32/f64 + WGSL parity);
                       N_BASIS=58; basis_name().
  recursion_model.rs   RecursionModel: load() + predict() — used for BOTH pred_recursion & pred_clip.
  config.rs            Config structs + serde defaults for every toml field.
  aesthetic.rs         AestheticScorer: Python sidecar lifecycle, score_blocking(), poll/request,
                       status_line().
  colormap.rs          apply_colormap(); turbo + 9 hand-written colormaps; smooth t/max_iter mapping.
  render_gpu.rs        wgpu/Vulkan batch renderer; gpu_available(); WGSL shader (mirrors apply_formula).
  display.rs           Terminal TUI (generation status, save log, score readouts).
  io.rs                save_genome()/load_genome() (.nn JSON), save_png().
  bin/viewer.rs        Standalone interactive viewer (drag-zoom, pan); inline Config literal.

aesthetic_scorer.py    Python sidecar: CLIP ViT-L/14 zero-shot + LAION MLP. stdin path → stdout scores.

scripts/
  evo_daemon.sh        Manage 2 parallel daemon instances (start/ensure/stop/restart/status).
  fit_recursion_model.py  Train recursion predictor → recursion_model.json (Ridge + 5-fold CV).
  fit_clip_model.py    Train CLIP predictor → clip_model.json (Ridge λ=10 + 5-fold CV).
  dedup.py             Multi-scale DCT near-duplicate removal.
  analyze_diversity.py Formula diversity report (Shannon basis entropy, pairwise distance, unique sigs).
  diversity_reapply.sh Auto-tune formula_diversity_weight at loop boundaries.
  colormap_chart.py / colormap_switch.py / colormap_finalize.sh   Colormap experiment tooling.
  loop_analyze.py / loop_report.py / LOOP_RUNBOOK.md              Long-run loop tooling.

config.toml            Instance 1 config (populations/).
config2.toml           Instance 2 config (populations2/).
recursion_model.json   Trained recursion predictor (n=2088, cv_pearson=0.513).
clip_model.json        Trained CLIP predictor (n=2985, cv_pearson=0.452).
fractals/              Saved genomes: {id}.nn (JSON genome+scores) + {id}.png (512×512).
populations/ , populations2/   Per-instance population checkpoints.
colormap_scores.png    Colormap experiment result chart.
```

---

## 18. Key Invariants

These must remain true; breaking them silently corrupts behavior.

1. **`recursion_features()` (Rust) ≡ `features()` (both Python scripts)** — byte-for-byte, same 61 dims in the same order. Any change to one must change all three, and both models must be retrained.
2. **`apply_formula` (Rust CPU) ≡ WGSL shader (GPU)** — the 58-basis math must match, or GPU and CPU renders diverge.
3. **Save gate uses raw fine PNG entropy** (`beauty_entropy`), not multiscale. Multiscale is selection-only.
4. **`best_entropy_view()` never mutates the stored genome** — only the CLIP render uses the optimized view; the archived `.nn` keeps the evolved view params.
5. **`pred_clip` and `pred_recursion` share one struct and one feature vector** — only `clip_model.json` vs `recursion_model.json` (and thus the trained weights) differ.
6. **Missing model JSON ⇒ that fitness term = 0**, GA still runs. Never assume a model is present.
7. **Both daemon instances write to the same `fractals/`** but separate `populations*/`. Dedup must run (it's the only thing keeping the shared pool clean under relaxed gates).
8. **Turbo colormap is load-bearing for CLIP.** Changing it will drop CLIP below the save threshold and collapse the save rate.

---

## 19. Operations Playbook

### Check status
```bash
bash scripts/evo_daemon.sh status
```

### After any Rust change
```bash
cargo build --release && bash scripts/evo_daemon.sh restart
```

### Retrain the proxy models (do when archive grows ~500+ genomes)
```bash
python3 scripts/fit_clip_model.py        # → clip_model.json, prints cv_pearson
python3 scripts/fit_recursion_model.py   # → recursion_model.json
bash scripts/evo_daemon.sh restart       # models are JSON; no rebuild needed
```
If a model's `cv_pearson < 0.20`, set its weight to 0 in config (the structure no longer predicts the target).

### Inspect recent quality
```bash
python3 - <<'PY'
import glob, json, os, statistics
recs = [json.load(open(f)) for f in glob.glob('fractals/*.nn')]
laion=[r['laion_score'] for r in recs if r.get('laion_score')]
clip =[r['clip_score']  for r in recs if r.get('clip_score')]
print(f"n={len(recs)}  LAION μ={statistics.mean(laion):.4f} max={max(laion):.4f}"
      f"  CLIP μ={statistics.mean(clip):.4f} max={max(clip):.4f}")
PY
```

### Current archive snapshot (2026-06-28)
- **3457 saved genomes**; LAION μ 5.317 (max 5.414), CLIP μ 0.519 (max 0.532)
- Formula diversity ~0.87; ~96% unique formula signatures

---

## 20. What to Try Next

Hypotheses, roughly by expected impact:

1. **Retrain `clip_model.json` on the now-3457-genome archive** (last trained at 2985). More data → better aesthetic steering.
2. **Automate proxy retraining** — cron-like retrain of both models every ~500 saves + daemon restart, fully unsupervised.
3. **Add a LAION predictor** (`laion_model.json` targeting `laion/10`). LAION has more variance than CLIP → potentially more learnable; add `laion_pred_weight` to fitness.
4. **Evolve view params toward higher entropy** — currently view_cx/cy/zoom mutate randomly. A small fitness signal toward higher-multiscale-entropy views would make genomes evolve better default framing, not just better formulas (today only the save-time best-view search compensates).
5. **Adaptive mutation scale** — when recent-save aesthetic variance is low (population converged), temporarily raise `mutation_scale` to re-explore.
6. **Curriculum on `bailout`/`max_iter`** — deeper iteration late in an epoch could surface finer recursion the shallow eval pass misses.
