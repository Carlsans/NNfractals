# NNfractals — Principles & Design Reference

*Last updated: 2026-06-28. Audience: developers and AI agents resuming work on this codebase.*

---

## What This Program Does

NNfractals is a **genetic algorithm that autonomously evolves mathematical fractal formulas** and saves the most aesthetically beautiful results. It runs indefinitely as a background daemon, generating and scoring fractal images, keeping only those that pass aesthetic quality gates.

The core loop: evolve formulas → render images → score aesthetics → save the beautiful ones → repeat.

---

## The Formula Representation

Every fractal is defined by a **sparse iterated map**:

```
z_new = Σ coeff_i · φ_basis_i(z, c)
```

where:
- `z` is the current complex iterate, `c` is the parameter point
- `φ_basis_i` is one of **58 holomorphic basis functions** (powers, trig, exp, rational, etc.)
- Each genome carries **2–8 terms**, each term = `(basis_index, re, im)` — a complex coefficient on one basis function
- The genome is **sparse**: most of the 58 bases are absent; only a handful drive each fractal's character

The 58 bases include: `z`, `z²`, `z³`, `z⁴`, `c`, `zc`, `z²c`, `1/(z²+c)`, `sin(z)`, `cos(z)`, `tan(z)`, `exp(z)`, `(z²−1)/(z²+1)`, `cosh`, `sinh`, `log(z)`, and ~40 more holomorphic combinations.

**Why sparse?** Dense weight vectors would make all fractals similar. Sparse evolution naturally produces different formula families (Mandelbrot-variants, Julia-like sets, rational maps, etc.) that look visually distinct.

---

## Genetic Algorithm Structure

### Population
- 40 individuals, evolved in discrete generations
- **Elitism**: top 6 by fitness survive unchanged each generation
- **Mutation**: 20% per-coefficient perturbation probability, scale 0.08 (small steps in coefficient space)
- **Crossover**: mix of basis terms from two parents + re-centering
- **Epoch restart**: after 30 stagnant generations (fitness not improving by >0.005), restart the population with archive seeds + fresh randoms

### Epoch Restart Seeding
When an epoch stagnates, the new population is seeded from:
1. **6 archive seeds** — ranked by `2·clip_score + 0.15·(laion/10) + 0.20·self_replication + 0.20·fractal_recursion`. Aesthetic quality (CLIP) is weighted 2× to bias restarts toward the most beautiful formula families.
2. **8 random injections**: 8 "exotic" genomes forced to use rare basis functions (z², sin, tan, abs, 1/(z²+1), (z²−1)/(z²+1)), promoting exploration.
3. **Remainder**: mutations of the seeds.

**Why aesthetic-first seeding?** Early epochs used recursion-heavy seeding. Beautiful-but-non-recursive genomes were deprioritized. Doubling the CLIP weight ensures restarts start from genuinely beautiful ancestors, not just technically impressive ones.

---

## Fitness Function (per generation)

Every generation, all 40 genomes are evaluated and ranked by:

```
fitness = multiscale_entropy
        + 0.45 · visual_novelty
        + 0.80 · pred_recursion
        + 0.30 · formula_diversity
        + 0.50 · pred_clip
```

Each term explained below.

### 1. Multiscale Entropy (primary — unlabeled weight = 1.0)

**What:** Geometric mean of PNG compression entropy at two scales:
- **Fine** (64×64): `compressed_size / raw_size` of the colormap-rendered eval image
- **Coarse** (16×16 average-pool of the 64×64): same metric after 4× spatial downsampling

**Formula:** `sqrt(fine_entropy × coarse_entropy)`

**Why geometric mean, not just fine entropy?** Pure fine-scale entropy rewards salt-and-pepper noise (noise compresses poorly → high entropy → high score). Noise averages to near-uniform at coarse scale (coarse entropy → 0). Structured fractals with genuine detail stay complex at both scales. The geometric mean collapses when either scale is uninformative — punishing noise while rewarding structure.

**Implementation:** `src/fitness.rs: multiscale_entropy()`

### 2. Visual Novelty (weight 0.45)

**What:** Average L2 distance to the k=5 nearest neighbors in a 32-bin escape-time histogram space.

**How:** Each genome's escape-time distribution (how many pixels escape at each iteration depth) forms a 32-bin histogram — its behavioral descriptor. Novelty = mean distance to the 5 most similar already-seen genomes in this descriptor space.

**Archive:** Rolling window of 150 most recently evaluated descriptors.

**Why:** Without novelty pressure, the GA converges to one formula family. Novelty forces exploration of visually distinct fractal types (spirals, web-like, bubble-like, recursive, etc.).

**Implementation:** `src/fitness.rs: behavior_descriptor()`, `novelty_score()`

### 3. Predicted Recursion (weight 0.80)

**What:** A formula-only linear predictor of how strongly the fractal contains embedded miniature copies of the whole set (baby-Mandelbrots / fractal recursion).

**How:** Ridge regression (`recursion_model.json`) trained on 2088 saved genomes with measured `fractal_recursion` scores. Feature vector (61-dim):
- `f[0..58]`: Σ |coeff| for each basis (how strongly each basis drives the formula)
- `f[58]`: number of terms
- `f[59]`: total coefficient magnitude
- `f[60]`: `basis7 × (basis0 + basis1 + basis2 + basis3)` — a c×z-power interaction term that correlates with Mandelbrot-like self-similarity

**Why formula-only?** Rendering 40 genomes per generation to measure true recursion would be slow. The linear predictor approximates it from formula structure in microseconds. Cross-validated Pearson r ≈ 0.51 — weak individually, but meaningful as a population-level bias.

**Measured at save time:** The actual `fractal_recursion` score (template-matching with dihedral variants + boundary descent) is only computed for genomes that pass the save gate. This label is what the predictor was trained on.

**Implementation:** `src/recursion_model.rs`, `src/genome.rs: recursion_features()`, `scripts/fit_recursion_model.py`

### 4. Formula Diversity (weight 0.30)

**What:** Average L2 distance to k=5 nearest neighbors in a **normalized 58-dim basis-weight space** — same k-NN novelty mechanism as visual novelty, but in formula structure space rather than rendering space.

**How:** Each genome's formula is projected to a 58-dim vector where each element = Σ|coeff| on that basis, then L2-normalized. This captures which basis functions dominate the formula (direction-sensitive, not scale-sensitive).

**Archive:** Rolling window of 150 most recently evaluated formula vectors.

**Why a separate diversity metric from visual novelty?** Visual novelty operates in rendering space (escape-time histograms). Two formulas that render to different escape-time statistics could still be structurally similar (same basis mix, different coefficients). Formula diversity rewards genuinely different formula families.

**Result of 3×2h diversity loop:** Formula diversity improved from 0.70 → 0.87 across iterations, with 96–97% unique formula signatures in each window. Weight 0.30 was stable throughout.

**Implementation:** `src/genome.rs: formula_basis_normalized()`, `src/optimizer.rs`

### 5. Predicted CLIP Score (weight 0.50)

**What:** A formula-only linear predictor of the CLIP aesthetic score, trained on the 3000+ saved archive genomes.

**How:** Same infrastructure as pred_recursion (`clip_model.json`, same Ridge regression, same 61-dim feature vector). Trained with stronger regularization (λ=10 vs λ=5) because CLIP scores vary in a narrow band [0.47, 0.53] — harder to predict than recursion [0, 1].

**Cross-validated Pearson r ≈ 0.45** — the formula partially predicts CLIP quality. The GA uses this to steer toward formula families that historically produce beautiful images.

**Why add this?** CLIP and LAION scores were stationary — the GA had no aesthetic signal in its fitness function. These were only a binary save gate. Adding pred_clip closes the loop: the GA now actively selects for formula families that CLIP prefers.

**Implementation:** `src/recursion_model.rs` (same struct), `scripts/fit_clip_model.py`

---

## GPU Evaluation

All 40 genomes per generation are batch-rendered on the GPU (Vulkan via wgpu) at 64×64 resolution in a single dispatch. The GPU shader evaluates the iterated map `z_new = Σ coeff_i · φ_i(z, c)` for every pixel simultaneously.

**Fallback:** If GPU is unavailable, Rayon parallelizes CPU rendering across threads.

**Implementation:** `src/render_gpu.rs`, `src/fractal.rs: evaluate_fitness_batch()`

---

## Save Gate (what gets saved)

A genome is saved as a PNG + JSON file only if it passes all stages:

### Stage 1 — PNG Entropy Prefilter (fast, Rust)
```
min_entropy_prefilter = 0.38  ≤  raw_fine_png_entropy  ≤  max_entropy_prefilter = 0.70
```
Rejects boring uniform fractals (below 0.38) and salt-and-pepper noise (above 0.70). Note: this uses **raw fine-scale** entropy, not multiscale — the noise rejection at coarse scale is for selection fitness, not save filtering.

### Stage 2 — Diversity Gate (fast, Rust)
Minimum L2 distance to all previously saved behavioral descriptors: `min_save_distance = 0.05`. Prevents saving near-duplicate fractals within a run. Dedup (every 2h) handles the rest.

### Stage 3 — Best-View Search (fast, Rust)
Before rendering the full 512×512 image, the save candidate undergoes a **3×3 view search**: 9 candidate views (±15%/zoom offset in both axes) are rendered at 64×64 and scored by multiscale entropy. The view with the highest score is used for the full render and CLIP scoring.

**Why:** The GA evolves view params (view_cx, view_cy, view_zoom) to maximize fitness, not CLIP composition. This search finds the locally best composition before the expensive CLIP call. The genome's own view params are unchanged — only the render changes.

**Implementation:** `src/fractal.rs: best_entropy_view()`

### Stage 4 — Full Render + Aesthetic Scoring (expensive, Python sidecar)
The 512×512 image is rendered, saved to a temp file, and scored by a persistent Python sidecar process (`aesthetic_scorer.py`) that runs:
- **CLIP ViT-L/14**: zero-shot aesthetic classification → `clip_score ∈ [0, 1]`
- **LAION MLP**: trained on human aesthetic ratings, fed CLIP embeddings → `laion_score ∈ [0, 10]`

### Stage 4 — OR Gate
```
passes = clip_score ≥ 0.522  OR  laion_score ≥ 5.30
```
A fractal saves if it is excellent on **either** metric. This privileges:
- Fractals with strong CLIP aesthetics (cool-blue, structured, photographic quality)
- Fractals with strong LAION ratings (human-preferred aesthetics, sometimes warmer)

Previously AND logic — changed to OR so high-LAION fractals aren't blocked by borderline CLIP, and vice versa.

---

## Colormap: Turbo (Fixed)

After a 4-hour experiment cycling through 13 colormaps (30-min windows each), **turbo** was decisive:

| Colormap | Saves (90 min) | CLIP mean |
|----------|---------------|-----------|
| **turbo** | **382** | **0.5227** |
| lava | 8 | 0.5189 |
| neon | 9 | 0.5085 |
| aurora | 2 | 0.5018 |
| galaxy | 2 | 0.4839 |

**Why turbo wins with CLIP:** Turbo maps low escape times (most of the image) to blue/cyan. CLIP ViT-L/14 — trained on photographic images — associates cool blues with high-quality imagery (ocean, sky). Warm palettes (lava, sunset) drop CLIP below 0.520 because they look more like graphic art than photographic subjects.

All 6 custom colormaps (lava, aurora, galaxy, sunset, arctic, ember) were also tested and all failed below 0.520 CLIP.

---

## Deduplication

Every 2 hours, `scripts/dedup.py` runs automatically and removes near-duplicate fractals from `fractals/`. Similarity is measured as a geometric mean of DCT-based similarity at 3 scales (16×16, 32×32, 64×64 grayscale). Pairs with similarity ≥ 0.97 are deduplicated (weaker-LAION copy deleted).

This allows a **relaxed save gate** (min_save_distance=0.05 instead of 0.08) while keeping the pool clean.

---

## Two Parallel Instances

Two evolution daemons run simultaneously, managed by `scripts/evo_daemon.sh`:

| Instance | Config | Population dir |
|----------|--------|---------------|
| 1 | `config.toml` | `populations/` |
| 2 | `config2.toml` | `populations2/` |

Both instances write saves to the shared `fractals/` directory. Independent populations explore different formula families simultaneously. Dedup reconciles any near-duplicates.

**Daemon commands:**
```bash
bash scripts/evo_daemon.sh status    # show pids for both instances
bash scripts/evo_daemon.sh ensure    # start any instance not running
bash scripts/evo_daemon.sh restart   # stop+start both (e.g. after rebuild)
bash scripts/evo_daemon.sh stop      # stop both
```

---

## Measured Scores at Save Time

Beyond the GA fitness signals, each saved genome carries measurements made only once, at save time:

### Fractal Recursion Score
Boundary descent: render at base zoom → find richest boundary point → zoom in 3 levels (8× each) searching for template matches with dihedral variants of the whole-set template. Score = best match found.

### Self-Replication Score
Render the fractal at two consecutive zoom levels; compare boundary complexity persistence. A Mandelbrot-like set maintains complex boundaries under zoom; simpler formulas decay.

Both scores travel with the `.nn` JSON file and are used in archive seed ranking but not in per-generation GA fitness (too expensive to compute for all 40 genomes).

---

## What Was Tried and Removed

| Approach | Why removed |
|----------|-------------|
| **Transformer + backprop** | Fought the GA; drove elites toward noise (escape-variance objective misaligned with beauty); ~7× slowdown. Removed in early development. |
| **Single-scale PNG entropy as fitness** | Rewards salt-and-pepper noise as strongly as structured fractals. Replaced with multiscale entropy (geometric mean fine×coarse). |
| **AND gate for CLIP+LAION** | Too restrictive — CLIP-excellent fractals blocked by borderline LAION and vice versa. Changed to OR logic. |
| **Relaxed AND gate (0.512/5.20)** | Too permissive — ~1000 saves/2h, quality diluted. Replaced with OR gate at original thresholds (0.522/5.30). |
| **Recursion-heavy seed ranking** | Deprioritized beautiful non-recursive genomes as seeds. Changed to 2×CLIP + LAION + small recursion bonus. |

---

## File Map

```
src/
  main.rs              — CLI entry point (--config, --render flags)
  optimizer.rs         — GA loop: step(), try_save(), load_archive_seeds()
  genome.rs            — Genome struct, mutation, crossover, feature vectors
  fractal.rs           — render_cpu(), evaluate_fitness_full(), best_entropy_view()
  fitness.rs           — multiscale_entropy(), behavior_descriptor(), novelty_score(), beauty_score_full()
  formula.rs           — apply_formula() macro, N_BASIS=58, basis_name()
  recursion_model.rs   — RecursionModel struct: load(), predict() [used for both recursion + CLIP]
  config.rs            — Config structs for all toml fields
  aesthetic.rs         — AestheticScorer: Python sidecar management, score_blocking()
  colormap.rs          — apply_colormap(), turbo + custom colormaps
  render_gpu.rs        — GPU batch rendering via wgpu/Vulkan
  bin/viewer.rs        — Interactive fractal viewer (separate binary)

scripts/
  evo_daemon.sh        — Manage 2 parallel daemon instances (start/stop/restart/ensure/status)
  fit_recursion_model.py — Train recursion predictor on archive → recursion_model.json
  fit_clip_model.py    — Train CLIP predictor on archive → clip_model.json
  dedup.py             — Near-duplicate removal (multi-scale DCT similarity)
  analyze_diversity.py — Formula diversity analysis (Shannon entropy, pairwise distance, unique signatures)
  diversity_reapply.sh — Auto-tune formula_diversity_weight at iteration boundaries
  colormap_chart.py    — Generate colormap comparison bar chart

config.toml            — Instance 1 config (populations/ dir)
config2.toml           — Instance 2 config (populations2/ dir)
recursion_model.json   — Trained recursion predictor (n=2088, cv_pearson=0.513)
clip_model.json        — Trained CLIP predictor (n=2985, cv_pearson=0.452)
fractals/              — Saved genomes: {id}.nn (JSON) + {id}.png
populations/           — Instance 1 population checkpoint
populations2/          — Instance 2 population checkpoint
```

---

## Current Archive Stats (2026-06-28)

- **Total saved genomes:** 3457
- **LAION:** mean 5.317, max 5.414, min 5.147
- **CLIP:** mean 0.519, max 0.532, min 0.475
- **Formula diversity (fdiv_mean):** ~0.87 across recent saves
- **Unique formula signatures:** ~96% of saves have distinct top-3 basis combinations

---

## Key Invariants

1. **The save gate uses raw fine PNG entropy**, not multiscale. The coarse scale is for selection fitness only.
2. **The genome's view params are unchanged by best_entropy_view()**. The archive keeps the evolved view; only the CLIP render uses the optimized view.
3. **pred_clip and pred_recursion use the same 61-dim feature vector** (`recursion_features()`). Both models are stored in `RecursionModel` structs; only the training target differs.
4. **Dedup runs inside the evolution loop** (via subprocess every 2h), not as an external cron. Both instances trigger it independently from their `spawn_dedup_cleaner()` call.
5. **The recursion model was trained on 2088 genomes** from an earlier phase; clip_model was trained on 2985. Both should be periodically retrained as the archive grows.

---

## How to Retrain the Proxy Models

```bash
# Retrain CLIP predictor (do this when archive grows by ~500+ genomes)
python3 scripts/fit_clip_model.py
# → overwrites clip_model.json, prints cv_pearson

# Retrain recursion predictor
python3 scripts/fit_recursion_model.py
# → overwrites recursion_model.json, prints cv_pearson

# Restart daemons to pick up new models (no rebuild needed — models are JSON)
bash scripts/evo_daemon.sh restart
```

---

## What to Try Next

Hypotheses worth testing, roughly in order of expected impact:

1. **Retrain clip_model.json periodically** — archive is now 3457 genomes (was 2985 when last trained). More data = better predictor.
2. **Periodic formula-space reseeding** — run `fit_clip_model.py` + `fit_recursion_model.py` on a cron-like cadence (every 500 saves) and restart without human intervention.
3. **LAION predictor** — train a separate `laion_model.json` targeting `laion_score/10` (more variance than CLIP → may be more learnable). Add `laion_pred_weight` to fitness.
4. **Adaptive mutation scale** — if CLIP/LAION variance in recent saves is low (population converged aesthetically), increase `mutation_scale` temporarily to explore.
5. **View parameter evolution pressure** — currently view_cx/cy/zoom are mutated randomly. Adding a small fitness signal toward views with higher multiscale entropy would help the genome evolve better default compositions, not just better formulas.
