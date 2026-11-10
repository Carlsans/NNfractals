# Principles from the six RABC planes (2026-10-18)

Raw 4D slices (`explorer quat-planes`, other two axes at 0, ±1.6, max_iter 60), no collapse or rendering.
Genomes: 120 rated S3 (499 pairs), 300 most-rated slice (1602 pairs), 34 Starred, best/ring tournament picks.

## 1. Symmetry is decided by the formula's ingredients
- Mandelbrot mode + only "rotation-friendly" ops (z^n, sin, conj, +, ·, real/complex constants): RA = RB = RC and the
  vector planes AB = AC = BC are perfect circles → the whole 4D set is ONE 2D picture (the RA plane) spun around the
  real axis (SO(3) symmetry). Every 3D view is a surface of revolution.
- Julia mode with a complex constant (has an A part): RB = RC, symmetric under rotation in the B–C plane (SO(2) about A)
  → lathe/ring balls around one axis. This is the MAJORITY class (~60% of all rated genomes).
- Only abs folds (absfold/absim) combined with Julia + warp reliably break everything ("none" class, ~22%); bulb also breaks it.
- Root cause: DAG constants only fill the (r, a) quaternion parts, so the (b, c) directions are interchangeable.

## 2. Carl prefers smooth escape fields (both modes)
Pairwise accuracy (how often the LOWER value won):
| plane property | S3 | slice |
|---|---|---|
| boundary length of the bounded set | 64.7% (z 6.5) | 60.3% (z 8.1) |
| bounded fraction | 58.6% | 62.0% |
| fine detail (Laplacian of log escape time) | 66.5% (z 7.4) | 50% |
Favourites: few, large, smooth bounded parts and a gently varying escape field. Rejects: solid blobs (slice, 66% bounded)
or dusty/high-frequency sets (S3).

## 3. Asymmetry matters in slice mode
Slice: the more asymmetric (RA≠RB≠RC) one won 56–59% (z 4.5). S3: no preference (the S3 projection already mixes axes).

## Combined (rank sums, no fitting)
- S3: −boundary −bounded −detail → 67.1% (taste model, in-sample: 72%).
- slice: −boundary −bounded +asymmetry → 63.0%.
