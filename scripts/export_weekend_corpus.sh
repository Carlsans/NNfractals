#!/usr/bin/env bash
# Builds a large, diverse complex-field (re/im/mag/tensor) training corpus
# across every existing vae-explore pool (multiple formulas + GA genomes)
# plus a batch of fresh novelty-ranked fractals_1 genomes — the weekend
# architecture-research dataset, not a single-formula sample.
#
# Each source gets its OWN subdirectory — zone_NNNN stems are only unique
# WITHIN a pool (e.g. mandelbrot_vae/zone_0000.nn and burning_ship_vae/
# zone_0000.nn are different zones), so dumping every pool into one flat
# directory would silently overwrite files across pools. Training scripts
# take --dirs as a list, so per-pool subdirectories cost nothing at train
# time — gather_pairs/gather_paths already accept multiple dirs.
set -e
cd "$(dirname "$0")/.."
OUT=explorer_out/weekend_complex_corpus
mkdir -p "$OUT"

echo "=== existing vae-explore pools ==="
./target/release/explorer complex-export explorer_out/mandelbrot_vae "$OUT/mandelbrot" --res 512 --limit 923
./target/release/explorer complex-export explorer_out/burning_ship_vae "$OUT/burning_ship" --res 512 --limit 590
./target/release/explorer complex-export explorer_out/celtic_mandelbrot_vae "$OUT/celtic_mandelbrot" --res 512 --limit 700
./target/release/explorer complex-export "explorer_out/tricorn_(mandelbar)_vae" "$OUT/tricorn" --res 512 --limit 463
./target/release/explorer complex-export explorer_out/9aa9885ad83f97cc_vae "$OUT/genome_9aa988" --res 512 --limit 273
./target/release/explorer complex-export explorer_out/9fd45ec11ffe7187_vae "$OUT/genome_9fd45e" --res 512 --limit 93

echo "=== fresh novelty-ranked fractals_1 genomes (single view each, top 250 by novelty_score) ==="
python3 -c "
import json, glob
rows = []
for f in glob.glob('fractals_1/*.nn'):
    try:
        g = json.load(open(f))
        rows.append((g.get('novelty_score', 0), f))
    except Exception:
        pass
rows.sort(reverse=True)
for _, f in rows[:250]:
    print(f)
" > /tmp/claude-1000/-home-carl-rust-projects-NNfractals/f717be76-d887-47cc-a431-78a4d7202c8c/scratchpad/top250_novelty.txt

# One complex-export call (symlinked into a scratch dir) instead of 250
# separate process spawns — cmd_complex_export accepts a directory input.
TOPDIR=/tmp/claude-1000/-home-carl-rust-projects-NNfractals/f717be76-d887-47cc-a431-78a4d7202c8c/scratchpad/top250_novelty_links
rm -rf "$TOPDIR"; mkdir -p "$TOPDIR"
while read -r nn; do
  ln -s "$(pwd)/$nn" "$TOPDIR/$(basename "$nn")"
done < /tmp/claude-1000/-home-carl-rust-projects-NNfractals/f717be76-d887-47cc-a431-78a4d7202c8c/scratchpad/top250_novelty.txt
./target/release/explorer complex-export "$TOPDIR" "$OUT/fractals_1_novelty" --res 512 --limit 250

echo "=== DONE. Per-pool counts: ==="
for d in "$OUT"/*/; do
  echo "$d: $(ls "$d"*_re.png 2>/dev/null | wc -l)"
done
