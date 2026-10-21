#!/usr/bin/env python3
"""Per-genome check sheet from quat-dag-thumbs --views output: rows = 3 camera angles
(views 0,2,4 of 6), columns = the 4 T values. Writes <stem>_check.png next to the .nn."""
import glob, os, sys
from PIL import Image, ImageDraw
for d in sys.argv[1:]:
    for nn in sorted(glob.glob(f'{d}/*.nn')):
        stem = nn[:-3]; tiles = []
        for a in (0, 2, 4):
            tiles.append([f'{stem}_view_{a * 4 + c:02d}.png' for c in range(4)])
        if not all(os.path.exists(p) for r in tiles for p in r): print('missing views', nn); continue
        S = 224; im = Image.new('RGB', (4 * S, 3 * S + 18), (0, 0, 0))
        for i, r in enumerate(tiles):
            for j, p in enumerate(r): im.paste(Image.open(p).convert('RGB'), (j * S, 18 + i * S))
        ImageDraw.Draw(im).text((4, 3), os.path.basename(stem) + '   cols: T0..T3   rows: angle 0/120/240', fill=(255, 255, 255))
        im.save(f'{stem}_check.png')
