#!/usr/bin/env python3
"""One sheet per recipe dir: row per genome; cols = angle 0 at T0..T3, then angles 120/240 at T0."""
import glob, os, sys
from PIL import Image, ImageDraw
S = 224
for d in sys.argv[1:]:
    nns = sorted(glob.glob(f'{d.rstrip("/")}/*.nn'))
    im = Image.new('RGB', (6 * S, len(nns) * (S + 16)), (0, 0, 0)); dr = ImageDraw.Draw(im)
    for i, nn in enumerate(nns):
        st = nn[:-3]; y = i * (S + 16)
        dr.text((4, y + 2), os.path.basename(st) + '  | T0 T1 T2 T3 (angle 0) | angle120 | angle240', fill=(255, 255, 255))
        for j, v in enumerate([0, 1, 2, 3, 8, 16]):
            p = f'{st}_view_{v:02d}.png'
            if os.path.exists(p): im.paste(Image.open(p).convert('RGB'), (j * S, y + 16))
    im.save(f'{d.rstrip("/")}/_recipe_sheet.png')
