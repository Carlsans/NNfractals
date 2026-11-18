"""Replay a Marlin pen-plot G-code file (G0/G1/G5 + Z pen) onto a canvas, independently
of the Rust writer, and compare it with the SVG raster. Usage: check_gcode.py FILE.gcode SVG.png W_MM H_MM PX_PER_MM OUT.png"""
import re, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
gc, svg_png, W, H, S, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
img = Image.new('L', (int(W * S), int(H * S)), 255)
dr = ImageDraw.Draw(img)
x = y = 0.0; z = None; down = False
downs = ups = draws_up = 0; xs = []; ys = []; g5 = 0
FLIP = not (len(sys.argv) > 7 and sys.argv[7] == "noflip")
px = (lambda X, Y: (X * S, (H - Y) * S)) if FLIP else (lambda X, Y: (X * S, Y * S))  # bed Y up -> image y down, unless noflip
for line in open(gc):
    line = line.split(';')[0].strip()
    if not line: continue
    w = dict((m[0], float(m[1:])) for m in re.findall(r'[A-Z]-?[\d.]+', line[2:] if line[:2] in ('G0','G1','G5') else line))
    cmd = line.split()[0]
    if cmd in ('G0', 'G1') and 'Z' in w:
        nd = w['Z'] <= 0.0
        downs += nd and not down; ups += (not nd) and down; down = nd
    if cmd in ('G0', 'G1', 'G5') and ('X' in w or 'Y' in w):
        nx, ny = w.get('X', x), w.get('Y', y)
        if cmd == 'G0' and down: draws_up += 1
        if cmd != 'G0' and not down: draws_up += 1      # drawing move with the pen up = bug
        if cmd == 'G5':
            g5 += 1
            c1 = (x + w['I'], y + w['J']); c2 = (nx + w['P'], ny + w['Q'])
            pts = []
            for k in range(33):
                t = k / 32; s = 1 - t
                pts.append(px(s**3*x + 3*s*s*t*c1[0] + 3*s*t*t*c2[0] + t**3*nx, s**3*y + 3*s*s*t*c1[1] + 3*s*t*t*c2[1] + t**3*ny))
            dr.line(pts, fill=0, width=3)
        elif cmd == 'G1' and down:
            dr.line([px(x, y), px(nx, ny)], fill=0, width=3)
        if down: xs.append(nx); ys.append(ny)
        x, y = nx, ny
img.save(out)
a = np.array(img) < 128
b = np.array(Image.open(svg_png).convert('L').resize(img.size)) < 230  # any non-white ink (yellow is light)
dil = lambda m: np.array(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
print(f'{gc.split("/")[-1][-9:]}: pen downs {downs}, ups {ups}, G5 {g5}, moves with wrong pen state {draws_up}; '
      f'drawn X {min(xs):.2f}..{max(xs):.2f} Y {min(ys):.2f}..{max(ys):.2f}; '
      f'ink missing vs SVG {np.mean(b & ~dil(a)) / max(np.mean(b), 1e-9):.4%}, extra {np.mean(a & ~dil(b)) / max(np.mean(a), 1e-9):.4%}')
