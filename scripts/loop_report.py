#!/usr/bin/env python3
"""Report on loop winners not yet reported: renders 3 angles x 4 T views (quat-dag-thumbs
--only), a one-row check strip per winner, and the terms taste / t_cont / res_diff plus
whether the BULB op survived. Usage: loop_report.py DIR COLLAPSE"""
import csv, glob, json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw
D, COL = sys.argv[1], sys.argv[2]
done_f = f'{D}/_reported.json'
done = set(json.load(open(done_f))) if os.path.exists(done_f) else set()
new = [f for f in sorted(glob.glob(f'{D}/*.nn'), key=os.path.getmtime) if os.path.basename(f) not in done]
if not new: print('no new winners'); sys.exit(0)
tmp = tempfile.mkdtemp(prefix='rep_', dir='/tmp')
for f in new: os.symlink(os.path.abspath(f), f'{tmp}/{os.path.basename(f)}')
subprocess.run(['target/release/explorer', 'quat-dag-thumbs', '--dir', tmp, '--views', '--t-linear', '--gpu', '--collapse', COL], capture_output=True)
subprocess.run(['target/release/explorer', 'quat-metric-terms', '--dir', tmp, '--collapse', COL, '--terms', 'taste,t_cont,res_diff', '--gpu', '--out', f'{tmp}/t.csv'], capture_output=True)
T = {r['file']: r for r in csv.DictReader(open(f'{tmp}/t.csv'))}
S = 224
sheet = Image.new('RGB', (6 * S, len(new) * (S + 16)))
dr = ImageDraw.Draw(sheet)
for k, f in enumerate(new):
    b = os.path.basename(f); st = f'{tmp}/{b[:-3]}'; y = k * (S + 16)
    g = json.load(open(f)); prog = g['program']
    def dyn(i):
        n = prog[i]
        if n['op'] in (0, 1): return True
        if n['op'] == 2: return False
        return dyn(n['a']) or (n['op'] in (17, 18, 19, 20, 21) and dyn(n['b']))
    bulbs = ['dyn' if dyn(n['b']) else f"{prog[n['b']]['kre']:.1f}" for n in prog if n['op'] == 21]
    r = T.get(b, {})
    line = f"{b[:60]}  taste {float(r.get('taste', 'nan')):.2f}  t_cont {float(r.get('t_cont', 'nan')):.2f}  res {float(r.get('res_diff', 'nan')):.3f}  bulbs {bulbs}"
    print(line)
    dr.text((4, y + 2), line, fill=(255, 255, 255))
    for j, v in enumerate([0, 1, 2, 3, 8, 16]):
        p = f'{st}_view_{v:02d}.png'
        if os.path.exists(p): sheet.paste(Image.open(p).convert('RGB'), (j * S, y + 16))
    done.add(b)
out = f'{D}/_report_{len(done):03d}.png'
sheet.save(out); print('sheet', out)
json.dump(sorted(done), open(done_f, 'w'))
