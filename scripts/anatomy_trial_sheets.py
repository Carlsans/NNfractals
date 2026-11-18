"""Rank screened anatomy trials and build labelled contact sheets (best first).
argv: run_dir [per_sheet=30] [cols=6]"""
import json, os, subprocess, sys
run = sys.argv[1]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 30
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
recs = [json.loads(l) for l in open(f'{run}/results.jsonl')]
ok = [r for r in recs if r.get('layers') is not None and os.path.exists(f"{run}/{r['id']}/thumb.png")]
ok.sort(key=lambda r: (-r['layers'], -(r['draw_m'] or 0)))
print(f'{len(recs)} trials, {len(ok)} with thumbnails; layers: ' + ' '.join(str(r['layers']) for r in ok[:40]))
with open(f'{run}/ranking.txt', 'w') as f:
    for i, r in enumerate(ok, 1):
        f.write(f"{i:3d} {r['layers']:3d} layers {r['inside']:5.1f}% inside {r['draw_m']:6.1f} m  {r['genome']} {r['plane']} p{r['phase']}\n")
os.makedirs(f'{run}/sheets', exist_ok=True)
for s in range(0, len(ok), per):
    tiles = []
    for i, r in enumerate(ok[s:s + per], s + 1):
        t = f"{run}/sheets/t{i:03d}.png"
        label = f"{i}. {r['layers']} layers  {os.path.basename(r['genome'])[:22]} {r['plane']}"
        subprocess.run(['magick', f"{run}/{r['id']}/thumb.png", '-resize', '300x', '-gravity', 'south', '-splice', '0x16', '-pointsize', '12', '-annotate', '+0+1', label, t])
        tiles.append(t)
    out = f"{run}/sheets/sheet_{s // per + 1:02d}.png"
    subprocess.run(['magick', 'montage', *tiles, '-tile', f'{cols}x', '-geometry', '+3+3', out])
    print(out)
