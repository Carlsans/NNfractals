"""Screen many anatomy mosaics at low resolution. Run from the NNfractals root.
argv: candidates.txt (lines: genome_path plane [phase]) out_dir. Appends results.jsonl; skips done ids.
Stop with: touch <out_dir>/STOP"""
import json, os, re, subprocess, sys
cands, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
res = f'{out}/results.jsonl'
done = set()
if os.path.exists(res):
    for l in open(res):
        try: done.add(json.loads(l)['id'])
        except Exception: pass
for line in open(cands):
    if os.path.exists(f'{out}/STOP'): break
    p = line.split()
    if len(p) < 2: continue
    g, plane = p[0], p[1]
    phase = p[2] if len(p) > 2 else '0.5'
    zoom = p[3] if len(p) > 3 else '0.6'
    tid = f"{os.path.basename(g)[:-3]}__{plane}_p{phase}" + ("" if zoom == '0.6' else f"_z{zoom}")
    if tid in done: continue
    d = f'{out}/{tid}'
    r = subprocess.run(['target/release/explorer', 'quat-anatomy', '--genome', g, '--plane', plane, '--phase', phase,
                        '--zoom', zoom, '--compose', 'mosaic', '--cell', '1.0', '--svg-only', '--out-dir', d],
                       capture_output=True, text=True, timeout=900)
    txt = r.stdout + r.stderr
    m = re.search(r'(\d+) layers with at least 1 m', txt)
    ins = re.search(r'([\d.]+)% inside', txt)
    dr = re.search(r'draw ([\d.]+) m', txt)
    rec = {'id': tid, 'genome': g, 'plane': plane, 'phase': phase, 'zoom': zoom, 'layers': int(m.group(1)) if m else None,
           'inside': float(ins.group(1)) if ins else None, 'draw_m': float(dr.group(1)) if dr else None, 'ok': r.returncode == 0}
    svgs = [f for f in os.listdir(d) if f.endswith('.svg')] if os.path.isdir(d) else []
    if svgs:
        subprocess.run(['rsvg-convert', '-w', '420', '-b', 'white', f'{d}/{svgs[0]}', '-o', f'{d}/thumb.png'])
    with open(res, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(rec, flush=True)
print('DONE', flush=True)
