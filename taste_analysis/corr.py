import json,glob,os,collections,math
import numpy as np
from scipy import stats
R='.'
def load(path):
    try: return json.load(open(path))
    except: return None
def metrics(g):
    return {k:float(v) for k,v in g.items() if k.startswith('quat_') and isinstance(v,(int,float)) and k not in('quat_me_cell',)}
def run(corpus, tag):
    G={}
    for f in glob.glob(f'{corpus}/*.nn'):
        g=load(f)
        if g: G[os.path.basename(f)]=g
    wins=collections.Counter(); games=collections.Counter(); pairs=[]
    for l in open(f'{corpus}/ratings.jsonl'):
        r=json.loads(l); w=os.path.basename(r['winner']); lo=os.path.basename(r['loser'])
        if w in G and lo in G:
            wins[w]+=1; games[w]+=1; games[lo]+=1; pairs.append((w,lo))
    star={os.path.basename(f) for f in glob.glob('Starred/*.nn')}
    names=sorted(n for n in G if games[n]>0)
    keys=sorted({k for n in names for k in metrics(G[n])} - {'quat_taste'})
    M={n:metrics(G[n]) for n in names}
    out=[]
    for k in keys:
        # pairwise
        ok=tot=0
        for w,l in pairs:
            a,b=M[w].get(k),M[l].get(k)
            if a is None or b is None or abs(a-b)<1e-9: continue
            tot+=1; ok+= a>b
        pw=(ok/tot if tot else float('nan'))
        z=(ok-tot/2)/math.sqrt(tot/4) if tot else 0
        # spearman vs win fraction (>=4 games)
        ns=[n for n in names if games[n]>=4 and k in M[n]]
        x=np.array([M[n][k] for n in ns]); y=np.array([wins[n]/games[n] for n in ns])
        rho_w=stats.spearmanr(x,y)[0] if len(ns)>10 and x.std()>0 else float('nan')
        # vs taste score
        ns2=[n for n in names if k in M[n]]
        x2=np.array([M[n][k] for n in ns2]); t=np.array([G[n].get('quat_taste',0.0) for n in ns2])
        rho_t=stats.spearmanr(x2,t)[0] if x2.std()>0 and t.std()>0 else float('nan')
        # starred AUC
        sn=[n for n in ns2 if n in star]
        auc=float('nan')
        if len(sn)>=3:
            xs=np.array([M[n][k] for n in sn]); xo=np.array([M[n][k] for n in ns2 if n not in star])
            if x2.std()>0: auc=stats.mannwhitneyu(xs,xo).statistic/(len(xs)*len(xo))
        out.append((k[5:],pw,z,rho_w,rho_t,auc,len(ns),len(sn)))
    print(f'== {tag}: {len(names)} rated genomes, {len(pairs)} pairs, {len([n for n in names if n in star])} starred')
    return out,names,M,G,wins,games,star
res={}
for corpus,tag in (('train_corpus_quat','slice'),('train_corpus_quat_s3','s3')):
    res[tag]=run(corpus,tag)
    out=res[tag][0]
    out.sort(key=lambda r:-abs(r[2]))
    print(f"{'metric':34s} pair-acc   z    rho(win) rho(taste) starAUC  n  nstar")
    for r in out[:22]:
        print(f"{r[0]:34s} {r[1]:.3f} {r[2]:6.1f}  {r[3]:7.2f}  {r[4]:7.2f}  {r[5]:7.2f} {r[6]:5d} {r[7]:3d}")
import pickle; pickle.dump({t:res[t][0] for t in res},open('taste_analysis/res.pkl','wb'))
