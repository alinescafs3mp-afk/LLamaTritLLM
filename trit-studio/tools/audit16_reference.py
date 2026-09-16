#!/usr/bin/env python3
"""Independent prefix mathematical specification. Does NOT execute C#, UI, OS pipes or native TorchSharp."""
from pathlib import Path
import hashlib,json,math
import numpy as np
R=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(716)
def case(layers,heads,kvheads,n,planes,seed):
    g=np.random.default_rng(seed);d=32;hd=d//heads;hidden=64
    def quant(a):
        r=a.copy().reshape(-1,8);out=np.zeros_like(r)
        for _ in range(planes):
            sc=np.maximum(np.abs(r).mean(1,keepdims=True),np.float32(1e-8))
            part=sc*np.where(np.abs(r)>sc*.5,np.sign(r),0).astype(np.float32)
            out+=part;r-=part
        return out.reshape(a.shape)
    def matrix(rows,cols):return quant(g.normal(0,.04,(rows,cols)).astype(np.float32))
    emb=matrix(262,d);w=[]
    for l in range(layers):
        w.append(dict(q=matrix(d,d),k=matrix(kvheads*hd,d),v=matrix(kvheads*hd,d),o=matrix(d,d),gate=matrix(hidden,d),up=matrix(hidden,d),down=matrix(d,hidden)))
    def norm(x):return x/(np.mean(x*x,dtype=np.float32)+np.float32(1e-5))**np.float32(.5)
    def rope(x,hh,p):
        x=x.copy().reshape(hh,hd)
        for j in range(0,hd,2):
            a=np.float32(p/(10000**(j/hd)));co=np.float32(np.cos(a));si=np.float32(np.sin(a))
            ev=x[:,j].copy();od=x[:,j+1].copy();x[:,j]=ev*co-od*si;x[:,j+1]=ev*si+od*co
        return x
    ids=g.integers(0,262,n+5).tolist()
    def run(fast):
        kk=[[]for _ in w];vv=[[]for _ in w];outputs=[];skips=0
        for pos,token in enumerate(ids):
            x=emb[token].copy();output=pos>=n-1
            for l,p in enumerate(w):
                z=norm(x);q=rope(p['q']@z,heads,pos);k=rope(p['k']@z,kvheads,pos);v=(p['v']@z).reshape(kvheads,hd)
                kk[l].append(k);vv[l].append(v)
                if fast and not output and l==layers-1:skips+=1;break
                attn=np.zeros((heads,hd),np.float32)
                for h in range(heads):
                    kv=h//(heads//kvheads)
                    scores=np.array([q[h]@t[kv]/math.sqrt(hd)for t in kk[l]],np.float32)
                    prob=np.exp(scores-scores.max());prob/=prob.sum()
                    for t in range(pos+1):attn[h]+=prob[t]*vv[l][t][kv]
                x+=p['o']@attn.reshape(-1);z=norm(x);a=p['gate']@z
                x+=p['down']@((a/(1+np.exp(-a)))*(p['up']@z))
            if output:outputs.append(emb@norm(x))
        return np.array(outputs),kk,vv,skips
    a,ak,av,_=run(False);b,bk,bv,skips=run(True)
    assert np.array_equal(a,b),(layers,heads,kvheads,n,np.abs(a-b).max())
    for l in range(layers):
        assert np.array_equal(ak[l],bk[l])and np.array_equal(av[l],bv[l])
    assert skips==n-1
    return dict(layers=layers,heads=heads,kvHeads=kvheads,prefix=n,planes=planes,continuedTokens=5,skippedOutputs=skips,maxAbsLogitDifference=float(np.abs(a-b).max()))
results=[]
for layers in (1,2,4):
 for heads,kv in ((2,1),(4,2),(4,4)):
  for n in (1,3,17,41):
   for planes in (1,2):results.append(case(layers,heads,kv,n,planes,100+len(results)))
# Independent pointer-state decisions, not .NET filesystem tests.
policies=[]
for pointer in ('absent','file','directory','link'):
 for revisions in (False,True):
  allowed=pointer=='file' or pointer=='absent'and not revisions
  policies.append(dict(pointer=pointer,revisions=revisions,allow=allowed))
# Detect a keyboard held across completion without dispatching framework events here.
def keys(events):
 held=False;sent=0;newlines=0
 for e in events:
  if e=='release':held=False
  elif e=='shift':newlines+=1
  elif e=='enter':
   if not held:sent+=1
   held=True
 return sent,newlines
assert keys(['enter','enter','release','shift','enter','release'])==(2,1)
manifest=json.loads((R/'data/DATASET_MANIFEST.json').read_text())
report=dict(status='passed',scope=__doc__,prefixCases=results,pointerPolicyCases=policies,keyboardPolicySimulation=True,
 corpusRecords=sum(manifest['counts'].values()),CSharp='NOT_RUN',PowerShell='NOT_RUN',GUI='NOT_RUN',CUDA='NOT_RUN',
 note='Exact equal outputs/KV in these NumPy cases. Not a throughput benchmark or execution of the changed C# code.')
(R/'reports/audit16-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='passed',prefixCases=len(results),maxLogitDifference=max(x['maxAbsLogitDifference']for x in results),corpusRecords=report['corpusRecords']),indent=2))
