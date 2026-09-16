#!/usr/bin/env python3
"""Development-only numerical oracle. Requires numpy and torch, NOT an application dependency.
It validates the math specification and writes a fixture for the C# executable tests.
It does not execute or certify the C# implementation.
"""
import json, math, pathlib
import numpy as np
import torch
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
c=dict(formatVersion=1, dimension=16, hiddenDimension=32, layers=1, heads=2, kvHeads=1,
       context=64, planes=2, groupSize=8, threshold=0.5, ropeTheta=10000, seed=42)
d=c['dimension'];hd=d//c['heads'];kv=hd*c['kvHeads'];g=c['groupSize']
shapes=[('embedding',262,d,True),('block.0.norm1',1,d,False),('block.0.q',d,d,True),('block.0.k',kv,d,True),
        ('block.0.v',kv,d,True),('block.0.out',d,d,True),('block.0.norm2',1,d,False),('block.0.gate',32,d,True),
        ('block.0.up',32,d,True),('block.0.down',d,32,True),('norm',1,d,False)]
weights={}
for n,(name,rows,cols,quant) in enumerate(shapes):
    a=np.sin(np.arange(rows*cols,dtype=np.float64)*0.11+n*0.317)*0.12 if quant else np.ones(rows*cols)
    weights[name]=a.astype(np.float32).reshape(rows,cols)

def numpy_quant(a):
    r=a.copy().reshape(-1,g);q=np.zeros_like(r)
    for _ in range(c['planes']):
        # Match the scalar accumulation and clamped scales in the portable exporter.
        sums=np.zeros((r.shape[0],1),dtype=np.float32)
        for col in range(g): sums[:,0]+=np.abs(r[:,col])
        scale=np.maximum(sums/g,np.float32(1e-8))
        trit=np.where(np.abs(r)>scale*c['threshold'],np.sign(r),0).astype(np.float32)
        plane=scale*trit;q+=plane;r-=plane
    return q.reshape(a.shape)
qweights={name:numpy_quant(weights[name]) if quant else weights[name].copy() for name,_,_,quant in shapes}

params={name:torch.tensor(a,requires_grad=True) for name,a in weights.items()}
def quant(name):
    w=params[name];r=w.detach().reshape(-1,g);q=torch.zeros_like(r)
    for _ in range(c['planes']):
        scale=r.abs().mean(1,keepdim=True).clamp_min(1e-8)
        t=torch.where(r.abs()>scale*c['threshold'],r.sign(),torch.zeros_like(r))
        piece=scale*t;q=q+piece;r=r-piece
    return w+(q.reshape(w.shape)-w).detach()
def norm(x,name):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*params[name]
def rope(x,heads):
    b,t,_=x.shape;x=x.reshape(b,t,heads,hd//2,2)
    a=torch.arange(t)[:,None]/(10000**(torch.arange(hd//2)*2/hd))[None,:]
    cos=a.cos()[None,:,None,:];sin=a.sin()[None,:,None,:]
    even,odd=x[...,0],x[...,1]
    return torch.stack((even*cos-odd*sin,even*sin+odd*cos),-1).reshape(b,t,heads,hd)
def forward(ids):
    b,t=ids.shape;embedding=quant('embedding');x=embedding[ids]
    p='block.0.';z=norm(x,p+'norm1')
    q=rope(z@quant(p+'q').T,2).transpose(1,2)
    k=rope(z@quant(p+'k').T,1).repeat_interleave(2,dim=2).transpose(1,2)
    v=(z@quant(p+'v').T).reshape(b,t,1,hd).repeat_interleave(2,dim=2).transpose(1,2)
    scores=q@k.transpose(-1,-2)/math.sqrt(hd)
    scores=scores.masked_fill(torch.ones(t,t,dtype=torch.bool).triu(1),float('-inf'))
    attn=(scores.softmax(-1)@v).transpose(1,2).reshape(b,t,d)
    x=x+attn@quant(p+'out').T;z=norm(x,p+'norm2');gate=z@quant(p+'gate').T
    x=x+(gate*gate.sigmoid()*(z@quant(p+'up').T))@quant(p+'down').T
    return norm(x,'norm')@embedding.T

def numpy_norm(x,name):return x/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-5))*qweights[name][0]
def numpy_rope(x,heads,pos):
    x=x.reshape(heads,hd).copy()
    for h in range(heads):
        for j in range(0,hd,2):
            angle=pos/(10000**(j/hd));co,si=np.float32(math.cos(angle)),np.float32(math.sin(angle))
            a,b=x[h,j:j+2].copy();x[h,j]=a*co-b*si;x[h,j+1]=a*si+b*co
    return x.reshape(-1)
def numpy_autoreg(tokens):
    keys=[];values=[];outputs=[]
    for pos,token in enumerate(tokens):
        x=qweights['embedding'][token].copy();p='block.0.';z=numpy_norm(x,p+'norm1')
        q=numpy_rope(qweights[p+'q']@z,2,pos);k=numpy_rope(qweights[p+'k']@z,1,pos);v=qweights[p+'v']@z
        keys.append(k);values.append(v);attn=np.zeros(d,dtype=np.float32)
        for h in range(2):
            scores=np.array([q[h*hd:(h+1)*hd]@k0/math.sqrt(hd) for k0 in keys],dtype=np.float32)
            pr=np.exp(scores-scores.max());pr/=pr.sum()
            attn[h*hd:(h+1)*hd]=sum(pr[j]*values[j] for j in range(len(keys)))
        x+=qweights[p+'out']@attn;z=numpy_norm(x,p+'norm2');gate=qweights[p+'gate']@z
        x+=qweights[p+'down']@((gate/(1+np.exp(-gate)))*(qweights[p+'up']@z))
        outputs.append(qweights['embedding']@numpy_norm(x,'norm'))
    return np.array(outputs)

tokens=[1,3,ord('q')+6,2,4]
with torch.no_grad(): expected=forward(torch.tensor([tokens]))[0].numpy()
np_result=numpy_autoreg(tokens)
error=float(np.max(np.abs(expected-np_result)))
assert error<0.002,error
packed_roundtrip_error=0.
for name,_,_,isq in shapes:
    if isq:
        with torch.no_grad(): packed_roundtrip_error=max(packed_roundtrip_error,float(np.max(np.abs(quant(name).numpy()-qweights[name]))))
assert packed_roundtrip_error<1e-6
fixture=dict(config=c,weights={name:a.reshape(-1).tolist() for name,a in weights.items()},tokens=tokens,logits=expected.tolist())
(ROOT/'tests/fixtures').mkdir(parents=True,exist_ok=True)
(ROOT/'tests/fixtures/reference.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')
# Real autograd verification on the analogous math. Not a claim that the C# worker was run.
ids=torch.tensor([[1,3,ord('q')+6,2,4,ord('a')+6]])
labels=torch.tensor([[-100,-100,-100,-100,ord('a')+6,2]])
opt=torch.optim.AdamW(params.values(),lr=.001,weight_decay=.01)
def loss():return F.cross_entropy(forward(ids).reshape(-1,262),labels.reshape(-1),ignore_index=-100)
before=float(loss().detach());original={k:v.detach().clone() for k,v in params.items()}
for i in range(32):
    opt.zero_grad();l=loss();l.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),1);opt.step()
after=float(loss().detach());changed=sum(int((original[k]!=v).sum()) for k,v in params.items())
assert changed>0 and after<before,(changed,before,after)
# Corpus checks are separate from the numerical reference and support true multi-turn data.
from validate_dataset import validate as validate_corpus
corpus_report = validate_corpus()
lengths = [x['max_sequence_bytes_with_roles'] for x in corpus_report['splits'].values()]
report=dict(scope='Independent Python numerical specification tests; C# build/runtime NOT executed',
    status='passed',torch_version=torch.__version__,forward_max_abs_error=error,quantizer_max_abs_error=packed_roundtrip_error,
    fitting_loss_before=before,fitting_loss_after=after,changed_parameter_values=changed,
    parameter_count=sum(v.numel() for v in params.values()),max_bundled_example_sequence_length=max(lengths),
    checks=['dense Torch vs incremental NumPy causal GQA forward','Torch QAT vs portable-quantizer specification',
            'actual gradient update and toy fitting','bundled examples fit default sequence length'])
(ROOT/'reports/numerical-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
