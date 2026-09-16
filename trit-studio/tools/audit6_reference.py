#!/usr/bin/env python3
"""Independent v6 algorithm specification. This does NOT execute C#, CUDA or the desktop app."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
f=np.float32

def pack(source,group,planes,threshold,local):
    n=len(source)//group;stride=(group+4)//5
    scales=np.zeros((planes,n),dtype='float32');trits=np.zeros((planes,n*stride),dtype='uint8')
    residual=source.copy() if not local else None
    def process(r,g,p):
        total=f(0)
        for value in r:total=f(total+abs(value))
        scale=f(max(f(total/f(group)),f(1e-8)));scales[p,g]=scale
        for j in range(0,group,5):
            value=0;factor=1
            for z in range(min(5,group-j)):
                i=j+z;t=int(np.sign(r[i])) if abs(r[i])>f(scale*f(threshold)) else 0
                value+=(t+1)*factor;factor*=3
                if not local or p+1<planes:r[i]=f(r[i]-f(scale*f(t)))
            trits[p,g*stride+j//5]=value
    if local:
        for g in range(n):
            r=source[g*group:(g+1)*group].copy()
            for p in range(planes):process(r,g,p)
    else:
        for p in range(planes):
            for g in range(n):process(residual[g*group:(g+1)*group],g,p)
    return scales,trits

def pack_checks():
    rng=np.random.default_rng(6006);cases=[]
    for group in (1,3,8,32,128):
        for planes in (1,2,3):
            for threshold in (0,.5,1):
                for kind in ('random','boundaries'):
                    a=(rng.standard_normal(group*19)*.17).astype('float32') if kind=='random' else np.resize(np.array([0,1,-1,.5,-.5],dtype='float32'),group*19)
                    original=a.tobytes();old=pack(a,group,planes,threshold,False);new=pack(a,group,planes,threshold,True)
                    assert a.tobytes()==original
                    assert old[0].tobytes()==new[0].tobytes() and old[1].tobytes()==new[1].tobytes()
                    cases.append(dict(group=group,planes=planes,threshold=threshold,kind=kind,bytes_identical=True))
    a=(rng.standard_normal(65536+128)*.12).astype('float32')
    old=pack(a,32,3,.5,False);new=pack(a,32,3,.5,True)
    assert all(x.tobytes()==y.tobytes() for x,y in zip(old,new))
    return {'cases':cases,'large_weights':len(a),'large_bytes_identical':True,
        'scope':'Sequential independent float32 algorithms; real C# parallel scheduling requires native tests.',
        'residual_memory_formula':{'v5_bytes_per_tensor':'4*N','v6_stack_bytes_per_active_worker':512,'packed_outputs_unchanged':True}}

def qat(source,group,planes,threshold,optimized):
    r=source.detach().reshape(-1,group).clone();q=torch.zeros_like(r)
    with torch.no_grad():
        for p in range(planes):
            magnitude=r.abs();scale=magnitude.mean(1,keepdim=True).clamp_min(1e-8)
            t=torch.where((magnitude if optimized else r.abs())>scale*threshold,r.sign(),torch.zeros_like(r))
            plane=scale*t;q.add_(plane)
            if not optimized or p+1<planes:r.sub_(plane)
    return source+(q.reshape(source.shape)-source).detach()

def qat_checks():
    generator=torch.Generator().manual_seed(66);rows=[]
    for group in (8,32):
        for planes in (1,2,3):
            for threshold in (0,.5,1):
                source=torch.randn(64,32,generator=generator)*.2;values=[];grads=[];grad=torch.randn(source.shape,generator=generator)
                for optimized in (False,True):
                    a=source.clone().requires_grad_();out=qat(a,group,planes,threshold,optimized);(out*grad).sum().backward()
                    assert torch.equal(a.detach(),source);values.append(out.detach());grads.append(a.grad)
                assert torch.equal(*values) and torch.equal(*grads) and torch.equal(grads[0],grad)
                rows.append(dict(group=group,planes=planes,threshold=threshold,values_identical=True,identity_gradient=True))
    return rows

def norm_checks():
    generator=torch.Generator().manual_seed(67);rows=[]
    for count in (1,8,24):
        gradients=[torch.randn((1+i)*17,generator=generator)*.03 for i in range(count)]
        old=torch.zeros([])
        for g in gradients:old=old+g.pow(2).sum()
        new=torch.zeros([])
        for g in gradients:new.add_(g.pow(2).sum())
        assert torch.equal(old,new)
        rows.append(dict(matrices=count,value=float(new),values_identical=True))
    return rows

def loop_checks():
    cases=0
    for context in (32,64,128):
        for prompt in (1,context//2,context-1):
            for limit in (1,2,4,64):
                # Deterministic token-producing oracle, not a trained model or actual C# inference.
                pos=prompt;old=[]
                while pos<context:
                    if len(old)>=limit:break
                    old.append(71);pos+=1
                pos2=prompt;new=[];advance=False
                while pos2<context:
                    if len(new)>=limit:break
                    if advance:
                        if pos2>=context-1:break
                        pos2+=1
                    new.append(71);advance=True
                assert old==new
                assert pos-pos2==1
                cases+=1
    return dict(cases=cases,tokens_unchanged=True,unused_final_forward_removed=True,scope='Loop/limit simulation only')

def main():
    report={'status':'passed','scope':'Independent Python/NumPy/PyTorch numerical and algorithm specification; NOT C# execution or measured application speedup',
            'group_local_packing':pack_checks(),'qat_temporaries':qat_checks(),'gradient_norm':norm_checks(),'generation_loop':loop_checks(),
            'csharp_build':'NOT_RUN','cuda':'NOT_RUN'}
    (ROOT/'reports/performance-reference-v6-additional.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'pack_cases':len(report['group_local_packing']['cases'])+1,'qat_cases':len(report['qat_temporaries']),
          'norm_cases':len(report['gradient_norm']),'loop_cases':report['generation_loop']['cases']}))
if __name__=='__main__':main()
