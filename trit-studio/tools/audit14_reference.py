#!/usr/bin/env python3
"""Independent numerical/state specifications, NOT execution of C#, GUI or CUDA."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(140014)
normalization=[];residual=[]
for size in (1,3,8,15,16,31,32,64,127,256,512):
    for width in (4,8,16):
        for kind in ('random','zeros','mixed'):
            x=rng.normal(size=size).astype(np.float32)
            if kind=='zeros': x[:]=0
            if kind=='mixed': x[::3]*=np.float32(100)
            gamma=rng.uniform(.1,1.9,size).astype(np.float32)
            # Use the SAME reduction on both sides; this is not a specification of .NET Dot/JIT lowering.
            squares=np.float32(0)
            for value in x: squares=np.float32(squares+np.float32(value*value))
            inv=np.float32(1)/np.sqrt(np.float32(squares/np.float32(size)+np.float32(1e-5)))
            expected=np.array([np.float32(np.float32(a*inv)*g) for a,g in zip(x,gamma)],dtype=np.float32)
            actual=np.empty_like(x)
            for start in range(0,size,width):
                stop=min(start+width,size)
                actual[start:stop]=(x[start:stop]*inv)*gamma[start:stop]
            assert np.array_equal(expected,actual)
            for alias in ('separate','input','gamma','all'):
                a=x.copy();g=gamma.copy()
                if alias=='all':g=a
                gold=np.array([np.float32(np.float32(v*inv)*h) for v,h in zip(a,g)],dtype=np.float32)
                out=a if alias in ('input','all') else g if alias=='gamma' else np.empty_like(a)
                for start in range(0,size,width):
                    stop=min(start+width,size);out[start:stop]=(a[start:stop]*inv)*g[start:stop]
                assert np.array_equal(gold,out)
            normalization.append(dict(size=size,lanes=width,case=kind,aliases=4,max_abs_error=float(np.max(np.abs(expected-actual)))))
for size,width,self_alias in itertools.product((0,1,3,16,31,128,512),(4,8,16),(False,True)):
    a=rng.normal(size=size).astype(np.float32);b=a if self_alias else rng.normal(size=size).astype(np.float32)
    expected=np.array([np.float32(x+y) for x,y in zip(a,b)],dtype=np.float32)
    for start in range(0,size,width):a[start:start+width]=a[start:start+width]+b[start:start+width]
    assert np.array_equal(expected,a)
    residual.append(dict(size=size,lanes=width,self_alias=self_alias))
# Opening policy simulation: refusal during read-only preparation cannot touch the live session.
opens=[]
for failure in ('malformed-id','oversized-id','bad-active','bad-hash','cancel-before-read','cancel-after-read',None):
    state=dict(model='old',draft='unsubmitted',epoch=5,lease='old',old_worker_alive=True)
    before=state.copy();events=[];destination_lease=True
    if failure is None:
        events+=['verified-packed-model','prepared-history','commit-conversation-id','begin-handoff']
        state.update(model='new',epoch=6,lease='new',old_worker_alive=False)
    else:
        events+=['preview-refused'];destination_lease=False
        assert state==before
    opens.append(dict(failure=failure,events=events,preserved_old=failure is not None,retained_destination_lease=destination_lease))
# Initial-reference lifetime: invalidation precedes updates/restores, including failures.
initials=[]
for event in ('none','evaluate','sampler-only','update','partial-update-failure','restore','failed-restore','mark-updated','dispose'):
    reference_valid=True;weight_version=0;step=0
    if event in ('update','partial-update-failure','restore','failed-restore','dispose'):reference_valid=False
    if event in ('update','mark-updated'):weight_version+=1
    if event=='update':step+=1
    shortcut=reference_valid and step==0 and weight_version==0
    assert shortcut==(event in ('none','evaluate','sampler-only'))
    initials.append(dict(event=event,reuse_allowed=shortcut))
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.jsonl')}
assert files['pretrain.jsonl']=='a1efc4e62aba0afbe0d2f062e76e8f68c8f54454c68fe985a88ece58a1d14973'
assert files['challenge-v13.jsonl']=='888d908276483030ca818c74d0acaca65a7da9ebedad5cc1762509d260495ef3'
report=dict(status='passed',scope='Independent Python/NumPy numerical and policy specifications. NOT C# execution, .NET SIMD, UI scheduling, filesystem failure or CUDA evidence.',
    normalization_cases=len(normalization),normalization_alias_checks=4*len(normalization),normalization=normalization,
    residual_cases=len(residual),residual=residual,workspace_policy_cases=opens,initial_reference_lifetime_cases=initials,
    frozen_baseline_sha256=files['pretrain.jsonl'],frozen_prior_holdout_sha256=files['challenge-v13.jsonl'],
    native_benchmark='initializationAndElementwise is DEFINED in C#, NOT RUN in this environment',
    maximum_normalization_error=max(x['max_abs_error'] for x in normalization))
(ROOT/'reports/audit14-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('normalization','residual')},indent=2))
