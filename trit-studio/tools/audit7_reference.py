#!/usr/bin/env python3
"""Independent numerical/specification checks, not C# execution or conversational evaluation."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
def run():
    checks=[]
    for shape in [(257,), (2,17,32), (4,3,64)]:
        a=torch.linspace(-16,16,math.prod(shape),dtype=torch.float32).reshape(shape).requires_grad_()
        b=a.detach().clone().requires_grad_()
        ref=a*a.sigmoid(); fused=torch.nn.functional.silu(b)
        ref.square().mean().backward(); fused.square().mean().backward()
        forward=(ref-fused).abs().max().item();grad=(a.grad-b.grad).abs().max().item()
        assert forward<1e-5 and grad<1e-5
        checks.append(dict(shape=list(shape),forward_max_abs=forward,gradient_max_abs=grad))
    # Equivalent output-storage policies under prompt capacity, changing limits and whole UTF-8 scalars.
    output_trials=0
    for context in (32,64,512,2048):
        for prompt in (1, context//2, context-4, context-1):
            for initial in (1,2,5,256,1024):
                for later in (1,4,512,1024):
                    cap=min(1024,context-prompt);buffer=bytearray(cap);old=[];count=0
                    text=('Привет 🌿 A\n'*256)
                    for char in text:
                        limit=initial if count<4 else later
                        bs=char.encode('utf-8')
                        if count+len(bs)>min(limit,context-prompt): break
                        old.extend(int(x)+6 for x in bs)
                        buffer[count:count+len(bs)]=bs;count+=len(bs)
                    assert len(buffer)==cap and count<=cap
                    assert bytes(x-6 for x in old).decode()==bytes(buffer[:count]).decode()
                    output_trials+=1
    manifest=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    load=lambda n:[json.loads(x) for x in (ROOT/'data'/f'{n}.jsonl').read_text().splitlines()]
    base,seed=load('pretrain'),load('seed')
    assert all('text' in x and 'answer' not in x and 'messages' not in x for x in base)
    assert not {x['id'] for x in base}&{x['id'] for x in seed}
    # Capacity and vocab are unchanged by curriculum selection; these are data-policy checks only.
    report=dict(status='passed',scope='Independent Python math/output-buffer/data-policy checks; NOT C#/GUI/worker/CUDA execution',
      silu=checks,output_storage_trials=output_trials,baseline_records=len(base),conversation_records=len(seed),
      baseline_has_dialogues=False,baseline_seed_exact_overlap=0,
      stage_policy='Random means zero optimizer steps; this invariant still requires actual worker regression on Grok host',
      torch_version=torch.__version__,measured_CSharp_speedup=None)
    (ROOT/'reports/audit7-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
