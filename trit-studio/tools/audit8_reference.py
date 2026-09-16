#!/usr/bin/env python3
"""Independent data/algorithm specification, not an execution of Trit Studio C# or its native tests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]

def receipt_checks():
    def parse(data):
        if not isinstance(data, dict) or type(data.get('success')) is not bool: raise ValueError('success')
        if 'cancelled' in data and type(data['cancelled']) is not bool: raise ValueError('cancelled')
        if data.get('error') is not None and not isinstance(data['error'], str): raise ValueError('error')
        if data['success'] and (data.get('cancelled', False) or (data.get('error') or '').strip()): raise ValueError('contradiction')
        return data
    fixtures=[{'success':True},{'success':False,'error':'failed'},{'success':False,'cancelled':True},
              {},{'success':'yes'},{'success':False,'error':{}},{'success':False,'cancelled':0},
              {'success':True,'error':'bad'},{'success':True,'cancelled':True},None]
    resolved=0
    for data in fixtures:
        pending={'request':False}; settled={}
        try:
            result=parse(data)  # Invalid data cannot remove the completion source before failure cleanup.
            pending.pop('request'); settled['request']=result
        except ValueError:
            for key in pending: settled[key]='failed'
            pending.clear()
        assert 'request' in settled and not pending
        resolved+=1
    return dict(cases=resolved,all_requests_settled_in_reference=True,
                native_evidence='Defined real broken-child C# fixtures, not run here')

def cache_checks():
    torch.set_num_threads(1)
    rng=np.random.default_rng(818)
    examples=[]
    for i in range(41):
        n=3+(i*11)%61; ids=rng.integers(0,262,n,dtype=np.int64); labels=np.full(n,-100,dtype=np.int64)
        labels[n//2:]=rng.integers(0,262,n-n//2); examples.append((ids,labels))
    generator=torch.Generator().manual_seed(118)
    base=torch.randn(262,262,generator=generator)
    modified=base+torch.eye(262)*0.25
    results=[]
    for batch in (1,4,8):
        ordered=sorted(enumerate(examples),key=lambda x:(len(x[1][0]),x[0]))
        def build(index):
            selected=[e for _,e in ordered[index*batch:(index+1)*batch]];length=max(len(e[0]) for e in selected)
            inp=np.zeros((len(selected),length),dtype=np.int64);targets=[];positions=[]
            for row,(tokens,labels) in enumerate(selected):
                inp[row,:len(tokens)]=tokens
                for j,v in enumerate(labels):
                    if v!=-100: positions.append(row*length+j);targets.append(int(v))
            return inp.reshape(-1),np.array(targets,dtype=np.int64),np.array(positions,dtype=np.int64)
        count=(len(ordered)+batch-1)//batch
        for budget in (0,128,16*1024*1024):
            slots={};payload=hits=builds=0;outputs=[]
            for weights in (base,modified):
                total=torch.zeros((),dtype=torch.float64);token_count=0
                for index in range(count):
                    if index in slots: packed=slots[index];hits+=1
                    else:
                        packed=build(index);builds+=1;size=sum(x.nbytes for x in packed)
                        if size<=budget-payload: slots[index]=packed;payload+=size
                    plain=build(index)
                    assert all(np.array_equal(a,b) for a,b in zip(packed,plain))
                    ids,labels,pos=packed
                    logits=weights[torch.from_numpy(ids[pos])]
                    loss=F.cross_entropy(logits,torch.from_numpy(labels),reduction='sum')
                    total+=loss.double();token_count+=len(labels)
                assert token_count==sum(np.count_nonzero(e[1]!=-100) for e in examples)
                outputs.append(float(total)/token_count)
            assert payload<=budget and outputs[0]!=outputs[1]
            if budget==0: assert hits==0 and builds==2*count
            if budget==16*1024*1024: assert hits==count and builds==count
            results.append(dict(batch=batch,budget_bytes=budget,retained_array_bytes=payload,
                                builds=builds,hits=hits,loss_before=outputs[0],loss_after=outputs[1]))
    return dict(scope='Independent CPU array-preparation/weighted-objective check. No timing claims.',
                shapes=results,inputs_and_targets_equal=True,updated_weights_recomputed=True)

def transaction_checks():
    # Fault-injection specification: candidate errors cannot move the old authoritative pointer.
    cases=[]
    for failure in ('verify','decode','optimizer','activate','replay',None):
        pointer='current';session='current';warning=False
        try:
            for stage in ('verify','decode','optimizer'):
                if failure==stage: raise ValueError(stage)
            session='parent'
            if failure=='activate': raise OSError('activate')
            pointer='parent'
            if failure=='replay': warning=True
        except (ValueError,OSError): session=pointer
        if failure in ('verify','decode','optimizer','activate'): assert pointer==session=='current'
        else: assert pointer==session=='parent'
        cases.append(dict(failure=failure,active=pointer,session=session,queue_warning=warning))
    return dict(scope='Independent state-ordering simulation, not filesystem durability or real native rollback',cases=cases)

def main():
    manifest=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    base=(ROOT/'data/pretrain.jsonl').read_bytes()
    assert hashlib.sha256(base).hexdigest()==manifest['frozen_baseline_sha256']
    counts={name:len([x for x in (ROOT/'data'/f'{name}.jsonl').read_text().splitlines() if x.strip()]) for name in manifest['files']}
    report=dict(status='passed',scope='Independent Python algorithm/data specification; NOT C# compilation, runtime, GUI or CUDA',
                completion=receipt_checks(),validation_preparation=cache_checks(),rollback=transaction_checks(),
                baseline_bytes_unchanged=True,baseline_sha256=hashlib.sha256(base).hexdigest(),records=counts,
                measured_CSharp_speedup=None,real_CSharp_tests='NOT_RUN')
    (ROOT/'reports/audit8-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
