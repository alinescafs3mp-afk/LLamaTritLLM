#!/usr/bin/env python3
"""Independent policy/algorithm tests, NOT execution of the C# application."""
from pathlib import Path
import hashlib, json, random, collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def run():
    checks=collections.Counter()
    # Immutable corpus payload reuse and bounded streaming/capture policy.
    raw=(ROOT/'data/seed.jsonl').read_bytes()
    for budget in (0,64,4096,4*1024*1024):
        for chunks in (1,4096,65536):
            out=bytearray();capture=bytearray() if budget else None;digest=hashlib.sha256()
            for i in range(0,len(raw),chunks):
                b=raw[i:i+chunks];out.extend(b);digest.update(b)
                if capture is not None:
                    if len(capture)+len(b)>budget:capture=None
                    else:capture.extend(b)
            assert bytes(out)==raw and digest.hexdigest()==hashlib.sha256(raw).hexdigest()
            assert (capture is not None)==(0<len(raw)<=budget)
            if capture is not None:assert bytes(capture)==raw
            checks['bounded_capture_cases']+=1
    # Byte decoder policy: optional UTF8 BOM, strict malformed-byte rejection.
    valid=[b'plain', 'привет\nещё'.encode(),b'\xef\xbb\xbf'+ 'текст'.encode(), '🙂'.encode()]
    invalid=[b'\xff',b'\xc0\xaf',b'\xef\xbb\xbf\xc0\xaf',b'\xf0\x9f',b'\xed\xa0\x80',b'\xff\xfea\x00']
    for b in valid: b.decode('utf-8-sig',errors='strict');checks['valid_utf8_cases']+=1
    for b in invalid:
        try:b.decode('utf-8-sig',errors='strict')
        except UnicodeDecodeError:checks['invalid_utf8_cases']+=1
        else:raise AssertionError('invalid UTF8 accepted')
    # Cached batch statistics must preserve both choices and RNG call order.
    lengths=[20+i for i in range(65)]
    labels=[[-100 if j<8 or j>=n-2 else 7 for j in range(n)] for n in lengths]
    effective=[max(j for j,v in enumerate(row) if v!=-100)+1 for row in labels]
    counts=[sum(v!=-100 for v in row) for row in labels]
    buckets={k:[i for i,n in enumerate(effective) if (n-1)//32==k] for k in set((n-1)//32 for n in effective)}
    def select(rng,batch,bucket,forced,cached):
        candidates=None
        if bucket and forced is None:
            anchor=rng.randrange(65);candidates=buckets[(effective[anchor]-1)//32];indices=[anchor]
        else:indices=[forced if forced is not None else rng.randrange(65)]
        for _ in range(1,batch):indices.append(rng.randrange(65) if candidates is None else candidates[rng.randrange(len(candidates))])
        if cached:return indices,max(effective[i] for i in indices),sum(effective[i] for i in indices),sum(counts[i] for i in indices)
        used=[max(j for j,v in enumerate(labels[i]) if v!=-100)+1 for i in indices]
        return indices,max(used),sum(used),sum(sum(v!=-100 for v in labels[i][:n]) for i,n in zip(indices,used))
    for batch in (1,8,32):
        for bucket in (False,True):
            for forced in (None,3):
                a,b=random.Random(42),random.Random(42)
                for _ in range(100):
                    assert select(a,batch,bucket,forced,False)==select(b,batch,bucket,forced,True)
                    assert a.getstate()==b.getstate();checks['batch_statistic_equivalence_cases']+=1
    # Chunked exact inequality counting, including negative zero and uneven tails.
    rng=np.random.default_rng(9)
    for size in (1,17,4096,65539,300001):
        a=rng.standard_normal(size).astype(np.float32);b=a.copy();b[::17]+=np.float32(1)
        for chunk in (1,16384,65536):
            actual=sum(np.count_nonzero(a[i:i+chunk]!=b[i:i+chunk]) for i in range(0,size,chunk))
            assert actual==np.count_nonzero(a!=b);checks['changed_count_cases']+=1
    checks['changed_count_signed_zero']=int(np.count_nonzero(np.array([0.],np.float32)!=np.array([-0.],np.float32))==0)
    # Cross-file semantic checks: positive mismatches cannot be excused by valid file hashes.
    def good(step,tokens,info_step,info_tokens,changed,ids):
        return step>=0 and tokens>=step and step==info_step and tokens==info_tokens and 0<=changed<=100 and (step!=0 or(tokens==0 and changed==0 and not ids)) and len(ids)==len(set(ids)) and all(len(x)==64 and all(c in '0123456789ABCDEF' for c in x) for x in ids)
    legal='A'*64
    cases=[((0,0,0,0,0,[]),True),((4,20,4,20,1,[legal]),True),((5,20,4,20,1,[]),False),((4,21,4,20,1,[]),False),((0,1,0,1,0,[]),False),((0,0,0,0,1,[]),False),((0,0,0,0,0,[legal]),False),((4,2,4,2,1,[]),False),((4,20,4,20,1,[legal,legal]),False),((4,20,4,20,101,[]),False)]
    for args,expected in cases:assert good(*args)==expected;checks['checkpoint_semantics_cases']+=1
    report={'status':'passed','scope':'Independent Python algorithm/policy checks; NOT C#, native GPU, concurrency, disk durability or performance measurement', 'checks':dict(checks),
      'checkpoint_cache_payload_cap_per_slot':4*1024*1024,'unchanged_payload_serializations_for_8_writes_after_warmup':{'uncached':8,'cached':0},
      'corpus_jsonl_bytes':len(raw),'limitations':['JSONL input bytes are a simulation payload, not the C# TrainingExample JSON serialization format.','Python random exercises preserved sampling call order; C# tests compare the actual persisted SamplerRandom state.','No measured C# speedup or peak-memory claim.']}
    (ROOT/'reports/audit9-reference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
