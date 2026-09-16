#!/usr/bin/env python3
"""Independent Python specifications only. Does NOT load/execute C#, SIMD JIT, GUI or OS pipe tests."""
from pathlib import Path
import json, math, hashlib, itertools, random
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def value_sum(v,p,vector_width):
    n,hd=v.shape; out=np.zeros(hd,np.float32);end=hd-hd%vector_width
    for t in range(n):
        for d in range(0,end,vector_width):
            out[d:d+vector_width]=np.add(out[d:d+vector_width],np.multiply(p[t],v[t,d:d+vector_width],dtype=np.float32),dtype=np.float32)
        for d in range(end,hd):out[d]=np.float32(out[d]+np.float32(p[t]*v[t,d]))
    return out

def attention_check():
    rng=np.random.default_rng(13013);cases=0;worst=0.
    for n,hd,heads in itertools.product((1,17,129),(3,8,16,33,64),(1,4)):
        kvh=1 if heads==1 else 2
        q=rng.normal(0,.3,(heads,hd)).astype(np.float32)
        k=rng.normal(0,.3,(n,kvh,hd)).astype(np.float32);v=rng.normal(0,.3,k.shape).astype(np.float32)
        for width in (4,8,16):
            for h in range(heads):
                kh=h//(heads//kvh)
                scores=np.sum(k[:,kh,:]*q[h],axis=-1,dtype=np.float32)*np.float32(1/math.sqrt(hd))
                exp=np.exp(scores-np.max(scores),dtype=np.float32);denom=np.float32(0)
                for x in exp:denom=np.float32(denom+x)
                p=exp*np.float32(1/denom)
                scalar=value_sum(v[:,kh,:],p,1);vector=value_sum(v[:,kh,:],p,width)
                assert np.array_equal(scalar.view(np.uint32),vector.view(np.uint32))
                worst=max(worst,float(np.max(np.abs(scalar-vector))))
            cases+=1
    return {'cases':cases,'max_abs_error':worst,'bitwise_equal_in_numpy':True,
        'scope':'Value reduction lanes only, same scores and time order. NOT .NET SIMD, JIT, native speed or parallel evidence.'}

def parse(line):
    if len(line)>1048576:raise ValueError('too large')
    row=json.loads(line.decode('utf-8'))
    if not isinstance(row,dict) or not isinstance(row.get('user'),str) or not isinstance(row.get('assistant'),str):raise ValueError('invalid text')
    for key in ('user','assistant'):row[key].encode('utf-8',errors='strict')
    return row

def reverse_tail(data,limit,budget):
    start=max(0,len(data)-budget);raw=data[start:];lower=0
    if start and data[start-1:start]!=b'\n':
        newline=raw.find(b'\n')
        if newline<0:return [],0,0
        lower=newline+1
    cursor=len(raw);rows=[];skipped=inspected=0
    while cursor>lower and len(rows)<limit:
        newline=raw.rfind(b'\n',lower,cursor);line_start=lower if newline<0 else newline+1
        line=raw[line_start:cursor];cursor=lower if newline<0 else newline
        if start==0 and line_start==0 and line.startswith(b'\xef\xbb\xbf'):line=line[3:]
        if not line.strip(b'\t\r\n '):continue
        inspected+=1
        try:rows.append(parse(line))
        except (ValueError,UnicodeError):skipped+=1
    return rows[::-1],skipped,inspected

def forward_oracle(data,limit,budget):
    start=max(0,len(data)-budget);raw=data[start:]
    if start and data[start-1:start]!=b'\n':
        at=raw.find(b'\n');raw=b'' if at<0 else raw[at+1:]
    elif not start and raw.startswith(b'\xef\xbb\xbf'):raw=raw[3:]
    rows=[]
    for line in raw.split(b'\n'):
        if not line.strip(b'\t\r\n '):continue
        try:rows.append(parse(line))
        except (ValueError,UnicodeError):pass
    return rows[-limit:]

def journal_check():
    rng=random.Random(13013);cases=0
    for count in (0,1,5,100,1000):
        for crlf in (False,True):
            records=[]
            for i in range(count):
                line=json.dumps({'user':f'реплика {i} 🙂','assistant':'ответ'*rng.randrange(1,20),'revision':i},ensure_ascii=False).encode()
                records.append(line+(b'\r\n' if crlf else b'\n'))
                if i%17==0:records.append(b'{broken\xff\n')
            raw=b''.join(records)
            for budget,limit,suffix in itertools.product((1024,4096,4*1024*1024),(1,7,250),(b'',b'{"partial',b'\n \t\r\n')):
                data=raw+suffix;result,skipped,inspected=reverse_tail(data,limit,budget)
                assert result==forward_oracle(data,limit,budget);assert inspected>=len(result);cases+=1
    dense=b''.join((json.dumps({'user':str(i),'assistant':'ok','revision':i})+'\n').encode() for i in range(10000))
    result,skipped,inspected=reverse_tail(dense,250,4*1024*1024)
    assert len(result)==250 and inspected==250 and result[0]['revision']==9750
    # A byte snapshot is independent of all bytes appended later; actual C# uses a growing stream fixture.
    snapshot=dense;appended=dense+b'{"user":"later","assistant":"later"}\n'
    assert reverse_tail(snapshot,250,4*1024*1024)[0]!=reverse_tail(appended,250,4*1024*1024)[0]
    return {'tail_cases':cases,'static_valid_journal_records':10000,'records_parsed_for_last250':inspected,
        'scope':'Independent fixed-window parsing model. NOT real .NET IO, ArrayPool memory or concurrent truncation evidence.'}

def draft_check():
    cases=0
    for draft,exclude,old_private in itertools.product((None,'',' ','new draft'),(False,True),(False,True)):
        restored=draft in (None,'');text='old text' if restored else draft
        private=(exclude or old_private) if restored else exclude
        if restored and old_private:assert private
        if not restored:assert text==draft and private==exclude
        cases+=1
    return {'cases':cases,'scope':'Policy truth table, NOT headless UI execution.'}

def run():
    m=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    result={'status':'passed','scope':'Independent Python numeric/data/state specifications only; C# build/runtime/GUI/CUDA NOT_RUN',
        'attention':attention_check(),'journal':journal_check(),'draftRecovery':draft_check(),
        'dataset':{'version':m['version'],'total':sum(m['counts'].values()),'pretrain_sha256':m['files']['pretrain']['sha256']}}
    (ROOT/'reports/audit13-reference.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':run()
