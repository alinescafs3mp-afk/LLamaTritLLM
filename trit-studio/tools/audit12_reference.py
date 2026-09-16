#!/usr/bin/env python3
"""Independent Python specifications, NOT C# execution, GUI scheduling, OS pipes or measured app speed."""
from pathlib import Path
import hashlib, json, math, random, unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    rng=np.random.default_rng(12); schedules=0; row_counts=0
    for n in (16,64,256):
        for sizes in ((17,), (64,65), (n,n//2,n//2), (257,33,19)):
            matrices=[rng.normal(size=(rows,n)).astype(np.float32) for rows in sizes]
            x=rng.normal(size=n).astype(np.float32)
            # Row arithmetic stays identical. This tests mapping/group coverage and float storage,
            # not SIMD width, TPL allocation or real multi-threaded .NET determinism.
            expected=[np.array([np.dot(a[row],x) for row in range(len(a))],dtype=np.float32) for a in matrices]
            offsets=np.cumsum([0,*sizes]); total=sum(sizes)
            for permute in (False,True):
                chunks=list(range((total+15)//16));random.Random(12).shuffle(chunks) if permute else None
                actual=[np.empty_like(a) for a in expected];seen=set()
                for chunk in chunks:
                    for logical in range(chunk*16,min(total,(chunk+1)*16)):
                        assert logical not in seen;seen.add(logical)
                        bank=int(np.searchsorted(offsets,logical,side='right')-1);row=logical-int(offsets[bank])
                        actual[bank][row]=np.dot(matrices[bank][row],x)
                assert seen==set(range(total))
                for a,b in zip(expected,actual):assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
                schedules+=1;row_counts+=total
    rope_cases=0;rope_values=0
    for hd in (8,16,32,64,128):
        for context in (32,512,1024):
            den=np.array([np.float32(math.pow(10000,float(np.float32(2*d)/np.float32(hd)))) for d in range(hd//2)],np.float32)
            for p in range(context):
                old=np.array([np.float32(p)/np.float32(math.pow(10000,float(np.float32(d)/np.float32(hd)))) for d in range(0,hd,2)],np.float32)
                new=np.float32(p)/den
                assert np.array_equal(old.view(np.uint32),new.view(np.uint32));rope_values+=len(new)
            rope_cases+=1
    manifest=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    def runtime(row):
        if 'text' in row:return dict(text=row['text'],answer=None,history=[])
        if 'messages' in row:
            m=row['messages'];return dict(text=m[-2]['content'],answer=m[-1]['content'],history=[(m[i]['content'],m[i+1]['content']) for i in range(0,len(m)-2,2)])
        return dict(text=row['prompt'],answer=row['answer'],history=[])
    corpora={name:[runtime(json.loads(line)) for line in (ROOT/'data'/value['file']).read_text().splitlines()] for name,value in manifest['files'].items()} if all('file'in v for v in manifest['files'].values()) else {
        name:[runtime(json.loads(line)) for line in (ROOT/'data'/(name+'.jsonl')).read_text().splitlines()] for name in manifest['files']}
    def required(row):return len(row['text'].encode())+(1 if row['answer']is None else len(row['answer'].encode())+4)
    def start(row,budget):
        used=required(row)
        if used>budget:raise ValueError('target too long')
        i=len(row['history'])
        while i>0:
            size=sum(len(s.encode()) for s in row['history'][i-1])+4
            if used+size>budget:break
            used+=size;i-=1
        return i
    def key(row,first=0):
        def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split()).upper()
        obj={'kind':'text'if row['answer']is None else'dialogue','text':norm(row['text']),'history':[[norm(a),norm(b)]for a,b in row['history'][first:]]}
        return hashlib.sha256(json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    keys=0;decisions=0;trimmed=0;savings=[]
    allrows=sum(corpora.values(),[])
    for budget in (32,64,128,512):
        controls=[e for e in corpora['validation']if required(e)<=budget]
        oldfull={key(e)for e in controls};oldeff={key(e,start(e,budget))for e in controls}
        newfull=set();neweff=set()
        for e in controls:
            full=key(e);first=start(e,budget);newfull.add(full);neweff.add(full if first==0 else key(e,first))
        assert (oldfull,oldeff)==(newfull,neweff)
        legacy_builds=0;new_builds=0
        for e in allrows:
            if required(e)>budget:continue
            full=key(e);first=start(e,budget);eff=key(e,first)
            assert eff==(full if first==0 else key(e,first));keys+=1;trimmed+=int(first>0)
            old=full in oldfull or eff in oldeff
            new=full in newfull or (full if first==0 else key(e,first))in neweff
            assert old==new;decisions+=1
            legacy_builds+=1+int(full not in oldfull)
            new_builds+=1+int(full not in newfull and first>0)
        savings.append(dict(sequence=budget,legacy_hashes=legacy_builds,optimized_hashes=new_builds))
    class Pairs(list):pass
    def unique(pairs):
        if not isinstance(pairs,Pairs):raise ValueError('object required')
        d={};seen=set()
        for k,v in pairs:
            if k.lower()in seen:raise ValueError('duplicate')
            seen.add(k.lower());d[k]=v
        return d
    def name(x,limit):return isinstance(x,str)and bool(x.strip())and len(x)<=limit and not any(unicodedata.category(c)=='Cc'for c in x)
    def parse(raw,expected=None):
        root={k.lower():v for k,v in unique(json.loads(raw,object_pairs_hook=Pairs)).items()}
        if not name(root.get('kind'),64)or not isinstance(root.get('data'),Pairs):raise ValueError('shape')
        ident=root.get('id')
        if ident is not None and not name(ident,128):raise ValueError('id')
        if root['kind']=='completed':
            if ident is None:raise ValueError('id')
            data=unique(root['data'])
            if not name(data.get('command'),64):raise ValueError('command')
            if expected is not None and data['command']!=expected:raise ValueError('correlation')
        return root
    good=['{"kind":"ready","data":{}}','{"kind":"future-event","data":{"x":1}}','{"kind":"completed","id":"1","data":{"command":"train","success":true}}']
    bad=['{}','[]','null','{"kind":"ready","data":[]}',
         '{"kind":"ready","id":"a","Id":"a","data":{}}',
         '{"kind":"completed","data":{"command":"train"}}',
         '{"kind":"completed","id":"1","data":{"success":true}}',
         '{"kind":"completed","id":"1","data":{"command":"wrong","success":true}}',
         '{"kind":"completed","id":"1","data":{"command":"train","success":false,"Success":true}}',
         '{"kind":"completed","id":"1","data":{"command":42}}',
         '{"kind":"completed","id":"1","data":{"command":null}}']
    for raw in good:assert parse(raw,'train')
    for raw in bad:
        try:parse(raw,'train')
        except ValueError:pass
        else:raise AssertionError(raw)
    # Explicit state-machine specification only. The C# headless test exercises real Tasks/gate/IO.
    class LoadPolicy:
        def __init__(self):self.current=None;self.jobs=[];self.active='last-verified'
        def submit(self,signature):
            if self.current and self.current['sig']==signature and self.current['state']=='running':return self.current
            if self.current and self.current['state']=='running':self.current['state']='cancelled'
            j={'sig':signature,'state':'running'};self.current=j;self.jobs.append(j);return j
        def finish(self,j,ok):
            if j['state']!='running':return
            j['state']='success'if ok else'failed'
            if j is self.current and ok:self.active=j['sig']
    cases=0
    for oldok,newok in ((True,True),(True,False),(False,True),(False,False)):
        p=LoadPolicy();old=p.submit((1,1,'a'));new=p.submit((1,2,'b'));same=p.submit((1,2,'b'))
        assert same is new and old['state']=='cancelled'and p.active=='last-verified'
        p.finish(old,oldok);assert p.active=='last-verified';p.finish(new,newok)
        assert p.active==((1,2,'b')if newok else'last-verified');cases+=1
    p=LoadPolicy();old=p.submit((1,2,'b'));new=p.submit((2,2,'b'));assert old is not new and old['state']=='cancelled';cases+=1
    report=dict(status='passed',scope='Independent Python numerical, split-key, framing and state-machine specifications. NOT C# compilation, TPL/GUI tasks, OS IO, native TorchSharp, CUDA or application throughput.',
        projection_schedule_cases=schedules,projection_rows_verified=row_counts,rope_configurations=rope_cases,rope_float32_angles_checked=rope_values,
        split_key_pairs=keys,split_decisions=decisions,trimmed_history_cases=trimmed,split_hash_operation_counts=savings,
        event_envelope_cases=len(good)+len(bad),snapshot_state_specifications=cases,corpus_version=manifest['version'],
        csharp='NOT_RUN',performance_timing='NOT_MEASURED')
    (ROOT/'reports/audit12-reference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
