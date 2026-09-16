#!/usr/bin/env python3
"""Independent byte/token/mask equivalence specifications. This never imports or executes C#."""
import hashlib, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def tokens(s): return [b+6 for b in s.encode('utf-8','strict')]
def parts(row):
    if 'text' in row: return row['text'],None,[]
    if 'prompt' in row: return row['prompt'],row['answer'],[]
    m=row['messages']; return m[-2]['content'],m[-1]['content'],[(m[i]['content'],m[i+1]['content']) for i in range(0,len(m)-2,2)]
def plan(row,limit):
    text,answer,h=parts(row)
    n=len(tokens(text))+(1 if answer is None else len(tokens(answer))+4)
    if n>limit: raise ValueError('target too long')
    first=len(h)
    if answer is not None:
        while first:
            u,a=h[first-1];need=len(tokens(u))+len(tokens(a))+4
            if n+need>limit:break
            n+=need;first-=1
    return n,first

def old_encode(row,limit):
    text,answer,h=parts(row); n,first=plan(row,limit); ids=[1];target=1
    if answer is not None:
        for u,a in h[first:]:ids += [3]+tokens(u)+[2,4]+tokens(a)+[2]
        ids += [3]+tokens(text)+[2,4];target=len(ids);ids+=tokens(answer)+[2]
    else:ids+=tokens(text)+[2]
    return ids[:-1],[ids[i+1] if i+1>=target else -100 for i in range(len(ids)-1)]

def direct_encode(row,limit):
    text,answer,h=parts(row);n,first=plan(row,limit)
    inputs=[None]*n;labels=[-100]*n;at=0;target=1
    def put(xs):
        nonlocal at
        assert at+len(xs)<=n
        inputs[at:at+len(xs)]=xs;at+=len(xs)
    put([1])
    if answer is not None:
        for u,a in h[first:]:put([3]);put(tokens(u));put([2,4]);put(tokens(a));put([2])
        put([3]);put(tokens(text));put([2,4]);target=at;put(tokens(answer))
    else:put(tokens(text))
    assert at==n
    labels[target-1:n-1]=inputs[target:n];labels[-1]=2
    return inputs,labels

def run():
    corpus_checks=0;streamed_checks=0;sequence_max=0
    for file in (ROOT/'data').glob('*.jsonl'):
        for line in file.read_text().splitlines():
            row=json.loads(line);a=old_encode(row,512);b=direct_encode(row,512)
            assert a==b,(file,row['id']);corpus_checks+=1;sequence_max=max(sequence_max,len(b[0]))
    rng=random.Random(71010);alphabet=['a','é','я','🙂','中','\n','\t','\u07ff','\u0800','\U0010ffff']
    def phrase(): return ''.join(rng.choice(alphabet) for _ in range(rng.randrange(1,18)))
    for _ in range(800):
        if rng.randrange(4)==0:row={'text':phrase()}
        else:
            messages=[]
            for j in range(rng.randrange(1,7)):
                messages += [{'role':'user','content':phrase()},{'role':'assistant','content':phrase()}]
            row={'messages':messages}
        for limit in [32,64,128,256,512,1024,2048]:
            try:a=old_encode(row,limit)
            except ValueError:
                try:direct_encode(row,limit)
                except ValueError:continue
                raise AssertionError('direct accepted truncated target')
            assert a==direct_encode(row,limit);streamed_checks+=1
    boundaries=0
    for count in [0,1,1023,1024,1025,16384,32768]:
        for s in ['x','я','🙂']:
            raw=(s*count).encode();out=[-1]*(len(raw)+9)
            out[4:4+len(raw)]=[b+6 for b in raw]
            assert out[:4]==[-1]*4 and out[4+len(raw):]==[-1]*5
            assert out[4:4+len(raw)]==tokens(s*count);boundaries+=1
    base=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    for stem,key in [('test','frozen_v2_test_sha256'),('challenge','frozen_v3_challenge_sha256'),('pretrain','frozen_baseline_sha256')]+[(f'challenge-v{v}',f'frozen_v{v}_challenge_sha256') for v in range(4,10)]:
        assert hashlib.sha256((ROOT/f'data/{stem}.jsonl').read_bytes()).hexdigest()==base[key]
    report={'status':'passed','scope':'Independent Python encoding specification, NOT C# execution, native speed or real-process fault evidence',
      'corpus_encoding_pairs':corpus_checks,'random_encoding_pairs_including_trimmed_histories':streamed_checks,'buffer_boundaries':boundaries,
      'max_corpus_sequence':sequence_max,'frozen_baseline_and_eight_holdouts':True,
      'not_executed':['C# byte encoder','C# pipe timeouts','worker fail-closed mode persistence','native reload avoidance','C# benchmarks','GUI','CUDA']}
    (ROOT/'reports/audit10-reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
