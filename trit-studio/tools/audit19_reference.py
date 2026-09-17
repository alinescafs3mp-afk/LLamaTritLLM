#!/usr/bin/env python3
"""Independent policy/corpus algorithms, NOT C# execution or model quality certification."""
from pathlib import Path
import math, json, hashlib, collections
import numpy as np
from learning_corpus_experiment import encode_row
ROOT=Path(__file__).resolve().parents[1]
def rate(steps,peak,warmup,i):
    if steps==1:return peak
    w=min(warmup,steps//10);n=i+1
    if w>0 and n<=w:return peak*n/w
    return peak*(.1+.9*.5*(1+math.cos(math.pi*(n-w-1)/max(1,steps-w-1))))
def cadence(steps,requested,existing,initial=False,auto=False):
    available=max(0,128-existing-int(initial));interval=requested
    if auto and steps>0 and available>0:interval=max(interval,math.ceil(steps/min(32,available)))
    needed=math.ceil(steps/interval)+int(initial)
    if needed>max(0,128-existing):raise ValueError('quota')
    return interval,needed

def run():
    cases=0
    for n in [1,2,3,9,10,20,100,1000,8000,100000]:
        for warm in [0,1,100,10000]:
            a=[rate(n,.001,warm,i)for i in range(n)];w=min(warm,n//10)
            assert all(0<x<=.001000000000001 for x in a)
            if n>1:assert abs(a[-1]-.0001)<1e-12
            assert all(a[i]<=a[i-1]+1e-12 for i in range(max(1,w+1),n));cases+=1
    budgets=0
    for n in [0,1,5,8000,100000]:
        for existing in [0,1,51,95,126,127,128]:
            for initial in [False,True]:
                for interval in [1,100,1000,100000]:
                    for auto in [False,True]:
                        try:p,count=cadence(n,interval,existing,initial,auto)
                        except ValueError:continue
                        assert p>=interval and p<=100000 and existing+count<=128
                        if auto and n>0:assert count-int(initial)<=32
                        budgets+=1
    assert cadence(8000,100,51,auto=True)==(250,32)
    assert cadence(8000,1000,51,auto=True)==(1000,8)
    def load(name):return[json.loads(x)for x in(ROOT/'data'/name).read_text().splitlines()]
    seed,base,control=load('seed.jsonl'),load('pretrain.jsonl'),load('validation.jsonl')
    ids={x['id']for x in base};staged=[x for x in base if x['id']not in ids]+seed
    assert staged==seed and len(ids)==48
    def baseline(train):
        counts=np.zeros((262,262),np.int64)
        for row in train:
            x,y=encode_row(row)
            for a,b in zip(x,y):
                if b!=-100:counts[a,b]+=1
        marginal=counts.sum(0);pred=counts.argmax(1);pred[counts.sum(1)==0]=marginal.argmax()
        good=total=0
        for row in control:
            x,y=encode_row(row)
            for a,b in zip(x,y):
                if b!=-100:total+=1;good+=int(pred[a]==b)
        return {'correct':good,'total':total,'accuracy_percent':100*good/total}
    def repeated(s):
        longest=max((len(list(g))for _,g in __import__('itertools').groupby(s)),default=0)
        grams={s[i:i+4]for i in range(len(s)-3)};distinct=len(grams)/max(1,len(s)-3)
        return longest>=12 or len(s)>=48 and distinct<.15
    assert repeated('о'*64) and repeated('🙂'*16) and repeated('аб'*64)
    assert not repeated('Привет! Как дела?') and not repeated('')
    result={'status':'passed','scope':__doc__,'schedule_cases':cases,'accepted_budget_cases':budgets,
        'current_data_version':json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())['version'],
        'v19_reference_seed_only':baseline(seed),'v19_reference_seed_and_basics':baseline(seed+base),
        'route_order':'Exact zero-step placeholder removal equals direct seed; actual Worker test still NOT_RUN',
        'frozen_baseline_sha256':hashlib.sha256((ROOT/'data/pretrain.jsonl').read_bytes()).hexdigest(),
        'csharp':'NOT_RUN','new_schedule_training_benefit':'NOT_MEASURED',
        'executed_learning_evidence':'reports/learning-investigation-v19.json, independent constant-LR v18 arms only'}
    (ROOT/'reports/audit19-reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
