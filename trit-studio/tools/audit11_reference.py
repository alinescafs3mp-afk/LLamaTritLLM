"""Independent Python specifications for audit11. Does NOT execute C# or OS subprocess semantics."""
from pathlib import Path
import itertools,json,random,hashlib,math
ROOT=Path(__file__).resolve().parents[1]

def main():
    replay_cases=0; consumed_bad=0; signature=hashlib.sha256()
    # Tuple strings model exact encoded row payloads, not trained neural behavior.
    for b,n,k in itertools.product([0,1,2,7,64,256,50000],[0,1,2,7,64],[1,2,3,4]):
        if not b+n:continue
        baseline=[('base',i) for i in range(b)]; learned=[('learned',i) for i in range(n)]
        recent=[('new',i) for i in range(k)]
        for step in sorted(set([0,max(0,b-1),b,b+n-1,b+n,(1<<63)-1])):
            legacy=recent+[ (baseline+learned)[(step+i)%(b+n)] for i in range(k)]
            cache={}; selected=list(recent)
            for i in range(k):
                j=(step+i)%(b+n)
                if j<b: selected.append(baseline[j])
                else:
                    cache.setdefault(j-b,learned[j-b]);selected.append(cache[j-b])
            assert selected==legacy and len(cache)<=min(n,k)
            signature.update(repr((b,n,k,step,selected)).encode());replay_cases+=1
            # Consume-time validation rejects a selected bad entry, never learns an unselected one.
            for bad in range(n):
                assert (bad in cache)==any(x==('learned',bad) for x in selected)
                consumed_bad+=1
    quota_cases=0
    for steps in [0,1,2,99,100,101,127,128,129,1999,100000]:
        for every in [1,2,3,7,100,999,1000]:
            for initial in [0,1]:
                count=(steps+every-1)//every+initial
                actual=len(range(0,steps,every))+initial
                assert count==actual
                for existing in [0,1,8,120,127,128,200]:
                    accepts=count<=max(0,128-existing)
                    assert accepts==(existing+count<=128 or (existing>=128 and count==0))
                    quota_cases+=1
    class BadEnvelope(ValueError):pass
    def obj(pairs):
        result={};seen=set()
        for k,v in pairs:
            # Apply duplicates at root below, JSON object nesting has its ordinary meaning.
            if k.lower() in seen:raise BadEnvelope('duplicate')
            seen.add(k.lower());result[k]=v
        return result
    def parse(raw):
        x=json.loads(raw,object_pairs_hook=obj)
        if not isinstance(x,dict):raise BadEnvelope('root')
        # Official .NET Web deserializer is case insensitive for the declared record fields.
        x={k.lower():v for k,v in x.items()}
        for key,limit in [('kind',64),('id',128)]:
            v=x.get(key)
            if not isinstance(v,str) or not v.strip() or len(v)>limit or any(ord(c)<32 or 127<=ord(c)<=159 for c in v):raise BadEnvelope('identity')
        if not isinstance(x.get('payload'),dict):raise BadEnvelope('payload')
        return x
    good=['{"kind":"mode","id":"1","payload":{"enabled":false}}',
          '{"kind":"unknown","id":"2","payload":{}}','{"Kind":"stop","Id":"3","Payload":{}}']
    bad=['{','[]','null','{}','{"kind":"x","id":1,"payload":{}}',
         '{"kind":"x","id":"a","Id":"b","payload":{}}',
         '{"kind":"x","id":"a","payload":false}',
         '{"kind":"x","id":"a","payload":null}',
         '{"kind":"x","id":"a","payload":[]}',
         json.dumps(dict(kind='x',id='a\nb',payload={})),json.dumps(dict(kind='x',id='a'*129,payload={}))]
    for x in good:assert parse(x)
    for x in bad:
        try:parse(x)
        except (ValueError,TypeError):pass
        else:raise AssertionError(x)
    # Side-effect models document intended UI policy; do NOT pretend to run C# continuations.
    stale_cases=0
    for epoch_ok,client_ok,pending_ok,closing,busy,ready in itertools.product([False,True],repeat=6):
        sends=epoch_ok and client_ok and pending_ok and not closing and not busy and ready
        assert sends==(all([epoch_ok,client_ok,pending_ok,ready]) and not (closing or busy));stale_cases+=1
    m=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    report=dict(status='passed',scope='Independent Python index/budget/framing-policy specifications, NOT C# execution, native speed or actual process/UI behavior',
        replay_pool_parity_cases=replay_cases,selected_vs_unselected_guard_cases=consumed_bad,
        replay_trace_sha256=signature.hexdigest(),checkpoint_publication_budget_cases=quota_cases,
        command_envelope_cases=len(good)+len(bad),ui_policy_boolean_combinations=stale_cases,
        bounded_learned_rows_per_online_pool=4,legacy_learned_rows_encoded_per_pool=64,
        corpus_version=m['version'],csharp_tests='NOT_RUN')
    (ROOT/'reports/audit11-reference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
