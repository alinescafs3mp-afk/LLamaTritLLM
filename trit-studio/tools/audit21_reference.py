#!/usr/bin/env python3
"""Independent specifications for v21. NOT C#, GUI, native scheduling or model-quality evidence."""
import hashlib,json,pathlib
from context_learning_experiment import prepare,load,expand,normal,key
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 old,new,control,held,n,nc=prepare()
 phases=[]
 for total in [1,2,3,5,17,100,3000,6000]:
  sequence=[0 if i*5<total else 1 if i*5<total*3 else 2 for i in range(total)]
  assert sequence==sorted(sequence)
  phases.append({'steps':total,'phase_counts':[sequence.count(i)for i in range(3)]})
 allrows=expand(load('seed.jsonl')+load('conversation-starter.jsonl')+load('conversation-context.jsonl'),{key(x)for x in load('validation.jsonl')})
 # The Python pools share tuple/list owners instead of copying encoded tokens.
 full_owners={id(x) for x in new[0][:n]}
 for pool in new:
  assert full_owners <= {id(x)for x in pool}
 assert len(new[0])==4*n and len(new[1])==2*n+n//2 and len(new[2])==n+n//4+n//2
 checks=[]
 for i in range(0,len(held),2):
  a,b=held[i],held[i+1];ma,mb=a['messages'],b['messages']
  assert ma[-2]['content']==mb[-2]['content']
  assert normal(ma[-1]['content'])!=normal(mb[-1]['content'])
  assert [x['content']for x in ma[::2]]!=[x['content']for x in mb[::2]]
  for guess in [ma[-1]['content'],mb[-1]['content'],'Понял.','Лена или Нина.','']:
   assert not (normal(guess)==normal(ma[-1]['content']) and normal(guess)==normal(mb[-1]['content']))
  checks.append({'pair':i//2,'constant_reply_cannot_pass':True})
 derived=[x for x in expand(held,set())]
 trainkeys={key(r)for r in allrows}
 assert not any(key(r)in trainkeys for r in derived)
 data=load('seed.jsonl')+load('conversation-starter.jsonl')+load('conversation-context.jsonl')+held
 from learning_corpus_experiment import encode_row
 maxseq=max(len(encode_row(r)[0])+1 for r in data)
 assert maxseq<=512
 # This is the production guard formula, not an alternate relaxed loss threshold.
 anchor=1.619321091802457;bad=2.219646608248914
 assert bad>anchor+max(.02,.2*anchor)
 report={'status':'passed','scope':__doc__,'source_rows_before_derivation':len(load('seed.jsonl'))+len(load('conversation-starter.jsonl'))+len(load('conversation-context.jsonl')),
 'unique_expanded_rows':n,'context_expanded_rows':nc,'pool_sizes_legacy':[len(x)for x in old],
 'pool_sizes_mixed':[len(x)for x in new],'phase_boundaries':phases,'contrast_pairs':checks,
 'all_derived_probe_inputs_excluded_from_training':True,'max_full_sequence':maxseq,
 'native_build':'NOT_RUN','data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((ROOT/'data').glob('*.jsonl'))}}
 (ROOT/'reports/audit21-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k!='data_sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
