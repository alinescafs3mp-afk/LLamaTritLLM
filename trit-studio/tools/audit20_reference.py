#!/usr/bin/env python3
"""Independent objective/prefix/course specifications. NOT C# runtime or GUI evidence."""
import json,pathlib,hashlib,unicodedata,collections,math
import numpy as np,torch
import torch.nn.functional as F
from learning_corpus_experiment import encode_row
ROOT=pathlib.Path(__file__).resolve().parents[1]
def rows(name):return [json.loads(l)for l in (ROOT/'data'/name).read_text().splitlines()]
def normalize(s):return ' '.join(unicodedata.normalize('NFKC',s).split()).upper()
def messages(r):return r.get('messages',[{'role':'user','content':r.get('prompt','')},{'role':'assistant','content':r.get('answer','')}])
def identity(r):
 if'text'in r:return ('text',r['text'].strip())
 return tuple((m['role'],m['content'])for m in messages(r))
def guardkey(r):
 if'text'in r:return ('text',normalize(r['text']))
 return tuple((m['role'],normalize(m['content']))for m in messages(r)[:-1])
def main():
 torch.set_num_threads(1);torch.manual_seed(23);cases=[]
 for counts in [(1,12),(5,2,70),(1,1),(17,9,4,33)]:
  for vocab in [7,262]:
   for dtype in [torch.float32,torch.float64]:
    n=sum(counts);a=torch.randn(n,vocab,dtype=dtype,requires_grad=True);b=a.detach().clone().requires_grad_();y=torch.randint(vocab,(n,));weights=torch.tensor([1/(len(counts)*c) for c in counts for _ in range(c)],dtype=dtype)
    actual=(F.cross_entropy(a,y,reduction='none')*weights).sum();offset=0;parts=[]
    for c in counts:parts.append(F.cross_entropy(b[offset:offset+c],y[offset:offset+c]));offset+=c
    expected=torch.stack(parts).mean();actual.backward();expected.backward();error=float((a.grad-b.grad).abs().max())
    assert abs(float(actual-expected))<1e-6 and error<1e-7
    assert abs(float(weights.sum())-1)<1e-6
    cases.append({'targets':counts,'vocab':vocab,'dtype':str(dtype),'gradient_max_error':error})
 raw=rows('seed.jsonl')+rows('conversation-starter.jsonl');controls={guardkey(r)for r in rows('validation.jsonl')};out={identity(r):r for r in raw};excluded=0;prefixes=0
 for r in raw:
  assert guardkey(r)not in controls
  if'text'in r:continue
  m=messages(r)
  for end in range(2,len(m),2):
   prefix={'messages':m[:end]};prefixes+=1
   if guardkey(prefix)in controls:excluded+=1;continue
   out.setdefault(identity(prefix),prefix)
 total_before=sum(sum(y>=0 for y in encode_row(r)[1])for r in raw)
 total_after=sum(sum(y>=0 for y in encode_row(r)[1])for r in out.values())
 assert all(len(encode_row(r)[0])<=512 for r in out.values())
 assert all(guardkey(r)not in controls for r in out.values())
 assert len(out)>len(raw)
 # Each pair gets an EOS target. No history or user token is made a target for that pair.
 for r in out.values():
  x,y=encode_row(r)
  if'text'not in r:assert sum(v>=0 for v in y)==len(messages(r)[-1]['content'].encode())+1
 counts=collections.Counter('text'if'text'in r else 'multi'if len(messages(r))>2 else 'single'for r in rows('seed.jsonl'))
 sourcebytes=sum(len(x)for x,y in [encode_row(r)for r in raw]);expandedbytes=sum(len(x)for x,y in [encode_row(r)for r in out.values()])
 stages=[]
 for n in [1,2,3,4,5,19,100,6000]:
  phase=[0 if i*5<n else 1 if i*5<n*3 else 2 for i in range(n)]
  assert len(phase)==n and phase==sorted(phase)
  stages.append({'steps':n,'phase_steps':[phase.count(k)for k in range(3)]})
 report={'status':'passed','scope':'Independent Python objective/prefix/curriculum specifications; NOT C# execution or conversation quality',
  'objective_cases':cases,'phase_cases':stages,'source_records':len(raw),'main_types':counts,'all_prefix_candidates':prefixes,
  'derived_control_exclusions':excluded,'unique_expanded_examples':len(out),'new_target_examples':len(out)-len(raw),
  'supervised_byte_positions_before':total_before,'supervised_byte_positions_after':total_after,
  'input_positions_before':sourcebytes,'input_positions_after':expandedbytes,
  'growth_semantics':'Derived prefixes are training views of existing dialogues, not newly authored conversations.',
  'data_sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()for n in ['seed.jsonl','conversation-starter.jsonl','validation.jsonl','pretrain.jsonl']}}
 (ROOT/'reports/audit20-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['objective_cases','data_sha256']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
