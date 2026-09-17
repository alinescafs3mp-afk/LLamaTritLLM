#!/usr/bin/env python3
"""Opt-in independent A/B with v21 data, v20 versus v21 sampling. NOT C#/CUDA.
Both arms see identical source/derived examples; only phase pools differ. All raw results retained.
Same initialization seed, optimizer, budget, guard cadence. Actual stopping guard is enforced by default.
The controls and contrast targets NEVER train. Fixed public diagnostics are development evidence.
"""
import argparse,collections,hashlib,json,math,pathlib,random,time,copy
import torch
import torch.nn.functional as F
from learning_reference import Model
from learning_corpus_experiment import encode_row,batch,evaluate
from conversation_course_experiment import reply
from conversation_full_experiment import key,msgs
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(n):return [json.loads(l)for l in (ROOT/'data'/n).read_text().splitlines() if l.strip()]
def expand(source,guard):
 out={key(r,True):r for r in source}
 for r in source:
  if 'text'in r:continue
  m=msgs(r)
  for end in range(2,len(m),2):
   p={'messages':m[:end]}
   if key(p)not in guard:out.setdefault(key(p,True),p)
 return list(out.values())
def prepare():
 starter=load('conversation-starter.jsonl');context=load('conversation-context.jsonl');control=load('validation.jsonl');held=load('context-challenge.jsonl')
 guard={key(r)for r in control};ctx=expand(context,guard);allrows=expand(load('seed.jsonl')+starter+context,guard)
 trainkeys={key(r) for r in allrows}
 assert not any(key(p) in trainkeys for p in expand(held,set())), 'Derived held-out prefix overlaps training'
 a=[encode_row(r)for r in allrows];b=[encode_row(r)for r in starter];c=[encode_row(r)for r in ctx];d=[e for r,e in zip(allrows,a)if 'text'not in r]
 def replay(pool,n):return [pool[i%len(pool)]for i in range(n)]
 old=[b,d+replay(b,max(1,len(d)//4)),a+replay(b,max(1,len(a)//4))]
 new=[a+replay(b,2*len(a))+replay(c,len(a)),a+replay(b,len(a)//2)+replay(c,len(a)),a+replay(b,len(a)//4)+replay(c,len(a)//2)]
 return old,new,[encode_row(r)for r in control],held,len(allrows),len(ctx)
def normal(s):return s.strip().rstrip('.!?').casefold().replace('ё','е')
@torch.no_grad()
def contrast(m,held):
 out=[]
 for r in held:
  history=[];messages=msgs(r)
  for msg in messages[::2]:
   q=msg['content'];answer=reply(m,q,history,limit=64);history.append((q,answer))
  expected=messages[-1]['content'];out.append({'id':r['id'],'turns':history,'expected':expected,'exact':normal(history[-1][1])==normal(expected)})
 pairs=[{'both_correct':out[i]['exact'] and out[i+1]['exact'],'same_answer':normal(out[i]['turns'][-1][1])==normal(out[i+1]['turns'][-1][1])}for i in range(0,len(out),2)]
 return {'pairs_passed':sum(p['both_correct']for p in pairs),'pairs_total':len(pairs),'same_answer_pairs':sum(p['same_answer']for p in pairs),'exact_cases':sum(r['exact']for r in out),'cases':out}
def run(args):
 if not 1<=args.steps<=100000 or not 1<=args.every<=1000: raise ValueError("steps must be1..100000; every must be1..1000")
 torch.set_num_threads(2);old,new,controls,held,n,nc=prepare();result={'scope':'Independent PyTorch CPU, identical v21 source data in both arms, differing sampling pools; not C#/CUDA, not a general conversation score. Real validation regression guard enforced at each interval.','parameters':821120,'initialization':args.initialization,'seed':42,'steps_requested':args.steps,'batch':8,'lr_peak':.001,'publication_interval':args.every,'data_sha256':{f:hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest()for f in ['seed.jsonl','conversation-starter.jsonl','conversation-context.jsonl','context-challenge.jsonl','validation.jsonl']},'unique_expanded_rows':n,'context_expanded_rows':nc,'arms':{}}
 for name,pools in [('v20_phase_pools',old),('v21_context_mixed',new)]:
  if args.only and name!=args.only:continue
  torch.manual_seed(42);rng=random.Random(42);m=Model(d=128,hid=384,n=4,heads=4,kv=2)
  if args.initialization=='small-residual':
   with torch.no_grad():
    for tensor_name,r,c,q in m.sh:
     if q and tensor_name!='embedding':
      factor=.02*math.sqrt(c)/(math.sqrt(2*m.n) if tensor_name.endswith(('_out','_down')) else 1)
      m.ps[tensor_name].mul_(factor)
  opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);buckets=[]
  for rr in pools:
   buck=collections.defaultdict(list)
   for i,(x,y)in enumerate(rr):buck[(len(x)-1)//32].append(i)
   buckets.append(buck)
  arm={'pool_sizes':[len(x)for x in pools],'observations':[]};result['arms'][name]=arm
  start=time.monotonic();best=evaluate(m,controls)['loss'];arm['initial_control_loss']=best;last=copy.deepcopy(m.state_dict());last_step=0
  for step in range(1,args.steps+1):
   phase=0 if (step-1)*5<args.steps else 1 if(step-1)*5<args.steps*3 else 2;rr=pools[phase]
   a=rng.randrange(len(rr));bucket=buckets[phase][(len(rr[a][0])-1)//32];ids,y=batch(rr,[a]+[rng.choice(bucket)for _ in range(7)])
   warm=min(100,args.steps//10);lr=.001*(step/warm if warm and step<=warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm-1)/max(1,args.steps-warm-1))))
   for pg in opt.param_groups:pg['lr']=lr
   opt.zero_grad();z=m(ids);per=F.cross_entropy(z.reshape(-1,262),y.reshape(-1),reduction='none',ignore_index=-100).reshape(y.shape);loss=(per.sum(1)/(y>=0).sum(1)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   if step%args.every and step!=args.steps:continue
   control=evaluate(m,controls);reject=control['loss']>best+max(.02,.2*best);o={'step':step,'phase':phase,'seconds':round(time.monotonic()-start,2),'control':control,'rejected':reject,'best_before':best,'lr':lr}
   arm['observations'].append(o)
   if reject:m.load_state_dict(last);arm['stop_reason']='validation_regression';arm['last_committed_step']=last_step
   else:best=min(best,control['loss']);last=copy.deepcopy(m.state_dict());last_step=step;arm['last_committed_step']=step
   pathlib.Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(name,json.dumps(o),flush=True)
   if reject:break
  else:arm['stop_reason']='requested_steps_completed'
  arm['final_control']=evaluate(m,controls);arm['context']=contrast(m,held)
  prompts=['Привет, как ты?','Давай немного поговорим про книги.','Я закончил рисунок, хочу поделиться.','Приветик!','Я дочитал книгу.']
  arm['free_answers']={q:reply(m,q,limit=112)for q in prompts};arm['elapsed_seconds']=round(time.monotonic()-start,2)
  # Public synthetic experiment model state, retained locally for reproducibility, never loaded into user models.
  torch.save(m.state_dict(),str(pathlib.Path(args.output).with_suffix(''))+'-'+name+'.pt')
  pathlib.Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(name,'DONE',arm['context']['pairs_passed'],arm['free_answers'],flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=3000);p.add_argument('--every',type=int,default=250);p.add_argument('--only',choices=['v20_phase_pools','v21_context_mixed']);p.add_argument('--initialization',choices=['legacy','small-residual'],default='legacy');p.add_argument('--output',required=True);run(p.parse_args())
