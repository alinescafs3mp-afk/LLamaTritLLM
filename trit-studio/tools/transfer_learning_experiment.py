#!/usr/bin/env python3
"""Actual independent PyTorch A/B, not C#/CUDA. SAME expanded v22 train and frozen controls.
Legacy v21 reference pools+length buckets vs explicit per-batch language/final-fact/general quotas.
Evaluate development holdouts ONLY at the end; log every reply, never use targets for generation.
"""
import argparse,collections,copy,hashlib,json,math,pathlib,random,time
import torch
import torch.nn.functional as F
from learning_reference import Model
from learning_corpus_experiment import encode_row,batch,evaluate
from conversation_course_experiment import reply
from conversation_full_experiment import key,msgs
from context_learning_experiment import expand,normal,contrast
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name):return [json.loads(x) for x in (ROOT/'data'/name).read_text().splitlines()if x.strip()]
PATTERNS=((1,1,0,1,2,1,0,2),(1,2,0,2,1,2,0,1),(1,0,2,1,0,2,0,1))
def prepare():
 control=load('validation.jsonl');guard={key(r)for r in control}
 starter=load('conversation-starter.jsonl');oldctx=load('conversation-context.jsonl')
 language=starter+load('conversation-language.jsonl');facts=oldctx+load('conversation-transfer.jsonl')
 allrows=expand(load('seed.jsonl')+language+facts,guard)
 mapping={key(r,True):i for i,r in enumerate(allrows)}
 lang=[mapping[key(r,True)] for r in language];fact=[mapping[key(r,True)] for r in facts]
 buckets=collections.defaultdict(list)
 for r,i in zip(facts,fact):buckets[msgs(r)[-2]['content']].append(i)
 factgroups=list(buckets.values());encoded=[encode_row(r)for r in allrows];n=len(encoded)
 oldbasic=[mapping[key(r,True)]for r in starter]
 oldcontext=[mapping[key(r,True)]for r in expand(oldctx,guard)]
 def replay(pool,k):return [pool[i%len(pool)]for i in range(k)]
 oldpools=[list(range(n))+replay(oldbasic,2*n)+replay(oldcontext,n),list(range(n))+replay(oldbasic,n//2)+replay(oldcontext,n),list(range(n))+replay(oldbasic,n//4)+replay(oldcontext,n//2)]
 allinputs={key(r)for r in allrows};tests=load('transfer-challenge.jsonl')
 assert not any(key(r)in allinputs for r in tests),'test input leak'
 return allrows,encoded,lang,factgroups,oldpools,[encode_row(r)for r in control],tests
@torch.no_grad()
def check_cases(m,rows):
 results=[]
 for r in rows:
  history=[];turns=[]
  for msg in msgs(r)[::2]:
   q=msg['content'];keep=[];used=len(q.encode())+4+112
   for u,a in reversed(history):
    size=len(u.encode())+len(a.encode())+4
    if used+size>1024:break
    used+=size;keep.insert(0,(u,a))
   answer=reply(m,q,keep,limit=112);turns.append({'user':q,'answer':answer,'dropped':len(history)-len(keep)})
   history.append((q,answer))
  expected=msgs(r)[-1]['content'];results.append({'id':r['id'],'family':r['family'],'turns':turns,'expected':expected,'exact':normal(history[-1][1])==normal(expected) and all(t['dropped']==0 for t in turns)})
 groups={f:{'exact':sum(r['exact']for r in results if r['family']==f),'total':sum(r['family']==f for r in results)}for f in sorted({r['family']for r in results})}
 return {'exact':sum(r['exact']for r in results),'total':len(results),'groups':groups,'cases':results}
def run(a):
 if not 1<=a.steps<=100000:raise ValueError('steps')
 torch.set_num_threads(2);torch.manual_seed(42);rng=random.Random(42)
 raw,rr,lang,factgroups,pools,control,tests=prepare();m=Model(d=128,hid=384,n=4,heads=4,kv=2)
 tables={}
 def rotate(x,heads):
  b,t,_=x.shape;hd=m.hd
  if t not in tables:
   angle=torch.arange(t)[:,None]/torch.pow(10000,torch.arange(hd//2)*2/hd)[None,:]
   tables[t]=(angle.cos()[None,:,None,:],angle.sin()[None,:,None,:])
  c,s=tables[t];v=x.reshape(b,t,heads,hd//2,2);ev,od=v[...,0],v[...,1]
  return torch.stack((ev*c-od*s,ev*s+od*c),-1).reshape(b,t,heads,hd)
 m.rotate=rotate
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);oldbuckets=[]
 for pool in pools:
  buck=collections.defaultdict(list)
  for j,i in enumerate(pool):buck[(len(rr[i][0])-1)//32].append(j)
  oldbuckets.append(buck)
 start=time.monotonic();initial=evaluate(m,control);best=initial['loss'];last=copy.deepcopy(m.state_dict());last_step=0
 report={'scope':'Independent CPU PyTorch actual learning, medium821120 B8, same v22 data/initial seed across arms; different from C# RNG/device. NOT fluent-chat acceptance. Guard enforced. Holdout responses evaluated only after training.','arm':a.arm,'steps_requested':a.steps,'seed':42,'batch':8,'lr_peak':.001,'enforce_guard':True,'parameters':sum(p.numel()for p in m.parameters()),'unique_views':len(rr),'language_views':len(lang),'fact_groups':len(factgroups),'initial':initial,'data_sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()for n in ['seed.jsonl','conversation-starter.jsonl','conversation-context.jsonl','conversation-language.jsonl','conversation-transfer.jsonl','validation.jsonl','context-challenge.jsonl','transfer-challenge.jsonl']},'observations':[]}
 out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 for step in range(1,a.steps+1):
  ph=0 if (step-1)*5<a.steps else 1 if(step-1)*5<a.steps*3 else 2
  if a.arm=='legacy':
   pp=pools[ph];anchor=rng.randrange(len(pp));buck=oldbuckets[ph][(len(rr[pp[anchor]][0])-1)//32];idx=[pp[anchor]]+[pp[rng.choice(buck)]for _ in range(7)]
  else:
   idx=[]
   for slot in range(8):
    role=PATTERNS[ph][((step-1)*8+slot)%8]
    if role==0:idx.append(rng.randrange(len(rr)))
    elif role==1:idx.append(rng.choice(lang))
    else:idx.append(rng.choice(rng.choice(factgroups)))
  ids,y=batch(rr,idx);warm=min(100,a.steps//10);lr=.001*(step/warm if warm and step<=warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,a.steps-warm))))
  for group in opt.param_groups:group['lr']=lr
  opt.zero_grad();z=m(ids);per=F.cross_entropy(z.reshape(-1,262),y.reshape(-1),reduction='none',ignore_index=-100).reshape(y.shape)
  loss=(per.sum(1)/(y>=0).sum(1)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%500 and step!=a.steps:continue
  val=evaluate(m,control);reject=val['loss']>best+max(.02,best*.2)
  report['observations'].append({'step':step,'phase':ph,'lr':lr,'control':val,'guard_rejected':reject,'seconds':round(time.monotonic()-start,2)})
  if reject:m.load_state_dict(last);report['stop_reason']='validation_regression';break
  best=min(best,val['loss']);last=copy.deepcopy(m.state_dict());last_step=step
  out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(a.arm,step,val,round(time.monotonic()-start,1),flush=True)
 else:report['stop_reason']='requested_steps_completed'
 report['last_accepted_step']=last_step;report['final_control']=evaluate(m,control)
 report['transfer']=check_cases(m,tests);report['old_context']=contrast(m,load('context-challenge.jsonl'))
 report['public_open']={q:reply(m,q,limit=128)for q in ['Привет, как ты?','Давай немного поговорим про книги.','Я закончил рисунок, хочу поделиться.','Я сегодня гулял.','Не хочу советов, просто послушай.','Что посоветуешь почитать?']}
 report['known']=check_cases(m,[load('conversation-language.jsonl')[i]for i in range(0,160,8)])
 report['elapsed_seconds']=round(time.monotonic()-start,2)
 torch.save(m.state_dict(),out.with_suffix('.pt'))
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(a.arm,'DONE',report['transfer']['exact'],report['old_context']['pairs_passed'],report['public_open'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['legacy','balanced'],required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--output',required=True);run(p.parse_args())
