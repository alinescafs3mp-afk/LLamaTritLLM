#!/usr/bin/env python3
"""Independent foundation experiment: real Python gradients, NOT C#/CUDA or full-course acceptance."""
import argparse,json,hashlib,time,math,pathlib,random
import numpy as np,torch
import torch.nn.functional as F
from learning_reference import Model
from learning_corpus_experiment import encode_row,batch,evaluate
ROOT=pathlib.Path(__file__).resolve().parents[1]
@torch.no_grad()
def reply(m,query,history=(),limit=160):
 ids=[1]
 for u,a in history:ids += [3]+[x+6 for x in u.encode()]+[2,4]+[x+6 for x in a.encode()]+[2]
 ids += [3]+[x+6 for x in query.encode()]+[2,4];out=[];remaining=0;lo=128;hi=191
 quant=m.quant; cached={n:quant(n)for n,r,c,q in m.sh if q};m.quant=lambda n:cached[n]
 try:
  for _ in range(limit):
   z=m(torch.tensor([ids]))[0,-1]; allowed=torch.zeros(262,dtype=torch.bool)
   if remaining==0:
    allowed[2]=True
    for b in [9,10,13]+list(range(32,127))+list(range(194,245)):
     need=1 if b<128 else 2 if b<224 else 3 if b<240 else 4
     if need<=limit-len(out):allowed[b+6]=True
   else:allowed[lo+6:hi+7]=True
   t=int(z.masked_fill(~allowed,-float('inf')).argmax())
   if t==2:break
   b=t-6;out.append(b);ids.append(t)
   if remaining:remaining-=1;lo=128;hi=191
   elif b>=128:
    remaining=1 if b<224 else 2 if b<240 else 3;lo=160 if b==224 else 144 if b==240 else 128;hi=159 if b==237 else 143 if b==244 else 191
  return bytes(out).decode('utf8','strict')
 finally:m.quant=quant

def run(a):
 torch.set_num_threads(2);torch.manual_seed(42);rng=random.Random(42)
 m=Model(d=128,hid=384,n=4,heads=4,kv=2) if a.medium else Model()
 raw=[json.loads(l)for l in (ROOT/'data/conversation-starter.jsonl').read_text().splitlines()];rows=[encode_row(r)for r in raw]
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 prompts=['Приветик!','Скажи своё имя.','Пиши сейчас покороче.','Я сегодня гулял.','Я дочитал книгу.','Я немного устал.',
  'Привет, как ты?','Давай немного поговорим про книги.','Я закончил рисунок, хочу поделиться.','Какое у тебя имя?']
 known={r['prompt']:r['answer']for r in raw};start=time.monotonic()
 report={'scope':'Independent Python FOUNDATION ONLY; not full curriculum, not C#/CUDA; known queries are recall, new queries require human review',
  'seed':42,'parameters':sum(p.numel()for p in m.parameters()),'records':len(rows),'batch':8,'steps':a.steps,'equal_example_loss':True,
  'starter_sha256':hashlib.sha256((ROOT/'data/conversation-starter.jsonl').read_bytes()).hexdigest(),'observations':[]}
 for step in range(a.steps+1):
  if step:
   ids,y=batch(rows,[rng.randrange(len(rows))for _ in range(8)])
   lr=.001*min(1,step/50)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-50)/max(1,a.steps-50))))
   for group in opt.param_groups:group['lr']=lr
   opt.zero_grad();z=m(ids);per=F.cross_entropy(z.reshape(-1,262),y.reshape(-1),reduction='none',ignore_index=-100).reshape(y.shape)
   loss=(per.sum(1)/(y>=0).sum(1)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step not in (0,400,800,1200,a.steps):continue
  replies={p:reply(m,p)for p in prompts};checks={p:replies[p]==known[p]for p in prompts if p in known}
  row={'step':step,'elapsed_seconds':round(time.monotonic()-start,2),'train':evaluate(m,rows),'replies':replies,'exact_known':checks}
  report['observations'].append(row);pathlib.Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1600);p.add_argument('--medium',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();run(a)
