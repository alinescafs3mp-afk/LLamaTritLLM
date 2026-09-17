#!/usr/bin/env python3
"""Opt-in independent CPU full-course experiment. NOT C#/CUDA or proof of broad fluency."""
import argparse,collections,json,math,pathlib,random,time,hashlib
import torch
import torch.nn.functional as F
from learning_reference import Model
from learning_corpus_experiment import encode_row,batch,evaluate
from conversation_course_experiment import reply
ROOT=pathlib.Path(__file__).resolve().parents[1]
def norm(s):return ' '.join(s.strip().split()).casefold()
def msgs(r):return r.get('messages',[{'role':'user','content':r.get('prompt','')},{'role':'assistant','content':r.get('answer','')}])
def key(r,answer=False):
 if 'text'in r:return ('text',norm(r['text']))
 m=msgs(r);return tuple((x['role'],norm(x['content']))for x in (m if answer else m[:-1]))
def load(n):return [json.loads(l)for l in (ROOT/'data'/n).read_text().splitlines()]
def prepare():
 starter=load('conversation-starter.jsonl');source=load('seed.jsonl')+starter;control=load('validation.jsonl');guard={key(r)for r in control};out={key(r,True):r for r in source};excluded=0
 for r in source:
  if'text'in r:continue
  m=msgs(r)
  for end in range(2,len(m),2):
   p={'messages':m[:end]}
   if key(p)in guard:excluded+=1;continue
   out.setdefault(key(p,True),p)
 allrows=list(out.values());allenc=[encode_row(r)for r in allrows];basic=[encode_row(r)for r in starter]
 dialogue=[e for r,e in zip(allrows,allenc)if'text'not in r]
 def mix(es):return es+[basic[i%len(basic)]for i in range(max(1,len(es)//4))]
 return [basic,mix(dialogue),mix(allenc)],[encode_row(r)for r in control],len(allrows),excluded

def run(a):
 torch.set_num_threads(2);torch.manual_seed(42);rng=random.Random(42);pools,control,n,excluded=prepare();buckets=[]
 for rr in pools:
  buck=collections.defaultdict(list)
  for i,(x,y)in enumerate(rr):buck[(len(x)-1)//32].append(i)
  buckets.append(buck)
 m=Model(d=128,hid=384,n=4,heads=4,kv=2);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);start=time.monotonic()
 tests=[('known-greeting',['Приветик!']),('novel-greeting',['Привет, как ты?']),('known-book',['Я дочитал книгу.']),('novel-book',['Давай немного поговорим про книги.']),('novel-sharing',['Я закончил рисунок, хочу поделиться.']),('context-name',['Меня зовут Арина.','Как меня зовут?']),('context-correction',['Папка синяя.','Я передумал: папка жёлтая.','Какого цвета папка теперь?'])]
 rep={'scope':'INDEPENDENT Python medium821120 CPU full curriculum; different initialization/sampling from C#, not target laptop, open responses must be read', 'steps':a.steps,'batch':8,'phases_fraction':[.2,.4,.4],'expanded_rows':n,'excluded_prefixes':excluded,'seed':42,'data_sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()for n in ['seed.jsonl','conversation-starter.jsonl','validation.jsonl']},'observations':[]}
 for step in range(a.steps+1):
  phase=0 if (step-1)*5<a.steps else 1 if(step-1)*5<a.steps*3 else 2
  if step:
   rr=pools[phase];anchor=rng.randrange(len(rr));bucket=buckets[phase][(len(rr[anchor][0])-1)//32];indices=[anchor]+[rng.choice(bucket)for _ in range(7)];ids,y=batch(rr,indices)
   warm=min(100,a.steps//10);lr=.001*(step/warm if step<=warm and warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm-1)/max(1,a.steps-warm-1))))
   for pg in opt.param_groups:pg['lr']=lr
   opt.zero_grad();z=m(ids);per=F.cross_entropy(z.reshape(-1,262),y.reshape(-1),reduction='none',ignore_index=-100).reshape(y.shape);loss=(per.sum(1)/(y>=0).sum(1)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%600 and step!=a.steps:continue
  generated=[]
  for name,questions in tests:
   history=[]
   for q in questions:history.append((q,reply(m,q,history,limit=112)))
   generated.append({'case':name,'turns':history})
  o={'step':step,'phase':phase,'elapsed_seconds':round(time.monotonic()-start,2),'control':evaluate(m,control),'generated':generated};rep['observations'].append(o);pathlib.Path(a.output).write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n');print(json.dumps(o,ensure_ascii=False),flush=True)
 return rep
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=6000);p.add_argument('--output',required=True);a=p.parse_args();run(a)
