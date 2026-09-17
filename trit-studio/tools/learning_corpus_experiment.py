#!/usr/bin/env python3
"""Independent Python experiment, NOT the C# trainer. This is opt-in and can take minutes.
To reproduce the recorded v19 investigation use the immutable v18 data from owner commit68727516.
New default v19 data will naturally change the result. Inputs/initialization are not the user's weights.
Autoregressive helper has a72-byte output cap and may show a replacement at a cut Unicode boundary;
this is NOT evidence that the production UTF8-safe decoder is faulty. Raw measured outputs are retained.
"""
import sys, math, json, pathlib, collections, argparse, time, copy
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from learning_reference import Model, prompt, gen
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
def encode_row(r):
 if 'text' in r: whole=[1]+[v+6 for v in r['text'].encode()]+[2];return whole[:-1],whole[1:]
 msgs=r.get('messages',[{'role':'user','content':r.get('prompt','')},{'role':'assistant','content':r.get('answer','')}])
 ids=[1]
 for m in msgs[:-1]: ids += [3 if m['role']=='user' else 4]+[v+6 for v in m['content'].encode()]+[2]
 ids += [4];start=len(ids)-1;ids += [v+6 for v in msgs[-1]['content'].encode()]+[2]
 return ids[:-1],[-100]*start+ids[start+1:]
def rows(name):return [encode_row(json.loads(l)) for l in (DATA/name).read_text().splitlines() if l.strip()]
def baseline():
 tr=rows('seed.jsonl')+rows('pretrain.jsonl'); va=rows('validation.jsonl'); counts=np.zeros((262,262),np.int64)
 for x,y in tr:
  for a,b in zip(x,y):
   if b>=0:counts[a,b]+=1
 uni=counts.sum(0);predict=counts.argmax(1);predict[counts.sum(1)==0]=uni.argmax()
 def ev(rr):
  n=sum(sum(b>=0 for b in y) for x,y in rr);correct=sum(sum(b>=0 and predict[a]==b for a,b in zip(x,y))for x,y in rr)
  return {'correct':int(correct),'total':n,'accuracy':correct/n,'unigram_accuracy':sum(sum(b==uni.argmax()for b in y)for x,y in rr)/n}
 return {'scope':'independent corpus counting only: byte-bigram trained ONLY on train+basic, evaluated teacher-forced','train':ev(tr),'control':ev(va),'train_utf8_bytes':sum(len(x) for x,y in tr)}
class Rng:
 def __init__(self,s=42): self.s=s
 def next(self,n):
  x=self.s;x^=x>>12;x^=(x<<25)&((1<<64)-1);x^=x>>27;self.s=x;return (x*2685821657736338717&((1<<64)-1))%n
def batch(rr,indices):
 T=max(len(rr[i][0])for i in indices);ids=torch.zeros((len(indices),T),dtype=torch.int64);labels=torch.full_like(ids,-100)
 for j,i in enumerate(indices):
  x,y=rr[i];ids[j,:len(x)]=torch.tensor(x);labels[j,:len(y)]=torch.tensor(y)
 return ids,labels
@torch.no_grad()
def evaluate(m,rr):
 loss=0.;right=0;total=0
 order=sorted(range(len(rr)),key=lambda i:len(rr[i][0]))
 for start in range(0,len(order),16):
  ids,y=batch(rr,order[start:start+16]);z=m(ids); mask=y>=0;n=int(mask.sum());loss+=float(F.cross_entropy(z[mask],y[mask]))*n;right+=int((z.argmax(-1)[mask]==y[mask]).sum());total+=n
 return {'loss':loss/total,'accuracy':right/total,'target_tokens':total}
def run(args):
 torch.set_num_threads(2);torch.manual_seed(42)
 if args.large: m=Model(d=256,hid=768,n=6,heads=8,kv=4)
 else: m=Model()
 tr=rows('seed.jsonl')+(rows('pretrain.jsonl')if args.basics else []);va=rows('validation.jsonl');rng=Rng();buckets=collections.defaultdict(list)
 for i,(x,y) in enumerate(tr):buckets[(len(x)-1)//32].append(i)
 opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=.01);start=time.monotonic();r={'scope':'Independent Python QAT training, not C#/CUDA. Same architecture/data/selection recipe; initialization differs from C#','parameters':sum(p.numel()for p in m.parameters()),'lr':args.lr,'baseline_added':args.basics,'batch':args.batch,'steps':args.steps,'checkpoints':[]}
 for i in range(args.steps):
  a=rng.next(len(tr));bucket=buckets[(len(tr[a][0])-1)//32];idx=[a]+[bucket[rng.next(len(bucket))]for _ in range(args.batch-1)];ids,y=batch(tr,idx)
  lr=args.lr
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad();z=m(ids);mask=y>=0;loss=F.cross_entropy(z[mask],y[mask]);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if (i+1)%args.every==0 or i==args.steps-1:
   val=evaluate(m,va);out={'step':i+1,'lr':lr,'batch_loss':float(loss.detach()),'validation':val,'generated':{p:gen(m,p)for p in ['Привет!','Как дела?','Как тебя зовут?']},'seconds':time.monotonic()-start};r['checkpoints'].append(out);print(json.dumps(out,ensure_ascii=False),flush=True);pathlib.Path(args.output).write_text(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--lr',type=float,default=.001);a.add_argument('--batch',type=int,default=16);a.add_argument('--steps',type=int,default=600);a.add_argument('--every',type=int,default=100);a.add_argument('--large',action='store_true');a.add_argument('--basics',action='store_true');a.add_argument('--data-dir',type=pathlib.Path,default=DATA);a.add_argument('--baseline',action='store_true');a.add_argument('--output',required=True);args=a.parse_args();DATA=args.data_dir
 if args.baseline:r=baseline();pathlib.Path(args.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
 else:run(args)
