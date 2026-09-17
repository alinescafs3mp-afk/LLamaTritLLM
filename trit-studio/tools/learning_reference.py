import math,json,time,pathlib,argparse,sys,random
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
# Independent port of actual TritStudio v16 QAT architecture/token conventions.
torch.set_num_threads(2)
class Model(nn.Module):
 def __init__(self,d=64,hid=192,n=2,heads=4,kv=2):
  super().__init__();self.d=d;self.hid=hid;self.n=n;self.h=heads;self.kv=kv;self.hd=d//heads;self.ps=nn.ParameterDict();self.sh=[]
  def add(name,r,c,q):
   val=torch.randn(r,c)*(.02 if name=='embedding' else 1/math.sqrt(c)) if q else torch.ones(r,c)
   self.ps[name]=nn.Parameter(val);self.sh.append((name,r,c,q))
  add('embedding',262,d,True)
  for i in range(n):
   p=f'b{i}_';add(p+'norm1',1,d,False)
   for name,rows,cols in [('q',d,d),('k',self.hd*kv,d),('v',self.hd*kv,d),('out',d,d)]:add(p+name,rows,cols,True)
   add(p+'norm2',1,d,False)
   for name,rows,cols in [('gate',hid,d),('up',hid,d),('down',d,hid)]:add(p+name,rows,cols,True)
  add('norm',1,d,False)
 def quant(self,name):
  w=self.ps[name];r=w.detach().reshape(-1,32).clone();out=torch.zeros_like(r)
  with torch.no_grad():
   for i in range(2):
    scale=r.abs().mean(1,keepdim=True).clamp_min(1e-8);plane=scale*torch.where(r.abs()>scale*.5,r.sign(),0)
    out.add_(plane)
    if i<1:r.sub_(plane)
  return w+(out.reshape_as(w)-w).detach()
 def norm(self,x,name):return x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*self.ps[name]
 def rotate(self,x,heads):
  b,t,_=x.shape;a=torch.arange(t)[:,None]/torch.pow(10000,torch.arange(self.hd//2)*2/self.hd)[None,:]
  c,s=a.cos()[None,:,None,:],a.sin()[None,:,None,:];v=x.reshape(b,t,heads,self.hd//2,2);even,odd=v[...,0],v[...,1]
  return torch.stack((even*c-odd*s,even*s+odd*c),-1).reshape(b,t,heads,self.hd)
 def forward(self,ids):
  b,t=ids.shape;e=self.quant('embedding');x=e[ids]
  for i in range(self.n):
   p=f'b{i}_';z=self.norm(x,p+'norm1');q=self.rotate(z@self.quant(p+'q').T,self.h).transpose(1,2)
   k=self.rotate(z@self.quant(p+'k').T,self.kv).repeat_interleave(self.h//self.kv,dim=2).transpose(1,2)
   v=(z@self.quant(p+'v').T).reshape(b,t,self.kv,self.hd).repeat_interleave(self.h//self.kv,dim=2).transpose(1,2)
   a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,self.d)
   x=x+a@self.quant(p+'out').T;z=self.norm(x,p+'norm2');x=x+(F.silu(z@self.quant(p+'gate').T)*(z@self.quant(p+'up').T))@self.quant(p+'down').T
  return self.norm(x,'norm')@e.T

def prompt(s):return [1,3]+[x+6 for x in s.encode()]+[2,4]
def encode(p,a):
 pre=prompt(p);ans=[x+6 for x in a.encode()]+[2];whole=pre+ans
 labels=[-100]*(len(pre)-1)+ans
 return whole[:-1],labels
@torch.no_grad()
def gen(m,p):
 ids=prompt(p);out=[];remaining=0;lo=128;hi=191
 for i in range(72):
  logits=m(torch.tensor([ids]))[0,-1];allowed=torch.zeros(262,dtype=torch.bool)
  if remaining==0:
   allowed[2]=True
   for b in [9,10,13]+list(range(32,127))+list(range(194,245)):allowed[b+6]=True
  else:allowed[lo+6:hi+7]=True
  t=int(logits.masked_fill(~allowed,-float('inf')).argmax())
  if t==2:break
  ids.append(t);b=t-6;out.append(b)
  if remaining:remaining-=1;lo=128;hi=191
  elif b>=128:
   remaining=1 if b<224 else 2 if b<240 else 3;lo=160 if b==224 else 144 if b==240 else 128;hi=159 if b==237 else 143 if b==244 else 191
 return bytes(out).decode('utf8',errors='replace')

def run(lr,steps):
 torch.manual_seed(42);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01)
 pairs=[('Привет!','Привет!'),('Как дела?','Хорошо.'),('Как тебя зовут?','Трит.'),('Спасибо.','Пожалуйста.'),('Пока!','До встречи!'),('Ты тут?','Да.')]
 rows=[encode(*p) for p in pairs];T=max(len(x) for x,y in rows);ids=torch.zeros((6,T),dtype=torch.long);labels=torch.full((6,T),-100,dtype=torch.long)
 for i,(x,y) in enumerate(rows):ids[i,:len(x)]=torch.tensor(x);labels[i,:len(y)]=torch.tensor(y)
 before=float(F.cross_entropy(m(ids).reshape(-1,262),labels.reshape(-1)).detach());report={'lr':lr,'steps':steps,'before':before,'parameter_count':sum(p.numel() for p in m.parameters()),'checkpoints':[]};start=time.monotonic()
 for i in range(steps):
  opt.zero_grad();loss=F.cross_entropy(m(ids).reshape(-1,262),labels.reshape(-1));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if (i+1)%100==0 or i==steps-1:
   answers=[gen(m,p) for p,a in pairs];n=sum(s==a for s,(_,a) in zip(answers,pairs));row={'step':i+1,'loss':float(loss.detach()),'exact':n,'answers':answers,'seconds':time.monotonic()-start};report['checkpoints'].append(row);print(json.dumps({'lr':lr,**row},ensure_ascii=False),flush=True)
   if n==len(pairs) and i+1>=300:break
 report['completed_steps']=i+1;report['elapsed']=time.monotonic()-start;report['scope']='Independent Python port, not C#; exact-fit training set, NOT generalization';return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--output',required=True);a=ap.parse_args();r=run(a.lr,a.steps);pathlib.Path(a.output).write_text(json.dumps(r,ensure_ascii=False,indent=2))
