#!/usr/bin/env python3
"""Create an EXPLICIT six-reply learned diagnostic fixture, never a default app model.
Uses independent Python/Torch training + a V1 binary writer + a separate NumPy reader/inference.
Does NOT execute C#, and six training replies are not general conversational ability.
"""
from pathlib import Path
import json,struct,hashlib,math
import numpy as np,torch
import torch.nn.functional as F
from learning_reference import Model,encode,prompt
R=Path(__file__).resolve().parents[1]
PAIRS=[('Привет!','Привет!'),('Как дела?','Хорошо.'),('Как тебя зовут?','Трит.'),('Спасибо.','Пожалуйста.'),('Пока!','До встречи!'),('Ты тут?','Да.')]
CONFIG=dict(formatVersion=1,dimension=64,hiddenDimension=192,layers=2,heads=4,kvHeads=2,context=128,planes=2,groupSize=32,threshold=.5,ropeTheta=10000,seed=42)
def pack(a,g=32,n=2):
 residual=a.copy().reshape(-1,g);result=[]
 for _ in range(n):
  s=np.zeros(len(residual),np.float32)
  for j in range(g):s+=np.abs(residual[:,j])
  s=np.maximum(s/np.float32(g),np.float32(1e-8));t=np.where(np.abs(residual)>s[:,None]*np.float32(.5),np.sign(residual),0).astype(np.int32)
  q=np.zeros((len(s),(g+4)//5),np.uint8)
  for j in range(g):q[:,j//5]+=((t[:,j]+1)*3**(j%5)).astype(np.uint8)
  result.append((s,q));residual-=s[:,None]*t.astype(np.float32)
 return result

def read(path,shapes):
 raw=memoryview(path.read_bytes());pos=0;n=raw[0];pos=1
 assert bytes(raw[pos:pos+n])==b'TRITSTUDIO_PACKED_V1';pos+=n
 n=struct.unpack_from('<i',raw,pos)[0];pos+=4;c=json.loads(bytes(raw[pos:pos+n]));pos+=n;assert c==CONFIG
 w={}
 for name,rows,cols,isq in shapes:
  if isq:
   groups=rows*cols//32;buf=np.zeros((groups,32),np.float32)
   for _ in range(2):
    s=np.frombuffer(raw[pos:pos+groups*4],'<f4');pos+=groups*4
    codes=np.frombuffer(raw[pos:pos+groups*7],np.uint8).reshape(groups,7);pos+=groups*7
    for j in range(32):buf[:,j]+=s*((codes[:,j//5].astype(np.int32)//3**(j%5))%3-1).astype(np.float32)
   w[name]=buf.reshape(rows,cols)
  else:w[name]=np.frombuffer(raw[pos:pos+rows*cols*4],'<f4').reshape(rows,cols).copy();pos+=rows*cols*4
  assert np.isfinite(w[name]).all()
 assert pos==len(raw)
 return w

class Inference:
 def __init__(self,w):self.w=w;self.pos=0;self.k=[[],[]];self.v=[[],[]]
 def norm(self,x,name):return x*(np.float32(1)/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-5)))*self.w[name][0]
 def rope(self,x,h):
  x=x.reshape(h,16).copy()
  for j in range(0,16,2):
   a=np.float32(self.pos)/np.float32(10000**(j/16));co,si=np.float32(np.cos(a)),np.float32(np.sin(a));even=x[:,j].copy();odd=x[:,j+1].copy();x[:,j]=even*co-odd*si;x[:,j+1]=even*si+odd*co
  return x
 def step(self,token):
  w=self.w;x=w['embedding'][token].copy()
  for l in range(2):
   p=f'b{l}_';z=self.norm(x,p+'norm1');q=self.rope(w[p+'q']@z,4);k=self.rope(w[p+'k']@z,2);v=(w[p+'v']@z).reshape(2,16)
   self.k[l].append(k);self.v[l].append(v);a=np.zeros((4,16),np.float32)
   for h in range(4):
    scores=np.array([q[h]@kk[h//2]/4 for kk in self.k[l]],np.float32);pr=np.exp(scores-scores.max());pr/=pr.sum()
    for i in range(len(pr)):a[h]+=pr[i]*self.v[l][i][h//2]
   x+=w[p+'out']@a.reshape(-1);z=self.norm(x,p+'norm2');gate=w[p+'gate']@z;x+=w[p+'down']@((gate/(1+np.exp(-gate)))*(w[p+'up']@z))
  self.pos+=1;return w['embedding']@self.norm(x,'norm')

def generate(w,p):
 s=Inference(w)
 for token in prompt(p):logits=s.step(token)
 out=[];remaining=0;lo=128;hi=191
 for i in range(64):
  allowed=np.zeros(262,bool)
  if remaining==0:
   allowed[2]=True
   for b in [9,10,13]+list(range(32,127))+list(range(194,245)):allowed[b+6]=True
  else:allowed[lo+6:hi+7]=True
  token=int(np.argmax(np.where(allowed,logits,-np.inf)))
  if token==2:break
  out.append(token-6);b=token-6
  if remaining:remaining-=1;lo=128;hi=191
  elif b>=128:
   remaining=1 if b<224 else 2 if b<240 else 3;lo=160 if b==224 else 144 if b==240 else 128;hi=159 if b==237 else 143 if b==244 else 191
  logits=s.step(token)
 return bytes(out).decode('utf-8')

def run():
 torch.set_num_threads(2);torch.manual_seed(42);m=Model();o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 rows=[encode(*x)for x in PAIRS];T=max(len(x)for x,y in rows);ids=torch.zeros((6,T),dtype=torch.int64);targets=torch.full((6,T),-100,dtype=torch.int64)
 for i,(x,y)in enumerate(rows):ids[i,:len(x)]=torch.tensor(x);targets[i,:len(y)]=torch.tensor(y)
 before=float(F.cross_entropy(m(ids).reshape(-1,262),targets.reshape(-1)).detach())
 for _ in range(300):
  o.zero_grad();loss=F.cross_entropy(m(ids).reshape(-1,262),targets.reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 after=float(F.cross_entropy(m(ids).reshape(-1,262),targets.reshape(-1)).detach())
 rawconfig=json.dumps(CONFIG,separators=(',',':')).encode();magic=b'TRITSTUDIO_PACKED_V1';payload=bytearray(bytes([len(magic)])+magic+struct.pack('<i',len(rawconfig))+rawconfig)
 for name,r,c,isq in m.sh:
  a=m.ps[name].detach().numpy()
  if isq:
   for scale,trits in pack(a):payload+=scale.astype('<f4').tobytes()+trits.tobytes()
  else:payload+=a.astype('<f4').tobytes()
 out=R/'tests/fixtures/learning-reference.tritmodel';out.write_bytes(payload)
 weights=read(out,m.sh);answers=[generate(weights,p)for p,_ in PAIRS];assert answers==[a for _,a in PAIRS],answers
 report=dict(scope=__doc__,status='passed-independent-python',parameter_count=sum(p.numel()for p in m.parameters()),steps=300,learning_rate=.001,
  training_loss_before=before,training_loss_after=after,file_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
  replies=[dict(prompt=p,expected=a,generated=s)for (p,a),s in zip(PAIRS,answers)],CSharp='NOT_RUN',CUDA='NOT_RUN',generalization='NOT_TESTED; fixture only memorizes six training replies')
 (R/'reports/learning-fixture-v17.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
