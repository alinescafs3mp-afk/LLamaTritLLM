#!/usr/bin/env python3
"""Independent Python checks of v8 retained optimization semantics. NOT execution of C# or a C# speed benchmark."""
from __future__ import annotations
import collections
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from validate_dataset import full_length

ROOT = Path(__file__).resolve().parents[1]

def attention_checks():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(142)
    rows = []
    for length, heads, kv_heads, head_dim in [(1,2,1,8),(7,4,2,8),(31,4,1,16),(96,4,2,16)]:
        originals=[torch.randn(2, heads, length, head_dim,generator=g),
                   torch.randn(2, kv_heads, length, head_dim,generator=g),
                   torch.randn(2, kv_heads, length, head_dim,generator=g)]
        gradients=[];outputs=[]
        for sdpa in (False,True):
            q,k,v=[x.clone().requires_grad_() for x in originals]
            keys=k.repeat_interleave(heads//kv_heads,dim=1);values=v.repeat_interleave(heads//kv_heads,dim=1)
            if sdpa: output=F.scaled_dot_product_attention(q,keys,values,dropout_p=0.0,is_causal=True)
            else:
                scores=q@keys.transpose(-1,-2)/math.sqrt(head_dim)
                scores=scores.masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),float('-inf'))
                output=scores.softmax(-1)@values
            output.square().mean().backward()
            outputs.append(output.detach());gradients.append([x.grad.clone() for x in (q,k,v)])
        error=float((outputs[0]-outputs[1]).abs().max())
        grad_error=max(float((a-b).abs().max()) for a,b in zip(*gradients))
        assert error<2e-5 and grad_error<2e-5,(error,grad_error)
        rows.append(dict(sequence_length=length,heads=heads,kv_heads=kv_heads,head_dimension=head_dim,forward_max_abs_error=error,gradient_max_abs_error=grad_error))
    return rows

class Sampler:
    def __init__(self,state): self.state=state or 0x9E3779B97F4A7C15
    def next(self,size):
        mask=(1<<64)-1;x=self.state;x^=x>>12;x^=(x<<25)&mask;x^=x>>27;self.state=x&mask
        return ((self.state*2685821657736338717)&mask)%size

def shape_checks():
    records=[json.loads(x) for x in (ROOT/'data/seed.jsonl').read_text().splitlines() if x.strip()]
    lengths=[full_length(row) for row in records]
    buckets=collections.defaultdict(list)
    for i,length in enumerate(lengths): buckets[(length-1)//32].append(i)
    def sample(bucketed):
        rng=Sampler(42);batch=8;steps=10000;inputs=positions=attention=0
        counts=collections.Counter()
        for _ in range(steps):
            anchor=rng.next(len(lengths));selected=[anchor]
            group=buckets[(lengths[anchor]-1)//32] if bucketed else None
            selected += [(group[rng.next(len(group))] if group else rng.next(len(lengths))) for _ in range(batch-1)]
            used=[lengths[i] for i in selected];size=max(used)
            counts.update(selected);inputs+=sum(used);positions+=batch*size;attention+=batch*size*size
        return dict(steps=steps,batch_size=batch,padding_fraction=1-inputs/positions,padded_token_positions=positions,
                    dense_attention_shape_positions=attention,input_positions=inputs),counts
    flat,_=sample(False);bucket,counts=sample(True)
    assert bucket['padded_token_positions']<flat['padded_token_positions']
    # Analytic marginal: P(bucket)*1/N_bucket = 1/N. Do not sample buckets uniformly.
    marginal_error=max(abs((len(items)/len(lengths))/len(items)-1/len(lengths)) for items in buckets.values())
    assert marginal_error<1e-15
    return dict(scope='Python shape-count simulation on real bundled lengths, not measured C# time',corpus_rows=len(lengths),
                uniform=flat,bucketed=bucket,marginal_probability_max_error=marginal_error,
                saved_padded_positions_fraction=1-bucket['padded_token_positions']/flat['padded_token_positions'],
                saved_dense_attention_positions_fraction=1-bucket['dense_attention_shape_positions']/flat['dense_attention_shape_positions'])


def quantize_value(w,planes,group,threshold,inplace):
    r=w.detach().reshape(-1,group).clone() if inplace else w.detach().reshape(-1,group)
    q=torch.zeros_like(r)
    with torch.no_grad():
        for _ in range(planes):
            scale=r.abs().mean(1,keepdim=True).clamp_min(1e-8)
            t=torch.where(r.abs()>scale*threshold,r.sign(),torch.zeros_like(r))
            plane=scale*t
            if inplace: q.add_(plane);r.sub_(plane)
            else: q=q+plane;r=r-plane
    return w+(q.reshape(w.shape)-w).detach()

def quantizer_checks():
    generator=torch.Generator().manual_seed(251);results=[]
    for planes in (1,2,3):
        for group in (8,32):
            for threshold in (0.0,0.5,1.0):
                source=torch.randn(64,32,generator=generator)*0.13;source[0]=0
                mask=torch.randn(source.shape,generator=generator)
                values=[];grads=[]
                for inplace in (False,True):
                    w=source.clone().requires_grad_();q=quantize_value(w,planes,group,threshold,inplace)
                    (q*mask).sum().backward();values.append(q.detach());grads.append(w.grad)
                    assert torch.equal(source,w.detach()),'In-place quantizer changed master weights'
                    assert torch.equal(w.grad,mask),'STE ceased to be identity'
                error=float((values[0]-values[1]).abs().max());grad=float((grads[0]-grads[1]).abs().max())
                assert error==0 and grad==0
                results.append(dict(planes=planes,group=group,threshold=threshold,value_max_error=error,gradient_max_error=grad))
    return results

def projection_checks():
    generator=torch.Generator().manual_seed(505);rows=[]
    for length,heads,kvheads,planes,sdpa in [(7,2,1,1,False),(13,4,2,2,True),(31,4,1,3,False),(48,4,2,2,True)]:
        batch=2;dim=32;hidden=64;hd=dim//heads;kv=kvheads*hd
        shapes={'embedding':(262,dim),'n1':(1,dim),'n2':(1,dim),'nf':(1,dim),'q':(dim,dim),'k':(kv,dim),
                'v':(kv,dim),'o':(dim,dim),'gate':(hidden,dim),'up':(hidden,dim),'down':(dim,hidden)}
        source={k:torch.ones(shape) if k in ('n1','n2','nf') else torch.randn(shape,generator=generator)*0.10 for k,shape in shapes.items()}
        ids=torch.randint(6,262,(batch,length),generator=generator)
        positions=torch.tensor([2,length-2,length-1,length+3,2*length-2])
        labels=torch.randint(6,262,(len(positions),),generator=generator)
        gradients=[];values=[];losses=[]
        for selected in (False,True):
            params={k:v.clone().requires_grad_() for k,v in source.items()}
            def quant(k):return quantize_value(params[k],planes,8,.5,selected)
            def norm(x,k):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*params[k]
            def linear(x,k):return x@quant(k).T
            def rotate(z,h):
                pair=z.reshape(batch,length,h,hd//2,2)
                angles=torch.arange(length)[:,None]/10000**(torch.arange(hd//2)*2/hd)[None,:]
                co=angles.cos()[None,:,None,:];si=angles.sin()[None,:,None,:]
                a,b=pair[...,0],pair[...,1]
                return torch.stack((a*co-b*si,a*si+b*co),-1).reshape(batch,length,h,hd)
            embedding=quant('embedding');x=embedding[ids];z=norm(x,'n1')
            q=rotate(linear(z,'q'),heads).transpose(1,2)
            k=rotate(linear(z,'k'),kvheads).repeat_interleave(heads//kvheads,2).transpose(1,2)
            v=linear(z,'v').reshape(batch,length,kvheads,hd).repeat_interleave(heads//kvheads,2).transpose(1,2)
            if sdpa:attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True)
            else:
                scores=(q@k.transpose(-1,-2)/math.sqrt(hd)).masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),float('-inf'))
                attn=scores.softmax(-1)@v
            x=x+linear(attn.transpose(1,2).reshape(batch,length,dim),'o');z=norm(x,'n2');gate=linear(z,'gate')
            x=x+linear(gate*gate.sigmoid()*linear(z,'up'),'down')
            if selected:x=x.reshape(batch*length,dim).index_select(0,positions)
            logits=norm(x,'nf')@embedding.T
            if not selected:logits=logits.reshape(batch*length,262).index_select(0,positions)
            loss=F.cross_entropy(logits,labels);loss.backward()
            assert all(torch.equal(source[k],v.detach()) for k,v in params.items())
            assert all(torch.isfinite(v.grad).all() for v in params.values())
            values.append(logits.detach());losses.append(float(loss.detach()));gradients.append([v.grad for v in params.values()])
        error=float((values[0]-values[1]).abs().max());gradient=max(float((a-b).abs().max()) for a,b in zip(*gradients))
        assert error<2e-5 and gradient<2e-5 and abs(losses[0]-losses[1])<2e-5
        rows.append(dict(length=length,heads=heads,kv_heads=kvheads,planes=planes,sdpa=sdpa,targets=len(positions),
                         output_max_abs_error=error,gradient_max_abs_error=gradient,loss_abs_error=abs(losses[0]-losses[1])))
    return rows

def projection_shapes():
    records=[json.loads(x) for x in (ROOT/'data/seed.jsonl').read_text().splitlines() if x.strip()]
    inputs=sum(full_length(row) for row in records)
    targets=sum(full_length(row) if 'text' in row else 1+len((row['messages'][-1]['content'] if 'messages' in row else row['answer']).encode()) for row in records)
    assert 0<targets<=inputs
    return dict(scope='One unpadded pass over the corpus, output projection positions only; attention is NOT skipped',
                full_positions=inputs,selected_positions=targets,removed_positions=inputs-targets,
                reduction_fraction=1-targets/inputs,not_a_wall_clock_speedup=True)

if __name__=='__main__':
    report=dict(status='passed',scope='Independent Python algorithm/numerical specification; C# build, native C# execution and target speedup NOT RUN',
                torch_version=torch.__version__,attention=attention_checks(),batching=shape_checks(),
                quantizer=quantizer_checks(),projection=projection_checks(),projection_corpus=projection_shapes(),
                csharp_benchmark='Defined: TritStudio.Trainer --benchmark [--cuda] [--output FILE]. Not run in this environment.',
                caveats=['Bucketed batches alter correlations, not per-example marginal sampling. Fitting quality must still be measured.',
                         'The shape-count reduction is not a measured runtime speedup.',
                         'SDPA chooses an available native implementation; this report does not certify FlashAttention on the laptop.'])
    (ROOT/'reports/performance-reference-v14.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
