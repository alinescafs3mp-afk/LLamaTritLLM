#!/usr/bin/env python3
"""Independent Python mathematical checks. NOT C# code, UI, persistence or CUDA execution."""
from pathlib import Path
from collections import deque
import hashlib,json,random
import torch
ROOT=Path(__file__).resolve().parents[1]

def run():
    torch.set_num_threads(2)
    rng=random.Random(18); cases=0
    for rows in [1,7,64,65,257]:
        for seed in range(4):
            torch.manual_seed(seed)
            logits=torch.randn(rows,262,requires_grad=True)
            labels=torch.randint(0,262,(rows,)); mask=torch.rand(rows)>.35
            if not mask.any(): mask[0]=True
            labels=labels.masked_fill(~mask,-100)
            loss=torch.nn.functional.cross_entropy(logits,labels)
            grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
            before=torch.get_rng_state().clone()
            selected=logits[mask]; targets=labels[mask]
            # Same math as target-selected model output, no masked tokens in accuracy denominator.
            n=targets.numel(); correct=int((selected.detach().argmax(-1)==targets).sum())
            oracle=sum(int(torch.argmax(logits[i]).item())==int(labels[i]) for i in range(rows) if int(labels[i])!=-100)
            assert correct==oracle and n==int(mask.sum())
            assert torch.equal(before,torch.get_rng_state())
            other=torch.nn.functional.cross_entropy(selected,targets)
            assert torch.allclose(loss,other,atol=1e-7)
            assert torch.equal(grad,torch.autograd.grad(other,logits)[0])
            # Validation must weight partial batches by number of supervised targets, not batch means.
            total=hits=0; weighted=0.
            for start in range(0,n,5):
                x=selected[start:start+5]; y=targets[start:start+5]
                hits+=int((x.argmax(-1)==y).sum()); total+=len(y)
                weighted+=float(torch.nn.functional.cross_entropy(x,y).detach())*len(y)
            assert hits==correct and total==n and abs(weighted/total-float(other.detach()))<2e-6
            cases+=1
    # Sliding token weighted observations, independent brute force vs incremental FIFO.
    records=deque();correct=total=0;window_cases=0
    for step in range(1,1001):
        n=rng.randint(1,64*512); k=rng.randint(0,n)
        if len(records)==32:
            old=records.popleft();correct-=old[0];total-=old[1]
        records.append((k,n));correct+=k;total+=n
        assert correct==sum(a for a,b in records) and total==sum(b for a,b in records)
        assert 0<=100*correct/total<=100
        window_cases+=1
    baseline=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    # Structural ownership checks, not simulation presented as runtime behavior.
    launch=(ROOT/'src/TritStudio.App/LaunchEditor.cs').read_text()
    assert 'if (_runEditorInitialized) return' in launch and 'LaunchDraftStore.Write' in launch
    assert 'CaptureRunDraft(); SaveRunDraft();' in (ROOT/'src/TritStudio.App/MainWindow.cs').read_text()
    constants=(ROOT/'src/TritStudio.Core/ModelConfig.cs').read_text();assert 'MaxBatchSize = 64' in constants
    result={'status':'passed','scope':'Independent PyTorch CPU math and source invariants; NOT execution of C# / UI / worker / CUDA',
            'masked_accuracy_loss_gradient_cases':cases,'weighted_window_cases':window_cases,'corpus_total':sum(baseline['counts'].values()),
            'rng_unchanged_by_accuracy':True,'extra_forward_for_accuracy':False,'csharp_tests':'NOT_RUN','cuda':'NOT_RUN'}
    (ROOT/'reports/audit18-reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
