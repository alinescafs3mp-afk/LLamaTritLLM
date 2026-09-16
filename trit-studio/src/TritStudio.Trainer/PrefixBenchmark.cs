using System.Diagnostics;
using TritStudio.Core;
namespace TritStudio.Trainer;

// CPU-only prefill comparison, including cache allocation in BOTH paths. No user model touched.
public static class PrefixBenchmark
{
    public static object Run(bool reverse)
    {
        var rows=new List<object>();
        foreach(int layers in new[]{1,3})
        foreach(int length in new[]{16,128})
        {
            var cfg=new ModelConfig{Dimension=64,HiddenDimension=128,Layers=layers,Heads=4,KvHeads=2,Context=256,GroupSize=32};
            var model=new ManagedInference(TernaryQuantizer.Quantize(WeightSet.Initialize(cfg)),0,1);
            var ids=Enumerable.Range(0,length).Select(i=>i==0?ByteTokenizer.Bos:ByteTokenizer.Offset+(i*17)%256).ToArray();
            var expected=Prepare(false);var actual=Prepare(true);
            for(int i=0;i<expected.Logits.Length;i++)if(!float.IsFinite(actual.Logits[i])||MathF.Abs(expected.Logits[i]-actual.Logits[i])>2e-5f)
                throw new Exception("Prefix preparation output parity failed.");
            if(actual.Skips!=length-1||expected.Skips!=0)throw new Exception("Unexpected skipped prefix-block count.");
            foreach(bool fast in reverse?new[]{true,false}:new[]{false,true})
            {
                for(int i=0;i<2;i++)_ = Prepare(fast);
                const int iterations=4;long allocation=GC.GetAllocatedBytesForCurrentThread();var timer=Stopwatch.StartNew();
                for(int i=0;i<iterations;i++)_ = Prepare(fast);
                timer.Stop();allocation=GC.GetAllocatedBytesForCurrentThread()-allocation;
                rows.Add(new{layers,prefixTokens=length,mode=fast?"last-block-output-elision":"audit15-reference",iterations,
                    elapsedMs=timer.Elapsed.TotalMilliseconds,allocatedBytes=allocation,skipsPerResponse=fast?length-1:0});
            }
            (float[] Logits,int Skips) Prepare(bool fast)
            {
                var session=model.NewSession(optimizePrefix:fast);var output=new float[ByteTokenizer.VocabularySize];
                for(int i=0;i<ids.Length;i++)session.Step(ids[i],computeLogits:i==ids.Length-1,logitsBuffer:output);
                return(output,session.PrefixFinalBlockSkips);
            }
        }
        return new{rows,note="CPU-only prompt prefill; retained last-block K/V, unchanged previous layers. Not CUDA throughput or fewer network parameters. No speed ratio gate."};
    }
}
