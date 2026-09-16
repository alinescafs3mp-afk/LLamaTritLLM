using System.Diagnostics;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Real C# measurements only, never imported Python timings. Workloads are synthetic, no owner data.
public static class CpuPreparationBenchmark
{
    public static object Run(bool reverse)
    {
        var projection = new List<object>();
        foreach (int dimension in new[] { 64, 256 }) foreach(int threads in new[]{1,Math.Max(1,Math.Min(4,Environment.ProcessorCount))}.Distinct())
        {
            float[] Values(int n)=>Enumerable.Range(0,n).Select(i=>(i%37-18)*.0078125f).ToArray();
            var input=Values(dimension);var q=Values(dimension*dimension);var k=Values(dimension*dimension/2);var v=Values(k.Length);
            var gate=Values(dimension*dimension*3);var up=Values(gate.Length);
            var a=new float[dimension];var b=new float[dimension/2];var c=new float[b.Length];var d=new float[dimension*3];var e=new float[d.Length];
            var options=new ParallelOptions{MaxDegreeOfParallelism=threads};
            void Old(float[] matrix,float[] x,float[] output)
            {
                int n=x.Length;
                if(matrix.Length<65536||threads==1)for(int row=0;row<output.Length;row++)output[row]=ManagedInference.Session.Dot(matrix,row*n,x,0,n);
                else Parallel.For(0,output.Length,options,row=>output[row]=ManagedInference.Session.Dot(matrix,row*n,x,0,n));
            }
            void Calculate(bool grouped)
            {
                if(grouped){CpuProjection.Triple(q,k,v,input,a,b,c,options);CpuProjection.Pair(gate,up,input,d,e,options);}
                else{Old(q,input,a);Old(k,input,b);Old(v,input,c);Old(gate,input,d);Old(up,input,e);}
            }
            Calculate(false);var expected=new[]{a.ToArray(),b.ToArray(),c.ToArray(),d.ToArray(),e.ToArray()};Calculate(true);
            foreach(var pair in new[]{a,b,c,d,e}.Zip(expected))
                if(!pair.First.Select(BitConverter.SingleToInt32Bits).SequenceEqual(pair.Second.Select(BitConverter.SingleToInt32Bits)))throw new Exception("Projection reference mismatch.");
            foreach(bool grouped in reverse?new[]{true,false}:new[]{false,true})
            {
                for(int i=0;i<4;i++)Calculate(grouped);
                long bytes=GC.GetTotalAllocatedBytes(true);var timer=Stopwatch.StartNew();const int iterations=32;
                for(int i=0;i<iterations;i++)Calculate(grouped);
                timer.Stop();projection.Add(new{dimension,threads,grouped,iterations,ms=timer.Elapsed.TotalMilliseconds,allThreadManagedAllocatedBytes=GC.GetTotalAllocatedBytes(true)-bytes});
            }
        }
        var validation=new List<object>();
        var controls=Enumerable.Range(0,32).Select(i=>Dataset.Make("control"+i,"answer",history:[new ChatTurn("earlier","reply",0)])).ToArray();
        var candidates=Enumerable.Range(0,256).Select(i=>Dataset.Make("training"+i,"answer",history:[new ChatTurn("earlier","reply",0)])).ToArray();
        foreach(bool optimized in reverse?new[]{true,false}:new[]{false,true})
        {
            var guard=new ValidationGuard(controls,512);
            var full=controls.Select(e=>ValidationGuard.InputKey(e)).ToHashSet();var effective=controls.Select(e=>ValidationGuard.InputKey(e,512)).ToHashSet();
            bool Contains(TrainingExample row)=>optimized?guard.Contains(row):full.Contains(ValidationGuard.InputKey(row))||effective.Contains(ValidationGuard.InputKey(row,512));
            foreach(var row in candidates)if(Contains(row))throw new Exception("Synthetic split unexpectedly overlaps.");
            long bytes=GC.GetAllocatedBytesForCurrentThread(),builds=guard.KeyBuilds;var timer=Stopwatch.StartNew();const int passes=8;
            for(int i=0;i<passes;i++)foreach(var row in candidates)if(Contains(row))throw new Exception("Split result changed.");
            timer.Stop();validation.Add(new{optimized,passes,candidates=candidates.Length,ms=timer.Elapsed.TotalMilliseconds,managedAllocatedBytes=GC.GetAllocatedBytesForCurrentThread()-bytes,
                keyBuilds=optimized?guard.KeyBuilds-builds:2L*passes*candidates.Length});
        }
        return new{scope="Actual managed C# projection and split-key benchmark; warmup excluded; no required speed ratio. All-thread allocations may include runtime noise.",projection,validation};
    }
}
