using System.Diagnostics;
using System.Numerics;
using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Isolated initial-copy and CPU elementwise measurements; never opens owner workspaces.
public static class InitializationBenchmark
{
    public static object Run(bool cuda, bool reverse)
    {
        var config = ModelConfig.Small;
        var initial = WeightSet.Initialize(config);
        var resources = new ResourceOptions { Threads=2, MemoryMiB=4096, BatchSize=2, SequenceLength=64, PreferCuda=cuda };
        using var session = new TrainingSession(initial, resources, new TrainingOptions());
        if (cuda && !session.DeviceName.StartsWith("CUDA:", StringComparison.Ordinal))
            throw new InvalidOperationException("CUDA initial-copy benchmark may not silently use CPU.");
        var publication = new List<object>();
        foreach (bool reuse in reverse ? new[]{true,false} : new[]{false,true})
        {
            _ = Capture(); Sync();long allocation=GC.GetAllocatedBytesForCurrentThread();var watch=Stopwatch.StartNew();
            WeightSet? result=null; const int iterations=5;
            for(int i=0;i<iterations;i++) result=Capture();
            Sync();watch.Stop();allocation=GC.GetAllocatedBytesForCurrentThread()-allocation;
            foreach(var shape in WeightLayout.For(config))
                if(!initial.Values[shape.Name].SequenceEqual(result!.Values[shape.Name]))throw new Exception("Initial publication parity failed.");
            publication.Add(new { mode=reuse?"immutable-initial":"native-copy-reference", iterations,elapsedMs=watch.Elapsed.TotalMilliseconds,
                managedThreadAllocatedBytes=allocation, modeledReturnedFloatBytes=reuse?0:iterations*4*config.ParameterCount });
            WeightSet Capture()=>reuse?session.CapturePublicationMaster():session.Model.CopyMaster();
        }
        var elementwise=new List<object>();
        foreach(int length in new[]{16,64,128,256,512})
        {
            var x=Enumerable.Range(0,length).Select(i=>MathF.Sin(i+.3f)).ToArray();var gamma=Enumerable.Range(0,length).Select(i=>1+MathF.Cos(i)*.1f).ToArray();
            var expected=new float[length];ScalarNorm(x,gamma,expected);var actual=new float[length];CpuElementwise.RmsNorm(x,gamma,actual);
            for(int i=0;i<length;i++)if(MathF.Abs(expected[i]-actual[i])>2e-6f*MathF.Max(1,MathF.Abs(expected[i])))throw new Exception("Elementwise parity failed.");
            foreach(bool vector in reverse?new[]{true,false}:new[]{false,true})
            {
                for(int i=0;i<20;i++)Operation();long allocation=GC.GetAllocatedBytesForCurrentThread();var timer=Stopwatch.StartNew();
                const int repeats=1000;for(int i=0;i<repeats;i++)Operation();timer.Stop();allocation=GC.GetAllocatedBytesForCurrentThread()-allocation;
                elementwise.Add(new{length,mode=vector?"portable-vector":"scalar-reference",repeats,elapsedMs=timer.Elapsed.TotalMilliseconds,allocatedBytes=allocation});
                void Operation()
                {
                    if(vector){CpuElementwise.RmsNorm(x,gamma,actual);CpuElementwise.AddInPlace(actual,x);}
                    else {ScalarNorm(x,gamma,actual);for(int i=0;i<length;i++)actual[i]+=x[i];}
                }
            }
        }
        return new { publication,elementwise,vectorWidth=Vector<float>.Count,hardwareAccelerated=Vector.IsHardwareAccelerated,
            note="Initial reuse avoids a weight copy only before any update/restore. Not a disk-write, peak-VRAM or full-training timing. CPU scalar paths may win at small sizes." };
        void Sync(){if(cuda)torch.cuda.synchronize();}
    }
    private static void ScalarNorm(float[] x,float[] gamma,float[] result)
    {
        float inv=1/MathF.Sqrt(ManagedInference.Session.Dot(x,0,x,0,x.Length)/x.Length+1e-5f);
        for(int i=0;i<x.Length;i++)result[i]=x[i]*inv*gamma[i];
    }
}
