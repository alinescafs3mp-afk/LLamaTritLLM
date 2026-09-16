using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Isolated synthetic workloads. Never opens a user workspace or updates the real corpus.
// No timing assertion: driver warmup, CPU governor and machine load vary.
public static class TrainerBenchmark
{
    public static int Run(bool requireCuda, string? output, bool reverse = false)
    {
        try
        {
            if(output is not null) JsonData.AtomicWrite(Path.GetFullPath(output),new{status="running",startedAtUtc=DateTimeOffset.UtcNow});
            torch.set_num_interop_threads(1);
            if(requireCuda && !torch.cuda.is_available())throw new InvalidOperationException("CUDA benchmark requested, but unavailable. No CPU result is substituted.");
            var config = new ModelConfig{Dimension=32,HiddenDimension=64,Layers=1,Heads=4,KvHeads=2,GroupSize=8,Context=128};
            var options = new TrainingOptions{LearningRate=0.001};
            var rows = Enumerable.Range(0,64).Select(i=>Dataset.Make("case"+i+new string('x',i<48?8:85),"answer"));
            var corpus = Dataset.EncodeAll(rows.ToArray(),128,1);
            var results = new List<object>();
            foreach(int variant in reverse ? new[]{2,1,0} : new[]{0,1,2})
            {
                var resources = new ResourceOptions{Threads=2,MemoryMiB=4096,BatchSize=8,SequenceLength=128,PreferCuda=requireCuda,UseSdpa=variant>0,BucketByLength=variant>0,ProjectOnlyTargets=variant==2};
                using var session = new TrainingSession(WeightSet.Initialize(config),resources,options);
                for(int i=0;i<3;i++)session.TrainStep(corpus,options.LearningRate,CancellationToken.None);
                Sync(requireCuda); var timer=Stopwatch.StartNew(); long targets=0,positions=0,inputs=0,outputPositions=0;
                const int measuredSteps=16;
                for(int i=0;i<measuredSteps;i++)
                {
                    var result=session.TrainStep(corpus,options.LearningRate,CancellationToken.None);
                    targets+=result.TargetTokens; positions+=session.LastStepPerformance!.PaddedPositions; inputs+=session.LastStepPerformance.InputTokens;outputPositions+=session.LastStepPerformance.OutputPositions;
                }
                Sync(requireCuda); timer.Stop();
                var evalTimer=Stopwatch.StartNew(); double loss=session.Evaluate(corpus); Sync(requireCuda);evalTimer.Stop();
                long builds=session.Model.QuantizationBuilds;
                var cacheTimer=Stopwatch.StartNew(); double cached=session.Evaluate(corpus);Sync(requireCuda);cacheTimer.Stop();
                if(!double.IsFinite(loss)||cached!=loss||session.Model.QuantizationBuilds!=builds)throw new Exception("Benchmark evaluation invariant failed.");
                using var process=Process.GetCurrentProcess();
                results.Add(new{mode=variant==2?"sdpa+buckets+selected-output":variant==1?"sdpa+buckets+full-output":"explicit+uniform+full-output",device=session.DeviceName,steps=measuredSteps,
                    synchronizedTrainingMs=timer.Elapsed.TotalMilliseconds,targetTokens=targets,targetTokensPerSecond=targets/Math.Max(timer.Elapsed.TotalSeconds,1e-9),
                    paddingFraction=1-(double)inputs/positions,validationMs=evalTimer.Elapsed.TotalMilliseconds,cachedValidationMs=cacheTimer.Elapsed.TotalMilliseconds,
                    outputProjectionPositions=outputPositions,validationLoss=loss,hostRssMiB=process.WorkingSet64/1048576,attention=session.Model.AttentionBackend});
            }
            var report=new{status="passed",scope="Actual C# native smoke benchmark on an isolated synthetic workload, not a fluency evaluation",createdAtUtc=DateTimeOffset.UtcNow,
                cudaRequired=requireCuda,reverseOrder=reverse,parameterCount=config.ParameterCount,results,managedInference=ManagedBufferCheck(),publicationPacking=PackingCheck(reverse),validationPreparation=ValidationPreparationCheck(reverse),checkpointCorpus=CheckpointCorpusCheck(reverse),textPreparation=EncodingBenchmark.Run(reverse),onlineReplayPreparation=ReplayPreparationBenchmark.Run(reverse),cpuPreparation=CpuPreparationBenchmark.Run(reverse),attentionAndJournal=AttentionJournalBenchmark.Run(reverse),initializationAndElementwise=InitializationBenchmark.Run(requireCuda,reverse),
                caveats=new[]{"Host RSS is not CUDA VRAM; allocator pools are retained across trials.","Bucket sampling retains marginals but changes individual batches, so losses are not a speed-equivalence proof.","No mandatory speedup ratio. Repeat both orders on the target machine before selecting a backend."}};
            string json=JsonSerializer.Serialize(report,new JsonSerializerOptions(JsonData.Options){WriteIndented=true});
            if(output is not null)JsonData.AtomicWrite(Path.GetFullPath(output),report);
            Console.WriteLine(json);return 0;
        }
        catch(Exception error)
        {
            Console.Error.WriteLine("BENCHMARK FAILED: "+error);
            if(output is not null) try{JsonData.AtomicWrite(Path.GetFullPath(output),new{status="failed",error=error.Message,createdAtUtc=DateTimeOffset.UtcNow});}catch(Exception writeError){Console.Error.WriteLine(writeError.Message);}
            return 1;
        }
    }
    private static object CheckpointCorpusCheck(bool reverse)
    {
        string root=Path.Combine(Path.GetTempPath(),"trit-cache-bench-"+Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            var corpus=Enumerable.Range(0,512).Select(i=>Dataset.Make("record"+i+new string('x',64),"answer"+i)).ToArray();
            var results=new List<object>(); string? expected=null;
            foreach(int budget in reverse?new[]{4*1024*1024,0}:new[]{0,4*1024*1024})
            {
                var cache=new CheckpointJsonCache(budget); cache.WriteNew(Path.Combine(root,"warm"+budget),corpus);
                long allocated=GC.GetAllocatedBytesForCurrentThread();var watch=Stopwatch.StartNew();
                for(int i=0;i<8;i++)
                {
                    string path=Path.Combine(root,$"{budget}-{i}.json");string hash=cache.WriteNew(path,corpus);
                    expected??=hash;if(hash!=expected)throw new Exception("Checkpoint corpus digest changed across cache variants.");
                }
                watch.Stop();allocated=GC.GetAllocatedBytesForCurrentThread()-allocated;
                // Disk verification is outside the timing; production restore still verifies full checkpoints.
                for(int i=0;i<8;i++)if(ModelFiles.Hash(Path.Combine(root,$"{budget}-{i}.json"))!=expected)throw new Exception("Cache wrote incorrect bytes.");
                results.Add(new{budget,ms=watch.Elapsed.TotalMilliseconds,allocatedBytes=allocated,hits=cache.Hits,serializations=cache.Serializations,retainedBytes=cache.RetainedBytes});
            }
            return new{scope="Actual C# JSON snapshot writes including fsync; warmup excluded; no speed-ratio assertion",results};
        }
        finally{Directory.Delete(root,true);}
    }
    private static object ManagedBufferCheck()
    {
        var c=new ModelConfig{Dimension=32,HiddenDimension=64,Layers=1,Heads=4,KvHeads=2,GroupSize=8,Context=64};
        var model=new ManagedInference(TernaryQuantizer.Quantize(WeightSet.Initialize(c)),0,1);
        model.NewSession().Step(1); // Build shared RoPE/JIT before measurements.
        var rows=new List<object>();
        foreach(bool buffered in new[]{false,true})
        {
            long begin=GC.GetAllocatedBytesForCurrentThread(); var watch=Stopwatch.StartNew();
            for(int run=0;run<8;run++)
            {
                var session=model.NewSession();float[]? output=buffered?new float[ByteTokenizer.VocabularySize]:null;
                for(int i=0;i<32;i++)session.Step(6+i,default,true,output);
            }
            watch.Stop();long bytes=GC.GetAllocatedBytesForCurrentThread()-begin;
            rows.Add(new{buffered,steps=256,ms=watch.Elapsed.TotalMilliseconds,managedAllocatedBytes=bytes});
        }
        return new{scope="Actual managed CPU single-thread fixture, allocation traffic is not RSS or VRAM",rows};
    }
    private static object PackingCheck(bool reverse)
    {
        // Includes managed allocations on worker threads; unrelated runtime allocations can contribute noise.
        var input=Enumerable.Range(0,262144).Select(i=>MathF.Sin(i*.013f)*.2f).ToArray();
        var expected=PackReferenceV5.Pack(input,32,3,.5f,1);
        static void Verify(PackedPlane[] a,PackedPlane[] b)
        {
            for(int p=0;p<a.Length;p++)if(!a[p].Scales.SequenceEqual(b[p].Scales)||!a[p].Trits.SequenceEqual(b[p].Trits))
                throw new Exception("Packing benchmark changed the v1 byte format.");
        }
        var rows=new List<object>();
        foreach(int threads in new[]{1,Math.Min(4,Math.Max(1,Environment.ProcessorCount))}.Distinct())
        {
            foreach(bool legacy in reverse?new[]{false,true}:new[]{true,false})
            {
                PackedPlane[] Run()=>legacy?PackReferenceV5.Pack(input,32,3,.5f,threads):TernaryQuantizer.Pack(input,32,3,.5f,threads);
                Verify(expected,Run());long begin=GC.GetTotalAllocatedBytes(precise:true);var timer=Stopwatch.StartNew();
                PackedPlane[]? last=null;const int repetitions=3;
                for(int i=0;i<repetitions;i++)last=Run();
                timer.Stop();long allocated=GC.GetTotalAllocatedBytes(precise:true)-begin;
                Verify(expected,last!);var unpackTimer=Stopwatch.StartNew();
                var decoded=TernaryQuantizer.Unpack(last!,input.Length,32,threads);unpackTimer.Stop();
                if(decoded.Any(x=>!float.IsFinite(x)))throw new Exception("Unpacked nonfinite weights.");
                rows.Add(new{mode=legacy?"v5-full-residual-reference":"v6-group-local",threads,repetitions,
                    packMs=timer.Elapsed.TotalMilliseconds,managedAllocatedBytes=allocated,unpackMs=unpackTimer.Elapsed.TotalMilliseconds});
            }
        }
        return new{scope="Actual C# CPU pack/unpack microbenchmark; exact bytes must match. Allocation traffic is not peak RSS/VRAM.",
            weights=input.Length,group=32,planes=3,removedFullResidualBytes=4L*input.Length,rows};
    }
    private static object ValidationPreparationCheck(bool reverse)
    {
        var examples=Dataset.EncodeAll(Enumerable.Range(0,128).Select(i=>Dataset.Make("query "+i+new string('x',i%64),"answer")).ToArray(),128,1);
        var rows=new List<object>();
        foreach(bool cached in reverse?new[]{true,false}:new[]{false,true})
        {
            var plan=new EvaluationBatchCache(examples,8,128,cached?EvaluationBatchCache.DefaultBudgetBytes:0);
            for(int i=0;i<plan.Count;i++)_=plan.Get(i); // JIT/preparation warmup is excluded.
            long start=GC.GetAllocatedBytesForCurrentThread(); var watch=Stopwatch.StartNew();long targetCount=0;
            for(int repeat=0;repeat<16;repeat++)for(int i=0;i<plan.Count;i++)targetCount+=plan.Get(i).Targets.Length;
            watch.Stop();long allocated=GC.GetAllocatedBytesForCurrentThread()-start;
            if(targetCount!=16L*examples.Sum(x=>x.Labels.Count(y=>y!=-100)))throw new Exception("Validation cache changed target coverage.");
            rows.Add(new{cached,passes=16,ms=watch.Elapsed.TotalMilliseconds,managedAllocatedBytes=allocated,retainedPayloadBytes=plan.RetainedPayloadBytes,targetCount});
        }
        return new{scope="Actual managed CPU preparation only, NOT network forward speed or GPU timing",rows};
    }
    private static void Sync(bool cuda){if(cuda)torch.cuda.synchronize();}
}
