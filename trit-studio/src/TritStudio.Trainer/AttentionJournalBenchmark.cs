using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Text.Json;
using TritStudio.Core;
namespace TritStudio.Trainer;

public static class AttentionJournalBenchmark
{
    public static object Run(bool reverse)
    {
        var attention=new List<object>();var journal=new List<object>();
        foreach(int positions in new[]{32,256,1024}) foreach(int width in new[]{16,33,64})
        {
            const int heads=4,kvHeads=2;int kvDim=kvHeads*width;
            float[] Data(int n)=>Enumerable.Range(0,n).Select(i=>(i%47-23)*.03125f).ToArray();
            var q=Data(heads*width);var k=Data(positions*kvDim);var v=Data(k.Length);var scores=new float[positions];var result=new float[q.Length];
            void Calculate(bool optimized)
            {if(optimized)CpuAttention.Compute(q,k,v,scores,result,positions,heads,kvHeads,width);else LegacyAttention(q,k,v,scores,result,positions,heads,kvHeads,width);}
            Calculate(false);var expected=result.ToArray();Calculate(true);
            float maxError=expected.Zip(result,(a,b)=>MathF.Abs(a-b)).Max();
            if(!float.IsFinite(maxError)||maxError>2e-5f)throw new Exception("CPU attention benchmark parity failed.");
            foreach(bool optimized in reverse?new[]{true,false}:new[]{false,true})
            {
                for(int i=0;i<4;i++)Calculate(optimized);
                long allocated=GC.GetAllocatedBytesForCurrentThread();var watch=Stopwatch.StartNew();const int iterations=24;
                for(int i=0;i<iterations;i++)Calculate(optimized);
                watch.Stop();attention.Add(new{positions,headDimension=width,heads,kvHeads,optimized,iterations,ms=watch.Elapsed.TotalMilliseconds,
                    threadAllocatedBytes=GC.GetAllocatedBytesForCurrentThread()-allocated,maxError});
            }
        }
        string temp=Path.Combine(Path.GetTempPath(),"trit-journal-bench-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(temp);
        try
        {
            string path=Path.Combine(temp,"chat.jsonl");
            using(var writer=new StreamWriter(path,false,new UTF8Encoding(false)))
                for(int i=0;i<10000;i++)writer.WriteLine(JsonSerializer.Serialize(new ChatTurn("вопрос "+i,"ответ",i),JsonData.Options));
            var reference=LegacyTail(path);var current=ChatJournal.ReadTail(path);
            if(!reference.Turns.SequenceEqual(current.Turns))throw new Exception("Journal benchmark tail mismatch.");
            foreach(bool optimized in reverse?new[]{true,false}:new[]{false,true})
            {
                ChatReadResult Read()=>optimized?ChatJournal.ReadTail(path):LegacyTail(path);
                _=Read();_=Read();long allocated=GC.GetAllocatedBytesForCurrentThread();var watch=Stopwatch.StartNew();const int iterations=8;ChatReadResult last=current;
                for(int i=0;i<iterations;i++)last=Read();
                watch.Stop();journal.Add(new{optimized,iterations,ms=watch.Elapsed.TotalMilliseconds,threadAllocatedBytes=GC.GetAllocatedBytesForCurrentThread()-allocated,
                    inspectedRecordsPerPass=last.RecordsInspected,returned=last.Turns.Length,payloadBytesRead=last.BytesRead});
            }
        }
        finally{Directory.Delete(temp,true);}
        return new{scope="Actual C# synthetic attention and static journal-tail benchmark. No user files or native GPU workloads.",
            simdEnabled=Vector.IsHardwareAccelerated,floatLanes=Vector<float>.Count,attention,journal,
            caveats=new[]{"New attention keeps time-order accumulation, but target-specific lowering requires numeric tolerances.",
                "Journal benchmark uses valid records in an unchanged file. Corruption/growth/cancellation are separate tests.",
                "Warmup excluded. No mandatory speedup ratio; run both orders and inspect target hardware."}};
    }
    // Actual v12 time-major scalar value loop, retained only as a comparison oracle.
    private static void LegacyAttention(float[] q,float[] keys,float[] values,float[] scores,float[] output,int n,int heads,int kvHeads,int hd)
    {
        int kvDimension=kvHeads*hd;float scale=1/MathF.Sqrt(hd);
        for(int h=0;h<heads;h++)
        {
            int kh=h/(heads/kvHeads);float max=float.NegativeInfinity;
            for(int t=0;t<n;t++){float score=ManagedInference.Session.Dot(q,h*hd,keys,t*kvDimension+kh*hd,hd)*scale;scores[t]=score;max=MathF.Max(max,score);}
            float sum=0;for(int t=0;t<n;t++){scores[t]=MathF.Exp(scores[t]-max);sum+=scores[t];}
            float inverse=1/sum;int outputOffset=h*hd;Array.Clear(output,outputOffset,hd);
            for(int t=0;t<n;t++){float probability=scores[t]*inverse;int valueOffset=t*kvDimension+kh*hd;for(int d=0;d<hd;d++)output[outputOffset+d]+=probability*values[valueOffset+d];}
        }
    }
    private static ChatReadResult LegacyTail(string path)
    {
        using var file=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.ReadWrite);long start=Math.Max(0,file.Length-4*1024*1024);file.Position=start;
        using var reader=new StreamReader(file,Encoding.UTF8,detectEncodingFromByteOrderMarks:start==0);if(start>0)_=reader.ReadLine();
        var turns=new Queue<ChatTurn>();int inspected=0;
        while(reader.ReadLine() is string line)
        {
            if(string.IsNullOrWhiteSpace(line))continue;inspected++;
            if(line.Length>1_048_576)throw new Exception("Invalid benchmark input.");
            var turn=JsonSerializer.Deserialize<ChatTurn>(line,JsonData.Options)??throw new Exception("Invalid benchmark JSON.");
            if(turn.User is null||turn.Assistant is null||turn.User.Length>32768||turn.Assistant.Length>32768)throw new Exception("Invalid benchmark text.");
            turns.Enqueue(turn);while(turns.Count>250)turns.Dequeue();
        }
        return new(turns.ToArray(),0,file.Length-start,inspected);
    }
}
