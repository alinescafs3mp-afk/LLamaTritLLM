using System.Text;
using System.Text.Json;
using TritStudio.Core;

internal static class Audit9Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("checkpoint corpus cache matches canonical JSON bytes and hash", () => Temp(d => {
            var rows = new[]{Dataset.Make("контекст", "ответ")}; var cache = new CheckpointJsonCache();
            string p=Path.Combine(d,"first.json"); string hash=cache.WriteNew(p,rows);
            Check(File.ReadAllBytes(p).SequenceEqual(JsonSerializer.SerializeToUtf8Bytes(rows,JsonData.Options)));
            Check(hash==ModelFiles.Hash(p) && cache.Serializations==1 && cache.Hits==0);
        })),
        ("checkpoint corpus cache reuses only the successful immutable payload", () => Temp(d => {
            var rows=new[]{Dataset.Make("один")};var cache=new CheckpointJsonCache();
            string h=cache.WriteNew(Path.Combine(d,"a"),rows);string h2=cache.WriteNew(Path.Combine(d,"b"),rows);
            Check(h==h2 && cache.Hits==1 && cache.Serializations==1 && cache.RetainedBytes>0);
            Check(File.ReadAllBytes(Path.Combine(d,"a")).SequenceEqual(File.ReadAllBytes(Path.Combine(d,"b"))));
        })),
        ("checkpoint corpus replacement invalidates cache even for equal contents", () => Temp(d => {
            var rows=new[]{Dataset.Make("один")};var cache=new CheckpointJsonCache();
            cache.WriteNew(Path.Combine(d,"a"),rows);cache.WriteNew(Path.Combine(d,"b"),rows.ToArray());
            Check(cache.Hits==0 && cache.Serializations==2);
        })),
        ("checkpoint changed dataset gets fresh bytes and digest", () => Temp(d => {
            var cache=new CheckpointJsonCache();string h=cache.WriteNew(Path.Combine(d,"a"),new[]{Dataset.Make("первый")});
            string h2=cache.WriteNew(Path.Combine(d,"b"),new[]{Dataset.Make("второй")});Check(h!=h2 && h2==ModelFiles.Hash(Path.Combine(d,"b")));
        })),
        ("checkpoint oversized cache payload streams once without retention", () => Temp(d => {
            var rows=new[]{Dataset.Make(new string('x',1000))};var cache=new CheckpointJsonCache(64);
            string p=Path.Combine(d,"a");Check(cache.WriteNew(p,rows)==ModelFiles.Hash(p));
            Check(cache.RetainedBytes==0 && cache.Serializations==1);
            cache.WriteNew(Path.Combine(d,"b"),rows);Check(cache.Serializations==2 && cache.Hits==0);
        })),
        ("checkpoint zero budget remains a valid streaming writer", () => Temp(d => {
            var cache=new CheckpointJsonCache(0);var rows=new[]{Dataset.Make("zero cache")};
            cache.WriteNew(Path.Combine(d,"a"),rows);cache.WriteNew(Path.Combine(d,"b"),rows);
            Check(cache.RetainedBytes==0 && cache.Hits==0 && cache.Serializations==2);
        })),
        ("checkpoint cached write never overwrites an existing file", () => Temp(d => {
            var cache=new CheckpointJsonCache();var rows=new[]{Dataset.Make("valid")};string p=Path.Combine(d,"a");
            cache.WriteNew(p,rows);string h=ModelFiles.Hash(p);Throws<IOException>(()=>cache.WriteNew(p,rows));Check(ModelFiles.Hash(p)==h && cache.Hits==0);
        })),
        ("checkpoint cache cancellation is honored on a warm hit", () => Temp(d => {
            var cache=new CheckpointJsonCache();var rows=new[]{Dataset.Make("valid")};cache.WriteNew(Path.Combine(d,"a"),rows);
            using var cts=new CancellationTokenSource();cts.Cancel();string target=Path.Combine(d,"b");
            Throws<OperationCanceledException>(()=>cache.WriteNew(target,rows,cts.Token));Check(!File.Exists(target) && cache.Hits==0);
        })),
        ("failed serialization does not publish a partial file or poison the cache", () => Temp(d => {
            var cache=new CheckpointJsonCache();var rows=new[]{Dataset.Make("ok")};cache.WriteNew(Path.Combine(d,"a"),rows);
            var loop=new Loop();loop.Next=loop;string p=Path.Combine(d,"bad");
            Throws<JsonException>(()=>cache.WriteNew(p,loop));Check(!File.Exists(p) && cache.RetainedBytes==0);
            cache.WriteNew(Path.Combine(d,"b"),rows);Check(cache.Serializations==2);
        })),
        ("checkpoint cache explicit clear releases the owner and payload", () => Temp(d => {
            var cache=new CheckpointJsonCache();var rows=new[]{Dataset.Make("ok")};cache.WriteNew(Path.Combine(d,"a"),rows);cache.Clear();
            Check(cache.RetainedBytes==0);cache.WriteNew(Path.Combine(d,"b"),rows);Check(cache.Hits==0 && cache.Serializations==2);
        })),
        ("checkpoint matching zero and trained counter files are accepted", () => {
            CheckpointStateGuard.Validate(Info(0,0,0),new(0,42,0),new([]),ModelConfig.Small);
            CheckpointStateGuard.Validate(Info(4,20,1),new(4,42,20),new([Dataset.Make("ok").Id]),ModelConfig.Small);
        }),
        ("checkpoint hash-valid wrong positive step is rejected", () => Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,1),new(5,42,20),new([]),ModelConfig.Small))),
        ("checkpoint mismatched target token count is rejected", () => Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,1),new(4,42,21),new([]),ModelConfig.Small))),
        ("checkpoint zero sampler and too few learned tokens are rejected", () => {
            Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,1),new(4,0,20),new([]),ModelConfig.Small));
            Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,2,1),new(4,42,2),new([]),ModelConfig.Small));
        }),
        ("checkpoint zero-stage training traces are rejected", () => {
            Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(0,1,0),new(0,42,1),new([]),ModelConfig.Small));
            Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(0,0,1),new(0,42,0),new([]),ModelConfig.Small));
            Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(0,0,0),new(0,42,0),new([Dataset.Make("x").Id]),ModelConfig.Small));
        }),
        ("checkpoint altered weight count cannot exceed parameter count", () => Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,ModelConfig.Small.ParameterCount+1),new(4,42,20),new([]),ModelConfig.Small))),
        ("checkpoint duplicate or forged online IDs are rejected", () => {
            string id=Dataset.Make("x").Id;
            foreach(var ids in new[]{new[]{id,id},new[]{"xyz"},new[]{new string('G',64)},new[]{id.ToLowerInvariant()}})
                Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,1),new(4,42,20),new(ids),ModelConfig.Small));
        }),
        ("checkpoint null commit list is rejected before native restore", () => Throws<InvalidDataException>(()=>CheckpointStateGuard.Validate(Info(4,20,1),new(4,42,20),new(null!),ModelConfig.Small))),
        ("strict TXT imports UTF8 with and without a BOM", () => Temp(d => {
            foreach(bool bom in new[]{false,true}) { string p=Path.Combine(d,"t.txt");File.WriteAllText(p,"привет\r\nвторая строка",new UTF8Encoding(bom,true));Check(Dataset.LoadTraining(p).Length==2); }
        })),
        ("strict JSONL imports UTF8 with BOM and CRLF", () => Temp(d => {
            string p=Path.Combine(d,"t.jsonl");File.WriteAllText(p,"{\"prompt\":\"вопрос\",\"answer\":\"ответ\"}\r\n",new UTF8Encoding(true,true));
            Check(Dataset.LoadTraining(p).Single().Answer=="ответ");
        })),
        ("strict TXT rejects malformed bytes instead of learning replacement characters", () => Temp(d => {
            foreach(var bytes in new[]{new byte[]{0x61,0xFF,0x62},new byte[]{0xEF,0xBB,0xBF,0x61,0xC0,0xAF},new byte[]{0xF0,0x9F}})
            { string p=Path.Combine(d,"t.txt");File.WriteAllBytes(p,bytes);Throws<InvalidDataException>(()=>Dataset.LoadTraining(p)); }
        })),
        ("strict JSONL rejects malformed UTF8 inside quoted text", () => Temp(d => {
            string p=Path.Combine(d,"t.jsonl");File.WriteAllBytes(p,Encoding.UTF8.GetBytes("{\"text\":\"").Concat(new byte[]{0xFF}).Concat(Encoding.UTF8.GetBytes("\"}" )).ToArray());
            Throws<InvalidDataException>(()=>Dataset.LoadTraining(p));
        })),
        ("UTF16 import requires explicit conversion rather than silent reinterpretation", () => Temp(d => {
            string p=Path.Combine(d,"t.txt");File.WriteAllText(p,"пример",Encoding.Unicode);Throws<InvalidDataException>(()=>Dataset.LoadTraining(p));
        })),
        ("strict dataset line reader bounds a line before parsing it", () => Temp(d => {
            string p=Path.Combine(d,"t.txt");File.WriteAllText(p,new string('x',32769));Throws<InvalidDataException>(()=>Dataset.LoadTraining(p));
            p=Path.Combine(d,"t.jsonl");File.WriteAllText(p,new string(' ',StrictDatasetLines.MaxJsonLineChars+100));Throws<InvalidDataException>(()=>Dataset.LoadTraining(p));
        })),
        ("strict dataset cancellation happens before accessing an absent path", () => {
            using var cts=new CancellationTokenSource();cts.Cancel();Throws<OperationCanceledException>(()=>StrictDatasetLines.Read("absent",100,cts.Token).ToArray());
        }),
        ("weight change count matches bounded parallel and sequential variants", () => {
            var a=WeightSet.Initialize(ModelConfig.Medium);var values=a.Values.ToDictionary(p=>p.Key,p=>(float[])p.Value.Clone());long expected=0;
            foreach(var v in values.Values) for(int i=0;i<v.Length;i+=17){v[i]+=1;expected++;}
            var b=new WeightSet(a.Config,values);Check(a.CountChanged(b)==expected && a.CountChanged(b,4)==expected && a.CountChanged(a,4)==0);
        }),
        ("weight change comparison rejects mismatched configurations", () => {
            var a=WeightSet.Initialize(ModelConfig.Small);var b=WeightSet.Initialize(ModelConfig.Small with {Seed=43});Throws<ArgumentException>(()=>a.CountChanged(b));
        }),
        ("weight change count honors cancellation before scanning", () => {
            var a=WeightSet.Initialize(ModelConfig.Small);using var cts=new CancellationTokenSource();cts.Cancel();Throws<OperationCanceledException>(()=>a.CountChanged(a,4,cts.Token));
        }),
        ("cached batch statistics preserve every selected example and RNG state", () => {
            var corpus=Enumerable.Range(0,65).Select(i=>new EncodedExample(Enumerable.Repeat(7,20+i).ToArray(),Enumerable.Range(0,20+i).Select(j=>j<8||j>=18+i?-100:7).ToArray(),i.ToString())).ToArray();
            foreach(bool bucket in new[]{false,true}) foreach(int batchSize in new[]{1,8,32}) foreach(bool forced in new[]{false,true})
            {
                var planner=new BatchPlanner(corpus);var a=new SamplerRandom(42);var b=new SamplerRandom(42);var required=forced?corpus[3]:null;
                for(int step=0;step<30;step++)
                {
                    var actual=planner.Select(batchSize,a,bucket,required);var expected=LegacySelect(corpus,batchSize,b,bucket,required);
                    Check(actual.Examples.Select(x=>x.Id).SequenceEqual(expected.Examples.Select(x=>x.Id)) && a.State==b.State);
                    Check(actual.Length==expected.Length && actual.InputTokens==expected.InputTokens && actual.TargetTokens==expected.TargetTokens);
                }
            }
        }),
        ("batch statistics preparation is cancellable", () => {
            using var cts=new CancellationTokenSource();cts.Cancel();Throws<OperationCanceledException>(()=>new BatchPlanner([new([1,2],[-100,2],"x")],ct:cts.Token));
        }),
    ];
    private static PlannedBatch LegacySelect(EncodedExample[] corpus,int size,SamplerRandom random,bool bucket,EncodedExample? required)
    {
        int[] lengths=corpus.Select(BatchPlanner.EffectiveLength).ToArray();var selected=new EncodedExample[size];int[]? candidates=null;
        if(bucket && required is null && corpus.Length>=32){int anchor=random.Next(corpus.Length);candidates=Enumerable.Range(0,corpus.Length).Where(i=>(lengths[i]-1)/32==(lengths[anchor]-1)/32).ToArray();selected[0]=corpus[anchor];}
        else selected[0]=required??corpus[random.Next(corpus.Length)];
        for(int i=1;i<size;i++)selected[i]=corpus[candidates is null?random.Next(corpus.Length):candidates[random.Next(candidates.Length)]];
        return new(selected,selected.Max(BatchPlanner.EffectiveLength),selected.Sum(e=>(long)BatchPlanner.EffectiveLength(e)),selected.Sum(e=>(long)e.Labels.Count(x=>x!=-100)));
    }
    private static RevisionInfo Info(long step,long tokens,long changed) => new(1,step,changed,null,"test",DateTimeOffset.UtcNow,"","",null,"","","",tokens,null);
    private static void Check(bool condition){if(!condition)throw new Exception("Audit9 assertion failed.");}
    private static void Throws<T>(Action action) where T:Exception {try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
    private static void Temp(Action<string> action){string path=Path.Combine(Path.GetTempPath(),"trit-a9-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(path);try{action(path);}finally{Directory.Delete(path,true);}}
    public sealed class Loop { public Loop? Next {get;set;} }
}
