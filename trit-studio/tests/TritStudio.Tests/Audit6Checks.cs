using System.Security.Cryptography;
using TritStudio.Core;

internal static class Audit6Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("bundled corpus refuses a mismatched application version", () => Temp(dir=>{
            string source=Path.Combine(AppContext.BaseDirectory,"data");
            foreach(string name in new[]{"seed.jsonl","DATASET_MANIFEST.json"})File.Copy(Path.Combine(source,name),Path.Combine(dir,name));
            string manifest=Path.Combine(dir,"DATASET_MANIFEST.json");File.WriteAllText(manifest,File.ReadAllText(manifest).Replace(BundledCorpus.Version,"conversation-ru-v0"));
            Throws<InvalidDataException>(()=>BundledCorpus.Load(dir,"seed.jsonl"));
        })),
        ("group-local packing preserves every byte of the v5 format", () => {
            foreach (int group in new[] {1,3,8,32,128}) foreach (int planes in new[] {1,2,3}) foreach(float threshold in new[]{0f,.5f,1f})
            {
                var weights=Enumerable.Range(0,group*19).Select(i=>MathF.Sin(i)*.27f).ToArray();
                Compare(ReferencePack(weights,group,planes,threshold),TernaryQuantizer.Pack(weights,group,planes,threshold));
            }
        }),
        ("group-local pack parallel path preserves bytes and input", () => {
            var a=Enumerable.Range(0,65536+128).Select(i=>MathF.Cos(i)*.17f).ToArray(); var old=(float[])a.Clone();
            Compare(ReferencePack(a,32,3,.5f),TernaryQuantizer.Pack(a,32,3,.5f,4)); Check(a.SequenceEqual(old));
        }),
        ("parallel unpack is identical to sequential unpack", () => {
            var a=Enumerable.Range(0,65536+128).Select(i=>MathF.Sin(i*.01f)).ToArray();
            var p=TernaryQuantizer.Pack(a,32,3,.5f);
            Check(TernaryQuantizer.Unpack(p,a.Length,32,1).SequenceEqual(TernaryQuantizer.Unpack(p,a.Length,32,4)));
        }),
        ("packing zero and threshold-boundary values is unchanged", () => {
            foreach (var a in new[]{new float[32], Enumerable.Repeat(1f,32).ToArray(),Enumerable.Range(0,32).Select(i=>i%2==0?-1f:1f).ToArray()})
                foreach(float threshold in new[]{0f,.5f,1f}) Compare(ReferencePack(a,8,3,threshold),TernaryQuantizer.Pack(a,8,3,threshold));
        }),
        ("group-local quantizer rejects nonfinite and overflow scales", () => {
            Throws(()=>TernaryQuantizer.Pack([float.NaN],1,1,.5f));
            Throws(()=>TernaryQuantizer.Pack([float.PositiveInfinity],1,1,.5f));
            Throws(()=>TernaryQuantizer.Pack([float.MaxValue,float.MaxValue],2,1,.5f));
        }),
        ("pack and unpack observe pre-cancel before doing work", () => {
            using var cancel=new CancellationTokenSource(); cancel.Cancel();
            Throws<OperationCanceledException>(()=>TernaryQuantizer.Pack(new float[32],8,2,.5f,4,cancel.Token));
            Throws<OperationCanceledException>(()=>TernaryQuantizer.Unpack([new(new float[4],new byte[8])],32,8,4,cancel.Token));
        }),
        ("parallel unpack rejects noncanonical trailing trits", () => {
            var p=TernaryQuantizer.Pack(new float[65536],8,1,.5f);p[0].Trits[^1]=27;
            Throws(()=>TernaryQuantizer.Unpack(p,65536,8,4));
        }),
        ("hash-on-write equals file hashing and leaves stream ownership", () => {
            using var memory=new MemoryStream(); var bytes=Enumerable.Range(0,150000).Select(i=>(byte)i).ToArray();
            using(var h=new HashingWriteStream(memory))
            {
                h.Write(bytes,0,7);h.Write(bytes.AsSpan(7));
                Check(h.Finish()==Convert.ToHexString(SHA256.HashData(bytes)));Throws(()=>h.WriteByte(1));Throws(()=>h.Finish());
            }
            Check(memory.CanWrite && memory.ToArray().SequenceEqual(bytes));
        }),
        ("hashing writer cancels between bounded chunks", () => {
            using var cancel=new CancellationTokenSource(); using var memory=new CancelAfterWrite(cancel);
            using var h=new HashingWriteStream(memory,cancel.Token);
            Throws<OperationCanceledException>(()=>h.Write(new byte[200000]));Check(memory.Length==65536);
        }),
        ("hashed model write round-trips with unchanged checksums", () => Temp(dir => {
            var weights=WeightSet.Initialize(Config());
            foreach(bool packed in new[]{false,true})
            {
                string path=Path.Combine(dir,packed?"packed":"master");
                Check(ModelFiles.WriteHashed(path,weights,packed,2)==ModelFiles.Hash(path));
                var result=ModelFiles.Read(path,packed,2);var expected=packed?TernaryQuantizer.Quantize(weights):weights;
                Check(expected.CountChanged(result)==0);
            }
        })),
        ("model write preserves a pre-existing file", () => Temp(dir => {
            string path=Path.Combine(dir,"existing");File.WriteAllText(path,"owner data");
            Throws(()=>ModelFiles.WriteHashed(path,WeightSet.Initialize(Config()),true));Check(File.ReadAllText(path)=="owner data");
        })),
        ("pre-cancelled model write creates no file", () => Temp(dir => {
            using var cancel=new CancellationTokenSource();cancel.Cancel();string path=Path.Combine(dir,"cancelled");
            Throws<OperationCanceledException>(()=>ModelFiles.Write(path,WeightSet.Initialize(Config()),true,1,cancel.Token));Check(!File.Exists(path));
        })),
        ("partial invalid model write removes only its newly-created file", () => Temp(dir => {
            var w=WeightSet.Initialize(Config());w.Values["embedding"][0]=float.NaN;string path=Path.Combine(dir,"bad");
            Throws(()=>ModelFiles.Write(path,w,true));Check(!File.Exists(path));
        })),
        ("model reads and hashes honor pre-cancellation", () => Temp(dir => {
            string path=Path.Combine(dir,"model");ModelFiles.Write(path,WeightSet.Initialize(Config()),true);
            using var cancel=new CancellationTokenSource();cancel.Cancel();
            Throws<OperationCanceledException>(()=>ModelFiles.Read(path,true,2,cancel.Token));Throws<OperationCanceledException>(()=>ModelFiles.Hash(path,cancel.Token));
        })),
        ("effective split guard catches history removed by token budgeting", () => {
            var control=Dataset.Make("q","yes",history:[new(new string('x',30),"one",0)]);
            var training=Dataset.Make("q","no",history:[new(new string('y',30),"two",0)]);
            Check(!new ValidationGuard([control]).Contains(training));
            Check(new ValidationGuard([control],16).Contains(training));
        }),
        ("effective split guard retains meaningful different histories", () => {
            var a=Dataset.Make("q","a",history:[new("one","ok",0)]);var b=Dataset.Make("q","a",history:[new("two","ok",0)]);
            Check(!new ValidationGuard([a],32).Contains(b));
        }),
        ("effective split comparison includes retained shared suffix", () => {
            var a=Dataset.Make("q","a",history:[new(new string('x',50),"old",0),new("same","ok",0)]);
            var b=Dataset.Make("q","z",history:[new(new string('y',50),"old",0),new("same","ok",0)]);
            Check(new ValidationGuard([a],16).Contains(b));Check(Dataset.RetainedHistoryStart(a,16)==1);
        }),
        ("context retention agrees with exactly encoded history", () => {
            var e=Dataset.Make("q","a",history:[new(new string('x',60),"old",0),new("🙂","ok",0)]);
            int length=Dataset.RequiredSequenceLength(e)+ByteTokenizer.TokenCount("🙂")+2+4;
            Check(Dataset.RetainedHistoryStart(e,length)==1);
            var actual=Dataset.Encode(e,length);var expected=Dataset.Encode(Dataset.Make("q","a",history:[e.History![1]]),length);
            Check(actual.Tokens.SequenceEqual(expected.Tokens)&&actual.Labels.SequenceEqual(expected.Labels));
        }),
        ("context fitting rejects insufficient complete-answer budget", () => {
            var e=Dataset.Make("question","a long answer");Throws(()=>Dataset.RetainedHistoryStart(e,8));
            Throws(()=>new ValidationGuard([e],8));Throws(()=>new ValidationGuard([e],4096));
        }),
        ("effective key is target-independent when retained context matches", () => {
            var e=Dataset.Make(" which? ","a",history:[new("one","ok",0)]);
            var z=Dataset.Make("WHICH?","different",history:e.History);
            Check(ValidationGuard.InputKey(e,64)==ValidationGuard.InputKey(z,64));
        }),
        ("greedy sampler agrees with sorted reference and stable ties", () => {
            var sampler=new TokenSampler();var rng=new Random(41);
            for(int n=0;n<100;n++)
            {
                var logits=Enumerable.Range(0,262).Select(_=>(float)rng.Next(-4,5)).ToArray();var seen=new HashSet<int>{36,43,56};
                var o=new SamplingOptions(0,40,.9,1.3);Func<int,bool> allow=i=>i==2||i>=32;
                int expected=Enumerable.Range(0,262).Where(i=>allow(i)&&(i>=6||i==2))
                    .OrderByDescending(i=>seen.Contains(i)?logits[i]<0?(float)(logits[i]*1.3):(float)(logits[i]/1.3):logits[i]).ThenBy(i=>i).First();
                Check(sampler.Sample(logits,seen,o,rng,allow)==expected);
            }
        }),
        ("greedy all-masked outputs fail rather than choose a role", () => {
            Throws(()=>new TokenSampler().Sample(new float[262],[],new SamplingOptions(0),new Random(1),_=>false));
        }),
        ("single-byte reply does not compute an unused next forward", () => {
            var model=ConstantByteModel();int[] prompt=[1,3,71,2,4];
            var result=model.GenerateDetailed(prompt,()=>new SamplingOptions(0,1,1,1,1),null,CancellationToken.None);
            Check(result.GeneratedByteTokens==1&&result.ForwardSteps==prompt.Length&&result.StopReason=="token_limit");
        }),
        ("multi-byte-limit reply computes exactly the needed forward steps", () => {
            var model=ConstantByteModel();int[] prompt=[1,3,71,2,4];
            var result=model.GenerateDetailed(prompt,()=>new SamplingOptions(0,1,1,1,4),null,CancellationToken.None);
            Check(result.GeneratedByteTokens==4&&result.ForwardSteps==prompt.Length+3);
        }),
        ("generation stops at context capacity without an extra step", () => {
            var model=ConstantByteModel();int[] prompt=Enumerable.Repeat(71,31).ToArray();
            var result=model.GenerateDetailed(prompt,()=>new SamplingOptions(0,1,1,1,5),null,CancellationToken.None);
            Check(result.GeneratedByteTokens==1&&result.ForwardSteps==31&&result.StopReason=="context_limit");
        }),
        ("bound layer references do not share conversational KV state", () => {
            var model=ConstantByteModel();var a=model.NewSession();var b=model.NewSession();
            float[] expected=a.Step(71);b.Step(90);b.Step(91);float[] another=model.NewSession().Step(71);Check(expected.SequenceEqual(another));
        })
    ];
    private static ModelConfig Config()=>new(){Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=32};
    private static ManagedInference ConstantByteModel()
    {
        var c=Config();var values=WeightLayout.For(c).ToDictionary(s=>s.Name,s=>new float[s.Count]);
        foreach(var s in WeightLayout.For(c))if(!s.Quantized)Array.Fill(values[s.Name],1f);
        var embedding=values["embedding"];for(int token=0;token<262;token++)Array.Fill(embedding,token==71?2f:.01f,token*c.Dimension,c.Dimension);
        return new ManagedInference(new WeightSet(c,values),0);
    }
    private static void Compare(PackedPlane[] a,PackedPlane[] b)
    {Check(a.Length==b.Length);for(int i=0;i<a.Length;i++){Check(a[i].Scales.SequenceEqual(b[i].Scales));Check(a[i].Trits.SequenceEqual(b[i].Trits));}}
    // Deliberately preserve the older algorithm as an independent byte-format regression oracle.
    private static PackedPlane[] ReferencePack(float[] a,int group,int count,float threshold)
    {
        var r=(float[])a.Clone();int groups=a.Length/group,n=(group+4)/5;var result=new PackedPlane[count];
        for(int p=0;p<count;p++)
        {
            var scales=new float[groups];var trits=new byte[groups*n];
            for(int g=0;g<groups;g++)
            {
                float sum=0;for(int j=0;j<group;j++)sum+=MathF.Abs(r[g*group+j]);float scale=MathF.Max(sum/group,1e-8f);scales[g]=scale;
                for(int j=0;j<group;j+=5){int value=0,factor=1;for(int z=0;z<5&&j+z<group;z++)
                {int i=g*group+j+z;int trit=MathF.Abs(r[i])>scale*threshold?Math.Sign(r[i]):0;value+=(trit+1)*factor;factor*=3;r[i]-=scale*trit;}trits[g*n+j/5]=(byte)value;}
            }
            result[p]=new(scales,trits);
        }
        return result;
    }
    private static void Check(bool condition){if(!condition)throw new Exception("Audit6 assertion failed.");}
    private static void Throws(Action action){try{action();}catch{return;}throw new Exception("Expected failure.");}
    private static void Throws<T>(Action action)where T:Exception{try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
    private static void Temp(Action<string> action){string d=Path.Combine(Path.GetTempPath(),"trit-audit6-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(d);try{action(d);}finally{Directory.Delete(d,true);}}
    private sealed class CancelAfterWrite(CancellationTokenSource cancel):MemoryStream
    {public override void Write(ReadOnlySpan<byte> buffer){base.Write(buffer);cancel.Cancel();}}
}
