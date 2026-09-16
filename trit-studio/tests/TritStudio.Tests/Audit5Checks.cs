using System.Text;
using TritStudio.Core;

internal static class Audit5Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("selected target batches preserve row-major loss positions", () => {
            var x = new EncodedExample(new[]{1,3,80,4,81},new[]{-100,-100,-100,81,2},"a");
            var y = new EncodedExample(new[]{1,90},new[]{90,2},"b");
            var b = SupervisedBatch.Build([x,y],5);
            Check(b.Positions.SequenceEqual(new long[]{3,4,5,6})); Check(b.Targets.SequenceEqual(new long[]{81,2,90,2}));
            Check(b.Inputs.SequenceEqual(new long[]{1,3,80,4,81,1,90,0,0,0}));
        }),
        ("selected targets reject truncation and preserve masked tails", () => {
            var a = new EncodedExample([1,2,3],[-100,2,-100],"a");
            Check(SupervisedBatch.Build([a],2).Targets.Length==1);
            Throws(()=>SupervisedBatch.Build([a with {Labels=[-100,2,4]}],2));
        }),
        ("selected targets reject out-of-vocabulary inputs before CUDA", () => {
            Throws(()=>SupervisedBatch.Build([new([1,262],[2,2],"x")],2));
            Throws(()=>SupervisedBatch.Build([new([-1],[2],"x")],1));
        }),
        ("selected targets reject invalid label shapes and values", () => {
            Throws(()=>SupervisedBatch.Build([new([1],[-1],"x")],1));
            Throws(()=>SupervisedBatch.Build([new([1],[262],"x")],1));
            Throws(()=>SupervisedBatch.Build([new([1,2],[2],"x")],2));
            Throws(()=>SupervisedBatch.Build([new([1],[-100],"x")],1));
        }),
        ("batch preparation observes cancellation before allocation", () => {
            using var c = new CancellationTokenSource(); c.Cancel();
            Throws(()=>SupervisedBatch.Build([new([1],[2],"x")],1,c.Token));
        }),
        ("holdout guard ignores the proposed target", () => {
            var original=Dataset.Make("Какой вариант?","Первый"); var g=new ValidationGuard([original]);
            Check(g.Contains(Dataset.Make("  какой   ВАРИАНТ?  ","Второй"))); Throws(()=>g.EnsureTraining([original with {Answer="Другой"}]));
        }),
        ("holdout guard keeps different conversation contexts distinct", () => {
            var a=Dataset.Make("А потом?","Второй",history:[new("План A","Первый",0)]);
            var b=Dataset.Make("А потом?","Второй",history:[new("План B","Первый",0)]);
            var guard=new ValidationGuard([a]); Check(!guard.Contains(b));
            Check(guard.Contains(a with { Answer="Иной ответ" }));
        }),
        ("plain text and supervised dialogue do not collapse into one split key", () => {
            var g=new ValidationGuard([Dataset.Make("Привет","Здравствуйте")]); Check(!g.Contains(Dataset.Make("Привет")));
        }),
        ("bounded JSON writes never exceed the allowed backing length", () => {
            using var file=new MemoryStream(); using var limit=new BoundedWriteStream(file,8);
            limit.Write(new byte[7]); Throws(()=>limit.Write(new byte[2])); Check(file.Length==7);
            limit.WriteByte(1); Throws(()=>limit.WriteByte(2)); Check(file.Length==8);
        }),
        ("bounded JSON stream is non-owning", () => {
            using var memory=new MemoryStream(); using(var limit=new BoundedWriteStream(memory,0)) Throws(()=>limit.WriteByte(1));
            Check(memory.CanWrite&&memory.Length==0);
        }),
        ("bounded reader preserves CRLF, empty lines and EOF tails", () => {
            var reader=new BoundedLineReader(new StringReader("abc\r\n\nxy"),3);
            Check(Read(reader)=="abc"); Check(Read(reader)==""); Check(Read(reader)=="xy"); Check(Read(reader) is null);
        }),
        ("bounded reader handles fragmented UTF16 and split CRLF", () => {
            var input=new FragmentReader("Привет🙂\r\nnext\n",1); var lines=new BoundedLineReader(input,8);
            Check(Read(lines)=="Привет🙂"); Check(Read(lines)=="next"); Check(Read(lines) is null);
        }),
        ("oversize protocol records fail without reading the entire tail", () => {
            var input=new FragmentReader(new string('x',100000),17); var lines=new BoundedLineReader(input,32);
            Throws(()=>Read(lines)); Check(input.CharactersRead<=51);
        }),
        ("diagnostic truncation drains until the next real line", () => {
            var lines=new BoundedLineReader(new FragmentReader(new string('x',50000)+"\nOK\r\n",127),16,true);
            Check(Read(lines)==new string('x',16)); Check(lines.LastLineTruncated);
            Check(Read(lines)=="OK"); Check(!lines.LastLineTruncated);
        }),
        ("bounded protocol reader supports exact fragmented payload budget", () => {
            var lines=new BoundedLineReader(new FragmentReader(new string('z',8192)+"\r\n",4096),8192);
            Check(Read(lines)!.Length==8192); Check(!lines.LastLineTruncated); Check(Read(lines) is null);
        }),
        ("bounded reader is cancellable before waiting for input", () => {
            using var cancel=new CancellationTokenSource();cancel.Cancel();
            Throws(()=>new BoundedLineReader(new StringReader("x"),8).ReadLineAsync(cancel.Token).AsTask().GetAwaiter().GetResult());
        }),
        ("prompt budget equals assembled context without building tokens in UI", () => {
            var history=Enumerable.Range(0,20).Select(i=>new ChatTurn("Вопрос "+i,"Ответ🙂 "+i,0)).ToArray();
            foreach(int context in new[]{64,128,256,512}) {
                var b=ByteTokenizer.Measure(history,"Привет",context,16);var p=ByteTokenizer.Plan(history,"Привет",context,16);
                Check(b.InputTokens==p.Tokens.Length&&b.DroppedTurns==p.DroppedTurns&&b.RetainedTurns==p.RetainedTurns);
            }
        }),
        ("prompt budget rejects invalid Unicode and overflowing input", () => {
            Throws(()=>ByteTokenizer.Measure([],"\uD800",128,16));Throws(()=>ByteTokenizer.Measure([],new string('x',80),64,8));
        }),
        ("UTF8 start tokens must fit the remaining byte budget", () => {
            var g=new Utf8Guard();Check(!g.AllowsWithinBudget(0xD0+6,1));Check(g.AllowsWithinBudget(0xD0+6,2));
            Check(!g.AllowsWithinBudget(0xE2+6,2));Check(g.AllowsWithinBudget(0xE2+6,3));
            Check(!g.AllowsWithinBudget(0xF0+6,3));Check(g.AllowsWithinBudget(0xF0+6,4));
            Check(g.AllowsWithinBudget(ByteTokenizer.Eos,0));
        }),
        ("UTF8 continuation budget never admits an unfinished scalar", () => {
            var g=new Utf8Guard();g.Accept(0xF0+6);Check(g.RemainingBytes==3);
            Check(!g.AllowsWithinBudget(0x90+6,2));Check(g.AllowsWithinBudget(0x90+6,3));
            Check(!g.AllowsWithinBudget(ByteTokenizer.Eos,3));g.Accept(0x90+6);g.Accept(0x80+6);g.Accept(0x80+6);Check(g.Complete);
        }),
        ("reusable sampler matches the v4 stable sort distribution", () => {
            var sampler=new TokenSampler();var seen=new HashSet<int>{7,20,ByteTokenizer.Eos};
            foreach(var options in new[]{new SamplingOptions(),new SamplingOptions(0,262,1,1,8),new SamplingOptions(1.5,200,0.5,1.7,8)}) {
                var a=new Random(712);var b=new Random(712);
                for(int i=0;i<250;i++){
                    var logits=Enumerable.Range(0,262).Select(j=>(float)Math.Sin((j+i)%37)).ToArray();
                    Check(sampler.Sample(logits,seen,options,a)==LegacySample(logits,seen,options,b));
                }
            }
        }),
        ("reusable sampler rejects masked-out vocab and resets between draws", () => {
            var s=new TokenSampler();var values=new float[262];values[80]=2;
            Check(s.Sample(values,[],new(0),new Random(1))==80);
            values[80]=-2;values[81]=3;Check(s.Sample(values,[],new(0),new Random(1))==81);
            Throws(()=>s.Sample(values,[],new(),new Random(1),_=>false));
        }),
        ("session logit buffers do not alias independent conversations", () => {
            var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=32};
            var model=new ManagedInference(TernaryQuantizer.Quantize(WeightSet.Initialize(c)),0);
            var a=model.NewSession();var b=model.NewSession();var buffer=new float[262];
            var expected=b.Step(1);Check(ReferenceEquals(buffer,a.Step(1,default,true,buffer)));Check(buffer.SequenceEqual(expected));
            a.Step(30);Check(b.Position==1);Throws(()=>b.Step(1,default,true,new float[2]));Check(b.Position==1);
        }),
        ("cancelled inference does not perform a token step", () => {
            var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=32};
            var m=new ManagedInference(WeightSet.Initialize(c),0);var s=m.NewSession();using var stop=new CancellationTokenSource();stop.Cancel();
            Throws(()=>s.Step(1,stop.Token));Check(s.Position==0);
        }),
        ("old resource JSON enables target projection and explicit opt-out persists", () => {
            var old=System.Text.Json.JsonSerializer.Deserialize<ResourceOptions>("{}",JsonData.Options)!;
            Check(old.ProjectOnlyTargets);
            var no=old with{ProjectOnlyTargets=false};
            var encoded=System.Text.Json.JsonSerializer.Serialize(no,JsonData.Options);
            Check(!System.Text.Json.JsonSerializer.Deserialize<ResourceOptions>(encoded,JsonData.Options)!.ProjectOnlyTargets);
        }),
        ("cached split keys are independent of later validation answer mutation", () => {
            var example=Dataset.Make("Кто герой?","Лиса");var guard=new ValidationGuard([example]);
            Check(guard.Contains(example with{Answer="Ёж"}));Check(!guard.Contains(Dataset.Make("Где герой?","Лиса")));
        }),
        ("bounded startup capture drains a large diagnostic but retains only its prefix", () => {
            var reader=new FragmentReader(new string('z',100000),1000);
            var output=BoundedTextCapture.ReadAsync(reader,64).GetAwaiter().GetResult();
            Check(output.Text==new string('z',64)&&output.Truncated&&reader.CharactersRead==100000);
        }),
        ("exact startup capture budget is not a truncation", () => {
            var output=BoundedTextCapture.ReadAsync(new StringReader("abcdef"),6).GetAwaiter().GetResult();
            Check(output.Text=="abcdef"&&!output.Truncated);
        }),
        ("empty startup capture is complete", () => {
            var output=BoundedTextCapture.ReadAsync(new StringReader(""),8).GetAwaiter().GetResult();
            Check(output.Text==""&&!output.Truncated);
        }),
        ("startup diagnostic capture honors cancellation", () => {
            using var stop=new CancellationTokenSource();stop.Cancel();
            Throws(()=>BoundedTextCapture.ReadAsync(new StringReader("noise"),8,stop.Token).GetAwaiter().GetResult());
        }),
    ];
    private static string? Read(BoundedLineReader reader) => reader.ReadLineAsync().AsTask().GetAwaiter().GetResult();
    private static void Check(bool condition) { if(!condition) throw new Exception("Audit5 assertion failed"); }
    private static void Throws(Action action) { try {action();} catch {return;} throw new Exception("Expected failure"); }
    private sealed class FragmentReader(string text,int chunk) : TextReader
    {
        public int CharactersRead { get; private set; }
        public override ValueTask<int> ReadAsync(Memory<char> buffer,CancellationToken cancellationToken=default)
        {
            cancellationToken.ThrowIfCancellationRequested();int n=Math.Min(Math.Min(chunk,buffer.Length),text.Length-CharactersRead);
            text.AsMemory(CharactersRead,n).CopyTo(buffer);CharactersRead+=n;return ValueTask.FromResult(n);
        }
    }
    private static int LegacySample(float[] source,HashSet<int> seen,SamplingOptions o,Random rng)
    {
        o=o.Clamp();var logits=(float[])source.Clone();
        for(int i=0;i<logits.Length;i++){
            if(i<ByteTokenizer.Offset&&i!=ByteTokenizer.Eos)logits[i]=float.NegativeInfinity;
            else if(seen.Contains(i))logits[i]=logits[i]<0?(float)(logits[i]*o.RepetitionPenalty):(float)(logits[i]/o.RepetitionPenalty);
        }
        var indices=Enumerable.Range(0,logits.Length).OrderByDescending(i=>logits[i]).Take(o.TopK).ToArray();
        if(o.Temperature<1e-6)return indices[0];var p=new double[indices.Length];double sum=0;
        for(int i=0;i<p.Length;i++){p[i]=Math.Exp((logits[indices[i]]-logits[indices[0]])/o.Temperature);sum+=p[i];}
        int keep=0;double cumulative=0;do{cumulative+=p[keep++]/sum;}while(keep<p.Length&&cumulative<o.TopP);
        double kept=0;for(int i=0;i<keep;i++)kept+=p[i];double draw=rng.NextDouble()*kept;
        for(int i=0;i<keep;i++){draw-=p[i];if(draw<=0)return indices[i];}return indices[keep-1];
    }
}
