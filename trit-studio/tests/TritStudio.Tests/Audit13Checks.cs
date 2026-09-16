using System.Text;
using System.Text.Json;
using TritStudio.Core;

public static class Audit13Checks
{
    public static IEnumerable<(string Name, Action Run)> All => new (string Name, Action Run)[]
    {
        ("SIMD attention agrees with v12 scalar channels across GQA and tail widths", () => {
            foreach (int n in new[]{1,17,129}) foreach(int hd in new[]{3,8,16,33}) foreach(int heads in new[]{1,4})
            {
                int kvHeads=heads==1?1:2;var rng=new Random(n+hd+heads);
                float[] Data(int size)=>Enumerable.Range(0,size).Select(_=>(float)(rng.NextDouble()-.5)).ToArray();
                var q=Data(heads*hd);var k=Data(n*kvHeads*hd);var v=Data(k.Length);
                var expected=Reference(q,k,v,n,heads,kvHeads,hd);var actual=new float[q.Length];var scores=new float[n];
                CpuAttention.Compute(q,k,v,scores,actual,n,heads,kvHeads,hd);
                Close(expected,actual); // Same order per lane; tolerate target-specific SIMD lowering.
                Array.Fill(actual,17f);CpuAttention.Compute(q,k,v,scores,actual,n,heads,kvHeads,hd);Close(expected,actual);
            }
        }),
        ("attention rejects short buffers before writing output", () => {
            var output=Enumerable.Repeat(7f,8).ToArray();Throws(()=>CpuAttention.Compute(new float[8],new float[7],new float[8],new float[2],output,2,2,1,4));
            Assert(output.All(x=>x==7));
        }),
        ("attention rejects invalid GQA and excessive position", () => {
            Throws(()=>CpuAttention.Compute(new float[8],new float[8],new float[8],new float[2],new float[8],2,2,3,4));
            Throws(()=>CpuAttention.Compute(new float[8],new float[8],new float[8],new float[2],new float[8],2049,2,1,4));
        }),
        ("attention disallows aliases of mutable scratch with inputs", () => {
            var q=new float[8];var keys=new float[8];var values=new float[8];var scores=new float[8];
            Throws(()=>CpuAttention.Compute(q,keys,values,scores,q,2,2,1,4));
            Throws(()=>CpuAttention.Compute(q,keys,values,keys,new float[8],2,2,1,4));
            Throws(()=>CpuAttention.Compute(q,keys,values,scores,scores,2,2,1,4));
        }),
        ("attention cancellation precedes buffer writes", () => {
            using var stop=new CancellationTokenSource();stop.Cancel();var output=Enumerable.Repeat(9f,8).ToArray();
            Throws<OperationCanceledException>(()=>CpuAttention.Compute(new float[8],new float[8],new float[8],new float[2],output,2,2,1,4,stop.Token));Assert(output.All(x=>x==9));
        }),
        ("attention refuses nonfinite scores instead of sampling corrupt probabilities", () => {
            var q=new float[8];q[0]=float.NaN;Throws<ArithmeticException>(()=>CpuAttention.Compute(q,new float[8],new float[8],new float[2],new float[8],2,2,1,4));
        }),
        ("journal parses only needed newest records and restores chronological order", () => {
            using var stream=StreamOf(Enumerable.Range(0,1000).Select(i=>Record(new("u"+i,"a",i))));
            var read=ChatJournal.ReadTail(stream,limit:7,maxBytes:262144);
            Assert(read.Turns.Select(t=>t.Revision).SequenceEqual(Enumerable.Range(993,7).Select(x=>(long)x)));
            Equal(7,read.RecordsInspected);Equal(0,read.SkippedRecords);Assert(stream.CanRead);
        }),
        ("journal finite snapshot does not chase appends", () => {
            byte[] original=Encoding.UTF8.GetBytes(Record(new("before","answer",1)));using var stream=new GrowingStream(original);
            var read=ChatJournal.ReadTail(stream,limit:10,maxBytes:2048);
            Equal(1,read.Turns.Length);Equal("before",read.Turns[0].User);Equal((long)original.Length,read.BytesRead);
            Assert(stream.Length>original.Length);Equal((long)original.Length,stream.Position);
        }),
        ("journal snapshot handles repeated short reads", () => {
            byte[] bytes=Encoding.UTF8.GetBytes(Record(new("русский 🙂","ответ",1))+Record(new("last","done",2)));
            using var stream=new ShortStream(bytes);var read=ChatJournal.ReadTail(stream);Equal(2,read.Turns.Length);Equal("русский 🙂",read.Turns[0].User);
        }),
        ("journal exact leading record boundary is not discarded", () => {
            string last=Record(new("last","answer",2));string prefix=Record(new("earlier","answer",1));
            // Exactly 1024 bytes remain and the preceding byte is LF.
            string suffix=last+new string(' ',1024-Encoding.UTF8.GetByteCount(last));
            using var stream=StreamOf([prefix,suffix]);var read=ChatJournal.ReadTail(stream,limit:5,maxBytes:1024);
            Equal(1,read.Turns.Length);Equal(2L,read.Turns[0].Revision);Equal(1024L,read.BytesRead);
        }),
        ("journal partial leading UTF8 record is skipped safely", () => {
            string first=Record(new(new string('я',1000),"answer",1));string last=Record(new("last","done",2));
            using var stream=StreamOf([first,last]);var read=ChatJournal.ReadTail(stream,limit:5,maxBytes:1024);
            Equal(1,read.Turns.Length);Equal("last",read.Turns[0].User);Assert(read.BytesRead<=1024);
        }),
        ("journal invalid UTF8 record skipped without replacement text", () => {
            byte[] bad=Encoding.UTF8.GetBytes(Record(new("bad","text",0)));bad[Array.IndexOf(bad,(byte)'b')]=0xFF;
            byte[] good=Encoding.UTF8.GetBytes(Record(new("safe","ok",1)));
            using var stream=new MemoryStream(bad.Concat(good).ToArray());var read=ChatJournal.ReadTail(stream);
            Equal(1,read.SkippedRecords);Equal(1,read.Turns.Length);Equal("safe",read.Turns[0].User);
        }),
        ("journal BOM CRLF blank lines and valid unterminated final record", () => {
            using var stream=StreamOf(["\uFEFF"+Record(new("first","ok",0)).Replace("\n","\r\n"),"\r\n \t\r\n",Record(new("last","ok",1)).TrimEnd('\n')]);
            var read=ChatJournal.ReadTail(stream);Equal(2,read.Turns.Length);Equal(0,read.SkippedRecords);
        }),
        ("journal corrupt tail remains counted while good earlier records survive", () => {
            using var stream=StreamOf([Record(new("good","a",0)),"{\"user\":\"truncated"]);
            var read=ChatJournal.ReadTail(stream);Equal(1,read.Turns.Length);Equal(1,read.SkippedRecords);
        }),
        ("journal oversized newest row never creates a giant decoded string", () => {
            using var stream=StreamOf([Record(new("good","ok",1)),new string('x',1_048_577)+"\n"]);
            var read=ChatJournal.ReadTail(stream,maxBytes:2*1024*1024);Equal(1,read.Turns.Length);Equal(1,read.SkippedRecords);
        }),
        ("journal ignores malformed captured history but keeps valid records", () => {
            var bad=new ChatTurn("q","a",0,false,"c",[new("secret","x",0,true,"c")]);
            using var stream=StreamOf([Record(bad),Record(new("valid","yes",1))]);var read=ChatJournal.ReadTail(stream);
            Equal(1,read.Turns.Length);Equal(1,read.SkippedRecords);
        }),
        ("journal cancellation is not swallowed as damaged data", () => {
            using var stream=StreamOf([Record(new("q","a",0))]);using var stop=new CancellationTokenSource();stop.Cancel();
            Throws<OperationCanceledException>(()=>ChatJournal.ReadTail(stream,ct:stop.Token));Equal(0L,stream.Position);
        }),
        ("journal rejects invalid limits before file lookup", () => {
            Throws<ArgumentOutOfRangeException>(()=>ChatJournal.ReadTail("absent-v13.jsonl",limit:0));
            using var stream=new MemoryStream();Throws<ArgumentOutOfRangeException>(()=>ChatJournal.ReadTail(stream,maxBytes:0));
        }),
        ("excluded failed prompt restores exclusion on retry", () => {
            var r=DraftRecovery.Restore("",false,"private prompt",true);Assert(r.Restored&&r.ExcludeNext);Equal("private prompt",r.Text);
        }),
        ("retry keeps a newly enabled exclusion too", () => {
            var r=DraftRecovery.Restore(null,true,"normal prompt",false);Assert(r.Restored&&r.ExcludeNext);
        }),
        ("new draft and its privacy setting are not overwritten", () => {
            foreach(string text in new[]{"next question"," ","\n"})
            {var r=DraftRecovery.Restore(text,false,"old private",true);Assert(!r.Restored&&!r.ExcludeNext);Equal(text,r.Text);}
        }),
        ("nonprivate retry does not invent an exclusion", () => {
            var r=DraftRecovery.Restore("",false,"normal",false);Assert(r.Restored&&!r.ExcludeNext);
        })
    };
    public static float[] Reference(float[] q,float[] keys,float[] values,int n,int heads,int kvHeads,int hd)
    {
        var output=new float[heads*hd];var scores=new float[n];int kvDimension=kvHeads*hd;float scale=1/MathF.Sqrt(hd);
        for(int h=0;h<heads;h++)
        {
            int kh=h/(heads/kvHeads);float max=float.NegativeInfinity;
            for(int t=0;t<n;t++){float s=ManagedInference.Session.Dot(q,h*hd,keys,t*kvDimension+kh*hd,hd)*scale;scores[t]=s;max=MathF.Max(max,s);}
            float sum=0;for(int t=0;t<n;t++){scores[t]=MathF.Exp(scores[t]-max);sum+=scores[t];}
            float inv=1/sum;
            for(int t=0;t<n;t++){float p=scores[t]*inv;for(int d=0;d<hd;d++)output[h*hd+d]+=p*values[t*kvDimension+kh*hd+d];}
        }
        return output;
    }
    private static void Close(float[] expected,float[] actual)
    { Assert(expected.Length==actual.Length);for(int i=0;i<actual.Length;i++)Assert(float.IsFinite(actual[i])&&MathF.Abs(expected[i]-actual[i])<=2e-6f+2e-5f*MathF.Abs(expected[i]),"Attention parity failed."); }
    private static string Record(ChatTurn turn)=>JsonSerializer.Serialize(turn,JsonData.Options)+"\n";
    private static MemoryStream StreamOf(IEnumerable<string> rows)=>new(Encoding.UTF8.GetBytes(string.Concat(rows)));
    private sealed class ShortStream(byte[] data):MemoryStream(data)
    { public override int Read(byte[] buffer,int offset,int count)=>base.Read(buffer,offset,Math.Min(7,count)); }
    private sealed class GrowingStream:MemoryStream
    {
        private bool _grew;
        public GrowingStream(byte[] initial){Write(initial);Position=0;}
        public override int Read(byte[] buffer,int offset,int count)
        {
            if(!_grew){_grew=true;long position=Position;Position=Length;Write(Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat(Record(new("after","not in snapshot",2)),200))));Position=position;}
            return base.Read(buffer,offset,count);
        }
    }
    private static void Assert(bool value,string text="Assertion failed"){if(!value)throw new Exception(text);}
    private static void Equal<T>(T expected,T actual){if(!EqualityComparer<T>.Default.Equals(expected,actual))throw new Exception($"Expected {expected}, got {actual}");}
    private static void Throws(Action action){try{action();}catch{return;}throw new Exception("Expected failure");}
    private static void Throws<T>(Action action) where T:Exception{try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
}
