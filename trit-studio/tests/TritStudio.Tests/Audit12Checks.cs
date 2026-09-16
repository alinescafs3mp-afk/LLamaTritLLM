using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TritStudio.Core;

public static class Audit12Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("fused GQA projections match separate Dot rows exactly", () => { foreach(int n in new[]{16,64,256})foreach(int workers in new[]{1,2,4})Projection(n,n,n/2,n/2,workers); }),
        ("paired FFN projections match reference including uneven row chunks", () => { foreach(int n in new[]{16,64,256})foreach(int rows in new[]{17,64,257})Pair(n,rows); }),
        ("single projection retains exact Dot accumulation", () => { var a=Values(256*263);var input=Values(256);var output=new float[263];CpuProjection.Multiply(a,input,output,new ParallelOptions{MaxDegreeOfParallelism=2});Same(output,Reference(a,input)); }),
        ("projection rejects an invalid last matrix before any output writes", () => { var a=Values(64);var input=Values(8);var x=Enumerable.Repeat(99f,8).ToArray();var y=(float[])x.Clone();var z=(float[])x.Clone();Throws<ArgumentException>(()=>CpuProjection.Triple(a,a,new float[63],input,x,y,z,new()));Check(x.All(v=>v==99)&&y.All(v=>v==99)&&z.All(v=>v==99)); }),
        ("projection rejects shared output buffers", () => { var output=new float[8];Throws<ArgumentException>(()=>CpuProjection.Pair(Values(64),Values(64),Values(8),output,output,new())); }),
        ("projection refuses input overwrite", () => { var input=Values(8);var old=(float[])input.Clone();Throws<ArgumentException>(()=>CpuProjection.Multiply(Values(64),input,input,new()));Same(input,old); }),
        ("projection cancellation precedes output mutation", () => { var output=Enumerable.Repeat(77f,8).ToArray();var options=new ParallelOptions{CancellationToken=new CancellationToken(true)};Throws<OperationCanceledException>(()=>CpuProjection.Multiply(Values(64),Values(8),output,options));Check(output.All(v=>v==77)); }),
        ("validation no-trim input keys are computed once", () => { var row=Dataset.Make("control twelve","answer");var guard=new ValidationGuard([row],128);Check(guard.KeyBuilds==1);Check(!guard.Contains(Dataset.Make("other twelve","answer")));Check(guard.KeyBuilds==2); }),
        ("validation answer change still cannot bypass input identity", () => { var row=Dataset.Make("control answer twelve","one");Check(new ValidationGuard([row],128).Contains(Dataset.Make(row.Text,"two"))); }),
        ("validation still detects a trimmed control as a short training input", () => { var row=Dataset.Make("Q","A",history:[new ChatTurn(new string('x',80),new string('y',80),0)]);var guard=new ValidationGuard([row],32);Check(guard.KeyBuilds==2&&guard.Contains(Dataset.Make("Q","other"))); }),
        ("validation optimized keys retain legacy byte hashes and decisions", GuardParity),
        ("cancelled validation constructor never enumerates input", () => Throws<OperationCanceledException>(()=>new ValidationGuard(Never(),32,new CancellationToken(true)))),
        ("validation cancellation is checked while enumerating", () => { using var cancel=new CancellationTokenSource();IEnumerable<TrainingExample> Rows(){yield return Dataset.Make("first","a");cancel.Cancel();yield return Dataset.Make("second","b");}Throws<OperationCanceledException>(()=>new ValidationGuard(Rows(),128,cancel.Token)); }),
        ("validation empty ensure still respects cancellation", () => Throws<OperationCanceledException>(()=>new ValidationGuard([],128).EnsureTraining([],new CancellationToken(true)))),
        ("event payload survives parser lifetime", () => { var e=EventEnvelope.Parse("{\"kind\":\"status\",\"id\":null,\"data\":{\"message\":\"ok\"}}");Check(e.Data.GetProperty("message").GetString()=="ok"); }),
        ("duplicate event identity is a fatal frame", () => { Reject("{\"kind\":\"completed\",\"id\":\"a\",\"id\":\"b\",\"data\":{}}");Reject("{\"kind\":\"status\",\"Id\":\"a\",\"id\":\"b\",\"data\":{}}"); }),
        ("event requires object data and well formed identity", () => { foreach(var raw in new[]{"[]","null","{}","{\"kind\":\"status\",\"data\":[]}","{\"kind\":\"completed\",\"data\":{}}","{\"kind\":\"status\",\"id\":42,\"data\":{}}"})Reject(raw); }),
        ("valid unknown event remains forward compatible", () => Check(EventEnvelope.Parse("{\"kind\":\"future-event\",\"data\":{}}").Kind=="future-event")),
        ("terminal command name must exist before releasing a request", () => { foreach(var value in new[]{new{command=(object?)null},new{command=(object?)42},new{command=(object?)""}})Throws<InvalidDataException>(()=>EventEnvelope.CompletionCommand(Protocol.Element(value)));Throws<InvalidDataException>(()=>EventEnvelope.CompletionCommand(Protocol.Element(new{success=true}))); }),
        ("terminal duplicate success cannot choose an arbitrary outcome", () => { using var doc=JsonDocument.Parse("{\"command\":\"train\",\"success\":false,\"Success\":true}");Throws<InvalidDataException>(()=>EventEnvelope.CompletionCommand(doc.RootElement)); }),
        ("terminal command is returned for exact request correlation", () => Check(EventEnvelope.CompletionCommand(Protocol.Element(new{command="train",success=true}))=="train")),
    ];
    private static IEnumerable<TrainingExample> Never()=>Enumerable.Range(0,1).Select<int,TrainingExample>(_=>throw new Exception("Enumeration before cancellation"));
    private static float[] Values(int n)=>Enumerable.Range(0,n).Select(i=>(i%37-18)*0.0078125f).ToArray();
    private static float[] Reference(float[] a,float[] input)=>Enumerable.Range(0,a.Length/input.Length).Select(i=>ManagedInference.Session.Dot(a,i*input.Length,input,0,input.Length)).ToArray();
    private static void Projection(int n,int qa,int kb,int vc,int workers)
    {
        var a=Values(n*qa);var b=Values(n*kb);var c=Values(n*vc);var input=Values(n);var x=new float[qa];var y=new float[kb];var z=new float[vc];
        CpuProjection.Triple(a,b,c,input,x,y,z,new ParallelOptions{MaxDegreeOfParallelism=workers});Same(x,Reference(a,input));Same(y,Reference(b,input));Same(z,Reference(c,input));
    }
    private static void Pair(int n,int rows)
    {
        var a=Values(n*rows);var b=Values(n*(rows+1));var input=Values(n);var x=new float[rows];var y=new float[rows+1];
        CpuProjection.Pair(a,b,input,x,y,new ParallelOptions{MaxDegreeOfParallelism=2});Same(x,Reference(a,input));Same(y,Reference(b,input));
    }
    private static string LegacyKey(TrainingExample e,int? length=null)
    {
        static string N(string s)=>string.Join(" ",s.Normalize(NormalizationForm.FormKC).Split((char[]?)null,StringSplitOptions.RemoveEmptyEntries)).ToUpperInvariant();
        int first=length is int l&&e.IsDialogue?Dataset.RetainedHistoryStart(e,l):0;
        var payload=new{kind=e.IsDialogue?"dialogue":"text",text=N(e.Text),history=(e.History??[]).Skip(first).Select(t=>new[]{N(t.User),N(t.Assistant)}).ToArray()};
        return Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(payload,JsonData.Options)));
    }
    private static void GuardParity()
    {
        foreach(int length in new[]{32,128,512})
        {
            var controls=Enumerable.Range(0,16).Select(i=>Dataset.Make("Q "+i,"A",history:Enumerable.Range(0,i%4).Select(j=>new ChatTurn("earlier "+j,"answer "+j,0)).ToArray())).ToArray();
            var guard=new ValidationGuard(controls,length);var full=controls.Select(e=>LegacyKey(e)).ToHashSet();var effective=controls.Select(e=>LegacyKey(e,length)).ToHashSet();
            foreach(var e in controls.Concat(controls.Select(e=>Dataset.Make(e.Text,"new answer",history:e.History))).Concat(Enumerable.Range(0,16).Select(i=>Dataset.Make("other "+i,"B"))))
            {Check(LegacyKey(e)==ValidationGuard.InputKey(e));Check(LegacyKey(e,length)==ValidationGuard.InputKey(e,length));Check(guard.Contains(e)==(full.Contains(LegacyKey(e))||effective.Contains(LegacyKey(e,length))));}
        }
    }
    private static void Same(float[] a,float[] b)=>Check(a.Length==b.Length&&a.Zip(b).All(p=>BitConverter.SingleToInt32Bits(p.First)==BitConverter.SingleToInt32Bits(p.Second)));
    private static void Check(bool value){if(!value)throw new Exception("Audit12 contract failed.");}
    private static void Throws<T>(Action action)where T:Exception{try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
    private static void Reject(string text){try{EventEnvelope.Parse(text);}catch(InvalidDataException){return;}catch(JsonException){return;}throw new Exception("Expected rejected event frame.");}
}
