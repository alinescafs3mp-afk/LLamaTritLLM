using System.Text.Json;
using TritStudio.Core;

internal static class Audit11Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("lazy replay equals old pool at base and learned boundaries", () => {
            foreach (int b in new[] { 0, 1, 7, 100 }) foreach (int n in new[] { 0, 1, 64 })
                if (b + n > 0) foreach (int recent in new[] { 1, 2, 4 })
                    foreach (long step in new long[] { 0, Math.Max(0,b-1), b, b+n-1, b+n, long.MaxValue })
                        ReplayParity(b,n,recent,step);
        }),
        ("lazy replay does not encode unused learned rows", () => {
            var baseline = Rows(12); var bad = Dataset.Make("overlong",new string('x',2000));
            var batch = OnlineReplay.Build(baseline,Rows(4),new[]{bad},0,128,EmptyGuard());
            Check(batch.LearnedExamplesEncoded==0 && batch.BaseSelections==4);
        }),
        ("selected overlong learned answer is never truncated", () => {
            var bad = Dataset.Make("too long",new string('x',2000));
            Throws<ArgumentException>(()=>OnlineReplay.Build(Rows(1),Rows(1),new[]{bad},1,128,EmptyGuard()));
        }),
        ("selected legacy control input is rejected before learning", () => {
            var control = Dataset.Make("private control fixture","reference");
            var sameInput = Dataset.Make(control.Text,"different answer");
            var guard = new ValidationGuard(new[]{control},128);
            Throws<InvalidDataException>(()=>OnlineReplay.Build(Rows(1),Rows(1),new[]{sameInput},1,128,guard));
        }),
        ("unselected legacy control remains outside the consumed pool", () => {
            var control=Dataset.Make("not consumed fixture","reference");
            var batch=OnlineReplay.Build(Rows(8),Rows(4),new[]{control},0,128,new ValidationGuard(new[]{control},128));
            Check(batch.LearnedExamplesEncoded==0);
        }),
        ("repeated selected learned row is encoded once", () => {
            var batch=OnlineReplay.Build([],Rows(4),new[]{Dataset.Make("retained","yes")},0,128,EmptyGuard());
            Check(batch.LearnedExamplesEncoded==1 && ReferenceEquals(batch.Examples[4],batch.Examples[7]));
        }),
        ("lazy replay keeps all new examples in their original order", () => {
            var pending=Rows(4);var batch=OnlineReplay.Build(Rows(3),pending,[],2,128,EmptyGuard());
            for(int i=0;i<4;i++)Check(ReferenceEquals(pending[i],batch.Examples[i]));
        }),
        ("lazy replay rejects empty pending batch", () => Throws<ArgumentException>(()=>OnlineReplay.Build(Rows(1),[],[],0,128,EmptyGuard()))),
        ("lazy replay rejects unbounded recent pool", () => Throws<ArgumentException>(()=>OnlineReplay.Build(Rows(1),Rows(5),[],0,128,EmptyGuard()))),
        ("lazy replay rejects negative optimizer step", () => Throws<ArgumentException>(()=>OnlineReplay.Build(Rows(1),Rows(1),[],-1,128,EmptyGuard()))),
        ("lazy replay rejects empty replay pool", () => Throws<InvalidDataException>(()=>OnlineReplay.Build([],Rows(1),[],0,128,EmptyGuard()))),
        ("lazy replay rejects too many learned records", () => Throws<ArgumentException>(()=>OnlineReplay.Build(Rows(1),Rows(1),Enumerable.Range(0,65).Select(i=>Dataset.Make("l"+i)).ToArray(),0,128,EmptyGuard()))),
        ("lazy replay honours cancellation before selection", () => Throws<OperationCanceledException>(()=>OnlineReplay.Build(Rows(1),Rows(1),[],0,128,EmptyGuard(),new CancellationToken(true)))),
        ("unchanged corpus reuses exact prior owner", () => {
            var data = new[]{Dataset.Make("a","b"),Dataset.Make("c","d")};
            Check(ReferenceEquals(data,Dataset.ReuseUnchanged(data,data.ToArray())));
        }),
        ("changed corpus does not reuse serialization identity", () => {
            var old=new[]{Dataset.Make("a","b")};var changed=new[]{Dataset.Make("a","c")};
            Check(ReferenceEquals(changed,Dataset.ReuseUnchanged(old,changed)));
        }),
        ("corpus order matters to reuse", () => {
            var old=new[]{Dataset.Make("a","b"),Dataset.Make("c","d")};var reordered=old.Reverse().ToArray();
            Check(ReferenceEquals(reordered,Dataset.ReuseUnchanged(old,reordered)));
        }),
        ("metadata-only change invalidates unchanged corpus reuse", () => {
            var old=new[]{Dataset.Make("a","b","one")};var candidate=new[]{Dataset.Make("a","b","two")};
            Check(old[0].Id==candidate[0].Id && ReferenceEquals(candidate,Dataset.ReuseUnchanged(old,candidate)));
        }),
        ("new history ownership is conservatively not reused", () => {
            var a=new[]{Dataset.Make("a","b",history:new[]{new ChatTurn("old","answer",0)})};
            var b=new[]{Dataset.Make("a","b",history:new[]{new ChatTurn("old","answer",0)})};
            Check(a[0].Id==b[0].Id && ReferenceEquals(b,Dataset.ReuseUnchanged(a,b)));
        }),
        ("corpus reuse supports empty arrays", () => { TrainingExample[] old=[];Check(ReferenceEquals(old,Dataset.ReuseUnchanged(old,new TrainingExample[0]))); }),
        ("corpus reuse checks cancellation even for same owner", () => { TrainingExample[] data=[];Throws<OperationCanceledException>(()=>Dataset.ReuseUnchanged(data,data,new CancellationToken(true))); }),
        ("publication plan counts initial and partial final snapshots", () => {
            Check(CheckpointBudget.Publications(0,100,true)==1 && CheckpointBudget.Publications(0,100)==0);
            Check(CheckpointBudget.Publications(100,100)==1 && CheckpointBudget.Publications(101,100)==2);
            Check(CheckpointBudget.Publications(1200,100,true)==13);
        }),
        ("publication plan accepts exact remaining capacity", () => CheckpointBudget.EnsureFits(120,8)),
        ("publication plan rejects excess before any file writes", () => Throws<IOException>(()=>CheckpointBudget.EnsureFits(120,9))),
        ("full history cannot spend another online update", () => Throws<IOException>(()=>CheckpointBudget.EnsureFits(128,1))),
        ("publication plan rejects unsupported counts", () => {
            Throws<ArgumentOutOfRangeException>(()=>CheckpointBudget.Publications(-1,1));
            Throws<ArgumentOutOfRangeException>(()=>CheckpointBudget.Publications(1,0));
            Throws<ArgumentOutOfRangeException>(()=>CheckpointBudget.Publications(100001,1));
            Throws<ArgumentOutOfRangeException>(()=>CheckpointBudget.EnsureFits(-1,1));
        }),
        ("valid command envelope preserves correlation and payload", () => {
            var c=CommandEnvelope.Parse("{\"kind\":\"mode\",\"id\":\"abc\",\"payload\":{\"enabled\":false}}");
            Check(c.Kind=="mode" && c.Id=="abc" && !c.Payload.GetProperty("enabled").GetBoolean());
        }),
        ("well-formed unknown command can receive correlated refusal", () => Check(CommandEnvelope.Parse("{\"kind\":\"unknown\",\"id\":\"x\",\"payload\":{}}").Id=="x")),
        ("missing command identity is fatal framing", () => ThrowsAny(()=>CommandEnvelope.Parse("{\"kind\":\"stop\",\"payload\":{}}"))),
        ("duplicate and case-aliased command fields are rejected", () => {
            ThrowsAny(()=>CommandEnvelope.Parse("{\"kind\":\"stop\",\"id\":\"a\",\"id\":\"b\",\"payload\":{}}"));
            ThrowsAny(()=>CommandEnvelope.Parse("{\"kind\":\"stop\",\"Id\":\"a\",\"id\":\"b\",\"payload\":{}}"));
        }),
        ("non-object command payload cannot be ignored", () => {
            foreach(string payload in new[]{"null","false","42","[]","\"text\""})
                ThrowsAny(()=>CommandEnvelope.Parse("{\"kind\":\"status\",\"id\":\"a\",\"payload\":"+payload+"}"));
        }),
        ("malformed command roots and identity types fail", () => {
            foreach(string raw in new[]{"{", "[]", "null", "{\"kind\":\"status\",\"id\":42,\"payload\":{}}"})ThrowsAny(()=>CommandEnvelope.Parse(raw));
        }),
        ("oversized command identity rejected within bounded frame", () => {
            string raw=JsonSerializer.Serialize(new WorkerCommand("status",new string('x',129),Protocol.Element(new{})),JsonData.Options);
            ThrowsAny(()=>CommandEnvelope.Parse(raw));
        }),
        ("control characters in command metadata rejected", () => {
            string raw=JsonSerializer.Serialize(new WorkerCommand("status","a\nb",Protocol.Element(new{})),JsonData.Options);
            ThrowsAny(()=>CommandEnvelope.Parse(raw));
        }),
    ];
    private static EncodedExample[] Rows(int count) => Enumerable.Range(0,count).Select(i=>Dataset.Encode(Dataset.Make("base "+i,"answer"),128)).ToArray();
    private static ValidationGuard EmptyGuard() => new(Array.Empty<TrainingExample>(),128);
    private static void ReplayParity(int baseCount,int learnedCount,int recentCount,long step)
    {
        var baseline=Rows(baseCount);var recent=Rows(recentCount);
        var learned=Enumerable.Range(0,learnedCount).Select(i=>Dataset.Make("learned "+i,"retained target")).ToArray();
        var encoded=learned.Select(e=>Dataset.Encode(e,128)).ToArray();
        var actual=OnlineReplay.Build(baseline,recent,learned,step,128,EmptyGuard());
        Check(actual.Examples.Length==recentCount*2 && actual.LearnedExamplesEncoded<=recentCount);
        for(int i=0;i<recentCount;i++)
        {
            Check(ReferenceEquals(recent[i],actual.Examples[i]));
            int index=(int)(((ulong)step+(uint)i)%(uint)(baseCount+learnedCount));
            var expected=index<baseCount?baseline[index]:encoded[index-baseCount];
            Check(expected.Tokens.SequenceEqual(actual.Examples[i+recentCount].Tokens) && expected.Labels.SequenceEqual(actual.Examples[i+recentCount].Labels));
        }
    }
    private static void Check(bool value) { if(!value)throw new Exception("Audit11 contract failed."); }
    private static void Throws<T>(Action action) where T:Exception { try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name); }
    private static void ThrowsAny(Action action) { try{action();}catch(InvalidDataException){return;}catch(JsonException){return;}throw new Exception("Expected rejected command frame."); }
}
