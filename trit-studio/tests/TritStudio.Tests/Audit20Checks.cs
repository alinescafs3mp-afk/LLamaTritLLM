using System.Text.Json;
using TritStudio.Core;

internal static class Audit20Checks
{
    public static (string, Action)[] All =>
    [
        ("old jobs have no silent course or loss change", () => {
            var o=JsonSerializer.Deserialize<TrainingOptions>("{}",JsonData.Options)!;
            Check(!o.ConversationCourse&&!o.EqualExampleWeight);
        }),
        ("course settings persist in launch plans without changing architecture", () => {
            var d=new LaunchDraft(ModelConfig.Small,new ResourceOptions(),new TrainingOptions{ConversationCourse=true,EqualExampleWeight=true});
            Check(JsonSerializer.Deserialize<LaunchDraft>(JsonSerializer.Serialize(d,JsonData.Options),JsonData.Options)==d);
        }),
        ("every supplied assistant turn becomes a target with preceding context", () => {
            var a=new ChatTurn("Имя?","Трит",0);var b=new ChatTurn("Как дела?","Хорошо",0);
            var row=Dataset.Make("Пока","До встречи",history:[a,b]);
            var p=ConversationSupervision.Expand([row],new ValidationGuard([Dataset.Make("контроль","ответ")],512));
            Check(p.Examples.Length==3&&p.AddedAssistantTargets==2);
            Check(p.Examples.Any(x=>x.Text==a.User&&x.Answer==a.Assistant&&x.History is null));
            Check(p.Examples.Single(x=>x.Text==b.User).History!.Length==1);
            Check(p.Examples[0].Id==row.Id);
        }),
        ("assistant expansion is idempotent and never expands control targets", () => {
            var row=Dataset.Make("последний","ответ",history:[new("контроль","другая цель",0)]);
            var g=new ValidationGuard([Dataset.Make("контроль","эталон")],512);
            var a=ConversationSupervision.Expand([row],g);Check(a.Examples.Length==1&&a.ExcludedControlPrefixes==1);
            var b=ConversationSupervision.Expand(a.Examples,g);Check(b.Examples.SequenceEqual(a.Examples));
        }),
        ("assistant prefixes keep UTF8 labels and EOS, not user targets", () => {
            var row=Dataset.Make("дальше","да",history:[new("начало","Хорошо.",0)]);
            var p=ConversationSupervision.Expand([row],new ValidationGuard([],512));
            foreach(var r in p.Examples){var e=Dataset.Encode(r,512);Check(e.Labels.Count(x=>x!=-100)==ByteTokenizer.TokenCount(r.Answer!)+1);Check(e.Labels[^1]==ByteTokenizer.Eos);}
        }),
        ("expansion rejects contaminated originals and observes cancellation", () => {
            var row=Dataset.Make("same","answer");Throws(()=>ConversationSupervision.Expand([row],new ValidationGuard([row],512)));
            using var c=new CancellationTokenSource();c.Cancel();Throws(()=>ConversationSupervision.Expand([row],new ValidationGuard([],512),c.Token));
        }),
        ("text rows remain text without invented conversation", () => {
            var row=Dataset.Make("Связный небольшой текст.");var p=ConversationSupervision.Expand([row],new ValidationGuard([],512));
            Check(p.Examples.Single()==row&&p.AddedAssistantTargets==0);
        }),
        ("course phase boundaries and reference-only replay are deterministic", () => {
            var a=Dataset.Encode(Dataset.Make("a","b"),32);var b=Dataset.Encode(Dataset.Make("c","d"),32);
            var c=new ConversationCurriculum([a,b],[a.Id]);
            Check(ReferenceEquals(c.At(0,100),c.Foundation)&&ReferenceEquals(c.At(19,100),c.Foundation));
            Check(ReferenceEquals(c.At(20,100),c.Dialogue)&&ReferenceEquals(c.At(59,100),c.Dialogue));
            Check(ReferenceEquals(c.At(60,100),c.Context)&&ReferenceEquals(c.At(99,100),c.Context));
            Check(c.Context.Examples.All(x=>ReferenceEquals(x,a)||ReferenceEquals(x,b)));
            Throws(()=>c.At(-1,100));Throws(()=>c.At(100,100));
        }),
        ("course cannot silently proceed without the starter", () => {
            var a=Dataset.Encode(Dataset.Make("a","b"),32);Throws(()=>new ConversationCurriculum([a],["missing"]));
        }),
        ("balanced loss gives equal total weight to short and long examples", () => {
            var b=SupervisedBatch.Build([Dataset.Encode(Dataset.Make("q","a"),64),Dataset.Encode(Dataset.Make("q","abcdef"),64)],64);
            var w=ExampleLossWeights.Build(b);Check(Math.Abs(w.Sum()-1)<1e-6);
            for(int row=0;row<2;row++)Check(Math.Abs(w.Where((_,i)=>b.Positions[i]/64==row).Sum()-.5)<1e-6);
        }),
        ("balanced loss rejects targetless rows and malformed positions", () => {
            Throws(()=>ExampleLossWeights.Build(new([1,2],[1],[0],2,1)));
            Throws(()=>ExampleLossWeights.Build(new([1],[1],[5],1,1)));
        }),
        ("starter is train and challenge stays held out end to end", () => {
            string root=Path.Combine(AppContext.BaseDirectory,"data");
            Check(BundledCorpus.Load(root,"conversation-starter.jsonl").Length>200);
            Check(BundledCorpus.Load(root,"challenge-v20.jsonl").Length>0);
            Throws(()=>Dataset.LoadTraining(Path.Combine(root,"challenge-v20.jsonl")));
            Check(UpdatePayloadPolicy.IsOwnedPayload("trainer-cuda/data/conversation-starter.jsonl"));
        }),
        ("every new data file belongs to small updates, not retained vendors", () => {
            foreach(string name in new[]{"conversation-starter.jsonl","challenge-v20.jsonl"})
            foreach(string prefix in new[]{"trainer/data/","trainer-cuda/data/","checks/data/"})Check(UpdatePayloadPolicy.IsOwnedPayload(prefix+name));
            Check(!UpdatePayloadPolicy.IsOwnedPayload("trainer-cuda/torch_cuda.dll"));
        }),
        ("conversation report distinguishes open review and simple checks", () => {
            var cases=ConversationProbe.Cases;Check(cases.Count>=10&&cases.Any(x=>!x.AutomaticCheck)&&cases.Any(x=>x.Questions.Length>=3));
            var c=cases.Single(x=>x.Id=="constraint");Check(ConversationProbe.Matches(c,"понятно"));Check(!ConversationProbe.Matches(c,"понятно."));
            var r=new ConversationProbeReport(1,2,10,"hash","scope",[new("open","manual",null,false,[]),new("closed","check",false,true,[])]);
            Check(r.Checked==1&&r.Passed==0&&r.Repetitive==1&&r.Summary.Contains("не процент"));
        }),
        ("course prefixes are not silently counted as new source documents", () => {
            var row=Dataset.Make("q2","a2",history:[new("q1","a1",0)]);var p=ConversationSupervision.Expand([row],new ValidationGuard([],512));
            Check(p.OriginalRows==1&&p.Examples.Length==2&&p.AddedAssistantTargets==1);
        }),
    ];
    private static void Check(bool ok){if(!ok)throw new Exception("Audit20 assertion failed");}
    private static void Throws(Action f){try{f();}catch{return;}throw new Exception("Expected explicit refusal");}
}
