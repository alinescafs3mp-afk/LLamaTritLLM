using System.Text.Json;
using TritStudio.Core;

internal static class Audit22Checks
{
    private static (EncodedExample[] Corpus, TrainingExample[] Language, TrainingExample[] Facts) Fixture()
    {
        var language = new[] { Dataset.Make("Поговорим?","О чём хочется поговорить?"),Dataset.Make("Я прочитал роман.","Как тебе концовка?") };
        var facts = new[] {
            Dataset.Make("Как зовут?","Аня.",history:[new("Меня зовут Аня.","Понял.",0)]),
            Dataset.Make("Как зовут?","Олег.",history:[new("Меня зовут Олег.","Понял.",0)]),
            Dataset.Make("Где билет?","На столе.",history:[new("Билет на столе.","Понял.",0)]) };
        var all=language.Concat(facts).Append(Dataset.Make("Промежуточная реплика","Понял.")).ToArray();
        return (Dataset.EncodeAll(all,256,1),language,facts);
    }
    public static (string,Action)[] All =>
    [
        ("old JSON never enables v22 silently", () => {
            var t=JsonSerializer.Deserialize<TrainingOptions>("{\"conversationCourse\":true,\"contextPractice\":true}",JsonData.Options)!;
            Check(!t.TransferPractice);t.Validate();
        }),
        ("transfer requires course and context flags", () => {
            Throws(()=>new TrainingOptions{TransferPractice=true}.Validate());
            Throws(()=>new TrainingOptions{TransferPractice=true,ConversationCourse=true}.Validate());
            new TrainingOptions{TransferPractice=true,ConversationCourse=true,ContextPractice=true}.Validate();
        }),
        ("v22 plan roundtrips with existing user settings", () => {
            var t=new TrainingOptions{TransferPractice=true,ConversationCourse=true,ContextPractice=true,LearningRate=.0003,Steps=4567};
            Check(JsonSerializer.Deserialize<TrainingOptions>(JsonSerializer.Serialize(t,JsonData.Options),JsonData.Options)==t);
        }),
        ("v22 phase quotas for a full eight-row batch", () => {
            Check(DialogueBatchPlanner.Quotas(8,0,100)==new DialogueMix(2,4,2));
            Check(DialogueBatchPlanner.Quotas(8,20,100)==new DialogueMix(2,3,3));
            Check(DialogueBatchPlanner.Quotas(8,60,100)==new DialogueMix(3,3,2));
        }),
        ("v22 batch64 multiplies disclosed quotas", () => {
            Check(DialogueBatchPlanner.Quotas(64,5,100)==new DialogueMix(16,32,16));
            Check(DialogueBatchPlanner.Quotas(64,21,100)==new DialogueMix(16,24,24));
        }),
        ("one-row batches rotate rather than permanently starve categories", () => {
            var a=Enumerable.Range(0,8).Select(i=>DialogueBatchPlanner.Quotas(1,i,100)).ToArray();
            Check(a.Sum(x=>x.Language)==4&&a.Sum(x=>x.General)==2&&a.Sum(x=>x.Facts)==2);
        }),
        ("all supported batch sizes have exactly the requested number of slots", () => {
            for(int b=1;b<=64;b++)for(int step=0;step<100;step++){
                var x=DialogueBatchPlanner.Quotas(b,step,100);Check(x.General+x.Language+x.Facts==b);
            }
        }),
        ("invalid plan refuses before random sampling", () => {
            Throws(()=>DialogueBatchPlanner.Quotas(0,0,100));Throws(()=>DialogueBatchPlanner.Quotas(65,0,100));
            Throws(()=>DialogueBatchPlanner.Quotas(8,100,100));Throws(()=>DialogueBatchPlanner.Quotas(8,-1,100));
        }),
        ("v22 selection retains actual encoded references and statistics", () => {
            var f=Fixture();var p=new DialogueBatchPlanner(f.Corpus,f.Language,f.Facts);var rng=new SamplerRandom(42);
            var b=p.Select(8,rng,0,100);
            Check(b.Examples.All(x=>f.Corpus.Any(y=>ReferenceEquals(x,y))));
            Check(b.Length==b.Examples.Max(BatchPlanner.EffectiveLength));
            Check(b.TargetTokens==b.Examples.Sum(x=>(long)x.Labels.Count(y=>y!=-100)));
            Check(p.LastMix==new DialogueMix(2,4,2)&&p.FactQuestionGroups==2);
        }),
        ("priority final facts do not include the unrelated acknowledgement", () => {
            var f=Fixture();var p=new DialogueBatchPlanner(f.Corpus,f.Language,f.Facts);var rng=new SamplerRandom(32);
            var keys=f.Facts.Select(x=>x.Id).ToHashSet();
            for(int i=0;i<20;i++){var b=p.Select(8,rng,i,100);Check(keys.Contains(b.Examples[4].Id)&&keys.Contains(b.Examples[7].Id));}
        }),
        ("v22 selection is reproducible including next RNG state", () => {
            var f=Fixture();var a=new DialogueBatchPlanner(f.Corpus,f.Language,f.Facts);var b=new DialogueBatchPlanner(f.Corpus,f.Language,f.Facts);
            var r=new SamplerRandom(99);var q=new SamplerRandom(99);
            for(int step=0;step<100;step++)Check(a.Select(7,r,step,100).Examples.Select(x=>x.Id).SequenceEqual(b.Select(7,q,step,100).Examples.Select(x=>x.Id)));
            Check(r.State==q.State);
        }),
        ("pre-cancelled selection leaves RNG untouched", () => {
            var f=Fixture();var p=new DialogueBatchPlanner(f.Corpus,f.Language,f.Facts);var rng=new SamplerRandom(42);ulong before=rng.State;
            using var stop=new CancellationTokenSource();stop.Cancel();Throws(()=>p.Select(8,rng,0,100,stop.Token));Check(rng.State==before);
        }),
        ("empty priority corpus is an explicit error", () => {
            var f=Fixture();Throws(()=>new DialogueBatchPlanner(f.Corpus,[],f.Facts));Throws(()=>new DialogueBatchPlanner(f.Corpus,f.Language,[]));
        }),
        ("duplicated encoded IDs are refused", () => {
            var f=Fixture();Throws(()=>new DialogueBatchPlanner([..f.Corpus,f.Corpus[0]],f.Language,f.Facts));
        }),
        ("v22 bundled train/eval files have correct roles and update ownership", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            foreach(string file in new[]{"conversation-language.jsonl","conversation-transfer.jsonl"}){
                Check(BundledCorpus.Load(data,file).Length>100);foreach(string prefix in new[]{"trainer","trainer-cuda","checks"})
                    Check(UpdatePayloadPolicy.IsOwnedPayload(prefix+"/data/"+file));
            }
            Check(BundledCorpus.Load(data,"transfer-challenge.jsonl").Length==62);
            Throws(()=>Dataset.LoadTraining(Path.Combine(data,"transfer-challenge.jsonl")));
        }),
        ("v22 source and derived targets do not contain old or new probe inputs", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            var training=new[]{"seed.jsonl","conversation-starter.jsonl","conversation-context.jsonl","conversation-language.jsonl","conversation-transfer.jsonl"}
                .SelectMany(n=>BundledCorpus.Load(data,n)).DistinctBy(x=>x.Id).ToArray();
            var expanded=ConversationSupervision.Expand(training,new ValidationGuard([],512)).Examples;
            var tests=BundledCorpus.Load(data,"transfer-challenge.jsonl").Concat(BundledCorpus.Load(data,"context-challenge.jsonl")).ToArray();
            new ValidationGuard(ConversationSupervision.Expand(tests,new ValidationGuard([],512)).Examples,512).EnsureTraining(expanded);
        }),
        ("authored unknown-name templates preserve Russian agreement", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            var rows=BundledCorpus.Load(data,"conversation-transfer.jsonl").Concat(BundledCorpus.Load(data,"transfer-challenge.jsonl")).ToArray();
            Check(rows.Any(x=>x.Text=="Моё имя тебе известно?"));
            Check(rows.All(x=>!x.Text.Contains("мой имя",StringComparison.OrdinalIgnoreCase)&&!x.Text.Contains("имя тебе известен",StringComparison.OrdinalIgnoreCase)));
        }),
        ("previously recorded training input is excluded even when expected answer changes", () => {
            var a=new GeneralizationCase("a","fact","fact",true,true,[]);var b=new GeneralizationCase("b","b","x",false,false,[]);
            var report=new GeneralizationReport(1,1,1,"hash",2,"scope",[a,b]);
            Check(report.Exact==0&&report.Eligible==1&&report.PreviouslyTrained==1);
            var known=new ValidationGuard([Dataset.Make("q","a")],512);Check(known.Contains(Dataset.Make("q","different")));
        }),
        ("generated history cannot make a recorded input look unseen", () => {
            var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
            var model=new ManagedInference(WeightSet.Initialize(cfg),0,1);
            const string first="Привет.", last="Какое имя?";
            string reply=model.Generate(ByteTokenizer.Prompt([],first,512,112),()=>new SamplingOptions(0,262,1,1,112),null,CancellationToken.None);
            var known=Dataset.Make(last,"ignored known target",history:[new(first,reply,0)]);
            var probe=Dataset.Make(last,"public target",history:[new(first,"Different reference acknowledgement.",0)]);
            var report=GeneralizationProbe.Run(model,0,"hash",[probe],[known]);
            Check(report.PreviouslyTrained==1&&report.Eligible==0);
        }),
        ("generalization report uses literal text and no semantic-success claim", () => {
            var t=new ConversationProbeTurn("q","<script>x</script>",GenerationHealth.Inspect("x"),0);
            var r=new GeneralizationReport(1,1,1,"hash",1,"scope",[new("id","a","<script>x</script>",false,false,[t])]);
            Check(GeneralizationProbe.Markdown(r).Contains("    <script>x</script>"));Check(r.Exact==0&&r.Eligible==1);
        }),
        ("cancelled generalization does not start generation", () => {
            var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=256};
            var model=new ManagedInference(WeightSet.Initialize(cfg),0,1);using var stop=new CancellationTokenSource();stop.Cancel();
            Throws(()=>GeneralizationProbe.Run(model,0,"h",[Dataset.Make("a","b")],[],ct:stop.Token));
        }),
    ];
    private static void Check(bool ok){if(!ok)throw new Exception("Audit22 assertion failed");}
    private static void Throws(Action a){try{a();}catch{return;}throw new Exception("Expected rejection");}
}
