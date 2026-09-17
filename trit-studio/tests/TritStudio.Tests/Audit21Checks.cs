using System.Text.Json;
using TritStudio.Core;

internal static class Audit21Checks
{
    private static EncodedExample Encode(string text) => Dataset.Encode(Dataset.Make(text,"ответ"),128);
    private static TrainingExample[] Pair() => [
        Dataset.Make("Напиши имя.","Лена.",history:[new("Зови меня Лена.","Понял.",0)]),
        Dataset.Make("Напиши имя.","Нина.",history:[new("Зови меня Нина.","Понял.",0)])];
    public static (string,Action)[] All =>
    [
        ("legacy JSON does not silently enable context practice", () => {
            var t=JsonSerializer.Deserialize<TrainingOptions>("{\"conversationCourse\":true}",JsonData.Options)!;
            Check(t.ConversationCourse&&!t.ContextPractice);
        }),
        ("context practice requires an explicit course", () => {
            Throws(()=>new TrainingOptions{ContextPractice=true}.Validate());
            new TrainingOptions{ContextPractice=true,ConversationCourse=true}.Validate();
        }),
        ("new context plan roundtrips without changing old values", () => {
            var t=new TrainingOptions{ContextPractice=true,ConversationCourse=true,Steps=1234,LearningRate=.0003,EqualExampleWeight=true};
            Check(JsonSerializer.Deserialize<TrainingOptions>(JsonSerializer.Serialize(t,JsonData.Options),JsonData.Options)==t);
        }),
        ("v21 all phases retain every original encoded example", () => {
            var a=Encode("a");var b=Encode("b");var c=Encode("c");var all=new[]{a,b,c};
            var course=new ConversationCurriculum(all,[a.Id],[b.Id]);
            foreach(var p in new[]{course.Foundation,course.Dialogue,course.Context})
                foreach(var item in all)Check(p.Examples.Any(x=>ReferenceEquals(x,item)));
        }),
        ("v21 phase sizes implement disclosed reference weights", () => {
            var all=Enumerable.Range(0,8).Select(i=>Encode("q"+i)).ToArray();
            var course=new ConversationCurriculum(all,[all[0].Id],[all[1].Id]);
            Check(course.Foundation.Examples.Length==32&&course.Dialogue.Examples.Length==20&&course.Context.Examples.Length==14);
            Check(course.Foundation.Examples.All(x=>all.Any(y=>ReferenceEquals(x,y))));
            Check(course.Foundation.Examples.Count(x=>ReferenceEquals(x,all[0]))==17);
        }),
        ("legacy course pools still have the old foundation", () => {
            var a=Encode("a");var b=Encode("b");var course=new ConversationCurriculum([a,b],[a.Id]);
            Check(course.Foundation.Examples.Length==1&&ReferenceEquals(course.Foundation.Examples[0],a));
        }),
        ("missing context rows are refused before training", () => {
            var a=Encode("a");Throws(()=>new ConversationCurriculum([a],[a.Id],["absent"]));
        }),
        ("context corpus is actual training material and belongs to every update copy", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            Check(BundledCorpus.Load(data,"conversation-context.jsonl").Length==388);
            foreach(string root in new[]{"trainer","trainer-cuda","checks"})
                Check(UpdatePayloadPolicy.IsOwnedPayload(root+"/data/conversation-context.jsonl"));
        }),
        ("context challenge loads only through evaluation and is pair coherent", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            var rows=BundledCorpus.Load(data,"context-challenge.jsonl");Check(rows.Length==32);ContextTransferProbe.Validate(rows);
            Throws(()=>Dataset.LoadTraining(Path.Combine(data,"context-challenge.jsonl")));
        }),
        ("training and derived assistant targets do not include new probe inputs", () => {
            string data=Path.Combine(AppContext.BaseDirectory,"data");
            var tests=BundledCorpus.Load(data,"context-challenge.jsonl");var allTests=ConversationSupervision.Expand(tests,new ValidationGuard([],512)).Examples;
            var controls=new ValidationGuard(allTests,512);
            var training=BundledCorpus.Load(data,"seed.jsonl").Concat(BundledCorpus.Load(data,"conversation-starter.jsonl"))
                .Concat(BundledCorpus.Load(data,"conversation-context.jsonl")).DistinctBy(x=>x.Id).ToArray();
            var expanded=ConversationSupervision.Expand(training,new ValidationGuard([],512)).Examples;
            controls.EnsureTraining(expanded);
        }),
        ("exact fact match refuses keywords in an incorrect or extra sentence", () => {
            Check(ContextTransferProbe.Matches("Лена.","лена!"));
            Check(!ContextTransferProbe.Matches("Лена.","Не Лена."));
            Check(!ContextTransferProbe.Matches("Лена.","Лена или Нина."));
            Check(ContextTransferProbe.Matches("Жёлтая.","желтая"));
        }),
        ("constant reply cannot pass a counterfactual pair", () => {
            var p=Pair();ContextTransferProbe.Validate(p);
            foreach(string answer in new[]{"Лена.","Нина.","", "Лена или Нина"})
                Check(!(ContextTransferProbe.Matches(p[0].Answer!,answer)&&ContextTransferProbe.Matches(p[1].Answer!,answer)));
        }),
        ("mismatched question or equal targets cannot be a contrast pair", () => {
            var p=Pair();Throws(()=>ContextTransferProbe.Validate([p[0],p[1] with {Text="другой вопрос"}]));
            Throws(()=>ContextTransferProbe.Validate([p[0],p[1] with {Answer="Лена."}]));
            Throws(()=>ContextTransferProbe.Validate([p[0]]));
        }),
        ("changing only reference assistant history is not a user fact contrast", () => {
            var p=Pair();Throws(()=>ContextTransferProbe.Validate([p[0],p[1] with {History=[new("Зови меня Лена.","Иная подсказка.",0)]}]));
        }),
        ("pre-cancelled context probe never generates", () => {
            using var cancel=new CancellationTokenSource();cancel.Cancel();
            var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Heads=2,KvHeads=1,Layers=1,GroupSize=8,Context=512};
            var model=new ManagedInference(WeightSet.Initialize(cfg),1,1);
            Throws(()=>ContextTransferProbe.Run(model,0,"hash",Pair(),ct:cancel.Token));
        }),
        ("real context probe is deterministic and records generated turns", () => {
            var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Heads=2,KvHeads=1,Layers=1,GroupSize=8,Context=512};
            var weights=WeightSet.Initialize(cfg);var before=weights.Values.ToDictionary(x=>x.Key,x=>x.Value.ToArray());
            var model=new ManagedInference(weights,12,1);var a=ContextTransferProbe.Run(model,0,"hash",Pair());
            var b=ContextTransferProbe.Run(model,0,"hash",Pair());
            Check(JsonSerializer.Serialize(a,JsonData.Options)==JsonSerializer.Serialize(b,JsonData.Options));
            Check(a.Pairs.Length==1&&a.Pairs[0].First.Turns.Length==2&&a.AvailablePairs==1);
            Check(weights.Values.All(x=>x.Value.SequenceEqual(before[x.Key])));
        }),
        ("context markdown quotes generated markup as text", () => {
            var turn=new ConversationProbeTurn("q","<script>bad</script>",GenerationHealth.Inspect("x"),0);
            var a=new ContextVariantResult("id","fact","answer",false,[turn]);
            var r=new ContextTransferReport(1,1,0,"hash",1,"scope",[new(0,false,true,a,a)]);
            Check(ContextTransferProbe.Markdown(r).Contains("    <script>bad</script>"));
            Check(r.PassedPairs==0&&r.SameAnswerPairs==1);
        }),
    ];
    private static void Check(bool ok){if(!ok)throw new Exception("Audit21 assertion failed");}
    private static void Throws(Action action){try{action();}catch{return;}throw new Exception("Expected explicit refusal");}
}
