using System.Text.Json;
using TritStudio.Core;

internal static class Audit19Checks
{
    public static (string, Action)[] All =>
    [
        ("legacy jobs keep constant LR and explicit cadence", () => {
            var old = JsonSerializer.Deserialize<TrainingOptions>("{\"steps\":8000,\"learningRate\":0.01}",JsonData.Options)!;
            Check(!old.WarmupCosine && !old.AutoSnapshotInterval);
            Check(ManualLearningRate.At(old,7999)==.01);
        }),
        ("LR upper bound is not a minimum", () => {
            foreach(double r in new[]{1e-6,3e-4,.001,.01})new TrainingOptions{LearningRate=r}.Validate();
            Throws(()=>new TrainingOptions{LearningRate=.1}.Validate());
            Throws(()=>new TrainingOptions{LearningRate=double.NaN}.Validate());
        }),
        ("warmup cosine starts low reaches peak and decays without increasing late", () => {
            var p=new TrainingOptions{Steps=8000,WarmupCosine=true,LearningRate=.001};
            Check(Math.Abs(ManualLearningRate.At(p,0)-.00001)<1e-12);
            Check(ManualLearningRate.At(p,99)==.001);
            Check(Math.Abs(ManualLearningRate.At(p,7999)-.0001)<1e-12);
            double previous=.001;for(int i=100;i<8000;i++){double r=ManualLearningRate.At(p,i);Check(r<=previous+1e-12 && r>=.0001-1e-12);previous=r;}
        }),
        ("schedule supports small runs and rejects invalid offsets", () => {
            foreach(int n in new[]{1,2,3,10,100}){var p=new TrainingOptions{Steps=n,WarmupCosine=true};for(int i=0;i<n;i++)Check(ManualLearningRate.At(p,i)>0);Throws(()=>ManualLearningRate.At(p,n));}
            Throws(()=>ManualLearningRate.At(new TrainingOptions{Steps=0},0));
            Throws(()=>ManualLearningRate.At(new TrainingOptions(),-1));
        }),
        ("explicit exhausted job still refuses without rewriting policy", () => {
            var p=new TrainingOptions{Steps=8000,PublishEvery=100};Throws(()=>CheckpointBudget.Plan(p,51));Check(p.PublishEvery==100);
        }),
        ("long run auto cadence fits owner 51 plus 8000 case without deleting", () => {
            var p=new TrainingOptions{Steps=8000,PublishEvery=100,AutoSnapshotInterval=true};var plan=CheckpointBudget.Plan(p,51);
            Check(plan.PublishEvery==250 && CheckpointBudget.Publications(plan.Steps,plan.PublishEvery)==32);
            Check(p.PublishEvery==100); // caller's intent is never mutated
        }),
        ("sparser requested cadence stays unchanged", () => {
            Check(CheckpointBudget.Plan(new TrainingOptions{Steps=8000,PublishEvery=1000,AutoSnapshotInterval=true},51).PublishEvery==1000);
        }),
        ("auto cadence handles initial reservation and true full history", () => {
            var p=new TrainingOptions{Steps=100000,PublishEvery=100,AutoSnapshotInterval=true};
            var plan=CheckpointBudget.Plan(p,126,true);Check(plan.PublishEvery==100000);
            Throws(()=>CheckpointBudget.Plan(p,127,true));Throws(()=>CheckpointBudget.Plan(p,128));
            Check(CheckpointBudget.Plan(p with{Steps=0},127,true).Steps==0);
        }),
        ("zero-step placeholder removal does not touch learned material", () => {
            var b=Dataset.Make("базовый текст");var c=Dataset.Make("мой вопрос","мой ответ");TrainingExample[] old=[b,c];
            Check(InitialTrainingMaterial.RemoveUnlearnedBaseline(old,[b],0).SequenceEqual(new[]{c}));
            Check(ReferenceEquals(InitialTrainingMaterial.RemoveUnlearnedBaseline(old,[b],1),old));
            Throws(()=>InitialTrainingMaterial.RemoveUnlearnedBaseline(old,[b],-1));
        }),
        ("first conversation data route equivalence retains order", () => {
            TrainingExample[] basic=[Dataset.Make("база")];TrainingExample[] seed=[Dataset.Make("вопрос","ответ"),Dataset.Make("текст")];
            var staged=InitialTrainingMaterial.RemoveUnlearnedBaseline(basic,basic,0).Concat(seed).DistinctBy(x=>x.Id).ToArray();
            Check(staged.SequenceEqual(seed));
        }),
        ("reference bigram excludes masked targets and never learns controls", () => {
            var t=new EncodedExample([10,10,11,11],[20,20,21,-100],"t");var v=new EncodedExample([10,11,11],[20,21,20],"v");
            var b=new ByteBaseline([t]);var first=b.Evaluate([v]);var second=b.Evaluate([v]);
            Check(first.Correct==2&&first.Total==3&&first==second);
        }),
        ("reference fallback and ties choose stable lowest token", () => {
            var b=new ByteBaseline([new EncodedExample([10,10],[21,20],"t")]);
            Check(b.Evaluate([new EncodedExample([10,12],[20,20],"v")]).Correct==2);
        }),
        ("reference rejects empty targets shapes and invalid ids", () => {
            Throws(()=>new ByteBaseline([]));Throws(()=>new ByteBaseline([new EncodedExample([1],[999],"t")]));
            Throws(()=>new ByteBaseline([new EncodedExample([1,2],[1],"t")]));
            Throws(()=>new ByteBaseline([new EncodedExample([1],[-100],"t")]));
        }),
        ("reference observes cancellation before iteration", () => {
            using var c=new CancellationTokenSource();c.Cancel();Throws(()=>new ByteBaseline([Dataset.Encode(Dataset.Make("abc"),16)],c.Token));
        }),
        ("repetition diagnostic is descriptive and Unicode aware", () => {
            var a=GenerationHealth.Inspect(new string('о',64));Check(a.Repetitive&&a.LongestIdenticalRun==64&&a.UnicodeScalars==64);
            var b=GenerationHealth.Inspect(string.Concat(Enumerable.Repeat("🙂",16)));Check(b.Repetitive&&b.UnicodeScalars==16);
            Check(!GenerationHealth.Inspect("Привет! Как проходит день?").Repetitive);
        }),
        ("cyclic degeneration is detected without rewriting output", () => {
            string s=string.Concat(Enumerable.Repeat("аб",64));var h=GenerationHealth.Inspect(s);
            Check(h.Repetitive&&h.LongestIdenticalRun==1&&s.Length==128);
            Check(!GenerationHealth.Inspect("").Repetitive);
        }),
        ("actual optimizer rate state JSON remains backwards compatible", () => {
            var old=JsonSerializer.Deserialize<TrainerState>("{\"step\":7,\"samplerState\":42,\"targetTokens\":200}",JsonData.Options)!;
            Check(old.LastLearningRate is null);var current=old with{LastLearningRate=.0003};
            Check(JsonSerializer.Deserialize<TrainerState>(JsonSerializer.Serialize(current,JsonData.Options),JsonData.Options)==current);
        }),
        ("schedule choices survive launch draft serialization", () => {
            var d=new LaunchDraft(ModelConfig.Small,new ResourceOptions(),new TrainingOptions{WarmupCosine=true,AutoSnapshotInterval=true});
            var copy=JsonSerializer.Deserialize<LaunchDraft>(JsonSerializer.Serialize(d,JsonData.Options),JsonData.Options)!;
            Check(copy==d&&copy.Training.WarmupCosine&&copy.Training.AutoSnapshotInterval);
        }),
    ];
    private static void Check(bool condition){if(!condition)throw new Exception("Audit19 assertion failed");}
    private static void Throws(Action a){try{a();}catch{return;}throw new Exception("Expected refusal");}
}
