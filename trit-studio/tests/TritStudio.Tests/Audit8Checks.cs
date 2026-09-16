using System.Text.Json;
using TritStudio.Core;

internal static class Audit8Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("completion success is parsed without inventing an error", () => {
            var result = Parse("{\"success\":true,\"cancelled\":false,\"error\":null}"); Check(result.Success && !result.Cancelled && result.Error is null);
        }),
        ("completion failure retains its explicit error", () => {
            var result = Parse("{\"success\":false,\"error\":\"disk full\"}"); Check(!result.Success && result.Error == "disk full");
        }),
        ("cancelled completion is not success", () => {
            var result = Parse("{\"success\":false,\"cancelled\":true}"); Check(!result.Success && result.Cancelled);
        }),
        ("missing or nonboolean completion success is rejected", () => {
            foreach (string json in new[] {"{}", "[]", "null", "{\"success\":\"yes\"}", "{\"success\":1}"})
                Throws<InvalidDataException>(() => Parse(json));
        }),
        ("nonstring error payload is rejected before releasing a request", () => {
            foreach (string value in new[] {"{}", "[]", "true", "123"}) Throws<InvalidDataException>(() => Parse("{\"success\":false,\"error\":" + value + "}"));
        }),
        ("contradictory completion flags cannot certify success", () => {
            Throws<InvalidDataException>(() => Parse("{\"success\":true,\"cancelled\":true}"));
            Throws<InvalidDataException>(() => Parse("{\"success\":true,\"error\":\"failed\"}"));
            Throws<InvalidDataException>(() => Parse("{\"success\":false,\"cancelled\":\"no\"}"));
        }),
        ("validation preparation reuses identical owned arrays", () => {
            var cache = new EvaluationBatchCache(Corpus(), 2, 128); var first = cache.Get(0);
            Check(ReferenceEquals(first, cache.Get(0)) && cache.Builds == 1 && cache.Hits == 1);
            Check(cache.RetainedPayloadBytes == 8L * (first.Inputs.Length + first.Positions.Length + first.Targets.Length));
        }),
        ("validation preparation has an explicit zero-retention mode", () => {
            var cache = new EvaluationBatchCache(Corpus(), 2, 128, 0); var first = cache.Get(0); var second = cache.Get(0);
            Check(!ReferenceEquals(first, second) && first.Inputs.SequenceEqual(second.Inputs));
            Check(cache.RetainedPayloadBytes == 0 && cache.Hits == 0 && cache.Builds == 2);
        }),
        ("validation preparation never retains beyond its payload budget", () => {
            var cache = new EvaluationBatchCache(Corpus(), 1, 128, 128);
            for (int pass=0;pass<3;pass++) for (int i=0;i<cache.Count;i++) { _=cache.Get(i); Check(cache.RetainedPayloadBytes <=128); }
        }),
        ("validation preparation matches uncached packed targets and positions", () => {
            var rows=Corpus(); var sorted=rows.OrderBy(BatchPlanner.EffectiveLength).ToArray();
            var cache=new EvaluationBatchCache(rows,2,128);
            for(int i=0;i<cache.Count;i++)
            {
                var examples=sorted.Skip(i*2).Take(2).ToArray(); var expected=SupervisedBatch.Build(examples,examples.Max(BatchPlanner.EffectiveLength));
                var actual=cache.Get(i); Check(actual.Inputs.SequenceEqual(expected.Inputs) && actual.Positions.SequenceEqual(expected.Positions) && actual.Targets.SequenceEqual(expected.Targets));
            }
        }),
        ("validation preparation counts each target once including a short last batch", () => {
            var rows=Corpus(); var cache=new EvaluationBatchCache(rows,2,128);
            Check(cache.Count==2 && cache.Get(1).Rows==1);
            Check(Enumerable.Range(0,cache.Count).Sum(i=>cache.Get(i).Targets.Length)==rows.Sum(e=>e.Labels.Count(x=>x!=-100)));
        }),
        ("validation preparation trims only ignored trailing positions", () => {
            var row=new EncodedExample([1,2,0,0],[-100,2,-100,-100],"padding");
            var cache=new EvaluationBatchCache([row],1,16); var batch=cache.Get(0);
            Check(batch.Length==2 && batch.Targets.SequenceEqual(new long[]{2}) && batch.Positions.SequenceEqual(new long[]{1}));
        }),
        ("validation preparation rejects oversized supervised targets", () => {
            Throws<ArgumentException>(()=>new EvaluationBatchCache([new EncodedExample(new int[17],Enumerable.Repeat(1,17).ToArray(),"long")],1,16));
        }),
        ("validation preparation does not cache invalid native indexing inputs", () => {
            var cache=new EvaluationBatchCache([new EncodedExample([1,999],[-100,2],"bad")],1,16);
            Throws<ArgumentException>(()=>cache.Get(0)); Check(cache.Builds==0 && cache.RetainedPayloadBytes==0);
        }),
        ("validation preparation checks cancellation even on cache hits", () => {
            var cache=new EvaluationBatchCache(Corpus(),2,128); _=cache.Get(0);
            using var cts=new CancellationTokenSource();cts.Cancel();
            Throws<OperationCanceledException>(()=>cache.Get(0,cts.Token)); Check(cache.Hits==0);
        }),
        ("validation preparation cancellation before construction allocates no plan", () => {
            using var cts=new CancellationTokenSource();cts.Cancel();
            Throws<OperationCanceledException>(()=>new EvaluationBatchCache(Corpus(),2,128,ct:cts.Token));
        }),
        ("validation preparation refuses invalid bounds", () => {
            Throws<ArgumentException>(()=>new EvaluationBatchCache([],2,128));
            Throws<ArgumentException>(()=>new EvaluationBatchCache(Corpus(),0,128));
            Throws<ArgumentException>(()=>new EvaluationBatchCache(Corpus(),2,128,-1));
            var cache=new EvaluationBatchCache(Corpus(),2,128);Throws<ArgumentOutOfRangeException>(()=>cache.Get(-1));
        }),
        ("first conversation stage cannot omit all conversation material", () => {
            Throws<ArgumentException>(()=>LearningStages.ValidateMaterialRequest(TrainingMaterial.Conversation,false,0,false));
            LearningStages.ValidateMaterialRequest(TrainingMaterial.Conversation,true,0,false);
        }),
        ("repeated conversation training can retain previously learned corpus", () => {
            LearningStages.ValidateMaterialRequest(TrainingMaterial.Conversation,false,0,true);
        }),
        ("basic pretraining cannot silently consume custom file selections", () => {
            Throws<ArgumentException>(()=>LearningStages.ValidateMaterialRequest(TrainingMaterial.BasicPretrain,true,1,false));
            LearningStages.ValidateMaterialRequest(TrainingMaterial.BasicPretrain,true,0,false);
        }),
        ("custom stage needs new or previously trained custom material", () => {
            Throws<ArgumentException>(()=>LearningStages.ValidateMaterialRequest(TrainingMaterial.CustomWithReplay,true,0,false));
            LearningStages.ValidateMaterialRequest(TrainingMaterial.CustomWithReplay,false,1,false);
            LearningStages.ValidateMaterialRequest(TrainingMaterial.CustomWithReplay,false,0,false,true);
        }),
        ("stage material provenance roundtrips while legacy defaults remain unknown", () => {
            var settings=new WorkspaceSettings(ModelConfig.Small,new(),new(),TrainingMaterial.Conversation,true,true);
            var value=JsonSerializer.Deserialize<WorkspaceSettings>(JsonSerializer.Serialize(settings,JsonData.Options),JsonData.Options)!;
            Check(value.ConversationTrained==true && value.CustomDataTrained==true);
            Check(new WorkspaceSettings(ModelConfig.Small,new(),new()).ConversationTrained is null);
        }),
        ("stage preservation receipt distinguishes first copy from existing reference", () => Temp(root=> {
            string zero=Fixture(root,0); var first=StageArchive.PreserveWithReceipt(root,zero,"00-untrained","zero");
            var second=StageArchive.PreserveWithReceipt(root,Fixture(root,1),"00-untrained","later");
            Check(first.Created && !second.Created && first.Reference==second.Reference && second.Reference.Step==0);
        })),
        ("trained weights cannot become a new untrained reference", () => Temp(root=> {
            Throws<InvalidDataException>(()=>StageArchive.Preserve(root,Fixture(root,1),"00-untrained","wrong"));
            Check(!Directory.Exists(Path.Combine(StageArchive.Root(root),"00-untrained")));
        })),
        ("zero-step weights cannot become a completed basic or conversation reference", () => Temp(root=> {
            string zero=Fixture(root,0);
            foreach(string key in new[]{"01-basic","02-conversation","03-custom"}) Throws<InvalidDataException>(()=>StageArchive.Preserve(root,zero,key,"wrong"));
        })),
        ("same validation sequence keeps the best historical guard", () => {
            Check(ValidationBaseline.Anchor(5,Info(128,3),128)==3);
            Check(ValidationBaseline.Anchor(2,Info(128,3),128)==2);
        }),
        ("changed validation context discards only incomparable historical best", () => {
            Check(ValidationBaseline.Anchor(5,Info(64,1),128)==5);
            Check(ValidationBaseline.Anchor(5,Info(null,1),128)==5);
            Check(ValidationBaseline.Anchor(5,null,128)==5);
        }),
        ("invalid validation metrics cannot pass the guard", () => {
            Throws<ArgumentException>(()=>ValidationBaseline.Anchor(double.NaN,null,128));
            Throws<InvalidDataException>(()=>ValidationBaseline.Anchor(5,Info(128,double.PositiveInfinity),128));
        }),
        ("bundled manifest size is bounded before parsing", () => Temp(root=> {
            File.WriteAllText(Path.Combine(root,"DATASET_MANIFEST.json"),new string(' ',65537));
            Throws<InvalidDataException>(()=>BundledCorpus.Load(root,"seed.jsonl"));
        })),
        ("bundled import and direct dataset import honor pre-cancellation", () => {
            using var cts=new CancellationTokenSource();cts.Cancel();
            Throws<OperationCanceledException>(()=>BundledCorpus.Load("absent","seed.jsonl",cts.Token));
            Throws<OperationCanceledException>(()=>Dataset.LoadTraining("absent.jsonl",cts.Token));
        }),
        ("v8 held-out examples remain prohibited as training input", () => {
            Throws<InvalidDataException>(()=>Dataset.LoadTraining(Path.Combine(AppContext.BaseDirectory,"data","challenge-v8.jsonl")));
        }),
    ];
    private static CommandCompletion Parse(string json) { using var doc=JsonDocument.Parse(json); return CommandCompletion.Parse(doc.RootElement); }
    private static EncodedExample[] Corpus()=>Dataset.EncodeAll([Dataset.Make("third, longer question","yes"),Dataset.Make("q","a"),Dataset.Make("middle","answer")],128,1);
    private static RevisionInfo Info(int? sequence,double best)=>new(1,1,0,best,"fixture",DateTimeOffset.UtcNow,"","",null,"","","",0,best,ValidationSequenceLength:sequence);
    private static string Fixture(string root,long step)
    {
        string path=ModelFiles.GetRevisionPath(root,$"r{step:D16}");Directory.CreateDirectory(path);
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=64};
        ModelFiles.Write(Path.Combine(path,"model.tritmodel"),WeightSet.Initialize(c with {Seed=42+(int)step}),true);
        JsonData.AtomicWrite(Path.Combine(path,"revision.json"),new RevisionInfo(step,step,0,null,"fixture",DateTimeOffset.UtcNow,"",ModelFiles.Hash(Path.Combine(path,"model.tritmodel")),null,"","","",0,null));
        return path;
    }
    private static void Check(bool ok){if(!ok)throw new Exception("Audit8 assertion failed");}
    private static void Throws<T>(Action action) where T:Exception {try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
    private static void Temp(Action<string> action){string dir=Path.Combine(Path.GetTempPath(),"trit-audit8-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(dir);try{action(dir);}finally{Directory.Delete(dir,true);}}
}
