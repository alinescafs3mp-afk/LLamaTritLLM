using System.Text.Json;
using TritStudio.Core;

internal static class Audit7Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("untrained mode forces zero steps without changing supplied options", () => {
            var supplied=new TrainingOptions{Steps=123}; var actual=LearningStages.CreationOptions(CreationMode.Untrained,supplied);
            Check(actual.Steps==0 && supplied.Steps==123);
        }),
        ("basic and conversation stages retain the explicitly selected positive step count", () => {
            foreach(var mode in new[]{CreationMode.BasicPretrain,CreationMode.Conversation})
                Check(LearningStages.CreationOptions(mode,new TrainingOptions{Steps=3}).Steps==3);
            Throws(()=>LearningStages.CreationOptions(CreationMode.BasicPretrain,new TrainingOptions{Steps=0}));
        }),
        ("unknown stage enum values fail rather than choosing an implicit dataset", () => {
            Throws(()=>LearningStages.Validate((CreationMode)99)); Throws(()=>LearningStages.Validate((TrainingMaterial)99));
            Throws(()=>LearningStages.Name((TrainingMaterial)99)); Throws(()=>LearningStages.ArchiveKey((TrainingMaterial)99));
        }),
        ("legacy creation and training calls retain conversation defaults", () => {
            var create=new CreateRequest(ModelConfig.Small,new(),new(),[]); Check(create.Mode==CreationMode.Conversation);
            Check(new TrainRequest([],new()).Material==TrainingMaterial.Conversation);
            Check(LearningStages.CreationOptions(CreationMode.Conversation,new TrainingOptions{Steps=0}).Steps==0);
        }),
        ("old snapshot settings accept absent stage metadata", () => {
            var old=new {config=ModelConfig.Small,resources=new ResourceOptions(),training=new TrainingOptions()};
            var restored=JsonSerializer.Deserialize<WorkspaceSettings>(JsonSerializer.Serialize(old,JsonData.Options),JsonData.Options)!;
            Check(restored.LastMaterial is null);
        }),
        ("baseline is separately manifested text-only data", () => {
            string root=Path.Combine(AppContext.BaseDirectory,"data"); var baseline=BundledCorpus.Load(root,"pretrain.jsonl");
            var conversation=BundledCorpus.Load(root,"seed.jsonl").Select(x=>x.Id).ToHashSet();
            Check(baseline.Length>0 && baseline.All(x=>!x.IsDialogue && (x.History?.Length??0)==0 && !conversation.Contains(x.Id)));
        }),
        ("new held-out stage cannot be imported into optimizer training", () =>
            Throws(()=>Dataset.LoadTraining(Path.Combine(AppContext.BaseDirectory,"data","challenge-v7.jsonl")))),
        ("teaching context survives later list pruning", () => {
            var history=new List<ChatTurn>{new("before","answer",0,false,"c")};
            var target=new ChatTurn("question","response",1,false,"c",TeachingContext.Capture(history,1,"c"));
            history.Clear(); Check(Dataset.LearningContext(history,target).Single().User=="before");
        }),
        ("teaching capture never includes discarded generation history", () => {
            ChatTurn[] history=[new("old","a",0,false,"c"),new("seen","b",0,false,"c")];
            Check(TeachingContext.Capture(history,1,"c").Single().User=="seen");
            Check(TeachingContext.Capture(history,0,"c").Length==0);
        }),
        ("private history is a boundary not a gap to splice across", () => {
            ChatTurn[] history=[new("old","a",0,false,"c"),new("secret","b",0,true,"c"),new("recent","c",0,false,"c")];
            Check(TeachingContext.Capture(history,3,"c").Single().User=="recent");
        }),
        ("captured teaching history has no recursively nested snapshots", () => {
            var nested=new ChatTurn("u","a",0,false,"c",[new("hidden","x",0,false,"c")]);
            var captured=TeachingContext.Capture([nested],1,"c"); Check(captured.Single().LearningHistory is null);
            Throws(()=>TeachingContext.Capture([nested],2,"c"));
        }),
        ("captured teaching metadata roundtrips through the local journal schema", () => {
            var turn=new ChatTurn("q","a",0,false,"c",[new("old","answer",0,false,"c")]);
            var restored=JsonSerializer.Deserialize<ChatTurn>(JsonSerializer.Serialize(turn,JsonData.Options),JsonData.Options)!;
            Check(Dataset.LearningContext([],restored).Single().User=="old");
        }),
        ("captured context retains only its bounded most recent eight pairs", () => {
            var history=Enumerable.Range(0,20).Select(i=>new ChatTurn(i.ToString(),"a",0,false,"c")).ToArray();
            var result=TeachingContext.Capture(history,20,"c");Check(result.Length==8 && result[0].User=="12");
        }),
        ("stage archive preserves a verified model without copying training state", () => Temp(root=> {
            var source=Fixture(root,0); string file=StageArchive.Preserve(root,source,"00-untrained","untrained");
            Check(ModelFiles.Hash(file)==ModelFiles.Hash(Path.Combine(source,"model.tritmodel")));
            var info=JsonData.Read<StageReference>(Path.Combine(Path.GetDirectoryName(file)!,"stage.json"));
            Check(info.Step==0 && info.InferenceOnly && Directory.GetFiles(Path.GetDirectoryName(file)!).Length==2);
        })),
        ("stage archive is immutable on subsequent successful training", () => Temp(root=> {
            string first=StageArchive.Preserve(root,Fixture(root,0),"00-untrained","zero");var hash=ModelFiles.Hash(first);
            string again=StageArchive.Preserve(root,Fixture(root,1),"00-untrained","later");
            Check(first==again && ModelFiles.Hash(again)==hash);
        })),
        ("stage archive outlives pruning of its source training revision", () => Temp(root=> {
            string source=Fixture(root,0);string file=StageArchive.Preserve(root,source,"00-untrained","zero");
            Directory.Delete(source,true);Check(ModelFiles.Read(file,true).Config.Dimension==16);
        })),
        ("stage archive rejects corrupt source and removes only its new staging", () => Temp(root=> {
            string source=Fixture(root,0);File.AppendAllText(Path.Combine(source,"model.tritmodel"),"bad");
            Throws(()=>StageArchive.Preserve(root,source,"00-untrained","zero"));
            Check(Directory.GetDirectories(StageArchive.Root(root)).Length==0);
        })),
        ("stage archive detects existing reference corruption without overwriting", () => Temp(root=> {
            string source=Fixture(root,0);string file=StageArchive.Preserve(root,source,"00-untrained","zero");
            File.AppendAllText(file,"corruption");long size=new FileInfo(file).Length;
            Throws(()=>StageArchive.Preserve(root,source,"00-untrained","zero"));Check(new FileInfo(file).Length==size);
        })),
        ("stage archive refuses path traversal and observes pre-cancellation", () => Temp(root=> {
            Throws(()=>StageArchive.Preserve(root,"absent","../escape","bad"));
            using var cts=new CancellationTokenSource();cts.Cancel();Throws(()=>StageArchive.Preserve(root,"absent","00-untrained","zero",cts.Token));
            Check(!Directory.Exists(StageArchive.Root(root)));
        })),
        ("valid immutable sampling settings are not reallocated on each token", () => {
            var options=new SamplingOptions();Check(ReferenceEquals(options,options.Clamp()));
            var invalid=new SamplingOptions(double.NaN,0,4,0,2000);var fixedOptions=invalid.Clamp();
            Check(fixedOptions.Temperature==.7 && fixedOptions.TopK==1 && fixedOptions.TopP==1 && fixedOptions.RepetitionPenalty==1 && fixedOptions.MaxNewTokens==1024);
        }),
    ];
    private static string Fixture(string root,long revision)
    {
        string path=ModelFiles.GetRevisionPath(root,$"r{revision:D16}");Directory.CreateDirectory(path);
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=64,Seed=(int)(42+revision)};
        ModelFiles.Write(Path.Combine(path,"model.tritmodel"),WeightSet.Initialize(c),true);
        var info=new RevisionInfo(revision,revision,0,null,"fixture",DateTimeOffset.UtcNow,"",ModelFiles.Hash(Path.Combine(path,"model.tritmodel")),null,"","","",0,null);
        JsonData.AtomicWrite(Path.Combine(path,"revision.json"),info);return path;
    }
    private static void Check(bool condition){if(!condition)throw new Exception("Audit7 assertion failed");}
    private static void Throws(Action action){try{action();}catch{return;}throw new Exception("Expected failure");}
    private static void Temp(Action<string> action){string d=Path.Combine(Path.GetTempPath(),"trit-audit7-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(d);try{action(d);}finally{Directory.Delete(d,true);}}
}
