using System.Globalization;
using TritStudio.Core;

internal static class Audit18Checks
{
    public static (string, Action)[] All =>
    [
        ("numeric launch input accepts both decimal separators independent of locale", () => {
            var prior = CultureInfo.CurrentCulture;
            try { foreach (string culture in new[]{"ru-RU","en-US"}) {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
                Check(ParameterNumberText.Parse("0,0003",0,1)==.0003m);
                Check(ParameterNumberText.Parse("0.0003",0,1)==.0003m);
                Check(ParameterNumberText.Parse(" 3e-4 ",0,1)==.0003m);
            }} finally { CultureInfo.CurrentCulture = prior; }
        }),
        ("launch numeric parser rejects ambiguous grouping and malformed input", () => {
            foreach(string s in new[]{""," ","1,024.0","1.024,0","1 024","NaN","Infinity","1e","1,2,3","--1",new string('1',97)})
                Throws<ArgumentException>(()=>ParameterNumberText.Parse(s,0,100000));
        }),
        ("launch number bounds and integer fields are enforced", () => {
            Check(ParameterNumberText.Parse("64",1,64,true)==64);
            Check(ParameterNumberText.Parse("6.4e1",1,64,true)==64);
            Throws<ArgumentException>(()=>ParameterNumberText.Parse("64.1",1,64,true));
            Throws<ArgumentException>(()=>ParameterNumberText.Parse("1.5",1,64,true));
            Throws<ArgumentException>(()=>ParameterNumberText.Parse("65",1,64,true));
            Throws<ArgumentException>(()=>ParameterNumberText.Parse("0",1,64,true));
        }),
        ("next-run settings roundtrip retains custom values and flags", () => Temp(root => {
            var d=Draft();LaunchDraftStore.Write(root,d);Check(LaunchDraftStore.Read(root,d.Config)==d);
            Check(!File.Exists(Path.Combine(root,"settings.json"))&&!File.Exists(Path.Combine(root,"active.json")));
        })),
        ("two models never share launch drafts", () => Temp(root => {
            string a=Path.Combine(root,"a"),b=Path.Combine(root,"b");Directory.CreateDirectory(a);Directory.CreateDirectory(b);
            var d=Draft();var other=d with{Training=d.Training with{Steps=123,LearningRate=.0007},Resources=d.Resources with{BatchSize=32}};
            LaunchDraftStore.Write(a,d);LaunchDraftStore.Write(b,other);
            Check(LaunchDraftStore.Read(a,d.Config)==d&&LaunchDraftStore.Read(b,d.Config)==other);
        })),
        ("missing draft read does not create files", () => Temp(root => {
            Check(LaunchDraftStore.Read(root,ModelConfig.Small) is null);Check(!Directory.EnumerateFileSystemEntries(root).Any());
        })),
        ("draft configuration mismatch is refused without changing source", () => Temp(root => {
            var d=Draft();LaunchDraftStore.Write(root,d);byte[] before=File.ReadAllBytes(LaunchDraftStore.FilePath(root));
            Throws<InvalidDataException>(()=>LaunchDraftStore.Read(root,d.Config with{Layers=3}));
            Check(before.SequenceEqual(File.ReadAllBytes(LaunchDraftStore.FilePath(root))));
        })),
        ("malformed draft is retained for reviewed recovery", () => Temp(root => {
            string p=LaunchDraftStore.FilePath(root);File.WriteAllText(p,"{");
            Throws<Exception>(()=>LaunchDraftStore.Read(root,ModelConfig.Small));Check(File.ReadAllText(p)=="{");
        })),
        ("draft write validates before replacing a good file", () => Temp(root => {
            var d=Draft();LaunchDraftStore.Write(root,d);byte[] bytes=File.ReadAllBytes(LaunchDraftStore.FilePath(root));
            Throws<ArgumentException>(()=>LaunchDraftStore.Write(root,d with{Resources=d.Resources with{BatchSize=65}}));
            Throws<ArgumentException>(()=>LaunchDraftStore.Write(root,d with{Training=d.Training with{LearningRate=double.NaN}}));
            Check(bytes.SequenceEqual(File.ReadAllBytes(LaunchDraftStore.FilePath(root))));
        })),
        ("draft directory collision is visible not treated as absent", () => Temp(root => {
            Directory.CreateDirectory(LaunchDraftStore.FilePath(root));
            Throws<IOException>(()=>LaunchDraftStore.Read(root,ModelConfig.Small));
            Throws<IOException>(()=>LaunchDraftStore.Write(root,Draft()));
        })),
        ("oversized draft refuses bounded read", () => Temp(root => {
            File.WriteAllText(LaunchDraftStore.FilePath(root),new string(' ',LaunchDraftStore.MaxBytes+1));
            Throws<Exception>(()=>LaunchDraftStore.Read(root,ModelConfig.Small));
        })),
        ("draft file format and stage are checked", () => {
            Throws<InvalidDataException>(()=>(Draft() with{Version=2}).Validate());
            Throws<ArgumentException>(()=>(Draft() with{Material=(TrainingMaterial)999}).Validate());
        }),
        ("separate creation draft preserves architecture and future launch values", () => Temp(root => {
            var d=Draft();var c=new CreationDraft("Тест",ModelConfig.Large,d.Resources with{SequenceLength=2048},d.Training,CreationMode.Untrained);
            string p=Path.Combine(root,"creation-draft.json");LaunchDraftStore.WriteCreation(p,c);
            Check(LaunchDraftStore.ReadCreation(p)==c);Check(!File.Exists(LaunchDraftStore.FilePath(root)));
        })),
        ("creation draft does not run training-memory preflight", () => {
            var c=new CreationDraft("large",ModelConfig.Large,new ResourceOptions{BatchSize=64,SequenceLength=1024,MemoryMiB=16384},new TrainingOptions());
            c.Validate(); // Legal next-run intent, even if later CPU allocations need a smaller actual batch.
            Throws<ArgumentException>(()=>c.Resources.ValidateForTraining(c.Config,1024,false));
        }),
        ("batch64 passes all configuration and initialization checks", () => {
            var r=Draft().Resources;r.ValidateInitialization(ModelConfig.Small);
            Check(ResourceOptions.MaxBatchSize==64);r.ValidateForTraining(ModelConfig.Small,64,false);
        }),
        ("batch65 fails every configuration entry point", () => {
            var r=Draft().Resources with{BatchSize=65};
            Throws<ArgumentException>(()=>r.ValidateConfiguration(ModelConfig.Small));
            Throws<ArgumentException>(()=>r.ValidateInitialization(ModelConfig.Small));
            Throws<ArgumentException>(()=>TrainingMemoryEstimate.For(ModelConfig.Small,r,0));
        }),
        ("batch64 sampler and supervised assembly cover every row", () => {
            var examples=Examples(3);var plan=new BatchPlanner(examples).Select(64,new SamplerRandom(7));
            var b=SupervisedBatch.Build(plan.Examples,plan.Length);
            Check(b.Rows==64&&b.Inputs.Length==64*plan.Length&&b.Targets.LongLength==plan.TargetTokens);
            Check(b.Positions.All(x=>x>=0&&x<b.Inputs.LongLength));
            Check(b.Positions.Select(x=>x/b.Length).Distinct().Count()==64);
        }),
        ("batch65 is refused by sampler and assembly", () => {
            var e=Examples(65);Throws<ArgumentException>(()=>new BatchPlanner(e).Select(65,new SamplerRandom(1)));
            Throws<ArgumentException>(()=>SupervisedBatch.Build(e,64));
        }),
        ("validation64 retains final partial batch without dropping targets", () => {
            var e=Examples(65);var cache=new EvaluationBatchCache(e,64,64);
            Check(cache.Count==2&&cache.Get(0).Rows==64&&cache.Get(1).Rows==1);
            Check(cache.Get(0).Targets.LongLength+cache.Get(1).Targets.LongLength==e.Sum(x=>(long)x.Labels.Count(v=>v!=-100)));
        }),
        ("validation65 limit fails early", () => Throws<ArgumentException>(()=>new EvaluationBatchCache(Examples(2),65,64))),
        ("64-batch initialization does not invent training activations", () => {
            var r=Draft().Resources with{SequenceLength=1024};var a=TrainingMemoryEstimate.For(ModelConfig.Large,r,0);
            var b=TrainingMemoryEstimate.For(ModelConfig.Large,r with{BatchSize=1},0);
            Check(a==b&&a.ActivationBytes==0);r.ValidateInitialization(ModelConfig.Large);
        }),
        ("new challenge is held-out even after manifest expansion", () => {
            string file=Path.Combine(AppContext.BaseDirectory,"data","challenge-v18.jsonl");
            Check(Dataset.Load(file).Length>0);Throws<InvalidDataException>(()=>Dataset.LoadTraining(file));
        }),
    ];
    private static LaunchDraft Draft()=>new(ModelConfig.Small,
        new ResourceOptions{Threads=1,MemoryMiB=16384,BatchSize=64,SequenceLength=512,PreferCuda=false,UseSdpa=false,BucketByLength=false,ProjectOnlyTargets=false},
        new TrainingOptions{Steps=731,LearningRate=.0003,PublishEvery=137},TrainingMaterial.Conversation,false);
    private static EncodedExample[] Examples(int n)=>Enumerable.Range(0,n).Select(i=>Dataset.Encode(Dataset.Make("номер "+i,"да"),64)).ToArray();
    private static void Temp(Action<string> test) {string p=Path.Combine(Path.GetTempPath(),"trit-a18-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(p);try{test(p);}finally{Directory.Delete(p,true);}}
    private static void Check(bool ok){if(!ok)throw new Exception("Audit18 contract failed");}
    private static void Throws<T>(Action f) where T:Exception{try{f();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
}
