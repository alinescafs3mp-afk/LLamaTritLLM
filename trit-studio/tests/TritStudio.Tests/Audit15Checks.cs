using System.Text.Json;
using TritStudio.Core;
internal static class Audit15Checks
{
    public static (string Name, Action Run)[] All =>
    [
        ("update includes all owned application and trainer entrypoints", () => {
            foreach(string f in new[]{"TritStudio.exe","TritStudio.dll","TritStudio.Core.dll","trainer/TritStudio.Trainer.dll","trainer-cuda/TritStudio.Trainer.exe","runner/TritStudio.Runner.dll","checks/TritStudio.Tests.exe"}) Assert(UpdatePayloadPolicy.IsOwnedBinary(f),f);
        }),
        ("update contains own resolution metadata", () => {
            foreach(string f in new[]{"TritStudio.deps.json","TritStudio.runtimeconfig.json","trainer/TritStudio.Trainer.deps.json","trainer-cuda/TritStudio.Trainer.runtimeconfig.json"}) Assert(UpdatePayloadPolicy.IsOwnedBinary(f),f);
        }),
        ("update excludes actual native and managed vendor library names", () => {
            foreach(string f in new[]{"Avalonia.Controls.dll","System.Private.CoreLib.dll","hostfxr.dll","trainer/torch_cpu.dll","trainer-cuda/torch_cuda.dll","trainer-cuda/cublas64_12.dll","trainer-cuda/LibTorchSharp.dll","trainer/TorchSharp.dll","runtimes/win-x64/native/libSkiaSharp.dll"}) Assert(!UpdatePayloadPolicy.IsOwnedBinary(f),f);
        }),
        ("update rejects prefix spoofing and unknown subdirectories", () => {
            foreach(string f in new[]{"TritStudio.Evil.dll","TritStudio.Core.dll.bak","backup/TritStudio.exe","trainer/nested/TritStudio.Core.dll","other/TritStudio.Trainer.dll"}) Assert(!UpdatePayloadPolicy.IsOwnedBinary(f),f);
        }),
        ("update rejects path traversal and windows separators", () => {
            foreach(string f in new[]{"../TritStudio.exe","/TritStudio.exe","C:/TritStudio.exe","trainer\\TritStudio.dll","a//TritStudio.exe","./TritStudio.exe"}) Throws(()=>UpdatePayloadPolicy.IsOwnedBinary(f));
        }),
        ("update includes adjacent CPU and CUDA corpora", () => {
            foreach(string root in new[]{"trainer","trainer-cuda","checks"})
            foreach(string file in new[]{"seed.jsonl","validation.jsonl","pretrain.jsonl","test.jsonl","challenge.jsonl","challenge-v14.jsonl","challenge-v15.jsonl","DATASET_MANIFEST.json"})
                Assert(UpdatePayloadPolicy.IsOwnedPayload(root+"/data/"+file));
        }),
        ("update includes actual test numerical fixture", () => Assert(UpdatePayloadPolicy.IsOwnedPayload("checks/reference.json"))),
        ("runtime manifest excludes mutable app data", () => {
            string[] paths={"trainer/data/seed.jsonl","trainer-cuda/data/seed.jsonl","checks/data/DATASET_MANIFEST.json","checks/reference.json","trainer/TorchSharp.dll","trainer-cuda/torch_cuda.dll"};
            var vendor=paths.Where(p=>!UpdatePayloadPolicy.IsOwnedPayload(p)).ToArray();Assert(vendor.Length==2&&vendor.All(x=>x.EndsWith(".dll")));
        }),
        ("update rejects arbitrary data filenames and vendor names", () => {
            foreach(string f in new[]{"trainer/data/evil.dll","trainer/data/unknown.json","checks/data/TorchSharp.dll","trainer/data/challenge-v.jsonl","trainer/data/challenge-v+15.jsonl","trainer/data/challenge-v3.jsonl"}) Assert(!UpdatePayloadPolicy.IsOwnedPayload(f));
        }),
        ("update preserves folder-specific dependency copies", () => {
            foreach(string p in new[]{"trainer/LibTorchSharp.dll","trainer-cuda/LibTorchSharp.dll","checks/System.Runtime.dll","Avalonia.Base.dll"}) Assert(!UpdatePayloadPolicy.IsOwnedPayload(p));
        }),
        ("update admits later numbered challenges but no nested code", () => { Assert(UpdatePayloadPolicy.IsOwnedPayload("checks/data/challenge-v16.jsonl"));Assert(!UpdatePayloadPolicy.IsOwnedPayload("checks/data/child/challenge-v15.jsonl")); }),
        ("update content rejects rooted or traversing paths", () => { Throws(()=>UpdatePayloadPolicy.IsOwnedContent("../checks/reference.json"));Throws(()=>UpdatePayloadPolicy.IsOwnedContent("checks/data/../reference.json")); }),
        ("all bundled published data paths belong to update payload", () => {
            using var manifest=JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"data","DATASET_MANIFEST.json")));
            foreach(var entry in manifest.RootElement.GetProperty("files").EnumerateObject())
                foreach(string folder in new[]{"trainer","trainer-cuda","checks"}) Assert(UpdatePayloadPolicy.IsOwnedPayload(folder+"/data/"+entry.Name+".jsonl"));
        }),
        ("catalog empty scan does not create model storage", () => Temp(root=>{
            string models=Path.Combine(root,"models");Assert(ModelLibrary.Scan(models).Length==0);Assert(!Directory.Exists(models));
        })),
        ("catalog sees complete and interrupted creations", () => Temp(root=>{
            string models=Path.Combine(root,"models");Directory.CreateDirectory(Path.Combine(models,"unfinished"));Directory.CreateDirectory(Path.Combine(models,"ready"));File.WriteAllText(Path.Combine(models,"ready","active.json"),"{}");
            var entries=ModelLibrary.Scan(models);Assert(entries.Length==2);Assert(entries.Single(x=>x.Name=="ready").Problem is null);Assert(entries.Single(x=>x.Name=="unfinished").Problem is not null);
        })),
        ("catalog does not parse model weights or unbounded metadata", () => Temp(root=>{
            string model=Path.Combine(root,"models","broken");Directory.CreateDirectory(model);File.WriteAllText(Path.Combine(model,"active.json"),"broken");
            Assert(ModelLibrary.Scan(Path.Combine(root,"models")).Length==1);
        })),
        ("catalog includes deduplicated external exported models", () => Temp(root=>{
            string f=Path.Combine(root,"outside.tritmodel");File.WriteAllText(f,"fixture");var rows=ModelLibrary.Scan(Path.Combine(root,"models"),[f,f]);
            Assert(rows.Length==1&&!rows[0].Workspace&&!rows[0].Managed);
        })),
        ("catalog reports actual direct ownership only", () => Temp(root=>{
            string models=Path.Combine(root,"models");Assert(ModelLibrary.IsManaged(models,Path.Combine(models,"one")));Assert(!ModelLibrary.IsManaged(models,models));Assert(!ModelLibrary.IsManaged(models,Path.Combine(models,"one","nested")));Assert(!ModelLibrary.IsManaged(models,Path.Combine(root,"models-other","one")));
        })),
        ("catalog pre-cancel observes cancellation before directory traversal", () => { using var c=new CancellationTokenSource();c.Cancel();Throws(()=>ModelLibrary.Scan(Path.Combine(Path.GetTempPath(),"missing-models"),ct:c.Token)); }),
        ("path comparison canonicalizes trailing separator and dots", () => Temp(root=>{Assert(ModelLibrary.SamePath(root,Path.Combine(root,".")));Assert(!ModelLibrary.SamePath(root,null));})),
        ("model trash preserves weights logs and metadata", () => Temp(root=>{
            string models=Path.Combine(root,"models"),model=Path.Combine(models,"a"),trash=Path.Combine(root,"trash");Directory.CreateDirectory(model);File.WriteAllText(Path.Combine(model,"weights.bin"),"weights");File.WriteAllText(Path.Combine(model,"chat.jsonl"),"conversation");
            var moved=ModelLibrary.MoveToTrash(models,model,trash);Assert(!Directory.Exists(model));Assert(File.ReadAllText(Path.Combine(moved.Directory,"model","weights.bin"))=="weights");Assert(File.ReadAllText(Path.Combine(moved.Directory,"model","chat.jsonl"))=="conversation");Assert(JsonData.Read<ModelTrashReceipt>(Path.Combine(moved.Directory,"restore.json")).OriginalPath==model);
        })),
        ("model trash refuses a live UI lease", () => Temp(root=>{
            string models=Path.Combine(root,"models"),model=Path.Combine(models,"a");Directory.CreateDirectory(model);using var lease=new FileStream(Path.Combine(model,".ui.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);
            Throws(()=>ModelLibrary.MoveToTrash(models,model,Path.Combine(root,"trash")));Assert(Directory.Exists(model));
        })),
        ("model trash refuses a live trainer lease", () => Temp(root=>{
            string models=Path.Combine(root,"models"),model=Path.Combine(models,"a");Directory.CreateDirectory(model);using var lease=new FileStream(Path.Combine(model,".trainer.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);
            Throws(()=>ModelLibrary.MoveToTrash(models,model,Path.Combine(root,"trash")));Assert(Directory.Exists(model));
        })),
        ("model trash refuses external or nested directory", () => Temp(root=>{
            string models=Path.Combine(root,"models"),nested=Path.Combine(models,"a","nested"),external=Path.Combine(root,"external");Directory.CreateDirectory(nested);Directory.CreateDirectory(external);
            Throws(()=>ModelLibrary.MoveToTrash(models,nested,Path.Combine(root,"trash")));Throws(()=>ModelLibrary.MoveToTrash(models,external,Path.Combine(root,"trash")));Assert(Directory.Exists(external)&&Directory.Exists(nested));
        })),
        ("model trash refuses the model root itself", () => Temp(root=>{string models=Path.Combine(root,"models");Directory.CreateDirectory(models);Throws(()=>ModelLibrary.MoveToTrash(models,models,Path.Combine(root,"trash")));})),
        ("model trash refuses recursive destination", () => Temp(root=>{string models=Path.Combine(root,"models"),model=Path.Combine(models,"a");Directory.CreateDirectory(model);Throws(()=>ModelLibrary.MoveToTrash(models,model,Path.Combine(model,"trash")));Assert(Directory.Exists(model));})),
        ("trash must not contaminate another catalog entry", () => Temp(root=>{string models=Path.Combine(root,"models"),model=Path.Combine(models,"a");Directory.CreateDirectory(model);Throws(()=>ModelLibrary.MoveToTrash(models,model,Path.Combine(models,"other","trash")));Assert(Directory.Exists(model));})),
        ("model trash pre-cancel keeps source and creates no receipt", () => Temp(root=>{string models=Path.Combine(root,"models"),model=Path.Combine(models,"a"),trash=Path.Combine(root,"trash");Directory.CreateDirectory(model);using var c=new CancellationTokenSource();c.Cancel();Throws(()=>ModelLibrary.MoveToTrash(models,model,trash,c.Token));Assert(Directory.Exists(model)&&!Directory.Exists(trash));})),
        ("model trash never overwrites a previous same-name removal", () => Temp(root=>{string models=Path.Combine(root,"models"),model=Path.Combine(models,"a"),trash=Path.Combine(root,"trash");Directory.CreateDirectory(model);var a=ModelLibrary.MoveToTrash(models,model,trash);Directory.CreateDirectory(model);var b=ModelLibrary.MoveToTrash(models,model,trash);Assert(a.Directory!=b.Directory&&Directory.Exists(a.Directory)&&Directory.Exists(b.Directory));})),
        ("trash refuses symlink roots without touching target", () => Temp(root=>{
            if(OperatingSystem.IsWindows())return; // Mandatory real Windows junction check is in the desktop checklist.
            string models=Path.Combine(root,"models"),outside=Path.Combine(root,"outside");Directory.CreateDirectory(models);Directory.CreateDirectory(outside);Directory.CreateSymbolicLink(Path.Combine(models,"link"),outside);
            Throws(()=>ModelLibrary.MoveToTrash(models,Path.Combine(models,"link"),Path.Combine(root,"trash")));Assert(Directory.Exists(outside));
        })),
        ("trash refuses linked lease files without touching target", () => Temp(root=>{
            if(OperatingSystem.IsWindows())return;
            string models=Path.Combine(root,"models"),model=Path.Combine(models,"a"),outside=Path.Combine(root,"outside.lock");Directory.CreateDirectory(model);File.WriteAllText(outside,"untouched");File.CreateSymbolicLink(Path.Combine(model,".ui.lock"),outside);
            Throws(()=>ModelLibrary.MoveToTrash(models,model,Path.Combine(root,"trash")));Assert(Directory.Exists(model)&&File.ReadAllText(outside)=="untouched");
        })),
        ("catalog marks linked entries non-removable", () => Temp(root=>{
            if(OperatingSystem.IsWindows())return;
            string models=Path.Combine(root,"models"),outside=Path.Combine(root,"outside");Directory.CreateDirectory(models);Directory.CreateDirectory(outside);Directory.CreateSymbolicLink(Path.Combine(models,"link"),outside);
            var item=ModelLibrary.Scan(models).Single();Assert(!item.Managed&&item.Problem is not null);
        })),
        ("chat clear persists a fresh conversation only", () => Temp(root=>{
            string old=Guid.NewGuid().ToString("N");JsonData.AtomicWrite(Path.Combine(root,"chat-state.json"),old);
            foreach(string name in new[]{"chat.jsonl","active.json","runtime.json","replay.json"})File.WriteAllText(Path.Combine(root,name),"untouched");
            string fresh=ChatSessionState.StartNew(root);Assert(fresh!=old&&Guid.TryParseExact(fresh,"N",out _));
            Assert(JsonData.Read<string>(Path.Combine(root,"chat-state.json"))==fresh);
            foreach(string name in new[]{"chat.jsonl","active.json","runtime.json","replay.json"})Assert(File.ReadAllText(Path.Combine(root,name))=="untouched");
        })),
        ("chat clear memory-only session creates unique IDs", () => {Assert(ChatSessionState.StartNew(null)!=ChatSessionState.StartNew(null));}),
        ("chat clear missing workspace creates nothing", () => Temp(root=>{string p=Path.Combine(root,"absent");Throws(()=>ChatSessionState.StartNew(p));Assert(!Directory.Exists(p));})),
        ("chat clear pre-cancel leaves ID unchanged", () => Temp(root=>{string path=Path.Combine(root,"chat-state.json");JsonData.AtomicWrite(path,"original");using var ct=new CancellationTokenSource();ct.Cancel();Throws(()=>ChatSessionState.StartNew(root,ct.Token));Assert(JsonData.Read<string>(path)=="original");})),
        ("chat clear failed state commit leaves journal intact", () => Temp(root=>{Directory.CreateDirectory(Path.Combine(root,"chat-state.json"));File.WriteAllText(Path.Combine(root,"chat.jsonl"),"original");Throws(()=>ChatSessionState.StartNew(root));Assert(File.ReadAllText(Path.Combine(root,"chat.jsonl"))=="original");})),
        ("chat clear refuses linked state without touching target", () => Temp(root=>{
            if(OperatingSystem.IsWindows())return;
            string w=Path.Combine(root,"workspace"),f=Path.Combine(root,"outside.json");Directory.CreateDirectory(w);File.WriteAllText(f,"original");File.CreateSymbolicLink(Path.Combine(w,"chat-state.json"),f);Throws(()=>ChatSessionState.StartNew(w));Assert(File.ReadAllText(f)=="original");
        })),
        ("learning stage and vocabulary contract unchanged by compact UI", () => { Assert(ByteTokenizer.VocabularySize==262);Assert(new ModelConfig().FormatVersion==1);Assert(new TrainingOptions{PublishEvery=1000}.Steps==1200); }),
        ("publish interval validates supported boundaries", () => {new TrainingOptions{PublishEvery=1}.Validate();new TrainingOptions{PublishEvery=1000}.Validate();Throws(()=>new TrainingOptions{PublishEvery=0}.Validate());new TrainingOptions{PublishEvery=100000}.Validate();Throws(()=>new TrainingOptions{PublishEvery=100001}.Validate());}),
    ];
    private static void Assert(bool ok,string? why=null){if(!ok)throw new Exception(why??"Assertion failed");}
    private static void Throws(Action action){try{action();}catch{return;}throw new Exception("Expected refusal");}
    private static void Temp(Action<string> action){string p=Path.Combine(Path.GetTempPath(),"trit-audit15-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(p);try{action(p);}finally{Directory.Delete(p,true);}}
}
