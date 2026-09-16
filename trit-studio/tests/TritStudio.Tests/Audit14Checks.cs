using System.Text.Json;
using TritStudio.Core;

public static class Audit14Checks
{
    public static IEnumerable<(string Name, Action Run)> All => new (string Name, Action Run)[]
    {
        ("vector RMSNorm matches the prior scalar channel order including tails", () => {
            foreach (int n in new[]{1,3,8,15,16,31,32,64,127,256,512})
            { var x=Data(n); var gamma=Data(n,1.2f); var expected=Norm(x,gamma);var actual=new float[n];CpuElementwise.RmsNorm(x,gamma,actual);Close(expected,actual); }
        }),
        ("vector residual addition agrees with scalar channels", () => {
            foreach(int n in new[]{0,1,3,16,31,128,512}) { var a=Data(n);var b=Data(n,.3f);var expected=a.Zip(b,(x,y)=>x+y).ToArray();CpuElementwise.AddInPlace(a,b);Close(expected,a); }
        }),
        ("RMSNorm permits exact in-place input alias", () => {
            var x=Data(37);var g=Data(37,1f);var expected=Norm(x,g);CpuElementwise.RmsNorm(x,g,x);Close(expected,x);
        }),
        ("RMSNorm permits output gamma and all-array aliases", () => {
            var x=Data(37);var g=Data(37,1f);var expected=Norm(x,g);CpuElementwise.RmsNorm(x,g,g);Close(expected,g);
            x=Data(37);expected=Norm(x,x);CpuElementwise.RmsNorm(x,x,x);Close(expected,x);
        }),
        ("residual self-alias doubles each channel", () => {
            var x=Data(37);var expected=x.Select(v=>v+v).ToArray();CpuElementwise.AddInPlace(x,x);Close(expected,x);
        }),
        ("RMSNorm bad dimensions leave output unchanged", () => {
            var output=Enumerable.Repeat(19f,16).ToArray();Throws<ArgumentException>(()=>CpuElementwise.RmsNorm(new float[16],new float[15],output));Assert(output.All(x=>x==19));
        }),
        ("residual bad dimensions leave target unchanged", () => {
            var x=Data(17);var before=(float[])x.Clone();Throws<ArgumentException>(()=>CpuElementwise.AddInPlace(x,new float[16]));Close(before,x);
        }),
        ("elementwise pre-cancellation precedes all writes", () => {
            using var stop=new CancellationTokenSource();stop.Cancel();var x=Data(32);var before=(float[])x.Clone();
            Throws<OperationCanceledException>(()=>CpuElementwise.AddInPlace(x,x,stop.Token));Throws<OperationCanceledException>(()=>CpuElementwise.RmsNorm(x,x,x,stop.Token));Close(before,x);
        }),
        ("RMSNorm rejects nonfinite reduction before output writes", () => {
            foreach(float bad in new[]{float.NaN,float.PositiveInfinity,float.MaxValue}) { var x=Data(16);x[3]=bad;var output=Enumerable.Repeat(11f,16).ToArray();Throws<ArithmeticException>(()=>CpuElementwise.RmsNorm(x,Data(16,1f),output));Assert(output.All(v=>v==11)); }
        }),
        ("zero RMSNorm remains exactly zero", () => { var output=new float[33];CpuElementwise.RmsNorm(new float[33],Data(33,1),output);Assert(output.All(x=>x==0)); }),
        ("workspace preview is read-only for an empty destination", () => Temp(root=>{
            var preview=WorkspacePreview.Load(root);Assert(preview.NeedsConversationState&&preview.Weights is null&&preview.History.Turns.Length==0);
            Assert(!Directory.EnumerateFileSystemEntries(root).Any());
        })),
        ("workspace preview preserves a saved conversation identity", () => Temp(root=>{
            string file=Path.Combine(root,"chat-state.json");JsonData.AtomicWrite(file,"conversation-14");var before=File.ReadAllBytes(file);
            var preview=WorkspacePreview.Load(root);Assert(!preview.NeedsConversationState&&preview.ConversationId=="conversation-14");Assert(before.SequenceEqual(File.ReadAllBytes(file)));
        })),
        ("workspace preview rejects malformed conversation state without rewriting", () => Temp(root=>{
            string file=Path.Combine(root,"chat-state.json");File.WriteAllText(file,"{");Throws<JsonException>(()=>WorkspacePreview.Load(root));Assert(File.ReadAllText(file)=="{");
        })),
        ("workspace preview rejects empty or oversized conversation ids", () => Temp(root=>{
            foreach(string id in new[]{"", " ",new string('a',257)}) { JsonData.AtomicWrite(Path.Combine(root,"chat-state.json"),id);Throws<InvalidDataException>(()=>WorkspacePreview.Load(root)); }
        })),
        ("workspace preview bounds conversation state before parsing", () => Temp(root=>{
            File.WriteAllText(Path.Combine(root,"chat-state.json"),new string(' ',4097));Throws<InvalidDataException>(()=>WorkspacePreview.Load(root));
        })),
        ("workspace preview refuses invalid active path without creating chat state", () => Temp(root=>{
            JsonData.AtomicWrite(Path.Combine(root,"active.json"),new ActiveRevision("../elsewhere"));Throws<InvalidDataException>(()=>WorkspacePreview.Load(root));Assert(!File.Exists(Path.Combine(root,"chat-state.json")));
        })),
        ("workspace preview loads verified model and finite history once", () => Temp(root=>{
            var info=CreateSnapshot(root);JsonData.AtomicWrite(Path.Combine(root,"chat-state.json"),"c14");
            ChatJournal.AppendAsync(Path.Combine(root,"chat.jsonl"),new("question","answer",info.Revision,false,"c14")).GetAwaiter().GetResult();
            var preview=WorkspacePreview.Load(root);Assert(preview.Weights is not null&&preview.Revision==info&&preview.History.Turns.Length==1);
            Assert(preview.InferenceFile==Path.Combine(ModelFiles.ActivePath(root)!,"model.tritmodel"));
        })),
        ("workspace preview rejects checksum mismatch without rewriting pointer", () => Temp(root=>{
            CreateSnapshot(root);string pointer=Path.Combine(root,"active.json");var before=File.ReadAllBytes(pointer);
            File.AppendAllText(Path.Combine(ModelFiles.ActivePath(root)!,"model.tritmodel"),"broken");Throws<InvalidDataException>(()=>WorkspacePreview.Load(root));Assert(before.SequenceEqual(File.ReadAllBytes(pointer)));
        })),
        ("workspace preview reports damaged tail but retains valid older exchange", () => Temp(root=>{
            ChatJournal.AppendAsync(Path.Combine(root,"chat.jsonl"),new("q","a",0)).GetAwaiter().GetResult();File.AppendAllText(Path.Combine(root,"chat.jsonl"),"{broken");
            var preview=WorkspacePreview.Load(root);Assert(preview.History.Turns.Length==1&&preview.History.SkippedRecords==1);
        })),
        ("cancelled preview and model reads do not access missing paths", () => {
            using var stop=new CancellationTokenSource();stop.Cancel();string path=Path.Combine(Path.GetTempPath(),Guid.NewGuid().ToString("N"));
            Throws<OperationCanceledException>(()=>WorkspacePreview.Load(path,ct:stop.Token));
            Throws<OperationCanceledException>(()=>ModelFiles.Read(path,true,ct:stop.Token));
            Throws<OperationCanceledException>(()=>ModelFiles.Hash(path,stop.Token));
            Throws<OperationCanceledException>(()=>ModelFiles.ReadInferenceRevision(path,ct:stop.Token));
        }),
    };
    public static RevisionInfo CreateSnapshot(string root)
    {
        string name="r0000000000000000",dir=ModelFiles.GetRevisionPath(root,name);Directory.CreateDirectory(dir);
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=64};
        string model=Path.Combine(dir,"model.tritmodel");ModelFiles.Write(model,WeightSet.Initialize(c),true);
        var info=new RevisionInfo(0,0,0,null,"preview test",DateTimeOffset.UtcNow,"",ModelFiles.Hash(model),null,"","","",0,null);
        JsonData.AtomicWrite(Path.Combine(dir,"revision.json"),info);JsonData.AtomicWrite(Path.Combine(root,"active.json"),new ActiveRevision(name));return info;
    }
    private static float[] Data(int n,float offset=0)=>Enumerable.Range(0,n).Select(i=>MathF.Sin(i+.5f)+offset).ToArray();
    private static float[] Norm(float[] x,float[] gamma)
    {
        float inv=1/MathF.Sqrt(ManagedInference.Session.Dot(x,0,x,0,x.Length)/x.Length+1e-5f);return x.Select((v,i)=>v*inv*gamma[i]).ToArray();
    }
    private static void Close(float[] a,float[] b) { Assert(a.Length==b.Length);for(int i=0;i<a.Length;i++)Assert(MathF.Abs(a[i]-b[i])<=2e-6f*MathF.Max(1,MathF.Abs(a[i])),"Elementwise value mismatch"); }
    private static void Assert(bool ok,string text="Audit14 contract failed") { if(!ok)throw new Exception(text); }
    private static void Throws<T>(Action action) where T:Exception { try{action();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name); }
    private static void Temp(Action<string> action) { string root=Path.Combine(Path.GetTempPath(),"trit-a14-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(root);try{action(root);}finally{Directory.Delete(root,true);} }
}
