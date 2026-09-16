using TritStudio.Core;

internal static class Audit16Checks
{
    public static (string, Action)[] All =>
    [
        ("prefix last-block skip matches full sessions: one layer", () => Prefix(1,1)),
        ("prefix last-block skip matches full sessions: GQA", () => Prefix(2,1)),
        ("prefix last-block skip matches full sessions: full KV", () => Prefix(3,2)),
        ("prefix cancellation leaves position unchanged", () => {
            var s=Model().NewSession();using var c=new CancellationTokenSource();c.Cancel();
            Throws<OperationCanceledException>(()=>s.Step(1,c.Token,false));Check(s.Position==0&&s.PrefixFinalBlockSkips==0);
        }),
        ("prefix invalid token leaves position unchanged", () => {
            var s=Model().NewSession();Throws<ArgumentOutOfRangeException>(()=>s.Step(999,computeLogits:false));Check(s.Position==0);
        }),
        ("logit-producing steps do not skip blocks", () => {
            var s=Model().NewSession();s.Step(1);s.Step(10);Check(s.Position==2&&s.PrefixFinalBlockSkips==0);
        }),
        ("missing active pointer with revisions is refused", () => Temp(p=>{
            Directory.CreateDirectory(Path.Combine(p,"revisions","r0000000000000000"));
            Throws<System.IO.InvalidDataException>(()=>ModelFiles.ActivePath(p));
            Check(!File.Exists(Path.Combine(p,"active.json")));
        })),
        ("fresh empty workspace remains valid", () => Temp(p=>Check(ModelFiles.ActivePath(p) is null))),
        ("staging-only workspace is not a committed revision", () => Temp(p=>{
            Directory.CreateDirectory(Path.Combine(p,"revisions",".stage-incomplete"));Check(ModelFiles.ActivePath(p) is null);
        })),
        ("active pointer directory is not an empty model", () => Temp(p=>{
            Directory.CreateDirectory(Path.Combine(p,"active.json"));Throws<InvalidDataException>(()=>WorkspacePreview.Load(p));
        })),
        ("conversation state directory fails preview", () => Temp(p=>{
            Directory.CreateDirectory(Path.Combine(p,"chat-state.json"));Throws<InvalidDataException>(()=>WorkspacePreview.Load(p));
        })),
        ("active pointer size budget applies before parse", () => Temp(p=>{
            File.WriteAllText(Path.Combine(p,"active.json"),new string(' ',4097));Throws<InvalidDataException>(()=>ModelFiles.ActivePath(p));
        })),
        ("null revision identifier is explicit invalid data", () => Throws<InvalidDataException>(()=>ModelFiles.GetRevisionPath(".",null!))),
        ("revision metadata size budget applies before parse", () => Temp(p=>{
            File.WriteAllText(Path.Combine(p,"revision.json"),new string(' ',65537));Throws<InvalidDataException>(()=>ModelFiles.ReadRevisionInfo(p));
        })),
        ("file instead of trash rejects before touching model", () => Temp(p=>{
            string models=Path.Combine(p,"models"),m=Path.Combine(models,"one"),trash=Path.Combine(p,"trash");
            Directory.CreateDirectory(m);File.WriteAllText(trash,"keep");File.WriteAllText(Path.Combine(m,"weights"),"keep");
            Throws<IOException>(()=>ModelLibrary.MoveToTrash(models,m,trash));Check(Directory.Exists(m)&&File.ReadAllText(trash)=="keep");
        })),
        ("linked active pointer is not followed", () => Temp(p=>{
            if(OperatingSystem.IsWindows())return;string outside=Path.Combine(p,"outside");File.WriteAllText(outside,"{}");
            File.CreateSymbolicLink(Path.Combine(p,"active.json"),outside);Throws<IOException>(()=>ModelFiles.ActivePath(p));
        })),
        ("linked revision directory is not read", () => Temp(p=>{
            if(OperatingSystem.IsWindows())return;string outside=Path.Combine(p,"outside");Directory.CreateDirectory(outside);
            string link=Path.Combine(p,"r0000000000000001");Directory.CreateSymbolicLink(link,outside);Throws<IOException>(()=>ModelFiles.ReadRevisionInfo(link));
        })),
        ("preview rejection does not create state", () => Temp(p=>{
            Directory.CreateDirectory(Path.Combine(p,"revisions","r0000000000000000"));Throws<InvalidDataException>(()=>WorkspacePreview.Load(p));
            Check(!File.Exists(Path.Combine(p,"chat-state.json")));
        })),
    ];
    private static ManagedInference Model(int layers=1,int kv=1)
    {
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Heads=2,KvHeads=kv,Layers=layers,Context=64,GroupSize=8};
        return new(TernaryQuantizer.Quantize(WeightSet.Initialize(cfg)),0);
    }
    private static void Prefix(int layers,int kv)
    {
        foreach(int n in new[]{1,2,7,23})
        {
            var m=Model(layers,kv);var reference=m.NewSession(optimizePrefix:false);var fast=m.NewSession();
            float[] left=[],right=[];
            for(int i=0;i<n;i++)
            {
                int token=i==0?ByteTokenizer.Bos:ByteTokenizer.Offset+(i*37)%256;
                left=reference.Step(token,computeLogits:i==n-1);right=fast.Step(token,computeLogits:i==n-1);
            }
            Equal(left,right);Check(fast.PrefixFinalBlockSkips==n-1&&reference.PrefixFinalBlockSkips==0);
            for(int i=0;i<4;i++)Equal(reference.Step(30+i),fast.Step(30+i));
            Check(reference.Position==fast.Position&&fast.Position==n+4);
        }
    }
    private static void Equal(float[] a,float[] b)
    {Check(a.Length==b.Length);for(int i=0;i<a.Length;i++)Check(float.IsFinite(a[i])&&float.IsFinite(b[i])&&Math.Abs(a[i]-b[i])<=2e-5f,"Prefix logit mismatch");}
    private static void Check(bool ok,string message="Assertion failed"){if(!ok)throw new Exception(message);}
    private static void Throws<T>(Action a) where T:Exception
    {try{a();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
    private static void Temp(Action<string> a)
    {string p=Path.Combine(Path.GetTempPath(),"trit-a16-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(p);try{a(p);}finally{Directory.Delete(p,true);}}
}
