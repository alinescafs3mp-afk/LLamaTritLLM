using System.Reflection;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;
internal static class Audit22UiChecks
{
    public static async Task Run(MainWindow window,string home)
    {
        T Field<T>(string n)=>(T)typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(window)!;
        void Set(string n,object? v)=>typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.SetValue(window,v);
        object? Invoke(string n,params object?[] a)=>typeof(MainWindow).GetMethod(n,BindingFlags.NonPublic|BindingFlags.Instance)!.Invoke(window,a);
        void Check(bool x,string detail){if(!x)throw new Exception("Audit22 UI: "+detail);}
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
        string file=Path.Combine(home,"a22.tritmodel");ModelFiles.Write(file,WeightSet.Initialize(cfg),true);
        await (Task)Invoke("OpenPackedPath",file)!;
        string folder=Path.Combine(home,"a22-workspace");Directory.CreateDirectory(folder);Set("_workspace",folder);Invoke("SetRunEditorContext",folder);
        var ready=new ReadyEvent(true,cfg,new ResourceOptions{Threads=1,MemoryMiB=4096},new TrainingOptions());
        Invoke("ReceiveRunSettings",ready);Invoke("UpdateState");
        Field<Button>("_conversationPresetButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));Dispatcher.UIThread.RunJobs();
        Check(LaunchDraftStore.Read(folder,cfg)!.Training.TransferPractice,"profile did not explicitly enable v22");
        var model=Field<ManagedInference>("_model");
        Field<CheckBox>("_transferPractice").IsChecked=false;Invoke("SaveRunDraft",true);
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(ready)),Field<long>("_epoch"));Dispatcher.UIThread.RunJobs();
        Check(!((LaunchDraft)Invoke("CaptureRunDraft")!).Training.TransferPractice,"Ready overwrote v22 opt-out");
        Field<CheckBox>("_transferPractice").IsChecked=true;Field<CheckBox>("_contextPractice").IsChecked=false;
        Check(Field<CheckBox>("_transferPractice").IsChecked==false&&!Field<CheckBox>("_transferPractice").IsEnabled,"v21 opt-out left invalid v22 enabled");
        Invoke("SaveRunDraft",true);Check(!LaunchDraftStore.Read(folder,cfg)!.Training.TransferPractice,"opt-out was not saved");
        Invoke("ShowConversationResult",Protocol.Element(new{revision=1,step=3,summary="старый отчёт",readable="old.md",transferSummary="проверка v22",transferReadable="transfer.md"}));
        Check(Field<TextBlock>("_conversationSummary").Text!.Contains("transfer.md"),"new report has no visible path");
        Check(ReferenceEquals(model,Field<ManagedInference>("_model")),"report changed weights");
        await (Task)Invoke("OpenPackedPath",file)!;
        Check(!Field<TextBlock>("_conversationSummary").Text!.Contains("transfer.md"),"old score leaked across model switch");
        Console.WriteLine("PASS audit22 real profile/opt-out/save, Ready fencing, report feedback and model identity");
    }
}
