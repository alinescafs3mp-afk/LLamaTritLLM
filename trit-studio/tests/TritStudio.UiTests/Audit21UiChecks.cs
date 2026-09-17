using System.Reflection;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;

internal static class Audit21UiChecks
{
    public static async Task Run(MainWindow window,string home)
    {
        T Field<T>(string n)=>(T)typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(window)!;
        void Set(string n,object? value)=>typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.SetValue(window,value);
        object? Invoke(string n,params object?[] values)=>typeof(MainWindow).GetMethod(n,BindingFlags.NonPublic|BindingFlags.Instance)!.Invoke(window,values);
        void Check(bool ok,string reason){if(!ok)throw new Exception("Audit21 UI: "+reason);}
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
        string file=Path.Combine(home,"a21.tritmodel");ModelFiles.Write(file,WeightSet.Initialize(cfg),true);
        await (Task)Invoke("OpenPackedPath",file)!;
        string path=Path.Combine(home,"a21-workspace");Directory.CreateDirectory(path);
        Set("_workspace",path);Invoke("SetRunEditorContext",path);
        var ready=new ReadyEvent(true,cfg,new ResourceOptions{MemoryMiB=4096,Threads=1},new TrainingOptions());
        Invoke("ReceiveRunSettings",ready);Invoke("UpdateState");
        Field<Button>("_conversationPresetButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));Dispatcher.UIThread.RunJobs();
        var draft=LaunchDraftStore.Read(path,cfg)!;
        Check(draft.Training.ContextPractice&&draft.Training.ConversationCourse,"Preset did not opt into v21 explicitly");
        var model=Field<ManagedInference>("_model");
        Field<CheckBox>("_contextPractice").IsChecked=false;Invoke("SaveRunDraft",true);
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(ready)),Field<long>("_epoch"));Dispatcher.UIThread.RunJobs();
        Check(!((LaunchDraft)Invoke("CaptureRunDraft")!).Training.ContextPractice,"Ready reenabled the context recipe");
        Field<CheckBox>("_contextPractice").IsChecked=true;Field<CheckBox>("_conversationCourse").IsChecked=false;
        Check(Field<CheckBox>("_contextPractice").IsChecked==false && !Field<CheckBox>("_contextPractice").IsEnabled,"Course opt-out left context practice enabled");
        Invoke("SaveRunDraft",true);Check(!LaunchDraftStore.Read(path,cfg)!.Training.ContextPractice,"Opt-out not durable");
        var payload=Protocol.Element(new{revision=12,step=99,summary="Общий отчёт",readable="main.md",contextSummary="Контекст 2/16",contextReadable="context.md",contextError=(string?)null});
        Invoke("ShowConversationResult",payload);
        Check(Field<TextBlock>("_conversationSummary").Text!.Contains("context.md"),"Context report has no visible result/path");
        Check(ReferenceEquals(model,Field<ManagedInference>("_model")),"Displaying a report changed active weights");
        await (Task)Invoke("OpenPackedPath",file)!;
        Check(!Field<TextBlock>("_conversationSummary").Text!.Contains("context.md"),"Other model retained context score");
        Console.WriteLine("PASS audit21 real profile click, independent opt-out/save, repeated Ready, context report and model identity");
    }
}
