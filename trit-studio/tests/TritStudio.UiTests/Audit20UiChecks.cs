using System.Reflection;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;

internal static class Audit20UiChecks
{
    public static async Task Run(MainWindow window,string home)
    {
        T Field<T>(string n)=>(T)typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(window)!;
        void Set(string n,object? x)=>typeof(MainWindow).GetField(n,BindingFlags.NonPublic|BindingFlags.Instance)!.SetValue(window,x);
        object? Invoke(string n,params object?[] a)=>typeof(MainWindow).GetMethod(n,BindingFlags.NonPublic|BindingFlags.Instance)!.Invoke(window,a);
        async Task Call(string n,params object?[] a)=>await (Task)Invoke(n,a)!;
        void Check(bool c,string m){if(!c)throw new Exception("Audit20 UI: "+m);}
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
        string file=Path.Combine(home,"a20.tritmodel");ModelFiles.Write(file,WeightSet.Initialize(cfg),true);
        await Call("OpenPackedPath",file);var model=Field<ManagedInference>("_model");byte[] saved=File.ReadAllBytes(file);
        string a=Path.Combine(home,"a20-a"),b=Path.Combine(home,"a20-b");Directory.CreateDirectory(a);Directory.CreateDirectory(b);
        var ready=new ReadyEvent(true,cfg,new ResourceOptions{Threads=1,MemoryMiB=4096,SequenceLength=512},new TrainingOptions());
        void Select(string path){Set("_workspace",path);Invoke("SetRunEditorContext",path);Invoke("ReceiveRunSettings",ready);Invoke("UpdateState");}
        Select(a);var creatorRate=Field<NumericUpDown>("_newRate").Value;
        Field<Button>("_conversationPresetButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));Dispatcher.UIThread.RunJobs();
        var chosen=LaunchDraftStore.Read(a,cfg)!;
        Check(chosen.Training.ConversationCourse&&chosen.Training.EqualExampleWeight&&chosen.Training.WarmupCosine&&chosen.Training.AutoSnapshotInterval,"Course opt-ins were not saved.");
        Check(chosen.Resources.BatchSize==16&&chosen.Training.Steps==6000&&chosen.Training.LearningRate==.001,"Profile launch settings differ.");
        Check(Field<NumericUpDown>("_newRate").Value==creatorRate&&ReferenceEquals(model,Field<ManagedInference>("_model")),"Profile changed creator/active weights.");
        Field<CheckBox>("_conversationCourse").IsChecked=false;Field<CheckBox>("_equalExamples").IsChecked=true;
        Field<NumericUpDown>("_steps").Value=213;Invoke("SaveRunDraft",true);var desired=LaunchDraftStore.Read(a,cfg)!;
        Check(!desired.Training.ConversationCourse&&desired.Training.EqualExampleWeight&&desired.Training.Steps==213,"Save replaced custom course choices.");
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(ready)),Field<long>("_epoch"));Dispatcher.UIThread.RunJobs();
        Check((LaunchDraft)Invoke("CaptureRunDraft")! == desired,"Ready overwrote the course editor.");
        Select(b);Field<CheckBox>("_equalExamples").IsChecked=false;Invoke("SaveRunDraft",true);Select(a);
        Check((LaunchDraft)Invoke("CaptureRunDraft")! == desired,"Course options leaked between models.");
        string summary=Field<TextBlock>("_conversationSummary").Text!;
        var receipt=Protocol.Element(new{revision=99L,step=100L,summary="outdated",readable="old.md"});
        Invoke("OnWorker",new WorkerEvent("conversation-result",null,receipt),Field<long>("_epoch")-1);Dispatcher.UIThread.RunJobs();
        Check(Field<TextBlock>("_conversationSummary").Text==summary,"Stale-model probe result was displayed.");
        Invoke("OnWorker",new WorkerEvent("conversation-result",null,receipt),Field<long>("_epoch"));Dispatcher.UIThread.RunJobs();
        Check(Field<TextBlock>("_conversationSummary").Text!.Contains("old.md"),"Probe receipt produced no visible outcome.");
        Check(File.ReadAllBytes(file).SequenceEqual(saved),"UI settings or probe receipt changed inference file.");
        await Call("OpenPackedPath",file);
        Check(!Field<Button>("_conversationProbeButton").IsEnabled,"Standalone inference file enabled worker diagnostics.");
        Check(!Field<TextBlock>("_conversationSummary").Text!.Contains("old.md"),"Prior model report survived model switch.");
        Console.WriteLine("PASS audit20 headless actual course-profile click, durable opt-outs, model isolation, stale receipts and no hidden training.");
    }
}
