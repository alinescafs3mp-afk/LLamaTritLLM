using System.Reflection;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Input.Raw;
using Avalonia.Interactivity;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;

// Real headless input routing and on-disk failures. No mocked model answers.
internal static class Audit16UiChecks
{
    public static async Task Run(MainWindow window, string home)
    {
        T Field<T>(string n)=>(T)typeof(MainWindow).GetField(n,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(window)!;
        void Set(string n,object? v)=>typeof(MainWindow).GetField(n,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,v);
        object? Invoke(string n,params object?[] a)=>typeof(MainWindow).GetMethod(n,BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,a);
        Task Call(string n,params object?[] a)=>(Task)Invoke(n,a)!;
        void Check(bool ok,string reason){if(!ok)throw new Exception("Audit16: "+reason);}
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=2,Heads=2,KvHeads=1,Context=512,GroupSize=8};
        string file=Path.Combine(home,"keyboard.tritmodel");ModelFiles.Write(file,WeightSet.Initialize(cfg),true);
        await Call("OpenPackedPath",file);
        Field<TabControl>("_tabs").SelectedIndex=0;
        var input=Field<TextBox>("_input");var history=Field<List<ChatTurn>>("_history");
        Set("_sampling",new SamplingOptions(0,1,1,1,4));
        void Press(RawInputModifiers modifiers=RawInputModifiers.None)=>window.KeyPress(Key.Enter,modifiers,PhysicalKey.Enter,"\r");
        void Release(RawInputModifiers modifiers=RawInputModifiers.None)=>window.KeyRelease(Key.Enter,modifiers,PhysicalKey.Enter,"\r");
        async Task Idle()
        {
            using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(10));
            do { Dispatcher.UIThread.RunJobs(); await Task.Delay(5,timeout.Token); }
            while(Field<bool>("_generating")||Field<HashSet<Button>>("_busyButtons").Count!=0);
        }
        input.Text="первая строка";input.CaretIndex=input.Text.Length;input.Focus();Dispatcher.UIThread.RunJobs();
        Check(input.IsFocused,"composer must own keyboard focus");
        int before=history.Count;
        var savedModel=Field<ManagedInference>("_model");Set("_model",null);Invoke("UpdateState");
        string unsent=input.Text!;Press();Release();
        Check(input.Text==unsent&&history.Count==before,"Enter with no model altered/submitted the draft");
        Set("_model",savedModel);Invoke("UpdateState");
        Press(RawInputModifiers.Shift);Release(RawInputModifiers.Shift);Dispatcher.UIThread.RunJobs();
        Check(history.Count==before&&!Field<bool>("_generating"),"Shift+Enter submitted a message");
        Check(input.Text!.Contains('\n')||input.Text.Contains('\r'),"Shift+Enter did not insert a newline through TextBox");
        input.Text="нажатие Enter";input.CaretIndex=input.Text.Length;Invoke("UpdateState");
        Check(Field<Button>("_send").IsEnabled,"send should be enabled before keyboard test");
        Press(); // Keep key down across completion: auto-repeat must not submit a second draft.
        await Idle();
        Check(history.Count==before+1&&history[^1].User=="нажатие Enter","plain Enter did not send exactly once without adding a newline");
        input.Text="следующий черновик";Press();Dispatcher.UIThread.RunJobs();
        Check(history.Count==before+1&&input.Text=="следующий черновик","held Enter resubmitted or changed the next draft");
        Release();Press();Release();await Idle();
        Check(history.Count==before+2&&history[^1].User=="следующий черновик","released Enter could not send the next message");
        Set("_navigationBusy",true);Invoke("UpdateState");input.Text="не отправлять во время переключения";
        Press();Release();Dispatcher.UIThread.RunJobs();
        Check(history.Count==before+2&&input.Text=="не отправлять во время переключения","Enter bypassed disabled send/navigation gate");
        Set("_navigationBusy",false);Invoke("UpdateState");
        Console.WriteLine("PASS audit16 actual keyboard routing: Shift+Enter newline, Enter sends once, held-key latch and busy gate.");

        // Missing active pointer is different from a fresh empty workspace.
        foreach(string kind in new[]{"active-directory","orphaned-revision","chat-state-directory"})
        {
            string root=Path.Combine(home,"v16-"+kind);Directory.CreateDirectory(root);
            if(kind=="active-directory")Directory.CreateDirectory(Path.Combine(root,"active.json"));
            if(kind=="orphaned-revision")Directory.CreateDirectory(Path.Combine(root,"revisions","r0000000000000000"));
            if(kind=="chat-state-directory")Directory.CreateDirectory(Path.Combine(root,"chat-state.json"));
            var model=Field<ManagedInference>("_model");var client=Field<WorkerClient?>("_client");long epoch=Field<long>("_epoch");
            input.Text="не потерять черновик";Field<CheckBox>("_private").IsChecked=true;int rows=history.Count;
            bool refused=false;try{await Call("OpenWorkspace",root);}catch(InvalidDataException){refused=true;}
            Check(refused&&ReferenceEquals(model,Field<ManagedInference>("_model"))&&ReferenceEquals(client,Field<WorkerClient?>("_client")),kind+" replaced usable model/client");
            Check(Field<long>("_epoch")==epoch&&history.Count==rows&&input.Text=="не потерять черновик"&&Field<CheckBox>("_private").IsChecked==true,kind+" lost context/draft/privacy");
            using(var lease=new FileStream(Path.Combine(root,".ui.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None)){}
            Check(!File.Exists(Path.Combine(root,"chat-state.json")),kind+" created a conversation state after failed preview");
        }
        // Exercise actual nested model dialogs, including programmatic double-click despite disabled state.
        Task manager=Call("ManageModels");Window? catalog=null;
        try
        {
            using var deadline=new CancellationTokenSource(TimeSpan.FromSeconds(5));
            while((catalog=window.OwnedWindows.FirstOrDefault(x=>x.Title=="Модели")) is null)
            {Dispatcher.UIThread.RunJobs();await Task.Delay(5,deadline.Token);}
            var content=(StackPanel)((Border)catalog.Content!).Child!;
            var row=content.Children.OfType<StackPanel>().Last();
            var remove=row.Children.OfType<Button>().First();
            remove.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            remove.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));Dispatcher.UIThread.RunJobs();
            Check(catalog.OwnedWindows.Count==1,"duplicate model-removal confirmation opened");
            var answer=catalog.OwnedWindows.Single();
            var answerContent=(StackPanel)((Border)answer.Content!).Child!;
            answerContent.Children.OfType<StackPanel>().Last().Children.OfType<Button>()
                .Single(x=>x.Content!.ToString()=="Отмена").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            Dispatcher.UIThread.RunJobs();
            Check(catalog.OwnedWindows.Count==0,"cancelled confirmation remained open");
        }
        finally
        {
            if(catalog is not null){foreach(var child in catalog.OwnedWindows.ToArray())child.Close(false);catalog.Close(null);}
            await manager.WaitAsync(TimeSpan.FromSeconds(5));
        }
        // A Ready payload failure must not leave the UI permanently suppressing control handlers.
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(new ReadyEvent(true,cfg,new ResourceOptions(),new TrainingOptions{OnlineLearningRate=double.MaxValue}))),Field<long>("_epoch"));
        Dispatcher.UIThread.RunJobs();
        Check(!Field<bool>("_settingControls"),"failed Ready control update left suppression enabled");
        Set("_workerReady",false);Invoke("UpdateState");
        Console.WriteLine("PASS audit16 failed workspace handoff preserves active session and malformed Ready releases control suppression.");
    }
}
