using System.Reflection;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;
using Avalonia.VisualTree;
using TritStudio.App;
using TritStudio.Core;

internal static class Audit18UiChecks
{
    public static async Task Run(MainWindow window, string home)
    {
        T Field<T>(string n)=>(T)typeof(MainWindow).GetField(n,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(window)!;
        void Set(string n,object? v)=>typeof(MainWindow).GetField(n,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,v);
        object? Invoke(string n,params object?[] a)=>typeof(MainWindow).GetMethod(n,BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,a);
        Task Call(string n,params object?[] a)=>(Task)Invoke(n,a)!;
        void Check(bool ok,string reason){if(!ok)throw new Exception("Audit18 UI: "+reason);}
        async Task PumpUntil(Func<bool> done)
        {
            using var deadline=new CancellationTokenSource(TimeSpan.FromSeconds(6));
            while(!done()){Dispatcher.UIThread.RunJobs();await Task.Delay(10,deadline.Token);}
            Dispatcher.UIThread.RunJobs();
        }
        var config=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
        string file=Path.Combine(home,"a18-model.tritmodel");ModelFiles.Write(file,WeightSet.Initialize(config),true);
        await Call("OpenPackedPath",file);
        string a=Path.Combine(home,"a18-model-a"),b=Path.Combine(home,"a18-model-b");Directory.CreateDirectory(a);Directory.CreateDirectory(b);
        void SelectContext(string path)
        {
            Set("_workspace",path);Invoke("SetRunEditorContext",path);
            Invoke("ReceiveRunSettings",new ReadyEvent(true,config,new ResourceOptions{Threads=1,MemoryMiB=4096,SequenceLength=512},new TrainingOptions()));
            Invoke("UpdateState");
        }
        SelectContext(a);Field<TabControl>("_tabs").SelectedIndex=1;Dispatcher.UIThread.RunJobs();
        var numbers=new[]{"_learningRate","_steps","_batch","_sequence","_publishEvery"};
        // Actual template TextBox input, not just direct numeric Value assignment.
        var rate=Field<ParameterNumber>("_learningRate");
        var inner=rate.GetVisualDescendants().OfType<TextBox>().First();
        inner.Focus();inner.SelectAll();window.KeyTextInput("0,0003");
        Field<Button>("_saveRunDraft").Focus();Dispatcher.UIThread.RunJobs();
        Check(rate.ReadChecked()==.0003m&&rate.Value==.0003m,"comma LR not committed from real text input");
        inner.Focus();inner.SelectAll();window.KeyTextInput("3e-4");
        Field<Button>("_saveRunDraft").Focus();Dispatcher.UIThread.RunJobs();
        Check(rate.ReadChecked()==.0003m,"scientific LR not committed");
        Field<NumericUpDown>("_batch").Value=64;Field<NumericUpDown>("_steps").Value=731;
        Field<NumericUpDown>("_sequence").Value=384;Field<NumericUpDown>("_publishEvery").Value=137;
        Field<CheckBox>("_refreshCorpus").IsChecked=false;
        Field<CheckBox>("_scheduledRate").IsChecked=true;Field<CheckBox>("_autoSnapshots").IsChecked=true;
        await PumpUntil(()=>File.Exists(LaunchDraftStore.FilePath(a))&&!Field<bool>("_runDraftDirty"));
        var desired=(LaunchDraft)Invoke("CaptureRunDraft")!;
        Check(LaunchDraftStore.Read(a,config)==desired,"auto-save did not preserve the actual typed options");
        Check(desired.Resources.BatchSize==64&&desired.Training.LearningRate==.0003&&desired.Training.Steps==731,"custom values changed before save");
        var model=Field<ManagedInference>("_model");byte[] packed=File.ReadAllBytes(file);
        var observed=new ReadyEvent(true,config,new ResourceOptions{BatchSize=8,Threads=1,MemoryMiB=8192},new TrainingOptions());
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(observed)),Field<long>("_epoch"));
        Dispatcher.UIThread.RunJobs();
        Check((LaunchDraft)Invoke("CaptureRunDraft")! == desired,"repeated ready reset pending run fields");
        Invoke("SetRunEditorContext",a);Invoke("ReceiveRunSettings",observed);
        Check((LaunchDraft)Invoke("CaptureRunDraft")! == desired,"same-model reconnect reset intent");
        var hidden=desired with{Training=desired.Training with{MaxValidationRegression=.31}};
        Invoke("ApplyRunDraft",hidden);
        Check(((LaunchDraft)Invoke("CaptureRunDraft")!).Training.MaxValidationRegression==.31,"editing exposed fields reset an unexposed training option");
        Invoke("ApplyRunDraft",desired);
        // Invalid user text survives blur and cannot silently save the previous valid value.
        inner.Focus();inner.SelectAll();window.KeyTextInput("не число");Field<Button>("_saveRunDraft").Focus();Dispatcher.UIThread.RunJobs();
        Check(rate.Text=="не число","invalid text was silently reverted on blur");
        bool rejected=false;try{Invoke("CaptureRunDraft");}catch(TargetInvocationException e)when(e.InnerException is ArgumentException){rejected=true;}
        Check(rejected&&LaunchDraftStore.Read(a,config)==desired,"invalid input overwrote the good draft");
        rate.Text="0.0003";Invoke("SaveRunDraft",true);
        // A different model owns different parameters. Reopening A reloads its own saved draft.
        SelectContext(b);Field<NumericUpDown>("_steps").Value=89;Field<NumericUpDown>("_batch").Value=32;Invoke("SaveRunDraft",true);
        var other=LaunchDraftStore.Read(b,config);SelectContext(a);
        Check((LaunchDraft)Invoke("CaptureRunDraft")! == desired,"model B fields leaked back into model A");
        Check(other!.Training.Steps==89&&other.Resources.BatchSize==32,"model B values were not stored");
        // Profile is explicit replacement of five fields plus two disclosed opt-ins, save is not. Neither alters weights or creator.
        Field<NumericUpDown>("_newRate").Value=.0008m;Field<NumericUpDown>("_newBatch").Value=64;
        Field<Button>("_trainingPresetButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));Dispatcher.UIThread.RunJobs();
        Check(Field<CheckBox>("_scheduledRate").IsChecked==true&&Field<CheckBox>("_autoSnapshots").IsChecked==true,"profile did not enable disclosed schedule/cadence");
        Check(rate.Value==.001m&&Field<NumericUpDown>("_steps").Value==2000&&Field<NumericUpDown>("_batch").Value==8,"explicit profile didn't fill its five fields");
        Check(Field<NumericUpDown>("_newRate").Value==.0008m&&Field<NumericUpDown>("_newBatch").Value==64,"profile altered creation options");
        Field<CheckBox>("_scheduledRate").IsChecked=false;Field<CheckBox>("_autoSnapshots").IsChecked=false;
        Field<NumericUpDown>("_batch").Value=24;rate.Value=.0004m;
        Field<Button>("_saveRunDraft").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        await PumpUntil(()=>!Field<HashSet<Button>>("_busyButtons").Contains(Field<Button>("_saveRunDraft")));
        Check(LaunchDraftStore.Read(a,config)!.Resources.BatchSize==24&&rate.Value==.0004m,"save button re-applied the profile");
        Check(!LaunchDraftStore.Read(a,config)!.Training.WarmupCosine&&!LaunchDraftStore.Read(a,config)!.Training.AutoSnapshotInterval,"save unexpectedly re-enabled schedule");
        Check(ReferenceEquals(model,Field<ManagedInference>("_model"))&&packed.SequenceEqual(File.ReadAllBytes(file)),"editing settings changed model weights");
        // An invalid hidden create-training field must not prevent saving a selected model's run or raw creation.
        Field<ParameterNumber>("_newRate").Text="invalid";Field<ParameterNumber>("_newBatch").Text="invalid";
        Invoke("SaveRunDraft",true);
        var raw=(ResourceOptions)Invoke("ZeroCreationResources",config)!;raw.ValidateInitialization(config);
        Check(raw.BatchSize==8,"raw creation consumed the hidden invalid batch editor");
        // Prior headless fixtures set kvHeads=3 to reject create. Restore a valid architecture so this
        // assertion tests independent creation-draft persistence, not that leftover invalid config.
        // A strict FlushEditorDrafts throw is not unwound by Avalonia 11.3 headless Dispatch.
        Field<NumericUpDown>("_kvHeads").Value=2;
        Field<ParameterNumber>("_newRate").SetNumber(.0008m);Field<ParameterNumber>("_newBatch").SetNumber(64);
        Invoke("FlushEditorDrafts",true,true);
        Check(LaunchDraftStore.ReadCreation(Path.Combine(AppPaths.Root,"creation-draft.json"))!.Resources.BatchSize==64,"creation draft not independent/persistent");
        // Help is LABEL-only and every transition must wait the full 2 seconds.
        string[] helped=["_dimension","_hidden","_layers","_heads","_kvHeads","_context","_planes","_group","_threshold",
            "_steps","_learningRate","_batch","_sequence","_publishEvery","_newSteps","_newRate","_newBatch","_newSequence","_newPublish"];
        foreach(string field in helped)
        {
            var n=Field<NumericUpDown>(field);var parent=(StackPanel)n.Parent!;var label=(TextBlock)parent.Children[0];
            Check(ToolTip.GetShowDelay(label)==2000&&ToolTip.GetBetweenShowDelay(label)==0,"wrong help timing: "+field);
            Check(ToolTip.GetTip(label) is TextBlock tip && (tip.Text?.Length ?? 0)>25 && ToolTip.GetTip(n) is null,"help absent or attached to editor rather than label: "+field);
        }
        foreach(string field in new[]{"_threads","_memory","_name","_newMemory"})
        {var control=Field<Control>(field);Check(ToolTip.GetTip(control) is null,"unrequested general tooltip");}
        // Real pointer dwell on isolated controls from the PRODUCTION label factory.
        // Avalonia 11.3 headless deadlocks inside Popup.IsOpen and on a second Window.Show
        // after the main window has been exercised. Overlay on the already-pumped shell;
        // cancel ToolTipOpening so the 2s DispatcherTimer still fires without a popup.
        // Labels need an explicit hit-testable size: a default TextBlock is only hittable
        // on glyph pixels. TranslatePoint is Avalonia.VisualExtensions in 11.3.
        var helpFactory=typeof(MainWindow).GetMethod("HelpField",BindingFlags.NonPublic|BindingFlags.Static)!;
        var helpA=(StackPanel)helpFactory.Invoke(null,new object[]{"Ширина",new Border{Height=10},"Объяснение первого параметра для теста."})!;
        var helpB=(StackPanel)helpFactory.Invoke(null,new object[]{"Пакет",new Border{Height=10},"Объяснение второго параметра для теста."})!;
        var labelA=(TextBlock)helpA.Children[0];var labelB=(TextBlock)helpB.Children[0];
        foreach(var label in new[]{labelA,labelB}){label.Width=80;label.Height=24;label.Background=Brushes.Transparent;}
        int openings=0;
        void Opening(object? sender,CancelRoutedEventArgs e){openings++;e.Cancel=true;}
        ToolTip.AddToolTipOpeningHandler(labelA,Opening);
        ToolTip.AddToolTipOpeningHandler(labelB,Opening);
        bool windowTips=ToolTip.GetServiceEnabled(window);
        ToolTip.SetServiceEnabled(window,false);
        ToolTip.SetServiceEnabled(labelA,true);
        ToolTip.SetServiceEnabled(labelB,true);
        var shell=Field<LayoutTransformControl>("_scaledShell");
        var grid=(Grid)shell.Child!;
        var canvas=new Canvas{Width=400,Height=200,Background=Brushes.Transparent};
        Canvas.SetLeft(helpA,16);Canvas.SetTop(helpA,16);Canvas.SetLeft(helpB,16);Canvas.SetTop(helpB,90);
        canvas.Children.Add(helpA);canvas.Children.Add(helpB);
        var overlay=new Border{Background=Brushes.Transparent,Child=canvas,HorizontalAlignment=Avalonia.Layout.HorizontalAlignment.Left,VerticalAlignment=Avalonia.Layout.VerticalAlignment.Top,ZIndex=1000};
        grid.Children.Add(overlay);
        try
        {
            Dispatcher.UIThread.RunJobs();
            var away=new Avalonia.Point(300,40);
            foreach(var label in new[]{labelA,labelB})
            {
                window.MouseMove(away);
                await Task.Delay(30);
                var mapped=Avalonia.VisualExtensions.TranslatePoint(label,new Avalonia.Point(label.Bounds.Width/2,label.Bounds.Height/2),window);
                Check(mapped is not null && label.Bounds.Width>0,"help label is not in the hover overlay");
                int before=openings;
                window.MouseMove(mapped!.Value);
                long start=Environment.TickCount64;
                while(Environment.TickCount64-start<1000)
                {
                    await Task.Delay(20);
                    Check(openings==before,"tooltip appeared before the requested delay");
                }
                while(openings==before)
                {
                    if(Environment.TickCount64-start>6000) throw new Exception("Audit18 UI: tooltip did not open after 2s pointer dwell");
                    await Task.Delay(20);
                }
                Check(Environment.TickCount64-start>=2000,"tooltip opened before the 2s dwell");
                window.MouseMove(away);
                await Task.Delay(50);
                Check(!ToolTip.GetIsOpen(label)&&openings==before+1,"tooltip remained after pointer exit");
            }
        }
        finally
        {
            ToolTip.RemoveToolTipOpeningHandler(labelA,Opening);
            ToolTip.RemoveToolTipOpeningHandler(labelB,Opening);
            ToolTip.SetServiceEnabled(window,windowTips);
            grid.Children.Remove(overlay);
            Dispatcher.UIThread.RunJobs();
        }
        // Real storage failure retains edits and last good file, with explicit visible feedback.
        File.Delete(LaunchDraftStore.FilePath(a));Directory.CreateDirectory(LaunchDraftStore.FilePath(a));
        rate.Value=.0005m;Invoke("FlushEditorDrafts",false,false);
        Check(Field<TextBlock>("_runDraftStatus").Text!.Contains("НЕ сохранены")&&rate.Value==.0005m,"save error reset editor or claimed success");
        Directory.Delete(LaunchDraftStore.FilePath(a));Invoke("SaveRunDraft",true);
        // Live accuracy is visible on BOTH tabs and has no coupling to the launch editor.
        var metric=new LearningProgress(new TokenAccuracy(75,100,4,512,2),new TokenAccuracy(30,50,3,512,1));
        Invoke("OnWorker",new WorkerEvent("status",null,Protocol.Element(new StatusEvent("учимся",Step:4,Busy:true,Stage:"train",Accuracy:metric))),Field<long>("_epoch"));
        Dispatcher.UIThread.RunJobs();
        Check(Field<TextBlock>("_trainingAccuracyText").Text!.Contains("75")&&Field<TextBlock>("_controlAccuracyText").Text!.Contains("60"),"live accuracy not rendered");
        foreach(int tab in new[]{0,1}){Field<TabControl>("_tabs").SelectedIndex=tab;Dispatcher.UIThread.RunJobs();Check(Field<TextBlock>("_trainingAccuracyText").IsEffectivelyVisible,"accuracy hidden with tab/sidebar");}
        var lower=metric with{Training=new TokenAccuracy(50,100,5,512,3)};
        Invoke("OnWorker",new WorkerEvent("status",null,Protocol.Element(new StatusEvent("учимся",Step:5,Accuracy:lower))),Field<long>("_epoch"));Dispatcher.UIThread.RunJobs();
        Check(Field<TextBlock>("_trainingAccuracyText").Text!.Contains("50"),"accuracy was forced to be monotonic");
        // Cleanly reset editor identity so later close does not point at a removed/trashed context.
        await Call("OpenPackedPath",file);
        Check(Field<TextBlock>("_trainingAccuracyText").Text!.Contains("ещё не измерена"),"old model's score leaked into standalone file");
        Check(!Field<Button>("_saveRunDraft").IsEnabled,"inference-only file enabled workspace draft writes");
        Console.WriteLine("PASS audit18 real headless numeric input, preset/save separation, autosave, model identity, ready fencing, batch64, help labels and storage failure.");
    }
}
