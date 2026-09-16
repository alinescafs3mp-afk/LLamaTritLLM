using System.Reflection;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Interactivity;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;

if(args.Length==2 && args[0]=="--workspace") return await ClientFaultChecks.Child(Path.GetFileName(args[1]));

string home = Path.Combine(Path.GetTempPath(), "trit-ui-" + Guid.NewGuid().ToString("N"));
Environment.SetEnvironmentVariable("TRITSTUDIO_HOME", home);
try
{
    await ClientFaultChecks.Run(home);
    await ClientFaultChecks.CheckWriteDeadline(home);
    using var session = HeadlessUnitTestSession.StartNew(typeof(HeadlessBuilder));
    using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(3));
    await session.Dispatch<int>(async () =>
    {
        var window = new MainWindow(); window.Show(); Dispatcher.UIThread.RunJobs();
        T Field<T>(string name) => (T)(typeof(MainWindow).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(window)!);
        Task Action(Button button, string title, Func<Task> action) => (Task)typeof(MainWindow)
            .GetMethod("RunButton", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(window, [button, title, action])!;
        void Check(bool ok, string message) { if (!ok) throw new Exception(message); }
        Check(!Field<Button>("_send").IsEnabled, "Send enabled without a model.");
        Check(!Field<Button>("_exportButton").IsEnabled, "Export enabled without a model.");
        Check(Field<ComboBox>("_creationMode").SelectedIndex==0 && Field<CheckBox>("_online").IsChecked==false,"Fresh evolution experiment must not train in the background.");
        Check(Field<Button>("_create").Content!.ToString()!.Contains("без обучения"),"No visible zero-stage action.");
        Field<ComboBox>("_creationMode").SelectedIndex=1; Dispatcher.UIThread.RunJobs();
        Check(Field<Button>("_create").Content!.ToString()!.Contains("базовый"),"Creation mode has no visible feedback.");
        Field<ComboBox>("_creationMode").SelectedIndex=0;
        int calls = 0;
        var complete = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var button = new Button { Content = "Fixture action" };
        Task first = Action(button, "Проверка фидбека", async () => { calls++; await complete.Task; });
        Check(!button.IsEnabled, "Running button must be disabled immediately.");
        Check(Field<TextBlock>("_actionTitle").Text!.StartsWith("Принято"), "No immediate receipt.");
        await Action(button, "Проверка фидбека", () => { calls++; return Task.CompletedTask; });
        Check(calls == 1, "Duplicate action executed.");
        complete.SetResult(true); await first;
        Check(button.IsEnabled, "Button stayed disabled after completion.");
        Check(Field<TextBlock>("_actionTitle").Text!.StartsWith("Завершено"), "No completion feedback.");
        await Action(button, "Проверка ошибки", () => throw new IOException("Fixture failure"));
        Check(Field<TextBlock>("_error").IsVisible && Field<TextBlock>("_error").Text!.Contains("Fixture failure"), "Error not visible.");
        Check(button.IsEnabled, "Button not usable after failure.");
        await Action(button, "Проверка отмены", () => throw new OperationCanceledException());
        Check(Field<TextBlock>("_actionTitle").Text!.StartsWith("Отменено"), "Cancellation presented as success.");
        var packed = Field<Button>("_packedButton"); var picked = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task picker = Action(packed,"Проверка исключения навигации",async () => { await picked.Task; });
        Check(!Field<Button>("_openButton").IsEnabled && !Field<Button>("_newChatButton").IsEnabled,"A second model switch can race the picker.");
        picked.SetResult(true); await picker;
        Check(Field<Button>("_openButton").IsEnabled,"Navigation remained blocked.");
        Check(Field<CheckBox>("_sdpa").IsChecked == true && Field<CheckBox>("_buckets").IsChecked == true && Field<CheckBox>("_targetProjection").IsChecked == true,"Optimization defaults are missing.");
        var dataPicker = Field<Button>("_pickDataButton"); var selectData = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task selection = Action(dataPicker,"Проверка выбора датасета",async()=>{await selectData.Task;});
        Check(!Field<Button>("_openButton").IsEnabled && !Field<Button>("_clearDataButton").IsEnabled,"Dataset picker did not lock navigation/clear.");
        selectData.SetResult(true);await selection;
        Check(dataPicker.IsEnabled,"Dataset picker stayed disabled.");
        // A navigation action cancels a not-yet-sent edit before it can target another workspace.
        using(var delayedSource=new CancellationTokenSource())
        {
            typeof(MainWindow).GetField("_modeDebounce",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,delayedSource);
            await Action(packed,"Отмена устаревшей настройки",()=>Task.CompletedTask);
            Check(delayedSource.IsCancellationRequested,"Navigation kept a stale online-rate timer alive.");
        }
        string echoExe=Path.Combine(AppContext.BaseDirectory,"TritStudio.UiTests"+(OperatingSystem.IsWindows()?".exe":""));
        await using(var echo=new WorkerClient(Path.Combine(home,"echo-mode"),echoExe))
        {
            echo.Start();await echo.WaitReadyAsync();int modeReceipts=0;
            echo.Received+=e=>{if(e.Kind=="completed")Interlocked.Increment(ref modeReceipts);};
            void Set(string name,object? value)=>typeof(MainWindow).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,value);
            Set("_client",echo);Set("_workerReady",true);
            long epoch=Field<long>("_epoch");
            Task stale=(Task)typeof(MainWindow).GetMethod("ApplyOnlineRateAfterDelay",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[])!;
            Set("_epoch",epoch+1);await stale;
            Check(Volatile.Read(ref modeReceipts)==0,"An old rate edit was submitted after changing workspace epoch.");
            Task<bool> applying=(Task<bool>)typeof(MainWindow).GetMethod("ApplyOnline",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[false])!;
            Check(!Field<Button>("_openButton").IsEnabled&&!Field<Button>("_create").IsEnabled,"Model-changing controls are enabled during a pending mode receipt.");
            Check(await applying,"A valid mode receipt was not acknowledged.");
            Set("_client",null);Set("_workerReady",false);
            typeof(MainWindow).GetMethod("UpdateState",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[]);
        }
        Check(!Field<Button>("_maintenanceButton").IsEnabled && !Field<Button>("_discardButton").IsEnabled,"Maintenance is enabled without a workspace.");
        Check(Field<Button>("_newChatButton").Content!.ToString()=="Очистить чат" && !Field<Button>("_newChatButton").IsEnabled,
            "Clear chat must be visible but disabled without a conversation. Real confirmation is tested in Audit15UiChecks.");
        Field<NumericUpDown>("_kvHeads").Value = 3;
        Field<Button>("_create").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Check(Field<TextBlock>("_error").IsVisible, "Invalid configuration did not show an error.");
        Check(Field<Button>("_create").IsEnabled, "Create button stuck after rejected config.");
        Check(!Directory.Exists(Path.Combine(home, "models")), "Invalid create mutated model directory.");
        var modelConfig = new ModelConfig { Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512 };
        var inference = new ManagedInference(TernaryQuantizer.Quantize(WeightSet.Initialize(modelConfig with {Dimension=128,HiddenDimension=256,Layers=2,GroupSize=32})),0);
        typeof(MainWindow).GetField("_model",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,inference);
        Field<TextBox>("_input").Text = new string('x',600); Dispatcher.UIThread.RunJobs();
        Check(!Field<Button>("_send").IsEnabled && Field<TextBlock>("_contextBudget").Text!.Contains("не помещается"),"Oversized prompt has no pre-send feedback.");
        Field<TextBox>("_input").Text = "привет"; Dispatcher.UIThread.RunJobs();
        Check(Field<Button>("_send").IsEnabled && Field<TextBlock>("_contextBudget").Text!.Contains("байт-токенов"),"Context budget did not recover.");
        Task Send() => (Task)typeof(MainWindow).GetMethod("StartSend",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[])!;
        string[] BubbleTexts() => Field<StackPanel>("_messages").Children.OfType<Border>()
            .Select(x=>(StackPanel)x.Child!).SelectMany(x=>x.Children).OfType<SelectableTextBlock>().Select(x=>x.Text??"").ToArray();
        // Invalid math produces a real GenerateDetailed exception, not a mock completed response.
        Array.Fill(inference.Weights.Values["embedding"],float.NaN);
        Field<TextBox>("_input").Text="верни мой запрос"; Field<CheckBox>("_private").IsChecked=true;
        await Action(Field<Button>("_send"),"Проверка ошибки генерации",Send);
        Check(Field<TextBlock>("_error").IsVisible,"Generation failure has no error receipt.");
        Check(BubbleTexts().Contains("Не удалось получить ответ.")&&!BubbleTexts().Contains("Генерирую ответ…"),"Failed answer still appears running.");
        Check(Field<TextBox>("_input").Text=="верни мой запрос","Failed generation lost input.");
        Check(Field<CheckBox>("_private").IsChecked==true,"Excluded failed prompt lost its privacy setting on retry.");
        var healthy=new ManagedInference(TernaryQuantizer.Quantize(WeightSet.Initialize(modelConfig with {Dimension=128,HiddenDimension=256,Layers=2,GroupSize=32})),0);
        typeof(MainWindow).GetField("_model",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,healthy);
        Field<TextBox>("_input").Text="останови этот ответ";
        Task cancelled=Action(Field<Button>("_send"),"Проверка остановки ответа",Send);
        Field<CancellationTokenSource>("_generationCts").Cancel();
        Field<TextBox>("_input").Text="новый черновик"; Field<CheckBox>("_private").IsChecked=false;
        await cancelled;
        Check(Field<TextBlock>("_actionTitle").Text!.StartsWith("Отменено"),"Cancelled generation presented as completed.");
        Check(Field<TextBox>("_input").Text=="новый черновик","Cancel overwrote the user's next draft.");
        Check(!BubbleTexts().Contains("Генерирую ответ…"),"Cancelled answer still appears running.");
        Check(Field<CheckBox>("_private").IsChecked==false,"Cancellation overwrote the new draft privacy choice.");
        Field<TextBox>("_input").Text="отмена приватной реплики";Field<CheckBox>("_private").IsChecked=true;
        Task retryCancelled=Action(Field<Button>("_send"),"Проверка приватной отмены",Send);
        Field<CancellationTokenSource>("_generationCts").Cancel();await retryCancelled;
        Check(Field<TextBox>("_input").Text=="отмена приватной реплики"&&Field<CheckBox>("_private").IsChecked==true,"Cancelled private prompt was restored as trainable.");
        Field<CheckBox>("_private").IsChecked=false;
        // Real completed generation followed by a deliberately unwritable journal target.
        // Failure to append must not be presented as failed generation or restore the already-sent input.
        string blockedJournal = Path.Combine(home, "blocked-journal"); Directory.CreateDirectory(Path.Combine(blockedJournal, "chat.jsonl"));
        typeof(MainWindow).GetField("_workspace",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,blockedJournal);
        typeof(MainWindow).GetField("_sampling",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,new SamplingOptions(0,1,1,1,1));
        int historyCount = Field<List<ChatTurn>>("_history").Count;
        Field<TextBox>("_input").Text="готовый ответ не потеряй";
        await Action(Field<Button>("_send"),"Проверка ошибки записи переписки",Send);
        Check(Field<List<ChatTurn>>("_history").Count==historyCount+1,"Completed answer was discarded after journal failure.");
        Check(Field<TextBox>("_input").Text=="","Journal failure restored an already answered prompt.");
        Check(Field<TextBlock>("_error").Text!.Contains("сохранить переписку"),"Journal failure is not distinguished from model failure.");
        Check(Field<TextBlock>("_status").Text!.StartsWith("Ответ за"),"Completed generation lost its completion receipt.");
        // Hold the real load gate to deterministically supersede a queued load before disk IO.
        string snapshots = Path.Combine(home, "latest-snapshot");
        string latestFolder = ModelFiles.GetRevisionPath(snapshots, "r0000000000000002");Directory.CreateDirectory(latestFolder);
        ModelFiles.Write(Path.Combine(latestFolder,"model.tritmodel"),WeightSet.Initialize(modelConfig),true);
        var latestInfo = new RevisionInfo(2,2,0,null,"UI load fixture",DateTimeOffset.UtcNow,"",ModelFiles.Hash(Path.Combine(latestFolder,"model.tritmodel")),null,"","","",2,null);
        JsonData.AtomicWrite(Path.Combine(latestFolder,"revision.json"),latestInfo);
        void SetField(string name,object? value)=>typeof(MainWindow).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,value);
        SetField("_workspace",snapshots);SetField("_revision",null);SetField("_pending",null);SetField("_publication",1L);
        long publicationEpoch=Field<long>("_epoch");var loader=Field<SemaphoreSlim>("_modelLoader");await loader.WaitAsync();
        Task Load(PublishEvent value,long number)=>(Task)typeof(MainWindow).GetMethod("LoadPublication",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[value,publicationEpoch,number])!;
        Task obsolete=Load(new PublishEvent("r0000000000000001",latestInfo with{Revision=1}),1); // This missing file must never be read.
        var obsoleteToken=Field<CancellationTokenSource>("_snapshotLoadCts").Token;
        SetField("_publication",2L);Task newest=Load(new PublishEvent("r0000000000000002",latestInfo),2);
        Check(obsoleteToken.IsCancellationRequested,"New publication did not cancel an obsolete load.");
        await obsolete.WaitAsync(TimeSpan.FromSeconds(3));
        SetField("_publication",3L);Task repeated=Load(new PublishEvent("r0000000000000002",latestInfo),3);
        Check(!repeated.IsCompleted,"Repeated publication reported completion before the shared load finished.");
        loader.Release();await Task.WhenAll(newest,repeated).WaitAsync(TimeSpan.FromSeconds(5));
        Check(Field<ManagedInference>("_model").Revision==2,"Newest valid publication was not activated.");
        Check(Field<CancellationTokenSource?>("_snapshotLoadCts") is null,"Snapshot load retained its disposed source.");
        var lastGood = Field<ManagedInference>("_model");SetField("_publication",4L);bool mismatchRefused=false;
        try { await Load(new PublishEvent("r0000000000000002",latestInfo with{ModelSha256=new string('0',64)}),4).WaitAsync(TimeSpan.FromSeconds(5)); }
        catch(InvalidDataException) { mismatchRefused=true; }
        Check(mismatchRefused&&ReferenceEquals(lastGood,Field<ManagedInference>("_model")),"Mismatched publication replaced the verified chat model.");
        // Bad destination must not tear down a live chat, change epoch, clear a draft, or leak its lease.
        string badWorkspace=Path.Combine(home,"bad-destination");Directory.CreateDirectory(badWorkspace);
        JsonData.AtomicWrite(Path.Combine(badWorkspace,"active.json"),new ActiveRevision("invalid-revision"));
        var priorModel=Field<ManagedInference>("_model");string? priorWorkspace=Field<string?>("_workspace");long priorEpoch=Field<long>("_epoch");
        Field<TextBox>("_input").Text="черновик должен остаться";int priorHistory=Field<List<ChatTurn>>("_history").Count;
        Task OpenBad()=>(Task)typeof(MainWindow).GetMethod("OpenWorkspace",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,[badWorkspace])!;
        await Action(Field<Button>("_openButton"),"Проверка плохой папки",OpenBad);
        Check(ReferenceEquals(priorModel,Field<ManagedInference>("_model"))&&Field<long>("_epoch")==priorEpoch,"Invalid destination tore down the active model or epoch.");
        Check(Field<string?>("_workspace")==priorWorkspace&&Field<List<ChatTurn>>("_history").Count==priorHistory,"Invalid destination discarded workspace/history.");
        Check(Field<TextBox>("_input").Text=="черновик должен остаться"&&Field<TextBlock>("_error").IsVisible,"Invalid destination lost draft or visible error.");
        using(var leaseProbe=new FileStream(Path.Combine(badWorkspace,".ui.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None)) { }
        // The preflight must also refuse malformed conversation state before disconnecting the client.
        File.Delete(Path.Combine(badWorkspace,"active.json"));File.WriteAllText(Path.Combine(badWorkspace,"chat-state.json"),"{");
        await Action(Field<Button>("_openButton"),"Проверка повреждённого состояния",OpenBad);
        Check(ReferenceEquals(priorModel,Field<ManagedInference>("_model"))&&Field<long>("_epoch")==priorEpoch,"Bad conversation state changed active model.");
        window.Width = 1000; window.Height = 760; Dispatcher.UIThread.RunJobs();
        Check(Field<TextBlock>("_actionTitle").Bounds.Width > 0, "Feedback not laid out at minimum supported width.");
        await Audit15UiChecks.Run(window,home);
        await Audit16UiChecks.Run(window,home);
        window.Close(); Dispatcher.UIThread.RunJobs();
        Console.WriteLine("PASS headless UI: initial state, immediate receipt, duplicate suppression, completion, persistent failure, new chat, config rejection, layout.");
        return 0;
    }, deadline.Token);
    return 0;
}
catch (Exception error) { Console.Error.WriteLine("FAIL headless UI: " + error); return 1; }
finally { if (Directory.Exists(home)) Directory.Delete(home, recursive: true); }
public sealed class HeadlessBuilder
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<StudioApp>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
}
