using System.Reflection;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;
using TritStudio.App;
using TritStudio.Core;

internal static class Audit15UiChecks
{
    public static async Task Run(MainWindow window, string home)
    {
        T Field<T>(string name)=>(T)typeof(MainWindow).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(window)!;
        void Set(string name,object? value)=>typeof(MainWindow).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(window,value);
        object? Invoke(string name,params object?[] args)=>typeof(MainWindow).GetMethod(name,BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,args);
        async Task Call(string name,params object?[] args)=>await (Task)Invoke(name,args)!;
        void Check(bool ok,string reason){if(!ok)throw new Exception(reason);}
        Invoke("SetSidebar",false,false);Check(!Field<ScrollViewer>("_chatSidebar").IsVisible,"Chat settings do not collapse.");
        Field<Button>("_chatSettingsButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Check(Field<ScrollViewer>("_chatSidebar").IsVisible,"Chat settings button has no visible effect.");
        Field<Button>("_chatSettingsButton").RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        Check(!Field<ScrollViewer>("_chatSidebar").IsVisible,"Chat settings button failed to hide panel.");
        foreach(double scale in new[]{0.5,0.65,0.8,1,1.25})
        {
            Invoke("ApplyUiScale",scale,false);Dispatcher.UIThread.RunJobs();
            Check(((ScaleTransform)Field<LayoutTransformControl>("_scaledShell").LayoutTransform!).ScaleX==scale,"UI scale not applied.");
        }
        Invoke("ApplyUiScale",double.NaN,false);
        Check(((ScaleTransform)Field<LayoutTransformControl>("_scaledShell").LayoutTransform!).ScaleX==0.8,"Invalid persisted UI scale was not normalized.");
        Field<NumericUpDown>("_publishEvery").Value=250;
        Check(((TrainingOptions)Invoke("SelectedTraining")!).PublishEvery==250,"Publish interval editor does not affect next training request.");
        // Ready belongs to active training resources, NEVER to the new-network architecture editor.
        decimal? edited=Field<NumericUpDown>("_dimension").Value;
        var cfg=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512};
        Invoke("OnWorker",new WorkerEvent("ready",null,Protocol.Element(new ReadyEvent(true,cfg,new ResourceOptions(),new TrainingOptions{LearningRate=0.0003,PublishEvery=250}))),Field<long>("_epoch"));
        Dispatcher.UIThread.RunJobs();
        Check(Field<NumericUpDown>("_dimension").Value==edited,"Active model overwrote the independent new-model editor.");
        Set("_workerReady",false);
        var feedbackHost = new StackPanel();
        var feedbackTurn = new ChatTurn("вопрос","ответ",0);
        Invoke("AddTeaching",feedbackHost,feedbackTurn);
        var fold = (Expander)feedbackHost.Children.Single(); Check(fold.Content is null,"Feedback editor allocated before expansion.");
        fold.IsExpanded = true; var editorContent = fold.Content; Check(editorContent is not null,"Expanded feedback has no controls.");
        fold.IsExpanded = false; fold.IsExpanded = true; Check(ReferenceEquals(editorContent,fold.Content),"Collapsing/reopening feedback lost its state.");
        string a=Path.Combine(home,"catalog-a.tritmodel"), b=Path.Combine(home,"catalog-b.tritmodel"), bad=Path.Combine(home,"catalog-broken.tritmodel");
        ModelFiles.Write(a,WeightSet.Initialize(cfg),true);ModelFiles.Write(b,WeightSet.Initialize(cfg with{Seed=73}),true);File.WriteAllText(bad,"not model bytes");
        await Call("OpenPackedPath",a);
        Check(ModelLibrary.SamePath(Field<ModelEntry[]>("_library").Single(x=>ModelLibrary.SamePath(x.Path,a)).Path,a),"Loaded file missing from library.");
        Check(ModelLibrary.SamePath(((ModelEntry)Field<ComboBox>("_modelPicker").SelectedItem!).Path,a),"Header selection differs from actual loaded model.");
        Check(!Field<Button>("_train").IsEnabled,"Inference-only file incorrectly enables training.");
        // Real confirmation buttons: cancel preserves the conversation; confirm clears only the current context.
        string clearRoot=Path.Combine(home,"clear-chat-fixture");Directory.CreateDirectory(clearRoot);
        string oldConversation=Field<string>("_conversationId");
        var turn=new ChatTurn("запомни синий блокнот","синий блокнот",0,ConversationId:oldConversation);
        ChatJournal.AppendAsync(Path.Combine(clearRoot,"chat.jsonl"),turn).GetAwaiter().GetResult();
        JsonData.AtomicWrite(Path.Combine(clearRoot,"chat-state.json"),oldConversation);
        File.WriteAllText(Path.Combine(clearRoot,"runtime.json"),"runtime untouched");
        File.WriteAllText(Path.Combine(clearRoot,"replay.json"),"queue untouched");
        Set("_workspace",clearRoot);Field<List<ChatTurn>>("_history").Add(turn);
        Invoke("AddBubble","Вы",turn.User,true);Invoke("AddBubble","Трит",turn.Assistant,false);
        Field<TextBox>("_input").Text="сохрани мой черновик";Field<CheckBox>("_private").IsChecked=true;Invoke("UpdateState");
        var clear=Field<Button>("_newChatButton");var originalModel=Field<ManagedInference>("_model");long originalEpoch=Field<long>("_epoch");
        var worker=Field<WorkerClient?>("_client");bool? online=Field<CheckBox>("_online").IsChecked;
        var sampling=Field<SamplingOptions>("_sampling");byte[] journal=File.ReadAllBytes(Path.Combine(clearRoot,"chat.jsonl"));
        Task ClearAction()=> (Task)Invoke("RunButton",clear,"Очистка чата",(Func<Task>)(()=>Call("ClearChat")))!;
        void AnswerDialog(bool yes)
        {
            Dispatcher.UIThread.RunJobs();var dialog=window.OwnedWindows.Single(x=>x.Title=="Очистить чат?");
            var content=(StackPanel)((Border)dialog.Content!).Child!;
            var buttons=content.Children.OfType<StackPanel>().Last();
            buttons.Children.OfType<Button>().Single(x=>x.Content!.ToString()==(yes?"Подтвердить":"Отмена")).RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        }
        Task cancelClear=ClearAction();Check(!clear.IsEnabled&&!Field<Button>("_send").IsEnabled,"Clear confirmation did not lock concurrent send.");
        AnswerDialog(false);await cancelClear;
        Check(Field<string>("_conversationId")==oldConversation&&Field<List<ChatTurn>>("_history").Count==1,"Cancelled clear changed context.");
        Task clearNow=ClearAction();AnswerDialog(true);await clearNow;
        Check(Field<string>("_conversationId")!=oldConversation&&Field<List<ChatTurn>>("_history").Count==0&&Field<StackPanel>("_messages").Children.Count==0,"Confirmed clear did not reset visible history.");
        Check(JsonData.Read<string>(Path.Combine(clearRoot,"chat-state.json"))==Field<string>("_conversationId"),"Clear did not persist conversation boundary.");
        Check(ReferenceEquals(originalModel,Field<ManagedInference>("_model"))&&ReferenceEquals(worker,Field<WorkerClient?>("_client"))&&Field<long>("_epoch")==originalEpoch,"Clear switched the active model/client.");
        Check(ReferenceEquals(sampling,Field<SamplingOptions>("_sampling"))&&Field<CheckBox>("_online").IsChecked==online,"Clear changed learning/generation settings.");
        Check(Field<TextBox>("_input").Text=="сохрани мой черновик"&&Field<CheckBox>("_private").IsChecked==true,"Clear lost next draft/privacy.");
        Check(File.ReadAllBytes(Path.Combine(clearRoot,"chat.jsonl")).SequenceEqual(journal)&&File.ReadAllText(Path.Combine(clearRoot,"replay.json"))=="queue untouched"&&File.ReadAllText(Path.Combine(clearRoot,"runtime.json"))=="runtime untouched","Clear erased logs or queue.");
        Invoke("ApplyHistory",ChatJournal.ReadTail(Path.Combine(clearRoot,"chat.jsonl")));
        Check(Field<List<ChatTurn>>("_history").Count==0&&Field<StackPanel>("_messages").Children.Count==0,"Cleared history returned on reload.");
        // Refused durable commit must leave old in-memory context visible.
        string currentConversation=Field<string>("_conversationId");
        Field<List<ChatTurn>>("_history").Add(turn with{ConversationId=currentConversation});Invoke("AddBubble","Вы","оставь меня",true);
        File.Delete(Path.Combine(clearRoot,"chat-state.json"));Directory.CreateDirectory(Path.Combine(clearRoot,"chat-state.json"));
        bool commitFailed=false;try{Invoke("ClearConversation");}catch{commitFailed=true;}
        Check(commitFailed&&Field<List<ChatTurn>>("_history").Count==1&&Field<string>("_conversationId")==currentConversation,"Failed clear lost current conversation.");
        Directory.Delete(Path.Combine(clearRoot,"chat-state.json"));Set("_workspace",null);Invoke("ClearConversation");
        Console.WriteLine("PASS audit15 chat clear: real modal cancel/confirm, durable boundary, unchanged weights/client/queue, draft retained and storage-failure preservation.");

        var before=Field<ManagedInference>("_model");long epoch=Field<long>("_epoch");Field<TextBox>("_input").Text="draft survives failed switch";
        bool refused=false;try{await Call("SwitchSelectedModel",new ModelEntry(bad,"broken",false,false));}catch{refused=true;}
        Check(refused&&ReferenceEquals(before,Field<ManagedInference>("_model"))&&Field<long>("_epoch")==epoch,"Broken selection destroyed active model.");
        Check(Field<TextBox>("_input").Text=="draft survives failed switch","Bad model selection lost draft.");
        Set("_datasetPaths",new[]{"old-model-only.jsonl"});
        await Call("SwitchSelectedModel",new ModelEntry(b,"catalog-b",false,false));
        Check(ModelLibrary.SamePath(Field<string>("_inferenceFile"),b)&&Field<string[]>("_datasetPaths").Length==0,"Switch routed to wrong model or leaked old dataset choices.");
        // Remove an INACTIVE managed directory using actual filesystem operations: active inference stays b.
        string inactive=Path.Combine(AppPaths.Models,"inactive-fixture");Directory.CreateDirectory(inactive);File.WriteAllText(Path.Combine(inactive,"weights.bin"),"keep me in trash");
        await Call("RemoveEntry",new ModelEntry(inactive,"inactive-fixture",true,true));
        Check(!Directory.Exists(inactive)&&ModelLibrary.SamePath(Field<string>("_inferenceFile"),b),"Removing inactive model switched active one.");
        Check(Directory.EnumerateFiles(AppPaths.Trash,"weights.bin",SearchOption.AllDirectories).Any(),"Trash destroyed model payload.");
        await Call("RemoveEntry",new ModelEntry(b,"catalog-b",false,false));
        Check(File.Exists(b)&&Field<ManagedInference?>("_model") is null&&!Field<Button>("_train").IsEnabled,"External removal deleted file or left a training target.");
        Console.WriteLine("PASS audit15 headless: scale/sidebar feedback, active identity, isolated create fields, failed/successful switching, real inactive trash and external forget.");
    }
}
