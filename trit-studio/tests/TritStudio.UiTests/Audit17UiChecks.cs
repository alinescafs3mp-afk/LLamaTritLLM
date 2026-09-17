using System.Reflection;
using Avalonia.Controls;
using Avalonia.VisualTree;
using TritStudio.App;
using TritStudio.Core;

internal static class Audit17UiChecks
{
    public static void Run(MainWindow window)
    {
        T Field<T>(string name)=>(T)typeof(MainWindow).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(window)!;
        object? Invoke(string name,params object?[] args)=>typeof(MainWindow).GetMethod(name,BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(window,args);
        void Check(bool ok,string text){if(!ok)throw new Exception(text);}
        // The actual visual tree contains only three direct training cards, not six independent sections.
        Field<TabControl>("_tabs").SelectedIndex=1;Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        var panels=window.GetVisualDescendants().OfType<Grid>().Single(x=>x.Name=="ModelPanels");
        Check(panels.Children.Count==3,"Training tab does not have exactly three top-level blocks.");
        Check(panels.Children.OfType<Border>().Select(x=>x.Name).ToHashSet().SetEquals(new[]{"ActiveModelPanel","CreateModelPanel","TrainModelPanel"}),"Wrong training block identities.");
        Field<NumericUpDown>("_batch").Value=32;Field<NumericUpDown>("_sequence").Value=1024;Field<NumericUpDown>("_learningRate").Value=.000001m;
        Field<NumericUpDown>("_newBatch").Value=8;Field<NumericUpDown>("_newSequence").Value=512;Field<NumericUpDown>("_newRate").Value=.001m;
        var fresh=(ResourceOptions)Invoke("CreationResources",ModelConfig.Large)!;
        Check(fresh.BatchSize==8&&fresh.SequenceLength==512,"New-model settings leaked from current-model fine-tuning controls.");
        Check(((TrainingOptions)Invoke("CreationTraining")!).LearningRate==.001,"Creation inherited the current model's 1e-6 LR.");
        Invoke("UpdateLearningHint");Check(Field<TextBlock>("_learningHint").Text!.Contains("Очень мало"),"Tiny LR has no visible warning.");
        var model=Field<ManagedInference?>("_model");Field<TextBox>("_input").Text="не трогай черновик";
        Invoke("ApplyLearningPreset");
        Check(Field<NumericUpDown>("_learningRate").Value==.001m&&Field<NumericUpDown>("_batch").Value==8,"Explicit learning profile not applied.");
        Check(ReferenceEquals(model,Field<ManagedInference?>("_model"))&&Field<TextBox>("_input").Text=="не трогай черновик","Learning profile changed weights or draft.");
        Check(Field<TextBlock>("_status").Text!.Contains("НЕ запущено"),"Profile falsely announces automatic training.");
        Check(Field<NumericUpDown>("_newRate").Value==.001m,"Profile changed independent creation options.");
        Field<TextBlock>("_qualitySummary").Text="old model's quality";Invoke("ClearModelSpecificInputs");
        Check(!Field<TextBlock>("_qualitySummary").Text!.Contains("old model"),"Quality result leaked to another model.");
        Console.WriteLine("PASS audit17 headless: three blocks, independent create/train fields, explicit profile, low-LR warning and quality identity.");
    }
}
