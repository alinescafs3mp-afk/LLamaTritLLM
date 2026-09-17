using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using TritStudio.Core;
namespace TritStudio.App;

public sealed partial class MainWindow
{
    private readonly Button _qualityButton = Button("Проверить обучение"), _trainingPresetButton = Button("Учебный старт: применить настройки");
    private readonly TextBlock _qualitySummary = Text("Проверка сравнивает тренер и CPU-файл и показывает свободные ответы без изменения весов.",12);
    private readonly TextBlock _learningHint = Text("",12);
    private readonly NumericUpDown _newMemory = Number(8192,512,65536,512), _newBatch = Number(8,1,32),
        _newSequence = Number(512,16,2048,16), _newSteps = Number(1200,1,100000,100),
        _newRate = Number(0.001m,0.000001m,0.01m,0.0001m,"0.######"), _newPublish = Number(100,1,1000,25);
    private readonly CheckBox _newCuda = new() { Content = "CUDA для новой модели", IsChecked = true };
    private StackPanel? _createTrainFields;
    private static WrapPanel Flow(params Control[] controls)
    {
        var panel = new WrapPanel { Orientation = Orientation.Horizontal };
        foreach(var control in controls) { control.Margin = new Thickness(0,0,8,4);panel.Children.Add(control); }
        return panel;
    }
    private static Control SectionTitle(string text) => new Border { Margin = new Thickness(0,8,0,4),
        BorderBrush = new SolidColorBrush(Color.Parse("#344156")), BorderThickness = new Thickness(0,1,0,0),
        Padding = new Thickness(0,6,0,0), Child = Text(text,14) };
    private static Grid FormFields(params Control[] controls)
    {
        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("*,*") };
        for(int row=0;row<(controls.Length+1)/2;row++)grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        for(int i=0;i<controls.Length;i++)
        { controls[i].Margin = new Thickness(i%2==0?0:10,0,0,6);Grid.SetRow(controls[i],i/2);Grid.SetColumn(controls[i],i%2);grid.Children.Add(controls[i]); }
        return grid;
    }
    private ResourceOptions CreationResources(ModelConfig config) => new()
    {
        Threads = Math.Max(1, Environment.ProcessorCount-2), MemoryMiB = Int(_newMemory), BatchSize = Int(_newBatch),
        SequenceLength = Math.Min(config.Context,Int(_newSequence)), PreferCuda = _newCuda.IsChecked == true
    };
    private TrainingOptions CreationTraining() => new() { Steps = Int(_newSteps), LearningRate = (double)(_newRate.Value ?? 0.001m), PublishEvery = Int(_newPublish) };
    private void UpdateLearningHint()
    {
        double rate = (double)(_learningRate.Value ?? 0);
        _learningHint.Foreground = rate < 0.00001 ? new SolidColorBrush(Color.Parse("#FFBCAD")) : Muted;
        _learningHint.Text = $"LR = {rate:0.000000} ({rate:0.###E+0}). " +
            (rate < 0.00001 ? "Очень мало для старта со случайных весов. 8000 шагов не компенсируют автоматически столь маленькое обновление." :
                "Это скорость обновления весов, не точность. Проверяйте свободные ответы и контрольную ошибку, не только количество шагов.");
    }
    private void ApplyLearningPreset()
    {
        _learningRate.Value = 0.001m; _batch.Value = 8; _sequence.Value = Math.Min(512,_model?.Weights.Config.Context ?? 512);
        _steps.Value = 2000; _publishEvery.Value = 200;
        UpdateLearningHint(); _status.Text = "LR 0,001; пакет 8; длина до 512; 2000 шагов; публикация каждые 200. Обучение НЕ запущено. Архитектура и бюджет RAM не изменены.";
        Notice("Учебный профиль применён", _status.Text);
    }
    private async Task CheckLearning()
    {
        NeedWorker();var client = _client!;long epoch = _epoch;
        _operationBusy = true;UpdateState();
        try
        {
            var receipt = await client.Request("quality",new { });
            if(epoch != _epoch || !ReferenceEquals(client,_client))throw new InvalidOperationException("Модель изменилась во время диагностики.");
            var result = receipt.Data.GetProperty("result"); string file = result.GetProperty("file").GetString()!;
            _qualitySummary.Text = result.GetProperty("summary").GetString() + $"\nШаг {result.GetProperty("step").GetInt64():N0}; " +
                $"сохранённый LR ручного обучения {result.GetProperty("learningRate").GetDouble():G6}; точность по целевым байтам небольшой учебной выборки {result.GetProperty("accuracy").GetDouble():P1}.\nОтчёт: {file}";
            _status.Text = _qualitySummary.Text;
            try { Process.Start(new ProcessStartInfo(Path.GetDirectoryName(file)!) { UseShellExecute = true }); } catch(Exception e){Log(e.Message);}
        }
        finally { _operationBusy = false;UpdateState(); }
    }
}
