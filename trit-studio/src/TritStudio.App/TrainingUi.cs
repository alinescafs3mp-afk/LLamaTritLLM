using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using TritStudio.Core;
namespace TritStudio.App;

public sealed partial class MainWindow
{
    private readonly Button _conversationPresetButton = Button("Подготовить разговорный курс"), _conversationProbeButton = Button("Проверить разговор");
    private readonly TextBlock _conversationSummary = Text("Разговорная проверка: ещё не выполнялась. Здесь важен текст ответа, а не только байтовая точность.",12);
    private readonly CheckBox _conversationCourse = new() { Content = "Курс: короткие ответы → диалоги → общий контекст; учить все ответы ассистента" },
        _equalExamples = new() { Content = "Равный вес примеров: короткий ответ не теряется за длинными текстами" },
        _newConversationCourse = new() { Content = "Разговорный курс (только при разговорном этапе)" },
        _newEqualExamples = new() { Content = "Равный вес учебных примеров" },
        _contextPractice = new() { Content = "v21: контекст и перефразировки, весь корпус с первого этапа" },
        _newContextPractice = new() { Content = "v21: смешанная контекстная практика курса" },
        _transferPractice = new() { Content = Text("v22: новые формулировки и баланс задач в каждом пакете",12) },
        _newTransferPractice = new() { Content = Text("v22: перенос на другие формулировки (включает материалы v22)",12) };
    private readonly Button _qualityButton = Button("Проверить обучение"), _trainingPresetButton = Button("Заполнить учебным профилем");
    private readonly TextBlock _qualitySummary = Text("Проверка сравнивает тренер и CPU-файл и показывает свободные ответы без изменения весов.",12);
    private readonly TextBlock _learningHint = Text("",12);
    private readonly NumericUpDown _newMemory = Number(8192,512,65536,512), _newBatch = Number(8,1,ResourceOptions.MaxBatchSize),
        _newSequence = Number(512,16,2048,16), _newSteps = Number(1200,1,100000,100),
        _newRate = Number(0.001m,0.000001m,0.01m,0.0001m,"0.######"), _newPublish = Number(100,1,100000,25);
    private readonly CheckBox _newCuda = new() { Content = "CUDA для новой модели", IsChecked = true };
    private readonly CheckBox _scheduledRate = new() { Content = "Разогрев и плавное снижение LR до 10% пика" },
        _autoSnapshots = new() { Content = "Автоподбор интервала снимков для длинного запуска (без удаления)" },
        _newScheduledRate = new() { Content = "Разогрев и плавное снижение LR" },
        _newAutoSnapshots = new() { Content = "Автоподбор интервала снимков (без удаления)" };
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
        Threads = Math.Max(1, Environment.ProcessorCount-2), MemoryMiB = LaunchInt(_newMemory), BatchSize = LaunchInt(_newBatch),
        SequenceLength = Math.Min(config.Context,LaunchInt(_newSequence)), PreferCuda = _newCuda.IsChecked == true
    };
    private ResourceOptions ZeroCreationResources(ModelConfig config) => new()
    {
        Threads = Math.Max(1, Environment.ProcessorCount - 2), MemoryMiB = LaunchInt(_newMemory),
        BatchSize = 8, SequenceLength = Math.Min(config.Context, 512), PreferCuda = _newCuda.IsChecked == true
    };
    private TrainingOptions CreationTraining() => _creationTrainingBase with { Steps = LaunchInt(_newSteps), LearningRate = (double)CheckedNumber(_newRate), PublishEvery = LaunchInt(_newPublish), WarmupCosine = _newScheduledRate.IsChecked == true, AutoSnapshotInterval = _newAutoSnapshots.IsChecked == true, ConversationCourse = _newConversationCourse.IsChecked == true, ContextPractice = _newConversationCourse.IsChecked == true && _newContextPractice.IsChecked == true, TransferPractice = _newConversationCourse.IsChecked == true && _newContextPractice.IsChecked == true && _newTransferPractice.IsChecked == true, EqualExampleWeight = _newEqualExamples.IsChecked == true };
    private void UpdateLearningHint()
    {
        double rate = (double)(_learningRate.Value ?? 0);
        _learningHint.Foreground = rate < 0.00001 || rate > 0.003 ? new SolidColorBrush(Color.Parse("#FFBCAD")) : Muted;
        _learningHint.Text = $"LR = {rate:0.000000} ({rate:0.###E+0}). " +
            (rate < 0.00001 ? "Очень мало для старта со случайных весов. 8000 шагов не компенсируют автоматически столь маленькое обновление." :
                rate > 0.003 ? "Высокий LR для AdamW: возможны ухудшение и повторение символов. 0,01: МАКСИМУМ, не минимум." :
                "Это скорость обновления весов, не точность. Проверяйте свободные ответы и контрольную ошибку, не только количество шагов.");
    }
    private void ApplyLearningPreset()
    {
        if (_closing || _operationBusy || _openingWorkspace || _navigationBusy || _modeApplying) return;
        _restoringLaunch = true;
        try
        {
            SetEditorValue(_learningRate, 0.001m); SetEditorValue(_batch, 8); SetEditorValue(_sequence, Math.Min(512,_model?.Weights.Config.Context ?? 512));
            SetEditorValue(_steps, 2000); SetEditorValue(_publishEvery, 200);
            _scheduledRate.IsChecked = true; _autoSnapshots.IsChecked = true;
        }
        finally { _restoringLaunch = false; }
        MarkRunEdited();
        if (_runContext is not null && _runEditorInitialized) SaveRunDraft(explicitSave:true);
        UpdateLearningHint(); _status.Text = "Пик LR 0,001 с разогревом и снижением до 0,0001; пакет 8; длина до 512; 2000 шагов; интервал снимка от 200 с автоподбором. Обучение НЕ запущено. Архитектура и бюджет RAM не изменены.";
        Notice("Поля заполнены учебным профилем", _status.Text);
    }
    private void ApplyConversationPreset()
    {
        if (_closing || _operationBusy || _openingWorkspace || _navigationBusy || _modeApplying || _model is null) return;
        _restoringLaunch = true;
        try
        {
            SetEditorValue(_learningRate, 0.001m); SetEditorValue(_batch, 16);
            SetEditorValue(_sequence, Math.Min(512, _model.Weights.Config.Context));
            SetEditorValue(_steps, 6000); SetEditorValue(_publishEvery, 500);
            _scheduledRate.IsChecked = true; _autoSnapshots.IsChecked = true;
            _conversationCourse.IsChecked = true; _contextPractice.IsChecked = true; _transferPractice.IsChecked = true; _equalExamples.IsChecked = true;
            _trainingMaterial.SelectedIndex = (int)TrainingMaterial.Conversation; _refreshCorpus.IsChecked = true;
        }
        finally { _restoringLaunch = false; }
        MarkRunEdited(); if (_runContext is not null && _runEditorInitialized) SaveRunDraft(explicitSave: true);
        UpdateLearningHint(); Notice("Разговорный курс подготовлен, но НЕ запущен", "6000 шагов; пик LR 0,001 со снижением; пакет 16; до 512 байт. Курс v22: перефразировки и независимые комбинации фактов; категории смешиваются внутри каждого пакета. Равный вес примеров. Старые курсы доступны при выключенном v22. Архитектура и RAM не менялись. Нажмите «Дообучить текущую модель».");
    }
    private async Task CheckConversation()
    {
        NeedWorker(); var client = _client!; long epoch = _epoch; _operationBusy = true; UpdateState();
        try
        {
            var result = await client.Request("conversation-check", new { });
            if (epoch != _epoch || !ReferenceEquals(client, _client)) throw new InvalidOperationException("Модель сменилась во время проверки.");
            var data = result.Data.GetProperty("result");
            ShowConversationResult(data);
            string folder = Path.GetDirectoryName(data.GetProperty("file").GetString()!)!;
            try { Process.Start(new ProcessStartInfo(folder) { UseShellExecute = true }); } catch (Exception e) { Log(e.Message); }
        }
        finally { _operationBusy = false; UpdateState(); }
    }
    private void ShowConversationResult(System.Text.Json.JsonElement data)
    {
        _conversationSummary.Text = $"r{data.GetProperty("revision").GetInt64()}, шаг {data.GetProperty("step").GetInt64()}: " +
            data.GetProperty("summary").GetString() + "\nОтчёт: " + data.GetProperty("readable").GetString();
        if (data.TryGetProperty("contextSummary", out var summary) && summary.ValueKind == System.Text.Json.JsonValueKind.String)
            _conversationSummary.Text += "\n" + summary.GetString() + "\nКонтекст: " + data.GetProperty("contextReadable").GetString();
        if (data.TryGetProperty("contextError", out var error) && error.ValueKind == System.Text.Json.JsonValueKind.String)
            _conversationSummary.Text += "\nКонтекстная проверка не завершена: " + error.GetString();
        if (data.TryGetProperty("transferSummary", out var transfer) && transfer.ValueKind == System.Text.Json.JsonValueKind.String)
            _conversationSummary.Text += "\n" + transfer.GetString() + "\nНовые формулировки: " + data.GetProperty("transferReadable").GetString();
        if (data.TryGetProperty("transferError", out var failed) && failed.ValueKind == System.Text.Json.JsonValueKind.String)
            _conversationSummary.Text += "\nПроверка новых формулировок не завершена: " + failed.GetString();
        Log(_conversationSummary.Text);
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
                $"LR последнего сохранённого шага {result.GetProperty("learningRate").GetDouble():G6}; точность по целевым байтам небольшой учебной выборки {result.GetProperty("accuracy").GetDouble():P1}.\nОтчёт: {file}";
            _status.Text = _qualitySummary.Text;
            try { Process.Start(new ProcessStartInfo(Path.GetDirectoryName(file)!) { UseShellExecute = true }); } catch(Exception e){Log(e.Message);}
        }
        finally { _operationBusy = false;UpdateState(); }
    }
}
