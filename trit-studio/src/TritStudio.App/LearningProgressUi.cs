using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using TritStudio.Core;
namespace TritStudio.App;

public sealed partial class MainWindow
{
    private readonly TextBlock _trainingAccuracyText = Text("Учебная точность: нет модели",12);
    private readonly TextBlock _controlAccuracyText = Text("Контроль: нет модели",12);
    private readonly TextBlock _accuracyMeaning = Text("Точное предсказание следующего байт-токена, не оценка связности речи.",11);
    private Control BuildAccuracyLine()
    {
        _accuracyMeaning.Foreground = Muted;
        var row = new WrapPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0,4,0,0) };
        foreach(var control in new[] { _trainingAccuracyText,_controlAccuracyText,_accuracyMeaning })
        { control.Margin = new Thickness(0,0,16,0); row.Children.Add(control); }
        return row; // Shared header, always visible regardless of tab or collapsed sidebars.
    }
    private void UpdateAccuracyLine()
    {
        if(_model is null)
        { _trainingAccuracyText.Text = "Учебная точность: нет модели"; _controlAccuracyText.Text = "Контроль: нет модели"; return; }
        // Worker events are epoch-fenced. A rejected candidate's restoration replaces its observations.
        // A standalone packed file contains no training metrics, so it must never inherit the previous model's numbers.
        var progress = _workerStatus?.Accuracy ?? _revision?.Accuracy;
        if(progress?.Training is { } train)
            _trainingAccuracyText.Text = $"Учебная точность: {train.Percent:F1}% · {train.Batches} пак. · шаг {train.Step:N0}";
        else _trainingAccuracyText.Text = "Учебная точность: ещё не измерена";
        if(progress?.Validation is { } control)
        {
            long liveStep = _workerStatus?.Step ?? _revision?.Step ?? 0;
            _controlAccuracyText.Text = $"Контроль: {control.Percent:F1}% · шаг {control.Step:N0} · длина {control.SequenceLength}" +
                (liveStep > control.Step ? " (последняя проверка)" : "");
        }
        else _controlAccuracyText.Text = "Контроль: ещё не измерен";
    }
}
