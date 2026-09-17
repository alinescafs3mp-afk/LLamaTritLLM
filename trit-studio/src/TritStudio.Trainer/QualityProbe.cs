using TorchSharp;
using TritStudio.Core;
using static TorchSharp.torch;
namespace TritStudio.Trainer;

// Explicit user action, reads the currently COMMITTED model. No optimizer steps or queue changes.
public static class QualityProbe
{
    public sealed record ProbeReply(string Prompt, string Reply, bool PresentAsTrainingInput, double MaxLogitError, bool PipelineParityPassed, GenerationHealth? Health = null, bool? ExactTrainingReply = null);
    public sealed record Report(long Revision, long Step, double SavedLearningRate, double TrainSampleLoss,
        double ControlSampleLoss, double TrainSampleAccuracy, ProbeReply[] Replies, string Interpretation, string Privacy, ByteBaselineScore? ByteReference = null, double? RequestedLearningRate = null);
    public static Report Run(TrainingSession session, string revisionPath, WorkspaceSettings settings,
        TrainingExample[] training, TrainingExample[] control, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if(training.Length == 0 || control.Length == 0) throw new InvalidDataException("Для диагностики нужен непустой учебный и контрольный набор.");
        var info = ModelFiles.VerifyRevision(revisionPath);
        if (info.Step != session.Step) throw new InvalidOperationException("Диагностика требует последнего опубликованного шага; завершите текущий запуск.");
        var cpu = new ManagedInference(ModelFiles.Read(Path.Combine(revisionPath, "model.tritmodel"), true, ct: ct), info.Revision, 1);
        var sample = Dataset.EncodeAll(training.Where(x => x.IsDialogue).DefaultIfEmpty(training[0]).Take(16).ToArray(), settings.Resources.SequenceLength, 1, ct);
        var controls = Dataset.EncodeAll(control.Take(16).ToArray(), settings.Resources.SequenceLength, 1, ct);
        double trainLoss = session.Evaluate(sample, ct);
        var trainAccuracy = session.LastEvaluationAccuracy ?? throw new InvalidDataException("Нет счётчиков учебной проверки.");
        // Evaluate now counts accuracy from its existing logits. Do not run the same sample again just to count argmax.
        double controlLoss = session.Evaluate(controls, ct);
        // Measure a language-free reference on the full frozen control, fitted ONLY to train labels.
        // No model update, held-out fitting or answer substitution is involved.
        var reference = new ByteBaseline(training.Select(x => Dataset.Encode(x, settings.Resources.SequenceLength)), ct)
            .Evaluate(control.Select(x => Dataset.Encode(x, settings.Resources.SequenceLength)), ct);
        var replies = new List<ProbeReply>();
        foreach (string prompt in new[] { "Привет!", "Как дела?", "Как тебя зовут?" })
        {
            ct.ThrowIfCancellationRequested(); var tokens = ByteTokenizer.Prompt([], prompt, settings.Config.Context, Math.Min(128, settings.Config.Context / 2));
            float[] expected;
            using (var scope = NewDisposeScope())
            using (var noGrad = no_grad())
            {
                var ids = tensor(tokens.Select(x => (long)x).ToArray(), dtype: ScalarType.Int64, device: session.Model.Device).reshape(1, -1);
                expected = session.Model.Forward(ids).select(1, tokens.Length - 1).cpu().data<float>().ToArray();
            }
            var incremental = cpu.NewSession(); float[] actual = [];
            for(int index=0;index<tokens.Length;index++) actual = incremental.Step(tokens[index], ct, computeLogits:index == tokens.Length-1);
            double maximum = 0; bool passed = true;
            for (int i = 0; i < expected.Length; i++)
            {
                double delta = Math.Abs(expected[i] - actual[i]);
                if(!double.IsFinite(delta)) throw new ArithmeticException("Диагностика получила нечисловые логиты. Численный путь неисправен; не маскируйте это дополнительным обучением.");
                maximum = Math.Max(maximum, delta);
                if (!double.IsFinite(delta) || delta > 0.002 + 0.001 * Math.Abs(expected[i])) passed = false;
            }
            var answer = cpu.Generate(tokens, () => new SamplingOptions(0, 262, 1, 1, Math.Min(128, settings.Config.Context / 2)), null, ct);
            var known = training.Where(x => x.IsDialogue && x.Text == prompt && (x.History?.Length ?? 0) == 0).ToArray();
            replies.Add(new(prompt, answer, known.Length > 0, maximum, passed, GenerationHealth.Inspect(answer),
                known.Length == 0 ? null : known.Any(x => string.Equals(x.Answer, answer, StringComparison.Ordinal))));
        }
        string interpretation = replies.Any(x => !x.PipelineParityPassed) ? "Обнаружено расхождение тренера и CPU-файла. Не лечите это дополнительными шагами: требуется исправление пути весов." :
            session.Step == 0 ? "Случайные веса: обучения ещё не было." :
            replies.Any(x => x.Health?.Repetitive == true) ? "Численный путь согласован, но свободная генерация зациклена. Снижение teacher-forced ошибки и байтовая точность НЕ означают владение диалогом. Уменьшите чрезмерный LR, проверьте расписание и примеры, а не маскируйте ответ фильтром." :
            settings.Training.LearningRate < 0.00001 ? "Численный путь согласован на проверенных запросах. Заданный пиковый/постоянный LR ручного обучения ниже 1e-5: вероятно слишком медленное обучение с нуля. Это гипотеза, не доказательство причины всех ошибок." :
            "На проверенных запросах путь тренер -> файл -> CPU согласован. Сравните точность по целевым токенам и свободные ответы; число шагов само по себе не доказывает качество.";
        interpretation += $" Биграммный ориентир без нейросети на полном контроле: {reference.Accuracy:P1}. Он не является собеседником.";
        return new(info.Revision, session.Step, session.LastLearningRate ?? settings.Training.LearningRate, trainLoss, controlLoss,
            (double)trainAccuracy.Correct / trainAccuracy.Total, replies.ToArray(), interpretation,
            "Only three fixed public prompts/generated answers and aggregate metrics. No raw conversation/dataset records are copied. Generated replies MAY contain learned private information: review before sharing. Sample losses are not whole-corpus or blind-test scores.", reference, settings.Training.LearningRate);
    }
}
