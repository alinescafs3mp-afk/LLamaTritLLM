namespace TritStudio.Core;

// Teacher-forced NEXT-TOKEN accuracy. Count only supervised targets (including EOS), never padding/history.
// This is not sentence accuracy, fluency, a probability of correctness, or a guaranteed monotone score.
public sealed record TokenAccuracy(long Correct, long Total, long Step, int SequenceLength, int Batches = 1)
{
    public double Percent => Total > 0 ? 100.0 * Correct / Total : 0;
    public void Validate(long currentStep, int context = 2048)
    {
        if (Total <= 0 || Correct < 0 || Correct > Total || Step < 0 || Step > currentStep ||
            SequenceLength < 1 || SequenceLength > context || Batches < 1)
            throw new InvalidDataException("Некорректные счётчики точности предсказания.");
    }
}
public sealed record LearningProgress(TokenAccuracy? Training = null, TokenAccuracy? Validation = null)
{
    public void Validate(long step, int context = 2048)
    {
        Training?.Validate(step, context); Validation?.Validate(step, context);
        if (Training is not null && (Training.Step == 0 || Training.Batches > AccuracyWindow.Capacity))
            throw new InvalidDataException("Учебная точность не может обозначать нулевой или слишком большой интервал.");
    }
}

// Single-owner, token-weighted moving window. Not an unweighted mean of differently sized batches.
public sealed class AccuracyWindow
{
    public const int Capacity = 32;
    private readonly Queue<(long Correct, long Total)> _batches = new();
    private long _correct, _total;
    public TokenAccuracy? Current { get; private set; }
    public void Add(long correct, long total, long completedStep, int sequenceLength)
    {
        var sample = new TokenAccuracy(correct, total, completedStep, sequenceLength);
        sample.Validate(completedStep);
        if (completedStep < 1 || Current is { } last && completedStep <= last.Step)
            throw new ArgumentException("Шаг точности должен увеличиваться после успешного обновления.");
        long removeCorrect = _batches.Count == Capacity ? _batches.Peek().Correct : 0;
        long removeTotal = _batches.Count == Capacity ? _batches.Peek().Total : 0;
        long newCorrect = checked(_correct - removeCorrect + correct), newTotal = checked(_total - removeTotal + total);
        if (_batches.Count == Capacity) _batches.Dequeue();
        _batches.Enqueue((correct,total)); _correct = newCorrect; _total = newTotal;
        Current = new(_correct, _total, completedStep, sequenceLength, _batches.Count);
    }
    public void Clear() { _batches.Clear(); _correct = _total = 0; Current = null; }
}
