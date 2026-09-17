namespace TritStudio.Core;

// Checks semantic agreement BEFORE allocating LibTorch or replacing the current session.
// Hashes prove byte integrity, not that separate files describe the same optimizer step.
public static class CheckpointStateGuard
{
    public static void Validate(RevisionInfo info, TrainerState state, CommitLedger commits, ModelConfig config)
    {
        if (state.LastLearningRate is double rate && (!double.IsFinite(rate) || rate is <= 0 or > 0.01 || state.Step == 0))
            throw new InvalidDataException("Недопустимая фактическая скорость последнего шага в снимке.");
        info.Accuracy?.Validate(info.Step, config.Context);
        if (state.Step < 0 || state.TargetTokens < 0 || state.SamplerState == 0 ||
            state.Step != info.Step || state.TargetTokens != info.TargetTokens || state.TargetTokens < state.Step)
            throw new InvalidDataException("Счётчики тренера не совпадают с метаданными снимка. Восстановление отменено.");
        if (info.ChangedWeights > config.ParameterCount)
            throw new InvalidDataException("Число изменённых весов превышает размер модели.");
        if (state.Step == 0 && (state.TargetTokens != 0 || info.ChangedWeights != 0))
            throw new InvalidDataException("Нулевой снимок содержит признаки обучения.");
        if (commits.AppliedOnlineIds is null || commits.AppliedOnlineIds.Length > Dataset.MaxExamples)
            throw new InvalidDataException("Недопустимый журнал обученных онлайн-примеров.");
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (string id in commits.AppliedOnlineIds)
            if (id is null || id.Length != 64 || id.Any(c => !(c is >= '0' and <= '9' or >= 'A' and <= 'F')) || !ids.Add(id))
                throw new InvalidDataException("Журнал онлайн-примеров содержит повреждённые или повторные идентификаторы.");
        if (state.Step == 0 && ids.Count != 0)
            throw new InvalidDataException("Случайная модель не может иметь уже обученные онлайн-примеры.");
    }
}
