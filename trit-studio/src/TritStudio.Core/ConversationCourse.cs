namespace TritStudio.Core;

public sealed record ConversationPreparation(TrainingExample[] Examples, int OriginalRows,
    int AddedAssistantTargets, int ExcludedControlPrefixes);

// Expand only explicitly supplied TRAIN dialogues. Expected answers from control/probe files never enter here.
// Each derived example teaches one assistant reply with the preceding real context. Original rows retain their IDs.
public static class ConversationSupervision
{
    public static ConversationPreparation Expand(TrainingExample[] source, ValidationGuard guard,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (source.Length > Dataset.MaxExamples) throw new ArgumentException("Too many source examples for the conversation course.");
        guard.EnsureTraining(source, ct);
        var result = source.ToDictionary(x => x.Id, StringComparer.Ordinal);
        int added = 0, excluded = 0;
        foreach (var row in source)
        {
            ct.ThrowIfCancellationRequested();
            var history = row.History ?? [];
            for (int i = 0; i < history.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                var prefix = Dataset.Make(history[i].User, history[i].Assistant, row.Source,
                    history.Take(i).ToArray());
                if (guard.Contains(prefix, ct)) { excluded++; continue; }
                if (result.TryAdd(prefix.Id, prefix)) added++;
                if (result.Count > Dataset.MaxExamples)
                    throw new ArgumentException("После выделения ответов ассистента больше 50000 примеров. Разделите корпус.");
            }
        }
        return new(result.Values.ToArray(), source.Length, added, excluded);
    }
}

public sealed record ConversationPhase(string Name, EncodedExample[] Examples);

// Three immutable pools per explicit job. It is a sampling curriculum, not three different models.
// The corpus stored in the checkpoint is the full union. A new manual job starts the course again.
public sealed class ConversationCurriculum
{
    public ConversationPhase Foundation { get; }
    public ConversationPhase Dialogue { get; }
    public ConversationPhase Context { get; }
    public ConversationCurriculum(EncodedExample[] all, IEnumerable<string> starterIds, IEnumerable<string>? contextIds = null)
    {
        if (all.Length == 0) throw new ArgumentException("Empty conversation course.");
        var known = starterIds.ToHashSet(StringComparer.Ordinal);
        var basic = all.Where(x => known.Contains(x.Id)).ToArray();
        if (basic.Length == 0) throw new ArgumentException("Разговорный курс требует свой проверенный начальный корпус. Включите обновление встроенных данных.");
        var dialogue = all.Where(x => x.Tokens.Contains(ByteTokenizer.Assistant)).ToArray();
        if (dialogue.Length == 0) throw new ArgumentException("Conversation course has no dialogue.");
        if (contextIds is not null)
        {
            var contextKnown = contextIds.ToHashSet(StringComparer.Ordinal);
            var context = all.Where(x => contextKnown.Contains(x.Id)).ToArray();
            if (context.Length == 0) throw new ArgumentException("Контекстная практика требует проверенных контекстных примеров.");
            // Every phase retains the entire training distribution. A starter-only phase can increase
            // full-control loss enough to trigger the unchanged guard before reaching the dialogue phase.
            // Repeated entries are REFERENCES, not copied token buffers or new unique documents.
            Foundation = new("Короткие ответы + контекст + общий материал", ContextMix(all, basic, context, 2 * all.Length, all.Length));
            Dialogue = new("Контекстная практика с повторением базы", ContextMix(all, basic, context, all.Length / 2, all.Length));
            Context = new("Общий корпус с поддержкой контекста", ContextMix(all, basic, context, all.Length / 4, all.Length / 2));
            return;
        }
        Foundation = new("Короткие фразы и ответы", basic);
        Dialogue = new("Ответы с сохранением короткой базы", ReplayMix(dialogue, basic));
        Context = new("Весь корпус и многоходовой контекст", ReplayMix(all, basic));
    }
    private static EncodedExample[] ContextMix(EncodedExample[] all, EncodedExample[] basic,
        EncodedExample[] context, int basicCount, int contextCount)
    {
        var result = new EncodedExample[checked(all.Length + basicCount + contextCount)];
        all.CopyTo(result, 0);
        for (int i = 0; i < basicCount; i++) result[all.Length + i] = basic[i % basic.Length];
        for (int i = 0; i < contextCount; i++) result[all.Length + basicCount + i] = context[i % context.Length];
        return result;
    }
    private static EncodedExample[] ReplayMix(EncodedExample[] all, EncodedExample[] basic)
    {
        // Add ~one starter reference per four ordinary references. This intentional repetition is
        // training replay, not a claim of additional unique records. No token arrays are duplicated.
        int extra = Math.Max(1, all.Length / 4);
        var mixed = new EncodedExample[checked(all.Length + extra)]; all.CopyTo(mixed, 0);
        for (int i = 0; i < extra; i++) mixed[all.Length + i] = basic[i % basic.Length];
        return mixed;
    }
    public ConversationPhase At(int completed, int total)
    {
        if (total < 1 || completed < 0 || completed >= total) throw new ArgumentOutOfRangeException(nameof(completed));
        // Use integer products, not floor boundaries that make one-step jobs have zero late phase.
        if ((long)completed * 5 < total) return Foundation;
        return (long)completed * 5 < (long)total * 3 ? Dialogue : Context;
    }
}

public static class ExampleLossWeights
{
    // Sum of weights within each example is 1/Rows. A short reply does not disappear under long paragraphs.
    // This is an optional training objective. Validation remains the original token-mean cross entropy.
    public static float[] Build(SupervisedBatch batch)
    {
        if (batch.Rows < 1 || batch.Length < 1 || batch.Positions.Length != batch.Targets.Length)
            throw new ArgumentException("Invalid supervised batch.");
        var counts = new int[batch.Rows];
        foreach (long p in batch.Positions)
        {
            if (p < 0 || p >= (long)batch.Rows * batch.Length) throw new ArgumentException("Invalid target position.");
            counts[(int)(p / batch.Length)]++;
        }
        if (counts.Any(x => x == 0)) throw new ArgumentException("Every course example needs a target.");
        var weights = new float[batch.Positions.Length];
        for (int i = 0; i < weights.Length; i++) weights[i] = 1f / (batch.Rows * (float)counts[(int)(batch.Positions[i] / batch.Length)]);
        return weights;
    }
}
