namespace TritStudio.Core;

public sealed record OnlineReplayBatch(EncodedExample[] Examples, int LearnedExamplesEncoded, int BaseSelections);

// Build exactly the same logical pool as v10, but encode only learned entries actually selected.
// The base corpus and pending examples are already validated/encoded by their owning worker.
public static class OnlineReplay
{
    public static OnlineReplayBatch Build(EncodedExample[] baseline, EncodedExample[] recent,
        IReadOnlyList<TrainingExample> learned, long step, int sequenceLength, ValidationGuard guard,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(baseline); ArgumentNullException.ThrowIfNull(recent);
        ArgumentNullException.ThrowIfNull(learned); ArgumentNullException.ThrowIfNull(guard);
        if (step < 0 || recent.Length is < 1 or > 4 || learned.Count > 64)
            throw new ArgumentException("Invalid bounded online replay request.");
        int total = checked(baseline.Length + learned.Count);
        if (total == 0) throw new InvalidDataException("Replay corpus is empty.");
        if (sequenceLength is < 16 or > 2048) throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        var mixed = new EncodedExample[recent.Length * 2];
        Array.Copy(recent, mixed, recent.Length);
        var prepared = new Dictionary<int, EncodedExample>(); int baseSelections = 0;
        for (int i = 0; i < recent.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            // Keep the legacy deterministic selection and RNG state exactly unchanged.
            int index = (int)(((ulong)step + (uint)i) % (uint)total);
            EncodedExample example;
            if (index < baseline.Length) { example = baseline[index]; baseSelections++; }
            else
            {
                int learnedIndex = index - baseline.Length;
                if (!prepared.TryGetValue(learnedIndex, out example!))
                {
                    var row = learned[learnedIndex];
                    guard.EnsureTraining([row], ct); // Consume-time guard is mandatory, including legacy replay.
                    example = Dataset.Encode(row, sequenceLength);
                    prepared.Add(learnedIndex, example);
                }
            }
            mixed[recent.Length + i] = example;
        }
        return new(mixed, prepared.Count, baseSelections);
    }
}
