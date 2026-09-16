namespace TritStudio.Core;

// Prepares the same immutable validation inputs only once, with a fixed retained payload budget.
// This is CPU array reuse, NOT a cache of GPU tensors or validation results. Recreate for a new corpus/batch size.
public sealed class EvaluationBatchCache
{
    public const long DefaultBudgetBytes = 16L * 1024 * 1024;
    private readonly EncodedExample[] _ordered;
    private readonly int[] _lengths;
    private readonly SupervisedBatch?[] _packed;
    private readonly int _batchSize;
    private readonly long _budget;
    public long RetainedPayloadBytes { get; private set; }
    public long Hits { get; private set; }
    public long Builds { get; private set; }
    public int Count => _packed.Length;
    public EvaluationBatchCache(EncodedExample[] examples, int batchSize, int sequenceLength,
        long budgetBytes = DefaultBudgetBytes, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (examples.Length is < 1 or > Dataset.MaxExamples || batchSize is < 1 or > 32 ||
            sequenceLength is < 1 or > 2048 || budgetBytes is < 0 or > 128L * 1024 * 1024)
            throw new ArgumentException("Invalid validation cache limits.");
        // Effective lengths avoid padding ignored tails. Sorting must not change target-token weighting.
        var prepared = new (EncodedExample Example, int Length, int Index)[examples.Length];
        for (int i = 0; i < examples.Length; i++)
        {
            ct.ThrowIfCancellationRequested(); int length = BatchPlanner.EffectiveLength(examples[i]);
            if (length > sequenceLength) throw new ArgumentException("Validation target exceeds the sequence budget.");
            prepared[i] = (examples[i], length, i);
        }
        Array.Sort(prepared, (a,b) => a.Length != b.Length ? a.Length.CompareTo(b.Length) : a.Index.CompareTo(b.Index));
        ct.ThrowIfCancellationRequested();
        _ordered = prepared.Select(p => p.Example).ToArray(); _lengths = prepared.Select(p => p.Length).ToArray();
        _batchSize = batchSize; _budget = budgetBytes;
        _packed = new SupervisedBatch?[(examples.Length + batchSize - 1) / batchSize];
    }
    public SupervisedBatch Get(int index, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
        if (_packed[index] is { } cached) { Hits++; return cached; }
        int start = index * _batchSize, count = Math.Min(_batchSize, _ordered.Length - start);
        int length = _lengths[start + count - 1];
        var batch = SupervisedBatch.Build(_ordered.AsSpan(start, count).ToArray(), length, ct);
        long bytes = checked(8L * (batch.Inputs.LongLength + batch.Targets.LongLength + batch.Positions.LongLength));
        ct.ThrowIfCancellationRequested(); Builds++;
        if (bytes <= _budget - RetainedPayloadBytes) { _packed[index] = batch; RetainedPayloadBytes += bytes; }
        return batch;
    }
}
