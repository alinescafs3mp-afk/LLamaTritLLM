namespace TritStudio.Core;

public sealed record PlannedBatch(EncodedExample[] Examples, int Length, long InputTokens, long TargetTokens)
{
    public long PaddedPositions => (long)Examples.Length * Length;
    public double PaddingFraction => PaddedPositions == 0 ? 0 : 1 - (double)InputTokens / PaddedPositions;
}

// Single-owner sampler. Select a bucket by a uniformly drawn example, NOT uniformly by bucket.
// Consequently each example retains the same marginal probability, even in an uneven corpus.
public sealed class BatchPlanner
{
    private readonly EncodedExample[] _corpus;
    private readonly Dictionary<int, int[]> _buckets;
    private readonly int[] _lengths, _targetCounts;
    private readonly int _width;
    public BatchPlanner(EncodedExample[] corpus, int width = 32, CancellationToken ct = default)
    {
        if (corpus.Length == 0 || width < 1) throw new ArgumentException("Empty corpus or invalid bucket width.");
        _corpus = corpus; _width = width;
        _lengths = new int[corpus.Length]; _targetCounts = new int[corpus.Length];
        for (int i = 0; i < corpus.Length; i++)
        {
            ct.ThrowIfCancellationRequested(); _lengths[i] = EffectiveLength(corpus[i]);
            for (int j = 0; j < _lengths[i]; j++) if (corpus[i].Labels[j] != -100) _targetCounts[i]++;
        }
        _buckets = Enumerable.Range(0, corpus.Length).GroupBy(i => (_lengths[i] - 1) / width)
            .ToDictionary(g => g.Key, g => g.ToArray());
    }
    public static int EffectiveLength(EncodedExample example)
    {
        if (example.Tokens.Length != example.Labels.Length) throw new ArgumentException("Input/label length mismatch.");
        int length = Array.FindLastIndex(example.Labels, x => x != -100) + 1;
        if (length == 0) throw new ArgumentException("Example has no supervised target.");
        return length;
    }
    public PlannedBatch Select(int batchSize, SamplerRandom random, bool bucketByLength = true, EncodedExample? required = null)
    {
        if (batchSize < 1 || batchSize > 32) throw new ArgumentOutOfRangeException(nameof(batchSize));
        var selected = new EncodedExample[batchSize];
        int length = 0; long inputs = 0, targets = 0;
        void SelectIndex(int slot, int index)
        {
            selected[slot] = _corpus[index]; int used = _lengths[index];
            length = Math.Max(length, used); inputs += used; targets += _targetCounts[index];
        }
        int[]? bucket = null;
        // Online pools are deliberately mixed for replay. Do not isolate the correction from its replay examples.
        if (bucketByLength && required is null && _corpus.Length >= 32)
        {
            int anchor = random.Next(_corpus.Length);
            bucket = _buckets[(_lengths[anchor] - 1) / _width];
            SelectIndex(0, anchor);
        }
        else if (required is not null)
        {
            selected[0] = required; length = EffectiveLength(required); inputs = length;
            for (int i = 0; i < length; i++) if (required.Labels[i] != -100) targets++;
        }
        else SelectIndex(0, random.Next(_corpus.Length));
        for (int i = 1; i < selected.Length; i++)
            SelectIndex(i, bucket is null ? random.Next(_corpus.Length) : bucket[random.Next(bucket.Length)]);
        return new(selected, length, inputs, targets);
    }
}
