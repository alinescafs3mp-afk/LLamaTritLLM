namespace TritStudio.Core;

// One instance per generation. Buffers are reused, not shared across concurrent callers.
public sealed class TokenSampler
{
    private readonly float[] _logits = new float[ByteTokenizer.VocabularySize];
    private readonly int[] _indices = new int[ByteTokenizer.VocabularySize];
    private readonly double[] _probabilities = new double[ByteTokenizer.VocabularySize];
    private readonly IComparer<int> _order;
    public TokenSampler() => _order = Comparer<int>.Create((a, b) =>
    { int order = _logits[b].CompareTo(_logits[a]); return order != 0 ? order : a.CompareTo(b); });
    public int Sample(float[] source, HashSet<int> seen, SamplingOptions options, Random rng, Func<int, bool>? allowed = null)
    {
        if (source.Length != ByteTokenizer.VocabularySize) throw new ArithmeticException("Invalid vocabulary length.");
        var o = options.Clamp(); int first = -1;
        for (int i = 0; i < source.Length; i++)
        {
            float value = source[i];
            if (!float.IsFinite(value)) throw new ArithmeticException("Invalid inference logits.");
            _indices[i] = i;
            if ((allowed is not null && !allowed(i)) || (i < ByteTokenizer.Offset && i != ByteTokenizer.Eos)) value = float.NegativeInfinity;
            else if (seen.Contains(i)) value = value < 0 ? (float)(value * o.RepetitionPenalty) : (float)(value / o.RepetitionPenalty);
            if (float.IsNaN(value) || float.IsPositiveInfinity(value)) throw new ArithmeticException("Invalid inference logits.");
            _logits[i] = value;
            if (first < 0 || value > _logits[first]) first = i;
        }
        // Greedy decoding is O(vocabulary), preserving lowest-index tie resolution.
        // Stochastic decoding retains the same sorted order and RNG consumption.
        if (!float.IsFinite(_logits[first])) throw new ArithmeticException("No valid output token.");
        if (o.Temperature < 1e-6) return first;
        Array.Sort(_indices, _order); first = _indices[0];
        double sum = 0;
        for (int i = 0; i < o.TopK; i++)
        { _probabilities[i] = Math.Exp((_logits[_indices[i]] - _logits[first]) / o.Temperature); sum += _probabilities[i]; }
        int keep = 0; double cumulative = 0;
        do { cumulative += _probabilities[keep++] / sum; } while (keep < o.TopK && cumulative < o.TopP);
        double kept = 0; for (int i = 0; i < keep; i++) kept += _probabilities[i];
        double draw = rng.NextDouble() * kept;
        for (int i = 0; i < keep; i++) { draw -= _probabilities[i]; if (draw <= 0) return _indices[i]; }
        return _indices[keep - 1];
    }
}
