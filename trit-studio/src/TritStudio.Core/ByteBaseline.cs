namespace TritStudio.Core;

public sealed record ByteBaselineScore(long Correct, long Total)
{
    public double Accuracy => Total > 0 ? (double)Correct / Total : throw new InvalidDataException("Empty reference targets.");
}

// Diagnostic baseline, NEVER used for responses or gradients. Fits only supervised TRAIN targets.
// A prediction sees the previous input byte token, not the target token or any control answers.
public sealed class ByteBaseline
{
    private readonly int[] _prediction;
    public ByteBaseline(IEnumerable<EncodedExample> training, CancellationToken ct = default)
    {
        int vocabulary = ByteTokenizer.VocabularySize;
        var counts = new long[vocabulary * vocabulary]; var global = new long[vocabulary];
        foreach (var example in training)
        {
            ct.ThrowIfCancellationRequested(); Check(example);
            for (int i = 0; i < example.Labels.Length; i++)
            {
                int target = example.Labels[i]; if (target == -100) continue;
                counts[example.Tokens[i] * vocabulary + target]++; global[target]++;
            }
        }
        if (global.Sum() == 0) throw new ArgumentException("No training targets for byte baseline.");
        int fallback = ArgMax(global, 0, vocabulary); _prediction = new int[vocabulary];
        for (int previous = 0; previous < vocabulary; previous++)
        {
            int next = ArgMax(counts, previous * vocabulary, vocabulary);
            _prediction[previous] = counts[previous * vocabulary + next] == 0 ? fallback : next;
        }
    }
    private static int ArgMax(long[] values, int start, int length)
    {
        int best = 0; for (int i = 1; i < length; i++) if (values[start + i] > values[start + best]) best = i;
        return best; // deterministic tie: smaller token ID
    }
    private static void Check(EncodedExample e)
    {
        if (e.Tokens.Length != e.Labels.Length) throw new ArgumentException("Mismatched byte baseline arrays.");
        for (int i = 0; i < e.Tokens.Length; i++)
            if (e.Tokens[i] < 0 || e.Tokens[i] >= ByteTokenizer.VocabularySize ||
                e.Labels[i] != -100 && (e.Labels[i] < 0 || e.Labels[i] >= ByteTokenizer.VocabularySize))
                throw new ArgumentException("Invalid token in byte baseline.");
    }
    public ByteBaselineScore Evaluate(IEnumerable<EncodedExample> control, CancellationToken ct = default)
    {
        long total = 0, correct = 0;
        foreach (var example in control)
        {
            ct.ThrowIfCancellationRequested(); Check(example);
            for (int i = 0; i < example.Labels.Length; i++)
            {
                int target = example.Labels[i]; if (target == -100) continue;
                total++; if (_prediction[example.Tokens[i]] == target) correct++;
            }
        }
        if (total == 0) throw new ArgumentException("No control targets for byte baseline.");
        return new(correct, total);
    }
}
