namespace TritStudio.Core;

// Prepare only real targets for the output projection/loss. History still participates in causal attention.
// A flat position addresses [batch, sequence] in row-major order. No targets are dropped or reweighted.
public sealed record SupervisedBatch(long[] Inputs, long[] Targets, long[] Positions, int Rows, int Length)
{
    public static SupervisedBatch Build(EncodedExample[] examples, int length, CancellationToken ct = default)
    {
        if (examples.Length is < 1 or > 32 || length is < 1 or > 2048) throw new ArgumentException("Invalid batch shape.");
        int count = 0;
        foreach (var example in examples)
        {
            ct.ThrowIfCancellationRequested();
            if (example.Tokens.Length != example.Labels.Length || example.Tokens.Length == 0)
                throw new ArgumentException("Input/label shape mismatch.");
            for (int i = 0; i < example.Labels.Length; i++)
            {
                int label = example.Labels[i];
                if (label != -100 && (label < 0 || label >= ByteTokenizer.VocabularySize))
                    throw new ArgumentException("Invalid target token before device transfer.");
                if (example.Tokens[i] < 0 || example.Tokens[i] >= ByteTokenizer.VocabularySize)
                    throw new ArgumentException("Invalid input token before device transfer.");
                if (label != -100)
                {
                    if (i >= length) throw new ArgumentException("Batch would truncate a supervised target.");
                    count++;
                }
            }
        }
        if (count == 0) throw new ArgumentException("Batch has no supervised targets.");
        var inputs = new long[checked(examples.Length * length)];
        var targets = new long[count]; var positions = new long[count]; int at = 0;
        for (int row = 0; row < examples.Length; row++)
        {
            ct.ThrowIfCancellationRequested(); var example = examples[row];
            for (int col = 0; col < Math.Min(length, example.Tokens.Length); col++)
            {
                int index = row * length + col; inputs[index] = example.Tokens[col];
                if (example.Labels[col] == -100) continue;
                positions[at] = index; targets[at++] = example.Labels[col];
            }
        }
        return new(inputs, targets, positions, examples.Length, length);
    }
}
