namespace TritStudio.Core;

public sealed record WeightShape(string Name, int Rows, int Cols, bool Quantized)
{
    public int Count => checked(Rows * Cols);
}

public static class WeightLayout
{
    public static IReadOnlyList<WeightShape> For(ModelConfig c)
    {
        c.Validate();
        var a = new List<WeightShape> { new("embedding", ByteTokenizer.VocabularySize, c.Dimension, true) };
        for (int l = 0; l < c.Layers; l++)
        {
            string p = $"block.{l}.";
            a.Add(new(p + "norm1", 1, c.Dimension, false));
            a.Add(new(p + "q", c.Dimension, c.Dimension, true));
            a.Add(new(p + "k", c.KvDimension, c.Dimension, true));
            a.Add(new(p + "v", c.KvDimension, c.Dimension, true));
            a.Add(new(p + "out", c.Dimension, c.Dimension, true));
            a.Add(new(p + "norm2", 1, c.Dimension, false));
            a.Add(new(p + "gate", c.HiddenDimension, c.Dimension, true));
            a.Add(new(p + "up", c.HiddenDimension, c.Dimension, true));
            a.Add(new(p + "down", c.Dimension, c.HiddenDimension, true));
        }
        a.Add(new("norm", 1, c.Dimension, false));
        return a;
    }
}

public sealed class WeightSet
{
    public ModelConfig Config { get; }
    public IReadOnlyDictionary<string, float[]> Values { get; }
    public WeightSet(ModelConfig config, IReadOnlyDictionary<string, float[]> values)
    {
        config.Validate(); Config = config; Values = values;
        foreach (var s in WeightLayout.For(config))
            if (!values.TryGetValue(s.Name, out var a) || a.Length != s.Count || a.Any(x => !float.IsFinite(x)))
                throw new InvalidDataException($"Invalid tensor: {s.Name}");
        if (values.Count != WeightLayout.For(config).Count) throw new InvalidDataException("Unexpected tensor entries.");
    }
    public static WeightSet Initialize(ModelConfig c, int threads = 1, CancellationToken ct = default)
    {
        var shapes = WeightLayout.For(c); var arrays = new float[shapes.Count][];
        void InitializeTensor(int index)
        {
            ct.ThrowIfCancellationRequested(); var shape = shapes[index];
            // Per-tensor RNG is deterministic regardless of task scheduling and avoids equal layer initializations.
            var rng = new Random(unchecked(c.Seed + index * 9973)); var a = new float[shape.Count];
            if (!shape.Quantized) Array.Fill(a, 1f);
            else
            {
                float std = shape.Name == "embedding" ? 0.02f : MathF.Sqrt(1f / shape.Cols);
                // Box-Muller yields two samples per logarithm/square-root.
                for (int i = 0; i < a.Length; i += 2)
                {
                    if ((i & 16383) == 0) ct.ThrowIfCancellationRequested();
                    double radius = Math.Sqrt(-2 * Math.Log(Math.Max(rng.NextDouble(), 1e-15)));
                    double angle = 2 * Math.PI * rng.NextDouble();
                    a[i] = std * (float)(radius * Math.Cos(angle));
                    if (i + 1 < a.Length) a[i + 1] = std * (float)(radius * Math.Sin(angle));
                }
            }
            arrays[index] = a;
        }
        if (c.ParameterCount < 200000 || threads <= 1) for (int i = 0; i < shapes.Count; i++) InitializeTensor(i);
        else Parallel.For(0, shapes.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(threads, 1, Math.Max(1, Environment.ProcessorCount)), CancellationToken = ct }, InitializeTensor);
        return new(c, shapes.Select((s, i) => (s.Name, Data: arrays[i])).ToDictionary(x => x.Name, x => x.Data));
    }
    public long CountChanged(WeightSet other, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(other);
        if (Config != other.Config) throw new ArgumentException("Cannot compare weights of different model configurations.");
        var shapes = WeightLayout.For(Config); long count = 0;
        long CountTensor(int index)
        {
            ct.ThrowIfCancellationRequested(); var a = Values[shapes[index].Name]; var b = other.Values[shapes[index].Name];
            if (ReferenceEquals(a, b)) return 0;
            long changed = 0;
            for (int i = 0; i < a.Length; i++)
            { if ((i & 16383) == 0) ct.ThrowIfCancellationRequested(); if (a[i] != b[i]) changed++; }
            return changed;
        }
        if (Config.ParameterCount < 200000 || threads <= 1)
            for (int i = 0; i < shapes.Count; i++) count += CountTensor(i);
        else
            Parallel.For(0, shapes.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(threads, 1, Math.Max(1, Environment.ProcessorCount)), CancellationToken = ct },
                i => Interlocked.Add(ref count, CountTensor(i)));
        return count;
    }
}
