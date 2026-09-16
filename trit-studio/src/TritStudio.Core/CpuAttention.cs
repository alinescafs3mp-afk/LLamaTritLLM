using System.Numerics;
namespace TritStudio.Core;

// Portable time-major attention. SIMD accelerates only the value-channel accumulation;
// score dot products, causal history, softmax and temporal summation order are unchanged.
public static class CpuAttention
{
    public static void Compute(float[] query, float[] keys, float[] values, float[] scores, float[] output,
        int positions, int heads, int kvHeads, int headDimension, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(query); ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values); ArgumentNullException.ThrowIfNull(scores); ArgumentNullException.ThrowIfNull(output);
        if (positions is < 1 or > 2048 || heads is < 1 or > 16 || kvHeads < 1 || kvHeads > heads || heads % kvHeads != 0 || headDimension is < 1 or > 512)
            throw new ArgumentException("Invalid attention shape.");
        int dimension = checked(heads * headDimension), kvDimension = checked(kvHeads * headDimension);
        if (query.Length != dimension || output.Length != dimension || keys.Length < positions * kvDimension ||
            values.Length < positions * kvDimension || scores.Length < positions)
            throw new ArgumentException("Attention buffers do not cover the declared shape.");
        if (ReferenceEquals(output, scores) || ReferenceEquals(output, query) || ReferenceEquals(output, keys) || ReferenceEquals(output, values) ||
            ReferenceEquals(scores, query) || ReferenceEquals(scores, keys) || ReferenceEquals(scores, values))
            throw new ArgumentException("Attention scratch/output must not alias input or each other.");
        float scale = 1 / MathF.Sqrt(headDimension);
        int width = Vector<float>.Count;
        int vectorEnd = Vector.IsHardwareAccelerated ? headDimension - headDimension % width : 0;
        for (int h = 0; h < heads; h++)
        {
            ct.ThrowIfCancellationRequested(); int queryOffset = h * headDimension, keyHeadOffset = h / (heads / kvHeads) * headDimension;
            float max = float.NegativeInfinity;
            for (int t = 0; t < positions; t++)
            {
                if ((t & 15) == 0) ct.ThrowIfCancellationRequested();
                float score = ManagedInference.Session.Dot(query, queryOffset, keys, t * kvDimension + keyHeadOffset, headDimension) * scale;
                scores[t] = score; max = MathF.Max(max, score);
            }
            float sum = 0;
            for (int t = 0; t < positions; t++) { scores[t] = MathF.Exp(scores[t] - max); sum += scores[t]; }
            if (!float.IsFinite(sum) || sum <= 0) throw new ArithmeticException("Nonfinite attention scores.");
            float inverse = 1 / sum;
            Array.Clear(output, queryOffset, headDimension);
            for (int t = 0; t < positions; t++)
            {
                if ((t & 15) == 0) ct.ThrowIfCancellationRequested();
                float probability = scores[t] * inverse; int valueOffset = t * kvDimension + keyHeadOffset, d = 0;
                // No new arrays or per-head tasks. Scalar tail supports any valid head width.
                if (vectorEnd > 0)
                {
                    var weight = new Vector<float>(probability);
                    for (; d < vectorEnd; d += width)
                    {
                        var contribution = weight * new Vector<float>(values, valueOffset + d);
                        (new Vector<float>(output, queryOffset + d) + contribution).CopyTo(output, queryOffset + d);
                    }
                }
                for (; d < headDimension; d++) output[queryOffset + d] += probability * values[valueOffset + d];
            }
        }
    }
}
