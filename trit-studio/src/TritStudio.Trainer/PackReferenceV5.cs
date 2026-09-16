// V5 baseline retained ONLY for isolated correctness/performance comparison.
// Derived from the project's MIT-licensed quantizer; not used by the application training path.
using TritStudio.Core;
namespace TritStudio.Trainer;
internal static class PackReferenceV5
{
    public static PackedPlane[] Pack(float[] weights, int groupSize, int planes, float threshold, int threads = 1)
    {
        if (groupSize is < 1 or > 128 || planes is < 1 or > 3 || !float.IsFinite(threshold) || threshold is < 0 or > 1 || weights.Any(x => !float.IsFinite(x)))
            throw new ArgumentException("Invalid quantizer arguments.");
        if (weights.Length % groupSize != 0) throw new ArgumentException("Groups must exactly divide tensor size.");
        int groups = weights.Length / groupSize;
        var residual = (float[])weights.Clone();
        var result = new PackedPlane[planes];
        int bytesPerGroup = (groupSize + 4) / 5;
        for (int p = 0; p < planes; p++)
        {
            var scales = new float[groups]; var trits = new byte[groups * bytesPerGroup];
            void PackGroup(int g)
            {
                int begin = g * groupSize; float sum = 0;
                for (int j = 0; j < groupSize; j++) sum += MathF.Abs(residual[begin + j]);
                float scale = MathF.Max(sum / groupSize, 1e-8f);
                if (!float.IsFinite(scale)) throw new ArithmeticException("Quantizer scale overflow.");
                scales[g] = scale;
                for (int j = 0; j < groupSize; j += 5)
                {
                    int packed = 0, factor = 1;
                    for (int z = 0; z < 5 && j + z < groupSize; z++)
                    {
                        int i = begin + j + z;
                        int t = MathF.Abs(residual[i]) > scale * threshold ? Math.Sign(residual[i]) : 0;
                        packed += (t + 1) * factor; factor *= 3;
                        residual[i] -= scale * t;
                    }
                    trits[g * bytesPerGroup + j / 5] = (byte)packed;
                }
            }
            if (threads <= 1 || weights.Length < 65536) for (int g = 0; g < groups; g++) PackGroup(g);
            else Parallel.For(0, groups, new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(threads, 1, Math.Max(1, Environment.ProcessorCount)) }, PackGroup);
            result[p] = new(scales, trits);
        }
        return result;
    }
}
