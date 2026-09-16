// Adapted from the mean-absolute residual multi-plane approach in
// virex-84/LLamaTritLLM (MIT). The on-disk V1 byte order is unchanged.
namespace TritStudio.Core;

public sealed record PackedPlane(float[] Scales, byte[] Trits);
public static class TernaryQuantizer
{
    private static readonly int[] PowersOfThree = [1, 3, 9, 27, 81, 243];
    private const int GroupsPerTask = 64;
    public static PackedPlane[] Pack(float[] weights, int groupSize, int planes, float threshold, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (groupSize is < 1 or > 128 || planes is < 1 or > 3 || !float.IsFinite(threshold) || threshold is < 0 or > 1)
            throw new ArgumentException("Invalid quantizer arguments.");
        if (weights.Length % groupSize != 0) throw new ArgumentException("Groups must exactly divide tensor size.");
        int groups = weights.Length / groupSize, bytesPerGroup = (groupSize + 4) / 5;
        var result = Enumerable.Range(0, planes).Select(_ => new PackedPlane(new float[groups], new byte[checked(groups * bytesPerGroup)])).ToArray();
        // Finish all planes of a group while its residual is in a tiny worker-local buffer.
        // No full-tensor residual clone, and no new parallel barrier for every plane.
        void PackRange(int task)
        {
            Span<float> residual = stackalloc float[128];
            int end = Math.Min(groups, (task + 1) * GroupsPerTask);
            for (int g = task * GroupsPerTask; g < end; g++)
            {
                ct.ThrowIfCancellationRequested();
                weights.AsSpan(g * groupSize, groupSize).CopyTo(residual);
                for (int j = 0; j < groupSize; j++) if (!float.IsFinite(residual[j])) throw new ArgumentException("Nonfinite quantizer input.");
                for (int p = 0; p < planes; p++)
                {
                    float sum = 0;
                    for (int j = 0; j < groupSize; j++) sum += MathF.Abs(residual[j]);
                    float scale = MathF.Max(sum / groupSize, 1e-8f);
                    if (!float.IsFinite(scale)) throw new ArithmeticException("Quantizer scale overflow.");
                    result[p].Scales[g] = scale;
                    float cutoff = scale * threshold;
                    for (int j = 0; j < groupSize; j += 5)
                    {
                        int packed = 0, factor = 1;
                        for (int z = 0; z < 5 && j + z < groupSize; z++)
                        {
                            int i = j + z;
                            int t = MathF.Abs(residual[i]) > cutoff ? Math.Sign(residual[i]) : 0;
                            packed += (t + 1) * factor; factor *= 3;
                            if (p + 1 < planes) residual[i] -= scale * t;
                        }
                        result[p].Trits[g * bytesPerGroup + j / 5] = (byte)packed;
                    }
                }
            }
        }
        RunGroups(groups, weights.Length, threads, ct, PackRange);
        return result;
    }
    public static float[] Unpack(PackedPlane[] planes, int length, int groupSize, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (groupSize is < 1 or > 128 || length < 0 || length % groupSize != 0 || planes.Length is < 1 or > 3)
            throw new ArgumentException("Invalid packed tensor shape.");
        int groups = length / groupSize, bytesPerGroup = (groupSize + 4) / 5;
        foreach (var plane in planes)
            if (plane.Scales.Length != groups || plane.Trits.Length != checked(groups * bytesPerGroup))
                throw new InvalidDataException("Packed tensor length mismatch.");
        var result = new float[length];
        void UnpackRange(int task)
        {
            int end = Math.Min(groups, (task + 1) * GroupsPerTask);
            for (int g = task * GroupsPerTask; g < end; g++)
            {
                ct.ThrowIfCancellationRequested();
                foreach (var plane in planes)
                {
                    float scale = plane.Scales[g];
                    if (!float.IsFinite(scale) || scale < 0) throw new InvalidDataException("Invalid scale.");
                    for (int j = 0; j < groupSize; j += 5)
                    {
                        int packed = plane.Trits[g * bytesPerGroup + j / 5];
                        int valid = Math.Min(5, groupSize - j);
                        if (packed >= PowersOfThree[valid]) throw new InvalidDataException("Invalid base-3 data.");
                        for (int z = 0; z < valid; z++)
                        { result[g * groupSize + j + z] += scale * (packed % 3 - 1); packed /= 3; }
                    }
                }
                for (int j = 0; j < groupSize; j++)
                    if (!float.IsFinite(result[g * groupSize + j])) throw new InvalidDataException("Nonfinite decompressed weights.");
            }
        }
        RunGroups(groups, length, threads, ct, UnpackRange);
        return result;
    }
    private static void RunGroups(int groups, int length, int threads, CancellationToken ct, Action<int> run)
    {
        int tasks = (groups + GroupsPerTask - 1) / GroupsPerTask;
        if (threads <= 1 || length < 65536) for (int i = 0; i < tasks; i++) run(i);
        else Parallel.For(0, tasks, new ParallelOptions { MaxDegreeOfParallelism = Math.Clamp(threads, 1, Math.Max(1, Environment.ProcessorCount)), CancellationToken = ct }, run);
        ct.ThrowIfCancellationRequested();
    }
    public static WeightSet Quantize(WeightSet master, int threads = 1, CancellationToken ct = default)
    {
        var c = master.Config; var values = new Dictionary<string, float[]>();
        foreach (var s in WeightLayout.For(c))
        {
            ct.ThrowIfCancellationRequested();
            values[s.Name] = s.Quantized
                ? Unpack(Pack(master.Values[s.Name], c.GroupSize, c.Planes, c.Threshold, threads, ct), s.Count, c.GroupSize, threads, ct)
                : (float[])master.Values[s.Name].Clone();
        }
        return new(c, values);
    }
}
