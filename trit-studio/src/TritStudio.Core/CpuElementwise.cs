using System.Numerics;
namespace TritStudio.Core;

// Portable channel-wise operations. Keep the existing Dot reduction for RMSNorm and
// the original multiply order. No approximation, joined buffers, added tasks or SIMD dependency.
public static class CpuElementwise
{
    public static void RmsNorm(float[] input, float[] gamma, float[] output, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gamma); ArgumentNullException.ThrowIfNull(output);
        if (input.Length == 0 || gamma.Length != input.Length || output.Length != input.Length)
            throw new ArgumentException("RMSNorm shapes must agree and be nonempty.");
        // Exact whole-array aliasing is supported. The reduction completes before any output write.
        float inv = 1 / MathF.Sqrt(ManagedInference.Session.Dot(input, 0, input, 0, input.Length) / input.Length + 1e-5f);
        if (!float.IsFinite(inv) || inv <= 0) throw new ArithmeticException("Nonfinite or overflowing RMSNorm input.");
        int i = 0, width = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated)
        {
            var scale = new Vector<float>(inv);
            for (; i <= input.Length - width; i += width)
            {
                if ((i & 255) == 0) ct.ThrowIfCancellationRequested();
                ((new Vector<float>(input, i) * scale) * new Vector<float>(gamma, i)).CopyTo(output, i);
            }
        }
        for (; i < input.Length; i++) { if ((i & 255) == 0) ct.ThrowIfCancellationRequested(); output[i] = input[i] * inv * gamma[i]; }
    }
    public static void AddInPlace(float[] target, float[] value, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(target); ArgumentNullException.ThrowIfNull(value);
        if (target.Length != value.Length) throw new ArgumentException("Residual shapes must agree.");
        int i = 0, width = Vector<float>.Count;
        if (Vector.IsHardwareAccelerated)
            for (; i <= target.Length - width; i += width)
            {
                if ((i & 255) == 0) ct.ThrowIfCancellationRequested();
                (new Vector<float>(target, i) + new Vector<float>(value, i)).CopyTo(target, i);
            }
        for (; i < target.Length; i++) { if ((i & 255) == 0) ct.ThrowIfCancellationRequested(); target[i] += value[i]; }
    }
}
