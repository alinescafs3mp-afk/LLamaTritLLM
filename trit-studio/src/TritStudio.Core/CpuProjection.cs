namespace TritStudio.Core;

// Managed, portable matrix-vector kernels. Independent projections share one scheduling pass,
// NOT a concatenated copy of their matrices. Each row retains the exact Dot accumulation order.
public static class CpuProjection
{
    private const int ParallelThreshold = 65536, RowsPerChunk = 16;
    private static void Validate(float[] matrix, float[] input, float[] output)
    {
        ArgumentNullException.ThrowIfNull(matrix); ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output);
        if (input.Length == 0 || output.Length == 0 || matrix.LongLength != (long)input.Length * output.Length)
            throw new ArgumentException("Matrix shape mismatch.");
        if (ReferenceEquals(input, output) || ReferenceEquals(matrix, output))
            throw new ArgumentException("Projection input and output storage must not alias.");
    }
    private static void Rows(float[] matrix, float[] input, float[] output, CancellationToken ct)
    {
        for (int row = 0; row < output.Length; row++)
        {
            if ((row & 15) == 0) ct.ThrowIfCancellationRequested();
            output[row] = ManagedInference.Session.Dot(matrix, row * input.Length, input, 0, input.Length);
        }
    }
    public static void Multiply(float[] matrix, float[] input, float[] output, ParallelOptions options)
    {
        Validate(matrix, input, output); options.CancellationToken.ThrowIfCancellationRequested();
        // Keep closure allocation in the out-of-line parallel method, not in the small sequential path.
        if (matrix.Length < ParallelThreshold || options.MaxDegreeOfParallelism == 1) Rows(matrix, input, output, options.CancellationToken);
        else ParallelRows(matrix, input, output, null, null, null, null, options);
    }
    public static void Pair(float[] a, float[] b, float[] input, float[] x, float[] y, ParallelOptions options)
    {
        Validate(a, input, x); Validate(b, input, y);
        if (ReferenceEquals(x, y) || ReferenceEquals(x, b) || ReferenceEquals(y, a)) throw new ArgumentException("Projection outputs must be disjoint.");
        options.CancellationToken.ThrowIfCancellationRequested();
        if (Math.Max(a.Length, b.Length) < ParallelThreshold || options.MaxDegreeOfParallelism == 1)
        { Rows(a, input, x, options.CancellationToken); Rows(b, input, y, options.CancellationToken); }
        else ParallelRows(a, input, x, b, y, null, null, options);
    }
    public static void Triple(float[] a, float[] b, float[] c, float[] input, float[] x, float[] y, float[] z, ParallelOptions options)
    {
        Validate(a, input, x); Validate(b, input, y); Validate(c, input, z);
        if (ReferenceEquals(x, y) || ReferenceEquals(x, z) || ReferenceEquals(y, z) ||
            ReferenceEquals(x, b) || ReferenceEquals(x, c) || ReferenceEquals(y, a) || ReferenceEquals(y, c) || ReferenceEquals(z, a) || ReferenceEquals(z, b))
            throw new ArgumentException("Projection outputs must be disjoint.");
        options.CancellationToken.ThrowIfCancellationRequested();
        if (Math.Max(a.Length, Math.Max(b.Length, c.Length)) < ParallelThreshold || options.MaxDegreeOfParallelism == 1)
        { Rows(a, input, x, options.CancellationToken); Rows(b, input, y, options.CancellationToken); Rows(c, input, z, options.CancellationToken); }
        else ParallelRows(a, input, x, b, y, c, z, options);
    }
    private static void ParallelRows(float[] a, float[] input, float[] x, float[]? b, float[]? y, float[]? c, float[]? z, ParallelOptions options)
    {
        int first = x.Length, second = first + (y?.Length ?? 0), total = second + (z?.Length ?? 0), n = input.Length;
        Parallel.For(0, (total + RowsPerChunk - 1) / RowsPerChunk, options, chunk =>
        {
            int end = Math.Min(total, chunk * RowsPerChunk + RowsPerChunk);
            for (int row = chunk * RowsPerChunk; row < end; row++)
            {
                options.CancellationToken.ThrowIfCancellationRequested();
                if (row < first) x[row] = ManagedInference.Session.Dot(a, row * n, input, 0, n);
                else if (row < second) y![row - first] = ManagedInference.Session.Dot(b!, (row - first) * n, input, 0, n);
                else z![row - second] = ManagedInference.Session.Dot(c!, (row - second) * n, input, 0, n);
            }
        });
    }
}
