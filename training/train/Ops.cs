//https://github.com/virex-84

//Ops.cs

using System;

namespace TernaryLLM;

public static class Ops
{
    public static Tensor Add(Tensor a, Tensor b)
    {
        if (a.Data.Length != b.Data.Length)
            throw new ArgumentException("Add: shape mismatch");

        var result = new Tensor(new float[a.Data.Length], a.Shape, a.RequiresGrad || b.RequiresGrad);

        for (int i = 0; i < result.Data.Length; i++)
            result.Data[i] = a.Data[i] + b.Data[i];

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (a.RequiresGrad && a.Grad != null && result.Grad != null)
                    for (int i = 0; i < a.Grad.Length; i++)
                        a.Grad[i] += result.Grad[i];

                if (b.RequiresGrad && b.Grad != null && result.Grad != null)
                    for (int i = 0; i < b.Grad.Length; i++)
                        b.Grad[i] += result.Grad[i];
            }, a, b);
        }

        return result;
    }

    public static Tensor Mul(Tensor a, Tensor b)
    {
        if (a.Data.Length != b.Data.Length)
            throw new ArgumentException("Mul: shape mismatch");

        var result = new Tensor(new float[a.Data.Length], a.Shape, a.RequiresGrad || b.RequiresGrad);

        for (int i = 0; i < result.Data.Length; i++)
            result.Data[i] = a.Data[i] * b.Data[i];

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (a.RequiresGrad && a.Grad != null && result.Grad != null)
                    for (int i = 0; i < a.Grad.Length; i++)
                        a.Grad[i] += result.Grad[i] * b.Data[i];

                if (b.RequiresGrad && b.Grad != null && result.Grad != null)
                    for (int i = 0; i < b.Grad.Length; i++)
                        b.Grad[i] += result.Grad[i] * a.Data[i];
            }, a, b);
        }

        return result;
    }

    /// <summary>
    /// Matrix multiplication: [M, K] x [K, N] = [M, N]
    /// </summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Shape.Length != 2 || b.Shape.Length != 2)
            throw new ArgumentException("MatMul: requires 2D tensors");

        int M = a.Shape[0], K = a.Shape[1];
        int K2 = b.Shape[0], N = b.Shape[1];

        if (K != K2)
            throw new ArgumentException($"MatMul: shape mismatch {M}x{K} x {K2}x{N}");

        var result = new Tensor(new float[M * N], new[] { M, N }, a.RequiresGrad || b.RequiresGrad);

        // Forward
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += a.Data[i * K + k] * b.Data[k * N + j];
                result.Data[i * N + j] = sum;
            }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (a.RequiresGrad && a.Grad != null && result.Grad != null)
                    for (int i = 0; i < M; i++)
                        for (int k = 0; k < K; k++)
                        {
                            float g = 0;
                            for (int j = 0; j < N; j++)
                                g += result.Grad[i * N + j] * b.Data[k * N + j];
                            a.Grad[i * K + k] += g;
                        }

                if (b.RequiresGrad && b.Grad != null && result.Grad != null)
                    for (int k = 0; k < K; k++)
                        for (int j = 0; j < N; j++)
                        {
                            float g = 0;
                            for (int i = 0; i < M; i++)
                                g += result.Grad[i * N + j] * a.Data[i * K + k];
                            b.Grad[k * N + j] += g;
                        }
            }, a, b);
        }

        return result;
    }

    public static Tensor Gelu(Tensor x)
    {
        const float c = 0.7978845608f;
        const float coeff = 0.044715f;

        var result = new Tensor(new float[x.Data.Length], x.Shape, x.RequiresGrad);
        var tanhCache = new float[x.Data.Length];

        for (int i = 0; i < x.Data.Length; i++)
        {
            float v = x.Data[i];
            float inner = c * (v + coeff * v * v * v);
            float t = MathF.Tanh(inner);
            tanhCache[i] = t;
            result.Data[i] = 0.5f * v * (1f + t);
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (x.Grad == null || result.Grad == null) return;
                for (int i = 0; i < x.Data.Length; i++)
                {
                    float v = x.Data[i];
                    float t = tanhCache[i];
                    float dGelu = 0.5f * (1f + t) + 0.5f * v * (1f - t * t) * c * (1f + 3f * coeff * v * v);
                    x.Grad[i] += result.Grad[i] * dGelu;
                }
            }, x);
        }

        return result;
    }

    /// <summary>
    /// RMSNorm по последней размерности.
    /// x: [seq, dim], gamma: [dim]
    /// </summary>
    public static Tensor RmsNorm(Tensor x, Tensor gamma)
    {
        if (x.Shape.Length != 2)
            throw new ArgumentException("RmsNorm: x must be 2D [seq, dim]");
        if (gamma.Shape.Length != 1)
            throw new ArgumentException("RmsNorm: gamma must be 1D [dim]");

        int seq = x.Shape[0], dim = x.Shape[1];
        if (gamma.Data.Length != dim)
            throw new ArgumentException($"RmsNorm: gamma dim {gamma.Data.Length} != x dim {dim}");

        var result = new Tensor(new float[x.Data.Length], x.Shape, x.RequiresGrad || gamma.RequiresGrad);
        var rmsInv = new float[seq];

        // Forward
        for (int s = 0; s < seq; s++)
        {
            float sumSq = 0;
            for (int d = 0; d < dim; d++)
            {
                float v = x.Data[s * dim + d];
                sumSq += v * v;
            }
            rmsInv[s] = 1f / MathF.Sqrt(sumSq / dim + 1e-5f);

            for (int d = 0; d < dim; d++)
            {
                float normed = x.Data[s * dim + d] * rmsInv[s];
                result.Data[s * dim + d] = normed * gamma.Data[d];
            }
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (result.Grad == null) return;

                for (int s = 0; s < seq; s++)
                {
                    // Вычисляем grad gamma и grad x
                    float gradSum = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        float g = result.Grad[s * dim + d];
                        float normed = x.Data[s * dim + d] * rmsInv[s];
                        if (gamma.RequiresGrad && gamma.Grad != null)
                            gamma.Grad[d] += g * normed;
                        gradSum += g * gamma.Data[d] * normed;
                    }

                    if (x.RequiresGrad && x.Grad != null)
                    {
                        float scale = rmsInv[s] / dim;
                        for (int d = 0; d < dim; d++)
                        {
                            float g = result.Grad[s * dim + d];
                            float normed = x.Data[s * dim + d] * rmsInv[s];
                            x.Grad[s * dim + d] += scale * (dim * g * gamma.Data[d] - normed * gradSum);
                        }
                    }
                }
            }, x, gamma);
        }

        return result;
    }

    public static Tensor Softmax(Tensor x)
    {
        if (x.Shape.Length != 2)
            throw new ArgumentException("Softmax: requires 2D [rows, cols]");

        int rows = x.Shape[0], cols = x.Shape[1];
        var result = new Tensor(new float[x.Data.Length], x.Shape, x.RequiresGrad);

        for (int r = 0; r < rows; r++)
        {
            int offset = r * cols;
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++)
                if (x.Data[offset + c] > max) max = x.Data[offset + c];

            float sum = 0;
            for (int c = 0; c < cols; c++)
            {
                float e = MathF.Exp(x.Data[offset + c] - max);
                result.Data[offset + c] = e;
                sum += e;
            }

            float inv = 1f / (sum + 1e-10f);
            for (int c = 0; c < cols; c++)
                result.Data[offset + c] *= inv;
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (x.Grad == null || result.Grad == null) return;
                for (int r = 0; r < rows; r++)
                {
                    int offset = r * cols;
                    float dot = 0;
                    for (int c = 0; c < cols; c++)
                        dot += result.Data[offset + c] * result.Grad[offset + c];
                    for (int c = 0; c < cols; c++)
                        x.Grad[offset + c] += result.Data[offset + c] * (result.Grad[offset + c] - dot);
                }
            }, x);
        }

        return result;
    }

    public static Tensor LogSoftmax(Tensor x)
    {
        if (x.Shape.Length != 2)
            throw new ArgumentException("LogSoftmax: requires 2D [rows, cols]");

        int rows = x.Shape[0], cols = x.Shape[1];
        var result = new Tensor(new float[x.Data.Length], x.Shape, x.RequiresGrad);
        var probs = new float[x.Data.Length];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * cols;
            float max = float.NegativeInfinity;
            for (int c = 0; c < cols; c++)
                if (x.Data[offset + c] > max) max = x.Data[offset + c];

            float sum = 0;
            for (int c = 0; c < cols; c++)
            {
                float e = MathF.Exp(x.Data[offset + c] - max);
                probs[offset + c] = e;
                sum += e;
            }

            float logSum = MathF.Log(sum + 1e-10f);
            for (int c = 0; c < cols; c++)
            {
                probs[offset + c] /= sum + 1e-10f;
                result.Data[offset + c] = x.Data[offset + c] - max - logSum;
            }
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (x.Grad == null || result.Grad == null) return;
                for (int r = 0; r < rows; r++)
                {
                    int offset = r * cols;
                    float sumGrad = 0;
                    for (int c = 0; c < cols; c++)
                        sumGrad += result.Grad[offset + c];
                    for (int c = 0; c < cols; c++)
                        x.Grad[offset + c] += result.Grad[offset + c] - probs[offset + c] * sumGrad;
                }
            }, x);
        }

        return result;
    }

    /// <summary>
    /// Cross-entropy loss: -sum(target * log_softmax(x)) / rows
    /// target: one-hot или soft labels [rows, cols]
    /// </summary>
    public static Tensor CrossEntropy(Tensor x, Tensor target)
    {
        if (x.Shape.Length != 2 || target.Shape.Length != 2)
            throw new ArgumentException("CrossEntropy: requires 2D tensors");
        if (x.Data.Length != target.Data.Length)
            throw new ArgumentException("CrossEntropy: shape mismatch");

        int rows = x.Shape[0], cols = x.Shape[1];

        var logProbs = LogSoftmax(x);

        float loss = 0;
        for (int i = 0; i < logProbs.Data.Length; i++)
            loss -= target.Data[i] * logProbs.Data[i];
        loss /= rows;

        var result = new Tensor(new[] { loss }, new[] { 1 }, x.RequiresGrad);

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (logProbs.Grad == null) return;
                float scale = 1f / rows;
                for (int i = 0; i < logProbs.Data.Length; i++)
                    logProbs.Grad[i] += -target.Data[i] * scale;
            }, logProbs);
        }

        return result;
    }
}