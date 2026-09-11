//https://github.com/virex-84

//LlamaTrainOps.cs

using System;

namespace TernaryLLM;

public static class LlamaTrainOps
{
    public static Tensor Silu(Tensor x)
    {
        var result = new Tensor(
            new float[x.Data.Length],
            x.Shape,
            x.RequiresGrad);

        var sigmoid = new float[x.Data.Length];

        for (int i = 0; i < x.Data.Length; i++)
        {
            float v = x.Data[i];
            float s = 1f / (1f + MathF.Exp(-v));

            sigmoid[i] = s;
            result.Data[i] = v * s;
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (x.Grad == null || result.Grad == null)
                    return;

                for (int i = 0; i < x.Data.Length; i++)
                {
                    float s = sigmoid[i];
                    float derivative = s + x.Data[i] * s * (1f - s);
                    x.Grad[i] += result.Grad[i] * derivative;
                }
            }, x);
        }

        return result;
    }

    public static Tensor ApplyRoPE(
        Tensor x,
        int heads,
        int headDim,
        float theta)
    {
        if (x.Shape.Length != 2)
            throw new ArgumentException("RoPE input must be 2D [seq, dim].");

        if (headDim % 2 != 0)
            throw new InvalidOperationException("RoPE requires even headDim.");

        int seq = x.Shape[0];
        int dim = x.Shape[1];
        int expectedDim = heads * headDim;

        if (dim != expectedDim)
        {
            throw new ArgumentException(
                $"ApplyRoPE dim mismatch: expected {expectedDim}, got {dim}.");
        }

        int half = headDim / 2;

        var result = new Tensor(
            new float[x.Data.Length],
            x.Shape,
            x.RequiresGrad);

        var cos = new float[seq * half];
        var sin = new float[seq * half];

        for (int pos = 0; pos < seq; pos++)
        {
            for (int d = 0; d < half; d++)
            {
                float exponent = 2.0f * d / headDim;
                float invFreq = 1.0f / MathF.Pow(theta, exponent);

                float angle = pos * invFreq;

                int idx = pos * half + d;

                cos[idx] = MathF.Cos(angle);
                sin[idx] = MathF.Sin(angle);
            }
        }

        for (int pos = 0; pos < seq; pos++)
        {
            int posOffset = pos * dim;

            for (int h = 0; h < heads; h++)
            {
                int headStart = posOffset + h * headDim;

                for (int d = 0; d < half; d++)
                {
                    int csIdx = pos * half + d;

                    float c = cos[csIdx];
                    float s = sin[csIdx];

                    int idx1 = headStart + d;
                    int idx2 = headStart + d + half;

                    float x1 = x.Data[idx1];
                    float x2 = x.Data[idx2];

                    result.Data[idx1] = x1 * c - x2 * s;
                    result.Data[idx2] = x1 * s + x2 * c;
                }
            }
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (x.Grad == null || result.Grad == null)
                    return;

                for (int pos = 0; pos < seq; pos++)
                {
                    int posOffset = pos * dim;

                    for (int h = 0; h < heads; h++)
                    {
                        int headStart = posOffset + h * headDim;

                        for (int d = 0; d < half; d++)
                        {
                            int csIdx = pos * half + d;

                            float c = cos[csIdx];
                            float s = sin[csIdx];

                            int idx1 = headStart + d;
                            int idx2 = headStart + d + half;

                            float g1 = result.Grad[idx1];
                            float g2 = result.Grad[idx2];

                            x.Grad[idx1] += g1 * c + g2 * s;
                            x.Grad[idx2] += -g1 * s + g2 * c;
                        }
                    }
                }
            }, x);
        }

        return result;
    }
}