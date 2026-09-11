//https://github.com/virex-84

//LlamaOps.cs

using System;

namespace TernaryLLM;

public static class LlamaOps
{
    public static void Silu(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }

    public static void ApplyRoPE(
        float[] q,
        float[] k,
        int seq,
        int startPos,
        int qHeads,
        int kvHeads,
        int headDim,
        float theta)
    {
        if (headDim % 2 != 0)
            throw new InvalidOperationException("RoPE requires even headDim.");

        int half = headDim / 2;
        int qDim = qHeads * headDim;
        int kDim = kvHeads * headDim;

        var invFreq = new float[half];

        for (int d = 0; d < half; d++)
        {
            float exponent = 2.0f * d / headDim;
            invFreq[d] = 1.0f / MathF.Pow(theta, exponent);
        }

        for (int pos = 0; pos < seq; pos++)
        {
            int absPos = startPos + pos;

            for (int d = 0; d < half; d++)
            {
                float angle = absPos * invFreq[d];
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                int qPosOffset = pos * qDim;

                for (int h = 0; h < qHeads; h++)
                {
                    int idx1 = qPosOffset + h * headDim + d;
                    int idx2 = idx1 + half;

                    float x1 = q[idx1];
                    float x2 = q[idx2];

                    q[idx1] = x1 * cos - x2 * sin;
                    q[idx2] = x1 * sin + x2 * cos;
                }

                int kPosOffset = pos * kDim;

                for (int h = 0; h < kvHeads; h++)
                {
                    int idx1 = kPosOffset + h * headDim + d;
                    int idx2 = idx1 + half;

                    float x1 = k[idx1];
                    float x2 = k[idx2];

                    k[idx1] = x1 * cos - x2 * sin;
                    k[idx2] = x1 * sin + x2 * cos;
                }
            }
        }
    }
}