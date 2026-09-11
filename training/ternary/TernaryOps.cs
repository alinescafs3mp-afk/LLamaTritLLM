//https://github.com/virex-84

//TernaryOps.cs

using System;

namespace TernaryLLM;

public static class TernaryOps
{
    public static float[] Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("a.Length != b.Length");

        var result = new float[a.Length];

        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];

        return result;
    }

    public static float[] RmsNorm(
        ReadOnlySpan<float> input,
        int rows,
        int dim,
        float[] gamma)
    {
        if (input.Length < rows * dim)
            throw new ArgumentException("input too small");

        var output = new float[rows * dim];

        for (int r = 0; r < rows; r++)
        {
            int offset = r * dim;

            float meanSq = 0f;

            for (int i = 0; i < dim; i++)
            {
                float v = input[offset + i];
                meanSq += v * v;
            }

            meanSq /= dim;

            float inv = 1f / MathF.Sqrt(meanSq + 1e-5f);

            for (int i = 0; i < dim; i++)
            {
                float g = i < gamma.Length ? gamma[i] : 1f;
                output[offset + i] = input[offset + i] * inv * g;
            }
        }

        return output;
    }

    public static float[] Requantize(
        float[] data,
        int rows,
        int dim,
        int planes,
        int groupSize,
        TritScaleMode scaleMode,
        float thresholdScaleFraction = 0.0f)
    {
        var tensor = MultiPlaneTritPacker.FromFloat(
            data,
            rows,
            dim,
            planes,
            groupSize,
            scaleMode,
            thresholdScaleFraction);

        var result = new float[data.Length];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                int group = groupSize > 0 ? c / groupSize : 0;
                float value = 0f;

                for (int plane = 0; plane < planes; plane++)
                {
                    float scale = tensor.GetScale(r, group, plane);
                    sbyte t = tensor.Get(plane, r, c);
                    value += scale * t;
                }

                result[r * dim + c] = value;
            }
        }

        return result;
    }
}