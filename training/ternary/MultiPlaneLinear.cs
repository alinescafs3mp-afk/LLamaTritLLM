//https://github.com/virex-84

//MultiPlaneLinear.sc

using System;

namespace TernaryLLM;

public sealed class MultiPlaneLinear
{
    public MultiPlaneTritPacker Weight;
    public float[] Bias;

    public int OutFeatures => Weight.Rows;
    public int InFeatures => Weight.Cols;

    public MultiPlaneLinear(MultiPlaneTritPacker weight, bool useBias = true)
    {
        Weight = weight;
        Bias = useBias ? new float[OutFeatures] : Array.Empty<float>();
    }

    public float[] Forward(ReadOnlySpan<float> input)
    {
        return ForwardSequence(input, 1);
    }

    public float[] ForwardSequence(ReadOnlySpan<float> input, int rows)
    {
        if (input.Length < rows * InFeatures)
            throw new ArgumentException("input too small");

        var output = new float[rows * OutFeatures];

        for (int r = 0; r < rows; r++)
        {
            var inRow = input.Slice(r * InFeatures, InFeatures);

            for (int o = 0; o < OutFeatures; o++)
            {
                float b = o < Bias.Length ? Bias[o] : 0f;
                output[r * OutFeatures + o] = b + Weight.DotRowFloat(o, inRow);
            }
        }

        return output;
    }

    public void SaveToStream(BinaryWriter w)
    {
        Weight.SaveToStream(w);
        TernarySerializer.WriteFloats(w, Bias);
    }

    public void LoadFromStream(BinaryReader r)
    {
        Weight.LoadFromStream(r);
        Bias = TernarySerializer.ReadFloats(r);
    }
}