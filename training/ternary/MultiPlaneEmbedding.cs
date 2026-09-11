//https://github.com/virex-84

//MultiPlaneEmbedding.cs

using System;

namespace TernaryLLM;

public sealed class MultiPlaneEmbedding
{
    public MultiPlaneTritPacker Table;

    public int VocabSize => Table.Rows;
    public int EmbeddingDim => Table.Cols;

    public MultiPlaneEmbedding(MultiPlaneTritPacker table)
    {
        Table = table;
    }

    public void GetRow(int tokenId, Span<float> dst)
    {
        if (tokenId < 0 || tokenId >= VocabSize)
        {
            dst.Slice(0, EmbeddingDim).Clear();
            return;
        }

        for (int i = 0; i < EmbeddingDim; i++)
        {
            float value = 0f;
            int group = Table.GroupSize > 0 ? i / Table.GroupSize : 0;

            for (int plane = 0; plane < Table.Planes; plane++)
            {
                float scale = Table.GetScale(tokenId, group, plane);
                sbyte t = Table.Get(plane, tokenId, i);
                value += scale * t;
            }

            dst[i] = value;
        }
    }

    public float[] ForwardSequence(int[] tokens)
    {
        var result = new float[tokens.Length * EmbeddingDim];

        for (int i = 0; i < tokens.Length; i++)
        {
            GetRow(tokens[i], result.AsSpan(i * EmbeddingDim, EmbeddingDim));
        }

        return result;
    }

    public float DotWithRow(int tokenId, ReadOnlySpan<float> x)
    {
        if (tokenId < 0 || tokenId >= VocabSize)
            return float.NegativeInfinity;

        if (x.Length < EmbeddingDim)
            return float.NegativeInfinity;

        // Table — это MultiPlaneTritPacker
        // DotRowFloat корректно обрабатывает PerGroup scales
        return Table.DotRowFloat(tokenId, x);
    }

    public void SaveToStream(BinaryWriter w)
    {
        Table.SaveToStream(w);
    }

    public void LoadFromStream(BinaryReader r)
    {
        Table.LoadFromStream(r);
    }
}