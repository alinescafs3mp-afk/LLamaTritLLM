//https://github.com/virex-84

//TernaryLlamaConfig.cs

using System;

namespace TernaryLLM;

public sealed class TernaryLlamaConfig
{
    public int VocabSize = 16000;

    public int EmbeddingDim = 128;

    /// <summary>
    /// Hidden dim для SwiGLU FFN.
    /// Обычно его подбирают меньше, чем 4*Emb, потому что появляется ещё Gate projection.
    /// </summary>
    public int HiddenDim = 256;

    public int NumHeads = 4;

    /// <summary>
    /// Количество KV-heads для Grouped-Query Attention.
    /// Если NumKVHeads == NumHeads, получается обычный MHA.
    /// </summary>
    public int NumKVHeads = 2;

    public int NumLayers = 2;

    public int MaxSeqLen = 512;

    public float RopeTheta = 10000f;

    public int EmbeddingPlanes = 2;
    public int AttentionQkvPlanes = 2;
    public int AttentionOutPlanes = 2;

    public int FfnGatePlanes = 2;
    public int FfnUpPlanes = 2;
    public int FfnDownPlanes = 2;

    public int LmHeadPlanes = 2;
    public int ActivationPlanes = 2;

    public int GroupSize = 32;
    public TritScaleMode ScaleMode = TritScaleMode.PerGroup;

    public AttentionMode AttentionMode = AttentionMode.Softmax;
    public int TopK = 8;

    public bool QuantizeActivations = false;
    public bool TiedOutput = true;

    public int HeadDim => NumHeads > 0 ? EmbeddingDim / NumHeads : 0;

    public int KVDim => NumKVHeads * HeadDim;

    public void SetUniformPlanes(int planes)
    {
        EmbeddingPlanes = planes;
        AttentionQkvPlanes = planes;
        AttentionOutPlanes = planes;

        FfnGatePlanes = planes;
        FfnUpPlanes = planes;
        FfnDownPlanes = planes;

        LmHeadPlanes = planes;
        ActivationPlanes = planes;
    }

    public void Validate()
    {
        if (VocabSize <= 0)
            throw new InvalidOperationException("VocabSize must be > 0.");

        if (EmbeddingDim <= 0)
            throw new InvalidOperationException("EmbeddingDim must be > 0.");

        if (HiddenDim <= 0)
            throw new InvalidOperationException("HiddenDim must be > 0.");

        if (NumHeads <= 0)
            throw new InvalidOperationException("NumHeads must be > 0.");

        if (NumKVHeads <= 0)
            throw new InvalidOperationException("NumKVHeads must be > 0.");

        if (NumKVHeads > NumHeads)
            throw new InvalidOperationException("NumKVHeads must be <= NumHeads.");

        if (NumHeads % NumKVHeads != 0)
            throw new InvalidOperationException("NumHeads must be divisible by NumKVHeads for GQA.");

        if (EmbeddingDim % NumHeads != 0)
            throw new InvalidOperationException("EmbeddingDim must be divisible by NumHeads.");

        if (HeadDim % 2 != 0)
            throw new InvalidOperationException("HeadDim must be even for RoPE.");

        if (NumLayers < 0)
            throw new InvalidOperationException("NumLayers must be >= 0.");

        if (MaxSeqLen <= 0)
            throw new InvalidOperationException("MaxSeqLen must be > 0.");
    }
}