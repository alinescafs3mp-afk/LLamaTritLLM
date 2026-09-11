//https://github.com/virex-84

//TernaryLlamaBlock.cs

using System;

namespace TernaryLLM;

public sealed class TernaryLlamaBlock
{
    private readonly int _emb;

    private float[] _norm1;
    private float[] _norm2;

    private readonly MultiPlaneLlamaAttention _attention;
    private readonly MultiPlaneSwiGLUFFN _ffn;

    private readonly bool _quantizeActivations;
    private readonly int _activationPlanes;
    private readonly int _groupSize;
    private readonly TritScaleMode _scaleMode;

    public float[] Norm1 => _norm1;
    public float[] Norm2 => _norm2;

    public MultiPlaneLlamaAttention Attention => _attention;
    public MultiPlaneSwiGLUFFN Ffn => _ffn;

    public TernaryLlamaBlock(
        int emb,
        int heads,
        int kvHeads,
        int hidden,
        int attnQkvPlanes,
        int attnOutPlanes,
        int ffnGatePlanes,
        int ffnUpPlanes,
        int ffnDownPlanes,
        int activationPlanes,
        int groupSize,
        TritScaleMode scaleMode,
        AttentionMode attentionMode,
        int topK,
        bool quantizeActivations,
        float ropeTheta)
    {
        _emb = emb;

        _norm1 = new float[emb];
        _norm2 = new float[emb];

        Array.Fill(_norm1, 1f);
        Array.Fill(_norm2, 1f);

        _attention = new MultiPlaneLlamaAttention(
            emb,
            heads,
            kvHeads,
            attnQkvPlanes,
            attnOutPlanes,
            groupSize,
            scaleMode,
            attentionMode,
            topK,
            ropeTheta);

        _ffn = new MultiPlaneSwiGLUFFN(
            emb,
            hidden,
            ffnGatePlanes,
            ffnUpPlanes,
            ffnDownPlanes,
            activationPlanes,
            groupSize,
            scaleMode,
            quantizeActivations);

        _quantizeActivations = quantizeActivations;
        _activationPlanes = activationPlanes;
        _groupSize = groupSize;
        _scaleMode = scaleMode;
    }

    public float[] Forward(ReadOnlySpan<float> input, int seq, LlamaKVCache cache)
    {
        float[] norm1 = TernaryOps.RmsNorm(input, seq, _emb, _norm1);
        float[] attn = _attention.Forward(norm1, seq, cache);

        float[] residual = TernaryOps.Add(input, attn);

        if (_quantizeActivations)
        {
            residual = TernaryOps.Requantize(
                residual,
                seq,
                _emb,
                _activationPlanes,
                _groupSize,
                _scaleMode);
        }

        float[] norm2 = TernaryOps.RmsNorm(residual, seq, _emb, _norm2);
        float[] ffn = _ffn.Forward(norm2, seq);

        float[] output = TernaryOps.Add(residual, ffn);

        if (_quantizeActivations)
        {
            output = TernaryOps.Requantize(
                output,
                seq,
                _emb,
                _activationPlanes,
                _groupSize,
                _scaleMode);
        }

        return output;
    }

    public void SaveToStream(BinaryWriter w)
    {
        TernarySerializer.WriteFloats(w, _norm1);
        TernarySerializer.WriteFloats(w, _norm2);

        _attention.SaveToStream(w);
        _ffn.SaveToStream(w);
    }

    public void LoadFromStream(BinaryReader r)
    {
        _norm1 = TernarySerializer.ReadFloats(r);
        _norm2 = TernarySerializer.ReadFloats(r);

        _attention.LoadFromStream(r);
        _ffn.LoadFromStream(r);
    }
}