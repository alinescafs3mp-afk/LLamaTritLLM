//https://github.com/virex-84

//MultiPlaneLlamaAttention.cs

using System;

namespace TernaryLLM;

public enum AttentionMode
{
    Softmax,
    TopK,
    Hardmax
}

public sealed class MultiPlaneLlamaAttention
{
    private readonly int _emb;
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly int _kvDim;
    private readonly int _groupSize;

    private readonly float _invScale;
    private readonly float _theta;

    private readonly MultiPlaneLinear _q;
    private readonly MultiPlaneLinear _k;
    private readonly MultiPlaneLinear _v;
    private readonly MultiPlaneLinear _out;

    private readonly AttentionMode _mode;
    private readonly int _topK;

    public MultiPlaneLinear Q => _q;
    public MultiPlaneLinear K => _k;
    public MultiPlaneLinear V => _v;
    public MultiPlaneLinear Out => _out;

    public MultiPlaneLlamaAttention(
        int emb,
        int heads,
        int kvHeads,
        int qkvPlanes,
        int outPlanes,
        int groupSize,
        TritScaleMode scaleMode,
        AttentionMode mode,
        int topK,
        float ropeTheta)
    {
        if (emb % heads != 0)
            throw new ArgumentException("emb % heads != 0");

        if (heads % kvHeads != 0)
            throw new ArgumentException("heads % kvHeads != 0");

        _emb = emb;
        _heads = heads;
        _kvHeads = kvHeads;
        _headDim = emb / heads;
        _kvDim = kvHeads * _headDim;
        _groupSize = heads / kvHeads;

        _invScale = 1f / MathF.Sqrt(_headDim);
        _theta = ropeTheta;

        _mode = mode;
        _topK = topK;

        float std = MathF.Sqrt(2f / emb);

        var qData = CreateRandomData(emb, emb, std);
        var kData = CreateRandomData(_kvDim, emb, std);
        var vData = CreateRandomData(_kvDim, emb, std);
        var outData = CreateRandomData(emb, emb, std);

        var qPacker = MultiPlaneTritPacker.FromFloat(
            qData,
            emb,
            emb,
            qkvPlanes,
            groupSize,
            scaleMode);

        var kPacker = MultiPlaneTritPacker.FromFloat(
            kData,
            _kvDim,
            emb,
            qkvPlanes,
            groupSize,
            scaleMode);

        var vPacker = MultiPlaneTritPacker.FromFloat(
            vData,
            _kvDim,
            emb,
            qkvPlanes,
            groupSize,
            scaleMode);

        var outPacker = MultiPlaneTritPacker.FromFloat(
            outData,
            emb,
            emb,
            outPlanes,
            groupSize,
            scaleMode);

        _q = new MultiPlaneLinear(qPacker, useBias: false);
        _k = new MultiPlaneLinear(kPacker, useBias: false);
        _v = new MultiPlaneLinear(vPacker, useBias: false);
        _out = new MultiPlaneLinear(outPacker, useBias: false);
    }

    private static float[] CreateRandomData(int rows, int cols, float std)
    {
        var data = new float[rows * cols];
        var rng = new Random(42);

        for (int i = 0; i < data.Length; i++)
        {
            double u1 = rng.NextDouble() + 1e-10;
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = (float)(std * z);
        }

        return data;
    }

    public float[] Forward(ReadOnlySpan<float> input, int seq, LlamaKVCache cache)
    {
        if (cache == null)
            throw new ArgumentNullException(nameof(cache));

        if (input.Length < seq * _emb)
            throw new ArgumentException("input too small");

        float[] q = _q.ForwardSequence(input, seq);
        float[] k = _k.ForwardSequence(input, seq);
        float[] v = _v.ForwardSequence(input, seq);

        int startPos = cache.Length;

        LlamaOps.ApplyRoPE(
            q,
            k,
            seq,
            startPos,
            _heads,
            _kvHeads,
            _headDim,
            _theta);

        cache.Append(k, v, seq);

        float[] context = new float[seq * _emb];

        for (int i = 0; i < seq; i++)
        {
            int absPos = startPos + i;
            int valid = absPos + 1;

            for (int h = 0; h < _heads; h++)
            {
                int kvh = h / _groupSize;
                int qBase = i * _emb + h * _headDim;

                float[] scores = new float[valid];

                for (int j = 0; j <= absPos; j++)
                {
                    int kBase = j * _kvDim + kvh * _headDim;

                    float dot = 0f;

                    for (int d = 0; d < _headDim; d++)
                    {
                        dot += q[qBase + d] * cache.K[kBase + d];
                    }

                    scores[j] = dot * _invScale;
                }

                float[] weights = ComputeWeights(scores);

                for (int d = 0; d < _headDim; d++)
                {
                    float acc = 0f;

                    for (int j = 0; j <= absPos; j++)
                    {
                        int vBase = j * _kvDim + kvh * _headDim + d;
                        acc += weights[j] * cache.V[vBase];
                    }

                    context[qBase + d] = acc;
                }
            }
        }

        return _out.ForwardSequence(context, seq);
    }

    private float[] ComputeWeights(float[] scores)
    {
        var weights = new float[scores.Length];

        if (scores.Length == 0)
            return weights;

        switch (_mode)
        {
            case AttentionMode.Softmax:
                {
                    float max = scores[0];

                    for (int i = 1; i < scores.Length; i++)
                    {
                        if (scores[i] > max)
                            max = scores[i];
                    }

                    float sum = 0f;

                    for (int i = 0; i < scores.Length; i++)
                    {
                        weights[i] = MathF.Exp(scores[i] - max);
                        sum += weights[i];
                    }

                    if (sum <= 1e-12f)
                    {
                        float uniform = 1f / scores.Length;

                        for (int i = 0; i < scores.Length; i++)
                            weights[i] = uniform;
                    }
                    else
                    {
                        float inv = 1f / (sum + 1e-10f);

                        for (int i = 0; i < scores.Length; i++)
                            weights[i] *= inv;
                    }

                    break;
                }

            case AttentionMode.Hardmax:
                {
                    int best = 0;

                    for (int i = 1; i < scores.Length; i++)
                    {
                        if (scores[i] > scores[best])
                            best = i;
                    }

                    weights[best] = 1f;
                    break;
                }

            case AttentionMode.TopK:
                {
                    int k = Math.Clamp(_topK, 1, scores.Length);

                    var indices = new int[scores.Length];

                    for (int i = 0; i < indices.Length; i++)
                        indices[i] = i;

                    Array.Sort(indices, (a, b) => scores[b].CompareTo(scores[a]));

                    float w = 1f / k;

                    foreach (int idx in indices)
                    {
                        weights[idx] = w;
                    }

                    break;
                }
        }

        return weights;
    }

    public void SaveToStream(BinaryWriter w)
    {
        _q.SaveToStream(w);
        _k.SaveToStream(w);
        _v.SaveToStream(w);
        _out.SaveToStream(w);
    }

    public void LoadFromStream(BinaryReader r)
    {
        _q.LoadFromStream(r);
        _k.LoadFromStream(r);
        _v.LoadFromStream(r);
        _out.LoadFromStream(r);
    }
}