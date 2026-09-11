//https://github.com/virex-84

//TrainableLlamaBlock.cs

using System;

namespace TernaryLLM;

public sealed class TrainableLlamaBlock
{
    public Tensor Norm1Gamma;

    public Tensor QWeight;
    public Tensor KWeight;
    public Tensor VWeight;
    public Tensor OutWeight;

    public Tensor Norm2Gamma;

    public Tensor FfnGateWeight;
    public Tensor FfnUpWeight;
    public Tensor FfnDownWeight;

    private readonly int _emb;
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly int _hidden;

    private readonly Random _rng = new(123);

    public TrainableLlamaBlock(
        int emb,
        int heads,
        int kvHeads,
        int hidden)
    {
        if (emb % heads != 0)
            throw new ArgumentException("emb % heads != 0");

        if (heads % kvHeads != 0)
            throw new ArgumentException("heads % kvHeads != 0");

        _emb = emb;
        _heads = heads;
        _kvHeads = kvHeads;
        _headDim = emb / heads;
        _hidden = hidden;

        int kvDim = kvHeads * _headDim;

        float std = MathF.Sqrt(2f / emb);

        Norm1Gamma = InitOnes(emb, true);

        QWeight = InitNormal(emb, emb, std, true);
        KWeight = InitNormal(kvDim, emb, std, true);
        VWeight = InitNormal(kvDim, emb, std, true);
        OutWeight = InitNormal(emb, emb, std, true);

        Norm2Gamma = InitOnes(emb, true);

        FfnGateWeight = InitNormal(hidden, emb, MathF.Sqrt(2f / emb), true);
        FfnUpWeight = InitNormal(hidden, emb, MathF.Sqrt(2f / emb), true);
        FfnDownWeight = InitNormal(emb, hidden, MathF.Sqrt(2f / hidden), true);
    }

    public Tensor Forward(Tensor x, TernaryLlamaConfig config)
    {
        var residual = x;

        var normed = x.RmsNorm(Norm1Gamma);

        var qQ = StraightThroughQuantize(QWeight, config.AttentionQkvPlanes, config);
        var qK = StraightThroughQuantize(KWeight, config.AttentionQkvPlanes, config);
        var qV = StraightThroughQuantize(VWeight, config.AttentionQkvPlanes, config);
        var qOut = StraightThroughQuantize(OutWeight, config.AttentionOutPlanes, config);

        var q = normed.MatMul(Transpose(qQ));
        var k = normed.MatMul(Transpose(qK));
        var v = normed.MatMul(Transpose(qV));

        q = LlamaTrainOps.ApplyRoPE(q, config.NumHeads, _headDim, config.RopeTheta);
        k = LlamaTrainOps.ApplyRoPE(k, config.NumKVHeads, _headDim, config.RopeTheta);

        var attn = MultiHeadCausalAttentionGQA(
            q,
            k,
            v,
            config.NumHeads,
            config.NumKVHeads,
            _headDim);

        var attnOut = attn.MatMul(Transpose(qOut));

        x = Ops.Add(residual, attnOut);

        residual = x;

        normed = x.RmsNorm(Norm2Gamma);

        var qGate = StraightThroughQuantize(FfnGateWeight, config.FfnGatePlanes, config);
        var qUp = StraightThroughQuantize(FfnUpWeight, config.FfnUpPlanes, config);
        var qDown = StraightThroughQuantize(FfnDownWeight, config.FfnDownPlanes, config);

        var gate = normed.MatMul(Transpose(qGate));
        var up = normed.MatMul(Transpose(qUp));

        var act = LlamaTrainOps.Silu(gate);

        var hidden = Ops.Mul(act, up);

        var down = hidden.MatMul(Transpose(qDown));

        x = Ops.Add(residual, down);

        return x;
    }

    private Tensor StraightThroughQuantize(
        Tensor w,
        int planes,
        TernaryLlamaConfig config)
    {
        int rows = w.Shape[0];
        int cols = w.Shape[1];

        var packer = MultiPlaneTritPacker.FromFloat(
            w.Data,
            rows,
            cols,
            planes,
            config.GroupSize,
            config.ScaleMode);

        var quantData = packer.ToFloat();

        var result = new Tensor(
            quantData,
            w.Shape,
            requiresGrad: w.RequiresGrad);

        if (w.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (result.Grad == null || w.Grad == null)
                    return;

                for (int i = 0; i < w.Grad.Length; i++)
                {
                    w.Grad[i] += result.Grad[i];
                }
            }, w);
        }

        return result;
    }

    private Tensor MultiHeadCausalAttentionGQA(
        Tensor q,
        Tensor k,
        Tensor v,
        int heads,
        int kvHeads,
        int headDim)
    {
        int seq = q.Shape[0];
        int emb = q.Shape[1];
        int kvDim = k.Shape[1];

        int group = heads / kvHeads;

        float scale = 1f / MathF.Sqrt(headDim);

        var outData = new float[seq * emb];

        var outTensor = new Tensor(
            outData,
            new[] { seq, emb },
            q.RequiresGrad || k.RequiresGrad || v.RequiresGrad);

        var qHeads = new float[heads][];
        var kHeads = new float[kvHeads][];
        var vHeads = new float[kvHeads][];

        for (int h = 0; h < heads; h++)
        {
            int hStart = h * headDim;

            qHeads[h] = new float[seq * headDim];

            for (int s = 0; s < seq; s++)
            {
                Array.Copy(
                    q.Data,
                    s * emb + hStart,
                    qHeads[h],
                    s * headDim,
                    headDim);
            }
        }

        for (int kv = 0; kv < kvHeads; kv++)
        {
            int kvStart = kv * headDim;

            kHeads[kv] = new float[seq * headDim];
            vHeads[kv] = new float[seq * headDim];

            for (int s = 0; s < seq; s++)
            {
                Array.Copy(
                    k.Data,
                    s * kvDim + kvStart,
                    kHeads[kv],
                    s * headDim,
                    headDim);

                Array.Copy(
                    v.Data,
                    s * kvDim + kvStart,
                    vHeads[kv],
                    s * headDim,
                    headDim);
            }
        }

        var attnWeightsPerHead = new float[heads][];

        for (int h = 0; h < heads; h++)
        {
            int kvh = h / group;

            var attnW = new float[seq * seq];

            for (int i = 0; i < seq; i++)
            {
                float max = float.NegativeInfinity;

                for (int j = 0; j <= i; j++)
                {
                    float dot = 0f;

                    for (int d = 0; d < headDim; d++)
                    {
                        dot += qHeads[h][i * headDim + d] * kHeads[kvh][j * headDim + d];
                    }

                    float score = dot * scale;
                    attnW[i * seq + j] = score;

                    if (score > max)
                        max = score;
                }

                float sum = 0f;

                for (int j = 0; j <= i; j++)
                {
                    attnW[i * seq + j] = MathF.Exp(attnW[i * seq + j] - max);
                    sum += attnW[i * seq + j];
                }

                float inv = 1f / (sum + 1e-10f);

                for (int j = 0; j <= i; j++)
                {
                    attnW[i * seq + j] *= inv;
                }
            }

            attnWeightsPerHead[h] = attnW;

            for (int i = 0; i < seq; i++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;

                    for (int j = 0; j <= i; j++)
                    {
                        acc += attnW[i * seq + j] * vHeads[kvh][j * headDim + d];
                    }

                    outData[i * emb + h * headDim + d] = acc;
                }
            }
        }

        if (outTensor.RequiresGrad)
        {
            outTensor.AddBackward(() =>
            {
                if (outTensor.Grad == null)
                    return;

                var dKHeads = new float[kvHeads][];
                var dVHeads = new float[kvHeads][];

                for (int kv = 0; kv < kvHeads; kv++)
                {
                    dKHeads[kv] = new float[seq * headDim];
                    dVHeads[kv] = new float[seq * headDim];
                }

                for (int h = 0; h < heads; h++)
                {
                    int kvh = h / group;
                    int hStart = h * headDim;

                    var attnW = attnWeightsPerHead[h];

                    var dAttnW = new float[seq * seq];
                    var dV = new float[seq * headDim];

                    for (int i = 0; i < seq; i++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            float grad = outTensor.Grad[i * emb + hStart + d];

                            for (int j = 0; j <= i; j++)
                            {
                                dAttnW[i * seq + j] += grad * vHeads[kvh][j * headDim + d];
                                dV[j * headDim + d] += grad * attnW[i * seq + j];
                            }
                        }
                    }

                    for (int i = 0; i < seq; i++)
                    {
                        float dot = 0f;

                        for (int j = 0; j <= i; j++)
                        {
                            dot += attnW[i * seq + j] * dAttnW[i * seq + j];
                        }

                        for (int j = 0; j <= i; j++)
                        {
                            dAttnW[i * seq + j] =
                                attnW[i * seq + j] * (dAttnW[i * seq + j] - dot) * scale;
                        }
                    }

                    var dQHead = new float[seq * headDim];
                    var dKHead = new float[seq * headDim];

                    for (int i = 0; i < seq; i++)
                    {
                        for (int j = 0; j <= i; j++)
                        {
                            float ds = dAttnW[i * seq + j];

                            for (int d = 0; d < headDim; d++)
                            {
                                dQHead[i * headDim + d] +=
                                    ds * kHeads[kvh][j * headDim + d];

                                dKHead[j * headDim + d] +=
                                    ds * qHeads[h][i * headDim + d];
                            }
                        }
                    }

                    if (q.Grad != null)
                    {
                        for (int s = 0; s < seq; s++)
                        {
                            for (int d = 0; d < headDim; d++)
                            {
                                q.Grad[s * emb + hStart + d] += dQHead[s * headDim + d];
                            }
                        }
                    }

                    var dk = dKHeads[kvh];
                    var dv = dVHeads[kvh];

                    for (int idx = 0; idx < dKHead.Length; idx++)
                    {
                        dk[idx] += dKHead[idx];
                        dv[idx] += dV[idx];
                    }
                }

                if (k.Grad != null)
                {
                    for (int kv = 0; kv < kvHeads; kv++)
                    {
                        int kvStart = kv * headDim;

                        for (int s = 0; s < seq; s++)
                        {
                            for (int d = 0; d < headDim; d++)
                            {
                                k.Grad[s * kvDim + kvStart + d] +=
                                    dKHeads[kv][s * headDim + d];
                            }
                        }
                    }
                }

                if (v.Grad != null)
                {
                    for (int kv = 0; kv < kvHeads; kv++)
                    {
                        int kvStart = kv * headDim;

                        for (int s = 0; s < seq; s++)
                        {
                            for (int d = 0; d < headDim; d++)
                            {
                                v.Grad[s * kvDim + kvStart + d] +=
                                    dVHeads[kv][s * headDim + d];
                            }
                        }
                    }
                }

            }, q, k, v);
        }

        return outTensor;
    }

    private static Tensor Transpose(Tensor w)
    {
        int rows = w.Shape[0];
        int cols = w.Shape[1];

        var result = new Tensor(
            new float[rows * cols],
            new[] { cols, rows },
            w.RequiresGrad);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.Data[j * rows + i] = w.Data[i * cols + j];
            }
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (result.Grad == null || w.Grad == null)
                    return;

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        w.Grad[i * cols + j] += result.Grad[j * rows + i];
                    }
                }
            }, w);
        }

        return result;
    }

    private Tensor InitNormal(int rows, int cols, float std, bool requiresGrad)
    {
        var data = new float[rows * cols];

        for (int i = 0; i < data.Length; i++)
        {
            double u1 = _rng.NextDouble() + 1e-10;
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = (float)(std * z);
        }

        return new Tensor(data, new[] { rows, cols }, requiresGrad);
    }

    private static Tensor InitOnes(int size, bool requiresGrad)
    {
        var data = new float[size];
        Array.Fill(data, 1f);

        return new Tensor(data, new[] { size }, requiresGrad);
    }
}