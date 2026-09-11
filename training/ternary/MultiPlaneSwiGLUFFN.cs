//https://github.com/virex-84

//MultiPlaneSwiGLUFFN.cs

using System;

namespace TernaryLLM;

public sealed class MultiPlaneSwiGLUFFN
{
    private readonly int _emb;
    private readonly int _hidden;

    private readonly MultiPlaneLinear _gate;
    private readonly MultiPlaneLinear _up;
    private readonly MultiPlaneLinear _down;

    private readonly bool _quantizeActivations;
    private readonly int _activationPlanes;
    private readonly int _groupSize;
    private readonly TritScaleMode _scaleMode;

    public MultiPlaneLinear Gate => _gate;
    public MultiPlaneLinear Up => _up;
    public MultiPlaneLinear Down => _down;

    public MultiPlaneSwiGLUFFN(
        int emb,
        int hidden,
        int gatePlanes,
        int upPlanes,
        int downPlanes,
        int activationPlanes,
        int groupSize,
        TritScaleMode scaleMode,
        bool quantizeActivations)
    {
        _emb = emb;
        _hidden = hidden;

        _quantizeActivations = quantizeActivations;
        _activationPlanes = activationPlanes;
        _groupSize = groupSize;
        _scaleMode = scaleMode;

        float stdUp = MathF.Sqrt(2f / emb);
        float stdDown = MathF.Sqrt(2f / hidden);

        var gateData = CreateRandomData(hidden, emb, stdUp);
        var upData = CreateRandomData(hidden, emb, stdUp);
        var downData = CreateRandomData(emb, hidden, stdDown);

        var gatePacker = MultiPlaneTritPacker.FromFloat(
            gateData,
            hidden,
            emb,
            gatePlanes,
            groupSize,
            scaleMode);

        var upPacker = MultiPlaneTritPacker.FromFloat(
            upData,
            hidden,
            emb,
            upPlanes,
            groupSize,
            scaleMode);

        var downPacker = MultiPlaneTritPacker.FromFloat(
            downData,
            emb,
            hidden,
            downPlanes,
            groupSize,
            scaleMode);

        _gate = new MultiPlaneLinear(gatePacker, useBias: false);
        _up = new MultiPlaneLinear(upPacker, useBias: false);
        _down = new MultiPlaneLinear(downPacker, useBias: false);
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

    public float[] Forward(ReadOnlySpan<float> input, int seq)
    {
        float[] gate = _gate.ForwardSequence(input, seq);
        float[] up = _up.ForwardSequence(input, seq);

        LlamaOps.Silu(gate);

        for (int i = 0; i < gate.Length; i++)
        {
            gate[i] *= up[i];
        }

        if (_quantizeActivations)
        {
            gate = TernaryOps.Requantize(
                gate,
                seq,
                _hidden,
                _activationPlanes,
                _groupSize,
                _scaleMode);
        }

        float[] down = _down.ForwardSequence(gate, seq);

        if (_quantizeActivations)
        {
            down = TernaryOps.Requantize(
                down,
                seq,
                _emb,
                _activationPlanes,
                _groupSize,
                _scaleMode);
        }

        return down;
    }

    public void SaveToStream(BinaryWriter w)
    {
        _gate.SaveToStream(w);
        _up.SaveToStream(w);
        _down.SaveToStream(w);
    }

    public void LoadFromStream(BinaryReader r)
    {
        _gate.LoadFromStream(r);
        _up.LoadFromStream(r);
        _down.LoadFromStream(r);
    }
}