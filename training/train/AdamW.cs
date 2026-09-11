//https://github.com/virex-84

//AdamW.cs

using System;
using System.Collections.Generic;

namespace TernaryLLM;

public sealed class AdamW
{
    public float LearningRate { get; set; }
    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.999f;
    public float Epsilon { get; set; } = 1e-8f;
    public float WeightDecay { get; set; } = 0.01f;
    public float GradientClip { get; set; } = 1.0f;

    private int _step;
    private float _beta1t = 1f;
    private float _beta2t = 1f;

    private readonly Dictionary<Tensor, (float[] m, float[] v)> _state = new();

    public AdamW(float lr)
    {
        LearningRate = lr;
    }

    public void Register(IEnumerable<Tensor> parameters)
    {
        foreach (var p in parameters)
        {
            if (p.RequiresGrad && !_state.ContainsKey(p))
            {
                _state[p] = (new float[p.Data.Length], new float[p.Data.Length]);
            }
        }
    }

    public void Step()
    {
        _step++;
        _beta1t *= Beta1;
        _beta2t *= Beta2;

        float bc1 = 1f / (1f - _beta1t);
        float bc2 = 1f / (1f - _beta2t);

        // Gradient clipping
        if (GradientClip > 0)
        {
            float totalNorm = 0;
            foreach (var (param, _) in _state)
            {
                if (param.Grad == null) continue;
                for (int i = 0; i < param.Grad.Length; i++)
                    totalNorm += param.Grad[i] * param.Grad[i];
            }
            totalNorm = MathF.Sqrt(totalNorm);

            if (totalNorm > GradientClip)
            {
                float scale = GradientClip / (totalNorm + 1e-6f);
                foreach (var (param, _) in _state)
                {
                    if (param.Grad == null) continue;
                    for (int i = 0; i < param.Grad.Length; i++)
                        param.Grad[i] *= scale;
                }
            }
        }

        // AdamW update
        foreach (var (param, (m, v)) in _state)
        {
            if (param.Grad == null) continue;

            for (int i = 0; i < param.Data.Length; i++)
            {
                float g = param.Grad[i];

                m[i] = Beta1 * m[i] + (1f - Beta1) * g;
                v[i] = Beta2 * v[i] + (1f - Beta2) * g * g;

                float mHat = m[i] * bc1;
                float vHat = v[i] * bc2;

                float update = LearningRate * mHat / (MathF.Sqrt(vHat) + Epsilon);

                // Weight decay (decoupled)
                if (WeightDecay > 0)
                    update += LearningRate * WeightDecay * param.Data[i];

                param.Data[i] -= update;
            }

            // Zero grad
            Array.Clear(param.Grad, 0, param.Grad.Length);
        }
    }
}