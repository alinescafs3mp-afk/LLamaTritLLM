//https://github.com/virex-84

//Tensor.cs

using System;
using System.Collections.Generic;

namespace TernaryLLM;

/// <summary>
/// Autograd tensor с поддержкой backward через замыкания.
/// </summary>
public sealed class Tensor
{
    public float[] Data;
    public float[]? Grad;
    public int[] Shape;

    private readonly List<Action> _backward = new();
    private readonly List<Tensor> _children = new();

    public bool RequiresGrad { get; set; }

    public Tensor(float[] data, int[] shape, bool requiresGrad = false)
    {
        Data = data;
        Shape = shape;
        RequiresGrad = requiresGrad;
        if (requiresGrad)
            Grad = new float[data.Length];
    }

    public void ZeroGrad()
    {
        if (Grad != null)
            Array.Clear(Grad, 0, Grad.Length);
    }

    public void Backward()
    {
        if (Grad == null)
            Grad = new float[Data.Length];

        // Заполняем единицами (dL/dL = 1)
        Array.Fill(Grad, 1f);

        // Топологическая сортировка
        var topo = new List<Tensor>();
        var visited = new HashSet<Tensor>();

        void Build(Tensor t)
        {
            if (!visited.Add(t)) return;
            foreach (var child in t._children)
                Build(child);
            topo.Add(t);
        }

        Build(this);

        // Обратный проход
        for (int i = topo.Count - 1; i >= 0; i--)
        {
            foreach (var b in topo[i]._backward)
                b();
        }
    }

    internal void AddBackward(Action gradFn, params Tensor[] children)
    {
        _backward.Add(gradFn);
        foreach (var c in children)
            if (c.RequiresGrad)
                _children.Add(c);
    }


    public Tensor Add(Tensor other) => Ops.Add(this, other);
    public Tensor MatMul(Tensor other) => Ops.MatMul(this, other);
    public Tensor Mul(Tensor other) => Ops.Mul(this, other);
    public Tensor Gelu() => Ops.Gelu(this);
    public Tensor RmsNorm(Tensor gamma) => Ops.RmsNorm(this, gamma);
    public Tensor Softmax() => Ops.Softmax(this);
    public Tensor LogSoftmax() => Ops.LogSoftmax(this);

}