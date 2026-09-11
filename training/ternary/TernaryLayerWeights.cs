//https://github.com/virex-84

//TernaryLayerWeights.cs

using System;

namespace TernaryLLM;

/// <summary>
/// Промежуточный DTO для переноса весов из обучаемой (мастер-)модели
/// в тернарную инференс-модель.
///
/// Живёт в ternary-пакете, чтобы TernaryLlamaModel НЕ зависел от
/// train-классов (TrainableLlamaBlock и пр.).
/// </summary>
public sealed class TernaryLayerWeights
{
    /// <summary>Нормы внимания (FP32).</summary>
    public float[] Norm1 = Array.Empty<float>();
    public float[] Norm2 = Array.Empty<float>();

    /// <summary>Веса внимания, row-major [rows, cols].</summary>
    public float[] Q = Array.Empty<float>();
    public float[] K = Array.Empty<float>();
    public float[] V = Array.Empty<float>();
    public float[] Out = Array.Empty<float>();

    /// <summary>Веса SwiGLU FFN, row-major [rows, cols].</summary>
    public float[] Gate = Array.Empty<float>();
    public float[] Up = Array.Empty<float>();
    public float[] Down = Array.Empty<float>();
}