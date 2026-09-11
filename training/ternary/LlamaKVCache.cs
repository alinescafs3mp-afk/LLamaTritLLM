//https://github.com/virex-84

//LlamaKVCache.cs

using System;

namespace TernaryLLM;

public sealed class LlamaKVCache
{
    public int MaxSeq { get; }
    public int KVDim { get; }
    public int Length { get; private set; }

    public float[] K { get; }
    public float[] V { get; }

    public LlamaKVCache(int maxSeq, int kvDim)
    {
        MaxSeq = maxSeq;
        KVDim = kvDim;
        Length = 0;

        K = new float[maxSeq * kvDim];
        V = new float[maxSeq * kvDim];
    }

    public void Reset()
    {
        Length = 0;
    }

    public void Append(float[] k, float[] v, int seq)
    {
        if (seq <= 0)
            return;

        if (Length + seq > MaxSeq)
            throw new InvalidOperationException(
                $"KV cache overflow: length={Length}, adding={seq}, maxSeq={MaxSeq}.");

        Array.Copy(k, 0, K, Length * KVDim, seq * KVDim);
        Array.Copy(v, 0, V, Length * KVDim, seq * KVDim);

        Length += seq;
    }
}