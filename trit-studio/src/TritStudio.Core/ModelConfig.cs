namespace TritStudio.Core;

public sealed record ModelConfig
{
    public int FormatVersion { get; init; } = 1;
    public int Dimension { get; init; } = 64;
    public int HiddenDimension { get; init; } = 192;
    public int Layers { get; init; } = 2;
    public int Heads { get; init; } = 4;
    public int KvHeads { get; init; } = 2;
    public int Context { get; init; } = 1024;
    public int Planes { get; init; } = 2;
    public int GroupSize { get; init; } = 32;
    public float Threshold { get; init; } = 0.5f;
    public float RopeTheta { get; init; } = 10000;
    public int Seed { get; init; } = 42;
    public int HeadDimension => Dimension / Heads;
    public int KvDimension => HeadDimension * KvHeads;
    // Includes RMSNorm parameters; tied embedding/output matrix is counted once.
    public long ParameterCount => (long)ByteTokenizer.VocabularySize * Dimension + Dimension +
        Layers * (2L * Dimension * Dimension + 2L * Dimension * KvDimension +
                  3L * Dimension * HiddenDimension + 2L * Dimension);
    public void Validate()
    {
        if (FormatVersion != 1) throw new InvalidDataException("Unsupported model format version.");
        if (Dimension is < 16 or > 512 || HiddenDimension is < 16 or > 2048 || Layers is < 1 or > 12)
            throw new ArgumentException("Model dimensions exceed the supported envelope.");
        if (Heads is < 1 or > 16 || KvHeads < 1 || KvHeads > Heads || Dimension % Heads != 0 ||
            Heads % KvHeads != 0 || HeadDimension % 2 != 0)
            throw new ArgumentException("Heads must divide the dimension, KV heads must divide heads, and head dimension must be even.");
        if (Context is < 32 or > 2048 || Planes is < 1 or > 3 || GroupSize is < 8 or > 128 ||
            Dimension % GroupSize != 0 || HiddenDimension % GroupSize != 0)
            throw new ArgumentException("Invalid context, planes, or quantization group size.");
        if (!float.IsFinite(Threshold) || Threshold is < 0 or > 1 || !float.IsFinite(RopeTheta) || RopeTheta < 100)
            throw new ArgumentException("Invalid quantization/RoPE settings.");
        if (ParameterCount > 25_000_000) throw new ArgumentException("Safety limit: 25 million parameters.");
    }
    public static ModelConfig Small => new();
    public static ModelConfig Medium => new() { Dimension = 128, HiddenDimension = 384, Layers = 4 };
    public static ModelConfig Large => new() { Dimension = 256, HiddenDimension = 768, Layers = 6, Heads = 8, KvHeads = 4 };
}

public sealed record ResourceOptions
{
    public int Threads { get; init; } = Math.Max(1, Environment.ProcessorCount - 2);
    public int MemoryMiB { get; init; } = 8192;
    public int BatchSize { get; init; } = 8;
    public int SequenceLength { get; init; } = 512;
    public bool PreferCuda { get; init; } = true;
    public bool UseSdpa { get; init; } = true;
    public bool BucketByLength { get; init; } = true;
    public bool ProjectOnlyTargets { get; init; } = true;
    public int EffectiveThreads => Math.Clamp(Threads, 1, Math.Max(1, Environment.ProcessorCount));
    public void Validate(ModelConfig config)
    {
        config.Validate();
        if (Threads < 1 || Threads > 1024 || MemoryMiB is < 512 or > 65536 ||
            BatchSize is < 1 or > 32 || SequenceLength < 16 || SequenceLength > config.Context)
            throw new ArgumentException("Invalid CPU/memory/batch/sequence resource budget.");
        // Conservative preflight estimate, not a measured peak or a hard OS allocation limit.
        long estimate = 96 * config.ParameterCount + 64L * BatchSize * SequenceLength * config.Dimension * config.Layers +
                        16L * BatchSize * config.Heads * SequenceLength * SequenceLength * config.Layers + 256L * 1024 * 1024;
        if (estimate > (long)MemoryMiB * 1024 * 1024)
            throw new ArgumentException($"Estimated training working set {estimate / 1048576} MiB exceeds the configured budget. Reduce size, batch, or sequence length.");
    }
}

public sealed record TrainingOptions
{
    public int Steps { get; init; } = 1200;
    public int PublishEvery { get; init; } = 100;
    public double LearningRate { get; init; } = 0.001;
    public double OnlineLearningRate { get; init; } = 0.0001;
    public double MaxValidationRegression { get; init; } = 0.20;
    public void Validate()
    {
        if (Steps is < 0 or > 100_000 || PublishEvery is < 1 or > 1000 ||
            !double.IsFinite(LearningRate) || LearningRate is <= 0 or > 0.01 ||
            !double.IsFinite(OnlineLearningRate) || OnlineLearningRate is <= 0 or > 0.001 ||
            !double.IsFinite(MaxValidationRegression) || MaxValidationRegression is < 0 or > 0.5)
            throw new ArgumentException("Invalid training options.");
    }
}

public sealed record SamplingOptions(double Temperature = 0.7, int TopK = 40, double TopP = 0.95,
    double RepetitionPenalty = 1.0, int MaxNewTokens = 256)
{
    public SamplingOptions Clamp()
    {
        double temperature = double.IsFinite(Temperature) ? Math.Clamp(Temperature, 0, 2) : 0.7;
        int topK = Math.Clamp(TopK, 1, ByteTokenizer.VocabularySize);
        double topP = double.IsFinite(TopP) ? Math.Clamp(TopP, 0.01, 1) : 0.95;
        double repetition = double.IsFinite(RepetitionPenalty) ? Math.Clamp(RepetitionPenalty, 1, 2) : 1.0;
        int maxTokens = Math.Clamp(MaxNewTokens, 1, 1024);
        // Sampling settings are immutable. Valid settings need no per-token replacement record.
        if (temperature == Temperature && topK == TopK && topP == TopP && repetition == RepetitionPenalty && maxTokens == MaxNewTokens) return this;
        return new(temperature, topK, topP, repetition, maxTokens);
    }
}
