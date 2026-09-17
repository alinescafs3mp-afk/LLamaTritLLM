namespace TritStudio.Core;

// Conservative model-specific planning envelope, NOT a measured allocator peak.
// Separates persistent storage from sequence-dependent forward/backward work.
public sealed record TrainingMemoryEstimate(long PersistentBytes, long ActivationBytes, int EffectiveLength)
{
    public long TotalBytes => checked(PersistentBytes + ActivationBytes);
    public static TrainingMemoryEstimate For(ModelConfig config, ResourceOptions resources, int effectiveLength)
    {
        config.Validate();
        if (resources.BatchSize is < 1 or > 32 || effectiveLength < 0 || effectiveLength > config.Context)
            throw new ArgumentOutOfRangeException(nameof(effectiveLength));
        long persistent = checked(96 * config.ParameterCount + 256L * 1048576);
        long activations = checked(64L * resources.BatchSize * effectiveLength * config.Dimension * config.Layers +
            16L * resources.BatchSize * config.Heads * effectiveLength * effectiveLength * config.Layers);
        return new(persistent, activations, effectiveLength);
    }
}
