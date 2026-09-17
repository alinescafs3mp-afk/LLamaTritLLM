namespace TritStudio.Core;

public static class InitialTrainingMaterial
{
    // Raw creation attaches these texts only as future material. They must not change first-conversation
    // sampling versus direct conversation creation. Once ANY step ran, retain all previous material.
    public static TrainingExample[] RemoveUnlearnedBaseline(TrainingExample[] existing, TrainingExample[] baseline, long step)
    {
        if (step < 0) throw new ArgumentOutOfRangeException(nameof(step));
        if (step > 0) return existing;
        var ids = baseline.Select(x => x.Id).ToHashSet(StringComparer.Ordinal);
        return existing.Where(x => !ids.Contains(x.Id)).ToArray();
    }
}
