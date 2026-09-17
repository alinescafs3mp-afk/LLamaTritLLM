namespace TritStudio.Core;

public static class CheckpointBudget
{
    public const int Limit = 128;
    public static int Publications(int steps, int publishEvery, bool includeInitial = false)
    {
        if (steps is < 0 or > 100000 || publishEvery is < 1 or > 100000)
            throw new ArgumentOutOfRangeException(nameof(steps), "Invalid bounded publication plan.");
        return checked((steps + publishEvery - 1) / publishEvery + (includeInitial ? 1 : 0));
    }
    // Only the explicitly enabled mode may lengthen cadence. It never removes a checkpoint.
    public static TrainingOptions Plan(TrainingOptions requested, int existing, bool includeInitial = false)
    {
        requested.Validate();
        if (existing < 0) throw new ArgumentOutOfRangeException(nameof(existing));
        int available = Math.Max(0, Limit - existing - (includeInitial ? 1 : 0));
        int interval = requested.PublishEvery;
        if (requested.AutoSnapshotInterval && requested.Steps > 0 && available > 0)
        {
            int targetPublications = Math.Min(32, available);
            interval = Math.Max(interval, (requested.Steps + targetPublications - 1) / targetPublications);
        }
        var result = requested with { PublishEvery = interval };
        EnsureFits(existing, Publications(result.Steps, interval, includeInitial));
        return result;
    }
    public static void EnsureFits(int existing, int required)
    {
        if (existing < 0 || required < 0) throw new ArgumentOutOfRangeException(nameof(existing));
        int available = Math.Max(0, Limit - existing);
        if (required > available)
            throw new IOException($"Для запуска требуется {required} новых снимков, доступно {available} из {Limit}. " +
                "Увеличьте интервал сохранения, включите его автоподбор или очистите старые снимки. Автоподбор ничего не удаляет. Этот запуск ещё не изменил веса.");
    }
}
