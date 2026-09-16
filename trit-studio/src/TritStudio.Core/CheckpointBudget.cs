namespace TritStudio.Core;

public static class CheckpointBudget
{
    public const int Limit = 128;
    public static int Publications(int steps, int publishEvery, bool includeInitial = false)
    {
        if (steps is < 0 or > 100000 || publishEvery is < 1 or > 1000)
            throw new ArgumentOutOfRangeException(nameof(steps), "Invalid bounded publication plan.");
        return checked((steps + publishEvery - 1) / publishEvery + (includeInitial ? 1 : 0));
    }
    public static void EnsureFits(int existing, int required)
    {
        if (existing < 0 || required < 0) throw new ArgumentOutOfRangeException(nameof(existing));
        int available = Math.Max(0, Limit - existing);
        if (required > available)
            throw new IOException($"Для запуска требуется {required} новых снимков, доступно {available} из {Limit}. " +
                "Уменьшите число шагов или очистите старые снимки на вкладке обучения. Этот запуск ещё не изменил веса.");
    }
}
