using System.Text.Json;
namespace TritStudio.Core;

public static class EventEnvelope
{
    public static WorkerEvent Parse(string line)
    {
        using var document = JsonDocument.Parse(line);
        var root = document.RootElement; UniqueFields(root);
        var value = root.Deserialize<WorkerEvent>(JsonData.Options);
        if (value is null || !ValidName(value.Kind, 64) || value.Data.ValueKind != JsonValueKind.Object ||
            (value.Id is not null && !ValidName(value.Id, 128)) || (value.Kind == "completed" && value.Id is null))
            throw new InvalidDataException("Пустая или неоднозначная запись протокола тренера.");
        // Clone: the record must outlive the parser document and its pooled backing buffer.
        return value with { Data = value.Data.Clone() };
    }
    public static string CompletionCommand(JsonElement data)
    {
        UniqueFields(data);
        if (!data.TryGetProperty("command", out var command) || command.ValueKind != JsonValueKind.String ||
            !ValidName(command.GetString(), 64)) throw new InvalidDataException("Результат не указывает выполненную команду.");
        return command.GetString()!;
    }
    private static bool ValidName(string? name, int limit) => !string.IsNullOrWhiteSpace(name) && name.Length <= limit && !name.Any(char.IsControl);
    private static void UniqueFields(JsonElement data)
    {
        if (data.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Protocol envelope must be an object.");
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in data.EnumerateObject())
            if (!names.Add(item.Name)) throw new InvalidDataException("Duplicate event/receipt field.");
    }
}
