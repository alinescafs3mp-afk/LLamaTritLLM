using System.Text.Json;
namespace TritStudio.Core;

// Decode before removing a pending request: malformed payloads must leave it reachable by Fail().
public sealed record CommandCompletion(bool Success, bool Cancelled, string? Error)
{
    public static CommandCompletion Parse(JsonElement data)
    {
        if (data.ValueKind != JsonValueKind.Object || !data.TryGetProperty("success", out var success) ||
            success.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new InvalidDataException("Результат команды не содержит корректного success.");
        bool cancelled = false; string? error = null;
        if (data.TryGetProperty("cancelled", out var flag))
        {
            if (flag.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                throw new InvalidDataException("Некорректный признак отмены команды.");
            cancelled = flag.GetBoolean();
        }
        if (data.TryGetProperty("error", out var problem))
        {
            if (problem.ValueKind is not (JsonValueKind.String or JsonValueKind.Null))
                throw new InvalidDataException("Ошибка команды должна быть строкой или null.");
            error = problem.ValueKind == JsonValueKind.String ? problem.GetString() : null;
        }
        if (success.GetBoolean() && (cancelled || !string.IsNullOrWhiteSpace(error)))
            throw new InvalidDataException("Противоречивый результат команды: успех вместе с отменой/ошибкой.");
        return new(success.GetBoolean(), cancelled, error);
    }
}
