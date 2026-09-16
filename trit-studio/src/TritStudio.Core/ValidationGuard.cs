using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
namespace TritStudio.Core;

// Single-owner split guard. Final answers are excluded from input identities. Full/effective keys
// share a computation when no history is dropped; this does not cache mutable caller records.
public sealed class ValidationGuard
{
    private readonly HashSet<string> _fullKeys = new(StringComparer.Ordinal), _effectiveKeys;
    private readonly int? _sequenceLength;
    public long KeyBuilds { get; private set; }
    public ValidationGuard(IEnumerable<TrainingExample> validation, int? sequenceLength = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (sequenceLength is int budget && (budget < 8 || budget > 2048)) throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        _sequenceLength = sequenceLength; _effectiveKeys = sequenceLength is null ? _fullKeys : new(StringComparer.Ordinal);
        foreach (var row in validation)
        {
            ct.ThrowIfCancellationRequested(); string full = BuildKey(row, 0, ct); _fullKeys.Add(full);
            if (sequenceLength is int length)
            {
                int first = row.IsDialogue ? Dataset.RetainedHistoryStart(row, length) : 0;
                _effectiveKeys.Add(first == 0 ? full : BuildKey(row, first, ct));
            }
        }
    }
    private string BuildKey(TrainingExample row, int first, CancellationToken ct)
    { string key = InputKeyFromStart(row, first, ct); KeyBuilds++; return key; }
    public bool Contains(TrainingExample example, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); string full = BuildKey(example, 0, ct);
        if (_fullKeys.Contains(full)) return true;
        if (_sequenceLength is not int length) return false;
        int first = example.IsDialogue ? Dataset.RetainedHistoryStart(example, length) : 0;
        return _effectiveKeys.Contains(first == 0 ? full : BuildKey(example, first, ct));
    }
    public void EnsureTraining(IEnumerable<TrainingExample> examples, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        foreach (var e in examples)
        {
            ct.ThrowIfCancellationRequested();
            if (Contains(e, ct)) throw new InvalidDataException($"Контрольный вход {e.Id[..Math.Min(8,e.Id.Length)]} совпал с учебным, в том числе после обрезки старого контекста. Изменение ответа не разрешает обучение.");
        }
    }
    public static string InputKey(TrainingExample example, int? sequenceLength = null)
    {
        int first = sequenceLength is int length && example.IsDialogue ? Dataset.RetainedHistoryStart(example, length) : 0;
        return InputKeyFromStart(example, first, default);
    }
    private static string InputKeyFromStart(TrainingExample example, int first, CancellationToken ct)
    {
        string Normalize(string text)
        {
            ct.ThrowIfCancellationRequested();
            return string.Join(" ", text.Normalize(NormalizationForm.FormKC).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToUpperInvariant();
        }
        var content = new { kind = example.IsDialogue ? "dialogue" : "text", text = Normalize(example.Text),
            history = (example.History ?? []).Skip(first).Select(t => new[] { Normalize(t.User), Normalize(t.Assistant) }).ToArray() };
        ct.ThrowIfCancellationRequested();
        return Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(content, JsonData.Options)));
    }
}
