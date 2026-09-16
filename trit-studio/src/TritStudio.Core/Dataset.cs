using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
namespace TritStudio.Core;

public static class Dataset
{
    public const int MaxExamples = 50000;
    // Single-owner immutable records only. Do not substitute ID-only equality: metadata/history may differ.
    public static TrainingExample[] ReuseUnchanged(TrainingExample[] previous, TrainingExample[] candidate,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(previous); ArgumentNullException.ThrowIfNull(candidate);
        if (ReferenceEquals(previous, candidate)) return previous;
        if (previous.Length != candidate.Length) return candidate;
        for (int i = 0; i < previous.Length; i++)
        {
            if ((i & 255) == 0) ct.ThrowIfCancellationRequested();
            if (previous[i] != candidate[i]) return candidate;
        }
        return previous;
    }
    public static string Identity(string text, string? answer) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new[] { text, answer }, JsonData.Options))));
    public static TrainingExample Make(string text, string? answer = null, string source = "dataset", IReadOnlyList<ChatTurn>? history = null)
    {
        if (text is null) throw new ArgumentException("Training text is null.");
        text = text.Trim(); answer = answer?.Trim();
        if (text.Length == 0 || answer?.Length == 0 || text.Length > 32768 || answer?.Length > 32768)
            throw new ArgumentException("Training text/answer is empty or exceeds 32768 characters.");
        // Reject invalid Unicode rather than learning replacement characters.
        var utf8 = new UTF8Encoding(false, true); _ = utf8.GetByteCount(text);
        if (answer is not null) _ = utf8.GetByteCount(answer);
        if (history?.Count > 64) throw new ArgumentException("At most 64 context pairs are supported.");
        ChatTurn[]? context = history?.Select(t => {
            if (t.ExcludedFromTraining) throw new ArgumentException("An excluded turn cannot enter training context.");
            if (string.IsNullOrWhiteSpace(t.User) || t.Assistant is null || t.User.Length > 32768 || t.Assistant.Length > 32768)
                throw new ArgumentException("Invalid dialogue context.");
            _ = utf8.GetByteCount(t.User); _ = utf8.GetByteCount(t.Assistant);
            return new ChatTurn(t.User, t.Assistant, 0);
        }).ToArray();
        if (context?.Length == 0) context = null;
        if (answer is null && context is not null) throw new ArgumentException("Text-only examples cannot have dialogue context.");
        // Preserve v1 IDs for single-turn examples. Context makes otherwise identical examples distinct.
        string id = context is null ? Identity(text, answer) : Convert.ToHexString(SHA256.HashData(
            Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new { text, answer, history = context.Select(t => new[] { t.User, t.Assistant }) }, JsonData.Options))));
        return new(id, text, answer, source, context);
    }
    public static int RequiredSequenceLength(TrainingExample e) => e.IsDialogue
        ? checked(ByteTokenizer.TokenCount(e.Text) + ByteTokenizer.TokenCount(e.Answer!) + 4)
        : checked(ByteTokenizer.TokenCount(e.Text) + 1);
    public static int FullSequenceLength(TrainingExample e) => checked(RequiredSequenceLength(e) +
        (e.History?.Sum(t => ByteTokenizer.TokenCount(t.User) + ByteTokenizer.TokenCount(t.Assistant) + 4) ?? 0));
    public static ChatTurn[] LearningContext(IReadOnlyList<ChatTurn> history, ChatTurn target)
    {
        int index = -1; for (int i = history.Count - 1; i >= 0; i--) if (ReferenceEquals(history[i], target) || history[i] == target) { index = i; break; }
        if (target.LearningHistory is not null) return TeachingContext.Capture(target.LearningHistory, target.LearningHistory.Length, target.ConversationId);
        return index < 0 ? [] : TeachingContext.Capture(history.Take(index).ToArray(), index, target.ConversationId);
    }
    public static TrainingExample[] LoadManyTraining(IEnumerable<string> paths, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        var selected = paths.Select(Path.GetFullPath).Distinct(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal).ToArray();
        if (selected.Length > 32) throw new ArgumentException("За один импорт допускается не более 32 файлов.");
        long bytes = selected.Sum(path => new FileInfo(path).Length);
        if (bytes > 128L * 1024 * 1024) throw new ArgumentException("Общий размер импорта превышает 128 МиБ.");
        var result = new Dictionary<string, TrainingExample>(StringComparer.Ordinal);
        foreach (string path in selected)
        {
            ct.ThrowIfCancellationRequested();
            foreach (var example in LoadInternal(path, forTraining: true, ct))
            {
                result.TryAdd(example.Id, example);
                if (result.Count > MaxExamples) throw new ArgumentException("Общий импорт превышает 50000 уникальных примеров.");
            }
        }
        return result.Values.ToArray();
    }
    public static TrainingExample[] Load(string path, CancellationToken ct = default) => LoadInternal(path, forTraining: false, ct);
    public static TrainingExample[] LoadTraining(string path, CancellationToken ct = default) => LoadInternal(path, forTraining: true, ct);
    private static TrainingExample[] LoadInternal(string path, bool forTraining, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (new FileInfo(path).Length > 64L * 1024 * 1024) throw new ArgumentException("Dataset import limit is 64 MiB per file.");
        string ext = Path.GetExtension(path).ToLowerInvariant();
        var result = new List<TrainingExample>();
        if (ext == ".txt")
        {
            foreach (var line in StrictDatasetLines.Read(path, ext == ".txt" ? 32768 : StrictDatasetLines.MaxJsonLineChars, ct)) { ct.ThrowIfCancellationRequested(); if (!string.IsNullOrWhiteSpace(line)) AddBounded(result, Make(line)); }
        }
        else if (ext == ".json")
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var doc = JsonDocument.Parse(stream);
            if (doc.RootElement.ValueKind != JsonValueKind.Array) throw new InvalidDataException("JSON must contain an array.");
            foreach (var item in doc.RootElement.EnumerateArray()) { ct.ThrowIfCancellationRequested(); AddBounded(result, Parse(item, forTraining)); }
        }
        else if (ext == ".jsonl")
        {
            int n = 0;
            foreach (string line in StrictDatasetLines.Read(path, ext == ".txt" ? 32768 : StrictDatasetLines.MaxJsonLineChars, ct))
            {
                ct.ThrowIfCancellationRequested();
                n++; if (string.IsNullOrWhiteSpace(line)) continue;
                try { using var doc = JsonDocument.Parse(line); AddBounded(result, Parse(doc.RootElement, forTraining)); }
                catch (OperationCanceledException) { throw; }
                catch (Exception e) { throw new InvalidDataException($"Invalid JSONL at line {n}: {e.Message}", e); }
            }
        }
        else throw new ArgumentException("Supported datasets: .txt, .json, .jsonl.");
        if (result.Count == 0 || result.Count > MaxExamples) throw new ArgumentException("Dataset must have 1..50000 records.");
        return result.DistinctBy(x => x.Id).ToArray();
    }
    private static void AddBounded(List<TrainingExample> result, TrainingExample example)
    {
        if (result.Count >= MaxExamples) throw new InvalidDataException("Dataset exceeds 50000 records.");
        result.Add(example);
    }
    private static TrainingExample Parse(JsonElement e, bool forTraining = false)
    {
        if (e.ValueKind == JsonValueKind.String)
        {
            string s = e.GetString()!; const string marker = " Assistant:";
            int at = s.IndexOf(marker, StringComparison.Ordinal);
            if (s.StartsWith("User:", StringComparison.Ordinal) && at > 5) return Make(s[5..at], s[(at + marker.Length)..]);
            return Make(s);
        }
        if (e.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Dataset record must be text or an object.");
        if (forTraining && e.TryGetProperty("split", out var split) && split.ValueKind == JsonValueKind.String && (split.GetString()?.Trim().ToLowerInvariant() is "validation" or "test" or "challenge" or "control" or "eval"))
            throw new InvalidDataException("Это контрольный/тестовый набор. Он не допускается в обучение; выберите seed.jsonl или собственный учебный файл.");
        if (e.TryGetProperty("messages", out var messages))
        {
            if (messages.ValueKind != JsonValueKind.Array) throw new InvalidDataException("messages must be an array.");
            var all = messages.EnumerateArray().ToArray();
            if (all.Length < 2 || all.Length > 130 || all.Length % 2 != 0) throw new InvalidDataException("messages must contain complete user/assistant pairs.");
            var pairs = new List<ChatTurn>();
            for (int i = 0; i < all.Length; i += 2)
            {
                if (all[i].GetProperty("role").GetString() != "user" || all[i + 1].GetProperty("role").GetString() != "assistant")
                    throw new InvalidDataException("Only alternating user/assistant messages are supported; system/tool roles are not silently discarded.");
                pairs.Add(new ChatTurn(all[i].GetProperty("content").GetString() ?? "", all[i + 1].GetProperty("content").GetString() ?? "", 0));
            }
            var last = pairs[^1]; return Make(last.User, last.Assistant, history: pairs.Take(pairs.Count - 1).ToArray());
        }
        if (e.TryGetProperty("prompt", out var p) && e.TryGetProperty("answer", out var a)) return Make(p.GetString()!, a.ValueKind == JsonValueKind.String ? a.GetString()! : throw new InvalidDataException("answer must be a string"));
        if (e.TryGetProperty("text", out var t)) return Make(t.GetString()!);
        throw new InvalidDataException("Expected text or {prompt, answer}.");
    }
    // Shared by encoding and split isolation: never hash context that the network does not actually see.
    public static int RetainedHistoryStart(TrainingExample example, int length)
    {
        if (length is < 8 or > 2048) throw new ArgumentOutOfRangeException(nameof(length));
        int used = RequiredSequenceLength(example);
        if (used > length) throw new ArgumentException($"Example {example.Id[..Math.Min(8, example.Id.Length)]} needs sequence >= {used}; configured {length}.");
        var history = example.History ?? []; int first = history.Length;
        while (first > 0)
        {
            var turn = history[first - 1];
            int size = checked(ByteTokenizer.TokenCount(turn.User) + ByteTokenizer.TokenCount(turn.Assistant) + 4);
            if ((long)used + size > length) break;
            used += size; first--;
        }
        return first;
    }
    public static EncodedExample Encode(TrainingExample e, int length)
    {
        if (length < 8 || length > 2048) throw new ArgumentOutOfRangeException(nameof(length));
        // Reserve the complete target and count retained history before allocating the final arrays.
        int first = e.IsDialogue ? RetainedHistoryStart(e, length) : 0;
        var history = e.IsDialogue ? e.History ?? [] : Array.Empty<ChatTurn>();
        int n = RequiredSequenceLength(e);
        if (n > length) throw new ArgumentException("Text must be split before encoding; silent truncation is disabled.");
        for (int i = first; i < history.Length; i++)
            n = checked(n + ByteTokenizer.TokenCount(history[i].User) + ByteTokenizer.TokenCount(history[i].Assistant) + 4);
        var input = new int[n]; var labels = new int[n]; Array.Fill(labels, -100);
        int at = 0, answerStart = 1; input[at++] = ByteTokenizer.Bos;
        void Text(string value) { at += ByteTokenizer.EncodeInto(value, input.AsSpan(at)); }
        if (e.IsDialogue)
        {
            for (int i = first; i < history.Length; i++)
            {
                var turn = history[i];
                input[at++] = ByteTokenizer.User; Text(turn.User); input[at++] = ByteTokenizer.Eos;
                input[at++] = ByteTokenizer.Assistant; Text(turn.Assistant); input[at++] = ByteTokenizer.Eos;
            }
            input[at++] = ByteTokenizer.User; Text(e.Text); input[at++] = ByteTokenizer.Eos;
            input[at++] = ByteTokenizer.Assistant; answerStart = at; Text(e.Answer!);
        }
        else Text(e.Text);
        // The final EOS is a target, not an input. Shift only supervised positions; history is still causal input.
        if (at != n) throw new InvalidOperationException("Training measurement/encoding mismatch.");
        input.AsSpan(answerStart).CopyTo(labels.AsSpan(answerStart - 1)); labels[^1] = ByteTokenizer.Eos;
        return new(input, labels, e.Id);
    }
    public static TrainingExample[] SplitLongTexts(IEnumerable<TrainingExample> source, int length, CancellationToken cancellationToken = default)
    {
        if (length is < 8 or > 2048) throw new ArgumentOutOfRangeException(nameof(length));
        var result = new List<TrainingExample>();
        foreach (var e in source)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (result.Count >= MaxExamples) throw new ArgumentException("Too many records after text chunking.");
            if (e.IsDialogue || ByteTokenizer.TokenCount(e.Text) <= length - 2) { result.Add(e); continue; }
            // Split by Unicode scalar values without corrupting UTF-8 boundaries. Preserve source semantics.
            var part = new StringBuilder(); int bytes = 0;
            foreach (var rune in e.Text.EnumerateRunes())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (result.Count >= MaxExamples) throw new ArgumentException("Too many records after text chunking.");
                if (bytes + rune.Utf8SequenceLength > length - 2)
                { if (!string.IsNullOrWhiteSpace(part.ToString())) result.Add(Make(part.ToString(), source: e.Source)); part.Clear(); bytes = 0; }
                part.Append(rune); bytes += rune.Utf8SequenceLength;
            }
            if (!string.IsNullOrWhiteSpace(part.ToString())) result.Add(Make(part.ToString(), source: e.Source));
        }
        if (result.Count > MaxExamples) throw new ArgumentException("Too many records after text chunking.");
        return result.ToArray();
    }
    public static EncodedExample[] EncodeAll(TrainingExample[] items, int length, int threads, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = new EncodedExample[items.Length];
        int workers = Math.Clamp(threads, 1, Math.Max(1, Environment.ProcessorCount));
        // The usual online update encodes <=4 new and <=4 selected learned records. Do not launch a
        // parallel loop per tiny pool; large imports still use bounded chunked parallel work.
        if (items.Length < 128 || workers == 1)
        {
            for (int i = 0; i < items.Length; i++) { cancellationToken.ThrowIfCancellationRequested(); result[i] = Encode(items[i], length); }
        }
        else Parallel.For(0, (items.Length + 31) / 32,
            new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, chunk =>
            {
                int end = Math.Min(items.Length, chunk * 32 + 32);
                for (int i = chunk * 32; i < end; i++) { cancellationToken.ThrowIfCancellationRequested(); result[i] = Encode(items[i], length); }
            });
        return result;
    }
}
