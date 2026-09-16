using System.Buffers;
using System.Text;
using System.Text.Json;
namespace TritStudio.Core;

public sealed record ChatReadResult(ChatTurn[] Turns, int SkippedRecords, long BytesRead, int RecordsInspected = 0);
// Local plaintext journal. A partial crash tail is retained for diagnosis, never joined to the next record.
public static class ChatJournal
{
    private const int MaxRecordBytes = 1_048_576;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    public static Task AppendAsync(string path, ChatTurn turn) => Task.Run(() =>
    {
        ValidateTurn(turn);
        byte[] record = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(turn, JsonData.Options) + "\n");
        if (record.Length > MaxRecordBytes) throw new ArgumentException("Chat record is too large.");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var file = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
        if (file.Length > 0)
        {
            file.Seek(-1, SeekOrigin.End);
            if (file.ReadByte() != (byte)'\n') file.WriteByte((byte)'\n');
        }
        file.Seek(0, SeekOrigin.End); file.Write(record); file.Flush(flushToDisk: true);
    });
    public static ChatReadResult ReadTail(string path, int limit = 250, int maxBytes = 4 * 1024 * 1024, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ValidateLimits(limit, maxBytes);
        if (!File.Exists(path)) return new([], 0, 0);
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        return ReadTail(file, limit, maxBytes, ct);
    }
    // The length is captured ONCE. Appends cannot extend this read or make it chase a moving EOF.
    // Leaves caller's seekable stream open. Reads at most maxBytes of payload plus one boundary byte.
    public static ChatReadResult ReadTail(Stream file, int limit = 250, int maxBytes = 4 * 1024 * 1024, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ValidateLimits(limit, maxBytes);
        ArgumentNullException.ThrowIfNull(file);
        if (!file.CanRead || !file.CanSeek) throw new ArgumentException("Journal snapshot requires a readable seekable stream.");
        long end = file.Length, start = Math.Max(0, end - maxBytes);
        bool boundary = start == 0;
        if (start > 0) { file.Position = start - 1; boundary = file.ReadByte() == (byte)'\n'; }
        int wanted = checked((int)(end - start)), read = 0;
        if (wanted == 0) return new([], 0, 0);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(wanted);
        try
        {
            file.Position = start;
            while (read < wanted)
            {
                ct.ThrowIfCancellationRequested();
                int n = file.Read(buffer, read, Math.Min(65536, wanted - read));
                if (n == 0) break; // Concurrent truncation is bounded too. Never retry indefinitely.
                read += n;
            }
            int lower = 0;
            if (!boundary)
            {
                int newline = buffer.AsSpan(0, read).IndexOf((byte)'\n');
                if (newline < 0) return new([], 0, read);
                lower = newline + 1; // Discard only a partial leading record, not one starting at a known boundary.
            }
            var turns = new List<ChatTurn>(limit); int skipped = 0, inspected = 0, cursor = read;
            // Parse newest first and stop when enough valid exchanges have been collected.
            // Old bytes outside the requested tail are not a whole-journal integrity audit.
            while (cursor > lower && turns.Count < limit)
            {
                ct.ThrowIfCancellationRequested();
                int newline = buffer.AsSpan(lower, cursor - lower).LastIndexOf((byte)'\n');
                int lineStart = newline < 0 ? lower : lower + newline + 1;
                var line = buffer.AsSpan(lineStart, cursor - lineStart);
                cursor = newline < 0 ? lower : lower + newline;
                if (start == 0 && lineStart == 0 && line.Length >= 3 && line[0] == 0xEF && line[1] == 0xBB && line[2] == 0xBF) line = line[3..];
                if (Blank(line)) continue;
                inspected++;
                try
                {
                    if (line.Length > MaxRecordBytes) throw new InvalidDataException("Oversized chat record.");
                    _ = StrictUtf8.GetCharCount(line); // Never train later on replacement-decoded damaged bytes.
                    var turn = JsonSerializer.Deserialize<ChatTurn>(line, JsonData.Options) ?? throw new InvalidDataException("Empty chat record.");
                    ValidateTurn(turn); turns.Add(turn);
                }
                catch (Exception error) when (error is JsonException or InvalidDataException or ArgumentException) { skipped++; }
            }
            turns.Reverse(); return new(turns.ToArray(), skipped, read, inspected);
        }
        finally { ArrayPool<byte>.Shared.Return(buffer, clearArray: true); }
    }
    private static bool Blank(ReadOnlySpan<byte> value)
    {
        foreach (byte b in value) if (b is not (9 or 10 or 13 or 32)) return false;
        return true;
    }
    private static void ValidateLimits(int limit, int bytes)
    {
        if (limit is < 1 or > 1000 || bytes is < 1024 or > 16 * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(limit));
    }
    private static void ValidateTurn(ChatTurn? turn)
    {
        static void ValidateText(ChatTurn? item)
        {
            if (item is null || item.User is null || item.Assistant is null || item.User.Length > 32768 || item.Assistant.Length > 32768 ||
                item.ConversationId is null || item.ConversationId.Length > 256)
                throw new InvalidDataException("Invalid chat text or conversation identifier.");
            _ = StrictUtf8.GetByteCount(item.User); _ = StrictUtf8.GetByteCount(item.Assistant);
        }
        ValidateText(turn);
        if (turn!.LearningHistory is not { } history) return; // Legacy records have no captured teaching context.
        if (history.Length > 8) throw new InvalidDataException("Too many captured teaching exchanges.");
        foreach (var item in history)
        {
            ValidateText(item);
            if (item.LearningHistory is not null || item.ExcludedFromTraining || item.ConversationId != turn.ConversationId)
                throw new InvalidDataException("Invalid captured teaching history.");
        }
    }
}
