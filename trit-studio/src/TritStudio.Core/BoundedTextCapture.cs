using System.Text;
namespace TritStudio.Core;

public sealed record CapturedText(string Text, bool Truncated);

// Retain a bounded prefix, but keep draining the pipe so native diagnostics cannot block a child process.
public static class BoundedTextCapture
{
    public static async Task<CapturedText> ReadAsync(TextReader reader, int limit, CancellationToken ct = default)
    {
        if (limit is < 1 or > 1_048_576) throw new ArgumentOutOfRangeException(nameof(limit));
        var buffer = new char[4096]; var text = new StringBuilder(Math.Min(256, limit)); bool truncated = false;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            int read = await reader.ReadAsync(buffer.AsMemory(), ct).ConfigureAwait(false);
            if (read == 0) return new(text.ToString(), truncated);
            int keep = Math.Min(read, limit - text.Length);
            text.Append(buffer, 0, keep); truncated |= keep < read;
        }
    }
}
