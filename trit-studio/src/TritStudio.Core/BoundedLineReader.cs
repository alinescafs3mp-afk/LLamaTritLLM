using System.Text;
namespace TritStudio.Core;

// Single-reader framing. Never accumulate more than maxChars + one optional CR terminator.
// Diagnostic streams drain excess characters instead of blocking the child's stderr pipe.
public sealed class BoundedLineReader(TextReader reader, int maxChars, bool drainOverflow = false)
{
    private readonly char[] _buffer = new char[4096];
    private int _start, _end;
    public bool LastLineTruncated { get; private set; }
    public async ValueTask<string?> ReadLineAsync(CancellationToken ct = default)
    {
        if (maxChars is < 1 or > 16_777_216) throw new ArgumentOutOfRangeException(nameof(maxChars));
        StringBuilder? result = null; bool any = false; LastLineTruncated = false;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            if (_start == _end)
            {
                _end = await reader.ReadAsync(_buffer.AsMemory(), ct).ConfigureAwait(false); _start = 0;
                if (_end == 0) return any ? Finish(result!) : null;
            }
            int newline = Array.IndexOf(_buffer, '\n', _start, _end - _start);
            int end = newline < 0 ? _end : newline;
            int count = end - _start;
            if (result is null && newline >= 0)
            {
                int payload = count > 0 && _buffer[end - 1] == '\r' ? count - 1 : count;
                if (payload > maxChars && !drainOverflow) throw TooLarge();
                LastLineTruncated = payload > maxChars;
                string text = new(_buffer, _start, Math.Min(payload, maxChars));
                _start = newline + 1; return text;
            }
            result ??= new StringBuilder(Math.Min(maxChars, 256));
            int room = maxChars + 1 - result.Length;
            if (count > room && !drainOverflow) throw TooLarge();
            int keep = Math.Min(room, count);
            result.Append(_buffer, _start, keep);
            if (keep < count) LastLineTruncated = true;
            if (result.Length > maxChars && result[result.Length - 1] != '\r')
            {
                if (!drainOverflow) throw TooLarge();
                LastLineTruncated = true;
            }
            any = true; _start = newline < 0 ? _end : newline + 1;
            if (newline >= 0) return Finish(result);
        }
    }
    private InvalidDataException TooLarge() => new($"Line exceeds the {maxChars} character protocol limit.");
    private string Finish(StringBuilder line)
    {
        if (!LastLineTruncated && line.Length > 0 && line[line.Length - 1] == '\r') line.Length--;
        if (line.Length > maxChars)
        {
            if (!drainOverflow) throw TooLarge();
            line.Length = maxChars; LastLineTruncated = true;
        }
        return line.ToString();
    }
}
