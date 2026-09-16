using System.Text;
namespace TritStudio.Core;

public static class StrictDatasetLines
{
    public const int MaxJsonLineChars = 1_048_576;
    public static IEnumerable<string> Read(string path, int maxChars, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (file.Length > 64L * 1024 * 1024) throw new InvalidDataException("Dataset import limit is 64 MiB per file.");
        // Accept an optional UTF-8 BOM. Never auto-switch to a replacement-fallback UTF-16/UTF-8 decoder.
        int a = file.ReadByte(), b = file.ReadByte(), c = file.ReadByte();
        file.Position = a == 0xEF && b == 0xBB && c == 0xBF ? 3 : 0;
        if ((a == 0xFF && b == 0xFE) || (a == 0xFE && b == 0xFF) || (a == 0 && b == 0))
            throw new InvalidDataException("Сохраните датасет в UTF-8. UTF-16/UTF-32 здесь не перекодируются автоматически.");
        using var reader = new StreamReader(file, new UTF8Encoding(false, true), detectEncodingFromByteOrderMarks: false);
        var lines = new BoundedLineReader(reader, maxChars); int number = 0;
        while (true)
        {
            string? line;
            try { line = lines.ReadLineAsync(ct).GetAwaiter().GetResult(); }
            catch (OperationCanceledException) { throw; }
            catch (Exception e) when (e is DecoderFallbackException or InvalidDataException)
            { throw new InvalidDataException($"Датасет {Path.GetFileName(path)}, около строки {number + 1}: повреждён UTF-8 или превышен предел {maxChars} символов на строку.", e); }
            if (line is null) yield break;
            number++; yield return line;
        }
    }
}
