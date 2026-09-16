namespace TritStudio.Core;

// Call only while holding the workspace trainer lease. Gaps are allowed; reused IDs are not.
public static class RevisionSequence
{
    private const long LargestId = 9_999_999_999_999_999;
    public static long PreserveHighWater(string workspace)
    {
        string path = Path.Combine(workspace, "revision-counter.json");
        long saved = File.Exists(path) ? JsonData.Read<long>(path) : -1;
        if (saved < -1 || saved > LargestId) throw new InvalidDataException("Invalid revision counter.");
        string directory = Path.Combine(workspace, "revisions");
        long observed = Directory.Exists(directory) ? Directory.EnumerateDirectories(directory, "r*")
            .Select(Path.GetFileName).Where(n => n is { Length: 17 } && n[1..].All(char.IsAsciiDigit))
            .Select(n => long.Parse(n![1..], System.Globalization.CultureInfo.InvariantCulture)).DefaultIfEmpty(-1).Max() : -1;
        long high = Math.Max(saved, observed);
        if (!File.Exists(path) || high != saved) JsonData.AtomicWrite(path, high);
        return high;
    }
    public static long Reserve(string workspace)
    {
        long high = PreserveHighWater(workspace);
        if (high >= LargestId) throw new InvalidOperationException("Revision ID space exhausted.");
        long next = high + 1;
        // Persist before writing a candidate. A crash can consume an ID, never recycle one.
        JsonData.AtomicWrite(Path.Combine(workspace, "revision-counter.json"), next);
        return next;
    }
}
