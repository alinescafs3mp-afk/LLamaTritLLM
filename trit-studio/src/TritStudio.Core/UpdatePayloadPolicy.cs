namespace TritStudio.Core;

// Shared by the packager and managed tests. An update replaces OUR binaries, never vendor runtime files.
public static class UpdatePayloadPolicy
{
    private static readonly HashSet<string> Owners = new(StringComparer.Ordinal)
    { "TritStudio", "TritStudio.App", "TritStudio.Core", "TritStudio.Trainer", "TritStudio.Runner", "TritStudio.Tests" };
    private static readonly string[] Suffixes = [".deps.json", ".runtimeconfig.json", ".exe", ".dll", ".pdb"];
    public static bool IsOwnedBinary(string relative)
    {
        string p = Normalize(relative); string[] parts = p.Split('/');
        if (parts.Length > 2 || parts.Length == 2 && parts[0] is not ("trainer" or "trainer-cuda" or "runner" or "checks")) return false;
        string name = parts[^1];
        foreach (string suffix in Suffixes) if (name.EndsWith(suffix, StringComparison.Ordinal)) return Owners.Contains(name[..^suffix.Length]);
        return Owners.Contains(name); // Unix apphost; Windows uses .exe.
    }
    // Each published trainer and test runner reads its OWN adjacent data folder before any root fallback.
    // Those copies are application content, not vendor dependencies; leaving v14 copies would reject v15.
    public static bool IsOwnedContent(string relative)
    {
        string p = Normalize(relative); string[] parts = p.Split('/');
        if (p is "checks/reference.json" or "checks/learning-reference.tritmodel") return true;
        if (parts.Length != 3 || parts[0] is not ("trainer" or "trainer-cuda" or "checks") || parts[1] != "data") return false;
        string name = parts[2];
        if (name is "conversation-language.jsonl" or "conversation-transfer.jsonl" or "transfer-challenge.jsonl" or "conversation-context.jsonl" or "conversation-starter.jsonl" or "context-challenge.jsonl" or "DATASET_MANIFEST.json" or "seed.jsonl" or "validation.jsonl" or "test.jsonl" or "challenge.jsonl" or "pretrain.jsonl") return true;
        if (name.Length <= 17 || !name.StartsWith("challenge-v",StringComparison.Ordinal) || !name.EndsWith(".jsonl",StringComparison.Ordinal)) return false;
        var digits = name.AsSpan(11,name.Length-17);
        foreach (char c in digits) if (c < '0' || c > '9') return false;
        return int.TryParse(digits,out int version) && version >= 4;
    }
    public static bool IsOwnedPayload(string relative) => IsOwnedBinary(relative) || IsOwnedContent(relative);
    public static string Normalize(string relative)
    {
        if (string.IsNullOrWhiteSpace(relative) || Path.IsPathRooted(relative) || relative.Contains('\\') || relative.Contains(':'))
            throw new InvalidDataException("Unsafe update path.");
        if (relative.Split('/').Any(x => x is "" or "." or "..")) throw new InvalidDataException("Unsafe update path.");
        return relative;
    }
}
