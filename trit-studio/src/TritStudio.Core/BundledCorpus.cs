using System.Text.Json;
namespace TritStudio.Core;

public static class BundledCorpus
{
    public const string Version = "conversation-ru-v17";
    public static TrainingExample[] Load(string directory, string name, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (name is not ("seed.jsonl" or "validation.jsonl" or "test.jsonl" or "challenge.jsonl" or "challenge-v4.jsonl" or "challenge-v5.jsonl" or "challenge-v6.jsonl" or "challenge-v7.jsonl" or "challenge-v8.jsonl" or "challenge-v9.jsonl" or "challenge-v10.jsonl" or "challenge-v11.jsonl" or "challenge-v12.jsonl" or "challenge-v13.jsonl" or "challenge-v14.jsonl" or "challenge-v15.jsonl" or "challenge-v16.jsonl" or "challenge-v17.jsonl" or "pretrain.jsonl"))
            throw new ArgumentException("Unknown bundled corpus file.");
        string manifestPath = Path.Combine(directory, "DATASET_MANIFEST.json");
        using var manifestStream = new FileStream(manifestPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (manifestStream.Length is < 2 or > 65536) throw new InvalidDataException("Bundled corpus manifest exceeds its budget.");
        using var manifest = JsonDocument.Parse(manifestStream);
        if (manifest.RootElement.GetProperty("version").GetString() != Version)
            throw new InvalidDataException("Bundled corpus version does not match this application. Restore the complete package.");
        string key = Path.GetFileNameWithoutExtension(name);
        var entry = manifest.RootElement.GetProperty("files").GetProperty(key);
        string path = Path.Combine(directory, name);
        if (new FileInfo(path).Length != entry.GetProperty("bytes").GetInt64() ||
            !string.Equals(ModelFiles.Hash(path, ct), entry.GetProperty("sha256").GetString(), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Bundled corpus checksum mismatch: {name}. Restore the complete application package.");
        var rows = name is "seed.jsonl" or "pretrain.jsonl" ? Dataset.LoadTraining(path, ct) : Dataset.Load(path, ct);
        if (rows.Length != entry.GetProperty("count").GetInt32()) throw new InvalidDataException("Bundled corpus count mismatch: " + name);
        return rows;
    }
}
