using System.Buffers;
namespace TritStudio.Core;

public sealed record StageReference(string Stage, string Name, long Revision, long Step,
    string ModelSha256, DateTimeOffset CreatedAt, bool InferenceOnly = true);

public sealed record StagePreservation(string File, StageReference Reference, bool Created);

public static class StageArchive
{
    private static void ValidateKey(string key)
    { if (key is not ("00-untrained" or "01-basic" or "02-conversation" or "03-custom")) throw new ArgumentException("Unknown evolution stage."); }
    private static void ValidateStep(string key, long step)
    {
        if (step < 0 || (key == "00-untrained" ? step != 0 : step == 0))
            throw new InvalidDataException("Эталон этапа не соответствует фактическому числу шагов обучения.");
    }
    public static string Root(string workspace) => Path.Combine(Path.GetFullPath(workspace), "evolution");
    private static void RejectLinks(string path)
    {
        for (string? item = Path.GetFullPath(path); item is not null; item = Path.GetDirectoryName(item))
            if ((Directory.Exists(item) || File.Exists(item)) && (File.GetAttributes(item) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Evolution archive cannot follow linked paths.");
    }
    // The workspace owner serializes writes. First successful stage is immutable, not silently refreshed later.
    public static string Preserve(string workspace, string revisionPath, string key, string name, CancellationToken ct = default)
        => PreserveWithReceipt(workspace, revisionPath, key, name, ct).File;
    public static StagePreservation PreserveWithReceipt(string workspace, string revisionPath, string key, string name, CancellationToken ct = default)
    {
        ValidateKey(key); ct.ThrowIfCancellationRequested();
        string root = Root(workspace); RejectLinks(root); Directory.CreateDirectory(root);
        string destination = Path.Combine(root, key); RejectLinks(destination);
        if (Directory.Exists(destination))
        {
            RejectLinks(Path.Combine(destination, "model.tritmodel")); RejectLinks(Path.Combine(destination, "stage.json"));
            var saved = JsonData.Read<StageReference>(Path.Combine(destination, "stage.json"), 65536);
            if (saved.Stage != key || !saved.InferenceOnly || saved.Revision < 0 || saved.Step < 0 ||
                ModelFiles.Hash(Path.Combine(destination, "model.tritmodel"), ct) != saved.ModelSha256)
                throw new InvalidDataException("Saved evolution reference failed integrity verification.");
            ValidateStep(key, saved.Step);
            return new(Path.Combine(destination, "model.tritmodel"), saved, false);
        }
        var info = ModelFiles.ReadRevisionInfo(revisionPath); ValidateStep(key, info.Step);
        string source = Path.Combine(revisionPath, "model.tritmodel"); RejectLinks(source);
        string staging = Path.Combine(root, ".stage-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(staging);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(131072);
        try
        {
            string hash;
            using (var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var output = new FileStream(Path.Combine(staging, "model.tritmodel"), FileMode.CreateNew, FileAccess.Write, FileShare.None))
            using (var hashed = new HashingWriteStream(output, ct))
            {
                if (input.Length is < 20 or > 600_000_000) throw new InvalidDataException("Invalid reference model size.");
                int read; while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                { ct.ThrowIfCancellationRequested(); hashed.Write(buffer.AsSpan(0, read)); }
                output.Flush(true); hash = hashed.Finish();
            }
            if (hash != info.ModelSha256) throw new InvalidDataException("Reference source checksum mismatch.");
            var reference = new StageReference(key, name, info.Revision, info.Step, hash, DateTimeOffset.UtcNow);
            JsonData.AtomicWrite(Path.Combine(staging, "stage.json"), reference, 65536);
            ct.ThrowIfCancellationRequested(); Directory.Move(staging, destination);
            return new(Path.Combine(destination, "model.tritmodel"), reference, true);
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); if (Directory.Exists(staging)) Directory.Delete(staging, true); }
    }
}
