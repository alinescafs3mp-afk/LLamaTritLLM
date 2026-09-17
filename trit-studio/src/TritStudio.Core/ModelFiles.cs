using System.Buffers;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
namespace TritStudio.Core;

public static class JsonData
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web) { WriteIndented = false };
    public const long MaxStateBytes = 128L * 1024 * 1024;
    public static T Read<T>(string path, long maxBytes = MaxStateBytes)
    {
        if (maxBytes is < 1 or > MaxStateBytes) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length > maxBytes) throw new InvalidDataException("JSON state exceeds its size budget.");
        return JsonSerializer.Deserialize<T>(stream, Options) ?? throw new InvalidDataException(path);
    }
    public static void AtomicWrite<T>(string path, T data, long maxBytes = MaxStateBytes)
    {
        if (maxBytes is < 1 or > MaxStateBytes) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        string temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                using var limited = new BoundedWriteStream(stream, maxBytes);
                JsonSerializer.Serialize(limited, data, Options);
                stream.Flush(true);
            }
            File.Move(temp, path, true);
        }
        finally { if (File.Exists(temp)) File.Delete(temp); }
    }
}
public sealed record RevisionInfo(long Revision, long Step, long ChangedWeights, double? ValidationLoss,
    string Reason, DateTimeOffset CreatedAt, string MasterSha256, string ModelSha256, long? Parent,
    string OptimizerSha256, string StateSha256, string CommitSha256, long TargetTokens, double? BestValidationLoss, int CheckpointVersion = 1, string? SettingsSha256 = null, string? TrainingDataSha256 = null, string? ValidationDataSha256 = null, int? ValidationSequenceLength = null, LearningProgress? Accuracy = null);
public sealed record ActiveRevision(string Directory);

public static class ModelFiles
{
    public static void Write(string path, WeightSet weights, bool packed, int threads = 1, CancellationToken ct = default)
        => WriteHashed(path, weights, packed, threads, ct);
    public static string WriteHashed(string path, WeightSet weights, bool packed, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); bool created = false;
        try
        {
            using var fs = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            created = true;
            using var hashed = new HashingWriteStream(fs, ct);
            using var w = new BinaryWriter(hashed, Encoding.UTF8, true);
            w.Write(packed ? "TRITSTUDIO_PACKED_V1" : "TRITSTUDIO_MASTER_V1");
            byte[] config = JsonSerializer.SerializeToUtf8Bytes(weights.Config, JsonData.Options);
            w.Write(config.Length); w.Write(config);
            foreach (var s in WeightLayout.For(weights.Config))
            {
                ct.ThrowIfCancellationRequested();
                if (packed && s.Quantized)
                {
                    var c = weights.Config;
                    foreach (var plane in TernaryQuantizer.Pack(weights.Values[s.Name], c.GroupSize, c.Planes, c.Threshold, threads, ct))
                    { WriteFloats(w, plane.Scales); w.Write(plane.Trits); }
                }
                else WriteFloats(w, weights.Values[s.Name]);
            }
            w.Flush(); fs.Flush(true); return hashed.Finish();
        }
        catch
        {
            // Never remove a pre-existing destination when CreateNew refused it.
            if (created) { try { File.Delete(path); } catch (IOException) { } }
            throw;
        }
    }
    public static WeightSet Read(string path, bool packed, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return ReadStream(stream, packed, threads, ct);
    }
    private static WeightSet ReadStream(Stream stream, bool packed, int threads, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (stream.Length is < 20 or > 600_000_000) throw new InvalidDataException("Invalid model file size.");
        using var r = new BinaryReader(stream, Encoding.UTF8, true);
        // BinaryReader.ReadString is avoided for untrusted variable-sized payloads except the bounded magic.
        int magicLength = r.ReadByte();
        string magic = Encoding.UTF8.GetString(ReadExact(r, magicLength));
        if (magic != (packed ? "TRITSTUDIO_PACKED_V1" : "TRITSTUDIO_MASTER_V1")) throw new InvalidDataException("Not a Trit Studio v1 model. Original LLamaTritLLM .bin files need a separate importer.");
        int size = r.ReadInt32();
        if (size is < 2 or > 16384) throw new InvalidDataException("Invalid config length.");
        var c = JsonSerializer.Deserialize<ModelConfig>(ReadExact(r, size), JsonData.Options) ?? throw new InvalidDataException("Missing config.");
        c.Validate();
        long expected = WeightLayout.For(c).Sum(s => packed && s.Quantized
            ? (long)c.Planes * (s.Count / c.GroupSize) * (4 + (c.GroupSize + 4) / 5) : (long)s.Count * 4);
        if (r.BaseStream.Length - r.BaseStream.Position != expected) throw new InvalidDataException("Model length does not match its declared shape.");
        var values = new Dictionary<string, float[]>();
        foreach (var s in WeightLayout.For(c))
        {
            if (packed && s.Quantized)
            {
                int groups = s.Count / c.GroupSize, nbytes = groups * ((c.GroupSize + 4) / 5);
                var planes = new PackedPlane[c.Planes];
                for (int p = 0; p < c.Planes; p++)
                {
                    var scales = ReadFloats(r, groups, ct);
                    planes[p] = new(scales, ReadExact(r, nbytes, ct));
                }
                values[s.Name] = TernaryQuantizer.Unpack(planes, s.Count, c.GroupSize, threads, ct);
            }
            else
            {
                var a = ReadFloats(r, s.Count, ct);
                values[s.Name] = a;
            }
        }
        if (r.BaseStream.Position != r.BaseStream.Length) throw new InvalidDataException("Unexpected trailing model data.");
        return new(c, values);
    }
    private static void WriteFloats(BinaryWriter w, float[] values)
    {
        if (BitConverter.IsLittleEndian) w.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        else foreach (float x in values) w.Write(x);
    }
    private static float[] ReadFloats(BinaryReader r, int count, CancellationToken ct)
    {
        var values = new float[count];
        if (BitConverter.IsLittleEndian) ReadChunks(r.BaseStream, MemoryMarshal.AsBytes(values.AsSpan()), ct);
        else for (int i = 0; i < count; i++) { if ((i & 16383) == 0) ct.ThrowIfCancellationRequested(); values[i] = r.ReadSingle(); }
        return values;
    }
    private static byte[] ReadExact(BinaryReader r, int n, CancellationToken ct = default)
    {
        var bytes = new byte[n]; ReadChunks(r.BaseStream, bytes, ct); return bytes;
    }
    private static void ReadChunks(Stream stream, Span<byte> bytes, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        while (!bytes.IsEmpty)
        {
            ct.ThrowIfCancellationRequested(); int count = Math.Min(bytes.Length, 65536);
            stream.ReadExactly(bytes[..count]); bytes = bytes[count..];
        }
    }
    public static string Hash(string path, CancellationToken ct = default)
    { ct.ThrowIfCancellationRequested(); using var fs = File.OpenRead(path); return HashStream(fs, ct); }
    private static string HashStream(Stream stream, CancellationToken ct)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(65536);
        try
        {
            while (true)
            {
                ct.ThrowIfCancellationRequested(); int read = stream.Read(buffer, 0, 65536);
                if (read == 0) break; hash.AppendData(buffer, 0, read);
            }
            return Convert.ToHexString(hash.GetHashAndReset());
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); }
    }
    public static string GetRevisionPath(string workspace, string name)
    {
        if (name is null || name.Length != 17 || !name.StartsWith("r", StringComparison.Ordinal) || name[1..].Any(x => x < '0' || x > '9'))
            throw new InvalidDataException("Invalid revision identifier.");
        return Path.Combine(Path.GetFullPath(workspace), "revisions", name);
    }
    public static string? ActivePath(string workspace)
    {
        string pointer = Path.Combine(workspace, "active.json");
        ModelLibrary.NoLinks(pointer);
        if (Directory.Exists(pointer)) throw new InvalidDataException("active.json является каталогом, а не указателем снимка.");
        if (!File.Exists(pointer))
        {
            string revisions = Path.Combine(workspace, "revisions");
            ModelLibrary.NoLinks(revisions);
            if (Directory.Exists(revisions) && Directory.EnumerateDirectories(revisions, "r*").Any())
                throw new InvalidDataException("Указатель active.json отсутствует, но сохранённые ревизии существуют. Восстановите указатель из проверенной резервной копии; новую модель поверх этих данных создавать нельзя.");
            return null;
        }
        return GetRevisionPath(workspace, JsonData.Read<ActiveRevision>(pointer, 4096).Directory);
    }
    public static RevisionInfo ReadRevisionInfo(string path)
    {
        ModelLibrary.NoLinks(path);
        string metadata = Path.Combine(path, "revision.json"); ModelLibrary.NoLinks(metadata);
        var info = JsonData.Read<RevisionInfo>(metadata, 65536);
        if (info.Revision < 0 || info.Step < 0 || info.ChangedWeights < 0 || info.TargetTokens < 0 ||
            (info.ValidationLoss is double loss && (!double.IsFinite(loss) || loss < 0)) ||
            (info.BestValidationLoss is double best && (!double.IsFinite(best) || best < 0)) ||
            (info.ValidationSequenceLength is int sequence && (sequence < 16 || sequence > 2048)) ||
            info.CheckpointVersion is < 1 or > 2 || Path.GetFileName(path) != $"r{info.Revision:D16}" ||
            (info.Parent is long parent && (parent < 0 || parent >= info.Revision)))
            throw new InvalidDataException("Invalid checkpoint metadata.");
        info.Accuracy?.Validate(info.Step);
        if (info.Accuracy?.Validation is { } control && (info.ValidationLoss is null || control.Step != info.Step || control.SequenceLength != info.ValidationSequenceLength))
            throw new InvalidDataException("Точность контрольного набора не соответствует снимку.");
        return info;
    }
    public static (WeightSet Weights, RevisionInfo Info) ReadInferenceRevision(string path, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); var info = ReadRevisionInfo(path);
        string packedFile = Path.Combine(path, "model.tritmodel"); ModelLibrary.NoLinks(packedFile);
        using var stream = new FileStream(packedFile, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length is < 20 or > 600_000_000) throw new InvalidDataException("Invalid model file size.");
        string hash = HashStream(stream, ct);
        if (!StringComparer.OrdinalIgnoreCase.Equals(hash, info.ModelSha256)) throw new InvalidDataException("Inference snapshot checksum mismatch.");
        stream.Position = 0;
        return (ReadStream(stream, true, threads, ct), info);
    }
    public static RevisionInfo VerifyRevision(string path)
    {
        var info = ReadRevisionInfo(path);
        if (Hash(Path.Combine(path, "master.weights")) != info.MasterSha256 || Hash(Path.Combine(path, "model.tritmodel")) != info.ModelSha256 ||
            Hash(Path.Combine(path, "optimizer.bin")) != info.OptimizerSha256 || Hash(Path.Combine(path, "state.json")) != info.StateSha256 ||
            Hash(Path.Combine(path, "commit.json")) != info.CommitSha256)
            throw new InvalidDataException("Checkpoint checksum mismatch; active model was not changed.");
        if (info.CheckpointVersion is < 1 or > 2) throw new InvalidDataException("Unsupported checkpoint version.");
        if (info.CheckpointVersion >= 2 && (Hash(Path.Combine(path, "settings.json")) != info.SettingsSha256 ||
            Hash(Path.Combine(path, "base-train.json")) != info.TrainingDataSha256 || Hash(Path.Combine(path, "validation.json")) != info.ValidationDataSha256))
            throw new InvalidDataException("Checkpoint input/settings checksum mismatch.");
        if (Path.GetFileName(path) != $"r{info.Revision:D16}") throw new InvalidDataException("Revision metadata does not match the folder.");
        return info;
    }
}
