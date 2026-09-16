using System.Text.Json;
namespace TritStudio.Core;

// One slot per immutable, session-owned corpus. A new array identity always invalidates the slot.
// Payload cap excludes serializer/file buffers and the source corpus. Not a shared or mutable-data cache.
public sealed class CheckpointJsonCache(int payloadBudget = 4 * 1024 * 1024)
{
    private object? _owner;
    private byte[]? _bytes;
    private string? _hash;
    public long Hits { get; private set; }
    public long Serializations { get; private set; }
    public int RetainedBytes => _bytes?.Length ?? 0;
    public void Clear() { _owner = null; _bytes = null; _hash = null; }

    // For NEW files in a private staged revision only. Never replace an existing destination.
    public string WriteNew<T>(string path, T value, CancellationToken ct = default) where T : class
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(value);
        if (payloadBudget is < 0 or > 16 * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(payloadBudget));
        bool hit = ReferenceEquals(value, _owner) && _bytes is not null;
        if (!hit) Clear(); // Drop old retained corpus before preparing a replacement, including on failure.
        bool created = false;
        try
        {
            using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            created = true;
            if (hit)
            {
                // Bytes and digest belong to one immutable successful serialization. Do not rehash them.
                for (int offset = 0; offset < _bytes!.Length; offset += 65536)
                { ct.ThrowIfCancellationRequested(); file.Write(_bytes.AsSpan(offset, Math.Min(65536, _bytes.Length - offset))); }
                ct.ThrowIfCancellationRequested(); file.Flush(true); Hits++; return _hash!;
            }
            using var hashed = new HashingWriteStream(file, ct);
            using var limited = new BoundedWriteStream(hashed, JsonData.MaxStateBytes);
            using var recording = new RecordingStream(limited, payloadBudget);
            JsonSerializer.Serialize(recording, value, JsonData.Options);
            ct.ThrowIfCancellationRequested(); recording.Flush(); file.Flush(true);
            string hash = hashed.Finish(); byte[]? bytes = recording.TakeBytes();
            Serializations++;
            // Publish the cache only after serialization and durable file flush have succeeded.
            if (bytes is not null) { _owner = value; _bytes = bytes; _hash = hash; }
            return hash;
        }
        catch
        {
            if (created) { try { File.Delete(path); } catch (IOException) { } }
            throw;
        }
    }

    // Tee in a SINGLE serialization pass. Oversized payloads keep streaming, without a second attempt.
    private sealed class RecordingStream(Stream inner, int budget) : Stream
    {
        private MemoryStream? _capture = budget > 0 ? new MemoryStream(Math.Min(budget, 4096)) : null;
        public byte[]? TakeBytes() => _capture?.ToArray();
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            inner.Write(buffer);
            if (_capture is null) return;
            if (buffer.Length > budget - _capture.Length) { _capture.Dispose(); _capture = null; return; }
            _capture.Write(buffer);
        }
        public override void WriteByte(byte value) { Span<byte> b = stackalloc byte[1]; b[0] = value; Write(b); }
        public override void Flush() => inner.Flush();
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => inner.CanWrite;
        public override long Length => inner.Length;
        public override long Position { get => inner.Position; set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) _capture?.Dispose(); base.Dispose(disposing); }
    }
}
