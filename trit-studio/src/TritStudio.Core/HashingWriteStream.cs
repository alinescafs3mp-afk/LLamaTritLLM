using System.Security.Cryptography;
namespace TritStudio.Core;

// Non-owning write-only stream. Hash the exact bytes sent to disk, without a second full read.
// Cancellation is observed between bounded writes; an OS write/fsync itself is not interruptible.
public sealed class HashingWriteStream(Stream inner, CancellationToken ct = default) : Stream
{
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    private bool _finished;
    public string Finish()
    {
        if (_finished) throw new InvalidOperationException("Hash already finalized.");
        _finished = true; return Convert.ToHexString(_hash.GetHashAndReset());
    }
    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
    public override void Write(ReadOnlySpan<byte> buffer)
    {
        if (_finished) throw new InvalidOperationException("Cannot write after finalizing a hash.");
        ct.ThrowIfCancellationRequested();
        while (!buffer.IsEmpty)
        {
            ct.ThrowIfCancellationRequested();
            var part = buffer[..Math.Min(buffer.Length, 65536)];
            inner.Write(part); _hash.AppendData(part); buffer = buffer[part.Length..];
        }
    }
    public override void WriteByte(byte value) { Span<byte> data = stackalloc byte[1]; data[0] = value; Write(data); }
    public override void Flush() => inner.Flush();
    public override bool CanRead => false;
    public override bool CanSeek => false;
    public override bool CanWrite => !_finished && inner.CanWrite;
    public override long Length => inner.Length;
    public override long Position { get => inner.Position; set => throw new NotSupportedException(); }
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    protected override void Dispose(bool disposing) { if (disposing) { _finished = true; _hash.Dispose(); } base.Dispose(disposing); }
}
