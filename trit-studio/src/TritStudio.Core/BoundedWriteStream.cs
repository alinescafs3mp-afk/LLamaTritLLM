namespace TritStudio.Core;

// Non-owning write-only adapter. Refuse the write BEFORE it grows a staging file past the state budget.
public sealed class BoundedWriteStream(Stream inner, long limit) : Stream
{
    private long _written;
    public override bool CanRead => false;
    public override bool CanSeek => false;
    public override bool CanWrite => inner.CanWrite;
    public override long Length => _written;
    public override long Position { get => _written; set => throw new NotSupportedException(); }
    private void Reserve(int count)
    {
        if (limit < 0 || count < 0 || _written > limit - count) throw new InvalidDataException("JSON state exceeds its size budget; previous state was retained.");
    }
    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
    public override void Write(ReadOnlySpan<byte> buffer)
    { Reserve(buffer.Length); inner.Write(buffer); _written += buffer.Length; }
    public override void WriteByte(byte value)
    { Reserve(1); inner.WriteByte(value); _written++; }
    public override void Flush() => inner.Flush();
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
}
