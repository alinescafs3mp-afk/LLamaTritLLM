namespace TritStudio.Core;
// Enforces canonical UTF-8 so byte-level generation never emits replacement-character soup.
public sealed class Utf8Guard
{
    private int _remaining, _minimum = 0x80, _maximum = 0xBF;
    public bool Complete => _remaining == 0;
    public int RemainingBytes => _remaining;
    public bool AllowsWithinBudget(int token, int remainingBytes)
    {
        if (!Allows(token)) return false;
        if (token == ByteTokenizer.Eos) return true;
        if (remainingBytes < 1) return false;
        if (_remaining > 0) return remainingBytes >= _remaining;
        int b = token - ByteTokenizer.Offset;
        int size = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        return size <= remainingBytes;
    }
    public bool Allows(int token)
    {
        if (token == ByteTokenizer.Eos) return Complete;
        int b = token - ByteTokenizer.Offset;
        if (b < 0 || b > 255) return false;
        if (_remaining != 0) return b >= _minimum && b <= _maximum;
        return b is 9 or 10 or 13 || b is >= 0x20 and <= 0x7E || b is >= 0xC2 and <= 0xF4;
    }
    public void Accept(int token)
    {
        if (!Allows(token) || token == ByteTokenizer.Eos) throw new ArgumentException("Invalid UTF-8 token.");
        int b = token - ByteTokenizer.Offset;
        if (_remaining > 0) { _remaining--; _minimum = 0x80; _maximum = 0xBF; return; }
        if (b < 0x80) return;
        _remaining = b < 0xE0 ? 1 : b < 0xF0 ? 2 : 3;
        _minimum = b == 0xE0 ? 0xA0 : b == 0xF0 ? 0x90 : 0x80;
        _maximum = b == 0xED ? 0x9F : b == 0xF4 ? 0x8F : 0xBF;
    }
}
