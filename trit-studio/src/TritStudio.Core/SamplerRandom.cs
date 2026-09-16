namespace TritStudio.Core;
public sealed class SamplerRandom
{
    public ulong State { get; set; }
    public SamplerRandom(ulong seed) => State = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
    public int Next(int max)
    {
        if (max <= 0) throw new ArgumentOutOfRangeException(nameof(max));
        ulong x = State; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; State = x;
        return (int)(unchecked(x * 2685821657736338717UL) % (uint)max);
    }
}
