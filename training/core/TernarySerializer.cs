//https://github.com/virex-84

//TernarySerializer.cs

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace TernaryLLM;

public static class TernarySerializer
{
    public static void WriteFloats(BinaryWriter w, float[] data)
    {
        w.Write(data.Length);
        if (data.Length > 0)
        {
            var bytes = MemoryMarshal.AsBytes(data.AsSpan());
            w.Write(bytes);
        }
    }

    public static float[] ReadFloats(BinaryReader r)
    {
        int len = r.ReadInt32();
        var data = new float[len];
        if (len > 0)
        {
            var bytes = MemoryMarshal.AsBytes(data.AsSpan());
            int read = r.BaseStream.Read(bytes);
            if (read != bytes.Length)
                throw new InvalidDataException("Неожиданный конец файла (float)");
        }
        return data;
    }
}