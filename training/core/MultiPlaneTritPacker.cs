//https://github.com/virex-84

//MultiPlaneTritPacker.cs

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace TernaryLLM;

/// <summary>
/// Режим хранения масштабов.
/// </summary>
public enum TritScaleMode
{
    /// <summary>
    /// Один масштаб на всю матрицу/вектор.
    /// </summary>
    Global,

    /// <summary>
    /// Один масштаб на строку.
    /// </summary>
    PerRow,

    /// <summary>
    /// Один масштаб на группу внутри строки.
    /// </summary>
    PerGroup
}

/// <summary>
/// Универсальный multi-plane ternary packer.
///
/// Каждый вес представляется как:
///
///     w ≈ Σ scale[p] * t[p]
///
/// где t[p] ∈ {-1, 0, +1}
///
/// Количество плоскостей задается через Planes.
/// Чем больше Planes, тем выше точность/емкость.
/// </summary>
public sealed class MultiPlaneTritPacker
{
    /// <summary>
    /// 64 трита на одну битовую маску.
    /// </summary>
    public const int TritsPerWord = 64;


    public int Rows { get; set; }

    public int Cols { get; set; }

    public int Planes { get; set; }

    public int WordsPerRow => WordsNeeded(Cols);

    public int Length => Rows * Cols;

    public int GroupSize { get; private set; }

    public TritScaleMode ScaleMode { get; private set; }

    /// <summary>
    /// P[plane][row * WordsPerRow + word]
    /// </summary>
    public ulong[][] P { get; set; }

    /// <summary>
    /// N[plane][row * WordsPerRow + word]
    /// </summary>
    public ulong[][] N { get; set; }

    /// <summary>
    /// Масштабы.
    ///
    /// Layout зависит от ScaleMode:
    ///
    /// Global:
    ///     Scales[plane]
    ///
    /// PerRow:
    ///     Scales[row * Planes + plane]
    ///
    /// PerGroup:
    ///     Scales[(row * GroupsPerRow + group) * Planes + plane]
    /// </summary>
    public float[] Scales { get; private set; }

    public int GroupsPerRow
    {
        get
        {
            int g = GroupSize <= 0 ? Cols : GroupSize;
            return (Cols + g - 1) / g;
        }
    }

    //====================================================================
    // Constructors
    //====================================================================

    /// <summary>
    /// Создает матричный packer.
    /// </summary>
    public MultiPlaneTritPacker(int rows, int cols, int planes)
    {
        if (rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows));

        if (cols <= 0)
            throw new ArgumentOutOfRangeException(nameof(cols));

        if (planes <= 0)
            throw new ArgumentOutOfRangeException(nameof(planes));

        Rows = rows;
        Cols = cols;
        Planes = planes;

        GroupSize = cols;
        ScaleMode = TritScaleMode.Global;
        Scales = new float[planes];

        int totalWords = rows * WordsPerRow;

        P = new ulong[planes][];
        N = new ulong[planes][];

        for (int p = 0; p < planes; p++)
        {
            P[p] = new ulong[totalWords];
            N[p] = new ulong[totalWords];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WordsNeeded(int tritCount)
        => (tritCount + TritsPerWord - 1) / TritsPerWord;

    //====================================================================
    // Quantization / Packing
    //====================================================================

    /// <summary>
    /// Создает multi-plane ternary представление из FP32-данных.
    /// </summary>
    public static MultiPlaneTritPacker FromFloat(
        float[] data,
        int rows,
        int cols,
        int planes,
        int groupSize = 0,
        TritScaleMode scaleMode = TritScaleMode.PerGroup,
        float thresholdScaleFraction = 0.0f)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        if (data.Length != rows * cols)
            throw new ArgumentException("data.Length != rows * cols");

        if (planes <= 0)
            throw new ArgumentOutOfRangeException(nameof(planes));

        var packer = new MultiPlaneTritPacker(rows, cols, planes);

        packer.GroupSize = groupSize <= 0 ? cols : groupSize;
        packer.ScaleMode = scaleMode;
        packer.Scales = new float[packer.GetScaleCount()];

        var residual = (float[])data.Clone();

        for (int plane = 0; plane < planes; plane++)
        {
            switch (scaleMode)
            {
                case TritScaleMode.Global:
                    {
                        float scale = MeanAbs(residual, 0, residual.Length);
                        packer.Scales[plane] = scale;

                        float threshold = scale * thresholdScaleFraction;

                        for (int i = 0; i < residual.Length; i++)
                        {
                            sbyte t = TernaryValue(residual[i], threshold);
                            packer.SetFlat(plane, i, t);
                            residual[i] -= scale * t;
                        }

                        break;
                    }

                case TritScaleMode.PerRow:
                    {
                        for (int row = 0; row < rows; row++)
                        {
                            int start = row * cols;

                            float scale = MeanAbs(residual, start, cols);
                            packer.SetScale(row, 0, plane, scale);

                            float threshold = scale * thresholdScaleFraction;

                            for (int col = 0; col < cols; col++)
                            {
                                int flat = start + col;

                                sbyte t = TernaryValue(residual[flat], threshold);
                                packer.Set(plane, row, col, t);

                                residual[flat] -= scale * t;
                            }
                        }

                        break;
                    }

                case TritScaleMode.PerGroup:
                default:
                    {
                        int groups = packer.GroupsPerRow;

                        for (int row = 0; row < rows; row++)
                        {
                            for (int group = 0; group < groups; group++)
                            {
                                int startCol = group * packer.GroupSize;
                                int len = Math.Min(packer.GroupSize, cols - startCol);

                                int startFlat = row * cols + startCol;

                                float scale = MeanAbs(residual, startFlat, len);
                                packer.SetScale(row, group, plane, scale);

                                float threshold = scale * thresholdScaleFraction;

                                for (int c = 0; c < len; c++)
                                {
                                    int col = startCol + c;
                                    int flat = startFlat + c;

                                    sbyte t = TernaryValue(residual[flat], threshold);
                                    packer.Set(plane, row, col, t);

                                    residual[flat] -= scale * t;
                                }
                            }
                        }

                        break;
                    }
            }
        }

        return packer;
    }

    //====================================================================
    // Get / Set
    //====================================================================

    public void Set(int plane, int row, int col, sbyte trit)
    {
        if ((uint)plane >= (uint)Planes)
            throw new ArgumentOutOfRangeException(nameof(plane));

        if ((uint)row >= (uint)Rows)
            throw new ArgumentOutOfRangeException(nameof(row));

        if ((uint)col >= (uint)Cols)
            throw new ArgumentOutOfRangeException(nameof(col));

        int word = row * WordsPerRow + (col >> 6);
        int bit = col & 63;

        ulong mask = 1UL << bit;

        switch (trit)
        {
            case 1:
                P[plane][word] |= mask;
                N[plane][word] &= ~mask;
                break;

            case -1:
                P[plane][word] &= ~mask;
                N[plane][word] |= mask;
                break;

            default:
                P[plane][word] &= ~mask;
                N[plane][word] &= ~mask;
                break;
        }
    }

    public sbyte Get(int plane, int row, int col)
    {
        int word = row * WordsPerRow + (col >> 6);
        int bit = col & 63;

        ulong mask = 1UL << bit;

        bool plus = (P[plane][word] & mask) != 0;
        bool minus = (N[plane][word] & mask) != 0;

        if (plus && !minus) return 1;
        if (!plus && minus) return -1;

        return 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetFlat(int plane, int index, sbyte trit)
    {
        int row = index / Cols;
        int col = index % Cols;

        Set(plane, row, col, trit);
    }

    public void ClearPlane(int plane)
    {
        Array.Clear(P[plane], 0, P[plane].Length);
        Array.Clear(N[plane], 0, N[plane].Length);
    }

    //====================================================================
    // Scales
    //====================================================================

    public float GetScale(int row, int group, int plane)
    {
        if (group < 0)
            group = 0;

        int groups = GroupsPerRow;

        if (group >= groups)
            group = groups - 1;

        switch (ScaleMode)
        {
            case TritScaleMode.Global:
                return Scales[plane];

            case TritScaleMode.PerRow:
                return Scales[row * Planes + plane];

            case TritScaleMode.PerGroup:
            default:
                return Scales[(row * groups + group) * Planes + plane];
        }
    }

    private void SetScale(int row, int group, int plane, float value)
    {
        if (group < 0)
            group = 0;

        int groups = GroupsPerRow;

        if (group >= groups)
            group = groups - 1;

        switch (ScaleMode)
        {
            case TritScaleMode.Global:
                Scales[plane] = value;
                break;

            case TritScaleMode.PerRow:
                Scales[row * Planes + plane] = value;
                break;

            case TritScaleMode.PerGroup:
            default:
                Scales[(row * groups + group) * Planes + plane] = value;
                break;
        }
    }

    private int GetScaleCount()
    {
        return ScaleMode switch
        {
            TritScaleMode.Global => Planes,
            TritScaleMode.PerRow => Rows * Planes,
            TritScaleMode.PerGroup => Rows * GroupsPerRow * Planes,
            _ => Planes
        };
    }

    //====================================================================
    // Dot products
    //====================================================================

    /// <summary>
    /// Быстрый dot-product одной плоскости одной строки с другой плоской другой строки.
    /// </summary>
    public int DotRowPlane(
        int rowA,
        int planeA,
        MultiPlaneTritPacker other,
        int rowB,
        int planeB)
    {
        if (Cols != other.Cols)
            throw new ArgumentException("Cols != other.Cols");

        int words = WordsNeeded(Cols);

        int baseA = rowA * WordsPerRow;
        int baseB = rowB * other.WordsPerRow;

        var aP = P[planeA];
        var aN = N[planeA];

        var bP = other.P[planeB];
        var bN = other.N[planeB];

        int plus = 0;
        int minus = 0;

        for (int w = 0; w < words; w++)
        {
            ulong same = (aP[baseA + w] & bP[baseB + w]) |
                         (aN[baseA + w] & bN[baseB + w]);

            ulong diff = (aP[baseA + w] & bN[baseB + w]) |
                         (aN[baseA + w] & bP[baseB + w]);

            plus += BitOperations.PopCount(same);
            minus += BitOperations.PopCount(diff);
        }

        return plus - minus;
    }

    /// <summary>
    /// Dot-product участка строки для одной плоскости.
    /// </summary>
    public int DotRowPlaneSegment(
        int rowA,
        int planeA,
        MultiPlaneTritPacker other,
        int rowB,
        int planeB,
        int colStart,
        int length)
    {
        if (length <= 0)
            return 0;

        if (colStart == 0 && length == Cols && Cols == other.Cols)
        {
            return DotRowPlane(rowA, planeA, other, rowB, planeB);
        }

        int sum = 0;

        for (int c = 0; c < length; c++)
        {
            int col = colStart + c;

            sbyte a = Get(planeA, rowA, col);
            sbyte b = other.Get(planeB, rowB, col);

            sum += a * b;
        }

        return sum;
    }

    /// <summary>
    /// Dot-product строки весов с FP32-вектором.
    ///
    /// Используется для linear-слоев:
    ///
    ///     y[o] = W[o, :] * x
    /// </summary>
    public float DotRowFloat(
        int row,
        ReadOnlySpan<float> x,
        int colStart = 0,
        int length = -1)
    {
        if (length < 0)
            length = Cols - colStart;

        if (length <= 0)
            return 0f;

        if (x.Length < colStart + length)
            throw new ArgumentException("x too small");

        if (colStart < 0 || colStart + length > Cols)
            throw new ArgumentOutOfRangeException(nameof(length));

        // Быстрый путь: вся строка, без per-group scales
        if (colStart == 0 &&
            length == Cols &&
            ScaleMode != TritScaleMode.PerGroup)
        {
            float sum = 0f;

            for (int plane = 0; plane < Planes; plane++)
            {
                float scale = GetScale(row, 0, plane);

                if (scale == 0f)
                    continue;

                float acc = 0f;

                for (int col = 0; col < Cols; col++)
                {
                    sbyte t = Get(plane, row, col);

                    if (t != 0)
                        acc += t * x[col];
                }

                sum += scale * acc;
            }

            return sum;
        }

        // Общий путь: участки и/или per-group scales
        float total = 0f;

        int offset = 0;

        while (offset < length)
        {
            int col = colStart + offset;

            int group = GroupSize <= 0 ? 0 : col / GroupSize;

            int chunk = GroupSize <= 0
                ? length - offset
                : GroupSize - (col % GroupSize);

            chunk = Math.Min(length - offset, chunk);

            for (int plane = 0; plane < Planes; plane++)
            {
                float scale = GetScale(row, group, plane);

                if (scale == 0f)
                    continue;

                float acc = 0f;

                for (int c = 0; c < chunk; c++)
                {
                    sbyte t = Get(plane, row, col + c);

                    if (t != 0)
                        acc += t * x[col + c];
                }

                total += scale * acc;
            }

            offset += chunk;
        }

        return total;
    }

    //====================================================================
    // Reconstruction
    //====================================================================

    /// <summary>
    /// Восстанавливает FP32-представление.
    /// Нужно в основном для отладки и тестов.
    /// </summary>
    public float[] ToFloat()
    {
        var result = new float[Length];

        for (int row = 0; row < Rows; row++)
        {
            for (int col = 0; col < Cols; col++)
            {
                int group = GroupSize <= 0 ? 0 : col / GroupSize;

                float value = 0f;

                for (int plane = 0; plane < Planes; plane++)
                {
                    float scale = GetScale(row, group, plane);

                    if (scale == 0f)
                        continue;

                    sbyte t = Get(plane, row, col);
                    value += scale * t;
                }

                result[row * Cols + col] = value;
            }
        }

        return result;
    }

    //====================================================================
    // Helpers
    //====================================================================

    private static float MeanAbs(float[] data, int start, int length)
    {
        if (length <= 0) return 0f;
        float sum = 0f;
        for (int i = 0; i < length; i++)
            sum += Math.Abs(data[start + i]);
        return sum / length; // делим на ВСЕ элементы, включая нули
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static sbyte TernaryValue(float value, float threshold)
    {
        if (value > threshold)
            return 1;

        if (value < -threshold)
            return -1;

        return 0;
    }

    public void SaveToStream(BinaryWriter w)
    {
        // Новый формат packer'а:
        // FP16 scales + base-3 packed trits
        w.Write("MPTP_V2");

        w.Write(Rows);
        w.Write(Cols);
        w.Write(Planes);
        w.Write(GroupSize);
        w.Write((int)ScaleMode);

        //------------------------------------------------------------
        // 1. Scales как FP16
        //------------------------------------------------------------
        w.Write(Scales.Length);

        for (int i = 0; i < Scales.Length; i++)
        {
            Half halfScale = (Half)Scales[i];
            ushort bits = BitConverter.HalfToUInt16Bits(halfScale);
            w.Write(bits);
        }

        //------------------------------------------------------------
        // 2. Trits в base-3 packing
        //    5 тритов -> 1 байт
        //------------------------------------------------------------
        int totalTrits = Rows * Cols;
        int expectedChunks = (totalTrits + 4) / 5;

        for (int p = 0; p < Planes; p++)
        {
            byte[] packedPlane = EncodePlaneBase3(p);

            if (packedPlane.Length != expectedChunks)
            {
                throw new InvalidDataException(
                    $"Base3 packing error: plane={p}, " +
                    $"expected {expectedChunks} chunks, got {packedPlane.Length}");
            }

            w.Write(packedPlane.Length);
            w.Write(packedPlane);
        }
    }

    public void LoadFromStream(BinaryReader r)
    {
        string marker = r.ReadString();

        if (marker != "MPTP_V2")
        {
            throw new InvalidDataException(
                $"Неверный формат MultiPlaneTritPacker: '{marker}'. Ожидалось MPTP_V2.");
        }

        Rows = r.ReadInt32();
        Cols = r.ReadInt32();
        Planes = r.ReadInt32();
        GroupSize = r.ReadInt32();
        ScaleMode = (TritScaleMode)r.ReadInt32();

        //------------------------------------------------------------
        // 1. Scales FP16
        //------------------------------------------------------------
        int scaleCount = r.ReadInt32();

        int expectedScaleCount = GetScaleCount();

        if (scaleCount != expectedScaleCount)
        {
            throw new InvalidDataException(
                $"Неверное количество scales: ожидалось {expectedScaleCount}, получено {scaleCount}.");
        }

        Scales = new float[scaleCount];

        for (int i = 0; i < scaleCount; i++)
        {
            ushort bits = r.ReadUInt16();
            Half halfScale = BitConverter.UInt16BitsToHalf(bits);
            Scales[i] = (float)halfScale;
        }

        //------------------------------------------------------------
        // 2. Trits base-3 packing
        //------------------------------------------------------------
        int totalTrits = Rows * Cols;
        int expectedChunks = (totalTrits + 4) / 5;

        int totalWords = Rows * WordsPerRow;

        P = new ulong[Planes][];
        N = new ulong[Planes][];

        for (int p = 0; p < Planes; p++)
        {
            P[p] = new ulong[totalWords];
            N[p] = new ulong[totalWords];
        }

        for (int p = 0; p < Planes; p++)
        {
            int len = r.ReadInt32();

            if (len != expectedChunks)
            {
                throw new InvalidDataException(
                    $"Неверный размер base3-данных для plane={p}: " +
                    $"ожидалось {expectedChunks}, получено {len}.");
            }

            byte[] packedPlane = r.ReadBytes(len);

            if (packedPlane.Length != len)
            {
                throw new InvalidDataException(
                    $"Неожиданный конец файла при чтении base3 plane={p}.");
            }

            DecodePlaneBase3(p, packedPlane);
        }
    }

    //====================================================================
    // BASE-3 TRIT PACKING
    //====================================================================

    private byte[] EncodePlaneBase3(int plane)
    {
        int total = Rows * Cols;
        int chunks = (total + 4) / 5;

        var bytes = new byte[chunks];

        int index = 0;

        for (int ch = 0; ch < chunks; ch++)
        {
            int value = 0;
            int multiplier = 1;

            for (int k = 0; k < 5 && index < total; k++, index++)
            {
                int row = index / Cols;
                int col = index % Cols;

                sbyte t = Get(plane, row, col);

                // trit: -1, 0, +1
                // digit:  0, 1, 2
                int digit;

                if (t == -1)
                    digit = 0;
                else if (t == 1)
                    digit = 2;
                else
                    digit = 1;

                value += digit * multiplier;
                multiplier *= 3;
            }

            bytes[ch] = (byte)value;
        }

        return bytes;
    }

    private void DecodePlaneBase3(int plane, byte[] bytes)
    {
        int total = Rows * Cols;
        int index = 0;

        // Очищаем текущие P/N маски
        ClearPlane(plane);

        foreach (byte b in bytes)
        {
            int value = b;

            for (int k = 0; k < 5 && index < total; k++, index++)
            {
                int digit = value % 3;
                value /= 3;

                // digit: 0, 1, 2
                // trit: -1, 0, +1
                sbyte t = (sbyte)(digit - 1);

                int row = index / Cols;
                int col = index % Cols;

                Set(plane, row, col, t);
            }
        }
    }
}