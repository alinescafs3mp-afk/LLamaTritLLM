using System.Globalization;
namespace TritStudio.Core;

// No thousands separators: 0,0003 and 0.0003 mean the same number on either OS locale.
public static class ParameterNumberText
{
    public static decimal Parse(string? text, decimal minimum, decimal maximum, bool integer = false)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length > 96) throw new ArgumentException("Введите число.");
        text = text.Trim();
        if (text.Contains(',') && text.Contains('.')) throw new ArgumentException("Используйте одну десятичную запятую или точку, без разделителей тысяч.");
        text = text.Replace(',', '.');
        if (!decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal value))
            throw new ArgumentException("Неверное число. Например: 32, 0,0003 или 3e-4.");
        if (value < minimum || value > maximum) throw new ArgumentOutOfRangeException(nameof(text), $"Число должно быть от {minimum} до {maximum}.");
        if (integer && decimal.Truncate(value) != value) throw new ArgumentException("Здесь нужно целое число.");
        return value;
    }
}
