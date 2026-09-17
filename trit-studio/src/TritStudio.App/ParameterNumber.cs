using System.Globalization;
using Avalonia.Controls;
using Avalonia.Data.Converters;
using Avalonia.Interactivity;
using TritStudio.Core;
namespace TritStudio.App;

// Preserve invalid user text instead of silently falling back to the previous valid/default value on blur.
public sealed class ParameterNumber : NumericUpDown
{
    protected override Type StyleKeyOverride => typeof(NumericUpDown);
    public bool IntegerOnly { get; init; }
    public ParameterNumber(string format, bool integer)
    { IntegerOnly = integer; TextConverter = new Converter(format, integer); }
    // Explicit profile/hydration must replace stale invalid text even when Value is already equal.
    public void SetNumber(decimal value)
    {
        Value = value;
        Text = value.ToString(FormatString, CultureInfo.CurrentCulture);
    }
    public decimal ReadChecked()
    {
        if (!IsInitialized && Text is null && Value is decimal initial) return initial;
        return ParameterNumberText.Parse(Text, Minimum, Maximum, IntegerOnly);
    }
    protected override void OnLostFocus(RoutedEventArgs e)
    {
        string? entered = Text; bool valid;
        try { _ = ReadChecked(); valid = true; } catch (ArgumentException) { valid = false; }
        base.OnLostFocus(e);
        if (!valid) SetCurrentValue(TextProperty, entered);
    }
    private sealed class Converter(string format, bool integer) : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
            string.IsNullOrEmpty(value as string) ? null : ParameterNumberText.Parse(value as string, decimal.MinValue, decimal.MaxValue, integer);
        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
            value is decimal number ? number.ToString(format, culture) : "";
    }
}
