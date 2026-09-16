namespace TritStudio.Core;

// A new context budget changes the effective control inputs even when the source JSON is identical.
public static class ValidationBaseline
{
    public static double Anchor(double current, RevisionInfo? previous, int sequenceLength)
    {
        if (!double.IsFinite(current) || current < 0 || sequenceLength is < 16 or > 2048)
            throw new ArgumentException("Invalid validation comparison.");
        // Legacy snapshots without this field cannot certify cross-run score comparability.
        double? best = previous?.ValidationSequenceLength == sequenceLength ? previous.BestValidationLoss : null;
        if (best is double saved && (!double.IsFinite(saved) || saved < 0))
            throw new InvalidDataException("Invalid historical validation score.");
        return Math.Min(current, best ?? current);
    }
}
