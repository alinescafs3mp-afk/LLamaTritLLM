namespace TritStudio.Core;

public sealed record DraftRestoration(string? Text, bool ExcludeNext, bool Restored);

// A retry of an excluded message must never silently become training material.
// A new draft (including deliberate whitespace) and its privacy choice belong to the user.
public static class DraftRecovery
{
    public static DraftRestoration Restore(string? currentDraft, bool excludeNext, string submitted, bool submittedExcluded)
    {
        ArgumentNullException.ThrowIfNull(submitted);
        return string.IsNullOrEmpty(currentDraft)
            ? new(submitted, excludeNext || submittedExcluded, true)
            : new(currentDraft, excludeNext, false);
    }
}
