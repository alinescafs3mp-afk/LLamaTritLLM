using System.Text;
namespace TritStudio.Core;

// Descriptive diagnostics only. Does not rewrite, stop, rerank or fabricate a model response.
public sealed record GenerationHealth(int UnicodeScalars, int LongestIdenticalRun, double DistinctFourGramFraction,
    bool Repetitive)
{
    public static GenerationHealth Inspect(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var scalars = text.EnumerateRunes().Select(x => x.Value).ToArray();
        int longest = 0, run = 0, previous = -1;
        foreach (int scalar in scalars) { run = scalar == previous ? run + 1 : 1; longest = Math.Max(longest, run); previous = scalar; }
        var grams = new HashSet<(int,int,int,int)>();
        for (int i = 0; i + 3 < scalars.Length; i++) grams.Add((scalars[i],scalars[i+1],scalars[i+2],scalars[i+3]));
        double distinct = scalars.Length < 4 ? 1 : (double)grams.Count / (scalars.Length - 3);
        return new(scalars.Length, longest, distinct, longest >= 12 || scalars.Length >= 48 && distinct < 0.15);
    }
}
