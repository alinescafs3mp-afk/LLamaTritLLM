using System.Buffers;
using System.Text;
namespace TritStudio.Core;

// Fixed UTF-8 vocabulary avoids BPE startup cost and vocabulary mutation during continual learning.
public static class ByteTokenizer
{
    public const int Pad = 0, Bos = 1, Eos = 2, User = 3, Assistant = 4, System = 5;
    public const int Offset = 6, VocabularySize = 262;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    public static int TokenCount(string text) => StrictUtf8.GetByteCount(text);
    public static int[] Encode(string text)
    {
        var result = new int[TokenCount(text)]; EncodeInto(text, result); return result;
    }
    // Write directly into the final token buffer. Small pieces use stack memory; larger pieces use
    // a temporary cleared pool buffer. No intermediate int array or LINQ iterator per text piece.
    public static int EncodeInto(string text, Span<int> destination)
    {
        int count = TokenCount(text);
        if (destination.Length < count) throw new ArgumentException("Token destination is too small.", nameof(destination));
        if (count == 0) return 0;
        byte[]? rented = null;
        Span<byte> bytes = count <= 1024 ? stackalloc byte[count] : (rented = ArrayPool<byte>.Shared.Rent(count));
        try
        {
            int written = StrictUtf8.GetBytes(text.AsSpan(), bytes);
            for (int i = 0; i < written; i++) destination[i] = bytes[i] + Offset;
            return written;
        }
        finally
        {
            if (rented is not null) ArrayPool<byte>.Shared.Return(rented, clearArray: true);
        }
    }
    public static string Decode(IEnumerable<int> ids) => Encoding.UTF8.GetString(ids.Where(x => x >= Offset && x < VocabularySize).Select(x => (byte)(x - Offset)).ToArray());
    public static int[] Prompt(IReadOnlyList<ChatTurn> history, string user, int context, int reserve)
        => Plan(history, user, context, reserve).Tokens;
    public static PromptBudget Measure(IReadOnlyList<ChatTurn> history, string user, int context, int reserve)
    {
        if (context < 8 || reserve < 1 || reserve >= context) throw new ArgumentOutOfRangeException(nameof(reserve));
        int latest = TokenCount(user);
        if ((long)latest + 4 + reserve > context)
            throw new ArgumentException($"Message uses {latest} byte tokens. Shorten it or reduce the response limit / choose a larger-context model.");
        int retained = 0, used = latest + 4;
        for (int i = history.Count - 1; i >= 0; i--)
        {
            var turn = history[i]; int size = checked(TokenCount(turn.User) + TokenCount(turn.Assistant) + 4);
            if ((long)used + size + reserve > context) break;
            used += size; retained++;
        }
        return new(used, latest, retained, history.Count - retained, reserve, context);
    }
    public static PromptPlan Plan(IReadOnlyList<ChatTurn> history, string user, int context, int reserve)
    {
        var budget = Measure(history, user, context, reserve);
        var tokens = new int[budget.InputTokens]; int at = 0; tokens[at++] = Bos;
        void Text(string value) { at += EncodeInto(value, tokens.AsSpan(at)); }
        for (int i = history.Count - budget.RetainedTurns; i < history.Count; i++)
        {
            var turn = history[i];
            tokens[at++] = User; Text(turn.User); tokens[at++] = Eos;
            tokens[at++] = Assistant; Text(turn.Assistant); tokens[at++] = Eos;
        }
        tokens[at++] = User; Text(user); tokens[at++] = Eos; tokens[at++] = Assistant;
        if (at != tokens.Length) throw new InvalidOperationException("Prompt measurement/encoding mismatch.");
        return new(tokens, budget.UserTokens, budget.RetainedTurns, budget.DroppedTurns, reserve, context);
    }
}
public sealed record PromptPlan(int[] Tokens, int UserTokens, int RetainedTurns, int DroppedTurns, int ReservedOutputTokens, int Context);
public sealed record ChatTurn(string User, string Assistant, long Revision, bool ExcludedFromTraining = false, string ConversationId = "", ChatTurn[]? LearningHistory = null);
public sealed record TrainingExample(string Id, string Text, string? Answer = null, string Source = "dataset", ChatTurn[]? History = null)
{
    public bool IsDialogue => Answer != null;
}
public sealed record EncodedExample(int[] Tokens, int[] Labels, string Id);

public sealed record PromptBudget(int InputTokens, int UserTokens, int RetainedTurns, int DroppedTurns, int ReservedOutputTokens, int Context);
