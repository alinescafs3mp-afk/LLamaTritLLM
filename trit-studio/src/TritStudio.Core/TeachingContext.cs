namespace TritStudio.Core;

public static class TeachingContext
{
    // Keep at most the history actually seen during generation, never nested snapshots.
    // Excluded exchanges act as a boundary: do not splice unrelated earlier answers across them.
    public static ChatTurn[] Capture(IReadOnlyList<ChatTurn> history, int retainedTurns, string conversationId)
    {
        if (retainedTurns < 0 || retainedTurns > history.Count) throw new ArgumentOutOfRangeException(nameof(retainedTurns));
        var result = new List<ChatTurn>();
        for (int i = history.Count - retainedTurns; i < history.Count; i++)
        {
            var turn = history[i];
            if (turn.ExcludedFromTraining || turn.ConversationId != conversationId) { result.Clear(); continue; }
            result.Add(turn with { LearningHistory = null });
        }
        return result.TakeLast(8).ToArray();
    }
}
