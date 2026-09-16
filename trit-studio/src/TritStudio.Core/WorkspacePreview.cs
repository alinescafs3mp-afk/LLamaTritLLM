namespace TritStudio.Core;

// Read-only phase of an open operation. The UI must hold the destination .ui.lock while using it.
// A bad destination must be rejected BEFORE the live model/client/history is disconnected.
public sealed record WorkspacePreview(string Root, string ConversationId, bool NeedsConversationState,
    WeightSet? Weights, RevisionInfo? Revision, string? InferenceFile, ChatReadResult History)
{
    public static WorkspacePreview Load(string root, int threads = 1, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); root = Path.GetFullPath(root);
        if (!Directory.Exists(root)) throw new DirectoryNotFoundException(root);
        ModelLibrary.NoLinks(root);
        string chatState = Path.Combine(root, "chat-state.json"); ModelLibrary.NoLinks(chatState);
        if (Directory.Exists(chatState)) throw new InvalidDataException("chat-state.json является каталогом.");
        bool exists = File.Exists(chatState);
        string conversation = exists ? JsonData.Read<string>(chatState, 4096) : Guid.NewGuid().ToString("N");
        if (string.IsNullOrWhiteSpace(conversation) || conversation.Length > 256)
            throw new InvalidDataException("Некорректный идентификатор разговора в chat-state.json. Текущая модель не переключена.");
        _ = ByteTokenizer.TokenCount(conversation);
        ct.ThrowIfCancellationRequested();
        string? active = ModelFiles.ActivePath(root);
        WeightSet? weights = null; RevisionInfo? info = null; string? file = null;
        if (active is not null)
        {
            (weights, info) = ModelFiles.ReadInferenceRevision(active, threads, ct);
            file = Path.Combine(active, "model.tritmodel");
        }
        var history = ChatJournal.ReadTail(Path.Combine(root, "chat.jsonl"), ct: ct);
        ct.ThrowIfCancellationRequested();
        return new(root, conversation, !exists, weights, info, file, history);
    }
}
