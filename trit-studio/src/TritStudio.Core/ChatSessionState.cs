namespace TritStudio.Core;

// Resetting chat is NOT data erasure or unlearning. Commit state before clearing the caller's UI.
public static class ChatSessionState
{
    public static string StartNew(string? workspace, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        string id = Guid.NewGuid().ToString("N");
        if (workspace is not null)
        {
            string root = Path.GetFullPath(workspace); ModelLibrary.NoLinks(root);
            if (!Directory.Exists(root)) throw new DirectoryNotFoundException(root);
            string destination = Path.Combine(root, "chat-state.json"); ModelLibrary.NoLinks(destination);
            ct.ThrowIfCancellationRequested();
            JsonData.AtomicWrite(destination, id, 4096);
        }
        // After the atomic write commits, return success even if cancellation arrives just afterwards.
        return id;
    }
}
