namespace TritStudio.Core;

public sealed record MaintenanceResult(int Kept, string[] Removed, long BytesFreed, bool Applied);
public static class WorkspaceMaintenance
{
    public static MaintenanceResult Prune(string workspace, int keep = 8, bool apply = false)
    {
        workspace = ExistingWorkspace(workspace);
        using var ui = new FileStream(Path.Combine(workspace, ".ui.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        return PruneWithUiLease(workspace, ui, keep, apply);
    }
    // The GUI keeps its UI lease, stops the trainer, invalidates/drains publication loads, then calls here.
    // Never release the UI lease just to run housekeeping: another application could claim the workspace.
    public static MaintenanceResult PruneWithUiLease(string workspace, FileStream uiLease, int keep = 8, bool apply = false)
    {
        workspace = ExistingWorkspace(workspace);
        var comparer = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (uiLease.SafeFileHandle.IsClosed || !uiLease.CanWrite ||
            !string.Equals(Path.GetFullPath(uiLease.Name), Path.Combine(workspace, ".ui.lock"), comparer))
            throw new InvalidOperationException("A live lease for this workspace is required.");
        using var trainer = new FileStream(Path.Combine(workspace, ".trainer.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        return PruneLocked(workspace, keep, apply);
    }
    private static string ExistingWorkspace(string workspace)
    {
        workspace = Path.GetFullPath(workspace);
        if (!Directory.Exists(workspace)) throw new DirectoryNotFoundException(workspace);
        return workspace;
    }
    private static MaintenanceResult PruneLocked(string workspace, int keep, bool apply)
    {
        if (keep is < 2 or > 128) throw new ArgumentOutOfRangeException(nameof(keep));
        string active = ModelFiles.ActivePath(workspace) ?? throw new InvalidDataException("Workspace has no active revision.");
        var info = ModelFiles.VerifyRevision(active);
        var revisions = Directory.GetDirectories(Path.Combine(workspace, "revisions"), "r*")
            .Where(p => Path.GetFileName(p).Length == 17 && Path.GetFileName(p)[1..].All(char.IsAsciiDigit))
            .OrderByDescending(p => Path.GetFileName(p), StringComparer.Ordinal).ToArray();
        // Preserve an actual rollback chain, not only the newest branch and one parent.
        var preserve = new HashSet<string>(StringComparer.Ordinal) { active };
        var next = info;
        while (preserve.Count < keep && next.Parent is long parent)
        {
            string path = ModelFiles.GetRevisionPath(workspace, $"r{parent:D16}");
            if (!Directory.Exists(path) || !preserve.Add(path)) break;
            next = ModelFiles.VerifyRevision(path);
        }
        foreach (string path in revisions) { if (preserve.Count >= keep) break; preserve.Add(path); }
        var remove = revisions.Where(p => !preserve.Contains(p)).ToArray(); long bytes = 0;
        foreach (string path in remove)
        {
            if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Refusing to prune a linked revision.");
            foreach (string entry in Directory.EnumerateFileSystemEntries(path))
                if ((File.GetAttributes(entry) & FileAttributes.ReparsePoint) != 0 || Directory.Exists(entry))
                    throw new IOException("Unexpected nested or linked checkpoint entry.");
            bytes += Directory.EnumerateFiles(path).Sum(p => new FileInfo(p).Length);
        }
        if (apply)
        {
            RevisionSequence.PreserveHighWater(workspace);
            foreach (var path in remove) Directory.Delete(path, recursive: true);
        }
        return new(revisions.Length - remove.Length, remove.Select(Path.GetFileName).Cast<string>().ToArray(), bytes, apply);
    }
}
