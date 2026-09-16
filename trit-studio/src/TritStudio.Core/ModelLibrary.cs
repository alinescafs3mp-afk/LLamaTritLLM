namespace TritStudio.Core;

public sealed record ModelEntry(string Path, string Name, bool Workspace, bool Managed, string? Problem = null)
{
    public override string ToString() => Name + (Workspace ? "" : " · только чат") + (Problem is null ? "" : " · проверить");
}
public sealed record ModelTrashReceipt(string OriginalPath, string ModelName, DateTimeOffset RemovedAt, string ModelDirectory = "model");
public sealed record ModelTrashResult(string Directory, string OriginalPath);

// A local catalog, not a recursive scan of disks. Listing never loads weights or starts training.
public static class ModelLibrary
{
    public static StringComparison PathComparison => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    public static bool SamePath(string? a, string? b) => a is not null && b is not null &&
        string.Equals(Path.TrimEndingDirectorySeparator(System.IO.Path.GetFullPath(a)), Path.TrimEndingDirectorySeparator(System.IO.Path.GetFullPath(b)), PathComparison);
    public static bool IsManaged(string modelsRoot, string path) =>
        SamePath(System.IO.Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(System.IO.Path.GetFullPath(path))), modelsRoot);
    public static void NoLinks(string path)
    {
        string full = Path.GetFullPath(path);
        for (string? p = full; p is not null; p = Path.GetDirectoryName(p))
        {
            // File.GetAttributes also observes link entries; absent components are permitted for a new destination.
            try { if ((File.GetAttributes(p) & FileAttributes.ReparsePoint) != 0) throw new IOException("Ссылки в пути модели не поддерживаются: " + p); }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
        }
    }
    public static ModelEntry[] Scan(string modelsRoot, IEnumerable<string>? remembered = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); modelsRoot = Path.GetFullPath(modelsRoot); NoLinks(modelsRoot);
        var paths = new HashSet<string>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        if (Directory.Exists(modelsRoot))
        {
            foreach (var p in Directory.EnumerateDirectories(modelsRoot))
            {
                ct.ThrowIfCancellationRequested(); if (paths.Count >= 2000) throw new IOException("В каталоге более 2000 моделей. Разделите каталог; частичный список не выдан за полный.");
                paths.Add(Path.GetFullPath(p));
            }
        }
        foreach (var p in (remembered ?? []).Take(256))
        {
            ct.ThrowIfCancellationRequested(); if (!string.IsNullOrWhiteSpace(p)) paths.Add(Path.GetFullPath(p));
        }
        var rows = new List<ModelEntry>();
        foreach (string path in paths)
        {
            ct.ThrowIfCancellationRequested();
            bool workspace = Directory.Exists(path), file = File.Exists(path);
            if (!workspace && !file) continue;
            bool managed = workspace && IsManaged(modelsRoot, path);
            string? problem = null;
            try
            {
                NoLinks(path);
                if (workspace && !File.Exists(Path.Combine(path, "active.json"))) problem = "Нет активного снимка; можно удалить незавершённую модель";
                if (file && !path.EndsWith(".tritmodel", StringComparison.OrdinalIgnoreCase)) continue;
            }
            catch (IOException e) { problem = e.Message; managed = false; }
            rows.Add(new(path, Path.GetFileName(Path.TrimEndingDirectorySeparator(path)), workspace, managed, problem));
        }
        return rows.OrderBy(x => x.Name, StringComparer.CurrentCultureIgnoreCase).ThenBy(x => x.Path, StringComparer.Ordinal).ToArray();
    }
    public static string ValidateRemoval(string modelsRoot, string path, string trashRoot)
    {
        path = Path.GetFullPath(path); modelsRoot = Path.GetFullPath(modelsRoot); trashRoot = Path.GetFullPath(trashRoot);
        if (!IsManaged(modelsRoot, path) || SamePath(modelsRoot, path)) throw new IOException("Удалять можно только отдельную модель внутри каталога приложения. Внешний путь можно убрать из списка.");
        NoLinks(modelsRoot); NoLinks(path); NoLinks(trashRoot);
        if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);
        if (File.Exists(trashRoot)) throw new IOException("Путь локальной корзины занят файлом. Модель не отключена и не перемещена.");
        NoLinks(Path.Combine(path, ".ui.lock")); NoLinks(Path.Combine(path, ".trainer.lock"));
        if (SamePath(trashRoot, path) || trashRoot.StartsWith(path + Path.DirectorySeparatorChar, PathComparison) || SamePath(trashRoot, modelsRoot) || trashRoot.StartsWith(modelsRoot + Path.DirectorySeparatorChar, PathComparison))
            throw new IOException("Корзина не должна находиться внутри удаляемой модели или внутри общего каталога моделей.");
        return path;
    }
    public static ModelTrashResult MoveToTrash(string modelsRoot, string path, string trashRoot, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); path = ValidateRemoval(modelsRoot, path, trashRoot);
        // Deny competing readers/writers, allow rename while the two lock handles remain open on Windows.
        // The existing v14 FileShare.None owner is incompatible and correctly blocks removal.
        NoLinks(Path.Combine(path, ".ui.lock")); NoLinks(Path.Combine(path, ".trainer.lock"));
        using var ui = new FileStream(Path.Combine(path, ".ui.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Delete);
        using var trainer = new FileStream(Path.Combine(path, ".trainer.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Delete);
        ValidateRemoval(modelsRoot, path, trashRoot); ct.ThrowIfCancellationRequested();
        Directory.CreateDirectory(trashRoot); NoLinks(trashRoot);
        string bin = Path.Combine(trashRoot, DateTime.UtcNow.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(bin); bool moved = false;
        try
        {
            JsonData.AtomicWrite(Path.Combine(bin, "restore.json"), new ModelTrashReceipt(path, Path.GetFileName(path), DateTimeOffset.UtcNow), 16 * 1024);
            ct.ThrowIfCancellationRequested();
            Directory.Move(path, Path.Combine(bin, "model")); moved = true;
            // No cancellation after rename: the removal has committed and its receipt must remain truthful.
            return new(bin, path);
        }
        finally { if (!moved && Directory.Exists(bin)) Directory.Delete(bin, true); }
    }
}
