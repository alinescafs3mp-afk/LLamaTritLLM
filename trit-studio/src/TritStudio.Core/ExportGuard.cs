namespace TritStudio.Core;

public static class ExportGuard
{
    public static void Validate(string source, string destination, string? workspace)
    {
        string target = Path.GetFullPath(destination);
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (string.Equals(Path.GetFullPath(source), target, comparison)) throw new ArgumentException("Источник и назначение совпадают.");
        if (workspace is not null)
        {
            string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(workspace));
            if (string.Equals(root, target, comparison) || target.StartsWith(Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar, comparison))
                throw new ArgumentException("Экспортируйте за пределы рабочей папки: нельзя перезаписывать настройки, журнал или снимки модели.");
        }
        // A reparse-point destination can hide a path back into the workspace. Do not follow it for writes.
        for (string? at = target; at is not null; at = Path.GetDirectoryName(at))
            if ((File.Exists(at) || Directory.Exists(at)) && (File.GetAttributes(at) & FileAttributes.ReparsePoint) != 0)
                throw new ArgumentException("Путь экспорта проходит через ссылку. Выберите обычную папку.");
    }
}
