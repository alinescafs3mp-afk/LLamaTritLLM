using TritStudio.Core;
namespace TritStudio.App;
public sealed record AppPreferences(string? LastWorkspace = null, bool OnlineLearning = false,
    SamplingOptions? Sampling = null, double UiScale = 0.8, string[]? RecentModels = null, bool ChatSettingsVisible = false);
public static class AppPaths
{
    public static string Root => Environment.GetEnvironmentVariable("TRITSTUDIO_HOME") is { Length: > 0 } path
        ? Path.GetFullPath(path) : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "TritStudio");
    public static string Models => Path.Combine(Root, "models");
    public static string Trash => Path.Combine(Root, "model-trash");
    public static string Preferences => Path.Combine(Root, "preferences.json");
    public static AppPreferences Load()
    {
        try { return File.Exists(Preferences) ? JsonData.Read<AppPreferences>(Preferences, 64 * 1024) : new(); }
        catch { return new(); }
    }
    public static string NewWorkspace(string name)
    {
        string safe = new(name.Trim().Where(c => char.IsLetterOrDigit(c) || c is '-' or '_').Take(50).ToArray());
        if (safe.Length == 0) safe = "model";
        return Path.Combine(Models, safe + "-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N")[..6]);
    }
}
