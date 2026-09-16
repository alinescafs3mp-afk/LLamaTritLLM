using System.Diagnostics;
using System.Text.Json;
using TritStudio.Core;
namespace TritStudio.App;

public static class TrainerLocator
{
    private static string Executable => OperatingSystem.IsWindows() ? "TritStudio.Trainer.exe" : "TritStudio.Trainer";
    private static readonly Dictionary<string, bool> ProbeCache = new(StringComparer.Ordinal);
    public static async Task<(string Path, string Note)> FindAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string? explicitPath = Environment.GetEnvironmentVariable("TRITSTUDIO_TRAINER");
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            explicitPath = Path.GetFullPath(explicitPath);
            if (!File.Exists(explicitPath)) throw new FileNotFoundException("TRITSTUDIO_TRAINER указывает на отсутствующий файл.", explicitPath);
            return (explicitPath, "Используется явно указанный тренер.");
        }
        string cpu = Path.Combine(AppContext.BaseDirectory, "trainer", Executable);
        string gpu = Path.Combine(AppContext.BaseDirectory, "trainer-cuda", Executable);
        if (File.Exists(gpu))
        {
            bool valid;
            lock (ProbeCache) valid = ProbeCache.TryGetValue(gpu, out bool saved) && saved;
            if (!valid)
            {
                valid = await ProbeCuda(gpu, cancellationToken).ConfigureAwait(false);
                // A transient failure must not make Reconnect permanently CPU-only until app restart.
                // Cache successful probes only; a failed probe is retried on the next explicit open.
                if (valid) { lock (ProbeCache) ProbeCache[gpu] = true; }
            }
            if (valid) return (gpu, "CUDA проверена. Обучение выберет GPU, когда включён переключатель CUDA.");
        }
        cancellationToken.ThrowIfCancellationRequested();
        if (!File.Exists(cpu)) throw new FileNotFoundException("CPU-тренер не найден. Перенесите всю готовую папку TritStudio, не один EXE.", cpu);
        return (cpu, File.Exists(gpu) ? "CUDA не прошла проверку. Выбран CPU-тренер; подробности в диагностике запуска." : "Доступен CPU-тренер. CUDA-комплект не установлен.");
    }
    private static async Task<bool> ProbeCuda(string executable, CancellationToken cancellationToken)
    {
        using var process = new Process { StartInfo = new ProcessStartInfo(executable)
        { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true } };
        process.StartInfo.ArgumentList.Add("--doctor");
        Task<CapturedText>? stdout = null, stderr = null;
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(TimeSpan.FromSeconds(60));
        try
        {
            process.Start();
            stdout = BoundedTextCapture.ReadAsync(process.StandardOutput, 65536, deadline.Token);
            stderr = BoundedTextCapture.ReadAsync(process.StandardError, 16384, deadline.Token);
            await process.WaitForExitAsync(deadline.Token).ConfigureAwait(false);
            var output = await stdout.ConfigureAwait(false); var errors = await stderr.ConfigureAwait(false);
            bool valid = false;
            if (process.ExitCode == 0 && !output.Truncated)
                foreach (string line in output.Text.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                    try { using var doc = JsonDocument.Parse(line); valid |= doc.RootElement.TryGetProperty("cuda", out var c) && c.ValueKind == JsonValueKind.True; } catch (JsonException) { }
            StartupLog.Write($"CUDA probe exit={process.ExitCode}; ready={valid}; stdoutTruncated={output.Truncated}; stderrTruncated={errors.Truncated}; stdout={output.Text}; stderr={errors.Text}");
            return valid;
        }
        catch (Exception error)
        {
            try { if (!process.HasExited) { process.Kill(entireProcessTree: true); await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false); } } catch { }
            StartupLog.Write("CUDA probe failed: " + error); cancellationToken.ThrowIfCancellationRequested(); return false;
        }
        finally
        {
            deadline.Cancel();
            try { await Task.WhenAll(stdout ?? Task.FromResult(new CapturedText("", false)), stderr ?? Task.FromResult(new CapturedText("", false))).WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false); } catch { }
        }
    }
}
public static class StartupLog
{
    private static readonly object Gate = new();
    public static void Write(string message)
    {
        lock (Gate) try
        {
            Directory.CreateDirectory(AppPaths.Root);
            string path = Path.Combine(AppPaths.Root, "startup.log");
            if (File.Exists(path) && new FileInfo(path).Length > 1_048_576) File.Move(path, path + ".previous", true);
            File.AppendAllText(path, DateTimeOffset.Now.ToString("O") + " " + message + Environment.NewLine);
        }
        catch { /* Diagnostics must not prevent chat from opening. */ }
    }
}
