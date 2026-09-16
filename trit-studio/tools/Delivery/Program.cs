using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

return await Delivery.Run(args);
internal static class Delivery
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
    private static readonly CancellationTokenSource Stop = new();
    private static string _root = "", _logs = "";
    private static int _commandNumber;
    public static async Task<int> Run(string[] args)
    {
        string? staging = null;
        FileStream? deliveryLease = null;
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; Stop.Cancel(); };
        try
        {
            _root = Directory.GetCurrentDirectory();
            if (!File.Exists(Path.Combine(_root, "TritStudio.sln"))) throw new Exception("Run from the Trit Studio source folder (contains TritStudio.sln).");
            string rid = "win-x64";
            bool cuda = !args.Contains("--cpu-only");
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--target" && i + 1 < args.Length) rid = args[++i];
                else if (args[i] is "--cpu-only" or "--with-cuda") { }
                else throw new ArgumentException("Usage: --target win-x64|linux-x64 [--with-cuda|--cpu-only]");
            }
            if (args.Contains("--cpu-only") && args.Contains("--with-cuda")) throw new ArgumentException("Choose one backend packaging mode.");
            if (rid is not ("win-x64" or "linux-x64")) throw new ArgumentException("Unsupported package runtime.");
            string artifacts = Path.Combine(_root, "artifacts"); Directory.CreateDirectory(artifacts);
            deliveryLease = new FileStream(Path.Combine(artifacts, ".delivery.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            await File.WriteAllTextAsync(Path.Combine(artifacts, "WHERE_TO_PICK_UP.txt"), "BUILD IN PROGRESS. A previous archive, if present, is not the result of this run.\n", Encoding.UTF8);
            await File.WriteAllTextAsync(Path.Combine(artifacts, "DELIVERY_RESULT.json"), JsonSerializer.Serialize(new { status = "building", startedAtUtc = DateTimeOffset.UtcNow }, Json), Encoding.UTF8);
            string runId = DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N")[..6];
            _logs = Path.Combine(artifacts, "build-logs", runId); Directory.CreateDirectory(_logs);
            // All host acceptance runs use CPU. Cross-publishing is never misreported as target execution.
            await Dotnet("--info");
            await Dotnet("restore", "TritStudio.sln", "-p:TorchBackend=cpu");
            await Dotnet("build", "TritStudio.sln", "-c", "Release", "--no-restore", "-p:TorchBackend=cpu");
            await Dotnet("run", "--project", "tests/TritStudio.Tests", "-c", "Release", "--no-build");
            await Dotnet("run", "--project", "tests/TritStudio.UiTests", "-c", "Release", "--no-build");
            await Dotnet("run", "--project", "src/TritStudio.Trainer", "-c", "Release", "--no-build", "--", "--self-test");
            await Dotnet("run", "--project", "src/TritStudio.Trainer", "-c", "Release", "--no-build", "--", "--benchmark", "--output", Path.Combine(_logs,"host-cpu-benchmark.json"));
            string hostExe = Path.Combine(_root, "src", "TritStudio.Trainer", "bin", "Release", "net10.0", "TritStudio.Trainer" + (OperatingSystem.IsWindows() ? ".exe" : ""));
            if (!File.Exists(hostExe)) throw new FileNotFoundException("Host trainer was not built.", hostExe);
            await Dotnet("run", "--project", "tests/TritStudio.Tests", "-c", "Release", "--no-build", "--", "--worker", hostExe);
            string name = "TritStudio-" + rid + "-portable";
            staging = Path.Combine(artifacts, ".stage-" + runId); Directory.CreateDirectory(staging);
            string folder = Path.Combine(staging, name); Directory.CreateDirectory(folder);
            await Publish("src/TritStudio.App", rid, folder);
            await Publish("src/TritStudio.Trainer", rid, Path.Combine(folder, "trainer"), "cpu");
            if (cuda) await Publish("src/TritStudio.Trainer", rid, Path.Combine(folder, "trainer-cuda"), rid == "win-x64" ? "cuda-windows" : "cuda-linux");
            await Publish("src/TritStudio.Runner", rid, Path.Combine(folder, "runner"));
            await Publish("tests/TritStudio.Tests", rid, Path.Combine(folder, "checks"));
            CopyTree(Path.Combine(_root, "packaging"), folder);
            CopyTree(Path.Combine(_root, "data"), Path.Combine(folder, "datasets"));
            CopyTree(Path.Combine(_root, "docs"), Path.Combine(folder, "docs"));
            CopyTree(Path.Combine(_root, "third_party"), Path.Combine(folder, "third_party"));
            foreach (string f in new[] { "README_RU.md", "LICENSE", "THIRD_PARTY_NOTICES.md" }) File.Copy(Path.Combine(_root, f), Path.Combine(folder, f));
            string version = File.ReadAllText(Path.Combine(_root, "VERSION")).Trim();
            await File.WriteAllTextAsync(Path.Combine(folder, "BUILD_REPORT.json"), JsonSerializer.Serialize(new
            {
                version, builtAtUtc = DateTimeOffset.UtcNow, target = rid, cudaIncluded = cuda,
                host = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                verified = new[] { "host managed core + corpus tests", "host headless UI", "host CPU gradients + optimizer resume", "actual host worker lifecycle", "isolated host CPU benchmark" },
                targetNativeUi = "not run by cross-publisher; execute laptop checklist",
                targetCuda = "not run by cross-publisher; run Check-laptop.cmd", buildLogs = _logs,
                dataset = JsonDocument.Parse(File.ReadAllText(Path.Combine(_root, "data", "DATASET_MANIFEST.json"))).RootElement.Clone()
            }, Json), Encoding.UTF8);
            var files = Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories).OrderBy(x => x, StringComparer.Ordinal)
                .ToDictionary(f => Path.GetRelativePath(folder, f).Replace('\\', '/'), Hash, StringComparer.Ordinal);
            await File.WriteAllTextAsync(Path.Combine(folder, "SHA256SUMS.json"), JsonSerializer.Serialize(files, Json), Encoding.UTF8);
            string stagedZip = Path.Combine(staging, name + ".zip");
            Console.WriteLine("[package] Creating portable ZIP (includes native libraries). No user workspace is included.");
            ZipFile.CreateFromDirectory(folder, stagedZip, CompressionLevel.Fastest, includeBaseDirectory: true);
            Stop.Token.ThrowIfCancellationRequested();
            string zipHash = Hash(stagedZip);
            string finalFolder = Path.Combine(artifacts, name), finalZip = Path.Combine(artifacts, name + ".zip");
            string previousFolder = finalFolder + ".previous-" + runId;
            if (Directory.Exists(finalFolder)) Directory.Move(finalFolder, previousFolder);
            try { Directory.Move(folder, finalFolder); }
            catch { if (Directory.Exists(previousFolder) && !Directory.Exists(finalFolder)) Directory.Move(previousFolder, finalFolder); throw; }
            File.Move(stagedZip, finalZip, overwrite: true);
            await File.WriteAllTextAsync(finalZip + ".sha256", zipHash + "  " + Path.GetFileName(finalZip) + "\n", Encoding.UTF8);
            string receipt = "BUILD SUCCEEDED\n" +
                "Portable archive: " + finalZip + "\n" +
                "Unpacked folder: " + finalFolder + "\n" +
                "SHA256: " + zipHash + "\n" +
                "Build logs: " + _logs + "\n" +
                "Target desktop and GPU remain to be tested on the laptop. Run Check-laptop.cmd on Windows.\n";
            await File.WriteAllTextAsync(Path.Combine(artifacts, "WHERE_TO_PICK_UP.txt"), receipt, Encoding.UTF8);
            await File.WriteAllTextAsync(Path.Combine(artifacts, "DELIVERY_RESULT.json"), JsonSerializer.Serialize(new
            {
                status = "succeeded", version, target = rid, cudaIncluded = cuda, archive = finalZip, unpackedDirectory = finalFolder,
                sha256 = zipHash, logs = _logs, completedAtUtc = DateTimeOffset.UtcNow,
                targetExecution = "NOT_RUN_BY_PACKAGER", previousFolder = Directory.Exists(previousFolder) ? previousFolder : null
            }, Json), Encoding.UTF8);
            Console.WriteLine("\n" + receipt);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("DELIVERY FAILED. No new package is accepted.\n" + error);
            if (_logs.Length > 0)
            {
                await File.WriteAllTextAsync(Path.Combine(_logs, "FAILURE.txt"), error.ToString());
                string artifacts = Path.Combine(_root, "artifacts");
                await File.WriteAllTextAsync(Path.Combine(artifacts, "WHERE_TO_PICK_UP.txt"), "LATEST BUILD FAILED. Do not treat an older archive as the result.\nLogs: " + _logs + "\n" + error.Message, Encoding.UTF8);
                await File.WriteAllTextAsync(Path.Combine(artifacts, "DELIVERY_RESULT.json"), JsonSerializer.Serialize(new { status = "failed", logs = _logs, error = error.Message }, Json), Encoding.UTF8);
            }
            return 1;
        }
        finally
        {
            try { if (staging is not null && Directory.Exists(staging)) Directory.Delete(staging, true); }
            catch (Exception error) { Console.Error.WriteLine("Staging cleanup warning: " + error.Message); }
            deliveryLease?.Dispose();
        }
    }
    private static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private static async Task Publish(string project, string rid, string output, string? backend = null)
    {
        var args = new List<string> { "publish", project, "-c", "Release", "-r", rid, "--self-contained", "true", "-p:PublishTrimmed=false", "-p:PublishSingleFile=false", "-o", output };
        if (backend is not null) args.Add("-p:TorchBackend=" + backend);
        await Dotnet(args.ToArray());
    }
    private static void CopyTree(string source, string target)
    {
        Directory.CreateDirectory(target);
        foreach (string sourceFile in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
        {
            string output = Path.Combine(target, Path.GetRelativePath(source, sourceFile)); Directory.CreateDirectory(Path.GetDirectoryName(output)!); File.Copy(sourceFile, output, true);
        }
    }
    private static async Task Dotnet(params string[] args)
    {
        Stop.Token.ThrowIfCancellationRequested();
        string command = "dotnet " + string.Join(' ', args.Select(a => a.Contains(' ') ? "\"" + a + "\"" : a));
        Console.WriteLine("\n[" + (++_commandNumber) + "] " + command);
        string logPath = Path.Combine(_logs, $"{_commandNumber:D2}.log");
        using var log = new StreamWriter(logPath, false, new UTF8Encoding(false)) { AutoFlush = true }; var gate = new object();
        log.WriteLine(command);
        var start = new ProcessStartInfo("dotnet") { WorkingDirectory = _root, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (string arg in args) start.ArgumentList.Add(arg);
        using var child = Process.Start(start) ?? throw new IOException("Could not start dotnet.");
        async Task Drain(StreamReader reader)
        {
            while (await reader.ReadLineAsync() is string line) { lock (gate) { log.WriteLine(line); Console.WriteLine(line); } }
        }
        Task stdout = Drain(child.StandardOutput), stderr = Drain(child.StandardError);
        try { await child.WaitForExitAsync(Stop.Token); await Task.WhenAll(stdout, stderr); }
        catch { if (!child.HasExited) child.Kill(entireProcessTree: true); await child.WaitForExitAsync(); await Task.WhenAll(stdout, stderr); throw; }
        log.WriteLine("EXIT=" + child.ExitCode);
        if (child.ExitCode != 0) throw new Exception(command + " failed with exit " + child.ExitCode + ". See " + logPath);
    }
}
