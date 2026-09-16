using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TritStudio.App;
using TritStudio.Core;

// Real redirected child processes. These fixtures do not load LibTorch or touch a user workspace.
internal static class ClientFaultChecks
{
    public static async Task<int> Child(string mode)
    {
        Console.WriteLine(JsonSerializer.Serialize(new WorkerEvent("ready",null,Protocol.Element(new ReadyEvent(false,null,null,null))),JsonData.Options));
        Console.Out.Flush();
        if (mode == "fault-not-reading") { await Task.Delay(TimeSpan.FromSeconds(30)); return 0; }
        string? request = await Console.In.ReadLineAsync();
        if (mode == "echo-mode")
        {
            while (request is not null)
            {
                var command=JsonSerializer.Deserialize<WorkerCommand>(request,JsonData.Options)!;
                if(command.Kind=="shutdown")return 0;
                await Task.Delay(150);
                Console.WriteLine(JsonSerializer.Serialize(new WorkerEvent("completed",command.Id,
                    Protocol.Element(new { command=command.Kind, success=true })),JsonData.Options));
                Console.Out.Flush(); request=await Console.In.ReadLineAsync();
            }
            return 0;
        }
        if (mode == "slow-valid-completion")
        {
            var command = JsonSerializer.Deserialize<WorkerCommand>(request!, JsonData.Options)!;
            await Task.Delay(TimeSpan.FromSeconds(2));
            Console.WriteLine(JsonSerializer.Serialize(new WorkerEvent("completed", command.Id,
                Protocol.Element(new { command = command.Kind, success = true })), JsonData.Options));
            Console.Out.Flush(); await Console.In.ReadLineAsync(); return 0;
        }
        if(mode=="fault-malformed") { Console.WriteLine("{ not JSON"); Console.Out.Flush(); }
        else if(mode=="fault-oversize") { Console.Write(new string('x',1_048_600)); Console.Out.Flush(); }
        else if(mode=="fault-completion-error")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine(JsonSerializer.Serialize(new {kind="completed",id=command.Id,data=new {success=false,error=new {code=123}}},JsonData.Options)); Console.Out.Flush();
        }
        else if(mode=="fault-completion-success")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine(JsonSerializer.Serialize(new {kind="completed",id=command.Id,data=new {success="yes"}},JsonData.Options)); Console.Out.Flush();
        }
        else if(mode=="fault-wrong-command")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine(JsonSerializer.Serialize(new {kind="completed",id=command.Id,data=new {command="another-command",success=true}},JsonData.Options));Console.Out.Flush();
        }
        else if(mode=="fault-missing-command")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine(JsonSerializer.Serialize(new {kind="completed",id=command.Id,data=new {success=true}},JsonData.Options));Console.Out.Flush();
        }
        else if(mode=="fault-duplicate-event")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine("{\"kind\":\"completed\",\"id\":\""+command.Id+"\",\"id\":\""+command.Id+"\",\"data\":{\"command\":\"fixture\",\"success\":true}}");Console.Out.Flush();
        }
        else if(mode=="fault-duplicate-terminal")
        {
            var command=JsonSerializer.Deserialize<WorkerCommand>(request!,JsonData.Options)!;
            Console.WriteLine("{\"kind\":\"completed\",\"id\":\""+command.Id+"\",\"data\":{\"command\":\"fixture\",\"success\":false,\"success\":true}}");Console.Out.Flush();
        }
        else if(mode=="fault-envelope") { Console.WriteLine("{}"); Console.Out.Flush(); }
        else if(mode=="fault-eof-alive")
        {
            Console.Out.Flush();
            // Console.Out.Dispose and close(1) each close one handle. .NET 10 keeps extra
            // dups of the redirected pipe; the parent only sees EOF when every write end is gone.
            if(OperatingSystem.IsWindows())
            { if(!CloseHandle(GetStdHandle(-11)))return 41; }
            else if(CloseStdoutPipeEnds()!=0)return 42;
        }
        else return 43;
        await Task.Delay(TimeSpan.FromSeconds(30)); // Parent must detect failure without waiting for this exit.
        return 0;
    }
    public static async Task Run(string home)
    {
        string executable=Path.Combine(AppContext.BaseDirectory,"TritStudio.UiTests"+(OperatingSystem.IsWindows()?".exe":""));
        if(!File.Exists(executable))throw new FileNotFoundException("UI fixture apphost missing",executable);
        foreach(string mode in new[]{"fault-malformed","fault-envelope","fault-eof-alive","fault-oversize","fault-completion-error","fault-completion-success","fault-wrong-command","fault-missing-command","fault-duplicate-event","fault-duplicate-terminal"})
        {
            var worker=new WorkerClient(Path.Combine(home,mode),executable);
            var terminated=new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
            worker.Terminated+=message=>terminated.TrySetResult(message);
            try
            {
                worker.Start();await worker.WaitReadyAsync().WaitAsync(TimeSpan.FromSeconds(10));
                bool failed=false;var watch=Stopwatch.StartNew();
                try { await worker.Request("fixture",new{}).WaitAsync(TimeSpan.FromSeconds(5)); }
                catch(Exception error) when(error is IOException or InvalidOperationException) { failed=true; }
                if(!failed)throw new Exception(mode+": pending command was not failed.");
                string detail=await terminated.Task.WaitAsync(TimeSpan.FromSeconds(2));
                if(mode=="fault-eof-alive" && !detail.Contains("процесс ещё не завершён"))throw new Exception("EOF fixture did not exercise a still-running child: "+detail);
                var child = (Process)typeof(WorkerClient).GetField("_process",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance)!.GetValue(worker)!;
                await child.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(4)); // No Dispose yet: transport failure itself must stop the orphan.
                if(watch.Elapsed>TimeSpan.FromSeconds(6))throw new Exception(mode+": waited for child exit instead of failing on pipe failure.");
            }
            finally { await worker.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(12)); }
            Console.WriteLine("PASS WorkerClient real child: "+mode);
        }
    }
    public static async Task CheckWriteDeadline(string home)
    {
        string executable = Path.Combine(AppContext.BaseDirectory, "TritStudio.UiTests" + (OperatingSystem.IsWindows() ? ".exe" : ""));
        foreach (bool blocked in new[] { true, false })
        {
            string mode = blocked ? "fault-not-reading" : "slow-valid-completion";
            await using var worker = new WorkerClient(Path.Combine(home, mode), executable, TimeSpan.FromMilliseconds(750));
            worker.Start(); await worker.WaitReadyAsync().WaitAsync(TimeSpan.FromSeconds(10));
            var watch = Stopwatch.StartNew();
            if (blocked)
            {
                bool failed = false;
                // Larger than OS pipe capacity, smaller than the protocol cap. Child never consumes stdin.
                try { await worker.Request("fixture", new { text = new string('x', 900000) }).WaitAsync(TimeSpan.FromSeconds(6)); }
                catch (IOException) { failed = true; }
                if (!failed) throw new Exception("A blocked write did not fail before the test deadline.");
                var child = (Process)typeof(WorkerClient).GetField("_process", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(worker)!;
                await child.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(4)); // Must die before normal Dispose.
            }
            else
            {
                await worker.Request("fixture", new { }).WaitAsync(TimeSpan.FromSeconds(8));
                if (watch.Elapsed < TimeSpan.FromSeconds(1)) throw new Exception("Delayed completion fixture did not exercise a long operation.");
            }
            Console.WriteLine("PASS actual command transport deadline: " + mode);
        }
    }

    [DllImport("libc",EntryPoint="close",SetLastError=true)] private static extern int CloseFileDescriptor(int fd);
    [DllImport("libc",EntryPoint="readlink",SetLastError=true)] private static extern int Readlink(string path, byte[] buf, int bufsiz);
    [DllImport("kernel32.dll",SetLastError=true)] private static extern IntPtr GetStdHandle(int id);
    [DllImport("kernel32.dll",SetLastError=true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CloseHandle(IntPtr handle);
    private static string? LinkTarget(string path)
    {
        var buf = new byte[256];
        int n = Readlink(path, buf, buf.Length);
        return n > 0 ? Encoding.UTF8.GetString(buf, 0, n) : null;
    }
    private static int CloseStdoutPipeEnds()
    {
        string? target = LinkTarget("/proc/self/fd/1");
        try { Console.Out.Dispose(); } catch { }
        int closed = 0;
        for (int fd = 1; fd < 1024; fd++)
        {
            if (fd == 2) continue;
            string? link = LinkTarget("/proc/self/fd/" + fd);
            if (link is null) continue;
            if (target is not null && link != target) continue;
            if (target is null && fd != 1) continue;
            if (CloseFileDescriptor(fd) == 0) closed++;
        }
        return closed > 0 ? 0 : -1;
    }
}
