using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using TritStudio.Core;

public static class WorkerProtocolChecks
{
    public static async Task<int> Run(string executable)
    {
        string root = Path.Combine(Path.GetTempPath(), "trit-protocol-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            await using (var worker = new Child(executable, root))
            {
                await worker.Wait(e => e.Kind == "ready");
                var c = new ModelConfig { Dimension = 16, HiddenDimension = 32, Layers = 1, Heads = 2, KvHeads = 1, Context = 512, GroupSize = 8 };
                var resources = new ResourceOptions { Threads = 1, MemoryMiB = 4096, BatchSize = 2, SequenceLength = 384, PreferCuda = false };
                var options = new TrainingOptions { Steps = 2, PublishEvery = 2, LearningRate = 0.0001, OnlineLearningRate = 0.00005, MaxValidationRegression = 0.5 };
                string command = await worker.Send("create", new CreateRequest(c, resources, options, []));
                await worker.Complete(command);
                string initial = ModelFiles.ActivePath(root)!;
                var first = ModelFiles.VerifyRevision(initial);
                Assert(first.Step == 2 && first.CheckpointVersion == 2, "Creation did not publish trained coherent inputs.");
                Assert(File.Exists(Path.Combine(initial, "settings.json")), "Snapshot settings missing.");
                var heldOut = JsonData.Read<TrainingExample[]>(Path.Combine(initial,"validation.json")).First(e=>e.IsDialogue);
                var changedTarget = Dataset.Make(heldOut.Text,"Другой ответ",history:heldOut.History);
                await worker.Complete(await worker.Send("online",new OnlineRequest(changedTarget)),expectSuccess:false,allowErrors:true);
                Assert(new ReplayLedger(root).PendingCount==0,"Changed hold-out target bypassed split isolation.");
                Assert(ModelFiles.ActivePath(root)==initial,"Rejected held-out example changed the active revision.");
                var example = Dataset.Make("hello", "hi", "protocol-test");
                await worker.Complete(await worker.Send("mode", new OnlineMode(true, 0.00005)));
                await worker.Complete(await worker.Send("online", new OnlineRequest(example)));
                var learned = await worker.Wait(e => e.Kind == "online-result");
                Assert(learned.Data.GetProperty("accepted").GetBoolean(), "Online update rejected on the fixture.");
                var updated = ModelFiles.VerifyRevision(ModelFiles.ActivePath(root)!);
                Assert(updated.Step == 6 && updated.ChangedWeights > 0, "No real online parameter update.");
                await worker.Complete(await worker.Send("rollback", new { }));
                Assert(ModelFiles.ActivePath(root) == initial, "Rollback pointer mismatch.");
                var duplicate = await worker.Complete(await worker.Send("online", new OnlineRequest(example)));
                Assert(duplicate.Data.GetProperty("result").GetProperty("queued").GetInt32() == 0, "Rolled-back example was silently retrained.");
                // Enqueue work and cancel immediately: old work must not start after a stop fence.
                string longJob = await worker.Send("train", new TrainRequest([], options with { Steps = 100000, PublishEvery = 1000 }));
                string stop = await worker.Send("stop", new { });
                var stopped = await worker.Complete(longJob, expectSuccess: false, allowErrors: true);
                Assert(!stopped.Data.GetProperty("success").GetBoolean(), "Cancelled queued job was reported as successful.");
                Assert(stopped.Data.GetProperty("cancelled").GetBoolean(), "Intentional cancellation must be distinct from failure.");
                await worker.Complete(stop, allowErrors: true);
                await worker.Complete(await worker.Send("mode", new OnlineMode(false,0.000037)));
                await worker.Complete(await worker.Send("online",new OnlineRequest(Dataset.Make("pending example","pending answer"))));
                var cleared = await worker.Complete(await worker.Send("discard-pending",new { }));
                Assert(cleared.Data.GetProperty("result").GetProperty("discarded").GetInt32()==1,"Pending queue did not discard exactly one new example.");
                Assert(new ReplayLedger(root).Summary.Discarded==1,"Pending discard is not durable.");
            }
            await using (var resumed = new Child(executable, root))
            {
                var ready = await resumed.Wait(e => e.Kind == "ready");
                Assert(ready.Data.GetProperty("hasModel").GetBoolean(), "Workspace did not reopen.");
                Assert(ready.Data.GetProperty("paused").GetBoolean() && !ready.Data.GetProperty("online").GetBoolean(), "Stop state was not durable.");
                Assert(Math.Abs(ready.Data.GetProperty("training").GetProperty("onlineLearningRate").GetDouble()-0.000037)<1e-12,"Online learning rate did not survive restart.");
                Assert(new ReplayLedger(root).PendingCount==0,"Discarded work restarted.");
                ModelFiles.VerifyRevision(ModelFiles.ActivePath(root)!);
            }
            // Simulate a durable queue created by an older version, outside a live trainer lease.
            string activeBefore=ModelFiles.ActivePath(root)!;
            var control=JsonData.Read<TrainingExample[]>(Path.Combine(activeBefore,"validation.json")).First(e=>e.IsDialogue);
            var contaminated=Dataset.Make(control.Text,"Другой эталон",history:control.History);
            new ReplayLedger(root).Add(contaminated);
            await using(var legacyQueue=new Child(executable,root))
            {
                await legacyQueue.Wait(e=>e.Kind=="ready");
                await legacyQueue.Complete(await legacyQueue.Send("mode",new OnlineMode(true,0.000037)));
                var error=await legacyQueue.Wait(e=>e.Kind=="error",allowErrors:true);
                Assert(error.Data.GetProperty("message").GetString()!.Contains("Контрольный вход"),"Old pending control input was not rechecked.");
                await legacyQueue.Complete(await legacyQueue.Send("status",new{}),allowErrors:true);
                Assert(ModelFiles.ActivePath(root)==activeBefore,"Legacy contaminated replay changed weights.");
                Assert(new ReplayLedger(root).PendingCount==1,"Rejected legacy queue was silently erased or trained.");
                await legacyQueue.Complete(await legacyQueue.Send("discard-pending",new{}),allowErrors:true);
            }
            await CheckModeFailureAndRefusal(executable, Path.Combine(root, "mode-failure"));
            await CheckMalformedCommandChannel(executable, Path.Combine(root, "command-frames"));
            await CheckEvolution(executable,Path.Combine(root,"stage-checks"));
            await CheckFailedRollback(executable,Path.Combine(root,"stage-checks"));
            await CheckCreationBudgetAndQuality(executable,Path.Combine(root,"audit17"));
            Console.WriteLine("PASS actual worker: creation, receipts, coherent snapshots, online gradients, rollback, idempotency, stop fence, restart.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine("FAIL actual worker: " + error); return 1; }
        finally { Directory.Delete(root, recursive: true); }
    }
    private static async Task CheckCreationBudgetAndQuality(string executable,string root)
    {
        var resources=new ResourceOptions{Threads=2,MemoryMiB=16384,BatchSize=32,SequenceLength=1024,PreferCuda=false};
        await using(var worker=new Child(executable,root))
        {
            await worker.Wait(e=>e.Kind=="ready");
            await worker.Complete(await worker.Send("create",new CreateRequest(ModelConfig.Large,resources,new TrainingOptions{Steps=8000},[],CreationMode.Untrained)));
            string active=ModelFiles.ActivePath(root)!;var before=ModelFiles.VerifyRevision(active);
            Assert(before.Step==0 && before.TargetTokens==0,"Zero creation took optimizer steps.");
        }
        await using(var worker=new Child(executable,root))
        {
            await worker.Wait(e=>e.Kind=="ready");
            string active=ModelFiles.ActivePath(root)!;var before=ModelFiles.VerifyRevision(active);
            string replay=Path.Combine(root,"replay.json");string? replayHash=File.Exists(replay)?ModelFiles.Hash(replay):null;
            string runtime=Path.Combine(root,"runtime.json");string? runtimeHash=File.Exists(runtime)?ModelFiles.Hash(runtime):null;
            var completed=await worker.Complete(await worker.Send("quality",new{}));
            var result=completed.Data.GetProperty("result");
            Assert(result.GetProperty("step").GetInt64()==0&&result.GetProperty("parity").GetBoolean(),"Quality probe failed numerical pipeline on current random model.");
            string report=result.GetProperty("file").GetString()!;Assert(File.Exists(report),"Diagnostic result file missing.");
            var after=ModelFiles.VerifyRevision(ModelFiles.ActivePath(root)!);
            Assert(before==after && active==ModelFiles.ActivePath(root),"Quality check changed active weights.");
            Assert(replayHash==(File.Exists(replay)?ModelFiles.Hash(replay):null),"Quality check changed replay queue.");
            Assert(runtimeHash==(File.Exists(runtime)?ModelFiles.Hash(runtime):null),"Quality check changed learning mode.");
        }
        Console.WriteLine("PASS audit17: large zero-step create/reopen ignores future activation estimate; actual model quality report preserves weights, queue and mode.");
    }
    private static async Task CheckModeFailureAndRefusal(string executable, string root)
    {
        var config = new ModelConfig { Dimension=16, HiddenDimension=32, Layers=1, Heads=2, KvHeads=1, Context=512, GroupSize=8 };
        var resources = new ResourceOptions { Threads=1, MemoryMiB=4096, BatchSize=1, SequenceLength=512, PreferCuda=false };
        var options = new TrainingOptions { Steps=1, PublishEvery=1, LearningRate=.0001 };
        await using var worker = new Child(executable, root);
        await worker.Wait(e => e.Kind == "ready");
        await worker.Complete(await worker.Send("create", new CreateRequest(config,resources,options,[],CreationMode.Untrained)));
        string active = ModelFiles.ActivePath(root)!; long publications = worker.Publications;
        await worker.Complete(await worker.Send("train", new TrainRequest([],options,resources,false,TrainingMaterial.Conversation)), expectSuccess:false,allowErrors:true);
        Assert(worker.Publications == publications, "Input refusal reloaded/republished an unchanged device session.");
        Assert(ModelFiles.ActivePath(root) == active, "Rejected stage changed active weights.");
        var refused = await worker.Complete(await worker.Send("train", new TrainRequest([],options with { Steps=100000 },resources,true,TrainingMaterial.BasicPretrain)),expectSuccess:false,allowErrors:true);
        Assert(refused.Data.GetProperty("error").GetString()!.Contains("снимков"), "Oversized publication plan was not refused before training.");
        Assert(worker.Publications==publications && ModelFiles.ActivePath(root)==active, "Capacity refusal changed or republished weights.");
        await worker.Complete(await worker.Send("online", new OnlineRequest(Dataset.Make("mode persistence fixture", "accepted target"))));
        string runtime = Path.Combine(root,"runtime.json"); byte[] saved = File.ReadAllBytes(runtime);
        File.Delete(runtime); Directory.CreateDirectory(runtime); // Deterministic write failure, even when tests run as admin.
        try
        {
            await worker.Complete(await worker.Send("mode", new OnlineMode(true,.0001)),expectSuccess:false,allowErrors:true);
            string status = await worker.Send("status", new { });
            var ready = await worker.Wait(e => e.Kind == "ready" && e.Id == status, allowErrors:true);
            Assert(!ready.Data.GetProperty("online").GetBoolean() && ready.Data.GetProperty("paused").GetBoolean(), "Failed enable left live training on.");
            await worker.Complete(status,allowErrors:true);
            Assert(ModelFiles.ActivePath(root) == active && new ReplayLedger(root).PendingCount == 1, "Failed enable trained pending material.");
        }
        finally { Directory.Delete(runtime); File.WriteAllBytes(runtime,saved); }
        await worker.Complete(await worker.Send("discard-pending",new { }),allowErrors:true);
        Console.WriteLine("PASS real worker: refused input avoids reload; failed mode persistence leaves live training paused.");
    }
    private static async Task CheckFailedRollback(string executable, string root)
    {
        string current = ModelFiles.ActivePath(root)!; var info = ModelFiles.ReadRevisionInfo(current);
        string parent = ModelFiles.GetRevisionPath(root, $"r{info.Parent!.Value:D16}");
        string statePath = Path.Combine(parent, "state.json"), metadataPath = Path.Combine(parent, "revision.json");
        byte[] oldState = File.ReadAllBytes(statePath), oldMetadata = File.ReadAllBytes(metadataPath);
        try
        {
            var state = JsonData.Read<TrainerState>(statePath);
            JsonData.AtomicWrite(statePath, state with { Step = state.Step + 1 });
            JsonData.AtomicWrite(metadataPath, ModelFiles.ReadRevisionInfo(parent) with { StateSha256 = ModelFiles.Hash(statePath) });
            await using var worker = new Child(executable, root); await worker.Wait(e => e.Kind == "ready");
            await worker.Complete(await worker.Send("rollback", new { }), expectSuccess: false, allowErrors: true);
            Assert(ModelFiles.ActivePath(root) == current, "Hash-valid but semantically mismatched state changed the active pointer.");
            ModelFiles.VerifyRevision(current);
            await worker.Complete(await worker.Send("status", new { }), allowErrors: true);
        }
        finally { File.WriteAllBytes(statePath, oldState); File.WriteAllBytes(metadataPath, oldMetadata); }
        Console.WriteLine("PASS failed rollback retains last active checkpoint after hash-valid state rejection.");
    }
    private static async Task CheckEvolution(string executable,string root)
    {
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,Context=512,GroupSize=8};
        var resources=new ResourceOptions{Threads=1,MemoryMiB=4096,BatchSize=1,SequenceLength=512,PreferCuda=false};
        var options=new TrainingOptions{Steps=2,PublishEvery=2,LearningRate=.0001,MaxValidationRegression=.5};
        await using(var worker=new Child(executable,root))
        {
            await worker.Wait(e=>e.Kind=="ready");
            await worker.Complete(await worker.Send("create",new CreateRequest(c,resources,options with{Steps=42},[],CreationMode.Untrained)));
            string zero=ModelFiles.ActivePath(root)!;var info=ModelFiles.VerifyRevision(zero);
            Assert(info.Step==0 && info.ChangedWeights==0,"Zero-stage creation performed hidden training.");
            Assert(ModelFiles.Read(Path.Combine(zero,"master.weights"),false).CountChanged(WeightSet.Initialize(c))==0,"Initial master differs from initialization.");
            var baseRows=JsonData.Read<TrainingExample[]>(Path.Combine(zero,"base-train.json"));
            Assert(baseRows.All(x=>!x.IsDialogue),"Untrained baseline attached conversation targets.");
            var flags=JsonData.Read<RuntimeFlags>(Path.Combine(root,"runtime.json"));Assert(!flags.Online&&flags.Paused,"Zero-stage auto-learning is not disabled.");
            string reference=Path.Combine(StageArchive.Root(root),"00-untrained","model.tritmodel");string hash=ModelFiles.Hash(reference);
            string validationHash=info.ValidationDataSha256!;
            await worker.Complete(await worker.Send("online",new OnlineRequest(Dataset.Make("queued but disabled","not learned"))));
            await worker.Complete(await worker.Send("status",new{}));Assert(ModelFiles.ActivePath(root)==zero,"Disabled queue learned on the zero stage.");
            await worker.Complete(await worker.Send("discard-pending",new{}));
            await worker.Complete(await worker.Send("train",new TrainRequest([],options,resources,false,TrainingMaterial.Conversation)),expectSuccess:false,allowErrors:true);
            Assert(ModelFiles.ActivePath(root)==zero,"Missing conversation material advanced the model.");
            Assert(!Directory.Exists(Path.Combine(StageArchive.Root(root),"02-conversation")),"False conversation reference was archived.");
            await worker.Complete(await worker.Send("train",new TrainRequest([],options,resources,true,TrainingMaterial.BasicPretrain)));
            string basic=ModelFiles.ActivePath(root)!;info=ModelFiles.VerifyRevision(basic);
            Assert(info.Step==2&&info.ChangedWeights>0,"Basic stage did not perform real learning.");
            Assert(JsonData.Read<TrainingExample[]>(Path.Combine(basic,"base-train.json")).All(x=>!x.IsDialogue),"Basic stage leaked conversation data.");
            Assert(info.ValidationDataSha256==validationHash,"Stage switch silently changed the control set.");
            Assert(File.Exists(Path.Combine(StageArchive.Root(root),"01-basic","model.tritmodel")),"Basic stage reference missing.");
            await worker.Complete(await worker.Send("train",new TrainRequest([],options,resources,true,TrainingMaterial.Conversation)));
            string conversation=ModelFiles.ActivePath(root)!;info=ModelFiles.VerifyRevision(conversation);
            Assert(info.Step==4&&JsonData.Read<TrainingExample[]>(Path.Combine(conversation,"base-train.json")).Any(x=>x.IsDialogue),"Conversation stage did not learn dialogues.");
            Assert(info.ValidationDataSha256==validationHash&&ModelFiles.Hash(reference)==hash,"Evolution modified its baseline reference.");
            flags=JsonData.Read<RuntimeFlags>(Path.Combine(root,"runtime.json"));Assert(!flags.Online&&flags.Paused,"Completed stages must leave auto-learning paused for comparison.");
        }
        await using(var reopened=new Child(executable,root))
        {
            await reopened.Wait(e=>e.Kind=="ready");
            var settings=JsonData.Read<WorkspaceSettings>(Path.Combine(ModelFiles.ActivePath(root)!,"settings.json"));
            Assert(settings.LastMaterial==TrainingMaterial.Conversation && settings.ConversationTrained==true,"Stage provenance was not durable.");
        }
        Console.WriteLine("PASS actual stage evolution: zero steps, independent text baseline, dialogue training, frozen controls/references, paused restart.");
    }
    private static async Task CheckMalformedCommandChannel(string executable, string root)
    {
        string[] frames = ["{", "{}", "{\"kind\":\"status\",\"id\":42,\"payload\":{}}",
            "{\"kind\":\"status\",\"id\":\"x\",\"payload\":false}",
            "{\"kind\":\"status\",\"id\":\"x\",\"id\":\"y\",\"payload\":{}}"];
        for (int i=0;i<frames.Length;i++)
        {
            string workspace=Path.Combine(root,i.ToString());
            await using var worker=new Child(executable,workspace);
            await worker.Wait(e=>e.Kind=="ready");
            await worker.Complete(await worker.Send("unknown-command",new{}),expectSuccess:false,allowErrors:true);
            await worker.AssertFatalFrame(frames[i]);
            Assert(ModelFiles.ActivePath(workspace) is null,"Malformed frame created or trained a model.");
        }
        Console.WriteLine("PASS actual worker: malformed command envelopes terminate without orphaned waiting callers; unknown valid commands get receipts.");
    }
    private static void Assert(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
    private sealed class Child : IAsyncDisposable
    {
        private readonly Process _process;
        private readonly Channel<WorkerEvent> _events = Channel.CreateUnbounded<WorkerEvent>();
        private readonly List<WorkerEvent> _deferred = new();
        private readonly Task _reader, _errors;
        private readonly StringBuilder _stderr = new();
        public Child(string executable, string root)
        {
            var start = new ProcessStartInfo(Path.GetFullPath(executable)) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardInputEncoding = new UTF8Encoding(false) };
            start.ArgumentList.Add("--workspace"); start.ArgumentList.Add(root);
            _process = Process.Start(start) ?? throw new IOException("Cannot start trainer.");
            _reader = Task.Run(async () =>
            {
                Exception? failure = null;
                try
                {
                    while (await _process.StandardOutput.ReadLineAsync() is string line)
                    {
                        var message = JsonSerializer.Deserialize<WorkerEvent>(line, JsonData.Options) ?? throw new InvalidDataException("Empty event.");
                        if (message.Kind == "published") Interlocked.Increment(ref _publications);
                        await _events.Writer.WriteAsync(message);
                    }
                }
                catch (Exception error) { failure = error; }
                finally { _events.Writer.TryComplete(failure); }
            });
            _errors = Task.Run(async () => { while (await _process.StandardError.ReadLineAsync() is string line) { lock (_stderr) _stderr.AppendLine(line); } });
        }
        private long _publications;
        public long Publications => Interlocked.Read(ref _publications);
        public async Task AssertFatalFrame(string frame)
        {
            await _process.StandardInput.WriteLineAsync(frame); await _process.StandardInput.FlushAsync();
            using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(10));
            await _process.WaitForExitAsync(timeout.Token); // Before Dispose: reader failure itself must terminate the worker.
            Assert(_process.ExitCode!=0,"Malformed command channel was reported as a normal exit.");
        }
        public async Task<string> Send<T>(string kind, T payload)
        {
            string id = Guid.NewGuid().ToString("N");
            await _process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(new WorkerCommand(kind, id, Protocol.Element(payload)), JsonData.Options));
            await _process.StandardInput.FlushAsync(); return id;
        }
        public async Task<WorkerEvent> Wait(Func<WorkerEvent, bool> predicate, bool allowErrors = false)
        {
            int previous = _deferred.FindIndex(e => predicate(e));
            if (previous >= 0) { var item = _deferred[previous]; _deferred.RemoveAt(previous); return item; }
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            try
            {
                while (await _events.Reader.WaitToReadAsync(timeout.Token))
                {
                    var e = await _events.Reader.ReadAsync(timeout.Token);
                    if (e.Kind == "error" && !allowErrors) throw new Exception(e.Data.ToString());
                    if (predicate(e)) return e;
                    if (e.Kind is "completed" or "published" or "online-result") _deferred.Add(e);
                }
                throw new IOException("Trainer closed before expected event.");
            }
            catch (Exception error) { lock (_stderr) throw new Exception(error.Message + "\nTrainer stderr: " + _stderr, error); }
        }
        public async Task<WorkerEvent> Complete(string id, bool expectSuccess = true, bool allowErrors = false)
        {
            var e = await Wait(e => e.Kind == "completed" && e.Id == id, allowErrors);
            Assert(e.Data.GetProperty("success").GetBoolean() == expectSuccess, e.Data.ToString()); return e;
        }
        public async ValueTask DisposeAsync()
        {
            try
            {
                if (!_process.HasExited)
                {
                    try { await Send("shutdown", new { }); _process.StandardInput.Close(); } catch { }
                    using var limit = new CancellationTokenSource(TimeSpan.FromSeconds(15));
                    try { await _process.WaitForExitAsync(limit.Token); }
                    catch (OperationCanceledException) { _process.Kill(entireProcessTree: true); await _process.WaitForExitAsync(); }
                }
                await Task.WhenAll(_reader, _errors);
            }
            finally { _process.Dispose(); }
        }
    }
}
