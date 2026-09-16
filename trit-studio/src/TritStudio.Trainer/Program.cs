using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Console.InputEncoding = Encoding.UTF8; Console.OutputEncoding = new UTF8Encoding(false);
        try
        {
            if (args.Contains("--self-test")) return TrainerSelfTest.Run(args.Contains("--cuda"));
            if (args.Contains("--benchmark"))
            {
                int index=Array.IndexOf(args,"--output");
                if(index>=0 && index+1>=args.Length)throw new ArgumentException("--output requires a path.");
                return TrainerBenchmark.Run(args.Contains("--cuda"),index<0?null:args[index+1],args.Contains("--reverse"));
            }
            if (args.Contains("--doctor"))
            {
                bool available = false; string? cudaError = null;
                using var t = torch.ones(new long[] { 2, 2 });
                try
                {
                    available = torch.cuda.is_available();
                    if (available) { using var gpu = torch.ones(new long[] { 2, 2 }, device: torch.CUDA); using var sum = gpu.sum(); available = sum.item<float>() == 4; }
                } catch (Exception e) { available = false; cudaError = e.Message; }
                using var cpuSum = t.sum();
                Console.WriteLine(JsonSerializer.Serialize(new { cpu = cpuSum.item<float>() == 4, cuda = available, cudaError }, JsonData.Options));
                return 0;
            }
            if (args.Length != 2 || args[0] != "--workspace")
            { Console.Error.WriteLine("Usage: TritStudio.Trainer --workspace PATH | --doctor | --self-test [--cuda] | --benchmark [--cuda] [--reverse] [--output FILE]"); return 2; }
            torch.set_num_interop_threads(1);
            using var worker = new Worker(args[1]);
            await worker.RunAsync(); return 0;
        }
        catch (Exception e) { Console.Error.WriteLine(e.ToString()); return 1; }
    }
}

internal sealed class Worker : IDisposable
{
    private readonly WorkspaceStore _store;
    private readonly ReplayLedger _replay;
    private TrainingSession? _session;
    private WeightSet? _publishedMaster;
    private ValidationGuard? _validationGuard;
    private WorkspaceSettings? _settings;
    private EncodedExample[] _base = [], _validation = [];
    private TrainingExample[] _trainingData = [], _validationData = [];
    private readonly HashSet<string> _applied = new(StringComparer.Ordinal);
    private readonly Channel<(WorkerCommand Command, long CancelEpoch)> _commands = Channel.CreateBounded<(WorkerCommand Command, long CancelEpoch)>(new BoundedChannelOptions(128) { SingleReader = true, SingleWriter = true });
    private readonly object _outputLock = new(), _operationLock = new();
    private CancellationTokenSource? _operation;
    private string _operationKind = "";
    private readonly CancellationTokenSource _lifetime = new();
    private bool _online, _paused;
    private double _onlineLr = 0.0001;
    private bool _hasRuntimeRate;
    private bool _needsRecovery;
    private RevisionInfo? _active;
    private string? _currentId;
    private int _stopRequested, _disableOnlineRequested;
    private int _completedSteps, _totalSteps;
    private string _stage = "idle";
    private object? _commandResult;
    private long _cancelEpoch, _currentCancelEpoch;
    private readonly Stopwatch _statusClock = Stopwatch.StartNew();
    private string _reportedStage = "";
    private bool _reportedBusy;
    public Worker(string workspace)
    {
        _store = new(workspace); _replay = new(workspace);
        if (File.Exists(FileAt("runtime.json")))
        {
            var flags = JsonData.Read<RuntimeFlags>(FileAt("runtime.json")); _paused = flags.Paused; _online = flags.Online;
            if (flags.OnlineLearningRate is double rate)
            {
                if (!double.IsFinite(rate) || rate is <= 0 or > 0.001) throw new InvalidDataException("Invalid persisted online learning rate.");
                _onlineLr = rate; _hasRuntimeRate = true;
            }
        }
    }
    private void SaveMode()
    {
        var flags = new RuntimeFlags(_paused, _online, _onlineLr);
        try { JsonData.AtomicWrite(FileAt("runtime.json"), flags); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // A failed Enable request must NEVER leave hidden training enabled in memory.
            // We cannot promise persistence while the storage is unavailable; make that explicit.
            _paused = true; _online = false;
            Emit("mode-state", new RuntimeFlags(true, false, _onlineLr));
            throw new IOException("Режим не сохранён. Автообучение выключено в памяти этого процесса. " +
                "Исправьте доступ к runtime.json перед повторным открытием: прежний сохранённый режим мог остаться на диске.", error);
        }
        _hasRuntimeRate = true; Emit("mode-state", flags);
    }
    private string FileAt(string name) => Path.Combine(_store.Root, name);
    private void Emit<T>(string kind, T data, string? id = null)
    {
        var value = new WorkerEvent(kind, id ?? _currentId, Protocol.Element(data));
        lock (_outputLock) { Console.WriteLine(JsonSerializer.Serialize(value, JsonData.Options)); Console.Out.Flush(); }
    }
    private static long RssMiB() { using var process = Process.GetCurrentProcess(); return process.WorkingSet64 / 1048576; }
    private void Status(string message, double? loss = null, double? speed = null, bool busy = false)
    {
        // Tiny models can produce thousands of steps/s. Progress must not saturate the pipe/UI dispatcher.
        // Stage transitions and terminal receipts are never throttled.
        if (busy && _reportedBusy && _reportedStage == _stage && _statusClock.ElapsedMilliseconds < 250) return;
        _reportedBusy = busy; _reportedStage = _stage; _statusClock.Restart();
        Emit("status", new StatusEvent(message, _session?.Step ?? 0, loss, speed, RssMiB(),
            _session?.DeviceName ?? "", _replay.PendingCount, busy, _stage, _completedSteps, _totalSteps,
            _replay.Summary, _store.RevisionCount, _session?.LastStepPerformance, _session?.LastValidationMilliseconds,
            _session?.ValidationCacheHits ?? 0, _session?.ValidationPreparedBytes ?? 0, _session?.ValidationBatchCacheHits ?? 0, _store.CorpusCacheHits, _store.CorpusCacheBytes));
    }
    private void Error(string message) => Emit("error", new { message });
    private void Interrupt(string? onlyKind = null)
    {
        lock (_operationLock) if (onlyKind is null || onlyKind == _operationKind) _operation?.Cancel();
    }
    private async Task ReadCommands()
    {
        try
        {
            var lines = new BoundedLineReader(Console.In, 1_048_576);
            while (!_lifetime.IsCancellationRequested)
            {
                string? line = await lines.ReadLineAsync(_lifetime.Token);
                if (line is null) break;
                // Never ignore a malformed command: its caller could otherwise await completion forever.
                var command = CommandEnvelope.Parse(line);
                // Cancellation is signalled immediately, rather than waiting behind the running job.
                if (command.Kind is "stop" or "shutdown") { Interlocked.Increment(ref _cancelEpoch); Interlocked.Exchange(ref _stopRequested, 1); Interrupt(); }
                if (command.Kind == "discard-pending")
                { Interlocked.Exchange(ref _disableOnlineRequested, 1); Interrupt("online"); }
                if (command.Kind == "mode")
                {
                    try { if (!Protocol.Payload<OnlineMode>(command.Payload).Enabled) { Interlocked.Exchange(ref _disableOnlineRequested, 1); Interrupt("online"); } }
                    catch (Exception e) { Emit("completed", new { command = command.Kind, success = false, error = e.Message }, command.Id); continue; }
                }
                if (!_commands.Writer.TryWrite((command, Volatile.Read(ref _cancelEpoch))))
                {
                    Emit("completed", new { command = command.Kind, success = false, error = "Очередь команд заполнена. Команда НЕ принята." }, command.Id);
                }
                else Emit("accepted", new { command = command.Kind, message = "Команда принята в очередь выполнения." }, command.Id);
                if (command.Kind == "shutdown") { _lifetime.Cancel(); break; }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception error)
        {
            Console.Error.WriteLine("Command channel failed; uncommitted work is cancelled: " + error.Message);
            throw;
        }
        finally { _lifetime.Cancel(); Interrupt(); _commands.Writer.TryComplete(); }
    }
    public async Task RunAsync()
    {
        // Start reading before loading LibTorch so the UI can cancel even during initialization.
        var reader = Task.Run(ReadCommands);
        try
        {
            if (ModelFiles.ActivePath(_store.Root) is not null) LoadWorkspace();
            Ready();
            while (!_lifetime.IsCancellationRequested)
            {
                if (Interlocked.Exchange(ref _stopRequested, 0) != 0) _paused = true;
                if (Interlocked.Exchange(ref _disableOnlineRequested, 0) != 0) _online = false;
                if (_commands.Reader.TryRead(out var queued))
                {
                    var command = queued.Command; _currentCancelEpoch = queued.CancelEpoch;
                    _currentId = command.Id; _commandResult = null; _needsRecovery = false;
                    string? failure = null; bool cancelled = false;
                    try
                    {
                        if (command.Kind == "shutdown") break;
                        if (_currentCancelEpoch != Volatile.Read(ref _cancelEpoch) && (command.Kind is "create" or "train" or "mode"))
                            throw new OperationCanceledException("Команда отменена до начала выполнения.");
                        await Handle(command);
                    }
                    catch (OperationCanceledException) { cancelled = true; failure = "Операция отменена"; RestoreAfterFailure(); _paused = true; _online = false; SaveMode(); Status("Обучение остановлено на последнем сохранённом снимке."); }
                    catch (Exception e)
                    {
                        failure = e.Message;
                        if (command.Kind is "create" or "train" or "rollback") { RestoreAfterFailure(); _paused = true; _online = false; SaveMode(); }
                        Error(e.Message); Status("Команда не выполнена. Подробности в сообщении об ошибке.");
                    }
                    finally
                    {
                        Emit("completed", new { command = command.Kind, success = failure is null, cancelled, error = failure, result = _commandResult });
                        _currentId = null;
                    }
                }
                else if (_online && !_paused && _session is not null && _replay.PendingCount > 0)
                {
                    try { _needsRecovery = false; await InOperation("online", LearnPending); }
                    catch (OperationCanceledException) { RestoreAfterFailure(); Status("Микрообучение остановлено; очередь сохранена."); }
                    catch (Exception e) { RestoreAfterFailure(); _paused = true; _online = false; SaveMode(); Error(e.Message); Status("Автообучение приостановлено из-за ошибки; чат доступен."); }
                }
                else if (!await _commands.Reader.WaitToReadAsync(_lifetime.Token)) break;
            }
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { }
        finally { Interrupt(); _lifetime.Cancel(); try { await reader.WaitAsync(TimeSpan.FromSeconds(1)); } catch (OperationCanceledException) { } catch (TimeoutException) { } }
    }
    private void Ready() => Emit("ready", new ReadyEvent(_session is not null, _settings?.Config, _settings?.Resources, _settings is null ? null : (_settings.Training with { OnlineLearningRate = _onlineLr }), _paused, _online, _settings?.LastMaterial));
    private Task InOperation(string kind, Action<CancellationToken> action)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(_lifetime.Token);
        lock (_operationLock)
        {
            _operation = cts; _operationKind = kind;
            if (kind == "offline" && _currentCancelEpoch != Volatile.Read(ref _cancelEpoch)) cts.Cancel();
            // A disable/stop may arrive after the scheduler condition but before _operation is assigned.
            if (kind == "online" && (Volatile.Read(ref _stopRequested) != 0 || Volatile.Read(ref _disableOnlineRequested) != 0)) cts.Cancel();
        }
        try { cts.Token.ThrowIfCancellationRequested(); action(cts.Token); }
        finally { _stage = "idle"; _completedSteps = 0; _totalSteps = 0; lock (_operationLock) { _operation = null; _operationKind = ""; } }
        return Task.CompletedTask;
    }
    private Task Handle(WorkerCommand command)
    {
        switch (command.Kind)
        {
            case "create": return InOperation("offline", ct => Create(Protocol.Payload<CreateRequest>(command.Payload), ct));
            case "train": return InOperation("offline", ct => Train(Protocol.Payload<TrainRequest>(command.Payload), ct));
            case "mode":
                var mode = Protocol.Payload<OnlineMode>(command.Payload);
                if (!double.IsFinite(mode.LearningRate) || mode.LearningRate is <= 0 or > 0.001) throw new ArgumentException("Недопустимая скорость онлайн-обучения.");
                _online = mode.Enabled; _onlineLr = mode.LearningRate; if (_online) _paused = false; SaveMode();
                Status(_online ? "Автообучение разрешено. Ответы модели не используются без подтверждения." : "Автообучение выключено."); break;
            case "online":
                var submitted = Protocol.Payload<OnlineRequest>(command.Payload).Example;
                var clean = Dataset.Make(submitted.Text, submitted.Answer, submitted.Source, submitted.History);
                if (clean.Id != submitted.Id) throw new InvalidDataException("Training example identity mismatch.");
                if (_session is null || _settings is null) throw new InvalidOperationException("Сначала создайте модель.");
                var chunks = Dataset.SplitLongTexts([clean], _settings.Resources.SequenceLength);
                // Validate before queuing. A long supervised answer must not silently lose its target.
                foreach (var x in chunks) _ = Dataset.Encode(x, _settings.Resources.SequenceLength);
                (_validationGuard ?? throw new InvalidDataException("Validation guard is not initialized.")).EnsureTraining(chunks);
                int queued = _replay.AddMany(chunks);
                _commandResult = new { queued, duplicates = chunks.Length - queued, pending = _replay.PendingCount, trainingEnabled = _online && !_paused };
                Status(queued > 0 ? $"В очередь добавлено примеров: {queued}." : "Этот пример уже есть в журнале; повторное обучение не запущено."); break;
            case "discard-pending":
                _online = false; _paused = true; SaveMode();
                int discarded = _replay.DiscardPending(); _commandResult = new { discarded, queue = _replay.Summary };
                Status($"Из очереди исключено примеров: {discarded}. Уже обученные веса не отменены. Автообучение выключено."); break;
            case "stop": _paused = true; _online = false; SaveMode(); Status("Обучение остановлено. Для автообучения снова включите переключатель в чате."); break;
            case "rollback": Rollback(); break;
            case "status": Status(_paused ? "Обучение приостановлено." : "Готово."); Ready(); break;
            default: throw new ArgumentException("Неизвестная команда: " + command.Kind);
        }
        return Task.CompletedTask;
    }
    private void Create(CreateRequest r, CancellationToken ct)
    {
        if (ModelFiles.ActivePath(_store.Root) is not null || _session is not null) throw new InvalidOperationException("В этой рабочей папке уже есть модель. Создайте новую папку модели.");
        r.Config.Validate(); r.Resources.Validate(r.Config);
        var training = LearningStages.CreationOptions(r.Mode, r.Training);
        _store.EnsureCapacity(CheckpointBudget.Publications(training.Steps, training.PublishEvery, includeInitial: true));
        bool conversation = r.Mode == CreationMode.Conversation;
        if (!conversation && r.DatasetPaths.Length != 0) throw new ArgumentException("На нулевом и базовом этапах свои датасеты не подключаются. Добавьте их отдельным запуском дообучения.");
        _stage = "dataset"; Status(conversation ? "Подготовка разговорного корпуса…" : "Подготовка базового корпуса. Разговорные примеры в обучение не попадут.", busy: true);
        var validation = LoadSeed("validation.jsonl", ct); var guard = new ValidationGuard(validation, r.Resources.SequenceLength, ct);
        var imported = Dataset.LoadManyTraining(r.DatasetPaths, ct); guard.EnsureTraining(imported, ct);
        // Untrained creation attaches only the tiny baseline as future material; it takes ZERO optimizer steps.
        var train = LoadSeed(conversation ? "seed.jsonl" : "pretrain.jsonl", ct).Where(x => !guard.Contains(x, ct)).Concat(imported).DistinctBy(x => x.Id).ToArray();
        train = Dataset.SplitLongTexts(train, r.Resources.SequenceLength, ct); guard.EnsureTraining(train, ct);
        var encoded = Dataset.EncodeAll(train, r.Resources.SequenceLength, r.Resources.Threads, ct);
        var val = Dataset.EncodeAll(validation, r.Resources.SequenceLength, r.Resources.Threads, ct);
        if (encoded.Length == 0) throw new ArgumentException("Нет обучающих примеров.");
        ct.ThrowIfCancellationRequested(); _stage = "initialize"; Status("Инициализация случайных весов…", busy: true);
        var weights = WeightSet.Initialize(r.Config, r.Resources.EffectiveThreads, ct);
        _needsRecovery = true; _settings = new(r.Config, r.Resources, training);
        _stage = "device"; Status("Подключение вычислительного устройства…", busy: true);
        _session = new(weights, r.Resources, training);
        _base = encoded; _validation = val; _trainingData = train; _validationData = validation; _validationGuard = guard;
        Emit("dataset", new DatasetSummaryEvent(train.Length, validation.Length, train.Max(Dataset.RequiredSequenceLength), conversation ? BundledCorpus.Version : "basic-ru-v7 · только материал, ещё не обучен"));
        JsonData.AtomicWrite(FileAt("origin.json"), new { createdAt = DateTimeOffset.UtcNow, tokenizer = "utf8-byte-v1", pretrained = false, creationMode = r.Mode });
        _stage = "checkpoint"; Status("Сохранение нулевого снимка, без шагов обучения…", busy: true);
        var initial = _store.Publish(_session, weights, null, "Создана, ещё не обучена", _applied, null, r.Resources.EffectiveThreads, _settings, _trainingData, _validationData, ct);
        _active = initial.Event.Info; _publishedMaster = initial.Master; Emit("published", initial.Event);
        _online = false; _paused = true; _onlineLr = training.OnlineLearningRate; SaveMode(); Ready();
        PreserveStage("00-untrained", "Случайные веса, шаг 0");
        if (training.Steps == 0)
        {
            Status("Создана НЕОБУЧЕННАЯ модель: 0 шагов. Автообучение выключено. Проверяйте ответы и запускайте базовый претрейн отдельно.");
            return;
        }
        var material = conversation ? TrainingMaterial.Conversation : TrainingMaterial.BasicPretrain;
        _settings = _settings with { LastMaterial = material, ConversationTrained = conversation, CustomDataTrained = imported.Length > 0 }; _paused = false; SaveMode();
        RunOffline(_base, training, ct);
        PreserveStage(LearningStages.ArchiveKey(material), LearningStages.Name(material));
        _paused = true; _online = false; SaveMode(); Ready();
    }
    private void PreserveStage(string key, string name)
    {
        // This optional inference-only copy must never roll back an already committed training result.
        try
        {
            string active = ModelFiles.ActivePath(_store.Root) ?? throw new InvalidDataException("No active model.");
            var saved = StageArchive.PreserveWithReceipt(_store.Root, active, key, name);
            Emit("stage-reference", new { key, name, file = saved.File, created = saved.Created,
                revision = saved.Reference.Revision, step = saved.Reference.Step,
                message = saved.Created ? $"Эталон этапа создан: r{saved.Reference.Revision}, шаг {saved.Reference.Step}." :
                    $"Сохранён прежний первый эталон: r{saved.Reference.Revision}, шаг {saved.Reference.Step}. Текущие веса в рабочей папке, эталон не перезаписан." });
        }
        catch (Exception error) { Emit("warning", new { message = "Веса сохранены, но отдельный эталон этапа не создан: " + error.Message }); }
    }
    private TrainingExample[] LoadSeed(string name, CancellationToken ct = default) => BundledCorpus.Load(Path.Combine(AppContext.BaseDirectory, "data"), name, ct);
    private void LoadWorkspace(string? revisionPath = null, bool announce = true, bool reconcile = true)
    {
        string path = revisionPath ?? ModelFiles.ActivePath(_store.Root) ?? throw new InvalidDataException("No active snapshot.");
        var info = ModelFiles.VerifyRevision(path);
        string inputRoot = info.CheckpointVersion >= 2 ? path : _store.Root;
        var settings = JsonData.Read<WorkspaceSettings>(Path.Combine(inputRoot, "settings.json"));
        settings.Resources.Validate(settings.Config); settings.Training.Validate();
        if (settings.LastMaterial is TrainingMaterial savedMaterial) LearningStages.Validate(savedMaterial);
        var state = JsonData.Read<TrainerState>(Path.Combine(path, "state.json"));
        var commits = JsonData.Read<CommitLedger>(Path.Combine(path, "commit.json"));
        CheckpointStateGuard.Validate(info, state, commits, settings.Config);
        var master = ModelFiles.Read(Path.Combine(path, "master.weights"), false);
        if (master.Config != settings.Config) throw new InvalidDataException("Конфигурация снимка не совпадает с моделью.");
        var trainData = JsonData.Read<TrainingExample[]>(Path.Combine(inputRoot, "base-train.json"));
        var valData = JsonData.Read<TrainingExample[]>(Path.Combine(inputRoot, "validation.json"));
        if (trainData.Length is < 1 or > Dataset.MaxExamples || valData.Length is < 1 or > Dataset.MaxExamples)
            throw new InvalidDataException("Invalid snapshot corpus size.");
        foreach (var e in trainData.Concat(valData))
            if (e is null || Dataset.Make(e.Text, e.Answer, e.Source, e.History).Id != e.Id)
                throw new InvalidDataException("Snapshot example identity mismatch.");
        var guard = new ValidationGuard(valData, settings.Resources.SequenceLength); guard.EnsureTraining(trainData);
        var encoded = Dataset.EncodeAll(trainData, settings.Resources.SequenceLength, settings.Resources.EffectiveThreads);
        var validation = Dataset.EncodeAll(valData, settings.Resources.SequenceLength, settings.Resources.EffectiveThreads);
        // Release the old device buffers only after cross-file semantic validation.
        _session?.Dispose(); _session = null;
        var candidate = new TrainingSession(master, settings.Resources, settings.Training);
        try { candidate.RestoreState(Path.Combine(path, "optimizer.bin"), state); }
        catch { candidate.Dispose(); throw; }
        _session = candidate; _publishedMaster = master; _settings = settings; _trainingData = trainData; _validationData = valData; _validationGuard = guard;
        _base = encoded; _validation = validation;
        _applied.Clear(); foreach (string id in commits.AppliedOnlineIds) _applied.Add(id);
        if (reconcile) _replay.Reconcile(_applied);
        _active = info; if (!_hasRuntimeRate) _onlineLr = settings.Training.OnlineLearningRate;
        if (announce)
        {
            Emit("dataset", new DatasetSummaryEvent(trainData.Length, valData.Length, trainData.Max(Dataset.RequiredSequenceLength), "workspace-snapshot"));
            Emit("published", new PublishEvent(Path.GetFileName(path), info));
            Status("Модель и состояние оптимизатора восстановлены.");
        }
    }
    private void Train(TrainRequest r, CancellationToken ct)
    {
        NeedSession(); r.Training.Validate(); LearningStages.Validate(r.Material);
        if (r.Training.Steps < 1) throw new ArgumentException("Для дообучения укажите хотя бы один шаг. Нулевой запуск не сохраняет новый датасет.");
        _store.EnsureCapacity(CheckpointBudget.Publications(r.Training.Steps, r.Training.PublishEvery));
        _stage = "dataset"; Status("Проверка датасетов и параметров дообучения…", busy: true);
        var settings = _settings!;
        var resources = r.Resources ?? settings.Resources; resources.Validate(settings.Config);
        // Validate only material this operation consumes. Scanning/encoding the entire historical replay
        // was both expensive and allowed an unused old long message to block a baseline-only stage.
        bool conversationPreviouslyTrained = settings.ConversationTrained == true;
        // Legacy LastMaterial alone did not prove which corpus was consumed. Explicitly attaching
        // the bundled corpus (the default) upgrades that provenance without guessing about old weights.
        LearningStages.ValidateMaterialRequest(r.Material, r.IncludeBundledUpdates, r.DatasetPaths.Length, conversationPreviouslyTrained,
            settings.CustomDataTrained == true);
        // Keep dataset identity split stable. Validation samples are never sent to the optimizer.
        bool basicOnly = r.Material == TrainingMaterial.BasicPretrain;
        if (basicOnly && (_applied.Count > 0 || (_active?.Step > 0 && settings.LastMaterial != TrainingMaterial.BasicPretrain)))
            throw new ArgumentException("Базовый этап доступен для новой модели или продолжения базового претрейна. После разговорного/онлайн-обучения используйте дообучение, откат или новую модель.");
        if (basicOnly && r.DatasetPaths.Length != 0) throw new ArgumentException("Базовый этап использует только pretrain.jsonl; свои файлы подключаются отдельным этапом.");
        var old = basicOnly ? Array.Empty<TrainingExample>() : _trainingData;
        var guard = new ValidationGuard(_validationData, resources.SequenceLength, ct); guard.EnsureTraining(old, ct);
        var imported = Dataset.LoadManyTraining(r.DatasetPaths, ct); guard.EnsureTraining(imported, ct);
        var bundled = basicOnly ? LoadSeed("pretrain.jsonl", ct) :
            r.Material == TrainingMaterial.Conversation && r.IncludeBundledUpdates ? LoadSeed("seed.jsonl", ct) : Array.Empty<TrainingExample>();
        var replay = basicOnly ? Array.Empty<TrainingExample>() : _replay.RecentLearned(256);
        var added = bundled.Concat(imported).Concat(replay).Where(x => !guard.Contains(x, ct));
        var all = Dataset.SplitLongTexts(old.Concat(added).DistinctBy(x => x.Id), resources.SequenceLength, ct).DistinctBy(x => x.Id).ToArray();
        if (all.Length == 0) throw new ArgumentException("Нет обучающих примеров для выбранного этапа.");
        all = Dataset.ReuseUnchanged(_trainingData, all, ct);
        bool sameBudget = resources.SequenceLength == settings.Resources.SequenceLength;
        // Resource/backend changes alone do not change byte tokenization. Keep exact corpus owners so
        // the serialization cache remains valid; any changed sequence budget forces re-encoding.
        var encoded = sameBudget && ReferenceEquals(all, _trainingData) ? _base :
            Dataset.EncodeAll(all, resources.SequenceLength, resources.Threads, ct);
        var validation = sameBudget ? _validation :
            Dataset.EncodeAll(_validationData, resources.SequenceLength, resources.Threads, ct);
        ct.ThrowIfCancellationRequested(); guard.EnsureTraining(all, ct);
        _needsRecovery = true; // From this point session/settings can differ from the last committed snapshot.
        if (resources != settings.Resources)
        {
            // Rebuild the trainer only at an operation boundary and carry the full optimizer state across devices.
            string active = ModelFiles.ActivePath(_store.Root)!; var info = ModelFiles.VerifyRevision(active);
            var master = ModelFiles.Read(Path.Combine(active, "master.weights"), false);
            var state = JsonData.Read<TrainerState>(Path.Combine(active, "state.json"));
            CheckpointStateGuard.Validate(info, state, JsonData.Read<CommitLedger>(Path.Combine(active, "commit.json")), master.Config);
            TrainingSession? candidate = null;
            _session!.Dispose(); _session = null;
            try
            {
                candidate = new(master, resources, r.Training);
                candidate.RestoreState(Path.Combine(active, "optimizer.bin"), state);
            }
            catch { candidate?.Dispose(); throw; }
            _session = candidate;
        }
        _base = encoded; _validation = validation; _validationGuard = guard;
        _trainingData = all; _settings = settings with { Training = r.Training, Resources = resources, LastMaterial = r.Material,
            ConversationTrained = conversationPreviouslyTrained || (r.Material == TrainingMaterial.Conversation && r.IncludeBundledUpdates),
            CustomDataTrained = settings.CustomDataTrained == true || imported.Length > 0 };
        Emit("dataset", new DatasetSummaryEvent(all.Length, _validationData.Length, all.Max(Dataset.RequiredSequenceLength), "workspace-snapshot"));
        _paused = false; _online = false; _onlineLr = r.Training.OnlineLearningRate; SaveMode(); RunOffline(_base, r.Training, ct);
        PreserveStage(LearningStages.ArchiveKey(r.Material), LearningStages.Name(r.Material));
        _paused = true; SaveMode(); Ready();
    }
    private void NeedSession() { if (_session is null || _settings is null) throw new InvalidOperationException("Сначала создайте или откройте модель."); }
    private void RunOffline(EncodedExample[] corpus, TrainingOptions options, CancellationToken ct)
    {
        NeedSession(); var session = _session!; _totalSteps = options.Steps; _completedSteps = 0;
        for (int start = 0; start < options.Steps; start += options.PublishEvery)
        {
            ct.ThrowIfCancellationRequested(); var before = _publishedMaster ?? throw new InvalidOperationException("Missing committed master snapshot.");
            _stage = "validate"; Status("Контрольная оценка перед обновлением…", busy: true);
            double baseline = session.Evaluate(_validation, ct);
            int steps = Math.Min(options.PublishEvery, options.Steps - start); var watch = Stopwatch.StartNew(); long tokens = 0; double loss = 0;
            for (int i = 0; i < steps; i++)
            {
                _stage = "train"; _completedSteps = start + i + 1;
                var result = session.TrainStep(corpus, options.LearningRate, ct); loss = result.Loss; tokens += result.TargetTokens;
                if (i % 2 == 0) Status($"Обучение: {start + i + 1}/{options.Steps}", loss, tokens / Math.Max(watch.Elapsed.TotalSeconds, 0.001), true);
            }
            _stage = "validate"; Status("Проверка кандидата перед публикацией…", busy: true);
            double validation = session.Evaluate(_validation, ct);
            double anchor = ValidationBaseline.Anchor(baseline, _active, _settings!.Resources.SequenceLength);
            if (validation > anchor + Math.Max(0.02, anchor * options.MaxValidationRegression))
            { throw new InvalidOperationException("Кандидат отклонён: ухудшение контрольной ошибки. Предыдущий снимок сохранён; уменьшите скорость обучения."); }
            _stage = "checkpoint"; Status("Упаковка и сохранение проверенных весов…", busy: true);
            Commit(before, validation, _settings!.LastMaterial == TrainingMaterial.BasicPretrain ? "Базовый претрейн (без разговорного корпуса)" : "Обучение на датасете", [], ct);
        }
        if (options.Steps == 0) Status("Шаги обучения не запускались. Веса не изменены.");
        else Status("Обучение на датасете завершено. Проверенные изменения опубликованы.");
    }
    private void LearnPending(CancellationToken ct)
    {
        NeedSession(); var session = _session!; var settings = _settings!; _stage = "online"; _totalSteps = 4; _completedSteps = 0;
        var pending = _replay.PeekPending(4); if (pending.Length == 0) return;
        _store.EnsureCapacity(1); // Refuse a full history before spending four gradient updates.
        var guard = _validationGuard ?? throw new InvalidDataException("Validation guard is not initialized.");
        // Durable replay may have been queued by an older build or for another sequence budget.
        // Recheck at consumption, not only when accepting the original UI command.
        guard.EnsureTraining(pending, ct);
        var recent = Dataset.EncodeAll(pending, settings.Resources.SequenceLength, settings.Resources.Threads, ct);
        // Half the mixed pool consists of the new examples; half comes from existing training/replay.
        var learned = _replay.RecentLearned(64);
        // Only up to four learned rows can be selected; do not encode all 64 on every micro-update.
        var replayBatch = OnlineReplay.Build(_base, recent, learned, session.Step,
            settings.Resources.SequenceLength, guard, ct);
        var mixed = replayBatch.Examples;
        var before = _publishedMaster ?? throw new InvalidOperationException("Missing committed master snapshot."); _stage = "validate"; Status("Проверка перед онлайн-обновлением…", busy: true);
        double baseline = session.Evaluate(_validation, ct); _stage = "online";
        _needsRecovery = true; // RNG, gradients, optimizer and weights can change even if the candidate fails.
        var watch = Stopwatch.StartNew(); long tokens = 0; double loss = 0;
        for (int i = 0; i < 4; i++)
        {
            _completedSteps = i + 1;
            var result = session.TrainStep(mixed, _onlineLr, ct, recent[i % recent.Length]); loss = result.Loss; tokens += result.TargetTokens;
            Status($"Онлайн-обучение: микрошага {i + 1}/4", loss, tokens / Math.Max(watch.Elapsed.TotalSeconds, 0.001), true);
        }
        _stage = "validate"; Status("Проверка онлайн-кандидата…", busy: true);
        double validation = session.Evaluate(_validation, ct);
        double anchor = ValidationBaseline.Anchor(baseline, _active, _settings!.Resources.SequenceLength);
        bool accepted = validation <= anchor + Math.Max(0.02, anchor * settings.Training.MaxValidationRegression);
        if (accepted) { _stage = "checkpoint"; Status("Сохранение онлайн-обновления…", busy: true); Commit(before, validation, "Онлайн-обучение", pending.Select(x => x.Id), ct); }
        else { RestoreAfterFailure(); Status("Онлайн-изменения отклонены по контрольной ошибке."); }
        _replay.Finish(pending.Select(x => x.Id), accepted);
        Emit("online-result", new { accepted, count = pending.Length, revision = _active?.Revision, message = accepted ? "Веса опубликованы" : "Кандидат отклонён; сохранены прежние веса" });
        Status(accepted ? "Веса обновлены; отвечающая модель получит снимок между ответами." : "Сохранена предыдущая версия весов.");
    }
    private void Commit(WeightSet previous, double validation, string reason, IEnumerable<string> ids, CancellationToken ct)
    {
        NeedSession(); var nextIds = _applied.Concat(ids).ToArray();
        var published = _store.Publish(_session!, previous, validation, reason, nextIds, _active?.Revision, _settings!.Resources.EffectiveThreads, _settings, _trainingData, _validationData, ct);
        foreach (string id in ids) _applied.Add(id);
        _active = published.Event.Info; _publishedMaster = published.Master; Emit("published", published.Event);
    }
    private void RestoreAfterFailure()
    {
        if (!_needsRecovery) return; // Refused input did not mutate the session: no full rehash/encode/GPU reload.
        _needsRecovery = false;
        try
        {
            string? path = ModelFiles.ActivePath(_store.Root);
            if (path is null) { _session?.Dispose(); _session = null; _publishedMaster = null; _settings = null; return; }
            LoadWorkspace(); // Restore inputs and device settings as well as weights, moments and RNG.
        }
        catch (Exception e) { Error("Не удалось восстановить тренер: " + e.Message); _session?.Dispose(); _session = null; _publishedMaster = null; }
    }
    private void Rollback()
    {
        NeedSession(); if (_active?.Parent is not long parent) throw new InvalidOperationException("Предыдущего снимка нет.");
        string name = $"r{parent:D16}", path = ModelFiles.GetRevisionPath(_store.Root, name);
        _paused = true; _online = false; SaveMode();
        // The previous active pointer stays authoritative if native optimizer restore or corpus decoding fails.
        // Do not emit a parent publication or mutate replay before this prospective load succeeds.
        _needsRecovery = true;
        LoadWorkspace(path, announce: false, reconcile: false);
        JsonData.AtomicWrite(FileAt("active.json"), new ActiveRevision(name));
        try { _replay.Reconcile(_applied); _replay.MarkRollback(_applied); }
        catch (Exception error)
        { Emit("warning", new { message = "Откат весов выполнен, но журнал очереди требует восстановления: " + error.Message }); }
        Emit("dataset", new DatasetSummaryEvent(_trainingData.Length, _validationData.Length, _trainingData.Max(Dataset.RequiredSequenceLength), "workspace-snapshot"));
        Emit("published", new PublishEvent(name, _active!)); Ready();
        Status("Откат выполнен. Обучение приостановлено, отменённые примеры повторно не запускаются.");
    }
    public void Dispose() { Interrupt(); _session?.Dispose(); _store.Dispose(); _lifetime.Dispose(); }
}
