using TritStudio.Core;
namespace TritStudio.Trainer;

public sealed class WorkspaceStore : IDisposable
{
    public int RevisionCount { get; private set; }
    public string Root { get; }
    private readonly FileStream _lease;
    private readonly CheckpointJsonCache _trainingCache = new(), _validationCache = new();
    public long CorpusCacheHits => _trainingCache.Hits + _validationCache.Hits;
    public long CorpusCacheBytes => _trainingCache.RetainedBytes + _validationCache.RetainedBytes;
    public WorkspaceStore(string root)
    {
        Root = Path.GetFullPath(root); bool createdDirectory = !Directory.Exists(Root); Directory.CreateDirectory(Root);
        if (createdDirectory && !OperatingSystem.IsWindows()) File.SetUnixFileMode(Root, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        _lease = new FileStream(Path.Combine(Root, ".trainer.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        Directory.CreateDirectory(Path.Combine(Root, "revisions"));
        RevisionCount = Directory.GetDirectories(Path.Combine(Root, "revisions"), "r*").Length;
    }
    public void EnsureCapacity(int publications)
    {
        // A single trainer owns the workspace. This is a count preflight, not a disk-space reservation.
        RevisionCount = Directory.EnumerateDirectories(Path.Combine(Root, "revisions"), "r*").Count();
        CheckpointBudget.EnsureFits(RevisionCount, publications);
    }
    public CheckpointPublication Publish(TrainingSession session, WeightSet previous, double? validation, string reason,
        IEnumerable<string> appliedOnlineIds, long? parent, int threads, WorkspaceSettings settings, TrainingExample[] trainingData, TrainingExample[] validationData, CancellationToken ct = default, LearningProgress? accuracy = null)
    {
        ct.ThrowIfCancellationRequested();
        accuracy?.Validate(session.Step, settings.Config.Context);
        if (accuracy?.Validation is { } control && (validation is null || control.Step != session.Step || control.SequenceLength != settings.Resources.SequenceLength))
            throw new InvalidDataException("Точность контроля не соответствует сохраняемым весам и длине.");
        string revisions = Path.Combine(Root, "revisions");
        // Online publication never deletes a revision being used by the UI. The offline maintenance command applies retention under exclusive workspace locks.
        var drive = new DriveInfo(Path.GetPathRoot(Root)!);
        long reserve = Math.Max(512L * 1024 * 1024, previous.Config.ParameterCount * 40);
        if (drive.AvailableFreeSpace < reserve) throw new IOException("Недостаточно свободного места для безопасного снимка. Освободите диск или перенесите рабочую папку.");
        if (Directory.EnumerateDirectories(revisions, "r*").Take(CheckpointBudget.Limit).Count() >= CheckpointBudget.Limit)
            throw new IOException("Достигнут предел 128 снимков. Используйте «Очистить старые снимки» на вкладке обучения или runner --prune после закрытия приложения. Активные веса сохранены.");
        long revision = RevisionSequence.Reserve(Root); string name = $"r{revision:D16}";
        string staging = Path.Combine(revisions, ".stage-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try
        {
            var master = session.CapturePublicationMaster(ct);
            string masterHash = ModelFiles.WriteHashed(Path.Combine(staging, "master.weights"), master, false, threads, ct);
            string packedHash = ModelFiles.WriteHashed(Path.Combine(staging, "model.tritmodel"), master, true, threads, ct);
            ct.ThrowIfCancellationRequested();
            session.SaveOptimizer(Path.Combine(staging, "optimizer.bin"));
            ct.ThrowIfCancellationRequested();
            using (var f = new FileStream(Path.Combine(staging, "optimizer.bin"), FileMode.Open, FileAccess.ReadWrite)) f.Flush(true);
            JsonData.AtomicWrite(Path.Combine(staging, "state.json"), session.State);
            JsonData.AtomicWrite(Path.Combine(staging, "settings.json"), settings);
            string trainingHash = _trainingCache.WriteNew(Path.Combine(staging, "base-train.json"), trainingData, ct);
            string validationHash = _validationCache.WriteNew(Path.Combine(staging, "validation.json"), validationData, ct);
            JsonData.AtomicWrite(Path.Combine(staging, "commit.json"), new CommitLedger(appliedOnlineIds.Distinct().ToArray()));
            var previousInfo = parent is long parentId ? ModelFiles.ReadRevisionInfo(ModelFiles.GetRevisionPath(Root, $"r{parentId:D16}")) : null;
            double? oldBest = previousInfo?.ValidationSequenceLength == settings.Resources.SequenceLength ? previousInfo.BestValidationLoss : null;
            double? best = validation is double v ? Math.Min(oldBest ?? v, v) : oldBest;
            var info = new RevisionInfo(revision, session.Step, master.CountChanged(previous, threads, ct), validation, reason, DateTimeOffset.UtcNow,
                masterHash, packedHash, parent,
                ModelFiles.Hash(Path.Combine(staging, "optimizer.bin"), ct), ModelFiles.Hash(Path.Combine(staging, "state.json"), ct),
                ModelFiles.Hash(Path.Combine(staging, "commit.json"), ct), session.TargetTokens, best, 2, ModelFiles.Hash(Path.Combine(staging, "settings.json"), ct),
                trainingHash, validationHash, settings.Resources.SequenceLength, accuracy);
            JsonData.AtomicWrite(Path.Combine(staging, "revision.json"), info);
            // Last cancellation point: once activation starts, report the committed revision, not cancellation.
            ct.ThrowIfCancellationRequested();
            var final = ModelFiles.GetRevisionPath(Root, name); Directory.Move(staging, final);
            JsonData.AtomicWrite(Path.Combine(Root, "active.json"), new ActiveRevision(name));
            RevisionCount = Directory.GetDirectories(revisions, "r*").Length;
            return new(new PublishEvent(name, info), master);
        }
        catch { if (Directory.Exists(staging)) Directory.Delete(staging, true); throw; }
    }
    public void Dispose() { _trainingCache.Clear(); _validationCache.Clear(); _lease.Dispose(); }
}

public sealed record CheckpointPublication(PublishEvent Event, WeightSet Master);
