using System.Text.Json;
namespace TritStudio.Core;
public sealed record WorkerCommand(string Kind, string Id, JsonElement Payload);
public sealed record WorkerEvent(string Kind, string? Id, JsonElement Data);
public sealed record CreateRequest(ModelConfig Config, ResourceOptions Resources, TrainingOptions Training, string[] DatasetPaths, CreationMode Mode = CreationMode.Conversation);
public sealed record TrainRequest(string[] DatasetPaths, TrainingOptions Training, ResourceOptions? Resources = null, bool IncludeBundledUpdates = true, TrainingMaterial Material = TrainingMaterial.Conversation);
public sealed record OnlineRequest(TrainingExample Example);
public sealed record OnlineMode(bool Enabled, double LearningRate = 0.0001);
public sealed record StatusEvent(string Message, long Step = 0, double? Loss = null, double? TokensPerSecond = null,
    long MemoryMiB = 0, string Device = "", int Queue = 0, bool Busy = false, string Stage = "idle", int CompletedSteps = 0, int TotalSteps = 0, QueueSummary? Replay = null, int Snapshots = 0, StepPerformance? Performance = null, double? ValidationMilliseconds = null, long ValidationCacheHits = 0, long ValidationPreparedBytes = 0, long ValidationBatchCacheHits = 0, long SnapshotCorpusCacheHits = 0, long SnapshotCorpusCacheBytes = 0);
public sealed record StepPerformance(double Milliseconds, int BatchSize, int SequenceLength, long InputTokens, long TargetTokens, long PaddedPositions, string AttentionBackend, long OutputPositions = 0)
{
    public double PaddingFraction => PaddedPositions == 0 ? 0 : Math.Clamp(1 - (double)InputTokens / PaddedPositions, 0, 1);
}
public sealed record PublishEvent(string RevisionDirectory, RevisionInfo Info);
public sealed record ReadyEvent(bool HasModel, ModelConfig? Config, ResourceOptions? Resources, TrainingOptions? Training, bool Paused = false, bool Online = false, TrainingMaterial? LastMaterial = null);
public sealed record DatasetSummaryEvent(int TrainingExamples, int ValidationExamples, int RequiredSequenceLength, string DatasetVersion);
public sealed record RuntimeFlags(bool Paused = false, bool Online = false, double? OnlineLearningRate = null);
public sealed record QueueSummary(int Pending, int Learned, int Rejected, int RolledBack, int Discarded);
public sealed record WorkspaceSettings(ModelConfig Config, ResourceOptions Resources, TrainingOptions Training, TrainingMaterial? LastMaterial = null, bool? ConversationTrained = null, bool? CustomDataTrained = null);
public sealed record TrainerState(long Step, ulong SamplerState, long TargetTokens);
public sealed record CommitLedger(string[] AppliedOnlineIds);
public static class Protocol
{
    public static JsonElement Element<T>(T value) => JsonSerializer.SerializeToElement(value, JsonData.Options);
    public static T Payload<T>(JsonElement e) => e.Deserialize<T>(JsonData.Options) ?? throw new InvalidDataException("Empty protocol payload.");
}
