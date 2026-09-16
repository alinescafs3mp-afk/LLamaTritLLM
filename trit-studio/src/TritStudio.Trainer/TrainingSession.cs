using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using TritStudio.Core;
using static TorchSharp.torch;
namespace TritStudio.Trainer;

public sealed class TrainingSession : IDisposable
{
    public TorchModel Model { get; }
    public AdamW Optimizer { get; }
    public long Step { get; private set; }
    public string DeviceName { get; }
    private readonly ResourceOptions _resources;
    private readonly SamplerRandom _rng;
    private WeightSet? _initialMaster;
    public long InitialMasterReuses { get; private set; }
    public WeightSet CapturePublicationMaster(CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (_initialMaster is { } initial && Step == 0 && TargetTokens == 0 && Model.WeightVersion == 0)
        { InitialMasterReuses++; return initial; }
        return Model.CopyMaster(ct);
    }
    public long TargetTokens { get; private set; }
    public TrainerState State => new(Step, _rng.State, TargetTokens);
    private EncodedExample[]? _plannedCorpus, _evaluatedCorpus;
    private BatchPlanner? _planner;
    private EncodedExample[]? _preparedCorpus;
    private EvaluationBatchCache? _validationBatches;
    public long ValidationBatchCacheHits => _validationBatches?.Hits ?? 0;
    public long ValidationPreparedBytes => _validationBatches?.RetainedPayloadBytes ?? 0;
    private long _evaluatedVersion = -1;
    private double _evaluatedLoss;
    public long ValidationPasses { get; private set; }
    public long ValidationCacheHits { get; private set; }
    public StepPerformance? LastStepPerformance { get; private set; }
    public double LastValidationMilliseconds { get; private set; }
    public TrainingSession(WeightSet initial, ResourceOptions resources, TrainingOptions training)
    {
        resources.Validate(initial.Config); training.Validate(); _resources = resources;
        int effectiveThreads = Math.Min(resources.EffectiveThreads, initial.Config.ParameterCount < 200000 ? 2 : initial.Config.ParameterCount < 1000000 ? 4 : resources.EffectiveThreads);
        set_num_threads(effectiveThreads);
        // Set inter-op threads once at process initialization, not each time a model is opened.
        bool cudaAvailable = false;
        try { cudaAvailable = resources.PreferCuda && cuda.is_available(); } catch (Exception e) { Console.Error.WriteLine("CUDA probe failed: " + e.Message); }
        Device device = cudaAvailable ? CUDA : CPU;
        DeviceName = cudaAvailable ? "CUDA:0" : $"CPU · {effectiveThreads} потоков" + (resources.PreferCuda ? " (CUDA недоступна в этой сборке/среде)" : "");
        Model = new(initial, device, resources.UseSdpa);
        try { Optimizer = optim.AdamW(Model.Parameters, lr: training.LearningRate, weight_decay: 0.01); }
        catch { Model.Dispose(); throw; }
        _rng = new(unchecked((ulong)initial.Config.Seed)); _initialMaster = initial;
    }
    public (double Loss, long TargetTokens) TrainStep(EncodedExample[] corpus, double lr, CancellationToken ct, EncodedExample? required = null)
    {
        ct.ThrowIfCancellationRequested();
        if (!double.IsFinite(lr) || lr is <= 0 or > 0.01) throw new ArgumentOutOfRangeException(nameof(lr));
        if (Step == long.MaxValue || TargetTokens > long.MaxValue - (long)_resources.BatchSize * _resources.SequenceLength)
            throw new InvalidOperationException("Training counters exhausted; weights were not changed.");
        CheckMemory();
        if (corpus.Length == 0) throw new ArgumentException("Empty training corpus.");
        using var scope = NewDisposeScope();
        var watch = Stopwatch.StartNew();
        int b = _resources.BatchSize;
        if (!ReferenceEquals(_plannedCorpus, corpus)) { _planner = new BatchPlanner(corpus, ct: ct); _plannedCorpus = corpus; }
        var plan = _planner!.Select(b, _rng, _resources.BucketByLength, required);
        var selected = plan.Examples;
        int t = plan.Length;
        if (t < 1 || t > _resources.SequenceLength) throw new ArgumentException("Invalid effective batch sequence length.");
        var batch = SupervisedBatch.Build(selected, t, ct); long targetTokens = batch.Targets.LongLength;
        foreach (var pg in Optimizer.ParamGroups) pg.LearningRate = lr;
        Optimizer.zero_grad();
        var ids = tensor(batch.Inputs, dtype: ScalarType.Int64, device: Model.Device).reshape(b, t);
        var targets = tensor(batch.Targets, dtype: ScalarType.Int64, device: Model.Device);
        var positions = tensor(batch.Positions, dtype: ScalarType.Int64, device: Model.Device);
        var logits = ForwardForLoss(ids, positions);
        var loss = nn.functional.cross_entropy(logits, targets);
        loss.backward();
        // Read loss and gradient norm together: one device-to-host synchronization before the update.
        var norm2 = SquaredGradientNorm(Model.Parameters, Model.Device);
        var metrics = stack(new[] { loss.detach(), norm2.sqrt() }).cpu().data<float>().ToArray();
        double value = metrics[0], norm = metrics[1];
        if (!double.IsFinite(value) || !double.IsFinite(norm)) throw new ArithmeticException("Nonfinite loss or gradients; weights were not updated.");
        if (norm > 1) { using var ng = no_grad(); foreach (var p in Model.Parameters) { using var gradientScope = NewDisposeScope(); if (p.grad is { } g) g.mul_(1 / norm); } }
        ct.ThrowIfCancellationRequested();
        _initialMaster = null; // BEFORE a possibly partial native update, not after its success.
        Optimizer.step(); Model.MarkUpdated(); Step++; TargetTokens += targetTokens;
        LastStepPerformance = new(watch.Elapsed.TotalMilliseconds, b, t, plan.InputTokens, targetTokens, plan.PaddedPositions, Model.AttentionBackend, _resources.ProjectOnlyTargets ? targetTokens : plan.PaddedPositions);
        return (value, targetTokens);
    }
    internal static Tensor SquaredGradientNorm(IEnumerable<Parameter> parameters, Device device)
    {
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        var sum = zeros(Array.Empty<long>(), dtype: ScalarType.Float32, device: device);
        foreach (var parameter in parameters)
        {
            // Release the full-size squared-gradient temporary before visiting the next matrix.
            using var gradientScope = NewDisposeScope();
            if (parameter.grad is { } gradient) sum.add_(gradient.pow(2).sum());
        }
        return sum.MoveToOuterDisposeScope();
    }
    private Tensor ForwardForLoss(Tensor ids, Tensor positions) => _resources.ProjectOnlyTargets
        ? Model.Forward(ids, positions)
        : Model.Forward(ids).reshape(-1, ByteTokenizer.VocabularySize).index_select(0, positions);
    public double Evaluate(EncodedExample[] examples, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (examples.Length == 0) throw new ArgumentException("Validation corpus is empty.");
        if (ReferenceEquals(_evaluatedCorpus, examples) && _evaluatedVersion == Model.WeightVersion)
        { ValidationCacheHits++; LastValidationMilliseconds = 0; return _evaluatedLoss; }
        var watch = Stopwatch.StartNew();
        if (!ReferenceEquals(_preparedCorpus, examples))
        {
            // Release old retained arrays before preparing another corpus. A failed build is never published.
            _validationBatches = null; _preparedCorpus = null; _evaluatedCorpus = null; _evaluatedVersion = -1;
            var prepared = new EvaluationBatchCache(examples, _resources.BatchSize, _resources.SequenceLength, ct: ct);
            _validationBatches = prepared; _preparedCorpus = examples;
        }
        using var evaluationScope = NewDisposeScope();
        using var noGrad = no_grad(); using var cachedWeights = Model.BeginEvaluation();
        var sum = zeros(Array.Empty<long>(), dtype: ScalarType.Float64, device: Model.Device); long count = 0;
        // Quantize each matrix once for the entire pass, not once for every validation batch.
        for (int index = 0; index < _validationBatches!.Count; index++)
        {
            ct.ThrowIfCancellationRequested(); using var scope = NewDisposeScope();
            var packed = _validationBatches.Get(index, ct); long tokens = packed.Targets.LongLength;
            var ids = tensor(packed.Inputs, dtype: ScalarType.Int64, device: Model.Device).reshape(packed.Rows, packed.Length);
            var labels = tensor(packed.Targets, dtype: ScalarType.Int64, device: Model.Device);
            var positions = tensor(packed.Positions, dtype: ScalarType.Int64, device: Model.Device);
            var logits = ForwardForLoss(ids, positions);
            var loss = nn.functional.cross_entropy(logits, labels);
            sum.add_(loss.to(ScalarType.Float64) * tokens); count += tokens;
        }
        if (count == 0) throw new InvalidDataException("Validation corpus has no target tokens.");
        // One scalar device read for the complete evaluation, not one synchronization per batch.
        double result = sum.item<double>() / count;
        if (!double.IsFinite(result)) throw new ArithmeticException("Nonfinite validation loss.");
        _evaluatedCorpus = examples; _evaluatedVersion = Model.WeightVersion; _evaluatedLoss = result;
        ValidationPasses++; LastValidationMilliseconds = watch.Elapsed.TotalMilliseconds;
        return result;
    }
    public void SaveOptimizer(string path) => Optimizer.save_state_dict(path);
    public void Restore(WeightSet weights, string optimizer, TrainerState state)
    {
        if (state.Step < 0 || state.TargetTokens < 0 || state.SamplerState == 0) throw new InvalidDataException("Invalid trainer state.");
        _initialMaster = null; _evaluatedVersion = -1;
        Model.Restore(weights); RestoreState(optimizer, state);
    }
    public void RestoreState(string optimizer, TrainerState state)
    {
        if (state.Step < 0 || state.TargetTokens < 0 || state.SamplerState == 0)
            throw new InvalidDataException("Invalid training counters or sampler state.");
        _initialMaster = null; // No initial-reference shortcut after state restoration, even on failure.
        Optimizer.load_state_dict(optimizer); Optimizer.to(Model.Device);
        Step = state.Step; _rng.State = state.SamplerState; TargetTokens = state.TargetTokens;
    }
    private void CheckMemory()
    {
        // Host RSS guard, not a CUDA VRAM query. VRAM allocation failures are handled by the worker boundary.
        using var process = Process.GetCurrentProcess();
        if (process.WorkingSet64 > (long)_resources.MemoryMiB * 1024 * 1024)
            throw new OutOfMemoryException("Trainer RSS exceeds the configured memory budget; reduce model/batch/sequence length.");
    }
    public void Dispose()
    {
        _validationBatches = null; _preparedCorpus = null; _evaluatedCorpus = null;
        _planner = null; _plannedCorpus = null;
        _initialMaster = null; Optimizer.Dispose(); Model.Dispose();
    }
}
