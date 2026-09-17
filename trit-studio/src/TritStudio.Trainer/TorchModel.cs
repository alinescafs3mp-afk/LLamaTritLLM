using TorchSharp;
using TorchSharp.Modules;
using TritStudio.Core;
using static TorchSharp.torch;
namespace TritStudio.Trainer;

// Training tensors stay on one device. Small metrics transfer per step; full weights transfer at publication.
// The quantizer has an identity straight-through derivative and is numerically matched to Core.
public sealed class TorchModel : IDisposable
{
    private readonly Dictionary<string, Parameter> _weights = new();
    private readonly ModelConfig _c;
    private readonly Device _device;
    private readonly WeightShape[] _layout;
    private readonly bool _useSdpa;
    private Parameter[] _parameters = [];
    private Dictionary<string, Tensor>? _evaluationWeights;
    public long WeightVersion { get; private set; }
    public long QuantizationBuilds { get; private set; }
    public string AttentionBackend => _useSdpa ? "SDPA (LibTorch auto)" : "explicit causal attention";
    private readonly Tensor _cos = null!, _sin = null!, _causal = null!;
    public IReadOnlyList<Parameter> Parameters => _parameters;
    public Device Device => _device;
    public ModelConfig Config => _c;
    public TorchModel(WeightSet weights, Device device, bool useSdpa = true)
    {
        using var initScope = NewDisposeScope();
        _c = weights.Config; _device = device; _layout = WeightLayout.For(_c).ToArray(); _useSdpa = useSdpa;
        try
        {
        foreach (var s in _layout)
        {
            using var tensorScope = NewDisposeScope();
            using var data = tensor(weights.Values[s.Name], dtype: ScalarType.Float32, device: device).reshape(s.Rows, s.Cols);
            _weights[s.Name] = nn.Parameter(data.clone());
            _weights[s.Name].DetachFromDisposeScope();
        }
        _parameters = _layout.Select(s => _weights[s.Name]).ToArray();
        // Cache RoPE once. SDPA owns its causal mask and kernel dispatch.
        int half = _c.HeadDimension / 2;
        var cos = new float[_c.Context * half]; var sin = new float[cos.Length];
        var denominators = new float[half];
        for (int d = 0; d < half; d++) denominators[d] = MathF.Pow(_c.RopeTheta, 2f * d / _c.HeadDimension);
        for (int p = 0; p < _c.Context; p++) for (int d = 0; d < half; d++)
        {
            float a = p / denominators[d];
            cos[p * half + d] = MathF.Cos(a); sin[p * half + d] = MathF.Sin(a);
        }
        _cos = tensor(cos, dtype: ScalarType.Float32, device: device).reshape(1, _c.Context, 1, half);
        _sin = tensor(sin, dtype: ScalarType.Float32, device: device).reshape(1, _c.Context, 1, half);
        _causal = _useSdpa ? empty(new long[] { 0 }, dtype: ScalarType.Bool, device: device) : ones(new long[] { _c.Context, _c.Context }, dtype: ScalarType.Bool, device: device).triu(1);
        _cos.DetachFromDisposeScope(); _sin.DetachFromDisposeScope(); _causal.DetachFromDisposeScope();
        }
        catch { foreach (var parameter in _weights.Values) parameter.Dispose(); throw; }
    }
    public Tensor Quantized(string name)
    {
        if (_evaluationWeights is not null)
        {
            if (!_evaluationWeights.TryGetValue(name, out var cached))
            {
                cached = BuildQuantized(name);
                cached.DetachFromDisposeScope();
                _evaluationWeights.Add(name, cached);
            }
            // A fresh managed alias belongs to the caller's scope; the cached owner survives all eval batches.
            return cached.alias();
        }
        return BuildQuantized(name);
    }
    private Tensor BuildQuantized(string name)
    {
        QuantizationBuilds++;
        var w = _weights[name];
        using var scope = NewDisposeScope();
        // Residual MUST own its storage: subtracting in place on w.detach() would corrupt master weights.
        var r = w.detach().reshape(-1, _c.GroupSize).clone();
        var q = zeros_like(r);
        using (var noGrad = no_grad())
        for (int p = 0; p < _c.Planes; p++)
        {
            // Release plane temporaries before constructing the next plane instead of accumulating all of them.
            using var planeScope = NewDisposeScope();
            var magnitude = r.abs();
            // TorchSharp 0.107 mean() takes long[] dimensions; the old int-dim overload is gone.
            var scale = magnitude.mean(new long[] { 1 }, keepdim: true).clamp_min(1e-8);
            var t = where(magnitude.gt(scale * _c.Threshold), r.sign(), zeros_like(r));
            var plane = scale * t; q.add_(plane);
            if (p + 1 < _c.Planes) r.sub_(plane);
        }
        // q contributes values, w contributes identity gradients. Scale/threshold are not differentiated.
        return (w + (q.reshape(w.shape) - w).detach()).MoveToOuterDisposeScope();
    }
    private Tensor Norm(Tensor x, string name) => x * (x.pow(2).mean(new long[] { -1 }, keepdim: true) + 1e-5).rsqrt() * _weights[name];
    private Tensor Linear(Tensor x, string name) => x.matmul(Quantized(name).transpose(0, 1));
    private Tensor Rotate(Tensor x, int heads)
    {
        long b = x.shape[0], t = x.shape[1]; int hd = _c.HeadDimension;
        var pairs = x.reshape(b, t, heads, hd / 2, 2);
        var even = pairs.select(-1, 0); var odd = pairs.select(-1, 1);
        var c = _cos.narrow(1, 0, t); var s = _sin.narrow(1, 0, t);
        return stack(new[] { even * c - odd * s, even * s + odd * c }, -1).reshape(b, t, heads, hd);
    }
    public Tensor Forward(Tensor ids, Tensor? targetPositions = null)
    {
        using var scope = NewDisposeScope();
        long b = ids.shape[0], t = ids.shape[1];
        if (t > _c.Context) throw new ArgumentException("Training sequence exceeds context.");
        var embedding = Quantized("embedding");
        var x = embedding.index_select(0, ids.reshape(-1)).reshape(b, t, _c.Dimension);
        for (int l = 0; l < _c.Layers; l++)
        {
            using var layerScope = NewDisposeScope();
            string p = $"block.{l}.";
            var norm = Norm(x, p + "norm1");
            var q = Rotate(Linear(norm, p + "q"), _c.Heads).transpose(1, 2);
            var k = Rotate(Linear(norm, p + "k"), _c.KvHeads);
            var v = Linear(norm, p + "v").reshape(b, t, _c.KvHeads, _c.HeadDimension);
            int group = _c.Heads / _c.KvHeads;
            k = k.unsqueeze(3).expand(b, t, _c.KvHeads, group, _c.HeadDimension).reshape(b, t, _c.Heads, _c.HeadDimension).transpose(1, 2);
            v = v.unsqueeze(3).expand(b, t, _c.KvHeads, group, _c.HeadDimension).reshape(b, t, _c.Heads, _c.HeadDimension).transpose(1, 2);
            Tensor attended;
            if (_useSdpa)
                // Dropout is explicitly zero; all three sequence lengths match. GQA heads are expanded above.
                // LibTorch selects its supported implementation; this is NOT a claim that FlashAttention is active.
                attended = nn.functional.scaled_dot_product_attention(q, k, v, null, 0.0, true);
            else
            {
                var scores = q.matmul(k.transpose(-1, -2)) / Math.Sqrt(_c.HeadDimension);
                scores = scores.masked_fill(_causal.narrow(0, 0, t).narrow(1, 0, t), double.NegativeInfinity);
                attended = scores.softmax(-1).matmul(v);
            }
            var attn = attended.transpose(1, 2).contiguous().reshape(b, t, _c.Dimension);
            x = x + Linear(attn, p + "out");
            norm = Norm(x, p + "norm2");
            var gate = Linear(norm, p + "gate");
            x = (x + Linear(nn.functional.silu(gate) * Linear(norm, p + "up"), p + "down")).MoveToOuterDisposeScope();
        }
        // Only the selected positions reach RMSNorm/output projection when computing a masked training loss.
        // Attention still sees the whole causal input. The full-logit API remains the reference/test path.
        if (targetPositions is not null) x = x.reshape(b * t, _c.Dimension).index_select(0, targetPositions);
        return Norm(x, "norm").matmul(embedding.transpose(0, 1)).MoveToOuterDisposeScope();
    }
    public WeightSet CopyMaster(CancellationToken ct = default)
    {
        using var scope = NewDisposeScope(); var result = new Dictionary<string, float[]>();
        foreach (var s in _layout)
        {
            ct.ThrowIfCancellationRequested(); using var tensorScope = NewDisposeScope();
            result[s.Name] = _weights[s.Name].detach().cpu().contiguous().data<float>().ToArray();
        }
        return new(_c, result);
    }
    public void Restore(WeightSet weights)
    {
        if (_evaluationWeights is not null) throw new InvalidOperationException("Cannot restore during evaluation.");
        WeightVersion++;
        using var scope = NewDisposeScope(); using var noGrad = no_grad();
        foreach (var s in _layout)
        {
            using var tensorScope = NewDisposeScope();
            _weights[s.Name].copy_(tensor(weights.Values[s.Name], dtype: ScalarType.Float32, device: _device).reshape(s.Rows, s.Cols));
        }
    }
    public void MarkUpdated()
    {
        if (_evaluationWeights is not null) throw new InvalidOperationException("Cannot update weights during evaluation.");
        WeightVersion++;
    }
    public IDisposable BeginEvaluation()
    {
        if (_evaluationWeights is not null) throw new InvalidOperationException("Evaluation cache is already active.");
        _evaluationWeights = new(StringComparer.Ordinal);
        return new EvaluationLease(this);
    }
    private void EndEvaluation()
    {
        if (_evaluationWeights is null) return;
        foreach (var tensor in _evaluationWeights.Values) tensor.Dispose();
        _evaluationWeights = null;
    }
    private sealed class EvaluationLease(TorchModel model) : IDisposable
    {
        private TorchModel? _owner = model;
        public void Dispose() => Interlocked.Exchange(ref _owner, null)?.EndEvaluation();
    }
    public void Dispose()
    {
        EndEvaluation();
        foreach (var p in _weights.Values) p.Dispose(); _cos.Dispose(); _sin.Dispose(); _causal.Dispose();
    }
}
