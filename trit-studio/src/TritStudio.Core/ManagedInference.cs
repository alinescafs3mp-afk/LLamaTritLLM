using System.Numerics;
using System.Text;
namespace TritStudio.Core;

// Portable inference: no TorchSharp, CUDA, unsafe pointers, or architecture-specific intrinsics.
// The active WeightSet is never mutated. Each response owns its KV cache and scratch buffers.
public sealed class ManagedInference
{
    public WeightSet Weights { get; }
    public long Revision { get; }
    private readonly int _threads;
    private readonly Lazy<RopeTable> _rope;
    private readonly LayerWeights[] _layers;
    public ManagedInference(WeightSet quantizedWeights, long revision, int threads = 1)
    {
        Weights = quantizedWeights; Revision = revision; _threads = Math.Max(1, threads);
        _layers = Enumerable.Range(0, Weights.Config.Layers).Select(i =>
        {
            string prefix = $"block.{i}."; var w = Weights.Values;
            return new LayerWeights(w[prefix + "norm1"], w[prefix + "q"], w[prefix + "k"], w[prefix + "v"],
                w[prefix + "out"], w[prefix + "norm2"], w[prefix + "gate"], w[prefix + "up"], w[prefix + "down"]);
        }).ToArray();
        _rope = new(() => new RopeTable(Weights.Config), LazyThreadSafetyMode.ExecutionAndPublication);
    }
    public Session NewSession() => new(Weights, _threads, _rope.Value, _layers);

    public string Generate(int[] prompt, Func<SamplingOptions> sampling, Action<string>? onText, CancellationToken ct)
        => GenerateDetailed(prompt, sampling, onText, ct).Text;
    public GenerationResult GenerateDetailed(int[] prompt, Func<SamplingOptions> sampling, Action<string>? onText, CancellationToken ct)
    {
        if (prompt.Length == 0 || prompt.Length >= Weights.Config.Context) throw new ArgumentException("Invalid prompt length.");
        ct.ThrowIfCancellationRequested();
        var session = NewSession(); var logits = new float[ByteTokenizer.VocabularySize];
        for (int i = 0; i < prompt.Length; i++) { ct.ThrowIfCancellationRequested(); session.Step(prompt[i], ct, i == prompt.Length - 1, logits); }
        var sampler = new TokenSampler();
        // MaxNewTokens is clamped to 1024; reserve the whole response once, including live limit increases.
        var output = new byte[Math.Min(1024, Weights.Config.Context - prompt.Length)]; int outputCount = 0;
        var seen = new HashSet<int>(prompt); var rng = new Random();
        var lastUpdate = System.Diagnostics.Stopwatch.StartNew(); var utf8 = new Utf8Guard(); int completeLength = 0; string stopReason = "context_limit";
        bool advance = false; int lastToken = -1, remaining = 0;
        Func<int, bool> allowed = id => utf8.AllowsWithinBudget(id, remaining);
        while (session.Position < Weights.Config.Context)
        {
            ct.ThrowIfCancellationRequested(); var options = sampling().Clamp();
            if (outputCount >= options.MaxNewTokens) { stopReason = utf8.Complete ? "token_limit" : "utf8_boundary"; break; }
            if (advance)
            {
                if (session.Position >= Weights.Config.Context - 1) { stopReason = "context_limit"; break; }
                // Do not compute an unused next-token prediction after the response limit was reached.
                session.Step(lastToken, ct, true, logits);
            }
            remaining = Math.Min(options.MaxNewTokens - outputCount, Weights.Config.Context - session.Position);
            if (remaining < utf8.RemainingBytes) { stopReason = "utf8_boundary"; break; }
            int id = sampler.Sample(logits, seen, options, rng, allowed);
            if (id == ByteTokenizer.Eos) { stopReason = "eos"; break; }
            output[outputCount++] = checked((byte)(id - ByteTokenizer.Offset)); seen.Add(id); utf8.Accept(id); if (utf8.Complete) completeLength = outputCount;
            if (lastUpdate.ElapsedMilliseconds >= 40)
            { onText?.Invoke(Encoding.UTF8.GetString(output, 0, completeLength)); lastUpdate.Restart(); }
            advance = true; lastToken = id;
        }
        var answer = Encoding.UTF8.GetString(output, 0, completeLength); onText?.Invoke(answer);
        return new(answer, outputCount, stopReason, session.Position);
    }
    public static int Sample(float[] source, HashSet<int> seen, SamplingOptions o, Random rng, Func<int, bool>? allowed = null)
        => new TokenSampler().Sample(source, seen, o, rng, allowed);

    // Bound once per immutable snapshot, instead of constructing names and doing dictionary lookups per token/layer.
    internal sealed record LayerWeights(float[] Norm1, float[] Q, float[] K, float[] V, float[] Out,
        float[] Norm2, float[] Gate, float[] Up, float[] Down);

    // Shared immutable position tables, independent of the conversational KV cache.
    internal sealed class RopeTable
    {
        public readonly float[] Cos, Sin;
        public RopeTable(ModelConfig c)
        {
            Cos = new float[c.Context * c.HeadDimension / 2]; Sin = new float[Cos.Length];
            // The denominator depends on the channel only. Keep division (not reciprocal multiplication)
            // to preserve the prior floating-point evaluation order at every position.
            var denominators = new float[c.HeadDimension / 2];
            for (int d = 0; d < denominators.Length; d++) denominators[d] = MathF.Pow(c.RopeTheta, 2f * d / c.HeadDimension);
            for (int p = 0; p < c.Context; p++) for (int d = 0; d < c.HeadDimension; d += 2)
            {
                float angle = p / denominators[d / 2];
                int i = p * (c.HeadDimension / 2) + d / 2; Cos[i] = MathF.Cos(angle); Sin[i] = MathF.Sin(angle);
            }
        }
    }

    public sealed class Session
    {
        private readonly LayerWeights[] _layers;
        private readonly float[] _embedding, _finalNorm;
        private readonly ModelConfig _c;
        private readonly ParallelOptions _parallel;
        private readonly float[][] _keys, _values;
        private readonly float[] _x, _normed, _q, _k, _v, _attention, _projected, _gate, _up, _hidden;
        private readonly float[] _ropeCos, _ropeSin, _scores;
        public int Position { get; private set; }
        internal Session(WeightSet weights, int threads, RopeTable rope, LayerWeights[] layers)
        {
            _layers = layers; _embedding = weights.Values["embedding"]; _finalNorm = weights.Values["norm"]; _c = weights.Config; _scores = new float[_c.Context];
            _parallel = new ParallelOptions { MaxDegreeOfParallelism = threads };
            _keys = Enumerable.Range(0, _c.Layers).Select(_ => new float[_c.Context * _c.KvDimension]).ToArray();
            _values = Enumerable.Range(0, _c.Layers).Select(_ => new float[_c.Context * _c.KvDimension]).ToArray();
            _x = new float[_c.Dimension]; _normed = new float[_c.Dimension]; _q = new float[_c.Dimension];
            _k = new float[_c.KvDimension]; _v = new float[_c.KvDimension]; _attention = new float[_c.Dimension];
            _projected = new float[_c.Dimension]; _gate = new float[_c.HiddenDimension];
            _up = new float[_c.HiddenDimension]; _hidden = new float[_c.HiddenDimension];
            _ropeCos = rope.Cos; _ropeSin = rope.Sin;
        }
        public float[] Step(int token, CancellationToken ct = default, bool computeLogits = true, float[]? logitsBuffer = null)
        {
            if (token < 0 || token >= ByteTokenizer.VocabularySize || Position >= _c.Context) throw new ArgumentOutOfRangeException(nameof(token));
            if (logitsBuffer is not null && logitsBuffer.Length != ByteTokenizer.VocabularySize) throw new ArgumentException("Invalid logits buffer.");
            ct.ThrowIfCancellationRequested(); _parallel.CancellationToken = ct;
            Array.Copy(_embedding, token * _c.Dimension, _x, 0, _c.Dimension);
            for (int l = 0; l < _c.Layers; l++)
            {
                ct.ThrowIfCancellationRequested(); var layer = _layers[l];
                CpuElementwise.RmsNorm(_x, layer.Norm1, _normed, ct);
                CpuProjection.Triple(layer.Q, layer.K, layer.V, _normed, _q, _k, _v, _parallel);
                Rotate(_q, _c.Heads); Rotate(_k, _c.KvHeads);
                Array.Copy(_k, 0, _keys[l], Position * _c.KvDimension, _k.Length);
                Array.Copy(_v, 0, _values[l], Position * _c.KvDimension, _v.Length);
                Attend(l, ct);
                MatVec(layer.Out, _attention, _projected); CpuElementwise.AddInPlace(_x, _projected, ct);
                CpuElementwise.RmsNorm(_x, layer.Norm2, _normed, ct);
                CpuProjection.Pair(layer.Gate, layer.Up, _normed, _gate, _up, _parallel);
                for (int i = 0; i < _hidden.Length; i++) _hidden[i] = _gate[i] / (1 + MathF.Exp(-_gate[i])) * _up[i];
                MatVec(layer.Down, _hidden, _projected); CpuElementwise.AddInPlace(_x, _projected, ct);
            }
            if (!computeLogits) { Position++; return Array.Empty<float>(); }
            CpuElementwise.RmsNorm(_x, _finalNorm, _normed, ct);
            var logits = logitsBuffer ?? new float[ByteTokenizer.VocabularySize]; MatVec(_embedding, _normed, logits);
            Position++; return logits;
        }
        private void Rotate(float[] a, int heads)
        {
            for (int h = 0; h < heads; h++) for (int d = 0; d < _c.HeadDimension; d += 2)
            {
                int i = h * _c.HeadDimension + d, r = Position * (_c.HeadDimension / 2) + d / 2;
                float x = a[i], y = a[i + 1], cos = _ropeCos[r], sin = _ropeSin[r];
                a[i] = x * cos - y * sin; a[i + 1] = x * sin + y * cos;
            }
        }
        private void Attend(int layer, CancellationToken ct) => CpuAttention.Compute(
            _q, _keys[layer], _values[layer], _scores, _attention, Position + 1,
            _c.Heads, _c.KvHeads, _c.HeadDimension, ct);
        private void MatVec(float[] matrix, float[] input, float[] output) => CpuProjection.Multiply(matrix, input, output, _parallel);
        public static float Dot(float[] a, int ai, float[] b, int bi, int n)
        {
            int width = Vector<float>.Count, j = 0; var sum = Vector<float>.Zero;
            for (; j <= n - width; j += width) sum += new Vector<float>(a, ai + j) * new Vector<float>(b, bi + j);
            float value = Vector.Sum(sum); for (; j < n; j++) value += a[ai + j] * b[bi + j]; return value;
        }

    }
}

public sealed record GenerationResult(string Text, int GeneratedByteTokens, string StopReason, int ForwardSteps = 0);
