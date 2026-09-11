//https://github.com/virex-84

//TernaryLlamaModel.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TernaryLLM;

public sealed class TernaryLlamaModel
{
    private readonly TernaryLlamaConfig _config;

    private readonly int _vocab;
    private readonly int _emb;
    private readonly int _maxSeq;

    private readonly MultiPlaneEmbedding _embedding;
    private readonly TernaryLlamaBlock[] _blocks;

    private float[] _finalNorm;

    private readonly MultiPlaneLinear? _lmHead;
    private readonly BPETokenizer? _tokenizer;

    public TernaryLlamaConfig Config => _config;

    public TernaryLlamaModel(TernaryLlamaConfig config, BPETokenizer? tokenizer = null)
    {
        config.Validate();

        _config = config;
        _tokenizer = tokenizer;

        _vocab = config.VocabSize;
        _emb = config.EmbeddingDim;
        _maxSeq = config.MaxSeqLen;

        var embData = CreateRandomData(_vocab, _emb, 0.02f);

        var embPacker = MultiPlaneTritPacker.FromFloat(
            embData,
            _vocab,
            _emb,
            config.EmbeddingPlanes,
            config.GroupSize,
            config.ScaleMode);

        _embedding = new MultiPlaneEmbedding(embPacker);

        _blocks = new TernaryLlamaBlock[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new TernaryLlamaBlock(
                _emb,
                config.NumHeads,
                config.NumKVHeads,
                config.HiddenDim,
                config.AttentionQkvPlanes,
                config.AttentionOutPlanes,
                config.FfnGatePlanes,
                config.FfnUpPlanes,
                config.FfnDownPlanes,
                config.ActivationPlanes,
                config.GroupSize,
                config.ScaleMode,
                config.AttentionMode,
                config.TopK,
                config.QuantizeActivations,
                config.RopeTheta);
        }

        _finalNorm = new float[_emb];
        Array.Fill(_finalNorm, 1f);

        if (!config.TiedOutput)
        {
            var lmData = CreateRandomData(_vocab, _emb, 0.02f);

            var lmPacker = MultiPlaneTritPacker.FromFloat(
                lmData,
                _vocab,
                _emb,
                config.LmHeadPlanes,
                config.GroupSize,
                config.ScaleMode);

            _lmHead = new MultiPlaneLinear(lmPacker, useBias: false);
        }
    }

    private static float[] CreateRandomData(int rows, int cols, float std)
    {
        var data = new float[rows * cols];
        var rng = new Random(42);

        for (int i = 0; i < data.Length; i++)
        {
            double u1 = rng.NextDouble() + 1e-10;
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = (float)(std * z);
        }

        return data;
    }

    public LlamaKVCache[] CreateCaches()
    {
        var caches = new LlamaKVCache[_blocks.Length];

        for (int i = 0; i < caches.Length; i++)
        {
            caches[i] = new LlamaKVCache(_maxSeq, _config.KVDim);
        }

        return caches;
    }

    public float[] Forward(int[] tokens, LlamaKVCache[] caches)
    {
        if (tokens == null || tokens.Length == 0)
            return Array.Empty<float>();

        if (caches == null)
            throw new ArgumentNullException(nameof(caches));

        if (caches.Length != _blocks.Length)
            throw new ArgumentException("caches.Length != blocks.Length");

        int seq = tokens.Length;

        if (caches.Length > 0 && caches[0].Length + seq > _maxSeq)
        {
            throw new InvalidOperationException(
                $"Sequence exceeds MaxSeqLen: cache={caches[0].Length}, seq={seq}, max={_maxSeq}.");
        }

        float[] hidden = _embedding.ForwardSequence(tokens);

        if (_config.QuantizeActivations)
        {
            hidden = TernaryOps.Requantize(
                hidden,
                seq,
                _emb,
                _config.ActivationPlanes,
                _config.GroupSize,
                _config.ScaleMode);
        }

        for (int i = 0; i < _blocks.Length; i++)
        {
            hidden = _blocks[i].Forward(hidden, seq, caches[i]);
        }

        hidden = TernaryOps.RmsNorm(hidden, seq, _emb, _finalNorm);

        int lastOffset = (seq - 1) * _emb;
        var last = hidden.AsSpan(lastOffset, _emb).ToArray();

        float[] logits = new float[_vocab];

        if (_config.TiedOutput)
        {
            for (int v = 0; v < _vocab; v++)
            {
                logits[v] = _embedding.DotWithRow(v, last);
            }
        }
        else
        {
            logits = _lmHead!.Forward(last);
        }

        return logits;
    }

    public string Predict(string userInput, float temperature = 0.7f)
    {
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer is not set.");

        string prompt = _tokenizer.FormatDialogue(userInput);
        return PredictRaw(prompt, temperature);
    }

    public string PredictRaw(string prompt, float temperature = 0.7f)
    {
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer is not set.");

        var tokenized = _tokenizer.Encode(prompt, addBos: true, addEos: false);

        if (tokenized.Count == 0)
            return string.Empty;

        int maxPrompt = Math.Max(1, _maxSeq - 10);

        if (tokenized.Count > maxPrompt)
            tokenized = tokenized.Take(maxPrompt).ToList();

        var caches = CreateCaches();

        float[] logits = Forward(tokenized.ToArray(), caches);

        var output = new List<int>();

        int contextLen = tokenized.Count;

        while (contextLen < _maxSeq - 1)
        {
            MaskLogits(logits);

            int nextId = temperature <= 0.05f
                ? ArgMax(logits)
                : SampleTopK(logits, temperature, topK: 40);

            if (nextId < 0 || nextId >= _vocab)
                break;

            if (nextId == _tokenizer.EosId)
                break;

            if (nextId == _tokenizer.PadId)
                break;

            output.Add(nextId);

            logits = Forward(new[] { nextId }, caches);
            contextLen++;
        }

        return _tokenizer.Decode(output);
    }

    private void MaskLogits(float[] logits)
    {
        if (_tokenizer == null)
            return;

        int validVocab = Math.Min(_vocab, _tokenizer.VocabSize);

        for (int i = validVocab; i < logits.Length; i++)
        {
            logits[i] = float.NegativeInfinity;
        }

        if (_tokenizer.PadId >= 0 && _tokenizer.PadId < logits.Length)
        {
            logits[_tokenizer.PadId] = float.NegativeInfinity;
        }
    }

    public int ArgMax(float[] logits)
    {
        int best = 0;

        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
                best = i;
        }

        return best;
    }

    private static int SampleTopK(float[] logits, float temperature, int topK)
    {
        if (logits == null || logits.Length == 0)
            return 0;

        temperature = Math.Clamp(temperature, 0.01f, 2.0f);
        int k = Math.Clamp(topK, 1, logits.Length);

        var indices = new int[logits.Length];

        for (int i = 0; i < indices.Length; i++)
            indices[i] = i;

        Array.Sort(indices, (a, b) => logits[b].CompareTo(logits[a]));

        float maxLogit = logits[indices[0]] / temperature;

        float[] probs = new float[k];
        float sum = 0f;

        for (int i = 0; i < k; i++)
        {
            float val = MathF.Exp(logits[indices[i]] / temperature - maxLogit);
            probs[i] = val;
            sum += val;
        }

        if (sum <= 1e-12f)
            return indices[0];

        float r = (float)Random.Shared.NextDouble();
        float cumulative = 0f;

        for (int i = 0; i < k; i++)
        {
            cumulative += probs[i] / sum;

            if (r <= cumulative)
                return indices[i];
        }

        return indices[k - 1];
    }

    public long TotalParameters()
    {
        long vocab = _config.VocabSize;
        long emb = _config.EmbeddingDim;
        long hidden = _config.HiddenDim;
        long layers = _config.NumLayers;
        long kvDim = _config.KVDim;

        long embWeights = vocab * emb;

        long qWeights = emb * emb;
        long kWeights = kvDim * emb;
        long vWeights = kvDim * emb;
        long attnOutWeights = emb * emb;

        long ffnGateWeights = hidden * emb;
        long ffnUpWeights = hidden * emb;
        long ffnDownWeights = emb * hidden;

        long lmHeadWeights = _config.TiedOutput ? 0L : vocab * emb;

        long weightsPerLayer =
            qWeights +
            kWeights +
            vWeights +
            attnOutWeights +
            ffnGateWeights +
            ffnUpWeights +
            ffnDownWeights;

        return embWeights + layers * weightsPerLayer + lmHeadWeights;
    }

    public void ImportWeights(
        MultiPlaneTritPacker embPacker,
        TernaryLayerWeights[] layers,
        float[] finalNormGamma,
        float[]? lmHeadData)
    {
        var cfg = _config;

        _embedding.Table = embPacker;
        _finalNorm = (float[])finalNormGamma.Clone();

        for (int i = 0; i < _blocks.Length && i < layers.Length; i++)
        {
            var lw = layers[i];
            var ib = _blocks[i];

            Array.Copy(lw.Norm1, ib.Norm1, _emb);
            Array.Copy(lw.Norm2, ib.Norm2, _emb);

            ib.Attention.Q.Weight = MultiPlaneTritPacker.FromFloat(
                lw.Q,
                _emb,
                _emb,
                cfg.AttentionQkvPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Attention.K.Weight = MultiPlaneTritPacker.FromFloat(
                lw.K,
                cfg.KVDim,
                _emb,
                cfg.AttentionQkvPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Attention.V.Weight = MultiPlaneTritPacker.FromFloat(
                lw.V,
                cfg.KVDim,
                _emb,
                cfg.AttentionQkvPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Attention.Out.Weight = MultiPlaneTritPacker.FromFloat(
                lw.Out,
                _emb,
                _emb,
                cfg.AttentionOutPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Ffn.Gate.Weight = MultiPlaneTritPacker.FromFloat(
                lw.Gate,
                cfg.HiddenDim,
                _emb,
                cfg.FfnGatePlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Ffn.Up.Weight = MultiPlaneTritPacker.FromFloat(
                lw.Up,
                cfg.HiddenDim,
                _emb,
                cfg.FfnUpPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);

            ib.Ffn.Down.Weight = MultiPlaneTritPacker.FromFloat(
                lw.Down,
                _emb,
                cfg.HiddenDim,
                cfg.FfnDownPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);
        }

        if (!cfg.TiedOutput && _lmHead != null && lmHeadData != null)
        {
            _lmHead.Weight = MultiPlaneTritPacker.FromFloat(
                lmHeadData,
                cfg.VocabSize,
                _emb,
                cfg.LmHeadPlanes,
                cfg.GroupSize,
                cfg.ScaleMode);
        }
    }

    public void Save(string path)
    {
        if (_tokenizer == null)
            throw new InvalidOperationException("Tokenizer is not set.");

        string? dir = Path.GetDirectoryName(path);

        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        using var stream = File.Open(path, FileMode.Create);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);

        writer.Write("TERNARY_LLAMA_V1");

        SaveConfig(writer, _config);

        _tokenizer.SaveToStream(writer);

        SaveToStream(writer);

        writer.Flush();
    }

    private void SaveToStream(BinaryWriter w)
    {
        _embedding.SaveToStream(w);

        TernarySerializer.WriteFloats(w, _finalNorm);

        foreach (var block in _blocks)
        {
            block.SaveToStream(w);
        }

        bool hasLmHead = _lmHead != null;
        w.Write(hasLmHead);

        if (hasLmHead)
        {
            _lmHead!.SaveToStream(w);
        }
    }

    public static TernaryLlamaModel Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}");

        using var stream = File.Open(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream, Encoding.UTF8);

        string magic = reader.ReadString();

        if (magic != "TERNARY_LLAMA_V1")
            throw new InvalidDataException($"Invalid file format: '{magic}'");

        var config = LoadConfig(reader);
        var tokenizer = BPETokenizer.LoadFromStream(reader);

        var model = new TernaryLlamaModel(config, tokenizer);

        model.LoadFromStream(reader);

        return model;
    }

    private void LoadFromStream(BinaryReader r)
    {
        _embedding.LoadFromStream(r);

        _finalNorm = TernarySerializer.ReadFloats(r);

        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadFromStream(r);
        }

        bool hasLmHead = r.ReadBoolean();

        if (hasLmHead)
        {
            if (_lmHead == null)
                throw new InvalidDataException("LMHead exists in file but model has TiedOutput=true.");

            _lmHead.LoadFromStream(r);
        }
    }

    private static void SaveConfig(BinaryWriter w, TernaryLlamaConfig config)
    {
        w.Write(config.VocabSize);
        w.Write(config.EmbeddingDim);
        w.Write(config.HiddenDim);
        w.Write(config.NumHeads);
        w.Write(config.NumKVHeads);
        w.Write(config.NumLayers);
        w.Write(config.MaxSeqLen);
        w.Write(config.RopeTheta);

        w.Write(config.EmbeddingPlanes);
        w.Write(config.AttentionQkvPlanes);
        w.Write(config.AttentionOutPlanes);

        w.Write(config.FfnGatePlanes);
        w.Write(config.FfnUpPlanes);
        w.Write(config.FfnDownPlanes);

        w.Write(config.LmHeadPlanes);
        w.Write(config.ActivationPlanes);

        w.Write(config.GroupSize);
        w.Write((int)config.ScaleMode);

        w.Write((int)config.AttentionMode);
        w.Write(config.TopK);

        w.Write(config.QuantizeActivations);
        w.Write(config.TiedOutput);
    }

    private static TernaryLlamaConfig LoadConfig(BinaryReader r)
    {
        return new TernaryLlamaConfig
        {
            VocabSize = r.ReadInt32(),
            EmbeddingDim = r.ReadInt32(),
            HiddenDim = r.ReadInt32(),
            NumHeads = r.ReadInt32(),
            NumKVHeads = r.ReadInt32(),
            NumLayers = r.ReadInt32(),
            MaxSeqLen = r.ReadInt32(),
            RopeTheta = r.ReadSingle(),

            EmbeddingPlanes = r.ReadInt32(),
            AttentionQkvPlanes = r.ReadInt32(),
            AttentionOutPlanes = r.ReadInt32(),

            FfnGatePlanes = r.ReadInt32(),
            FfnUpPlanes = r.ReadInt32(),
            FfnDownPlanes = r.ReadInt32(),

            LmHeadPlanes = r.ReadInt32(),
            ActivationPlanes = r.ReadInt32(),

            GroupSize = r.ReadInt32(),
            ScaleMode = (TritScaleMode)r.ReadInt32(),

            AttentionMode = (AttentionMode)r.ReadInt32(),
            TopK = r.ReadInt32(),

            QuantizeActivations = r.ReadBoolean(),
            TiedOutput = r.ReadBoolean()
        };
    }
}