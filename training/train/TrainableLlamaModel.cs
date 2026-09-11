//https://github.com/virex-84

//TrainableLlamaModel.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TernaryLLM;

public sealed class TrainableLlamaModel
{
    public TernaryLlamaConfig Config { get; }

    public Tensor TokenEmbedding { get; }

    public List<TrainableLlamaBlock> Blocks { get; } = new();

    public Tensor FinalNormGamma { get; }

    public Tensor? LmHead { get; }

    private readonly Random _rng = new(42);

    public BPETokenizer Tokenizer { get; }

    public int Epoch { get; set; }
    public float LabelSmoothing { get; set; }

    public TrainableLlamaModel(
        TernaryLlamaConfig config,
        BPETokenizer tokenizer)
    {
        config.Validate();

        Config = config;
        Tokenizer = tokenizer;

        if (Config.VocabSize != tokenizer.VocabSize)
        {
            Config.VocabSize = tokenizer.VocabSize;
        }

        int vocab = Config.VocabSize;
        int emb = Config.EmbeddingDim;
        int hidden = Config.HiddenDim;

        TokenEmbedding = InitNormal(vocab, emb, 0.02f, requiresGrad: true);

        for (int i = 0; i < Config.NumLayers; i++)
        {
            Blocks.Add(new TrainableLlamaBlock(
                emb,
                Config.NumHeads,
                Config.NumKVHeads,
                hidden));
        }

        FinalNormGamma = InitOnes(emb, requiresGrad: true);

        if (!Config.TiedOutput)
        {
            LmHead = InitNormal(vocab, emb, 0.02f, requiresGrad: true);
        }
    }

    public List<Tensor> Parameters()
    {
        var list = new List<Tensor>
        {
            TokenEmbedding,
            FinalNormGamma
        };

        foreach (var block in Blocks)
        {
            list.Add(block.Norm1Gamma);

            list.Add(block.QWeight);
            list.Add(block.KWeight);
            list.Add(block.VWeight);
            list.Add(block.OutWeight);

            list.Add(block.Norm2Gamma);

            list.Add(block.FfnGateWeight);
            list.Add(block.FfnUpWeight);
            list.Add(block.FfnDownWeight);
        }

        if (LmHead != null)
        {
            list.Add(LmHead);
        }

        return list;
    }

    public Tensor ForwardAllPositions(int[] tokenIds)
    {
        var x = EmbedTokens(tokenIds);

        foreach (var block in Blocks)
        {
            x = block.Forward(x, Config);
        }

        x = x.RmsNorm(FinalNormGamma);

        return ComputeLogitsAllPositions(x, tokenIds.Length);
    }

    public Tensor Forward(int[] tokenIds)
    {
        var x = EmbedTokens(tokenIds);

        foreach (var block in Blocks)
        {
            x = block.Forward(x, Config);
        }

        x = x.RmsNorm(FinalNormGamma);

        int seq = tokenIds.Length;
        int emb = Config.EmbeddingDim;

        var last = SliceLastRow(x, seq, emb);

        return ComputeLogits(last);
    }

    private Tensor EmbedTokens(int[] tokenIds)
    {
        int seq = tokenIds.Length;
        int emb = Config.EmbeddingDim;
        int vocab = Config.VocabSize;

        var packer = MultiPlaneTritPacker.FromFloat(
            TokenEmbedding.Data,
            vocab,
            emb,
            Config.EmbeddingPlanes,
            Config.GroupSize,
            Config.ScaleMode);

        var quantEmbData = packer.ToFloat();

        var xData = new float[seq * emb];

        for (int i = 0; i < seq; i++)
        {
            int tid = tokenIds[i];

            if (tid < 0 || tid >= vocab)
                continue;

            Array.Copy(
                quantEmbData,
                tid * emb,
                xData,
                i * emb,
                emb);
        }

        var x = new Tensor(
            xData,
            new[] { seq, emb },
            requiresGrad: true);

        x.AddBackward(() =>
        {
            if (x.Grad == null || TokenEmbedding.Grad == null)
                return;

            for (int i = 0; i < seq; i++)
            {
                int tid = tokenIds[i];

                if (tid < 0 || tid >= vocab)
                    continue;

                for (int d = 0; d < emb; d++)
                {
                    TokenEmbedding.Grad[tid * emb + d] += x.Grad[i * emb + d];
                }
            }
        }, TokenEmbedding);

        return x;
    }

    private Tensor ComputeLogits(Tensor last)
    {
        int vocab = Config.VocabSize;
        int emb = Config.EmbeddingDim;

        Tensor weight = LmHead ?? TokenEmbedding;

        int planes = LmHead == null
            ? Config.EmbeddingPlanes
            : Config.LmHeadPlanes;

        var quantWeight = StraightThroughQuantize(weight, planes, Config);

        return last.MatMul(TransposeWeight(quantWeight, vocab, emb));
    }

    private Tensor ComputeLogitsAllPositions(Tensor x, int seq)
    {
        int vocab = Config.VocabSize;
        int emb = Config.EmbeddingDim;

        Tensor weight = LmHead ?? TokenEmbedding;

        int planes = LmHead == null
            ? Config.EmbeddingPlanes
            : Config.LmHeadPlanes;

        var quantWeight = StraightThroughQuantize(weight, planes, Config);

        Tensor weightT = TransposeWeight(quantWeight, vocab, emb);

        return x.MatMul(weightT);
    }

    private Tensor StraightThroughQuantize(
        Tensor w,
        int planes,
        TernaryLlamaConfig config)
    {
        int rows = w.Shape[0];
        int cols = w.Shape[1];

        var packer = MultiPlaneTritPacker.FromFloat(
            w.Data,
            rows,
            cols,
            planes,
            config.GroupSize,
            config.ScaleMode);

        var quantData = packer.ToFloat();

        var result = new Tensor(
            quantData,
            w.Shape,
            requiresGrad: w.RequiresGrad);

        if (w.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (result.Grad == null || w.Grad == null)
                    return;

                for (int i = 0; i < w.Grad.Length; i++)
                {
                    w.Grad[i] += result.Grad[i];
                }
            }, w);
        }

        return result;
    }

    private static Tensor TransposeWeight(Tensor w, int rows, int cols)
    {
        var result = new Tensor(
            new float[cols * rows],
            new[] { cols, rows },
            w.RequiresGrad);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.Data[j * rows + i] = w.Data[i * cols + j];
            }
        }

        if (result.RequiresGrad)
        {
            result.AddBackward(() =>
            {
                if (result.Grad == null || w.Grad == null)
                    return;

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        w.Grad[i * cols + j] += result.Grad[j * rows + i];
                    }
                }
            }, w);
        }

        return result;
    }

    private static Tensor SliceLastRow(Tensor x, int seq, int emb)
    {
        var lastData = new float[emb];

        Array.Copy(
            x.Data,
            (seq - 1) * emb,
            lastData,
            0,
            emb);

        var last = new Tensor(
            lastData,
            new[] { 1, emb },
            requiresGrad: true);

        last.AddBackward(() =>
        {
            if (last.Grad == null || x.Grad == null)
                return;

            for (int d = 0; d < emb; d++)
            {
                x.Grad[(seq - 1) * emb + d] += last.Grad[d];
            }
        }, x);

        return last;
    }

    public void SyncToInferenceModel(TernaryLlamaModel target)
    {
        var cfg = target.Config;

        var embPacker = MultiPlaneTritPacker.FromFloat(
            TokenEmbedding.Data,
            cfg.VocabSize,
            cfg.EmbeddingDim,
            cfg.EmbeddingPlanes,
            cfg.GroupSize,
            cfg.ScaleMode);

        var layers = new TernaryLayerWeights[Blocks.Count];

        for (int i = 0; i < Blocks.Count; i++)
        {
            var tb = Blocks[i];

            layers[i] = new TernaryLayerWeights
            {
                Norm1 = tb.Norm1Gamma.Data,
                Norm2 = tb.Norm2Gamma.Data,

                Q = tb.QWeight.Data,
                K = tb.KWeight.Data,
                V = tb.VWeight.Data,
                Out = tb.OutWeight.Data,

                Gate = tb.FfnGateWeight.Data,
                Up = tb.FfnUpWeight.Data,
                Down = tb.FfnDownWeight.Data
            };
        }

        target.ImportWeights(
            embPacker,
            layers,
            FinalNormGamma.Data,
            LmHead?.Data);

        Console.WriteLine("[Sync] FP32 -> multi-plane ternary Llama: OK");
    }

    public string Predict(string userInput, float temperature = 0.7f)
    {
        return PredictRaw(Tokenizer.FormatDialogue(userInput), temperature);
    }

    public string PredictRaw(string prompt, float temperature = 0.7f)
    {
        var tokenized = Tokenizer.Encode(prompt, addBos: true, addEos: false);

        if (tokenized.Count == 0)
            return string.Empty;

        if (tokenized.Count > Config.MaxSeqLen - 10)
        {
            tokenized = tokenized
                .Take(Config.MaxSeqLen - 10)
                .ToList();
        }

        var context = tokenized.ToList();
        var output = new List<int>();

        for (int step = 0; step < Config.MaxSeqLen; step++)
        {
            if (context.Count >= Config.MaxSeqLen - 1)
                break;

            var logitsTensor = Forward(context.ToArray());
            float[] logits = logitsTensor.Data;

            MaskLogits(logits);

            int nextId = temperature <= 0.05f
                ? ArgMax(logits)
                : SampleTopK(logits, temperature, topK: 40);

            if (nextId < 0 || nextId >= Config.VocabSize)
                break;

            if (nextId == Tokenizer.EosId)
                break;

            if (nextId == Tokenizer.PadId)
                break;

            output.Add(nextId);
            context.Add(nextId);
        }

        return Tokenizer.Decode(output);
    }

    private void MaskLogits(float[] logits)
    {
        int validVocab = Math.Min(Config.VocabSize, Tokenizer.VocabSize);

        for (int i = validVocab; i < logits.Length; i++)
        {
            logits[i] = float.NegativeInfinity;
        }

        if (Tokenizer.PadId >= 0 && Tokenizer.PadId < logits.Length)
        {
            logits[Tokenizer.PadId] = float.NegativeInfinity;
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

        var indices = Enumerable
            .Range(0, logits.Length)
            .ToArray();

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
        float cum = 0f;

        for (int i = 0; i < k; i++)
        {
            cum += probs[i] / sum;

            if (r <= cum)
                return indices[i];
        }

        return indices[k - 1];
    }

    public void Train(
        List<int> tokens,
        int epochs,
        int batchSize = 4,
        float? lr = null,
        float? weightDecay = null,
        float? gradientClip = null,
        float? labelSmoothing = null,
        bool verbose = true)
    {
        var parameters = Parameters();

        var optimizer = new AdamW(lr ?? 0.001f)
        {
            WeightDecay = weightDecay ?? 0.01f,
            GradientClip = gradientClip ?? 1.0f
        };

        optimizer.Register(parameters);

        if (labelSmoothing.HasValue)
        {
            LabelSmoothing = labelSmoothing.Value;
        }

        var trainer = new LlamaTrainer(this, optimizer)
        {
            Epoch = Epoch,
            LabelSmoothing = LabelSmoothing
        };

        trainer.Train(
            tokens,
            epochs,
            batchSize: batchSize,
            verbose: verbose,
            startEpoch: Epoch);

        Epoch = trainer.Epoch;
        LabelSmoothing = trainer.LabelSmoothing;
    }

    private Tensor InitNormal(int rows, int cols, float std, bool requiresGrad)
    {
        var data = new float[rows * cols];

        for (int i = 0; i < data.Length; i++)
        {
            double u1 = _rng.NextDouble() + 1e-10;
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = (float)(std * z);
        }

        return new Tensor(data, new[] { rows, cols }, requiresGrad);
    }

    private static Tensor InitOnes(int size, bool requiresGrad)
    {
        var data = new float[size];
        Array.Fill(data, 1f);

        return new Tensor(data, new[] { size }, requiresGrad);
    }

    public void SaveWeightsOnly(string path)
    {
        string? dir = Path.GetDirectoryName(path);

        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        using var stream = File.Open(path, FileMode.Create);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);

        writer.Write("TRAINABLE_LLAMA_WEIGHTS_V1");

        SaveConfig(writer, Config);

        Tokenizer.SaveToStream(writer);

        SaveWeights(writer);

        writer.Flush();
    }

    public static TrainableLlamaModel LoadWeightsOnly(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}");

        using var stream = File.Open(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream, Encoding.UTF8);

        string magic = reader.ReadString();

        if (magic != "TRAINABLE_LLAMA_WEIGHTS_V1")
            throw new InvalidDataException($"Invalid format: '{magic}'");

        var config = LoadConfig(reader);
        var tokenizer = BPETokenizer.LoadFromStream(reader);

        var model = new TrainableLlamaModel(config, tokenizer);

        model.LoadWeights(reader);

        return model;
    }

    private void SaveWeights(BinaryWriter w)
    {
        TernarySerializer.WriteFloats(w, TokenEmbedding.Data);
        TernarySerializer.WriteFloats(w, FinalNormGamma.Data);

        w.Write(Blocks.Count);

        foreach (var block in Blocks)
        {
            TernarySerializer.WriteFloats(w, block.Norm1Gamma.Data);

            TernarySerializer.WriteFloats(w, block.QWeight.Data);
            TernarySerializer.WriteFloats(w, block.KWeight.Data);
            TernarySerializer.WriteFloats(w, block.VWeight.Data);
            TernarySerializer.WriteFloats(w, block.OutWeight.Data);

            TernarySerializer.WriteFloats(w, block.Norm2Gamma.Data);

            TernarySerializer.WriteFloats(w, block.FfnGateWeight.Data);
            TernarySerializer.WriteFloats(w, block.FfnUpWeight.Data);
            TernarySerializer.WriteFloats(w, block.FfnDownWeight.Data);
        }

        bool hasLmHead = LmHead != null;
        w.Write(hasLmHead);

        if (hasLmHead)
        {
            TernarySerializer.WriteFloats(w, LmHead!.Data);
        }
    }

    private void LoadWeights(BinaryReader r)
    {
        ReadTensor(r, TokenEmbedding, nameof(TokenEmbedding));
        ReadTensor(r, FinalNormGamma, nameof(FinalNormGamma));

        int blockCount = r.ReadInt32();

        if (blockCount != Blocks.Count)
        {
            throw new InvalidDataException(
                $"Invalid block count: expected {Blocks.Count}, got {blockCount}.");
        }

        for (int i = 0; i < blockCount; i++)
        {
            var block = Blocks[i];

            ReadTensor(r, block.Norm1Gamma, $"Blocks[{i}].Norm1Gamma");

            ReadTensor(r, block.QWeight, $"Blocks[{i}].QWeight");
            ReadTensor(r, block.KWeight, $"Blocks[{i}].KWeight");
            ReadTensor(r, block.VWeight, $"Blocks[{i}].VWeight");
            ReadTensor(r, block.OutWeight, $"Blocks[{i}].OutWeight");

            ReadTensor(r, block.Norm2Gamma, $"Blocks[{i}].Norm2Gamma");

            ReadTensor(r, block.FfnGateWeight, $"Blocks[{i}].FfnGateWeight");
            ReadTensor(r, block.FfnUpWeight, $"Blocks[{i}].FfnUpWeight");
            ReadTensor(r, block.FfnDownWeight, $"Blocks[{i}].FfnDownWeight");
        }

        bool hasLmHead = r.ReadBoolean();

        if (hasLmHead)
        {
            if (LmHead == null)
            {
                throw new InvalidDataException(
                    "LM head exists in file but model was created with TiedOutput=true.");
            }

            ReadTensor(r, LmHead, nameof(LmHead));
        }
        else
        {
            if (LmHead != null)
            {
                throw new InvalidDataException(
                    "LM head missing in file but model was created with TiedOutput=false.");
            }
        }

        foreach (var p in Parameters())
        {
            p.ZeroGrad();
        }
    }

    private static void ReadTensor(BinaryReader r, Tensor tensor, string name)
    {
        float[] data = TernarySerializer.ReadFloats(r);

        if (data.Length != tensor.Data.Length)
        {
            throw new InvalidDataException(
                $"Invalid tensor size for {name}: " +
                $"expected {tensor.Data.Length}, got {data.Length}.");
        }

        Array.Copy(data, tensor.Data, data.Length);

        tensor.ZeroGrad();
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

    public long TotalParameters()
    {
        long vocab = Config.VocabSize;
        long emb = Config.EmbeddingDim;
        long hidden = Config.HiddenDim;
        long layers = Config.NumLayers;
        long kvDim = Config.KVDim;

        long embWeights = vocab * emb;

        long qWeights = emb * emb;
        long kWeights = kvDim * emb;
        long vWeights = kvDim * emb;
        long attnOutWeights = emb * emb;

        long ffnGateWeights = hidden * emb;
        long ffnUpWeights = hidden * emb;
        long ffnDownWeights = emb * hidden;

        long lmHeadWeights = Config.TiedOutput ? 0L : vocab * emb;

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
}