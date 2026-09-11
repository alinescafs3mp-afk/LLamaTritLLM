//https://github.com/virex-84

//LlamaTrainer.cs

using System;
using System.Collections.Generic;
using System.Linq;

namespace TernaryLLM;

public sealed class LlamaTrainer
{
    public TrainableLlamaModel Model { get; }
    public AdamW Optimizer { get; }

    public float LabelSmoothing { get; set; } = 0.0f;
    public int Epoch { get; set; }

    public int MaxSeqLen => Model.Config.MaxSeqLen;

    public LlamaTrainer(
        TrainableLlamaModel model,
        AdamW optimizer)
    {
        Model = model;
        Optimizer = optimizer;
        Optimizer.Register(model.Parameters());
    }

    public void Train(
        List<int> tokens,
        int epochs,
        int batchSize = 4,
        bool verbose = true,
        int startEpoch = -1)
    {
        int maxSeq = MaxSeqLen;
        int baseEpoch = startEpoch >= 0 ? startEpoch : Epoch;

        var examples = new List<int[]>();

        int bos = Model.Tokenizer.BosId;
        int eos = Model.Tokenizer.EosId;

        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i] != bos)
                continue;

            int end = Math.Min(tokens.Count, i + maxSeq);

            for (int j = i + 1; j < end; j++)
            {
                if (tokens[j] == eos)
                {
                    end = j + 1;
                    break;
                }
            }

            if (end - i >= 2)
            {
                examples.Add(tokens.GetRange(i, end - i).ToArray());
            }
        }

        if (examples.Count == 0)
        {
            for (int i = 0; i + maxSeq <= tokens.Count; i++)
            {
                var slice = new int[maxSeq];

                for (int j = 0; j < maxSeq; j++)
                {
                    slice[j] = tokens[i + j];
                }

                examples.Add(slice);
            }
        }

        if (examples.Count == 0)
        {
            Console.WriteLine("  No examples for training.");
            return;
        }

        if (verbose)
        {
            Console.WriteLine($"\n  Examples: {examples.Count}");
            Console.WriteLine($"  Batch size: {batchSize}");
            Console.WriteLine($"  Epochs: {epochs}");
            Console.WriteLine($"  LR: {Optimizer.LearningRate}");
            Console.WriteLine($"  Weight decay: {Optimizer.WeightDecay}");
            Console.WriteLine();
        }

        for (int e = 0; e < epochs; e++)
        {
            int epoch = baseEpoch + e;

            var shuffled = examples
                .OrderBy(_ => Random.Shared.Next())
                .ToList();

            float epochLoss = 0;
            int epochBatches = 0;

            for (int b = 0; b < shuffled.Count; b += batchSize)
            {
                var batch = shuffled
                    .Skip(b)
                    .Take(batchSize)
                    .ToList();

                float batchLoss = TrainBatch(batch);

                epochLoss += batchLoss;
                epochBatches++;
            }

            if (verbose && (e % 5 == 0 || e == epochs - 1))
            {
                float avgLoss = epochLoss / Math.Max(1, epochBatches);
                float ppl = MathF.Exp(Math.Min(avgLoss, 20f));

                Console.WriteLine(
                    $"[Epoch {epoch + 1,3}/{baseEpoch + epochs}] " +
                    $"Loss={avgLoss:F4} PPL={ppl:F2}");
            }

            Epoch = epoch + 1;
        }
    }

    private float TrainBatch(List<int[]> batch)
    {
        foreach (var p in Model.Parameters())
        {
            p.ZeroGrad();
        }

        float totalLoss = 0;

        int vocab = Model.Config.VocabSize;

        foreach (var example in batch)
        {
            int seqLen = example.Length;

            var input = example[..^1];
            var targets = example[1..];

            var logits = Model.ForwardAllPositions(input);

            int rows = seqLen - 1;

            var targetTensor = new Tensor(
                new float[rows * vocab],
                new[] { rows, vocab });

            float smooth = LabelSmoothing / vocab;

            for (int t = 0; t < rows; t++)
            {
                int tid = targets[t];

                if (tid < 0 || tid >= vocab)
                    continue;

                for (int v = 0; v < vocab; v++)
                {
                    targetTensor.Data[t * vocab + v] = smooth;
                }

                targetTensor.Data[t * vocab + tid] =
                    1f - LabelSmoothing + smooth;
            }

            var loss = Ops.CrossEntropy(logits, targetTensor);

            totalLoss += loss.Data[0];

            loss.Backward();
        }

        Optimizer.Step();

        return totalLoss / Math.Max(1, batch.Count);
    }
}