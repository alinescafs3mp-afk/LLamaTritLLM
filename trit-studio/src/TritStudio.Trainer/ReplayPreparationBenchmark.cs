using System.Diagnostics;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Managed CPU preparation only. Both paths build identical input/label arrays before any gradient step.
internal static class ReplayPreparationBenchmark
{
    public static object Run(bool reverse)
    {
        const int sequence = 256;
        var baseline = Dataset.EncodeAll(Enumerable.Range(0, 256).Select(i => Dataset.Make("base-" + i, "answer")).ToArray(), sequence, 1);
        var learned = Enumerable.Range(0, 64).Select(i => Dataset.Make("remember-" + i, "retained correction",
            history: new[] { new ChatTurn("Earlier detail", "Acknowledged", 0) })).ToArray();
        var recent = Dataset.EncodeAll(Enumerable.Range(0, 4).Select(i => Dataset.Make("new-" + i, "correction")).ToArray(), sequence, 1);
        var guard = new ValidationGuard(new[] { Dataset.Make("reserved-control-input", "reference") }, sequence);
        long[] steps = [0, 254, 256, 319]; var trials = new List<object>();
        EncodedExample[] Legacy(long step)
        {
            guard.EnsureTraining(learned);
            var all = Dataset.EncodeAll(learned, sequence, 1);
            var pool = new EncodedExample[8]; Array.Copy(recent, pool, 4);
            for (int i = 0; i < 4; i++)
            {
                int index = (int)(((ulong)step + (uint)i) % (uint)(baseline.Length + all.Length));
                pool[4 + i] = index < baseline.Length ? baseline[index] : all[index - baseline.Length];
            }
            return pool;
        }
        foreach (long step in steps)
        {
            var expected = Legacy(step); var actual = OnlineReplay.Build(baseline, recent, learned, step, sequence, guard);
            for (int i = 0; i < expected.Length; i++)
                if (!expected[i].Tokens.SequenceEqual(actual.Examples[i].Tokens) || !expected[i].Labels.SequenceEqual(actual.Examples[i].Labels))
                    throw new InvalidOperationException("Replay preparation changed the training pool.");
            foreach (bool lazy in reverse ? new[] { true, false } : new[] { false, true })
            {
                EncodedExample[] RunOnce() => lazy ? OnlineReplay.Build(baseline, recent, learned, step, sequence, guard).Examples : Legacy(step);
                for (int i = 0; i < 3; i++) _ = RunOnce();
                const int repetitions = 32;
                long begin = GC.GetAllocatedBytesForCurrentThread(); var clock = Stopwatch.StartNew(); long positions = 0;
                for (int i = 0; i < repetitions; i++) foreach (var example in RunOnce()) positions += example.Tokens.Length;
                clock.Stop(); long allocated = GC.GetAllocatedBytesForCurrentThread() - begin;
                trials.Add(new { step, mode = lazy ? "selected-replay-only" : "v10-encode-all-64", repetitions,
                    milliseconds = clock.Elapsed.TotalMilliseconds, managedAllocatedBytes = allocated, positions,
                    learnedEncodesPerPool = lazy ? actual.LearnedExamplesEncoded : learned.Length });
            }
        }
        return new { scope = "Actual C# managed replay preparation, not total training speed or peak RSS/VRAM. Parity checked outside timing.", trials };
    }
}
