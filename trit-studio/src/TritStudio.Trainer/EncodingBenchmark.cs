using System.Diagnostics;
using System.Text;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Actual C# measurement of CPU preparation only. No user files, model updates or timing pass ratio.
internal static class EncodingBenchmark
{
    private static readonly UTF8Encoding LegacyUtf8 = new(false, true);
    public static object Run(bool reverse)
    {
        var examples = Enumerable.Range(0, 128).Select(i => i % 3 == 0
            ? Dataset.Make("Текст " + i + ": спокойный вечер и открытая книга.")
            : Dataset.Make("Что выбрали " + i + "?", "Зелёный вариант 🙂", history: i % 3 == 1 ? null :
                new[] { new ChatTurn("Сначала был синий.", "Понял.", 0), new ChatTurn("Теперь зелёный.", "Выбор изменён.", 0) })).ToArray();
        foreach (var example in examples)
        {
            var expected = Legacy(example, 512); var actual = Dataset.Encode(example, 512);
            if (!expected.Tokens.SequenceEqual(actual.Tokens) || !expected.Labels.SequenceEqual(actual.Labels))
                throw new InvalidOperationException("Encoding benchmark equivalence check failed.");
        }
        var trials = new List<object>();
        foreach (bool direct in reverse ? new[] { true, false } : new[] { false, true })
        {
            for (int warmup = 0; warmup < 3; warmup++) foreach (var row in examples) _ = Encode(row, direct);
            long startBytes = GC.GetAllocatedBytesForCurrentThread(); var watch = Stopwatch.StartNew(); long positions = 0, targets = 0;
            const int rounds = 16;
            for (int round = 0; round < rounds; round++) foreach (var row in examples)
            {
                var encoded = Encode(row, direct); positions += encoded.Tokens.Length;
                foreach (int label in encoded.Labels) if (label != -100) targets++;
            }
            watch.Stop(); long allocated = GC.GetAllocatedBytesForCurrentThread() - startBytes;
            trials.Add(new { mode = direct ? "direct-final-buffers" : "v9-list-reference", examples = examples.Length * rounds,
                milliseconds = watch.Elapsed.TotalMilliseconds, allocatedBytes = allocated, positions, supervisedTargets = targets });
        }
        return new { scope = "Actual single-thread C# encoding, equal input/label arrays; not native training throughput or peak memory", trials };
    }
    private static EncodedExample Encode(TrainingExample row, bool direct) => direct ? Dataset.Encode(row, 512) : Legacy(row, 512);
    private static int[] Bytes(string text) => LegacyUtf8.GetBytes(text).Select(x => (int)x + ByteTokenizer.Offset).ToArray();
    private static EncodedExample Legacy(TrainingExample row, int length)
    {
        if (length < 8 || length > 2048) throw new ArgumentOutOfRangeException(nameof(length));
        var ids = new List<int> { ByteTokenizer.Bos }; int answerStart = 1;
        if (row.IsDialogue)
        {
            var h = row.History ?? [];
            for (int i = Dataset.RetainedHistoryStart(row, length); i < h.Length; i++)
            {
                ids.Add(ByteTokenizer.User); ids.AddRange(Bytes(h[i].User)); ids.Add(ByteTokenizer.Eos);
                ids.Add(ByteTokenizer.Assistant); ids.AddRange(Bytes(h[i].Assistant)); ids.Add(ByteTokenizer.Eos);
            }
            ids.Add(ByteTokenizer.User); ids.AddRange(Bytes(row.Text)); ids.Add(ByteTokenizer.Eos);
            ids.Add(ByteTokenizer.Assistant); answerStart = ids.Count; ids.AddRange(Bytes(row.Answer!)); ids.Add(ByteTokenizer.Eos);
        }
        else
        {
            ids.AddRange(Bytes(row.Text)); ids.Add(ByteTokenizer.Eos);
            if (ids.Count > length + 1) throw new ArgumentException("Text must be split before encoding; silent truncation is disabled.");
        }
        int n = ids.Count - 1; var input = ids.Take(n).ToArray(); var labels = Enumerable.Repeat(-100, n).ToArray();
        for (int i = 0; i < n; i++) if (i + 1 >= answerStart) labels[i] = ids[i + 1];
        return new(input, labels, row.Id);
    }
}
