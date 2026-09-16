using System.Text;
using TritStudio.Core;

internal static class Audit10Checks
{
    public static readonly (string Name, Action Run)[] All =
    [
        ("direct UTF8 encoding equals legacy for scalar boundaries", () => {
            foreach (var text in new[] { "", "ASCII", "Привет", "日本語", "🙂\n\t", "\u007F\u0080\u07FF\u0800\uFFFF", "\U00010000\U0010FFFF", "e\u0301", "\0x" })
                Check(ByteTokenizer.Encode(text).SequenceEqual(LegacyBytes(text)));
        }),
        ("direct UTF8 stack/pool boundaries preserve exact bytes", () => {
            foreach (int n in new[] { 1023, 1024, 1025, 16384, 32768 })
                foreach (string unit in new[] { "x", "ж", "🙂" })
                { string text = string.Concat(Enumerable.Repeat(unit, n)); Check(ByteTokenizer.Encode(text).SequenceEqual(LegacyBytes(text))); }
        }),
        ("direct encode writes only the requested destination window", () => {
            var buffer = Enumerable.Repeat(-77, 80).ToArray(); string text = "Тест 🙂";
            int n = ByteTokenizer.EncodeInto(text, buffer.AsSpan(7, 60));
            Check(buffer.Take(7).All(x => x == -77) && buffer.Skip(7 + n).All(x => x == -77));
            Check(buffer.Skip(7).Take(n).SequenceEqual(LegacyBytes(text)));
        }),
        ("short UTF8 destination fails before mutation", () => {
            int[] buffer = [-1, -1]; Throws<ArgumentException>(() => ByteTokenizer.EncodeInto("🙂", buffer)); Check(buffer.All(x => x == -1));
        }),
        ("invalid Unicode fails before direct destination mutation", () => {
            int[] buffer = [-1, -1, -1, -1]; Throws<EncoderFallbackException>(() => ByteTokenizer.EncodeInto("x\uD800", buffer)); Check(buffer.All(x => x == -1));
            Throws<EncoderFallbackException>(() => ByteTokenizer.Encode("\uDC00"));
        }),
        ("empty direct encoding leaves destination unchanged", () => {
            int[] buffer = [9]; Check(ByteTokenizer.EncodeInto("", buffer) == 0 && buffer[0] == 9);
        }),
        ("direct prompt encoding matches list reference across budgets", () => {
            var history = new ChatTurn[] { new("Первый вопрос", "🙂 ответ", 0), new("日本語", "да", 1), new("third", "", 2) };
            foreach (int context in new[] { 32, 64, 128, 512 }) foreach (int reserve in new[] { 1, 8, 16 })
            {
                var actual = ByteTokenizer.Plan(history, "new", context, reserve);
                Check(actual.Tokens.SequenceEqual(LegacyPrompt(history, "new", context, reserve)));
                Check(actual.Tokens.Length == ByteTokenizer.Measure(history, "new", context, reserve).InputTokens);
            }
        }),
        ("direct prompt retains no partial old exchange", () => {
            var history = new ChatTurn[] { new(new string('a', 500), "older", 0), new("last", "reply", 1) };
            var plan = ByteTokenizer.Plan(history, "now", 64, 8);
            Check(plan.RetainedTurns == 1 && plan.DroppedTurns == 1 && plan.Tokens.SequenceEqual(LegacyPrompt(history, "now", 64, 8)));
        }),
        ("exact training encoding matches complete legacy corpus", () => {
            string data = Path.Combine(AppContext.BaseDirectory, "data");
            foreach (string path in Directory.GetFiles(data, "*.jsonl")) foreach (var example in Dataset.Load(path))
                Compare(example, 512);
        }),
        ("exact training encoding matches trimmed multibyte history", () => {
            var e = Dataset.Make("дальше?", "хорошо 🙂", history: new[] { new ChatTurn(new string('я', 80), "early", 0), new ChatTurn("中", "🙂", 0) });
            foreach (int length in new[] { 64, 128, 256, 512 }) Compare(e, length);
        }),
        ("exact text encoding preserves one terminal EOS target", () => {
            var e = Dataset.Encode(Dataset.Make("hi 🙂"), 32);
            Check(e.Tokens.Length == ByteTokenizer.TokenCount("hi 🙂") + 1 && e.Labels[^1] == ByteTokenizer.Eos);
            Check(e.Labels.All(x => x >= 0)); Compare(Dataset.Make("hi 🙂"), 32);
        }),
        ("exact dialogue encoding masks every history and prompt target", () => {
            var e = Dataset.Make("question", "abc", history: new[] { new ChatTurn("old", "reply", 0) });
            var row = Dataset.Encode(e, 64);
            Check(row.Labels.Count(x => x != -100) == 4 && row.Labels[^1] == ByteTokenizer.Eos); Compare(e, 64);
        }),
        ("exact length accepts boundary and rejects target truncation", () => {
            var e = Dataset.Make("qq", "aaaa"); int n = Dataset.RequiredSequenceLength(e);
            Check(Dataset.Encode(e, n).Tokens.Length == n); Throws<ArgumentException>(() => Dataset.Encode(e, n - 1));
            Throws<ArgumentException>(() => Dataset.Encode(Dataset.Make(new string('x', 40)), 16));
        }),
        ("encoding preparation parallel and sequential outputs match", () => {
            var rows = Enumerable.Range(0, 257).Select(i => Dataset.Make("вопрос" + i, "ответ🙂")).ToArray();
            var one = Dataset.EncodeAll(rows, 128, 1); var many = Dataset.EncodeAll(rows, 128, 4);
            Check(one.Length == many.Length);
            for (int i = 0; i < one.Length; i++) Check(one[i].Tokens.SequenceEqual(many[i].Tokens) && one[i].Labels.SequenceEqual(many[i].Labels) && one[i].Id == many[i].Id);
        }),
        ("tiny and empty encoding pools honor pre-cancellation", () => {
            using var stop = new CancellationTokenSource(); stop.Cancel();
            Throws<OperationCanceledException>(() => Dataset.EncodeAll([], 64, 4, stop.Token));
            Throws<OperationCanceledException>(() => Dataset.EncodeAll([Dataset.Make("test")], 64, 1, stop.Token));
        }),
        ("large encoding pools honor pre-cancellation", () => {
            using var stop = new CancellationTokenSource(); stop.Cancel();
            var rows = Enumerable.Range(0, 129).Select(i => Dataset.Make("test" + i)).ToArray();
            Throws<OperationCanceledException>(() => Dataset.EncodeAll(rows, 64, 4, stop.Token));
        }),
    ];
    private static int[] LegacyBytes(string text) => new UTF8Encoding(false, true).GetBytes(text).Select(b => (int)b + ByteTokenizer.Offset).ToArray();
    private static int[] LegacyPrompt(IReadOnlyList<ChatTurn> history, string user, int context, int reserve)
    {
        var budget = ByteTokenizer.Measure(history, user, context, reserve); var tokens = new List<int> { ByteTokenizer.Bos };
        for (int i = history.Count - budget.RetainedTurns; i < history.Count; i++)
        {
            tokens.Add(ByteTokenizer.User); tokens.AddRange(LegacyBytes(history[i].User)); tokens.Add(ByteTokenizer.Eos);
            tokens.Add(ByteTokenizer.Assistant); tokens.AddRange(LegacyBytes(history[i].Assistant)); tokens.Add(ByteTokenizer.Eos);
        }
        tokens.Add(ByteTokenizer.User); tokens.AddRange(LegacyBytes(user)); tokens.Add(ByteTokenizer.Eos); tokens.Add(ByteTokenizer.Assistant); return tokens.ToArray();
    }
    private static void Compare(TrainingExample e, int length)
    {
        var tokens = new List<int> { ByteTokenizer.Bos }; int answerStart = 1;
        if (e.IsDialogue)
        {
            var h = e.History ?? [];
            for (int i = Dataset.RetainedHistoryStart(e, length); i < h.Length; i++)
            {
                tokens.Add(ByteTokenizer.User); tokens.AddRange(LegacyBytes(h[i].User)); tokens.Add(ByteTokenizer.Eos);
                tokens.Add(ByteTokenizer.Assistant); tokens.AddRange(LegacyBytes(h[i].Assistant)); tokens.Add(ByteTokenizer.Eos);
            }
            tokens.Add(ByteTokenizer.User); tokens.AddRange(LegacyBytes(e.Text)); tokens.Add(ByteTokenizer.Eos);
            tokens.Add(ByteTokenizer.Assistant); answerStart = tokens.Count; tokens.AddRange(LegacyBytes(e.Answer!)); tokens.Add(ByteTokenizer.Eos);
        }
        else { tokens.AddRange(LegacyBytes(e.Text)); tokens.Add(ByteTokenizer.Eos); }
        var labels = Enumerable.Range(0, tokens.Count - 1).Select(i => i + 1 >= answerStart ? tokens[i + 1] : -100).ToArray();
        var actual = Dataset.Encode(e, length);
        Check(actual.Tokens.SequenceEqual(tokens.Take(tokens.Count - 1)) && actual.Labels.SequenceEqual(labels) && actual.Id == e.Id);
    }
    private static void Check(bool ok) { if (!ok) throw new Exception("Audit10 contract failed."); }
    private static void Throws<T>(Action action) where T : Exception
    { try { action(); } catch (T) { return; } throw new Exception("Expected " + typeof(T).Name); }
}
