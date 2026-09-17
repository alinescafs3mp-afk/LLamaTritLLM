using System.Text;
namespace TritStudio.Core;

public sealed record ContextVariantResult(string Id, string ExpectedFact, string FinalAnswer,
    bool ExactFact, ConversationProbeTurn[] Turns);
public sealed record ContextPairResult(int Pair, bool BothCorrect, bool SameFinalAnswer,
    ContextVariantResult First, ContextVariantResult Second);
public sealed record ContextTransferReport(int Schema, long Revision, long Step, string PackedSha256,
    int AvailablePairs, string Scope, ContextPairResult[] Pairs)
{
    public int PassedPairs => Pairs.Count(x => x.BothCorrect);
    public int SameAnswerPairs => Pairs.Count(x => x.SameFinalAnswer);
    public string Summary => $"Контекстные пары: {PassedPairs}/{Pairs.Length}; одинаковый ответ при разных фактах: {SameAnswerPairs}. " +
        $"Проверено {Pairs.Length} из {AvailablePairs} пар. Это точный ответ на ограниченную задачу, не оценка общего интеллекта.";
}

// Public paired DEVELOPMENT diagnostic. The final question is identical, while its answer depends
// on the preceding user facts. A constant answer can never pass both variants. Expected replies are
// used ONLY after generation, never injected into input/history or used by the production decoder.
public static class ContextTransferProbe
{
    public static string NormalizeFact(string text) => text.Normalize(NormalizationForm.FormKC)
        .Trim().TrimEnd('.', '!', '?').Trim().ToLowerInvariant().Replace('ё', 'е');
    public static bool Matches(string expected, string answer) => NormalizeFact(expected).Length > 0 &&
        string.Equals(NormalizeFact(expected), NormalizeFact(answer), StringComparison.Ordinal);
    public static void Validate(TrainingExample[] tests)
    {
        if (tests.Length is < 2 or > 128 || tests.Length % 2 != 0)
            throw new ArgumentException("Контекстная проверка требует от 1 до 64 полных пар.");
        var ids = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < tests.Length; i += 2)
        {
            var a = tests[i]; var b = tests[i + 1];
            foreach (var row in new[] { a, b })
            {
                if (row.Answer is null || row.History is null || row.History.Length is < 1 or > 4 ||
                    !ids.Add(row.Id) || NormalizeFact(row.Answer).Length == 0)
                    throw new ArgumentException("Неполная или повторная контекстная задача.");
            }
            if (a.Text != b.Text || a.History!.Length != b.History!.Length || Matches(a.Answer!, b.Answer!))
                throw new ArgumentException("В паре нужен одинаковый последний вопрос и разные правильные ответы.");
            if (a.History.Select(x => x.User).SequenceEqual(b.History.Select(x => x.User), StringComparer.Ordinal))
                throw new ArgumentException("Проверка различается только эталонными ответами; факты пользователя должны различаться.");
        }
    }
    public static ContextTransferReport Run(ManagedInference model, long step, string packedHash,
        TrainingExample[] tests, int maxPairs = 16, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); Validate(tests);
        if (maxPairs is < 1 or > 64) throw new ArgumentOutOfRangeException(nameof(maxPairs));
        int pairs = Math.Min(maxPairs, tests.Length / 2), available = tests.Length / 2;
        var result = new List<ContextPairResult>();
        // Spread bounded automatic checks across the file instead of only testing its first category.
        for (int sample = 0; sample < pairs; sample++)
        {
            int pair = sample * available / pairs;
            var first = Generate(tests[2 * pair]); var second = Generate(tests[2 * pair + 1]);
            bool passed = first.ExactFact && second.ExactFact && first.Turns.Concat(second.Turns)
                .All(x => x.DroppedHistory == 0 && !x.Health.Repetitive);
            result.Add(new(pair, passed, NormalizeFact(first.FinalAnswer) == NormalizeFact(second.FinalAnswer), first, second));
        }
        return new(1, model.Revision, step, packedHash, available,
            "Public paired context transfer. Test rows never train. Exact final fact matching is a narrow task, not semantic conversation grading. " +
            "Real generated replies are used in history. Separate variants start empty. Generated text may contain learned private information.", result.ToArray());

        ContextVariantResult Generate(TrainingExample test)
        {
            var history = new List<ChatTurn>(); var turns = new List<ConversationProbeTurn>();
            foreach (var question in test.History!.Select(x => x.User).Append(test.Text))
            {
                ct.ThrowIfCancellationRequested();
                int room = model.Weights.Config.Context - ByteTokenizer.TokenCount(question) - 4;
                if (room < 1) throw new ArgumentException("Вопрос контекстной проверки не помещается в контекст модели.");
                int reserve = Math.Min(64, room);
                var plan = ByteTokenizer.Plan(history, question, model.Weights.Config.Context, reserve);
                var answer = model.Generate(plan.Tokens, () => new SamplingOptions(0, ByteTokenizer.VocabularySize, 1, 1, reserve), null, ct);
                turns.Add(new(question, answer, GenerationHealth.Inspect(answer), plan.DroppedTurns));
                history.Add(new(question, answer, model.Revision));
            }
            string final = turns[^1].Answer;
            return new(test.Id, test.Answer!, final, Matches(test.Answer!, final), turns.ToArray());
        }
    }
    public static string Markdown(ContextTransferReport report)
    {
        var text = new StringBuilder($"# Контекстная проверка: r{report.Revision}, шаг {report.Step}\n\n{report.Summary}\n\n");
        text.AppendLine($"SHA256 весов: `{report.PackedSha256}`. Это открытая проверка переноса на другие формулировки, не скрытый экзамен.\n");
        text.AppendLine("Каждая пара меняет факт, но сохраняет последний вопрос. Эталонные ответы НЕ передаются модели. Ключевое слово внутри неверной фразы не засчитывается.\n");
        foreach (var pair in report.Pairs)
        {
            text.AppendLine($"## Пара {pair.Pair}: {(pair.BothCorrect ? "обе верны" : "есть ошибка")}\n");
            foreach (var variant in new[] { pair.First, pair.Second })
            {
                text.AppendLine("### " + variant.Id + "\n");
                foreach (var turn in variant.Turns)
                {
                    text.AppendLine("Вопрос:\n" + Plain(turn.Question));
                    text.AppendLine("Ответ:\n" + Plain(turn.Answer));
                    if (turn.DroppedHistory > 0) text.AppendLine($"Утеряно обменов контекста: {turn.DroppedHistory}.\n");
                }
                text.AppendLine("Ожидаемый факт (только для проверки, не часть запроса):\n" + Plain(variant.ExpectedFact));
            }
        }
        return text.ToString();
        static string Plain(string value) => string.Join("\n", value.Replace("\r", "").Split('\n').Select(x => "    " + x)) + "\n";
    }
}
