using System.Text;
namespace TritStudio.Core;

public sealed record GeneralizationCase(string Id, string Expected, string Actual, bool Exact,
    bool InputAlreadyTrained, ConversationProbeTurn[] Turns);
public sealed record GeneralizationReport(int Schema, long Revision, long Step, string PackedSha256,
    int Available, string Scope, GeneralizationCase[] Cases)
{
    public int PreviouslyTrained => Cases.Count(x => x.InputAlreadyTrained);
    public int Eligible => Cases.Length - PreviouslyTrained;
    public int Exact => Cases.Count(x => !x.InputAlreadyTrained && x.Exact);
    public string Summary => $"Новые формулировки: {Exact}/{Eligible} точных эталонных ответов; проверено {Cases.Length}/{Available}. " +
        $"Уже присутствуют в записанном обучении: {PreviouslyTrained}. Не общий процент качества речи; приемлемая перефразировка может не совпасть с эталоном.";
}

// Development evaluation ONLY. The generation loop sees user prompts and ITS OWN replies, not expected answers.
// Exact comparison is a deliberately strict reproducible diagnostic, not a semantic judge of natural conversation.
public static class GeneralizationProbe
{
    public static GeneralizationReport Run(ManagedInference model, long step, string hash,
        TrainingExample[] tests, TrainingExample[] knownTraining, int maxCases = 128,
        Action<int, int>? progress = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(tests); ArgumentNullException.ThrowIfNull(knownTraining);
        if (tests.Length is < 1 or > 128 || maxCases is < 1 or > 128 || step < 0) throw new ArgumentException("Invalid bounded generalization probe.");
        if (tests.Any(x => x is null || !x.IsDialogue || x.History?.Length > 4) || tests.Select(x => x.Id).Distinct().Count() != tests.Length)
            throw new ArgumentException("Incomplete or duplicate evaluation cases.");
        // This index answers only whether a diagnostic case was already a training input.
        // It neither inserts evaluation labels into training nor filters decoder output.
        var known = new ValidationGuard(knownTraining, model.Weights.Config.Context, ct);
        int count = Math.Min(maxCases, tests.Length); var results = new List<GeneralizationCase>(count);
        for (int i = 0; i < count; i++)
        {
            ct.ThrowIfCancellationRequested(); var test = tests[i * tests.Length / count];
            var history = new List<ChatTurn>(); var turns = new List<ConversationProbeTurn>();
            foreach (var question in (test.History ?? []).Select(x => x.User).Append(test.Text))
            {
                ct.ThrowIfCancellationRequested(); int room = model.Weights.Config.Context - ByteTokenizer.TokenCount(question) - 4;
                if (room < 1) throw new ArgumentException("Вопрос проверки не помещается в контекст этой модели.");
                int reserve = Math.Min(112, room);
                var plan = ByteTokenizer.Plan(history, question, model.Weights.Config.Context, reserve);
                var answer = model.Generate(plan.Tokens, () => new SamplingOptions(0,ByteTokenizer.VocabularySize,1,1,reserve),null,ct);
                turns.Add(new(question,answer,GenerationHealth.Inspect(answer),plan.DroppedTurns));
                history.Add(new(question,answer,model.Revision));
            }
            // A generated acknowledgement can turn a nominally new reference history into an actual
            // recorded training input. Check BOTH contexts before claiming transfer.
            var actualInput = Dataset.Make(test.Text, test.Answer, history: history.Take(history.Count - 1).ToArray());
            bool seen = known.Contains(test, ct) || known.Contains(actualInput, ct);
            string actual = turns[^1].Answer;
            bool exact = ContextTransferProbe.Matches(test.Answer!, actual) && turns.All(t => t.DroppedHistory == 0 && !t.Health.Repetitive);
            results.Add(new(test.Id, test.Answer!, actual, exact, seen, turns.ToArray()));
            progress?.Invoke(i + 1, count);
        }
        return new(1,model.Revision,step,hash,tests.Length,
            "Public development paraphrase/task diagnostic. Reference targets used ONLY after free generation. Exact match is not semantic quality. " +
            "Historical/deleted training data or semantic overlap may be unknown; previously trained exclusion checks reference AND actual generated inputs against supplied recorded sources. " +
            "Generated text may reveal learned private information; review before sharing.",results.ToArray());
    }
    public static string Markdown(GeneralizationReport report)
    {
        var text = new StringBuilder($"# Новые формулировки: r{report.Revision}, шаг {report.Step}\n\n{report.Summary}\n\nSHA256: `{report.PackedSha256}`\n\n");
        text.AppendLine("Это открытая диагностика. Читайте целые ответы: точное несовпадение не всегда означает смысловую ошибку. Прошлые реплики модели не заменяются эталонами.\n");
        foreach (var c in report.Cases)
        {
            text.AppendLine($"## {c.Id}: {(c.InputAlreadyTrained ? "уже в обучении, вне оценки переноса" : c.Exact ? "совпал с эталоном" : "не совпал, требуется чтение")}\n");
            foreach (var turn in c.Turns)
            {
                text.AppendLine("Вы:\n" + Literal(turn.Question)); text.AppendLine("Модель:\n" + Literal(turn.Answer));
                if (turn.DroppedHistory != 0) text.AppendLine($"Потеряно обменов контекста: {turn.DroppedHistory}\n");
            }
            text.AppendLine("Эталон проверяющего (НЕ передавался в запросе):\n" + Literal(c.Expected));
        }
        return text.ToString();
        static string Literal(string s) => string.Join("\n",s.Replace("\r","").Split('\n').Select(x => "    " + x)) + "\n";
    }
}
