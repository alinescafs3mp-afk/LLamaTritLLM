using System.Text;
namespace TritStudio.Core;

public sealed record ConversationProbeCase(string Id, string Category, string[] Questions,
    string[] RequiredWords, bool AutomaticCheck);
public sealed record ConversationProbeTurn(string Question, string Answer, GenerationHealth Health, int DroppedHistory);
public sealed record ConversationProbeResult(string Id, string Category, bool? SimpleCheckPassed,
    bool Repetitive, ConversationProbeTurn[] Turns);
public sealed record ConversationProbeReport(int Schema, long Revision, long Step, string PackedSha256,
    string Scope, ConversationProbeResult[] Results)
{
    public int Checked => Results.Count(x => x.SimpleCheckPassed.HasValue);
    public int Passed => Results.Count(x => x.SimpleCheckPassed == true);
    public int Repetitive => Results.Count(x => x.Repetitive);
    public string Summary => $"Простые разговорные задания: {Passed}/{Checked}; зацикливание: {Repetitive}/{Results.Length}. " +
        "Это открытые проверочные задания, не процент интеллекта. Открытые ответы прочитайте в отчёте.";
}

// PUBLIC DEVELOPMENT checks, not hidden test scores. Never import these expected words into training.
// Later turns use the model's OWN previous answers, not teacher-forced reference replies.
public static class ConversationProbe
{
    public static IReadOnlyList<ConversationProbeCase> Cases { get; } = Array.AsReadOnly<ConversationProbeCase>(
    [
        new("greeting", "открытый ответ", ["Привет! Давай немного поболтаем."], [], false),
        new("company", "открытый ответ", ["У меня сегодня был спокойный день. Хочу просто поделиться."], [], false),
        new("question", "открытый ответ", ["Задай один простой вопрос о моём дне."], [], false),
        new("rewrite", "открытый ответ", ["Помоги вежливо написать другу, что я немного опоздаю."], [], false),
        new("name", "память разговора", ["В этом разговоре называй меня Ариной.", "Как меня называть?"], ["арин"], true),
        new("change", "изменение решения", ["Я выбираю синюю папку.", "Передумал, выбираю жёлтую.", "Какую папку я в итоге выбрал?"], ["жёлт"], true),
        new("location", "факт из запроса", ["Мой блокнот лежит под лампой, а не в шкафу. Где мой блокнот?"], ["под ламп"], true),
        new("draft", "намерение и действие", ["Я написал черновик, но письмо не отправлял. Оно уже отправлено?"], ["нет"], true),
        new("constraint", "точное короткое действие", ["Ответь только словом «понятно» без знаков препинания."], ["понятно"], true),
        new("two_people", "разделение фактов", ["У Аллы красный зонт, у Игоря зелёный. Какой зонт у Аллы?"], ["красн"], true),
    ]);
    private static string Normalize(string s) => s.Normalize(NormalizationForm.FormKC).ToLowerInvariant().Replace('ё','е');
    public static bool Matches(ConversationProbeCase probe, string answer)
    {
        string normal = Normalize(answer.Trim());
        if (probe.Id == "constraint") return normal == "понятно";
        return normal.Length > 0 && probe.RequiredWords.All(word => normal.Contains(Normalize(word), StringComparison.Ordinal));
    }
    public static ConversationProbeReport Run(ManagedInference model, long step, string packedHash,
        CancellationToken ct = default)
    {
        var results = new List<ConversationProbeResult>(); int context = model.Weights.Config.Context;
        foreach (var probe in Cases)
        {
            ct.ThrowIfCancellationRequested(); var history = new List<ChatTurn>(); var turns = new List<ConversationProbeTurn>();
            foreach (string question in probe.Questions)
            {
                int room = context - ByteTokenizer.TokenCount(question) - 4;
                if (room < 1) throw new ArgumentException("Для разговорной проверки нужен больший контекст модели. Веса не изменены.");
                int reserve = Math.Min(160, room);
                var plan = ByteTokenizer.Plan(history, question, context, reserve);
                string answer = model.Generate(plan.Tokens, () => new SamplingOptions(0, ByteTokenizer.VocabularySize, 1, 1, reserve), null, ct);
                var health = GenerationHealth.Inspect(answer);
                turns.Add(new(question, answer, health, plan.DroppedTurns));
                history.Add(new ChatTurn(question, answer, model.Revision));
            }
            bool repetitive = turns.Any(x => x.Health.Repetitive);
            bool? passed = probe.AutomaticCheck ? !repetitive && turns.All(x => x.DroppedHistory == 0) && Matches(probe, turns[^1].Answer) : null;
            results.Add(new(probe.Id, probe.Category, passed, repetitive, turns.ToArray()));
        }
        return new(1, model.Revision, step, packedHash,
            "Public development probes. Simple lexical checks are NOT semantic or blind evaluation; generated text can contain learned private information. Greedy temperature0, per-case empty history, own generated history within a case.", results.ToArray());
    }
    public static string Markdown(ConversationProbeReport report)
    {
        var s = new StringBuilder($"# Разговорная проверка r{report.Revision}, шаг {report.Step}\n\n{report.Summary}\n\n");
        s.AppendLine("Ответы ниже сгенерированы по весам. Совпадение ключевого слова не доказывает правильность всего ответа. Проверьте текст перед передачей отчёта.\n");
        foreach (var r in report.Results)
        {
            s.AppendLine($"## {r.Id}: {(r.SimpleCheckPassed is null ? "нужна оценка человеком" : r.SimpleCheckPassed.Value ? "простая проверка выполнена" : "простая проверка не выполнена")}\n");
            foreach (var t in r.Turns)
            {
                s.AppendLine("**Вы:** " + t.Question + "\n");
                // Indented plain text prevents generated markup from acting as an HTML/script block.
                s.AppendLine(string.Join("\n", t.Answer.Replace("\r", "").Split('\n').Select(x => "    " + x)) + "\n");
                if (t.DroppedHistory > 0) s.AppendLine($"Контекст сокращён: {t.DroppedHistory} обменов. Этот случай не получает положительную оценку памяти.\n");
            }
        }
        return s.ToString();
    }
}
