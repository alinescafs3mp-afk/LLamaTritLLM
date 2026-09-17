namespace TritStudio.Core;

public sealed record DialogueMix(int General, int Language, int Facts)
{
    public string Summary => $"Состав пакета: общий {General}, разговорные ответы {Language}, факты из истории {Facts}.";
}

// Opt-in v22 sampling only. The network, labels, objective and decoder do not inspect a recipe.
// Crucially, bucket-by-length MUST NOT resample this plan after its category quotas are selected.
public sealed class DialogueBatchPlanner
{
    private static readonly int[][] Patterns =
    [ [1, 1, 0, 1, 2, 1, 0, 2], [1, 2, 0, 2, 1, 2, 0, 1], [1, 0, 2, 1, 0, 2, 0, 1] ];
    public EncodedExample[] Corpus { get; }
    private readonly int[] _language;
    private readonly int[][] _facts;
    private readonly int[] _lengths, _targets;
    public DialogueMix? LastMix { get; private set; }
    public int LanguageRows => _language.Length;
    public int FactQuestionGroups => _facts.Length;
    public DialogueBatchPlanner(EncodedExample[] corpus, IEnumerable<TrainingExample> language,
        IEnumerable<TrainingExample> facts, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(language); ArgumentNullException.ThrowIfNull(facts);
        if (corpus.Length is < 1 or > Dataset.MaxExamples) throw new ArgumentException("Invalid course corpus size.");
        Corpus = corpus;
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        _lengths = new int[corpus.Length]; _targets = new int[corpus.Length];
        for (int i = 0; i < corpus.Length; i++)
        {
            ct.ThrowIfCancellationRequested();
            if (!ids.TryAdd(corpus[i].Id, i)) throw new ArgumentException("Duplicate course example ID.");
            _lengths[i] = BatchPlanner.EffectiveLength(corpus[i]);
            for (int j = 0; j < _lengths[i]; j++) if (corpus[i].Labels[j] != -100) _targets[i]++;
        }
        // Only final ORIGINAL targets join this priority pool. Expanded intermediate acknowledgements
        // remain available in the general pool, rather than multiplying their priority weight.
        _language = Match(language).Select(x => ids[x.Id]).Distinct().ToArray();
        _facts = Match(facts).GroupBy(x => x.Text, StringComparer.Ordinal)
            .Select(g => g.Select(x => ids[x.Id]).Distinct().ToArray()).ToArray();
        if (_language.Length == 0 || _facts.Length == 0)
            throw new ArgumentException("v22 требует непустые разговорные и контекстные учебные группы после исключения контроля.");

        IEnumerable<TrainingExample> Match(IEnumerable<TrainingExample> rows)
        {
            foreach (var row in rows)
            {
                ct.ThrowIfCancellationRequested();
                if (row is null || !row.IsDialogue) throw new ArgumentException("Course priority rows must be dialogue targets.");
                if (ids.ContainsKey(row.Id)) yield return row;
            }
        }
    }
    public static int Phase(int completed, int total)
    {
        if (total is < 1 or > 100000 || completed < 0 || completed >= total) throw new ArgumentOutOfRangeException(nameof(completed));
        return (long)completed * 5 < total ? 0 : (long)completed * 5 < (long)total * 3 ? 1 : 2;
    }
    public static string PhaseName(int completed, int total) => Phase(completed,total) switch
    {
        0 => "v22: разговорная база с общим материалом",
        1 => "v22: разные формулировки и факты в контексте",
        _ => "v22: закрепление беседы и общего материала"
    };
    public static DialogueMix Quotas(int batchSize, int completed, int total)
    {
        if (batchSize is < 1 or > ResourceOptions.MaxBatchSize) throw new ArgumentOutOfRangeException(nameof(batchSize));
        int phase = Phase(completed, total); var counts = new int[3];
        for (int i = 0; i < batchSize; i++) counts[Patterns[phase][((long)completed * batchSize + i) % 8]]++;
        return new(counts[0], counts[1], counts[2]);
    }
    public PlannedBatch Select(int batchSize, SamplerRandom random, int completed, int total, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(random);
        var mix = Quotas(batchSize, completed, total); int phase = Phase(completed,total);
        var selected = new EncodedExample[batchSize]; int length = 0; long inputs = 0, targets = 0;
        for (int slot = 0; slot < batchSize; slot++)
        {
            ct.ThrowIfCancellationRequested();
            int kind = Patterns[phase][((long)completed * batchSize + slot) % 8];
            int index;
            if (kind == 0) index = random.Next(Corpus.Length);
            else if (kind == 1) index = _language[random.Next(_language.Length)];
            else
            {
                var group = _facts[random.Next(_facts.Length)];
                index = group[random.Next(group.Length)];
            }
            selected[slot] = Corpus[index]; length = Math.Max(length, _lengths[index]);
            inputs += _lengths[index]; targets += _targets[index];
        }
        LastMix = mix;
        return new(selected, length, inputs, targets);
    }
}
