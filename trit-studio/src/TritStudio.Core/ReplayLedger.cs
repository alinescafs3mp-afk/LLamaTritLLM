namespace TritStudio.Core;

public sealed record ReplayEntry(TrainingExample Example, string State = "pending");
public sealed record ReplayDocument(ReplayEntry[] Entries);
// Owned by exactly one trainer process. Pending work is durable; a handled/rejected ID is not reapplied on restart.
public sealed class ReplayLedger
{
    private readonly string _path;
    private Dictionary<string, ReplayEntry> _items;
    private TrainingExample[] _pending = [], _learned = [];
    private QueueSummary _summary = new(0, 0, 0, 0, 0);
    public ReplayLedger(string workspace)
    {
        _path = Path.Combine(workspace, "replay.json");
        _items = File.Exists(_path) ? JsonData.Read<ReplayDocument>(_path).Entries.ToDictionary(x => x.Example.Id) : new();
        if (_items.Count > Dataset.MaxExamples) throw new InvalidDataException("Replay ledger exceeds the record limit.");
        foreach (var row in _items.Values)
        {
            if (row.State is not ("pending" or "learned" or "rejected" or "rolledback" or "discarded")) throw new InvalidDataException("Unknown replay state.");
            var e = row.Example;
            if (Dataset.Make(e.Text, e.Answer, e.Source, e.History).Id != e.Id) throw new InvalidDataException("Replay identity mismatch.");
        }
        RebuildViews();
    }
    public QueueSummary Summary => _summary;
    private void RebuildViews()
    {
        var pending = new List<TrainingExample>(); var learned = new List<TrainingExample>();
        int rejected = 0, rolledback = 0, discarded = 0;
        foreach (var row in _items.Values)
            switch (row.State)
            {
                case "pending": pending.Add(row.Example); break;
                case "learned": learned.Add(row.Example); break;
                case "rejected": rejected++; break;
                case "rolledback": rolledback++; break;
                case "discarded": discarded++; break;
            }
        _pending = pending.ToArray(); _learned = learned.ToArray();
        _summary = new(_pending.Length, _learned.Length, rejected, rolledback, discarded);
    }
    // Discard removes pending work only. It never claims to unlearn already committed weights.
    public int DiscardPending()
    {
        int count = PendingCount;
        if (count != 0) Replace(_items.ToDictionary(x => x.Key, x => x.Value.State == "pending" ? x.Value with { State = "discarded" } : x.Value));
        return count;
    }
    public int PendingCount => _pending.Length;
    public TrainingExample[] Pending => (TrainingExample[])_pending.Clone();
    public TrainingExample[] Learned => (TrainingExample[])_learned.Clone();
    public TrainingExample[] PeekPending(int limit) => _pending.AsSpan(0, Math.Clamp(limit, 0, _pending.Length)).ToArray();
    public TrainingExample[] RecentLearned(int limit)
    {
        int count = Math.Clamp(limit, 0, _learned.Length);
        return _learned.AsSpan(_learned.Length - count, count).ToArray();
    }
    public bool Add(TrainingExample e) => AddMany([e]) > 0;
    public int AddMany(IEnumerable<TrainingExample> examples)
    {
        var added = examples.DistinctBy(e => e.Id).Where(e => !_items.ContainsKey(e.Id)).ToArray();
        if (_items.Count + added.Length > Dataset.MaxExamples) throw new InvalidOperationException("Replay limit reached. Export/archive this workspace before collecting more data.");
        if (added.Length == 0) return 0;
        foreach (var e in added)
            if (Dataset.Make(e.Text, e.Answer, e.Source, e.History).Id != e.Id) throw new InvalidDataException("Replay identity mismatch.");
        var newEntries = added.Select(e => new ReplayEntry(e)).ToArray();
        JsonData.AtomicWrite(_path, new ReplayDocument(_items.Values.Concat(newEntries).ToArray()));
        foreach (var e in newEntries) _items.Add(e.Example.Id, e);
        RebuildViews(); return added.Length;
    }
    public void Finish(IEnumerable<string> ids, bool accepted)
    {
        var changes = ids.ToHashSet(StringComparer.Ordinal);
        Replace(_items.ToDictionary(x => x.Key, x => changes.Contains(x.Key) ? x.Value with { State = accepted ? "learned" : "rejected" } : x.Value));
    }
    public void Reconcile(IEnumerable<string> committedIds)
    {
        var keep = committedIds.ToHashSet(StringComparer.Ordinal);
        var next = _items.ToDictionary(x => x.Key, x => x.Value);
        foreach (var id in keep) if (next.TryGetValue(id, out var e) && e.State != "learned") next[id] = e with { State = "learned" };
        foreach (var id in next.Keys.ToArray()) if (!keep.Contains(id) && next[id].State == "learned") next[id] = next[id] with { State = "rolledback" };
        Replace(next);
    }
    public void MarkRollback(IEnumerable<string> appliedIds)
    {
        var keep = appliedIds.ToHashSet(StringComparer.Ordinal);
        Replace(_items.ToDictionary(x => x.Key, x => x.Value.State == "learned" && !keep.Contains(x.Key) ? x.Value with { State = "rolledback" } : x.Value));
    }
    private void Replace(Dictionary<string, ReplayEntry> next)
    {
        if (next.Count == _items.Count && next.All(p => _items.TryGetValue(p.Key, out var old) && old == p.Value)) return;
        JsonData.AtomicWrite(_path, new ReplayDocument(next.Values.ToArray())); _items = next; RebuildViews();
    }
}
