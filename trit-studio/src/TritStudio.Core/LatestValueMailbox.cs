namespace TritStudio.Core;

// At most one queued callback; slow renderers see the newest text rather than an unbounded backlog.
// Dispose invalidates already scheduled callbacks, preventing stale text after a terminal result.
public sealed class LatestValueMailbox<T>(Action<Action> schedule, Action<T> consume) : IDisposable
{
    private readonly object _gate = new();
    private bool _scheduled, _hasValue, _closed;
    private T? _latest;
    public void Post(T value)
    {
        lock (_gate)
        {
            if (_closed) return;
            _latest = value; _hasValue = true;
            if (_scheduled) return;
            _scheduled = true;
        }
        try { schedule(Drain); }
        catch { lock (_gate) _scheduled = false; throw; }
    }
    private void Drain()
    {
        T value;
        lock (_gate)
        {
            _scheduled = false;
            if (_closed || !_hasValue) return;
            value = _latest!; _hasValue = false; _latest = default;
        }
        consume(value);
    }
    public void Dispose()
    {
        lock (_gate) { _closed = true; _hasValue = false; _latest = default; }
    }
}
