using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using TritStudio.Core;
namespace TritStudio.App;

public sealed class WorkerClient : IAsyncDisposable
{
    private readonly Process _process;
    private readonly SemaphoreSlim _write = new(1, 1);
    private readonly TimeSpan _writeTimeout;
    private readonly CancellationTokenSource _life = new();
    private readonly TaskCompletionSource<ReadyEvent> _ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private sealed record PendingRequest(string Kind, TaskCompletionSource<WorkerEvent> Receipt);
    private readonly ConcurrentDictionary<string, PendingRequest> _pending = new();
    private Task? _output, _errors;
    private volatile bool _disposing;
    private int _failed;
    private readonly object _disposeGate = new();
    private Task? _disposeTask;
    public event Action<string>? Terminated;
    public event Action<WorkerEvent>? Received;
    public event Action<string>? Faulted;
    public event Action<string>? Diagnostic;
    public WorkerClient(string workspace, string executable, TimeSpan? writeTimeout = null)
    {
        _writeTimeout = writeTimeout ?? TimeSpan.FromSeconds(10);
        if (_writeTimeout < TimeSpan.FromMilliseconds(100) || _writeTimeout > TimeSpan.FromMinutes(1))
            throw new ArgumentOutOfRangeException(nameof(writeTimeout));
        if (!File.Exists(executable)) throw new FileNotFoundException("Тренер не найден.", executable);
        var start = new ProcessStartInfo(executable) { UseShellExecute = false, RedirectStandardInput = true,
            RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true,
            StandardInputEncoding = new UTF8Encoding(false), StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8 };
        start.ArgumentList.Add("--workspace"); start.ArgumentList.Add(Path.GetFullPath(workspace));
        _process = new Process { StartInfo = start };
    }
    public void Start()
    {
        if (!_process.Start()) throw new InvalidOperationException("Не удалось запустить тренер.");
        try { _process.PriorityClass = ProcessPriorityClass.BelowNormal; } catch { }
        _output = Task.Run(ReadOutput); _errors = Task.Run(ReadErrors);
    }
    public Task<ReadyEvent> WaitReadyAsync() => _ready.Task.WaitAsync(TimeSpan.FromMinutes(3), _life.Token);
    private async Task ReadOutput()
    {
        try
        {
            var lines = new BoundedLineReader(_process.StandardOutput, 1_048_576);
            while (!_life.IsCancellationRequested)
            {
                string? line = await lines.ReadLineAsync(_life.Token).ConfigureAwait(false); if (line is null) break;
                if (line.Length > 1_048_576) throw new InvalidDataException("Trainer protocol record is too large.");
                WorkerEvent? message;
                try { message = EventEnvelope.Parse(line); }
                catch (JsonException e) { throw new InvalidDataException("Нарушен протокол тренера. Подробности в диагностике: " + line[..Math.Min(line.Length, 500)], e); }
                if (message is null || string.IsNullOrWhiteSpace(message.Kind) || message.Data.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
                    throw new InvalidDataException("Тренер прислал пустую или неполную запись протокола.");
                if (message.Kind == "completed" && string.IsNullOrWhiteSpace(message.Id))
                    throw new InvalidDataException("У результата команды отсутствует идентификатор.");
                if (message.Kind == "ready") _ready.TrySetResult(Protocol.Payload<ReadyEvent>(message.Data));
                if (message.Kind == "completed")
                {
                    // Parse all fields first. Otherwise GetString() on a forged error object could remove a
                    // request from _pending, then throw before completing it, leaving the UI waiting forever.
                    var completion = CommandCompletion.Parse(message.Data);
                    string completedKind = EventEnvelope.CompletionCommand(message.Data);
                    if (message.Id is string id)
                    {
                        if (_pending.TryGetValue(id, out var expected) && completedKind != expected.Kind)
                            throw new InvalidDataException("Идентификатор ответа совпал, но имя выполненной команды отличается от запроса.");
                        if (_pending.TryRemove(id, out var pending))
                        {
                            if (completion.Success) pending.Receipt.TrySetResult(message);
                            else if (completion.Cancelled) pending.Receipt.TrySetCanceled();
                            else pending.Receipt.TrySetException(new InvalidOperationException(completion.Error ?? "Команда не выполнена."));
                        }
                    }
                }
                Received?.Invoke(message);
            }
            if (!_life.IsCancellationRequested && !_disposing)
            {
                string detail = _process.HasExited ? $"код {_process.ExitCode}" : "поток ответов закрыт, процесс ещё не завершён";
                Fail(new IOException($"Связь с тренером потеряна ({detail}). Чат на последнем снимке остаётся доступен."));
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e) { if (!_life.IsCancellationRequested) Fail(e); }
    }
    private void Fail(Exception exception)
    {
        // Protocol-frame failures are a dead trainer link: fail the pending request as IOException
        // (same public type as EOF), then kill the child. InvalidDataException is for parsers, not callers.
        if (exception is InvalidDataException)
            exception = new IOException(exception.Message, exception);
        bool first = Interlocked.Exchange(ref _failed, 1) == 0;
        _ready.TrySetException(exception);
        foreach (var item in _pending.ToArray()) if (_pending.TryRemove(item.Key, out var tcs)) tcs.Receipt.TrySetException(exception);
        if (!_disposing && first)
        {
            // A broken protocol must not leave an invisible child continuing to change weights.
            try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); } catch { }
            Terminated?.Invoke(exception.Message);
        }
    }
    private async Task ReadErrors()
    {
        try
        {
            var lines = new BoundedLineReader(_process.StandardError, 16_384, drainOverflow: true);
            while (!_life.IsCancellationRequested)
            {
                string? line = await lines.ReadLineAsync(_life.Token).ConfigureAwait(false); if (line is null) break;
                if (lines.LastLineTruncated) line += " [диагностическая строка сокращена]";
                StartupLog.Write(line); if (!_disposing) Diagnostic?.Invoke(line[..Math.Min(line.Length, 1500)]);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e) { if (!_life.IsCancellationRequested && !_disposing) Faulted?.Invoke(e.Message); }
    }
    private async Task WriteCommand<T>(string id, string kind, T payload)
    {
        if (_disposing && kind != "shutdown" || Volatile.Read(ref _failed) != 0) throw new IOException("Тренер отключён. Переподключите его на вкладке обучения.");
        string json = JsonSerializer.Serialize(new WorkerCommand(kind, id, Protocol.Element(payload)), JsonData.Options);
        if (json.Length > 1_048_576) throw new ArgumentException("Command exceeds the protocol size limit.");
        bool acquired = false;
        using var transfer = CancellationTokenSource.CreateLinkedTokenSource(_life.Token);
        Task? operation = null;
        try
        {
            if (!await _write.WaitAsync(_writeTimeout, _life.Token).ConfigureAwait(false))
                throw new TimeoutException("Истёк срок ожидания передачи команды тренеру.");
            acquired = true;
            if (Volatile.Read(ref _failed) != 0 || (_disposing && kind != "shutdown"))
                throw new IOException("Соединение закрыто до отправки команды.");
            transfer.CancelAfter(_writeTimeout);
            operation = WriteAndFlush(json, transfer.Token);
            // The token requests cancellation of the underlying I/O. WaitAsync also bounds our
            // wait on a pipe implementation that does not promptly honor cancellation.
            await operation.WaitAsync(_writeTimeout, _life.Token).ConfigureAwait(false);
        }
        catch (Exception error) when (error is IOException or TimeoutException or OperationCanceledException or ObjectDisposedException)
        {
            transfer.Cancel();
            if (operation is not null) _ = operation.ContinueWith(t => { _ = t.Exception; },
                CancellationToken.None, TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            var failure = new IOException("Команда не передана или передана не полностью. Соединение закрыто; автоматического повтора нет. " +
                "Переподключите тренер и проверьте последний сохранённый снимок. Это ограничение передачи, не времени обучения.", error);
            Fail(failure); throw failure;
        }
        finally { if (acquired) _write.Release(); }
    }
    private async Task WriteAndFlush(string json, CancellationToken ct)
    {
        await _process.StandardInput.WriteLineAsync(json.AsMemory(), ct).ConfigureAwait(false);
        await _process.StandardInput.FlushAsync(ct).ConfigureAwait(false);
    }
    public Task Send<T>(string kind, T payload) => WriteCommand(Guid.NewGuid().ToString("N"), kind, payload);
    // A receipt means the durable queue accepted the example, not that its training improved quality.
    public async Task<WorkerEvent> Request<T>(string kind, T payload)
    {
        string id = Guid.NewGuid().ToString("N"); var receipt = new TaskCompletionSource<WorkerEvent>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = new PendingRequest(kind, receipt);
        try { await WriteCommand(id, kind, payload).ConfigureAwait(false); return await receipt.Task.WaitAsync(_life.Token).ConfigureAwait(false); }
        finally
        {
            _pending.TryRemove(id, out _);
            // A transport write can fail before Request starts awaiting the receipt that Fail completed.
            if (receipt.Task.IsFaulted) _ = receipt.Task.Exception;
        }
    }
    public ValueTask DisposeAsync()
    {
        // Every concurrent closer awaits the SAME shutdown, not an early no-op.
        lock (_disposeGate) return new ValueTask(_disposeTask ??= DisposeCoreAsync());
    }
    private async Task DisposeCoreAsync()
    {
        _disposing = true;
        Task? shutdownWrite = null;
        try
        {
            if (!_process.HasExited)
            {
                // Bound pipe writes as well as native-kernel exit. Never wait forever while closing the GUI.
                try
                {
                    shutdownWrite = Send("shutdown", new { });
                    await shutdownWrite.WaitAsync(TimeSpan.FromSeconds(2)).ConfigureAwait(false);
                    _process.StandardInput.Close();
                }
                catch { /* Terminate below if the graceful transfer cannot finish. */ }
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                try { await _process.WaitForExitAsync(timeout.Token); }
                catch (OperationCanceledException) { if (!_process.HasExited) { _process.Kill(entireProcessTree: true); await _process.WaitForExitAsync(); } }
            }
        }
        catch { try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); } catch { } }
        finally
        {
            _life.Cancel(); Fail(new OperationCanceledException("Тренер закрыт."));
            // WaitAsync does not own/cancel its underlying task. A delayed failed shutdown write
            // must be observed even when the shorter graceful-shutdown wait has already timed out.
            if (shutdownWrite is not null) _ = shutdownWrite.ContinueWith(t => { _ = t.Exception; },
                CancellationToken.None, TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            try { await Task.WhenAll(_output ?? Task.CompletedTask, _errors ?? Task.CompletedTask).WaitAsync(TimeSpan.FromSeconds(5)); } catch { }
            _process.Dispose();
            // Do not dispose the semaphore while an abandoned pipe write could still complete its finally block.
        }
    }
}
