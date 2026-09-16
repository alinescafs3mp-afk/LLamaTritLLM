using System.Text;
using TritStudio.Core;
Console.InputEncoding = Encoding.UTF8; Console.OutputEncoding = new UTF8Encoding(false);
try
{
    if (args.Length >= 2 && args[0] == "--prune")
    {
        int keep = args.Length >= 3 && int.TryParse(args[2], out var count) ? count : 8;
        var result = WorkspaceMaintenance.Prune(args[1], keep, args.Contains("--apply"));
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(result, JsonData.Options));
        if (!result.Applied) Console.WriteLine("Dry run only. Close the GUI and repeat with --apply to remove old snapshots. Active and immediate parent are retained.");
        return 0;
    }
    if (args.Length == 0) { Console.WriteLine("Usage: TritStudio.Runner model.tritmodel [prompt] | --prune WORKSPACE [keep] [--apply]"); return 2; }
    var weights = ModelFiles.Read(Path.GetFullPath(args[0]), true);
    var model = new ManagedInference(weights, -1, Math.Max(1, Math.Min(4, Environment.ProcessorCount - 1)));
    var history = new List<ChatTurn>();
    Console.WriteLine($"Trit Studio CPU runner | {weights.Config.ParameterCount:N0} parameters | UTF-8 byte tokenizer");
    Console.WriteLine("No training/CUDA runtime is loaded. /exit exits; /clear resets conversation context.");
    var sampling = new SamplingOptions(MaxNewTokens: Math.Min(128, weights.Config.Context / 4));
    while (true)
    {
        string? text;
        if (args.Length > 1) text = args[1]; else { Console.Write("You> "); text = Console.ReadLine(); }
        if (text is null or "/exit") break;
        if (text == "/clear") { history.Clear(); if (args.Length > 1) break; continue; }
        if (string.IsNullOrWhiteSpace(text)) { if (args.Length > 1) break; continue; }
        using var stop = new CancellationTokenSource();
        ConsoleCancelEventHandler handler = (_, e) => { e.Cancel = true; stop.Cancel(); };
        Console.CancelKeyPress += handler;
        try
        {
            var prompt = ByteTokenizer.Prompt(history, text, weights.Config.Context, sampling.MaxNewTokens);
            string answer = model.Generate(prompt, () => sampling, null, stop.Token);
            Console.WriteLine("Trit> " + answer); history.Add(new(text, answer, -1)); if (history.Count > 250) history.RemoveAt(0);
        }
        catch (OperationCanceledException) { Console.WriteLine("Cancelled."); }
        finally { Console.CancelKeyPress -= handler; }
        if (args.Length > 1) break;
    }
    return 0;
}
catch (Exception e) { Console.Error.WriteLine(e.Message); return 1; }
