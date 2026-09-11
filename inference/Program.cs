//https://github.com/virex-84

//inference Program.cs

using System;
using System.Text;
using TernaryLLM;

class Program
{
    static TernaryLlamaModel? model;

    static void Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;

        // ── Параметры командной строки ───────────────────────
        string modelPath = args.Length > 0 ? args[0] : "ternary_model.bin";
        float temperature = 0.7f;
        string? oncePrompt = null;

        for (int i = 1; i < args.Length; i++)
        {
            string a = args[i];

            if (a == "--temp" && i + 1 < args.Length)
            {
                temperature = Math.Clamp(float.Parse(args[i + 1]), 0.01f, 2.0f);
                i++;
            }
            else if (a == "--prompt" && i + 1 < args.Length)
            {
                oncePrompt = args[i + 1];
                i++;
            }
            else if (a == "--help")
            {
                PrintUsage();
                return;
            }
        }

        // ── Загрузка модели ──────────────────────────────────
        if (!Path.IsPathRooted(modelPath))
            modelPath = Path.Combine(AppContext.BaseDirectory, modelPath);

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Файл не найден: {modelPath}");
            PrintUsage();
            return;
        }

        try
        {
            Console.WriteLine($"Загрузка тернарной модели из {modelPath}...");

            model = TernaryLlamaModel.Load(modelPath);

            Console.WriteLine(" ✓ Модель загружена");
            Console.WriteLine($"   Параметров: {model.TotalParameters():N0}");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка загрузки модели: {ex.Message}");
            return;
        }

        // ── Одиночный промпт из командной строки ─────────────
        if (oncePrompt != null)
        {
            Console.WriteLine($"[temp={temperature:F1}] You: {oncePrompt}");
            OnPrompt(oncePrompt, temperature);
            return;
        }

        // ── Интерактивный чат ────────────────────────────────
        Console.WriteLine("╔═══════════════════════════════════════╗");
        Console.WriteLine("║  Ternary LLM — инференс (без обучения)║");
        Console.WriteLine("╠═══════════════════════════════════════╣");
        Console.WriteLine("║  exit   — выход                       ║");
        Console.WriteLine("║  temp N — установить temperature      ║");
        Console.WriteLine("╚═══════════════════════════════════════╝");

        while (true)
        {
            Console.Write($"\n[temp={temperature:F1}] You: ");
            string? input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input)) continue;

            string trimmed = input.Trim();

            if (trimmed.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            if (trimmed.StartsWith("temp ", StringComparison.OrdinalIgnoreCase))
            {
                if (float.TryParse(trimmed[5..].Trim(), out float t))
                {
                    temperature = Math.Clamp(t, 0.01f, 2.0f);
                    Console.WriteLine($"  Temperature установлена: {temperature:F2}");
                }
                continue;
            }

            OnPrompt(trimmed, temperature);
        }
    }

    static void OnPrompt(string text, float temperature)
    {
        var m = model;

        if (m == null)
        {
            Console.WriteLine("  Модель не загружена.");
            return;
        }

        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            string response = m.Predict(text, temperature);
            sw.Stop();

            Console.WriteLine($"Assistant: {response}");
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  [{sw.ElapsedMilliseconds} ms]");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Ошибка генерации: {ex.Message}");
        }
    }

    static void PrintUsage()
    {
        Console.WriteLine("Использование:");
        Console.WriteLine("  TernaryInference.exe [path/to/ternary_model.bin] [--temp T] [--prompt \"текст\"]");
        Console.WriteLine("");
        Console.WriteLine("  Без --prompt запускается интерактивный чат.");
    }
}