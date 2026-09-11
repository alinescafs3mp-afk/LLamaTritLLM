//https://github.com/virex-84

//training Program.cs

using System;
using System.Data;
using System.Text;
using TernaryLLM;

class Program
{
    // ── Пути по умолчанию ────────────────────────────────────
    const string DefaultPretrainData = "data/pretraining_data.json";
    const string DefaultChatData = "data/chat_training_data.json";

    const string MasterModel = "models/master_model.bin";
    const string TernaryModel = "models/ternary_model.bin";

    static TrainableLlamaModel? master;

    static void Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;

    Main:
        Console.WriteLine("╔════════════════════════════════╗");
        Console.WriteLine("║   Ternary LLM  — Главное меню  ║");
        Console.WriteLine("╠════════════════════════════════╣");
        Console.WriteLine("║  1. Загрузить модель           ║");
        Console.WriteLine("║  2. Создать модель             ║");
        Console.WriteLine("╚════════════════════════════════╝");
        Console.Write("\nВыберите опцию (1 или 2): ");

        var choice = Console.ReadLine()?.Trim();
        if (choice == "1") { master = TryLoadModel(); if (master is null) goto Main; }
        else if (choice == "2") { master = CreateModel(); if (master is null) goto Main; }
        else goto Main;

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("╔═══════════════════════════════╗");
            Console.WriteLine("║          Меню действий        ║");
            Console.WriteLine("╠═══════════════════════════════╣");
            Console.WriteLine("║  Мастер-модель:               ║");
            Console.WriteLine("║                               ║");
            Console.WriteLine("║  1. Pretrain                  ║");
            Console.WriteLine("║  2. Tune                      ║");
            Console.WriteLine("║  3. Сохранить модель          ║");
            Console.WriteLine("║  4. Информация о модели       ║");
            Console.WriteLine("║  5. Экспорт тернарной модели  ║");
            Console.WriteLine("║  6. Тест токенизатора         ║");
            Console.WriteLine("╠═══════════════════════════════╣");
            Console.WriteLine("║  Тернарная-модель:            ║");
            Console.WriteLine("║                               ║");
            Console.WriteLine("║  7. Тест Pretrain             ║");
            Console.WriteLine("║  8. Интерактивный чат         ║");
            Console.WriteLine("╠═══════════════════════════════╣");
            Console.WriteLine("║  9. Выход                     ║");
            Console.WriteLine("╚═══════════════════════════════╝");
            Console.Write("\nВыберите опцию: ");

            var action = Console.ReadLine()?.Trim();
            switch (action)
            {
                case "1": PretrainModel(master); break;
                case "2": TuneModel(master); break;
                case "3": SaveModel(master); break;
                case "4": ShowModelInfo(master); break;
                case "5": ExportToTernary(master); break;
                case "6": TestTokenizer(master.Tokenizer); break;

                case "7": TestPretrainTernary(master); break;
                case "8": InteractiveChatTernary(master); break;

                case "9":
                    Console.WriteLine("Выход из программы.");
                    return;
                default:
                    Console.WriteLine("Неверная опция.");
                    break;
            }
        }
    }

    private static TrainableLlamaModel? CreateModel()
    {
        int embDim = ReadIntWithDefault("Embedding dim", 64, 32, 1024);
        int hidDim = ReadIntWithDefault("FFN hidden dim", 64, 64, 4096);

        int numHeads = ReadIntWithDefault("Num heads", 1, 1, 32);
        int numKvHeads = ReadIntWithDefault("Num KV heads", 1, 1, numHeads);

        int numLayers = ReadIntWithDefault("Num layers", 1, 1, 24);
        int maxSeqLen = ReadIntWithDefault("Max seq len", 512, 16, 8192);

        int vocabSize = ReadIntWithDefault("BPE vocab size", 16000, 300, 8192);
        int planes = ReadIntWithDefault("Trit planes 1 (1.58 bit), 2 (INT3), 6 (INT6), etc", 1, 1, 100);

        var dataset = new DatasetLoader(
            ResolvePath(DefaultPretrainData),
            ResolvePath(DefaultChatData),
            DatasetType.JSON);

        var pretrainTexts = dataset.PretrainingData
            .Select(t => t.Replace("</s>", "").Trim())
            .ToList();

        var finetuneTexts = dataset.ChatTrainingData
            .Select(FormatChatForBpe)
            .Where(t => t != null)
            .Select(t => t!)
            .ToList();

        var allTexts = pretrainTexts.Concat(finetuneTexts).ToList();

        // Токенизатор обучается ДО модели
        var tokenizer = new BPETokenizer();

        tokenizer.Train(
            corpus: allTexts,
            vocabSize: vocabSize,
            minFrequency: 1,
            verbose: true);

        int actualVocabSize = tokenizer.VocabSize;

        if (actualVocabSize != vocabSize)
        {
            Console.WriteLine(
                $"BPE vocab: requested {vocabSize}, actual {actualVocabSize}. Using actual.");

            vocabSize = actualVocabSize;
        }

        var config = new TernaryLlamaConfig
        {
            VocabSize = vocabSize,
            EmbeddingDim = embDim,
            HiddenDim = hidDim,
            NumHeads = numHeads,
            NumKVHeads = numKvHeads,
            NumLayers = numLayers,
            MaxSeqLen = maxSeqLen,

            GroupSize = Math.Min(embDim, 64),
            ScaleMode = TritScaleMode.PerGroup,

            AttentionMode = AttentionMode.Softmax,
            QuantizeActivations = false,
            TiedOutput = true,

            RopeTheta = 10000f
        };

        config.SetUniformPlanes(planes);

        return new TrainableLlamaModel(config, tokenizer);
    }

    private static TrainableLlamaModel? TryLoadModel()
    {
        Console.Write($"\nПуть к файлу модели (по умолчанию: {MasterModel}): ");

        string? filePath = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(filePath))
            filePath = $"{MasterModel}";

        if (!Path.IsPathRooted(filePath))
            filePath = Path.Combine(AppContext.BaseDirectory, filePath);

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Файл не найден: {filePath}");
            return null;
        }

        try
        {
            Console.WriteLine($"Загрузка модели из {filePath}...");

            var loaded = TrainableLlamaModel.LoadWeightsOnly(filePath);

            long sizeKB = new FileInfo(filePath).Length / 1024;

            Console.WriteLine($" ✓ Модель загружена ({sizeKB} KB)");

            return loaded;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка загрузки: {ex.Message}");
            return null;
        }
    }

    private static void PretrainModel(TrainableLlamaModel model)
    {
        Console.WriteLine("\n=== PRETRAIN ===\n");

        int pretrainEpochs = ReadIntWithDefault("Pretrain epochs", 300, 1, 100000);
        float pretrainLr = ReadFloatWithDefault("Pretrain LR (0.0001..0.01)", 0.001f);
        int batchSize = ReadIntWithDefault("Batch size", 4, 1, 64);
        float weightDecay = ReadFloatWithDefault("Weight decay", 0.01f);
        float gradientClip = ReadFloatWithDefault("Gradient clip", 1.0f);

        string pretrainPath = ResolvePath(DefaultPretrainData);

        var dataset = new DatasetLoader(pretrainPath, "", DatasetType.JSON);

        var pretrainTexts = dataset.PretrainingData
            .Select(t => model.Tokenizer.FormatPretraining(t.Replace("</s>", "").Trim()))
            .Where(t => !string.IsNullOrEmpty(t))
            .ToList();

        if (pretrainTexts.Count == 0)
        {
            Console.WriteLine("  Нет pretrain данных.");
            return;
        }

        Console.WriteLine($"\n  Текстов: {pretrainTexts.Count}");

        var allTokens = new List<int>();

        foreach (var text in pretrainTexts)
        {
            var ids = model.Tokenizer.Encode(text, addBos: true, addEos: true);
            allTokens.AddRange(ids);
        }

        Console.WriteLine($"  Токенов: {allTokens.Count}");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        model.Train(
            allTokens,
            pretrainEpochs,
            batchSize: batchSize,
            lr: pretrainLr,
            weightDecay: weightDecay,
            gradientClip: gradientClip,
            labelSmoothing: 0.0f,
            verbose: true);

        sw.Stop();

        Console.WriteLine($"\n  Обучение завершено за {sw.Elapsed.TotalSeconds:F1}s");
    }

    private static void TuneModel(TrainableLlamaModel model)
    {
        Console.WriteLine("\n=== FINETUNE (CHAT) ===\n");

        int finetuneEpochs = ReadIntWithDefault("Finetune epochs", 700, 1, 100000);
        float finetuneLr = ReadFloatWithDefault("Finetune LR (0.0001..0.01)", 0.001f);
        int batchSize = ReadIntWithDefault("Batch size", 4, 1, 64);
        float weightDecay = ReadFloatWithDefault("Weight decay", 0.01f);
        float gradientClip = ReadFloatWithDefault("Gradient clip", 1.0f);

        string funetunePath = ResolvePath(DefaultChatData);

        var dataset = new DatasetLoader("", funetunePath, DatasetType.JSON);

        var finetuneTexts = dataset.ChatTrainingData
            .Select(FormatChatForBpe)
            .Where(t => t != null)
            .Select(t => t!)
            .ToList();

        if (finetuneTexts.Count == 0)
        {
            Console.WriteLine("  Нет funetune данных.");
            return;
        }

        Console.WriteLine($"\n  Текстов: {finetuneTexts.Count}");

        var allTokens = new List<int>();

        foreach (var text in finetuneTexts)
        {
            var ids = model.Tokenizer.Encode(text, addBos: true, addEos: true);
            allTokens.AddRange(ids);
        }

        Console.WriteLine($"  Токенов: {allTokens.Count}");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        model.Train(
            allTokens,
            finetuneEpochs,
            batchSize: batchSize,
            lr: finetuneLr,
            weightDecay: weightDecay,
            gradientClip: gradientClip,
            labelSmoothing: 0.0f,
            verbose: true);

        sw.Stop();

        Console.WriteLine($"\n  Обучение завершено за {sw.Elapsed.TotalSeconds:F1}s");
    }

    private static void SaveModel(TrainableLlamaModel mastermodel)
    {
        Console.Write($"\nПуть для сохранения (по умолчанию: {MasterModel}): ");

        string? filePath = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(filePath))
            filePath = $"{MasterModel}";

        if (!Path.IsPathRooted(filePath))
            filePath = Path.Combine(AppContext.BaseDirectory, filePath);

        try
        {
            mastermodel.SaveWeightsOnly(filePath);

            long sizeKB = new FileInfo(filePath).Length / 1024;

            Console.WriteLine($" ✓ Модель сохранена: {filePath}");
            Console.WriteLine($"  Размер: {sizeKB:N0} KB");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка сохранения: {ex.Message}");
        }
    }

    private static void ShowModelInfo(TrainableLlamaModel mastermodel)
    {
        var cfg = mastermodel.Config;

        Console.WriteLine("");
        Console.WriteLine(" ИНФОРМАЦИЯ О МОДЕЛИ");
        Console.WriteLine("");

        Console.WriteLine($" Архитектура:  TernaryLlama / multi-plane ternary");
        Console.WriteLine($" Параметры: {mastermodel.TotalParameters():N0} latent weights");

        Console.WriteLine($" Vocab size: {cfg.VocabSize:N0}");
        Console.WriteLine($" Embedding: {cfg.EmbeddingDim}");
        Console.WriteLine($" FFN hidden: {cfg.HiddenDim}");

        Console.WriteLine($" Num heads: {cfg.NumHeads}");
        Console.WriteLine($" Num KV heads: {cfg.NumKVHeads}");
        Console.WriteLine($" Head dim: {cfg.HeadDim}");
        Console.WriteLine($" KV dim: {cfg.KVDim}");

        Console.WriteLine($" Num layers: {cfg.NumLayers}");
        Console.WriteLine($" Max seq len: {cfg.MaxSeqLen}");
        Console.WriteLine($" RoPE theta: {cfg.RopeTheta}");

        Console.WriteLine("");
        Console.WriteLine(" ПЛОТНОСТЬ MULTI-PLANE TERNARY");
        Console.WriteLine("");

        Console.WriteLine($" Embedding     : {cfg.EmbeddingPlanes} planes");
        Console.WriteLine($" Attention QKV : {cfg.AttentionQkvPlanes} planes");
        Console.WriteLine($" Attention Out : {cfg.AttentionOutPlanes} planes");

        Console.WriteLine($" FFN Gate      : {cfg.FfnGatePlanes} planes");
        Console.WriteLine($" FFN Up        : {cfg.FfnUpPlanes} planes");
        Console.WriteLine($" FFN Down      : {cfg.FfnDownPlanes} planes");

        if (cfg.TiedOutput)
        {
            Console.WriteLine($" LM head       : tied with embedding");
        }
        else
        {
            Console.WriteLine($" LM head       : {cfg.LmHeadPlanes} planes");
        }

        if (cfg.QuantizeActivations)
        {
            Console.WriteLine($" Activations   : {cfg.ActivationPlanes} planes");
        }
        else
        {
            Console.WriteLine($" Activations   : FP32");
        }

        Console.WriteLine("");
    }

    private static void ExportToTernary(TrainableLlamaModel mastermodel)
    {
        Console.Write($"\nПуть к файлу модели (по умолчанию: {TernaryModel}): ");

        string? filePath = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(filePath))
            filePath = $"{TernaryModel}";

        var ternaryModel = new TernaryLlamaModel(mastermodel.Config, mastermodel.Tokenizer);
        mastermodel.SyncToInferenceModel(ternaryModel);

        ternaryModel.Save(filePath);
    }

    private static void TestTokenizer(BPETokenizer tokenizer)
    {
        Console.WriteLine("\n═══ Тест BPE токенизатора ═══");
        Console.WriteLine("Введите текст (пустая строка = выход):");

        while (true)
        {
            Console.Write("\n> ");
            string? input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input)) break;

            var ids = tokenizer.Encode(input, addBos: false, addEos: false);
            string decoded = tokenizer.Decode(ids);

            Console.WriteLine(
                $"  Token IDs [{ids.Count}]: [{string.Join(", ", ids)}]");

            var subwords = ids
                .Select(id => tokenizer.ToVocab().DecodeToken(id) ?? "?")
                .ToList();
            Console.WriteLine(
                $"  Подслова: [{string.Join("|", subwords)}]");
            Console.WriteLine($"  Decoded: \"{decoded}\"");
        }
    }

    private static void TestPretrainTernary(TrainableLlamaModel model)
    {
        var dataset = new DatasetLoader(
            ResolvePath(DefaultPretrainData),
            ResolvePath(DefaultChatData),
            DatasetType.JSON);

        var ternaryModel = new TernaryLlamaModel(model.Config, model.Tokenizer);
        model.SyncToInferenceModel(ternaryModel);

        foreach (var item in dataset.PretrainingData)
        {
            var f = item.Split(" ").First();

            Console.WriteLine(f + " -> " + ternaryModel.PredictRaw(f, temperature: 0f) + ".");
        }
    }

    static void InteractiveChatTernary(TrainableLlamaModel mastermodel)
    {
        var ternaryModel = new TernaryLlamaModel(mastermodel.Config, mastermodel.Tokenizer);
        mastermodel.SyncToInferenceModel(ternaryModel);

        Console.WriteLine("\n╔══════════════════════════════════════╗");
        Console.WriteLine("║         Интерактивный чат            ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine("║  exit   — выход                      ║");
        Console.WriteLine("║  temp N — установить temperature     ║");
        Console.WriteLine("║  clear  — очистить экран             ║");
        Console.WriteLine("╚══════════════════════════════════════╝");

        float temperature = 0.7f;

        while (true)
        {
            Console.Write($"\n[temp={temperature:F1}] You: ");
            string? userInput = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userInput)) continue;

            string trimmed = userInput.Trim();

            if (trimmed.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Выход из чата.");
                break;
            }

            if (trimmed.Equals("clear", StringComparison.OrdinalIgnoreCase))
            {
                Console.Clear();
                continue;
            }

            if (trimmed.StartsWith("temp ", StringComparison.OrdinalIgnoreCase))
            {
                if (float.TryParse(trimmed[5..].Trim(), out float newTemp))
                {
                    temperature = Math.Clamp(newTemp, 0.01f, 2.0f);
                    Console.WriteLine(
                        $"  Temperature установлена: {temperature:F2}");
                }
                continue;
            }

            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                string response = ternaryModel.Predict(trimmed, temperature);
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
    }

    static int ReadIntWithDefault(
        string prompt, int defaultValue,
        int min = 0, int max = int.MaxValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue}): ");
        string? input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input)) return defaultValue;
        if (int.TryParse(input, out int value))
            return Math.Clamp(value, min, max);
        Console.WriteLine($"  Некорректный ввод, используется {defaultValue}");
        return defaultValue;
    }

    static float ReadFloatWithDefault(string prompt, float defaultValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue:G4}): ");
        string? input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input)) return defaultValue;

        if (float.TryParse(input,
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,
            out float value))
            return value;

        if (float.TryParse(input, out value))
            return value;

        Console.WriteLine(
            $"  Некорректный ввод, используется {defaultValue:G4}");
        return defaultValue;
    }

    static string ResolvePath(string relativePath)
    {
        if (Path.IsPathRooted(relativePath))
            return relativePath;
        return Path.Combine(AppContext.BaseDirectory, relativePath);
    }

    static string? FormatChatForBpe(string text)
    {
        string clean = text.Replace("</s>", "").Trim();

        int userIdx = clean.IndexOf(
            "User:", StringComparison.OrdinalIgnoreCase);
        int assistantIdx = clean.IndexOf(
            "Assistant:", StringComparison.OrdinalIgnoreCase);

        if (userIdx >= 0 && assistantIdx > userIdx)
        {
            string userText = clean[(userIdx + 5)..assistantIdx]
                .Trim().TrimEnd(':').Trim();
            string assistantText = clean[(assistantIdx + 10)..]
                .Trim().TrimEnd(':').Trim();

            if (!string.IsNullOrEmpty(userText) &&
                !string.IsNullOrEmpty(assistantText))
                return $"<user> {userText}<assistant> {assistantText}";
        }

        if (!string.IsNullOrEmpty(clean))
            return clean.Trim();

        return null;
    }

}