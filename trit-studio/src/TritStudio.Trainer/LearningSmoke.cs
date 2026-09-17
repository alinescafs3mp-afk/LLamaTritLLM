using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

// This gate tests LEARNING + AUTOREGRESSIVE RECALL, not just decreasing teacher-forced loss.
// It only trains a fresh isolated fixture. Nothing here substitutes replies in the application.
public static class LearningSmoke
{
    public static readonly (string Prompt, string Answer)[] Examples =
    [ ("Привет!", "Привет!"), ("Как дела?", "Хорошо."), ("Как тебя зовут?", "Трит."),
      ("Спасибо.", "Пожалуйста."), ("Пока!", "До встречи!"), ("Ты тут?", "Да.") ];
    public static void Run(bool requireCuda, string root)
    {
        var config = new ModelConfig { Dimension = 64, HiddenDimension = 192, Layers = 2,
            Heads = 4, KvHeads = 2, Context = 128, Planes = 2, GroupSize = 32, Seed = 42 };
        var resources = new ResourceOptions { Threads = 2, MemoryMiB = 4096, BatchSize = 8, SequenceLength = 96, PreferCuda = requireCuda };
        var options = new TrainingOptions { LearningRate = 0.001 };
        var examples = Dataset.EncodeAll(Examples.Select(x => Dataset.Make(x.Prompt, x.Answer)).ToArray(), 96, 1);
        using var session = new TrainingSession(WeightSet.Initialize(config), resources, options);
        if (requireCuda && session.Model.Device.type != DeviceType.CUDA)
            throw new InvalidOperationException("Learning smoke requested CUDA but is running on CPU.");
        double before = session.Evaluate(examples); string[] answers = []; int exact = 0; double after = before;
        string file = Path.Combine(root, "learning-smoke.tritmodel");
        for (int step = 1; step <= 1000; step++)
        {
            session.TrainStep(examples, options.LearningRate, CancellationToken.None);
            if (step % 100 != 0) continue;
            after = session.Evaluate(examples);
            if (File.Exists(file)) File.Delete(file);
            ModelFiles.Write(file, session.CapturePublicationMaster(), true);
            var cpu = new ManagedInference(ModelFiles.Read(file, true), step, 1);
            answers = Examples.Select(x => cpu.Generate(ByteTokenizer.Prompt([], x.Prompt, config.Context, 64),
                () => new SamplingOptions(0, 262, 1, 1, 64), null, CancellationToken.None)).ToArray();
            exact = answers.Zip(Examples).Count(x => x.First == x.Second.Answer);
            Console.Error.WriteLine($"Learning smoke {session.DeviceName}: step={step}, loss={after:F4}, recall={exact}/{Examples.Length}");
            if (step >= 200 && exact == Examples.Length && after < 0.20) break;
        }
        var report = new { scope = "Training-set recall only, not unseen conversation quality", device = session.DeviceName,
            before, after, steps = session.Step, exact, total = Examples.Length, answers,
            pipeline = "real gradient updates -> packed disk -> production CPU autoregressive generation" };
        JsonData.AtomicWrite(Path.Combine(root, "learning-smoke.json"), report);
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(report, JsonData.Options));
        if (exact != Examples.Length || after >= 0.20)
            throw new InvalidOperationException("FAIL learning smoke: cannot recall six trained short Russian replies through the real disk/CPU path. " +
                "Investigate training/encoding/export; do not weaken this gate or insert canned responses.");
    }
}
