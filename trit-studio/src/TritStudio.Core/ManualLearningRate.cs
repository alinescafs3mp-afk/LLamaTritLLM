namespace TritStudio.Core;

// Run-local schedule: every explicit manual run starts a NEW plan, not a hidden continuation.
// Existing constant-LR jobs are unchanged. Checkpoints retain the actual applied LR in TrainerState.
public static class ManualLearningRate
{
    public static double At(TrainingOptions options, int completedBeforeStep)
    {
        options.Validate();
        if (completedBeforeStep < 0 || completedBeforeStep >= options.Steps)
            throw new ArgumentOutOfRangeException(nameof(completedBeforeStep));
        if (!options.WarmupCosine || options.Steps == 1) return options.LearningRate;
        int warmup = Math.Min(options.WarmupSteps, options.Steps / 10);
        int next = completedBeforeStep + 1;
        if (warmup > 0 && next <= warmup) return options.LearningRate * next / warmup;
        double progress = (double)(next - warmup - 1) / Math.Max(1, options.Steps - warmup - 1);
        return options.LearningRate * (0.1 + 0.9 * 0.5 * (1 + Math.Cos(Math.PI * progress)));
    }
}
