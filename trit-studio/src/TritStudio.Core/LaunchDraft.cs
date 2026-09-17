namespace TritStudio.Core;

// UI next-run intent, NOT a checkpoint and NOT evidence of the options used by existing weights.
// Stored separately from trainer-owned settings.json. Only the UI holding the workspace lease writes it.
public sealed record LaunchDraft(ModelConfig Config, ResourceOptions Resources, TrainingOptions Training,
    TrainingMaterial Material = TrainingMaterial.Conversation, bool IncludeBundledUpdates = true, int Version = 1)
{
    public void Validate()
    {
        if (Version != 1 || Config is null || Resources is null || Training is null)
            throw new InvalidDataException("Некорректный черновик параметров запуска.");
        Resources.ValidateConfiguration(Config); Training.Validate(); LearningStages.Validate(Material);
    }
}
public sealed record CreationDraft(string Name, ModelConfig Config, ResourceOptions Resources, TrainingOptions Training,
    CreationMode Mode = CreationMode.Untrained, int Version = 1)
{
    public void Validate()
    {
        if (Version != 1 || Name is null || Name.Length > 256 || Config is null || Resources is null || Training is null ||
            Mode is < CreationMode.Untrained or > CreationMode.Conversation)
            throw new InvalidDataException("Некорректный черновик создания модели.");
        // Maximum requested length may be above a new model's context; creation explicitly caps it at launch.
        (Resources with { SequenceLength = Math.Min(Config.Context, Resources.SequenceLength) }).ValidateConfiguration(Config);
        if (Resources.SequenceLength is < 16 or > 2048) throw new ArgumentException("Неверная длина нового запуска.");
        Config.Validate(); Training.Validate();
    }
}
public static class LaunchDraftStore
{
    public const int MaxBytes = 65536;
    public static string FilePath(string workspace) => Path.Combine(Path.GetFullPath(workspace), "next-run.json");
    public static LaunchDraft? Read(string workspace, ModelConfig config)
    {
        string path = FilePath(workspace); Guard(path);
        if (!File.Exists(path)) return null;
        var draft = JsonData.Read<LaunchDraft>(path, MaxBytes); draft.Validate();
        if (draft.Config != config) throw new InvalidDataException("Черновик запуска относится к другой архитектуре. Исходный файл сохранён: " + path);
        return draft;
    }
    public static void Write(string workspace, LaunchDraft draft)
    {
        draft.Validate(); string path = FilePath(workspace); Guard(path); JsonData.AtomicWrite(path, draft, MaxBytes);
    }
    public static CreationDraft? ReadCreation(string path)
    {
        Guard(path); if (!File.Exists(path)) return null;
        var draft = JsonData.Read<CreationDraft>(path, MaxBytes); draft.Validate(); return draft;
    }
    public static void WriteCreation(string path, CreationDraft draft)
    { draft.Validate(); Guard(path); JsonData.AtomicWrite(path, draft, MaxBytes); }
    private static void Guard(string path)
    {
        ModelLibrary.NoLinks(path);
        if (Directory.Exists(path)) throw new IOException("Вместо файла настроек расположен каталог: " + path);
    }
}
