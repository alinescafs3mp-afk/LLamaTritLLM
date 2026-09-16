namespace TritStudio.Core;

// Stable protocol values. Legacy callers without a mode retain v6's one-click conversation path.
public enum CreationMode { Untrained = 0, BasicPretrain = 1, Conversation = 2 }
public enum TrainingMaterial { Conversation = 0, BasicPretrain = 1, CustomWithReplay = 2 }
public static class LearningStages
{
    public static void Validate(CreationMode mode)
    { if (!Enum.IsDefined(typeof(CreationMode), mode)) throw new ArgumentException("Unknown creation mode."); }
    public static void Validate(TrainingMaterial material)
    { if (!Enum.IsDefined(typeof(TrainingMaterial), material)) throw new ArgumentException("Unknown training material."); }
    public static TrainingOptions CreationOptions(CreationMode mode, TrainingOptions options)
    {
        Validate(mode); options.Validate();
        if (mode == CreationMode.Untrained) return options with { Steps = 0 };
        if (mode == CreationMode.BasicPretrain && options.Steps < 1) throw new ArgumentException("Для выбранного этапа обучения нужен хотя бы один шаг. Для случайных весов выберите «Без обучения».");
        return options;
    }
    public static void ValidateMaterialRequest(TrainingMaterial material, bool includeBundled, int selectedFileCount, bool conversationPreviouslyTrained, bool customPreviouslyTrained = false)
    {
        Validate(material);
        if (selectedFileCount < 0) throw new ArgumentOutOfRangeException(nameof(selectedFileCount));
        if (material == TrainingMaterial.BasicPretrain && selectedFileCount != 0)
            throw new ArgumentException("Базовый претрейн не использует выбранные файлы. Уберите файлы или выберите другой этап.");
        if (material == TrainingMaterial.Conversation && !includeBundled && !conversationPreviouslyTrained)
            throw new ArgumentException("Разговорный корпус ещё не обучался. Включите добавление встроенного корпуса или выберите режим своих данных; базовые тексты нельзя назвать разговорным этапом.");
        if (material == TrainingMaterial.CustomWithReplay && selectedFileCount == 0 && !customPreviouslyTrained)
            throw new ArgumentException("Для отдельного этапа своих данных выберите хотя бы один файл. Для повторения разговорного корпуса выберите разговорное дообучение.");
    }
    public static string Name(TrainingMaterial material) => material switch
    {
        TrainingMaterial.BasicPretrain => "Базовый претрейн (только короткие тексты)",
        TrainingMaterial.Conversation => "Разговорный датасет",
        TrainingMaterial.CustomWithReplay => "Свои данные и повторение ранее изученного",
        _ => throw new ArgumentException("Unknown training material.")
    };
    public static string ArchiveKey(TrainingMaterial material) => material switch
    {
        TrainingMaterial.BasicPretrain => "01-basic",
        TrainingMaterial.Conversation => "02-conversation",
        TrainingMaterial.CustomWithReplay => "03-custom",
        _ => throw new ArgumentException("Unknown training material.")
    };
}
