using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Threading;
using TritStudio.Core;
namespace TritStudio.App;

public sealed partial class MainWindow
{
    private readonly TextBlock _runDraftStatus = Text("Параметры следующего запуска сохраняются отдельно от весов.", 12);
    private readonly Button _saveRunDraft = Button("Сохранить настройки запуска");
    private readonly DispatcherTimer _draftTimer = new() { Interval = TimeSpan.FromMilliseconds(450) };
    private bool _restoringLaunch, _runEditorInitialized, _runDraftDirty, _creationDraftDirty, _draftSaveBlocked;
    private string? _runContext;
    private TrainingOptions _runTrainingBase = new(), _creationTrainingBase = new();
    private ModelConfig _creationConfigBase = new();
    private bool _confirmingUnsavedClose;
    private string CreationDraftPath => Path.Combine(AppPaths.Root, "creation-draft.json");
    // Lazily constructed once, not allocated on every telemetry/UI-state refresh.
    private NumericUpDown[]? _runNumbersCache, _creationNumbersCache, _creationBaseNumbersCache;
    private Control[]? _runEditorsCache, _creationEditorsCache;
    private NumericUpDown[] RunNumbers => _runNumbersCache ??= [_steps, _learningRate, _batch, _sequence, _publishEvery, _threads, _memory];
    private NumericUpDown[] CreationBaseNumbers => _creationBaseNumbersCache ??= [_dimension, _hidden, _layers, _heads, _kvHeads, _context, _planes, _group, _threshold, _newMemory];
    private NumericUpDown[] CreationNumbers => _creationNumbersCache ??= [..CreationBaseNumbers, _newBatch, _newSequence, _newSteps, _newRate, _newPublish];
    private Control[] RunEditors => _runEditorsCache ??= [..RunNumbers, _cuda, _sdpa, _buckets, _targetProjection, _trainingMaterial, _refreshCorpus];
    private Control[] CreationEditors => _creationEditorsCache ??= [..CreationNumbers, _name, _newCuda, _creationMode, _preset];

    private static void ValidateEditorNumbers(IEnumerable<NumericUpDown> numbers)
    {
        foreach (var n in numbers)
        {
            try { _ = CheckedNumber(n); }
            catch (ArgumentException error) { throw new ArgumentException($"{n.Tag ?? "Параметр"}: {error.Message}", error); }
        }
    }
    private LaunchDraft CaptureRunDraft()
    {
        var config = _model?.Weights.Config ?? throw new InvalidOperationException("Сначала выберите модель.");
        ValidateEditorNumbers(RunNumbers);
        var draft = new LaunchDraft(config, SelectedResources(), SelectedTraining(),
            (TrainingMaterial)_trainingMaterial.SelectedIndex, _refreshCorpus.IsChecked == true);
        draft.Validate(); return draft;
    }
    private CreationDraft CaptureCreationDraft()
    {
        ValidateEditorNumbers(CreationNumbers); var config = SelectedConfig();
        var r = CreationResources(config) with { SequenceLength = LaunchInt(_newSequence) };
        var draft = new CreationDraft(_name.Text ?? "", config, r, CreationTraining(), (CreationMode)_creationMode.SelectedIndex);
        draft.Validate(); return draft;
    }
    private void WireLaunchEditors()
    {
        foreach (var c in RunEditors)
            c.PropertyChanged += (_,e) => { if (e.Property.Name is "Text" or "Value" or "IsChecked" or "SelectedIndex") MarkRunEdited(); };
        foreach (var c in CreationEditors)
            c.PropertyChanged += (_,e) => { if (e.Property.Name is "Text" or "Value" or "IsChecked" or "SelectedIndex") MarkCreationEdited(); };
        _draftTimer.Tick += (_,_) => { _draftTimer.Stop(); FlushEditorDrafts(strict:false); };
        _saveRunDraft.Click += async (_,_) => await RunButton(_saveRunDraft, "Сохранение настроек запуска", () =>
        { SaveRunDraft(explicitSave:true); return Task.CompletedTask; });
        try
        {
            if (LaunchDraftStore.ReadCreation(CreationDraftPath) is { } saved)
            {
                _restoringLaunch = true;
                try
                {
                    _creationTrainingBase = saved.Training;
                    _preset.SelectedIndex = -1; ApplyPreset(saved.Config); _name.Text = saved.Name;
                    SetEditorValue(_newMemory, saved.Resources.MemoryMiB); _newCuda.IsChecked = saved.Resources.PreferCuda;
                    SetEditorValue(_newBatch, saved.Resources.BatchSize); SetEditorValue(_newSequence, saved.Resources.SequenceLength);
                    SetEditorValue(_newSteps, saved.Training.Steps); SetEditorValue(_newRate, (decimal)saved.Training.LearningRate);
                    SetEditorValue(_newPublish, saved.Training.PublishEvery); _creationMode.SelectedIndex = (int)saved.Mode;
                }
                finally { _restoringLaunch = false; }
                _creationDraftDirty = false;
            }
        }
        catch (Exception e) { SetError("Настройки создания не восстановлены; исходный файл сохранён. " + e.Message); }
    }
    private void MarkRunEdited()
    {
        if (_restoringLaunch || _settingControls || _closing || !_runEditorInitialized || _runContext is null) return;
        _runDraftDirty = true; _runDraftStatus.Foreground = Muted;
        _runDraftStatus.Text = "Изменено для следующего запуска. Сохраняю… Текущие веса не меняются.";
        _draftTimer.Stop(); _draftTimer.Start();
    }
    private void MarkCreationEdited()
    {
        if (_restoringLaunch || _settingControls || _closing) return;
        _creationDraftDirty = true; _draftTimer.Stop(); _draftTimer.Start();
    }
    private void SetRunEditorContext(string? workspace)
    {
        if (ModelLibrary.SamePath(_runContext, workspace) && _runEditorInitialized) return;
        _runContext = workspace; _runEditorInitialized = false; _runDraftDirty = false; _draftSaveBlocked = false;
        _runDraftStatus.Text = workspace is null ? "Файл открыт только для чата. Настройки дообучения недоступны." : "Загружаю настройки выбранной модели…";
    }
    private void ApplyRunDraft(LaunchDraft d)
    {
        bool previous = _settingControls; _restoringLaunch = true; _settingControls = true;
        try
        {
            _runTrainingBase = d.Training;
            SetEditorValue(_threads, Math.Min(d.Resources.Threads, Environment.ProcessorCount));
            SetEditorValue(_memory, d.Resources.MemoryMiB); SetEditorValue(_batch, d.Resources.BatchSize); SetEditorValue(_sequence, d.Resources.SequenceLength);
            _cuda.IsChecked = d.Resources.PreferCuda; _sdpa.IsChecked = d.Resources.UseSdpa;
            _buckets.IsChecked = d.Resources.BucketByLength; _targetProjection.IsChecked = d.Resources.ProjectOnlyTargets;
            SetEditorValue(_steps, d.Training.Steps); SetEditorValue(_learningRate, (decimal)d.Training.LearningRate); SetEditorValue(_publishEvery, d.Training.PublishEvery);
            _trainingMaterial.SelectedIndex = (int)d.Material; _refreshCorpus.IsChecked = d.IncludeBundledUpdates;
        }
        finally { _settingControls = previous; _restoringLaunch = false; }
        UpdateLearningHint();
    }
    private void ReceiveRunSettings(ReadyEvent r)
    {
        // Ready/status are observations of committed state. They are NEVER permission to overwrite user intent.
        // Existing editors survive repeated ready, stop, quality, publication, rollback and reconnect.
        if (!r.HasModel) return;
        if (r.Config is null || r.Resources is null || r.Training is null)
            throw new InvalidDataException("Тренер объявил модель готовой без её настроек.");
        var training = r.Training ?? new TrainingOptions();
        // Validate repeated observations as well, but never copy them over edited next-run fields.
        r.Resources.ValidateConfiguration(r.Config); training.Validate();
        if (_runEditorInitialized) return;
        bool previousSetting = _settingControls; _settingControls = true;
        try { SetEditorValue(_onlineRate, (decimal)training.OnlineLearningRate); }
        finally { _settingControls = previousSetting; }
        var material = r.LastMaterial is null && training.Steps == 0 ? TrainingMaterial.BasicPretrain : TrainingMaterial.Conversation;
        var draft = new LaunchDraft(r.Config, r.Resources, training with { Steps = training.Steps == 0 ? 200 : training.Steps }, material);
        draft.Validate();
        bool restored = false;
        if (_runContext is not null)
        {
            try { if (LaunchDraftStore.Read(_runContext, r.Config) is { } saved) { draft = saved; restored = true; } }
            catch (Exception e)
            {
                _draftSaveBlocked = true; SetError("Черновик запуска не восстановлен и не перезаписан. Исправьте поля и явно сохраните настройки. " + e.Message);
            }
        }
        ApplyRunDraft(draft); _runEditorInitialized = true; _runDraftDirty = false;
        _runDraftStatus.Text = (restored ? "Восстановлены ВАШИ настройки следующего запуска. " : "Начальные настройки взяты из сохранённой модели. ") +
            "Редактирование сохраняется автоматически; учебный профиль только заполняет пять полей.";
    }
    private void SaveRunDraft(bool explicitSave = false)
    {
        if (_runContext is null || _model is null) throw new InvalidOperationException("Для сохранения выберите рабочую папку модели, не отдельный файл весов.");
        if (_draftSaveBlocked && !explicitSave) throw new IOException("Повреждённый черновик оставлен без изменений. Нажмите «Сохранить настройки запуска», чтобы явно заменить его текущими полями.");
        var draft = CaptureRunDraft(); LaunchDraftStore.Write(_runContext, draft);
        _runDraftDirty = false; _draftSaveBlocked = false; _runDraftStatus.Foreground = Muted;
        _runDraftStatus.Text = $"Сохранено для следующего запуска: LR {draft.Training.LearningRate:G6}; пакет {draft.Resources.BatchSize}; длина {draft.Resources.SequenceLength}; шагов {draft.Training.Steps}; сохранение через {draft.Training.PublishEvery}. Обучение НЕ запущено.";
        if (explicitSave) { _status.Text = _runDraftStatus.Text; Notice("Настройки сохранены", _runDraftStatus.Text); }
    }
    private void FlushEditorDrafts(bool strict, bool includeCreation = true)
    {
        _draftTimer.Stop();
        if (_runDraftDirty)
        {
            try { SaveRunDraft(); }
            catch (Exception e)
            {
                _runDraftStatus.Foreground = new SolidColorBrush(Color.Parse("#FFBCAD"));
                _runDraftStatus.Text = "Настройки НЕ сохранены: " + e.Message;
                if (strict) throw;
            }
        }
        if (includeCreation && _creationDraftDirty)
        {
            try { LaunchDraftStore.WriteCreation(CreationDraftPath, CaptureCreationDraft()); _creationDraftDirty = false; }
            catch (Exception e) { if (strict) throw; _estimate.Text = "Настройки создания НЕ сохранены: " + e.Message; }
        }
    }
    private void UpdateLaunchEditorAvailability()
    {
        bool free = !_closing && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        foreach(var c in RunEditors) c.IsEnabled = free && _runEditorInitialized && _runContext is not null;
        foreach(var c in CreationEditors) c.IsEnabled = free && !_generating;
        _saveRunDraft.IsEnabled = free && _runEditorInitialized && _model is not null && _workspace is not null;
        _trainingPresetButton.IsEnabled = free && _runEditorInitialized && _model is not null && _workspace is not null;
    }
}
