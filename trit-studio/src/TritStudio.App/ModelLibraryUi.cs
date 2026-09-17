using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Layout;
using Avalonia.Media;
using TritStudio.Core;

namespace TritStudio.App;

public sealed partial class MainWindow
{
    private readonly ComboBox _modelPicker = new() { PlaceholderText = "Выберите модель", MinWidth = 200, MaxDropDownHeight = 320, HorizontalAlignment = HorizontalAlignment.Stretch };
    private readonly Button _selectModelButton = Button("Открыть"), _manageModelsButton = Button("Модели…"), _chatSettingsButton = Button("Параметры ответа ▸"), _diagnosticsButton = Button("Сохранить диагностику");
    private readonly ComboBox _uiScale = new() { ItemsSource = new[] { "50%", "65%", "80%", "100%", "125%" }, MinWidth = 78, SelectedIndex = 2 };
    private readonly NumericUpDown _publishEvery = Number(100,1,1000,25);
    private readonly TextBlock _selectedModelInfo = Text("Выберите модель в шапке или создайте новую.",13);
    private LayoutTransformControl? _scaledShell;
    private Grid? _chatGrid;
    private ScrollViewer? _chatSidebar;
    private Expander? _newModelPane;
    private ModelEntry[] _library = [];
    private bool _updatingCatalog, _settingScale;
    private long _catalogScan;
    private string? CurrentModelPath => _workspace ?? _inferenceFile;

    private void WireLibrary()
    {
        _uiScale.SelectionChanged += (_,_) =>
        {
            if (_settingScale) return;
            double[] scales = [0.5,0.65,0.8,1,1.25];
            ApplyUiScale(scales[Math.Clamp(_uiScale.SelectedIndex,0,4)],true);
        };
        _chatSettingsButton.Click += (_,_) => SetSidebar(_chatSidebar?.IsVisible != true,true);
        _selectModelButton.Click += async (_,_) =>
        {
            if (_modelPicker.SelectedItem is ModelEntry entry) await RunButton(_selectModelButton,"Выбор модели",()=>SwitchSelectedModel(entry));
        };
        _modelPicker.SelectionChanged += async (_,_) =>
        {
            if (_updatingCatalog || _modelPicker.SelectedItem is not ModelEntry selected || ModelLibrary.SamePath(selected.Path,CurrentModelPath)) return;
            // The visible selection continues to identify actual weights while another model is prepared.
            SelectActualModel();
            await RunButton(_selectModelButton,"Выбор модели",()=>SwitchSelectedModel(selected));
        };
        _manageModelsButton.Click += async (_,_) => await RunButton(_manageModelsButton,"Управление моделями",ManageModels);
        _diagnosticsButton.Click += async (_,_) => await RunButton(_diagnosticsButton,"Диагностика модели",SaveDiagnostics);
    }
    private void ApplyUiScale(double scale, bool persist)
    {
        double[] supported = [0.5,0.65,0.8,1,1.25];
        if (!supported.Contains(scale)) scale = 0.8;
        _settingScale = true;
        try
        {
            if (_scaledShell is not null) _scaledShell.LayoutTransform = new ScaleTransform(scale,scale);
            _uiScale.SelectedIndex = Array.IndexOf(supported,scale);
            _preferences = _preferences with { UiScale = scale };
        }
        finally { _settingScale = false; }
        if (persist) { SavePreferences(); Notice("Масштаб применён", $"{scale:P0}. Параметры модели не изменены."); }
    }
    private void SetSidebar(bool visible, bool persist)
    {
        if (_chatSidebar is null || _chatGrid is null) return;
        _chatSidebar.IsVisible = visible;
        _chatGrid.ColumnDefinitions = new ColumnDefinitions(visible ? "*,270" : "*,0");
        _chatSettingsButton.Content = visible ? "Скрыть параметры ◂" : "Параметры ответа ▸";
        _preferences = _preferences with { ChatSettingsVisible = visible };
        if (persist) { SavePreferences(); Notice(visible ? "Параметры открыты" : "Параметры скрыты","Настройки генерации сохранены; текущие веса не изменены."); }
    }
    private void UpdateLibraryState()
    {
        bool free = !_closing && !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _modelPicker.IsEnabled = free;
        _selectModelButton.IsEnabled = free && _library.Length > 0;
        _manageModelsButton.IsEnabled = free;
        _diagnosticsButton.IsEnabled = !_closing && _model is not null && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
    }
    private void UpdateSelectedInfo()
    {
        var c = _model?.Weights.Config;
        string text = c is null ? "Активной модели нет. Создание другой сети находится в отдельном блоке ниже." :
            $"{Path.GetFileName(Path.TrimEndingDirectorySeparator(CurrentModelPath ?? ""))}\n" +
            $"{c.ParameterCount:N0} параметров · ревизия {_model!.Revision} · шаг {_workerStatus?.Step ?? _revision?.Step ?? 0:N0}\n" +
            $"Ширина {c.Dimension} · FFN {c.HiddenDimension} · слоёв {c.Layers} · голов {c.Heads}/{c.KvHeads}\n" +
            $"Контекст {c.Context} байт-токенов · плоскостей {c.Planes} · группа {c.GroupSize} · порог {c.Threshold:F2}\n" +
            $"Контрольная ошибка: {(_revision?.ValidationLoss is double loss ? loss.ToString("F4") : "нет оценки")}\n" +
            (_workspace is null ? "Только чат. Для обучения выберите рабочую папку, содержащую мастер-веса." : "Обе вкладки работают с этой моделью. Архитектура только для чтения.");
        if (_selectedModelInfo.Text != text) _selectedModelInfo.Text = text;
    }
    private void RememberModel(string path)
    {
        path = Path.GetFullPath(path);
        var previous = _preferences.RecentModels ?? [];
        _preferences = _preferences with { RecentModels = new[] { path }.Concat(previous.Where(x=>!ModelLibrary.SamePath(x,path))).Take(128).ToArray() };
        SavePreferences();
    }
    private void SelectActualModel()
    {
        _updatingCatalog = true;
        try { _modelPicker.SelectedItem = _library.FirstOrDefault(x=>ModelLibrary.SamePath(x.Path,CurrentModelPath)); }
        finally { _updatingCatalog = false; }
    }
    private async Task RefreshLibrary()
    {
        long request = ++_catalogScan;
        var remembered = (_preferences.RecentModels ?? []).ToArray();
        if (CurrentModelPath is { } active) remembered = remembered.Append(active).ToArray();
        ModelEntry[] entries;
        try { entries = await Task.Run(()=>ModelLibrary.Scan(AppPaths.Models,remembered,_windowLife.Token),_windowLife.Token); }
        catch(Exception e) when(e is IOException or UnauthorizedAccessException or ArgumentException)
        {
            _error.Text = "Каталог не обновлён полностью: " + e.Message + ". Открытая модель не отключена."; _error.IsVisible = true; Log(_error.Text);
            entries = _library;
            if(CurrentModelPath is { } current && !entries.Any(x=>ModelLibrary.SamePath(x.Path,current)))
                entries = entries.Append(new ModelEntry(current,Path.GetFileName(current),_workspace is not null,false,"Каталог не проверен")).ToArray();
        }
        if (_closing || request != _catalogScan) return;
        _updatingCatalog = true;
        try { _library = entries; _modelPicker.ItemsSource = entries; _modelPicker.SelectedItem = entries.FirstOrDefault(x=>ModelLibrary.SamePath(x.Path,CurrentModelPath)); }
        finally { _updatingCatalog = false; }
        UpdateLibraryState();
    }
    private void ClearModelSpecificInputs()
    {
        _qualitySummary.Text = "Проверка этой модели ещё не выполнялась.";
        _datasetPaths = []; _pathsLabel.Text = "Выбранные пользовательские файлы сброшены при переключении модели.";
        _dataSummary.Text = "Учебный материал появится после подключения тренера.";
        _input.Text = ""; _private.IsChecked = false; _followTail = true;
    }
    private async Task SwitchSelectedModel(ModelEntry selected)
    {
        if (ModelLibrary.SamePath(selected.Path,CurrentModelPath)) { _status.Text = "Эта модель уже активна в обеих вкладках."; return; }
        if (_workerStatus?.Busy == true && !await Confirm("Переключить модель?","Текущий тренер будет остановлен. Сохраняется последний опубликованный снимок; ещё не опубликованный шаг может быть потерян.")) throw new OperationCanceledException();
        try
        {
            if (selected.Workspace) await OpenWorkspace(selected.Path); else await OpenPackedPath(selected.Path);
            _status.Text = selected.Workspace ? "Выбрана модель: " + selected.Name + (_workerReady ? ". Чат и обучение используют её." : ". CPU-чат доступен; тренер не подключён.") : "Выбрана модель только для чата: " + selected.Name;
        }
        finally { SelectActualModel(); UpdateSelectedInfo(); }
    }
    private bool UsesEntry(ModelEntry entry)
    {
        string? current = CurrentModelPath;
        return ModelLibrary.SamePath(entry.Path,current) || entry.Workspace && current is not null &&
            Path.GetFullPath(current).StartsWith(Path.GetFullPath(entry.Path)+Path.DirectorySeparatorChar,ModelLibrary.PathComparison);
    }
    private async Task DisconnectSelectedModel()
    {
        FlushEditorDrafts(strict:true, includeCreation:false);
        CancelPendingModeEdit(); ++_epoch; ++_publication; CancelSnapshotLoad(); _workerReady = false;
        if (_client is not null) { await _client.DisposeAsync(); _client = null; }
        await _modelLoader.WaitAsync();
        try { _uiLease?.Dispose(); _uiLease = null; }
        finally { _modelLoader.Release(); }
        _workspace = null; SetRunEditorContext(null); _inferenceFile = null; _model = null; _revision = null; _pending = null; _workerStatus = null;
        _history.Clear(); _messages.Children.Clear(); _workspaceLabel.Text = ""; _modelLabel.Text = "Модель не выбрана";
        _conversationId = Guid.NewGuid().ToString("N"); ClearModelSpecificInputs(); SetOnlineCheck(false);
        UpdateState(); UpdateContextBudget(); UpdateStats(); SelectActualModel();
    }
    private async Task RemoveEntry(ModelEntry entry)
    {
        if (!entry.Managed)
        {
            if (entry.Workspace && ModelLibrary.IsManaged(AppPaths.Models,entry.Path))
                throw new IOException("Этот каталог автоматически обнаружен, но его безопасное перемещение не подтверждено (например, это ссылка). Файлы не изменены; проверьте путь вручную.");
            if (UsesEntry(entry)) await DisconnectSelectedModel();
            _preferences = _preferences with { RecentModels = (_preferences.RecentModels ?? []).Where(p=>!ModelLibrary.SamePath(p,entry.Path)).ToArray() };
            SavePreferences(); _status.Text = "Внешняя модель убрана из списка. Файлы на диске НЕ удалены.";
            await RefreshLibrary(); return;
        }
        // Validate before disconnecting. Directory.Move is non-destructive and uses two live lock handles.
        ModelLibrary.ValidateRemoval(AppPaths.Models,entry.Path,AppPaths.Trash);
        bool current = UsesEntry(entry);
        string? previousWorkspace = _workspace, previousFile = _inferenceFile;
        string? previousDraft = _input.Text; bool previousPrivate = _private.IsChecked == true;
        if (current) await DisconnectSelectedModel();
        try
        {
            var result = await Task.Run(()=>ModelLibrary.MoveToTrash(AppPaths.Models,entry.Path,AppPaths.Trash,_windowLife.Token),_windowLife.Token);
            _preferences = _preferences with { RecentModels = (_preferences.RecentModels ?? []).Where(p=>!ModelLibrary.SamePath(p,entry.Path) && !Path.GetFullPath(p).StartsWith(entry.Path+Path.DirectorySeparatorChar,ModelLibrary.PathComparison)).ToArray() };
            SavePreferences(); _status.Text = "Модель перемещена в локальную корзину: " + result.Directory + ". Веса, снимки и переписка сохранены там; место не освобождено.";
        }
        catch
        {
            // If removal failed, attempt to restore the original active workspace, but never hide the failure.
            if (current && !_closing && Directory.Exists(entry.Path))
                try
                {
                    if (previousWorkspace is not null) await OpenWorkspace(previousWorkspace);
                    else if (previousFile is not null) await OpenPackedPath(previousFile);
                    _input.Text = previousDraft; _private.IsChecked = previousPrivate;
                }
                catch(Exception e) { Log("Не удалось восстановить подключение после отказа удаления: " + e.Message); }
            throw;
        }
        finally { if (!_closing) await RefreshLibrary(); }
    }
    private async Task ManageModels()
    {
        await RefreshLibrary();
        var dialog = new Window { Title = "Модели", Width = 640, Height = 320, MinWidth = 440, WindowStartupLocation = WindowStartupLocation.CenterOwner, Background = Background };
        var chooser = new ComboBox { ItemsSource = _library, HorizontalAlignment = HorizontalAlignment.Stretch, SelectedItem = _library.FirstOrDefault(x=>ModelLibrary.SamePath(x.Path,CurrentModelPath)) ?? _library.FirstOrDefault() };
        var detail = Text("",12); var remove = Button("Удалить выбранную модель…"); var trash = Button("Открыть корзину"); var close = Button("Закрыть");
        void Describe()
        {
            var entry = chooser.SelectedItem as ModelEntry; remove.IsEnabled = entry is not null;
            remove.Content = entry?.Managed == true ? "Переместить в корзину…" : "Убрать внешний путь из списка";
            detail.Text = entry is null ? "Моделей в каталоге пока нет." : entry.Path + "\n" + (entry.Managed ? "В корзину перемещается вся рабочая папка, включая переписку и снимки. Это не безвозвратное удаление." : "Внешние файлы остаются на диске.") + "\n" + entry.Problem;
        }
        chooser.SelectionChanged += (_,_)=>Describe(); Describe();
        bool confirming = false;
        remove.Click += async (_,_)=>
        {
            if (confirming || chooser.SelectedItem is not ModelEntry selected) return;
            confirming = true; remove.IsEnabled = false; chooser.IsEnabled = false; string? confirmFailure = null;
            try
            {
            var answer = new Window { Title = "Подтвердите модель", Width = 500, SizeToContent = SizeToContent.Height, WindowStartupLocation = WindowStartupLocation.CenterOwner, Background = Background };
            var name = new TextBox { Watermark = "Введите точное имя модели" }; var yes = Button("Подтвердить"); yes.IsEnabled = false; var no = Button("Отмена");
            name.PropertyChanged += (_,e)=> { if(e.Property.Name=="Text") yes.IsEnabled = name.Text == selected.Name; };
            yes.Click += (_,_)=>answer.Close(true); no.Click += (_,_)=>answer.Close(false);
            answer.Content = new Border { Padding = new Thickness(14), Child = Column(Text(selected.Name,16), Text(selected.Managed ? "Переместить в локальную корзину? Если это активная модель, её тренер будет остановлен." : "Убрать из списка, не изменяя внешний файл?"), Text(selected.Path,11), name, Row(yes,no)) };
            if (await answer.ShowDialog<bool>(dialog)) dialog.Close(selected);
            }
            catch (Exception error) { confirmFailure = "Удаление не подтверждено: " + error.Message; }
            finally { confirming = false; chooser.IsEnabled = true; Describe(); if (confirmFailure is not null) detail.Text = confirmFailure; }
        };
        trash.Click += (_,_)=> { try { ModelLibrary.NoLinks(AppPaths.Trash); Directory.CreateDirectory(AppPaths.Trash); Process.Start(new ProcessStartInfo(AppPaths.Trash) { UseShellExecute = true }); } catch(Exception e) { detail.Text = e.Message; } };
        close.Click += (_,_)=>dialog.Close(null);
        dialog.Content = new Border { Padding = new Thickness(14), Child = Column(Text("Локальные модели и открытые файлы",17),chooser,detail,Row(remove,trash,close)) };
        var chosen = await dialog.ShowDialog<ModelEntry?>(this);
        if (chosen is not null) await RemoveEntry(chosen); else _status.Text = "Каталог моделей обновлён. Выбранная модель не менялась.";
    }
    private async Task SaveDiagnostics()
    {
        var model = _model ?? throw new InvalidOperationException("Нет выбранной модели.");
        var revision = _revision; var status = _workerStatus; var sampling = _sampling;
        string? active = _workspace is null || _inferenceFile is null ? null : Path.GetDirectoryName(_inferenceFile);
        WorkspaceSettings? saved = active is not null && File.Exists(Path.Combine(active,"settings.json")) ? JsonData.Read<WorkspaceSettings>(Path.Combine(active,"settings.json"),64*1024) : null;
        string file = Path.Combine(AppPaths.Root,"diagnostics","model-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N")[..6]+".json");
        var report = new { version = "17.0.0-audit17", capturedAt = DateTimeOffset.UtcNow, modelName = Path.GetFileName(CurrentModelPath),
            architecture = model.Weights.Config, parameterCount = model.Weights.Config.ParameterCount,
            published = revision is null ? null : new { revision.Revision, revision.Step, revision.TargetTokens, revision.ValidationLoss, revision.BestValidationLoss, revision.ValidationSequenceLength },
            liveStep = status?.Step, liveTrainingLoss = status?.Loss, device = status?.Device, performance = status?.Performance,
            savedSettings = saved, nextRun = new { training = SelectedTraining(), resources = SelectedResources(), material = _trainingMaterial.SelectedIndex },
            sampling, onlineEnabled = _online.IsChecked == true,
            privacy = "No conversation text, corpus contents or process logs. Model display name is included. Unpublished live steps may be newer than published weights." };
        await Task.Run(()=>JsonData.AtomicWrite(file,report,128*1024));
        _status.Text = "Диагностика сохранена: " + file + ". Тексты переписки не включены.";
        try { Process.Start(new ProcessStartInfo(Path.GetDirectoryName(file)!) { UseShellExecute = true }); } catch(Exception e) { Log("Папку диагностики не удалось открыть: " + e.Message); }
    }
}
