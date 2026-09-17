using System.Diagnostics;
using System.Text.Json;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using TritStudio.Core;
namespace TritStudio.App;

// Event-driven desktop UI; selected-model and creation state are separate. No timer rebuilds the tree; text is updated in-place.
public sealed partial class MainWindow : Window
{
    private static readonly IBrush PanelBrush = new SolidColorBrush(Color.Parse("#172033"));
    private static readonly IBrush Muted = new SolidColorBrush(Color.Parse("#ACBCD0"));
    private readonly TabControl _tabs = new();
    private readonly TextBlock _status = Text("Готово к созданию модели."), _modelLabel = Text("Модель не выбрана", 12),
        _stats = Text("Параметров: нет модели", 14), _error = Text("", 13), _pathsLabel = Text("Разговорный корпус подключается только на разговорном этапе. Базовый претрейн использует отдельный маленький набор текстов."),
        _estimate = Text(""), _workspaceLabel = Text(""), _details = Text(""), _log = Text("", 12);
    private readonly ProgressBar _progress = new() { Height = 7, Minimum = 0, Maximum = 100 };
    private readonly StackPanel _messages = new() { Spacing = 6 };
    private readonly ScrollViewer _scroll = new() { HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled };
    private readonly TextBox _input = new() { AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MinHeight = 48, MaxHeight = 140, Watermark = "Сообщение… Enter отправляет, Shift+Enter переносит строку" };
    private readonly TextBox _name = new() { Text = "Моя модель", Watermark = "Имя модели" };
    private readonly CheckBox _online = new() { Content = "Учиться на моих сообщениях", IsChecked = false };
    private readonly CheckBox _private = new() { Content = "Следующее сообщение не использовать для обучения" };
    private readonly CheckBox _refreshCorpus = new() { Content = "При дообучении добавлять новые примеры встроенного корпуса", IsChecked = true };
    private readonly CheckBox _sdpa = new() { Content = "Оптимизированное attention (SDPA)", IsChecked = true };
    private readonly CheckBox _buckets = new() { Content = "Группировать примеры по длине", IsChecked = true };
    private readonly CheckBox _targetProjection = new() { Content = "Считать выходной слой только для целевых токенов", IsChecked = true };
    private readonly CheckBox _cuda = new() { Content = "Обучать на CUDA, если проверка GPU успешна", IsChecked = true };
    private readonly Button _send = Button("Отправить"), _stopGeneration = Button("Стоп ответа"), _create = Button("Создать без обучения"), _train = Button("Запустить базовый претрейн"), _stopTraining = Button("Остановить обучение");
    private readonly ComboBox _creationMode = new() { ItemsSource = new[] { "0 · Без обучения: случайные веса", "1 · Создать и пройти базовый претрейн", "2 · Сразу обучить разговорному корпусу" }, SelectedIndex = 0, HorizontalAlignment = HorizontalAlignment.Stretch };
    private readonly ComboBox _trainingMaterial = new() { ItemsSource = new[] { "Разговорный корпус + выбранные файлы", "Базовый претрейн: только короткие тексты", "Свои файлы + ранее изученное" }, SelectedIndex = 1, HorizontalAlignment = HorizontalAlignment.Stretch };
    private readonly TextBlock _stageHint = Text("Сначала создайте нулевую модель. Ни разговорный корпус, ни сообщения чата не изменят веса без отдельного запуска обучения.", 12);
    private readonly ComboBox _preset = new() { ItemsSource = new[] { "Быстрый старт", "Побольше", "Экспериментальный" }, SelectedIndex = 0, HorizontalAlignment = HorizontalAlignment.Stretch };
    private readonly NumericUpDown _dimension = Number(64, 16, 512, 16), _hidden = Number(192, 16, 2048, 16),
        _layers = Number(2, 1, 12), _heads = Number(4, 1, 16), _kvHeads = Number(2, 1, 16),
        _context = Number(1024, 32, 2048, 32), _planes = Number(2, 1, 3), _group = Number(32, 8, 128, 8),
        _threshold = Number(0.5m, 0, 1, 0.05m, "0.00"), _threads = Number(Math.Max(1, Environment.ProcessorCount - 2), 1, Math.Max(1, Environment.ProcessorCount)),
        _memory = Number(8192, 512, 65536, 512), _batch = Number(8, 1, ResourceOptions.MaxBatchSize), _sequence = Number(512, 16, 2048, 16),
        _steps = Number(1200, 0, 100000, 20), _learningRate = Number(0.001m, 0.000001m, 0.01m, 0.0001m, "0.######"),
        _onlineRate = Number(0.0001m, 0.000001m, 0.001m, 0.00001m, "0.######"),
        _maxTokens = Number(256, 1, 1024, 16), _topK = Number(40, 1, 262);
    private readonly Slider _temperature = new() { Minimum = 0, Maximum = 2, Value = 0.7 },
        _topP = new() { Minimum = 0.1, Maximum = 1, Value = 0.95 },
        _repetition = new() { Minimum = 1, Maximum = 2, Value = 1.0 };
    private AppPreferences _preferences;
    private SamplingOptions _sampling = new();
    private string[] _datasetPaths = [];
    private readonly List<ChatTurn> _history = new();
    private readonly Queue<string> _logs = new();
    private WorkerClient? _client;
    private FileStream? _uiLease;
    private string? _workspace, _inferenceFile;
    private string _conversationId = Guid.NewGuid().ToString("N");
    private ManagedInference? _model;
    private (ManagedInference Model, RevisionInfo Info, string Path)? _pending;
    private RevisionInfo? _revision;
    private CancellationTokenSource? _generationCts;
    private Task? _generationTask;
    private bool _generating, _followTail = true, _closeAllowed, _settingControls, _workerReady;
    private long _epoch, _publication;
    private readonly SemaphoreSlim _modelLoader = new(1, 1);
    private CancellationTokenSource? _snapshotLoadCts;
    private (long Epoch, long Revision, string Hash)? _snapshotLoadSignature;
    private TaskCompletionSource<bool>? _snapshotLoadCompletion;
    private readonly DispatcherTimer _telemetry;
    private StatusEvent? _workerStatus;
    private bool _operationBusy, _openingWorkspace, _closing, _modeApplying;
    private readonly HashSet<Button> _busyButtons = new();
    private const string LearnedButtonTag = "trit-confirmed";
    private readonly Dictionary<Button, (string Label, DateTimeOffset Started)> _jobs = new();
    private readonly TextBlock _actionTitle = Text("Готово", 16), _actionDetail = Text("Выберите модель или создайте новую на второй вкладке.", 13),
        _activity = Text("Нет выполняющихся действий", 12), _dataSummary = Text("Корпус ещё не загружен", 12), _deviceNote = Text("Устройство будет проверено при открытии модели.", 12);
    private Button? _openButton, _packedButton, _exportButton, _newChatButton, _rollbackButton;
    private readonly Button _latest = Button("К последнему сообщению ↓");
    private DateTimeOffset? _generationStarted;
    private CancellationTokenSource? _modeDebounce;
    private string _lastStage = "";
    private long _actionSerial;
    private bool _navigationBusy, _promptValid = true;
    private readonly CancellationTokenSource _windowLife = new();
    private CancellationTokenSource? _preparationCts;
    private readonly TextBlock _contextBudget = Text("Контекст появится после открытия модели.", 12);
    private Button? _maintenanceButton, _discardButton, _reconnectButton, _pickDataButton, _clearDataButton;



    public MainWindow()
    {
        Title = "Trit Studio • аудит 22 • лаборатория собственных моделей";
        Width = 1280; Height = 860; MinWidth = 780; MinHeight = 540;
        Background = new SolidColorBrush(Color.Parse("#101624"));
        _preferences = AppPaths.Load(); _sampling = _preferences.Sampling ?? new();
        _online.IsChecked = _preferences.OnlineLearning;
        _temperature.Value = _sampling.Temperature; _topK.Value = _sampling.TopK; _topP.Value = _sampling.TopP;
        _repetition.Value = _sampling.RepetitionPenalty; _maxTokens.Value = _sampling.MaxNewTokens;
        _scaledShell = new LayoutTransformControl { Child = BuildShell(), LayoutTransform = new ScaleTransform(1, 1) };
        Content = _scaledShell; ApplyUiScale(_preferences.UiScale, false); Wire(); WireLibrary(); WireLaunchEditors(); UpdateEstimate(); UpdateLearningControls(); UpdateState();
        _telemetry = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _telemetry.Tick += (_, _) => { UpdateStats(); UpdateActivity(); }; _telemetry.Start();
        Opened += async (_, _) => await Safe(async () =>
        {
            await RefreshLibrary();
            if (_preferences.LastWorkspace is string path && Directory.Exists(path)) await OpenWorkspace(path);
            else { _tabs.SelectedIndex = 1; if (_newModelPane is not null) _newModelPane.IsExpanded = true; }
        });
        Closing += async (_, e) =>
        {
            if (_closeAllowed) return; e.Cancel = true; if (_closing || _confirmingUnsavedClose) return;
            try { FlushEditorDrafts(strict:true); }
            catch (Exception error)
            {
                SetError("Параметры не сохранены: " + error.Message); _confirmingUnsavedClose = true;
                bool exitWithoutSaving;
                try { exitWithoutSaving = await Confirm("Параметры не сохранены", error.Message + "\nЗакрыть без сохранения изменённых параметров? Последние корректные файлы останутся без изменений."); }
                finally { _confirmingUnsavedClose = false; }
                if (!exitWithoutSaving) return;
                _runDraftDirty = false; _creationDraftDirty = false;
            }
            _draftTimer.Stop(); _closing = true; _windowLife.Cancel(); _preparationCts?.Cancel(); IsEnabled = false; _telemetry.Stop(); CancelPendingModeEdit(); _generationCts?.Cancel();
            _actionTitle.Text = "Закрытие"; _actionDetail.Text = "Останавливаю вычисления. Последний подтверждённый снимок останется на диске.";
            if (_generationTask is not null) try { await _generationTask; } catch { }
            if (_client is not null) await _client.DisposeAsync();
            // Retain workspace ownership until a snapshot load / explicit maintenance finishes.
            await _modelLoader.WaitAsync();
            try { _uiLease?.Dispose(); } finally { _modelLoader.Release(); }
            SavePreferences(); _closeAllowed = true; Close();
        };
    }
    private static TextBlock Text(string text, double size = 14) => new() { Text = text, FontSize = size, TextWrapping = TextWrapping.Wrap };
    private static Button Button(string text) => new() { Content = text, Padding = new Thickness(10, 5), HorizontalAlignment = HorizontalAlignment.Left };
    private static NumericUpDown Number(decimal value, decimal min, decimal max, decimal step = 1, string format = "0") =>
        new ParameterNumber(format, format == "0")
        { Minimum = min, Maximum = max, Increment = step, FormatString = format, Value = value, HorizontalAlignment = HorizontalAlignment.Stretch };
    private static void SetEditorValue(NumericUpDown control, decimal value)
    {
        if (control is ParameterNumber number) number.SetNumber(value);
        else control.Value = value;
    }
    private static int LaunchInt(NumericUpDown x) => decimal.ToInt32(CheckedNumber(x));
    private static decimal CheckedNumber(NumericUpDown x) => x is ParameterNumber p ? p.ReadChecked() : x.Value ?? throw new ArgumentException("Введите число.");
    private static StackPanel Column(params Control[] items) { var s = new StackPanel { Spacing = 6 }; foreach (var c in items) s.Children.Add(c); return s; }
    private static StackPanel Row(params Control[] items) { var s = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 }; foreach (var c in items) s.Children.Add(c); return s; }
    private static Control Field(string label, Control input)
    { input.Tag = label; return Column(new TextBlock { Text = label, Foreground = Muted, FontSize = 12 }, input); }
    private static Border Card(Control c) => new() { Background = PanelBrush, CornerRadius = new CornerRadius(6), BorderBrush = new SolidColorBrush(Color.Parse("#344156")), BorderThickness = new Thickness(1), Padding = new Thickness(10), Child = c };
    private Control BuildShell()
    {
        var open = _openButton = Button("Папка…"); open.Click += async (_, _) => await RunButton(open, "Открытие модели", OpenFolder);
        var packed = _packedButton = Button("Файл…"); packed.Click += async (_, _) => await RunButton(packed, "Открытие модели для чата", OpenPacked);
        var export = _exportButton = Button("Экспорт"); export.Click += async (_, _) => await RunButton(export, "Экспорт модели", Export);
        var header = new Grid { ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto"), RowDefinitions = new RowDefinitions("Auto,Auto,Auto,Auto"), Margin = new Thickness(12, 8) };
        var brand = Text("TRIT STUDIO", 17); brand.VerticalAlignment = VerticalAlignment.Center; brand.Margin = new Thickness(0,0,16,0); header.Children.Add(brand);
        var choose = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto,Auto"), Margin = new Thickness(0,0,12,0), VerticalAlignment = VerticalAlignment.Center };
        _modelPicker.MinHeight = 30; choose.Children.Add(_modelPicker);
        Grid.SetColumn(_selectModelButton,1); choose.Children.Add(_selectModelButton);
        Grid.SetColumn(_manageModelsButton,2); choose.Children.Add(_manageModelsButton);
        Grid.SetColumn(choose,1); header.Children.Add(choose);
        _uiScale.MinHeight = 30;
        var scaleLabel = Text("Масштаб",12); scaleLabel.VerticalAlignment = VerticalAlignment.Center;
        var right = Row(open, packed, export, scaleLabel, _uiScale); right.VerticalAlignment = VerticalAlignment.Center;
        right.HorizontalAlignment = HorizontalAlignment.Right; Grid.SetColumn(right,2); header.Children.Add(right);
        foreach (var control in new Control[] { open, packed, export, _selectModelButton, _manageModelsButton, _modelPicker, _uiScale })
        { control.VerticalAlignment = VerticalAlignment.Center; control.MinHeight = 30; }
        _modelLabel.TextWrapping = TextWrapping.NoWrap; _modelLabel.TextTrimming = TextTrimming.CharacterEllipsis;
        _modelLabel.Margin = new Thickness(0,5,0,0); Grid.SetRow(_modelLabel,1); Grid.SetColumnSpan(_modelLabel,3); header.Children.Add(_modelLabel);
        var accuracyLine = BuildAccuracyLine(); Grid.SetRow(accuracyLine,2); Grid.SetColumnSpan(accuracyLine,3); header.Children.Add(accuracyLine);
        bool narrowHeader = false;
        header.PropertyChanged += (_,e) =>
        {
            if (e.Property.Name != "Bounds" || header.Bounds.Width <= 0) return;
            bool narrow = header.Bounds.Width < 1050; if (narrow == narrowHeader) return; narrowHeader = narrow;
            header.ColumnDefinitions = new ColumnDefinitions(narrow ? "Auto,*" : "Auto,*,Auto");
            Grid.SetColumn(right,narrow ? 0 : 2); Grid.SetRow(right,narrow ? 1 : 0); Grid.SetColumnSpan(right,narrow ? 2 : 1);
            Grid.SetRow(_modelLabel,narrow ? 2 : 1); Grid.SetColumnSpan(_modelLabel,narrow ? 2 : 3);
            Grid.SetRow(accuracyLine,narrow ? 3 : 2); Grid.SetColumnSpan(accuracyLine,narrow ? 2 : 3);
        };
        _tabs.ItemsSource = new[] { new TabItem { Header = "Чат", Content = BuildChat() }, new TabItem { Header = "Модель и обучение", Content = BuildTraining() } };
        _tabs.Margin = new Thickness(12,0,12,6);
        _error.Foreground = new SolidColorBrush(Color.Parse("#FFBCAD")); _error.IsVisible = false;
        _actionTitle.FontSize = 12; _actionDetail.FontSize = 11; _activity.FontSize = 11;
        _actionDetail.MaxHeight = 30; _actionDetail.ClipToBounds = true; _activity.MaxHeight = 26; _activity.ClipToBounds = true;
        _status.IsVisible = false; // Same information stays in the compact terminal receipt and the log.
        foreach (var feedback in new[] { _actionDetail, _status })
            feedback.PropertyChanged += (_, e) => { if (e.Property.Name == "Text") ToolTip.SetTip(feedback, feedback.Text); };
        _status.PropertyChanged += (_,e) => { if(e.Property.Name == "Text" && (_jobs.Count > 0 || _workerStatus?.Busy == true)) _actionDetail.Text = _status.Text; };
        _progress.Height = 3; _progress.IsVisible = false;
        var receiptLine = new Grid { ColumnDefinitions = new ColumnDefinitions("Auto,*") };
        _actionTitle.Margin = new Thickness(0,0,12,0); receiptLine.Children.Add(_actionTitle); Grid.SetColumn(_actionDetail,1); receiptLine.Children.Add(_actionDetail);
        var bottom = new Border { BorderBrush = new SolidColorBrush(Color.Parse("#344156")), BorderThickness = new Thickness(0,1,0,0), Padding = new Thickness(12,5),
            Child = Column(receiptLine, _activity, _progress,
                new ScrollViewer { Content = _error, MaxHeight = 64, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled }) };
        var grid = new Grid { RowDefinitions = new RowDefinitions("Auto,*,Auto") };
        grid.Children.Add(header); Grid.SetRow(_tabs,1); grid.Children.Add(_tabs); Grid.SetRow(bottom,2); grid.Children.Add(bottom); return grid;
    }
    private Control SliderField(string label, Slider slider)
    {
        var number = Text($"{slider.Value:F2}");
        slider.PropertyChanged += (_, e) => { if (e.Property.Name == "Value") number.Text = $"{slider.Value:F2}"; };
        return Column(Row(Text(label), number), slider);
    }
    private Control BuildChat()
    {
        var newChat = _newChatButton = Button("Очистить чат");
        ToolTip.SetTip(newChat, "Начать разговор без прежнего контекста. Сохранённый журнал, веса и очередь обучения останутся.");
        newChat.Click += async (_, _) => await RunButton(newChat, "Очистка чата", ClearChat);
        var rollback = _rollbackButton = Button("Откатить веса");
        rollback.Click += async (_, _) => await RunButton(rollback, "Откат весов", async () =>
        {
            NeedWorker(); _operationBusy = true; UpdateState();
            try { await _client!.Request("rollback", new { }); SetOnlineCheck(false); _status.Text = "Предыдущий снимок восстановлен. Автообучение выключено; включите его явно для новых изменений."; }
            finally { _operationBusy = false; }
        });
        var help = Text("Число параметров не растёт при обучении: меняются их значения. Архитектура задаётся на второй вкладке.", 12); help.Foreground = Muted;
        var privacy = Text("Обучение меняет веса и сохраняет текст локально. Ошибочные ответы модели не считаются эталоном. Исключение сообщения не удаляет его из истории.", 12); privacy.Foreground = Muted;
        var sidebar = Column(Text("Во время разговора", 18), SliderField("Температура", _temperature),
            Field("Top-k", _topK), SliderField("Top-p", _topP), SliderField("Штраф за повторы", _repetition), Field("Максимум новых байт-токенов", _maxTokens),
            _online, Field("Скорость онлайн-обучения", _onlineRate), privacy, _stats, help, rollback, _deviceNote);
        _scroll.Content = _messages; _scroll.Margin = new Thickness(0, 0, 0, 6);
        _scroll.ScrollChanged += (_, e) =>
        {
            if (e.OffsetDelta.Y < 0) _followTail = false;
            if (Math.Abs(e.OffsetDelta.Y) > 0 && _scroll.Extent.Height - _scroll.Viewport.Height - _scroll.Offset.Y < 25) _followTail = true;
        };
        _latest.IsVisible = false;
        _latest.Click += (_, _) => { _followTail = true; _scroll.ScrollToEnd(); _latest.IsVisible = false; Notice("Показаны новые сообщения", "Автопрокрутка снова включена."); };
        var composer = Column(_latest, _private, _input, _contextBudget, Row(_send, _stopGeneration, newChat));
        var chatTools = Row(_chatSettingsButton);
        var conversation = new Grid { RowDefinitions = new RowDefinitions("Auto,*,Auto") };
        conversation.Children.Add(chatTools); Grid.SetRow(_scroll,1); conversation.Children.Add(_scroll); Grid.SetRow(composer,2); conversation.Children.Add(composer);
        _chatGrid = new Grid { ColumnDefinitions = new ColumnDefinitions("*,0"), Margin = new Thickness(0,6,0,0) };
        _chatGrid.Children.Add(conversation);
        _chatSidebar = new ScrollViewer { Content = Card(sidebar), Margin = new Thickness(10,0,0,0), HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, IsVisible = false };
        Grid.SetColumn(_chatSidebar,1); _chatGrid.Children.Add(_chatSidebar);
        SetSidebar(_preferences.ChatSettingsVisible, false); return _chatGrid;
    }

    private Control BuildTraining()
    {
        var pick = _pickDataButton = Button("Добавить датасеты…");
        pick.Click += async (_, _) => await RunButton(pick, "Выбор датасетов", PickDatasets);
        var clear = _clearDataButton = Button("Убрать выбранные");
        clear.Click += (_, _) => { _datasetPaths = []; _pathsLabel.Text = "Выбор файлов очищен. Обученные веса не изменены."; Notice("Выбор очищен", _pathsLabel.Text); };
        var maintain = _maintenanceButton = Button("Очистить старые снимки");
        maintain.Click += async (_, _) => await RunButton(maintain, "Очистка истории снимков", MaintainSnapshots);
        var discard = _discardButton = Button("Очистить очередь обучения");
        discard.Click += async (_, _) => await RunButton(discard, "Очистка очереди", async () =>
        {
            NeedWorker();
            if (!await Confirm("Очистить очередь?", "Ожидающие примеры будут исключены из обучения. Уже изменённые веса и переписка сохранятся. Автообучение выключится.")) throw new OperationCanceledException();
            var receipt = await _client!.Request("discard-pending", new { });
            int count = receipt.Data.GetProperty("result").GetProperty("discarded").GetInt32();
            SetOnlineCheck(false); _status.Text = $"Из очереди исключено: {count}. Веса не отменены, автообучение выключено.";
        });
        var reconnect = _reconnectButton = Button("Переподключить тренер");
        reconnect.Click += async (_, _) => await RunButton(reconnect, "Переподключение тренера", async () =>
        {
            string path = _workspace ?? throw new InvalidOperationException("Откройте рабочую папку модели.");
            await OpenWorkspace(path); _status.Text = _workerReady ? "Тренер снова подключён. Настройки восстановлены." : "Чат доступен, тренер не подключён.";
        });
        var service = new Expander { Header = "Очередь, снимки и восстановление", Content = Column(discard, maintain, reconnect,
            Text("Очистка сохраняет последние 8 снимков цепочки отката. Она останавливает обучение, но не удаляет переписку или текущую модель.", 12)) };
        var references = Button("Открыть папку эталонов этапов");
        references.Click += async (_, _) => await RunButton(references, "Эталоны эволюции", () =>
        {
            string path = StageArchive.Root(_workspace ?? throw new InvalidOperationException("Откройте рабочую папку модели."));
            if (!Directory.Exists(path)) throw new InvalidOperationException("Эталоны ещё не созданы. Они сохраняются после создания и завершения первого запуска каждого этапа.");
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
            _status.Text = "Эталоны для CPU-чата: " + path + ". Открывайте model.tritmodel через «Файл…» в шапке.";
            return Task.CompletedTask;
        });

        _conversationProbeButton.Click += async (_,_) => await RunButton(_conversationProbeButton, "Разговорная проверка", CheckConversation);
        _conversationPresetButton.Click += async (_,_) => await Safe(() => { ApplyConversationPreset(); return Task.CompletedTask; });
        _qualityButton.Click += async (_,_) => await RunButton(_qualityButton, "Проверка обучения", CheckLearning);
        _trainingPresetButton.Click += async (_,_) => await Safe(() => { ApplyLearningPreset(); return Task.CompletedTask; });
        var active = Card(Column(Text("Текущая модель",18), _selectedModelInfo, _workspaceLabel,
            Flow(_diagnosticsButton, _qualityButton, _conversationProbeButton, references), _qualitySummary, _conversationSummary,
            new Expander { Header = "Обслуживание модели и журнал", Content = Column(service,
                new ScrollViewer { Content = _log, MaxHeight = 200 }) }));
        active.Name = "ActiveModelPanel";
        var architecture = FormFields(
            HelpField("Ширина",_dimension,WidthHelp), HelpField("FFN",_hidden,FfnHelp), HelpField("Слои",_layers,LayersHelp), HelpField("Головы",_heads,HeadsHelp),
            HelpField("KV-головы",_kvHeads,KvHelp), HelpField("Контекст",_context,ContextHelp), HelpField("Троичные плоскости",_planes,PlanesHelp),
            HelpField("Группа квантования",_group,GroupHelp), HelpField("Порог",_threshold,ThresholdHelp));
        _newModelPane = new Expander { Header = "Архитектура новой сети", Content = architecture };
        _createTrainFields = Column(Text("Обучение сразу после создания",14),
            FormFields(HelpField("Шаги",_newSteps,StepsHelp),HelpField("Learning rate",_newRate,RateHelp),HelpField("Пакет",_newBatch,BatchHelp),
                HelpField("Длина обучения",_newSequence,LengthHelp),HelpField("Сохранять через N шагов",_newPublish,PublishHelp)),
            _newScheduledRate, _newAutoSnapshots, _newConversationCourse, _newContextPractice, _newTransferPractice, _newEqualExamples, Text("Только встроенный материал выбранного этапа. Свои файлы подключаются после создания в соседнем блоке.",12));
        var create = Card(Column(Text("Создать новую модель",18), Field("Имя",_name),Field("Размер",_preset),
            _newModelPane, _estimate, Field("После создания",_creationMode),_stageHint,
            Flow(_newCuda,Field("Бюджет RAM создания, МиБ",_newMemory)),_createTrainFields,_create));
        create.Name = "CreateModelPanel";
        var training = Card(Column(Text("Дообучить выбранную модель",18),
            Text("Меняются веса модели из шапки. Поля блока «Создать новую модель» к этому запуску не относятся.",12),
            SectionTitle("Материал"),Field("Этап",_trainingMaterial),Flow(pick,clear),_pathsLabel,_dataSummary,_refreshCorpus,
            SectionTitle("Параметры запуска"), Flow(_conversationPresetButton, _trainingPresetButton, _saveRunDraft),
            FormFields(HelpField("Дополнительные шаги",_steps,StepsHelp),HelpField("Learning rate",_learningRate,RateHelp),
                HelpField("Пакет",_batch,BatchHelp),HelpField("Максимальная длина",_sequence,LengthHelp),HelpField("Сохранять через N шагов",_publishEvery,PublishHelp)),
            _scheduledRate, _autoSnapshots, _conversationCourse, _contextPractice, _transferPractice, _equalExamples, Text("Курс начинается заново при каждом ручном запуске. Для обычного продолжения готовой модели снимите «Курс», оставив равный вес примеров.",12), _learningHint,_runDraftStatus,SectionTitle("Устройство и память"),_cuda,
            FormFields(Field("Потоки CPU",_threads),Field("Бюджет RAM тренера, МиБ",_memory)),
            Text("Бюджет RAM задаёт лимит процесса, а не измеряет свободную память. VRAM не равна RAM. Нулевое создание не выделяет активации учебного пакета.",12),
            new Expander { Header = "Оптимизации", Content = Column(_sdpa,_buckets,_targetProjection) },
            Flow(_train,_stopTraining),SectionTitle("Последний запуск"),_details));
        training.Name = "TrainModelPanel";
        var grid = new Grid { Name = "ModelPanels", ColumnDefinitions = new ColumnDefinitions("2*,3*"),
            RowDefinitions = new RowDefinitions("Auto,Auto,Auto"), Margin = new Thickness(0,6,0,0) };
        Grid.SetColumnSpan(active,2);active.Margin = new Thickness(0,0,0,10);grid.Children.Add(active);
        Grid.SetRow(create,1);create.Margin = new Thickness(0,0,10,0);grid.Children.Add(create);
        Grid.SetRow(training,1);Grid.SetColumn(training,1);grid.Children.Add(training);
        bool stacked = false;
        grid.PropertyChanged += (_,e) =>
        {
            if (e.Property.Name != "Bounds" || grid.Bounds.Width <= 0) return;
            bool narrow = grid.Bounds.Width < 960;if(narrow==stacked)return;stacked=narrow;
            grid.ColumnDefinitions = new ColumnDefinitions(narrow ? "*" : "2*,3*");
            Grid.SetColumnSpan(active,narrow?1:2);Grid.SetRow(training,narrow?2:1);Grid.SetColumn(training,narrow?0:1);
            create.Margin = narrow ? new Thickness(0,0,0,10) : new Thickness(0,0,10,0);
        };
        return new ScrollViewer { Content = grid, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled };
    }

    private void Wire()
    {
        _creationMode.SelectionChanged += (_, _) => { UpdateLearningControls(); UpdateEstimate(); };
        _trainingMaterial.SelectionChanged += (_, _) => UpdateLearningControls();
        foreach (var c in new AvaloniaObject[] { _temperature, _topP, _repetition, _maxTokens, _topK })
            c.PropertyChanged += (_, e) => { if (e.Property.Name == "Value") UpdateSampling(); };
        foreach (var c in new[] { _dimension, _hidden, _layers, _heads, _kvHeads, _context, _planes, _group, _threshold })
            c.PropertyChanged += (_, e) => { if (e.Property.Name == "Value") UpdateEstimate(); };
        _learningRate.PropertyChanged += (_,e) => { if(e.Property.Name == "Value") UpdateLearningHint(); };
        _newMemory.PropertyChanged += (_,e) => { if(e.Property.Name == "Value") UpdateEstimate(); };
        _preset.SelectionChanged += (_, _) => { if (!_restoringLaunch && _preset.SelectedIndex >= 0) ApplyPreset(_preset.SelectedIndex switch { 1 => ModelConfig.Medium, 2 => ModelConfig.Large, _ => ModelConfig.Small }); };
        _online.PropertyChanged += async (_, e) =>
        {
            if (e.Property.Name == "IsChecked" && !_settingControls) await Safe(async () =>
            {
                CancelPendingModeEdit();
                if (await ApplyOnline())
                    Notice("Режим обучения подтверждён", _online.IsChecked == true ? "Автообучение включено. Веса обновятся после обработки очереди." : "Автообучение выключено. Очередь сохранена.");
                else if (!_workerReady && _model is null)
                    Notice("Выбран режим обучения", "Тренер ещё не подключён. Нулевой и базовый этапы не включают автообучение автоматически.");
            });
        };
        _onlineRate.PropertyChanged += async (_, e) =>
        {
            if (e.Property.Name != "Value" || _settingControls || !_workerReady) return;
            await ApplyOnlineRateAfterDelay();
        };
        _input.PropertyChanged += (_, e) => { if (e.Property.Name == "Text") UpdateContextBudget(); };
        _send.Click += async (_, _) => await RunButton(_send, "Ответ модели", StartSend);
        _stopGeneration.Click += (_, _) => { _generationCts?.Cancel(); Notice("Остановка ответа запрошена", "Текущая генерация завершится; неполный ответ не станет обучающим примером."); };
        // TextBox consumes Return before a bubbling handler. Handle plain Enter on the tunnel.
        // Shift+Enter belongs to the TextBox; Ctrl/Alt/Meta combinations are not implicit sends.
        _input.AddHandler(InputElement.KeyDownEvent, ComposerKeyDown, RoutingStrategies.Tunnel);
        _input.AddHandler(InputElement.KeyUpEvent, (_, e) => { if (e.Key == Key.Enter) _composerEnterDown = false; }, RoutingStrategies.Tunnel, handledEventsToo: true);
        _input.LostFocus += (_, _) => _composerEnterDown = false;
        _create.Click += async (_, _) => await RunButton(_create, "Создание и обучение", Create);
        _train.Click += async (_, _) => await RunButton(_train, "Дообучение модели", Train);
        _stopTraining.Click += async (_, _) => await RunButton(_stopTraining, "Остановка обучения", async () =>
        {
            if (_preparationCts is not null)
            {
                _preparationCts.Cancel(); _status.Text = "Отмена подготовки принята. Новая модель не будет создана."; return;
            }
            NeedWorker(); SetOnlineCheck(false); CancelPendingModeEdit();
            _status.Text = "Отмена отправлена. Текущий вычислительный шаг завершается; новые шаги не запускаются.";
            await _client!.Request("stop", new { }); _status.Text = "Обучение остановлено. Чат продолжает работать на сохранённых весах.";
        });
    }
    private void UpdateLearningControls()
    {
        var mode = (CreationMode)Math.Clamp(_creationMode.SelectedIndex, 0, 2);
        if (!_busyButtons.Contains(_create)) _create.Content = mode switch
        { CreationMode.Untrained => "Создать без обучения", CreationMode.BasicPretrain => "Создать и пройти базовый претрейн", _ => "Создать и обучить диалогам" };
        if (!_busyButtons.Contains(_train)) _train.Content = _trainingMaterial.SelectedIndex == 1 ? "Запустить базовый претрейн" : "Дообучить текущую модель";
        if (_createTrainFields is not null) _createTrainFields.IsVisible = mode != CreationMode.Untrained;
        UpdateLearningHint();
        _stageHint.Text = mode switch
        {
            CreationMode.Untrained => "Случайные веса, строго 0 шагов. Настройки дообучения соседнего блока не используются.",
            CreationMode.BasicPretrain => "Только короткие базовые тексты, без разговорного датасета и своих файлов. После завершения автообучение останется выключенным.",
            _ => "Создание и обучение на встроенном разговорном корпусе. Нулевой эталон сохранится отдельно; свои файлы добавляйте при дообучении."
        };
    }
    private static int Int(NumericUpDown x) => (int)(x.Value ?? 0);
    private ModelConfig SelectedConfig() => _creationConfigBase with { Dimension = LaunchInt(_dimension), HiddenDimension = LaunchInt(_hidden), Layers = LaunchInt(_layers), Heads = LaunchInt(_heads),
        KvHeads = LaunchInt(_kvHeads), Context = LaunchInt(_context), Planes = LaunchInt(_planes), GroupSize = LaunchInt(_group), Threshold = (float)CheckedNumber(_threshold) };
    private ResourceOptions SelectedResources() => new() { Threads = LaunchInt(_threads), MemoryMiB = LaunchInt(_memory), BatchSize = LaunchInt(_batch), SequenceLength = LaunchInt(_sequence), PreferCuda = _cuda.IsChecked == true, UseSdpa = _sdpa.IsChecked == true, BucketByLength = _buckets.IsChecked == true, ProjectOnlyTargets = _targetProjection.IsChecked == true };
    private TrainingOptions SelectedTraining() => _runTrainingBase with { Steps = LaunchInt(_steps), LearningRate = (double)CheckedNumber(_learningRate), OnlineLearningRate = (double)CheckedNumber(_onlineRate), PublishEvery = LaunchInt(_publishEvery), WarmupCosine = _scheduledRate.IsChecked == true, AutoSnapshotInterval = _autoSnapshots.IsChecked == true, ConversationCourse = _conversationCourse.IsChecked == true, ContextPractice = _conversationCourse.IsChecked == true && _contextPractice.IsChecked == true, TransferPractice = _conversationCourse.IsChecked == true && _contextPractice.IsChecked == true && _transferPractice.IsChecked == true, EqualExampleWeight = _equalExamples.IsChecked == true };
    private void ApplyPreset(ModelConfig c)
    {
        _creationConfigBase = c;
        SetEditorValue(_dimension, c.Dimension); SetEditorValue(_hidden, c.HiddenDimension); SetEditorValue(_layers, c.Layers); SetEditorValue(_heads, c.Heads); SetEditorValue(_kvHeads, c.KvHeads);
        SetEditorValue(_context, c.Context); SetEditorValue(_planes, c.Planes); SetEditorValue(_group, c.GroupSize); SetEditorValue(_threshold, (decimal)c.Threshold); UpdateEstimate();
    }
    private void UpdateEstimate()
    {
        try { var c = SelectedConfig(); c.Validate(); _estimate.Text = $"{c.ParameterCount:N0} параметров · мастер-веса ≈ {c.ParameterCount * 4 / 1048576.0:F2} МиБ.\nСоздание/загрузка: оценка ≈ {TrainingMemoryEstimate.For(c, ZeroCreationResources(c), 0).PersistentBytes / 1048576.0:F0} МиБ, бюджет {Int(_newMemory):N0} МиБ.\nАктивации обучения в нулевое создание не входят."; }
        catch (Exception e) { _estimate.Text = "Проверьте архитектуру: " + e.Message; }
    }
    private void UpdateSampling()
    {
        Volatile.Write(ref _sampling, new SamplingOptions(_temperature.Value, Int(_topK), _topP.Value, _repetition.Value, Int(_maxTokens)).Clamp());
        UpdateContextBudget();
    }
    private void UpdateContextBudget()
    {
        _promptValid = true;
        if (_model is null) { _contextBudget.Text = "Контекст появится после открытия модели."; UpdateState(); return; }
        try
        {
            var c = _model.Weights.Config; var options = _sampling;
            var plan = ByteTokenizer.Measure(_history, _input.Text ?? "", c.Context, Math.Min(options.MaxNewTokens, c.Context - 4));
            _contextBudget.Foreground = Muted;
            _contextBudget.Text = $"Вход: {plan.InputTokens}/{plan.Context} байт-токенов · ответ: до {plan.ReservedOutputTokens} · история: {plan.RetainedTurns}/{_history.Count} обменов." +
                (plan.DroppedTurns > 0 ? " Старые обмены не поместились и не войдут в следующий запрос." : "");
        }
        catch (ArgumentException error)
        {
            _promptValid = false; _contextBudget.Foreground = new SolidColorBrush(Color.Parse("#FFBCAD"));
            _contextBudget.Text = "Сообщение не помещается. Сократите текст или лимит ответа. " + error.Message;
        }
        UpdateState();
    }
    private void UpdateState()
    {
        _send.IsEnabled = _model is not null && !_generating && !_openingWorkspace && !_navigationBusy && _preparationCts is null && _promptValid;
        _stopGeneration.IsEnabled = _generating;
        _create.IsEnabled = !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _creationMode.IsEnabled = _create.IsEnabled; _trainingMaterial.IsEnabled = !_operationBusy && !_openingWorkspace && !_navigationBusy;
        _train.IsEnabled = _workerReady && _model is not null && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _stopTraining.IsEnabled = _preparationCts is not null || (_workerReady && (_operationBusy || _workerStatus?.Busy == true || _online.IsChecked == true));
        if (_openButton is not null) _openButton.IsEnabled = !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        if (_packedButton is not null) _packedButton.IsEnabled = !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        if (_exportButton is not null) _exportButton.IsEnabled = _model is not null && !_openingWorkspace && !_navigationBusy;
        if (_newChatButton is not null) _newChatButton.IsEnabled = _model is not null && _messages.Children.Count > 0 &&
            !_generating && !_openingWorkspace && !_navigationBusy && !_modeApplying && !_operationBusy && !_closing;
        bool parentAvailable = _workspace is not null && _revision?.Parent is long parent &&
            Directory.Exists(ModelFiles.GetRevisionPath(_workspace, $"r{parent:D16}"));
        if (_rollbackButton is not null) _rollbackButton.IsEnabled = _workerReady && !_operationBusy && !_openingWorkspace && parentAvailable && !_navigationBusy && !_modeApplying;
        if (_maintenanceButton is not null) _maintenanceButton.IsEnabled = _workspace is not null && _model is not null && !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        if (_discardButton is not null) _discardButton.IsEnabled = _workerReady && (_workerStatus?.Queue ?? 0) > 0 && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        if (_reconnectButton is not null) _reconnectButton.IsEnabled = _workspace is not null && !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _online.IsEnabled = (_workerReady || _model is null) && !_modeApplying && !_openingWorkspace && !_operationBusy && !_navigationBusy;
        _onlineRate.IsEnabled = (_workerReady || _model is null) && !_modeApplying && !_openingWorkspace && !_operationBusy && !_navigationBusy;
        if (_pickDataButton is not null) _pickDataButton.IsEnabled = !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        if (_clearDataButton is not null) _clearDataButton.IsEnabled = !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _qualityButton.IsEnabled = _workerReady && _model is not null && !_generating && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _conversationProbeButton.IsEnabled = _qualityButton.IsEnabled;
        _conversationPresetButton.IsEnabled = _model is not null && !_operationBusy && !_openingWorkspace && !_navigationBusy && !_modeApplying;
        _trainingPresetButton.IsEnabled = !_operationBusy && !_openingWorkspace && !_navigationBusy;
        UpdateLibraryState(); UpdateLaunchEditorAvailability();
        foreach (var b in _busyButtons) b.IsEnabled = false;
    }
    private void Log(string text)
    {
        _logs.Enqueue(DateTime.Now.ToString("HH:mm:ss") + " " + text);
        while (_logs.Count > 100) _logs.Dequeue(); _log.Text = string.Join("\n", _logs);
    }
    private void Notice(string title, string detail)
    {
        _actionTitle.Text = title; _actionDetail.Text = detail; Log(title + ": " + detail);
    }
    private async Task RunButton(Button button, string title, Func<Task> work)
    {
        if (_closing || _busyButtons.Contains(button) || !button.IsEnabled) return;
        bool navigation = button == _openButton || button == _packedButton || button == _exportButton ||
            button == _maintenanceButton || button == _discardButton || button == _reconnectButton || button == _pickDataButton ||
            button == _selectModelButton || button == _manageModelsButton || button == _diagnosticsButton || button == _newChatButton || button == _qualityButton || button == _conversationProbeButton;
        if (navigation) { if (_navigationBusy || _modeApplying) return; CancelPendingModeEdit(); _navigationBusy = true; }
        if (button == _create || button == _train || button == _rollbackButton) CancelPendingModeEdit();
        long serial = ++_actionSerial; object? label = button.Content;
        _busyButtons.Add(button); _jobs[button] = (title, DateTimeOffset.Now); button.Content = title + "…";
        _error.IsVisible = false; Notice("Принято · " + title, "Действие выполняется. Повторное нажатие не требуется."); UpdateState(); UpdateActivity();
        try
        {
            await work();
            Log("Завершено · " + title + ": " + _status.Text);
            if (serial == _actionSerial) { _actionTitle.Text = (_error.IsVisible ? "Есть ограничение · " : "Завершено · ") + title; _actionDetail.Text = _error.IsVisible ? _error.Text : _status.Text; }
        }
        catch (OperationCanceledException)
        {
            _status.Text = "Действие отменено; незавершённый результат не опубликован.";
            Notice("Отменено · " + title, _status.Text);
        }
        catch (Exception error) { SetError(title + ": " + error.Message); }
        finally { if (navigation) _navigationBusy = false; _busyButtons.Remove(button); _jobs.Remove(button); button.Content = label; button.IsEnabled = !Equals(button.Tag, LearnedButtonTag); UpdateLearningControls(); UpdateState(); UpdateActivity(); }
    }
    private void UpdateActivity()
    {
        var tasks = _jobs.Values.Select(x => $"{x.Label} · {(DateTimeOffset.Now - x.Started).TotalSeconds:F0} с").ToList();
        if (_generating && _generationStarted is { } began) tasks.Add($"Генерация r{_model?.Revision} · {(DateTimeOffset.Now - began).TotalSeconds:F0} с");
        if (_workerStatus?.Busy == true)
        {
            string stage = _workerStatus.Stage switch { "dataset" => "подготовка корпуса", "initialize" => "создание весов", "device" => "подключение устройства", "train" => "обучение", "online" => "обучение на разговоре", "validate" => "контрольная проверка", "checkpoint" => "сохранение снимка", _ => _workerStatus.Stage };
            string steps = _workerStatus.TotalSteps > 0 ? $" · {_workerStatus.CompletedSteps}/{_workerStatus.TotalSteps}" : "";
            tasks.Add("Тренер: " + stage + steps);
        }
        _activity.Text = tasks.Count == 0 ? "Нет выполняющихся действий" : string.Join("  |  ", tasks);
        bool busy = _openingWorkspace || _operationBusy || _navigationBusy || _generating || _jobs.Count > 0 || _workerStatus?.Busy == true;
        bool measured = busy && _workerStatus is { TotalSteps: > 0, Stage: "train" or "online" };
        _progress.IsIndeterminate = busy && !measured;
        _progress.Value = measured ? 100.0 * _workerStatus!.CompletedSteps / _workerStatus.TotalSteps : 0;
        _progress.IsVisible = busy || _generating || _jobs.Count > 0;
        _activity.IsVisible = tasks.Count > 0;
        _latest.IsVisible = !_followTail && _messages.Children.Count > 0;
    }
    private void SetOnlineCheck(bool value)
    {
        bool previous = _settingControls; _settingControls = true;
        try { _online.IsChecked = value; } finally { _settingControls = previous; }
        SavePreferences();
    }
    private static void AddGenerationWarning(StackPanel body, string answer)
    {
        if (GenerationHealth.Inspect(answer).Repetitive)
            body.Children.Add(Text("Сильные повторы: это не признак успешного диалога. Ответ оставлен без изменений. Проверьте обучение на пустом контексте.",12));
    }
    private void UpdateStats()
    {
        var c = _model?.Weights.Config; var r = _revision; using var process = Process.GetCurrentProcess(); long rss = process.WorkingSet64 / 1048576;
        _stats.Text = c is null ? "Параметров: нет модели" : $"Параметров: {c.ParameterCount:N0}\nРевизия: {_model!.Revision}\nИзменено в ревизии: {r?.ChangedWeights ?? 0:N0}\nШаг обучения: {_workerStatus?.Step ?? r?.Step ?? 0:N0}\nОбучающих токенов: {r?.TargetTokens ?? 0:N0}\nОчередь: {_workerStatus?.Queue ?? 0}\nПринято / отклонено: {_workerStatus?.Replay?.Learned ?? 0} / {_workerStatus?.Replay?.Rejected ?? 0}\nСнимков: {_workerStatus?.Snapshots ?? 0}\nRAM приложения: {rss:N0} МиБ\nRAM тренера: {_workerStatus?.MemoryMiB ?? 0:N0} МиБ\nУстройство: {_workerStatus?.Device ?? "CPU, только чат"}\nКонтрольная ошибка: {(r?.ValidationLoss is double loss ? loss.ToString("F4") : "ещё не измерена")}\n{(_pending is null ? "" : "Новый снимок ожидает конца ответа.")}";
        UpdateSelectedInfo(); UpdateAccuracyLine();
        var performance = _workerStatus?.Performance;
        _details.Text = (_workerStatus?.EffectiveLearningRate is double actualRate ?
            $"LR последнего шага: {actualRate:G6}; сохранение через {_workerStatus.EffectivePublishEvery}; " +
            (_workerStatus.WarmupCosine ? "разогрев + снижение" : "постоянный LR") + ".\n" : "") +
            (_workerStatus?.TokensPerSecond is double speed ? $"Обучение: {speed:F0} целевых токенов/с. " : "") +
            (performance is null ? "" : $"Пакет {performance.BatchSize} × {performance.SequenceLength}; пустые позиции {performance.PaddingFraction:P0}.\n{(performance.DialogueMix is null ? "" : performance.DialogueMix.Summary + "\n")}{performance.AttentionBackend}. Выходных позиций: {performance.OutputPositions}/{performance.PaddedPositions}. " +
                $"Контроль: {_workerStatus!.ValidationMilliseconds ?? 0:F0} мс; повторных проверок пропущено: {_workerStatus.ValidationCacheHits}.\n" +
                $"Кеш подготовки контроля: {_workerStatus.ValidationPreparedBytes / 1048576.0:F2} МиБ CPU; повторно использовано пакетов: {_workerStatus.ValidationBatchCacheHits}.\n" +
                $"Кеш данных снимка: {_workerStatus.SnapshotCorpusCacheBytes / 1048576.0:F2} МиБ CPU; повторных записей без сериализации: {_workerStatus.SnapshotCorpusCacheHits}.\n") +
            "Скорость и ошибка не являются оценкой разговорного качества.";
    }
    private void CancelPendingModeEdit()
    {
        // Only the owning async handler disposes its CTS, after its delay/request has finished.
        var pending = _modeDebounce; _modeDebounce = null; pending?.Cancel();
    }
    private async Task ApplyOnlineRateAfterDelay()
    {
        CancelPendingModeEdit();
        long epoch = _epoch; var client = _client;
        using var pending = CancellationTokenSource.CreateLinkedTokenSource(_windowLife.Token);
        _modeDebounce = pending;
        try
        {
            await Task.Delay(400, pending.Token);
            // An edit belongs to the model/connection in which it was made, not the next opened model.
            if (!ReferenceEquals(_modeDebounce, pending) || epoch != _epoch || !ReferenceEquals(client, _client) ||
                _closing || _openingWorkspace || _navigationBusy || _operationBusy || !_workerReady) return;
            double rate = (double)CheckedNumber(_onlineRate);
            if (await ApplyOnline() && !pending.IsCancellationRequested && ReferenceEquals(_modeDebounce, pending))
                Notice("Скорость обучения подтверждена", $"Learning rate: {rate}");
        }
        catch (OperationCanceledException) { }
        catch (Exception error) { if (!_closing && epoch == _epoch) SetError(error.Message); }
        finally { if (ReferenceEquals(_modeDebounce, pending)) _modeDebounce = null; }
    }
    private async Task<bool> ApplyOnline(bool allowOwnedOperation = false)
    {
        SavePreferences();
        if (_modeApplying || _closing || _openingWorkspace || _navigationBusy || (_operationBusy && !allowOwnedOperation)) return false;
        var client = _client; long epoch = _epoch;
        if (!_workerReady || client is null) return false;
        _modeApplying = true; UpdateState();
        Notice("Принято · настройка обучения", "Ожидаю подтверждение тренера. Настройка не считается применённой до ответа.");
        try
        {
            await client.Request("mode", new OnlineMode(_online.IsChecked == true, (double)CheckedNumber(_onlineRate)));
            return !_closing && epoch == _epoch && ReferenceEquals(client, _client);
        }
        finally { _modeApplying = false; UpdateState(); }
    }
    private void SavePreferences()
    {
        try { _preferences = _preferences with { LastWorkspace = _workspace, OnlineLearning = _online.IsChecked == true, Sampling = _sampling }; JsonData.AtomicWrite(AppPaths.Preferences, _preferences); }
        catch (Exception e) { SetError("Не удалось сохранить настройки: " + e.Message); }
    }
    private void NeedWorker() { if (_client is null || !_workerReady) throw new InvalidOperationException("Тренер не подключён. Откройте рабочую папку модели или создайте новую."); }
    private async Task Safe(Func<Task> action, bool clearError = true)
    {
        try { if (clearError) _error.IsVisible = false; await action(); }
        catch (OperationCanceledException) { _status.Text = "Операция отменена."; }
        catch (Exception e) { SetError(e.Message); }
        finally { UpdateState(); }
    }
    private void SetError(string message)
    {
        _actionTitle.Text = "Ошибка · действие не подтверждено"; _actionDetail.Text = message; StartupLog.Write(message);
        _error.Text = message.Length > 1200 ? message[..1200] : message; _error.IsVisible = true;
        Log("ОШИБКА: " + message);
    }
    private async Task OpenWorkspace(string path)
    {
        FlushEditorDrafts(strict:true, includeCreation:false);
        _windowLife.Token.ThrowIfCancellationRequested();
        if (_openingWorkspace) throw new InvalidOperationException("Открытие модели уже выполняется.");
        CancelPendingModeEdit(); _openingWorkspace = true; UpdateState(); UpdateActivity();
        try { await OpenWorkspaceCore(path); }
        finally { _openingWorkspace = false; UpdateState(); UpdateActivity(); }
    }
    private async Task OpenWorkspaceCore(string path)
    {
        if (_generating) throw new InvalidOperationException("Сначала остановите текущий ответ.");
        path = Path.GetFullPath(path); ModelLibrary.NoLinks(path); bool createdDirectory = !Directory.Exists(path); Directory.CreateDirectory(path);
        if (createdDirectory && !OperatingSystem.IsWindows()) File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        bool same = _uiLease is not null && string.Equals(path, _workspace, comparison);
        ModelLibrary.NoLinks(Path.Combine(path, ".ui.lock"));
        var nextLease = same ? _uiLease! : new FileStream(Path.Combine(path, ".ui.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        bool installed = false, handoff = false;
        try
        {
            _status.Text = "Проверка новой рабочей папки. Текущая модель пока остаётся подключённой…";
            // A verified CPU snapshot and journal are prepared BEFORE any live state is torn down.
            // Reuse these exact objects after the switch: no second model hash/unpack or history read.
            var preview = await Task.Run(() => WorkspacePreview.Load(path,
                Math.Max(1, Math.Min(4, Environment.ProcessorCount)), _windowLife.Token), _windowLife.Token);
            _windowLife.Token.ThrowIfCancellationRequested();
            if (preview.NeedsConversationState) JsonData.AtomicWrite(Path.Combine(path, "chat-state.json"), preview.ConversationId, 4096);
            handoff = true; long epoch = ++_epoch; ++_publication; CancelSnapshotLoad(); _workerReady = false;
            if (_client is not null) { await _client.DisposeAsync(); _client = null; }
            // Keep the former workspace locked until its cancelled publication reader drains.
            await _modelLoader.WaitAsync(); _modelLoader.Release();
            _windowLife.Token.ThrowIfCancellationRequested();
            if (!same) _uiLease?.Dispose(); _uiLease = nextLease; installed = true;
            _workspace = path; _conversationId = preview.ConversationId; SetRunEditorContext(path);
            if (!same) ClearModelSpecificInputs();
            _model = null; _revision = null; _pending = null; _inferenceFile = null;
            _workerStatus = null; _operationBusy = false; _history.Clear(); _messages.Children.Clear();
            _workspaceLabel.Text = path; _modelLabel.Text = Path.GetFileName(path);
            if (preview.Weights is not null && preview.Revision is not null && preview.InferenceFile is not null)
                ApplySnapshot((new ManagedInference(preview.Weights, preview.Revision.Revision,
                    Math.Max(1, Math.Min(4, Environment.ProcessorCount - 1))), preview.Revision, preview.InferenceFile));
            ApplyHistory(preview.History);
            SavePreferences();
            try
            {
                _status.Text = "Проверка CPU/CUDA-тренера…";
                var located = await TrainerLocator.FindAsync(_windowLife.Token); _windowLife.Token.ThrowIfCancellationRequested(); _deviceNote.Text = located.Note; Log(located.Note);
                var client = new WorkerClient(path, located.Path); _client = client;
                client.Received += message => OnWorker(message, epoch);
                client.Diagnostic += text => Dispatcher.UIThread.Post(() => { if (epoch == _epoch) Log(text); });
                client.Faulted += text => Dispatcher.UIThread.Post(() => { if (epoch == _epoch) SetError(text); });
                client.Terminated += text => Dispatcher.UIThread.Post(() =>
                {
                    if (_closing || epoch != _epoch) return;
                    _workerReady = false; _operationBusy = false;
                    if (_workerStatus is not null) _workerStatus = _workerStatus with { Busy = false, Step = _revision?.Step ?? 0,
                        Accuracy = _revision?.Accuracy, Performance = null };
                    _progress.IsIndeterminate = false; UpdateState(); UpdateActivity(); UpdateStats();
                    SetError(text + " Можно переподключить тренер на вкладке обучения. Чат на сохранённых весах доступен.");
                });
                client.Start();
                var ready = await client.WaitReadyAsync(); _windowLife.Token.ThrowIfCancellationRequested(); ReceiveRunSettings(ready); _workerReady = true;
                SetOnlineCheck(ready.HasModel ? ready.Online && !ready.Paused : _preferences.OnlineLearning);
                _status.Text = ready.HasModel ? "Модель открыта. Чат и тренер готовы." : "Тренер подключён. Можно создавать модель.";
            }
            catch (Exception e)
            {
                if (_client is not null) { await _client.DisposeAsync(); _client = null; }
                _workerReady = false;
                if (_model is null || e is OperationCanceledException) throw;
                SetError(e.Message + " Чат на сохранённых весах доступен, обучение недоступно.");
            }
            RememberModel(path); await RefreshLibrary();
            UpdateState(); UpdateStats();
        }
        catch
        {
            if (!handoff && !_closing) _status.Text = "Новая папка не открыта. Предыдущая модель, переписка и черновик сохранены.";
            throw;
        }
        finally { if (!installed && !same) nextLease.Dispose(); }
    }
    private void OnWorker(WorkerEvent message, long epoch)
    {
        Dispatcher.UIThread.Post(async () =>
        {
            if (_closing || epoch != _epoch) return;
            await Safe(async () =>
            {
                switch (message.Kind)
                {
                    case "published": await LoadPublication(Protocol.Payload<PublishEvent>(message.Data), epoch, ++_publication); break;
                    case "ready":
                        var r = Protocol.Payload<ReadyEvent>(message.Data);
                        ReceiveRunSettings(r); _workerReady = true;
                        break;
                    case "dataset":
                        var dataset = Protocol.Payload<DatasetSummaryEvent>(message.Data);
                        _dataSummary.Text = $"Датасет {dataset.DatasetVersion}: обучение {dataset.TrainingExamples:N0}, контроль {dataset.ValidationExamples:N0}. Минимальная длина для целых ответов: {dataset.RequiredSequenceLength}.";
                        Log(_dataSummary.Text); break;
                    case "accepted": Log("Тренер принял: " + message.Data.GetProperty("command").GetString()); break;
                    case "mode-state":
                        var flags = Protocol.Payload<RuntimeFlags>(message.Data); SetOnlineCheck(flags.Online && !flags.Paused);
                        if (flags.OnlineLearningRate is double rate) { bool previous = _settingControls; _settingControls = true; try { SetEditorValue(_onlineRate, (decimal)rate); } finally { _settingControls = previous; } }
                        break;
                    case "status":
                        var incoming = Protocol.Payload<StatusEvent>(message.Data);
                        incoming.Accuracy?.Validate(incoming.Step, _model?.Weights.Config.Context ?? 2048);
                        _workerStatus = incoming; _status.Text = _workerStatus.Message;
                        if (_lastStage != _workerStatus.Stage) { _lastStage = _workerStatus.Stage; Log(_workerStatus.Message); }
                        UpdateStats(); UpdateActivity(); break;
                    case "error": SetError(message.Data.GetProperty("message").GetString() ?? "Ошибка тренера"); break;
                    case "completed":
                        if (message.Data.TryGetProperty("success", out var success) && success.ValueKind == JsonValueKind.False)
                        {
                            string? failedCommand = message.Data.GetProperty("command").GetString();
                            if (failedCommand is "create" or "train" or "rollback")
                            {
                                SetOnlineCheck(false);
                                if (_workerStatus is not null) _workerStatus = _workerStatus with { Busy = false };
                            }
                        }
                        break;
                    case "conversation-result": ShowConversationResult(message.Data); break;
                    case "stage-reference":
                        Log(message.Data.GetProperty("message").GetString() + " Файл: " + message.Data.GetProperty("file").GetString()); break;
                    case "warning":
                        _error.Text = message.Data.GetProperty("message").GetString(); _error.IsVisible = true; Log("ПРЕДУПРЕЖДЕНИЕ: " + _error.Text); break;
                    case "online-result":
                        Log(message.Data.ToString());
                        break;
                }
            }, clearError: false);
        });
    }
    private void CancelSnapshotLoad()
    {
        // Called on the UI dispatcher. The async load owns disposal; callers request cancellation only.
        var previous = _snapshotLoadCts; _snapshotLoadCts = null; previous?.Cancel();
    }
    private async Task LoadPublication(PublishEvent publication, long epoch, long version)
    {
        if (_closing || epoch != _epoch || version != _publication) return;
        string workspace = _workspace ?? throw new InvalidOperationException("Нет рабочей папки.");
        var signature = (epoch, publication.Info.Revision, publication.Info.ModelSha256);
        if (_snapshotLoadSignature == signature && _snapshotLoadCompletion is { } inFlight && !inFlight.Task.IsCompleted)
        {
            // The command's completion and the published event may announce the SAME snapshot.
            // They must await one load, not cancel each other and report readiness before activation.
            await inFlight.Task; return;
        }
        CancelSnapshotLoad();
        var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        _snapshotLoadSignature = signature; _snapshotLoadCompletion = completion;
        using var load = CancellationTokenSource.CreateLinkedTokenSource(_windowLife.Token);
        _snapshotLoadCts = load; bool entered = false;
        try
        {
            await _modelLoader.WaitAsync(load.Token); entered = true;
            if (_closing || epoch != _epoch || load.IsCancellationRequested) return;
            if ((_revision?.Revision == publication.Info.Revision && _revision.ModelSha256 == publication.Info.ModelSha256) ||
                (_pending is { } existing && existing.Info.Revision == publication.Info.Revision && existing.Info.ModelSha256 == publication.Info.ModelSha256)) return;
            var result = await Task.Run(() =>
            {
                string folder = ModelFiles.GetRevisionPath(workspace, publication.RevisionDirectory);
                var (weights, info) = ModelFiles.ReadInferenceRevision(folder, Math.Max(1, Math.Min(4, Environment.ProcessorCount)), load.Token);
                if (info.Revision != publication.Info.Revision || !StringComparer.OrdinalIgnoreCase.Equals(info.ModelSha256, publication.Info.ModelSha256))
                    throw new InvalidDataException("Уведомление о публикации не совпадает с проверенным снимком. Последняя модель чата сохранена.");
                load.Token.ThrowIfCancellationRequested();
                string file = Path.Combine(folder, "model.tritmodel");
                return (Model: new ManagedInference(weights, info.Revision, Math.Max(1, Math.Min(4, Environment.ProcessorCount - 1))), Info: info, Path: file);
            }, load.Token);
            if (_closing || epoch != _epoch || load.IsCancellationRequested) return;
            if (_generating) _pending = result;
            else ApplySnapshot(result);
        }
        catch (OperationCanceledException) when (load.IsCancellationRequested && !_windowLife.IsCancellationRequested)
        {
            // Superseded publication is not a failed user action. The last verified model stays usable.
        }
        catch (Exception error)
        {
            completion.TrySetException(error); _ = completion.Task.Exception; throw;
        }
        finally
        {
            completion.TrySetResult(true);
            if (ReferenceEquals(_snapshotLoadCts, load))
            { _snapshotLoadCts = null; _snapshotLoadSignature = null; _snapshotLoadCompletion = null; }
            if (entered) _modelLoader.Release();
        }
    }
    private void ApplySnapshot((ManagedInference Model, RevisionInfo Info, string Path) snapshot)
    {
        _model = snapshot.Model; _revision = snapshot.Info; _inferenceFile = snapshot.Path;
        _modelLabel.Text = $"{Path.GetFileName(_workspace)} · {_model.Weights.Config.ParameterCount:N0} параметров · r{snapshot.Info.Revision}";
        UpdateContextBudget(); UpdateState(); UpdateStats(); Log($"Снимок r{snapshot.Info.Revision} загружен в чат. Шаг {snapshot.Info.Step}.");
        if (snapshot.Info.Step == 0) _status.Text = "Модель создана, но ещё не обучена. Первые ответы могут быть бессмысленными.";
    }
    private async Task Create()
    {
        ValidateEditorNumbers((CreationMode)_creationMode.SelectedIndex == CreationMode.Untrained ? CreationBaseNumbers : CreationNumbers);
        FlushEditorDrafts(strict:false); // An unrelated unfinished editor is not a launch parameter.
        var mode = (CreationMode)Math.Clamp(_creationMode.SelectedIndex, 0, 2);
        bool onlineAfterCreate = mode == CreationMode.Conversation && _online.IsChecked == true;
        var config = SelectedConfig();
        // Raw creation deliberately ignores its hidden future batch/LR/step editors.
        var resources = mode == CreationMode.Untrained ? ZeroCreationResources(config) : CreationResources(config);
        var selectedTraining = mode == CreationMode.Untrained ? new TrainingOptions { Steps = 0 } : CreationTraining();
        if (mode != CreationMode.Untrained && selectedTraining.Steps == 0) throw new ArgumentException("Выберите «Без обучения» для нулевого этапа или укажите положительное число шагов.");
        var training = LearningStages.CreationOptions(mode, selectedTraining);
        config.Validate(); resources.ValidateInitialization(config); training.Validate();
        _ = CheckpointBudget.Plan(training, 0, includeInitial: true);
        if (_operationBusy) throw new InvalidOperationException("Дождитесь текущего обучения или остановите его.");
        if (training.Steps > 0 && training.LearningRate > 0.003 &&
            !await Confirm("Высокий learning rate", "LR выше 0,003 может сорвать обучение AdamW. 0,01 — верхняя граница, не минимум. Продолжить с выбранным LR=" + training.LearningRate.ToString("G6") + "?"))
            throw new OperationCanceledException();
        if(training.Steps > 0 && training.LearningRate < 0.00001)
        {
            _operationBusy = true; UpdateState();
            try
            {
                if(!await Confirm("Маленький LR новой модели", $"Выбрано {training.LearningRate:G6}. Это может оставить случайную сеть почти необученной даже после тысяч шагов. Продолжить с этим значением?"))
                    throw new OperationCanceledException();
            }
            finally { _operationBusy = false; UpdateState(); }
        }
        // Fail fast on user data before opening a new workspace or initializing native compute.
        var selectedPaths = Array.Empty<string>(); // Creation never silently borrows the selected model's files.
        using var preparation = CancellationTokenSource.CreateLinkedTokenSource(_windowLife.Token);
        _preparationCts = preparation; _operationBusy = true; _status.Text = "Проверяю датасеты до создания рабочей папки…"; UpdateState();
        try
        {
            await Task.Run(() =>
            {
                var examples = Dataset.SplitLongTexts(Dataset.LoadManyTraining(selectedPaths, preparation.Token), resources.SequenceLength, preparation.Token);
                foreach (var example in examples) { preparation.Token.ThrowIfCancellationRequested(); _ = Dataset.Encode(example, resources.SequenceLength); }
            }, preparation.Token);
            preparation.Token.ThrowIfCancellationRequested();
        }
        finally { _preparationCts = null; _operationBusy = false; UpdateState(); }
        _windowLife.Token.ThrowIfCancellationRequested();
        string path = AppPaths.NewWorkspace(_name.Text ?? "model"); await OpenWorkspace(path);
        if (_client is null) throw new InvalidOperationException("Нет исполняемого тренера для создания модели.");
        _operationBusy = true; _progress.IsIndeterminate = true; UpdateState();
        // Commands are serialized after the worker handshake; creating a model does not require a second click.
        _tabs.SelectedIndex = 0;
        try
        {
            await _client.WaitReadyAsync();
            await _client.Request("create", new CreateRequest(config, resources, training, selectedPaths, mode));
            // Await the final snapshot here as well; protocol completion alone does not mean the UI loaded it.
            string? active = ModelFiles.ActivePath(path);
            if (active is not null) await LoadPublication(new PublishEvent(Path.GetFileName(active), JsonData.Read<RevisionInfo>(Path.Combine(active, "revision.json"))), _epoch, ++_publication);
            if (!_runEditorInitialized && active is not null)
            {
                var saved = JsonData.Read<WorkspaceSettings>(Path.Combine(active, "settings.json"), 65536);
                ReceiveRunSettings(new ReadyEvent(true, saved.Config, saved.Resources, saved.Training, LastMaterial: saved.LastMaterial));
            }
            SetOnlineCheck(onlineAfterCreate); await ApplyOnline(allowOwnedOperation: true);
            _trainingMaterial.SelectedIndex = training.Steps == 0 ? 1 : 0;
            if (training.Steps == 0) SetEditorValue(_steps, 200);
            else if (mode == CreationMode.BasicPretrain) SetEditorValue(_steps, 1200);
            MarkRunEdited(); FlushEditorDrafts(strict:false);
            _status.Text = training.Steps == 0 ? $"Создана НЕОБУЧЕННАЯ модель: 0 шагов. Автообучение выключено. Проверьте ответы, затем отдельно запустите базовый претрейн. Папка: {path}" :
                mode == CreationMode.BasicPretrain ? $"Базовый претрейн завершён. Разговорный корпус ещё не использовался. Следующий этап: разговорное обучение. Папка: {path}" : $"Создание и разговорное обучение завершены. Модель активна в чате: {path}";
        }
        finally { _operationBusy = false; await RefreshLibrary(); }
    }
    private async Task Train()
    {
        NeedWorker(); if (_model is null) return;
        var draft = CaptureRunDraft(); SaveRunDraft();
        var resources = draft.Resources; resources.ValidateInitialization(_model.Weights.Config); var training = draft.Training; training.Validate();
        _operationBusy = true; _progress.IsIndeterminate = true; UpdateState();
        try
        {
            if (training.LearningRate < 0.00001 && !await Confirm("Очень маленький learning rate",
                $"Сейчас LR={training.LearningRate:G6}. Это в {0.001/training.LearningRate:N0} раз меньше учебного старта 0,001. " +
                "При обучении с нуля даже тысячи шагов могут оставить почти случайную речь. Продолжить с выбранным значением?")) throw new OperationCanceledException();
            if (training.LearningRate > 0.003 && !await Confirm("Высокий learning rate",
                "0,01 — верхняя допустимая граница, не рекомендуемый минимум. Большие шаги AdamW могут разрушить уже выученное. Продолжить с LR=" + training.LearningRate.ToString("G6") + "?")) throw new OperationCanceledException();
            var material = draft.Material;
            if (material == TrainingMaterial.BasicPretrain && _datasetPaths.Length != 0)
                throw new ArgumentException("Для чистого базового этапа уберите выбранные файлы. Они не будут использоваться молча.");
            var paths = material == TrainingMaterial.BasicPretrain ? Array.Empty<string>() : (string[])_datasetPaths.Clone();
            Log($"Запрошен запуск: LR={training.LearningRate:G6}, пакет={resources.BatchSize}, максимум={resources.SequenceLength}, шагов={training.Steps}, публикация={training.PublishEvery}. Это параметры запроса, не отчёт о завершении.");
            await _client!.Request("train", new TrainRequest(paths, training, resources, draft.IncludeBundledUpdates, material));
            string? active = _workspace is null ? null : ModelFiles.ActivePath(_workspace);
            if (active is not null) await LoadPublication(new PublishEvent(Path.GetFileName(active), JsonData.Read<RevisionInfo>(Path.Combine(active, "revision.json"))), _epoch, ++_publication);
            SetOnlineCheck(false);
            if (material == TrainingMaterial.BasicPretrain) _trainingMaterial.SelectedIndex = 0;
            MarkRunEdited(); FlushEditorDrafts(strict:false);
            _status.Text = "Этап обучения завершён. Снимок опубликован. Путь к эталону или предупреждение о его сохранении показаны в журнале. Автообучение выключено для чистого сравнения.";
        }
        finally { _operationBusy = false; }
    }
    private bool _composerEnterDown;
    private async void ComposerKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Handled || e.Key != Key.Enter || e.KeyModifiers != KeyModifiers.None) return;
        e.Handled = true; // No newline even when Send is currently unavailable.
        if (_composerEnterDown) return;
        _composerEnterDown = true;
        if (!_send.IsEnabled)
        {
            Notice("Сообщение не отправлено", _generating ? "Дождитесь ответа или остановите его. Черновик сохранён." :
                _model is null ? "Сначала выберите модель. Черновик сохранён." :
                !_promptValid ? "Запрос превышает доступный контекст. Сократите текст или лимит ответа." : "Дождитесь завершения текущего действия. Черновик сохранён.");
            return;
        }
        await RunButton(_send, "Ответ модели", StartSend);
    }
    private Task StartSend()
    {
        if (_generating) return Task.CompletedTask;
        if (_model is null) throw new InvalidOperationException("Сначала создайте или откройте модель.");
        if (string.IsNullOrWhiteSpace(_input.Text)) throw new ArgumentException("Введите сообщение перед отправкой.");
        _generationTask = Send(); return _generationTask;
    }
    private async Task Send()
    {
        var model = _model!; string input = _input.Text!.Trim(); var options = _sampling;
        var plan = ByteTokenizer.Plan(_history, input, model.Weights.Config.Context, Math.Min(options.MaxNewTokens, model.Weights.Config.Context - 4));
        var prompt = plan.Tokens;
        var learningHistory = TeachingContext.Capture(_history, plan.RetainedTurns, _conversationId);
        bool excluded = _private.IsChecked == true; _private.IsChecked = false;
        _generating = true; _generationStarted = DateTimeOffset.Now; _generationCts = new(); var ct = _generationCts.Token;
        _input.Text = ""; _followTail = true; AddBubble("Вы" + (excluded ? " · не обучать" : ""), input, true);
        var response = AddBubble($"Трит · r{model.Revision}", "Генерирую ответ…", false);
        UpdateState(); var watch = Stopwatch.StartNew();
        using var textUpdates = new LatestValueMailbox<string>(action => Dispatcher.UIThread.Post(action), text =>
        {
            if (!ct.IsCancellationRequested && !_closing)
            { response.Text.Text = string.IsNullOrEmpty(text) ? "…" : text; FollowTail(); }
        });
        try
        {
            if (!excluded && _online.IsChecked == true && _workerReady && _client is not null)
                // Do not let a queued training job delay the first generated token.
                _ = QueueUserExample(_client, input, _epoch);
            var generated = await Task.Run(() => model.GenerateDetailed(prompt, () => Volatile.Read(ref _sampling), textUpdates.Post, ct), ct);
            textUpdates.Dispose(); // An older queued partial must not overwrite the completed answer.
            string answer = generated.Text;
            response.Text.Text = string.IsNullOrEmpty(answer) ? "[Модель завершила ответ без текста]" : answer;
            var turn = new ChatTurn(input, answer, model.Revision, excluded, _conversationId, learningHistory); _history.Add(turn); if (_history.Count > 250) _history.RemoveAt(0);
            if (_workspace is not null)
            {
                try { await ChatJournal.AppendAsync(Path.Combine(_workspace, "chat.jsonl"), turn); }
                catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException)
                {
                    // The answer already exists. A journal failure is NOT a model failure or a reason
                    // to restore the old prompt and generate the same exchange again.
                    response.Body.Children.Add(Text("Ответ получен, но переписку не удалось записать на диск. Этот обмен пока только в памяти окна.", 12));
                    SetError("Не удалось сохранить переписку: " + error.Message + " Ответ получен; проверьте свободное место и права на рабочую папку.");
                }
            }
            AddGenerationWarning(response.Body, answer);
            AddTeaching(response.Body, turn);
            string finish = generated.StopReason switch { "eos" => "модель завершила ответ", "token_limit" => "достигнут лимит ответа", "utf8_boundary" => "лимит изменён внутри символа; неполный символ не показан", _ => "заполнен контекст" };
            _status.Text = $"Ответ за {watch.Elapsed.TotalSeconds:F2} с · r{model.Revision} · {generated.GeneratedByteTokens} байт-токенов · {finish}.";
        }
        catch (OperationCanceledException)
        {
            textUpdates.Dispose();
            if (response.Text.Text is "Генерирую ответ…" or "…") response.Text.Text = "Ответ остановлен до выдачи текста.";
            response.Body.Children.Add(Text("Остановлено. Неполный ответ не сохранён и не используется как эталон.", 12));
            RestoreFailedDraft(input, excluded);
            throw; // Let RunButton show cancelled, not a misleading completed receipt.
        }
        catch (Exception error)
        {
            textUpdates.Dispose();
            if (response.Text.Text is "Генерирую ответ…" or "…") response.Text.Text = "Не удалось получить ответ.";
            response.Body.Children.Add(Text("Ошибка: " + error.Message + " Ответ не подтверждён как учебный пример.", 12));
            // Do not overwrite a draft the user already typed while generation was running.
            RestoreFailedDraft(input, excluded);
            throw;
        }
        finally
        {
            _generationCts.Dispose(); _generationCts = null; _generating = false; _generationStarted = null;
            if (_pending is { } next) { _pending = null; ApplySnapshot(next); }
            UpdateContextBudget(); UpdateState(); _input.Focus();
        }
    }
    private void RestoreFailedDraft(string input, bool excluded)
    {
        var restored = DraftRecovery.Restore(_input.Text, _private.IsChecked == true, input, excluded);
        if (!restored.Restored) return;
        _input.Text = restored.Text; _private.IsChecked = restored.ExcludeNext;
    }
    private (SelectableTextBlock Text, StackPanel Body) AddBubble(string who, string content, bool user)
    {
        var label = Text(who, 12); label.Foreground = Muted;
        var text = new SelectableTextBlock { Text = content, TextWrapping = TextWrapping.Wrap, FontSize = 14 };
        var body = Column(label, text);
        var border = Card(body); border.Background = new SolidColorBrush(Color.Parse(user ? "#1E3045" : "#172033"));
        _messages.Children.Add(border); while (_messages.Children.Count > 500) _messages.Children.RemoveAt(0);
        FollowTail(); return (text, body);
    }
    private async Task QueueUserExample(WorkerClient client, string input, long epoch)
    {
        try
        {
            await client.Request("online", new OnlineRequest(Dataset.Make(input, source: "chat-user")));
            if (epoch == _epoch) Log("Сообщение записано в обучающую очередь. Это не эталон ответа.");
        }
        catch (Exception error) { if (epoch == _epoch) SetError("Сообщение НЕ принято для обучения: " + error.Message); }
    }
    private void AddTeaching(StackPanel body, ChatTurn turn)
    {
        // Excluded messages may remain in chat context but must not re-enter training through a feedback button.
        if (turn.ExcludedFromTraining) { body.Children.Add(Text("Этот обмен исключён из обучения. Кнопки обучения отключены.", 12)); return; }
        if (string.IsNullOrWhiteSpace(turn.Assistant)) { body.Children.Add(Text("Пустой ответ не может стать эталоном обучения.", 12)); return; }
        // Capture context NOW, not when the user expands feedback after history pruning or a model switch.
        var fixedContext = Dataset.LearningContext(_history, turn); long teachingEpoch = _epoch;
        var feedback = new Expander { Header = "Оценить или исправить ответ", FontSize = 11 };
        feedback.PropertyChanged += (_,e) =>
        {
            if (e.Property.Name == "IsExpanded" && feedback.IsExpanded && feedback.Content is null)
                feedback.Content = BuildTeachingContent(turn,fixedContext,teachingEpoch);
        };
        body.Children.Add(feedback);
    }
    private Control BuildTeachingContent(ChatTurn turn, ChatTurn[] fixedContext, long teachingEpoch)
    {
        var correct = Button("Ответ верный"); correct.FontSize = 12;
        var editor = new TextBox { Text = turn.Assistant, AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MinHeight = 70, MaxHeight = 180 };
        var teach = Button("Обучить на исправлении"); teach.FontSize = 12;
        var receipt = Text("Ответ пока не используется как эталон.", 12); receipt.Foreground = Muted;
        async Task Queue(string answer)
        {
            NeedWorker(); long ownerEpoch = _epoch; var ownerClient = _client!;
            if (teachingEpoch != _epoch) throw new OperationCanceledException("Эта реплика относится к другой рабочей модели.");
            var context = fixedContext;
            var result = await ownerClient.Request("online", new OnlineRequest(Dataset.Make(turn.User, answer, "chat-approved", context)));
            if (_closing || ownerEpoch != _epoch) throw new OperationCanceledException("Рабочая модель уже переключена.");
            bool queued = result.Data.TryGetProperty("result", out var resultData) && resultData.ValueKind == JsonValueKind.Object && resultData.GetProperty("queued").GetInt32() > 0;
            receipt.Text = queued ? "Пример сохранён в очереди. Изменение весов будет отдельным событием." : "Этот пример уже есть в журнале. Дублирующее обучение не запущено.";
            if (_online.IsChecked != true) receipt.Text += " Включите автообучение для обработки очереди.";
            _status.Text = receipt.Text;
        }
        correct.Click += async (_, _) => await RunButton(correct, "Подтверждение ответа", async () => { await Queue(turn.Assistant); correct.Tag = LearnedButtonTag; });
        teach.Click += async (_, _) => await RunButton(teach, "Обучение на исправлении", () => Queue(editor.Text ?? ""));
        var expansion = new Expander { Header = "Исправить ответ и обучить", Content = Column(editor, teach), FontSize = 12 };
        return Column(correct, expansion, receipt);
    }
    private void FollowTail() { Dispatcher.UIThread.Post(() => { if (_followTail) _scroll.ScrollToEnd(); _latest.IsVisible = !_followTail && _messages.Children.Count > 0; }, DispatcherPriority.Background); }
    private async Task LoadHistory(string workspace)
    {
        var recovered = await Task.Run(() => ChatJournal.ReadTail(Path.Combine(workspace, "chat.jsonl"), ct: _windowLife.Token));
        ApplyHistory(recovered);
    }
    private void ApplyHistory(ChatReadResult recovered)
    {
        var turns = recovered.Turns;
        if (recovered.SkippedRecords > 0) Log($"В просмотренном хвосте журнала пропущено повреждённых записей: {recovered.SkippedRecords}. Более старые записи не проверялись; исходный файл сохранён.");
        _windowLife.Token.ThrowIfCancellationRequested();
        foreach (var turn in turns.Where(x => x.ConversationId == _conversationId))
        {
            _history.Add(turn); AddBubble("Вы" + (turn.ExcludedFromTraining ? " · не обучать" : ""), turn.User, true);
            var response = AddBubble($"Трит · r{turn.Revision}", turn.Assistant, false); AddGenerationWarning(response.Body, turn.Assistant); AddTeaching(response.Body, turn);
        }
        UpdateContextBudget();
    }
    private async Task OpenFolder()
    {
        var result = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Рабочая папка модели Trit Studio", AllowMultiple = false });
        string? path = result.FirstOrDefault()?.TryGetLocalPath();
        if (path is null) throw new OperationCanceledException("Выбор файла отменён.");
        if (!File.Exists(Path.Combine(path, "active.json"))) throw new InvalidDataException("Выбрана не рабочая папка модели: отсутствует active.json. Для новой модели используйте вкладку создания.");
        await OpenWorkspace(path);
    }
    private async Task OpenPacked()
    {
        if (_generating) throw new InvalidOperationException("Остановите текущий ответ.");
        var result = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { Title = "Готовая модель для CPU-чата", AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Trit Studio") { Patterns = new[] { "*.tritmodel" } } } });
        string? path = result.FirstOrDefault()?.TryGetLocalPath(); if (path is null) throw new OperationCanceledException("Выбор файла отменён.");
        await OpenPackedPath(path);
    }
    private async Task OpenPackedPath(string path)
    {
        FlushEditorDrafts(strict:true, includeCreation:false);
        if (_generating) throw new InvalidOperationException("Сначала остановите ответ.");
        path = Path.GetFullPath(path); ModelLibrary.NoLinks(path);
        var weights = await Task.Run(() => ModelFiles.Read(path, true, Math.Max(1, Math.Min(4, Environment.ProcessorCount)), _windowLife.Token)); _windowLife.Token.ThrowIfCancellationRequested();
        ++_epoch; ++_publication; CancelSnapshotLoad(); if (_client is not null) { await _client.DisposeAsync(); _client = null; }
        await _modelLoader.WaitAsync(); _modelLoader.Release();
        _windowLife.Token.ThrowIfCancellationRequested();
        _uiLease?.Dispose(); _uiLease = null; _workerReady = false; _operationBusy = false; _workspace = null; SetRunEditorContext(null); _pending = null; _workerStatus = null; _revision = null;
        _model = new(weights, -1, Math.Max(1, Math.Min(4, Environment.ProcessorCount))); _inferenceFile = path;
        _workspaceLabel.Text = path + " · только инференс"; _deviceNote.Text = "CPU-чат. Обучающий процесс не подключён."; SetOnlineCheck(false);
        _messages.Children.Clear(); _history.Clear(); _conversationId = Guid.NewGuid().ToString("N"); ClearModelSpecificInputs(); _modelLabel.Text = Path.GetFileName(path) + " · только чат";
        _status.Text = "Для дообучения нужна рабочая папка с мастер-весами и оптимизатором, а не только экспорт.";
        _tabs.SelectedIndex = 0; RememberModel(path); await RefreshLibrary(); SavePreferences(); UpdateContextBudget(); UpdateState(); UpdateStats();
    }
    private async Task Export()
    {
        if (_inferenceFile is null) throw new InvalidOperationException("Нет модели для экспорта.");
        string source = _inferenceFile;
        var target = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions { Title = "Экспорт переносимой модели", SuggestedFileName = "model.tritmodel",
            DefaultExtension = "tritmodel", FileTypeChoices = new[] { new FilePickerFileType("Trit Studio") { Patterns = new[] { "*.tritmodel" } } } });
        string? path = target?.TryGetLocalPath(); if (path is null) throw new OperationCanceledException("Выбор файла отменён.");
        _windowLife.Token.ThrowIfCancellationRequested();
        string destination = Path.GetFullPath(path);
        ExportGuard.Validate(source, destination, _workspace);
        await Task.Run(() =>
        {
            string staged = destination + ".stage-" + Guid.NewGuid().ToString("N");
            try { File.Copy(source, staged, false); File.Move(staged, destination, true); }
            finally { if (File.Exists(staged)) File.Delete(staged); }
        }); _status.Text = $"Экспорт готов: {destination}. Для переноса обучения копируйте всю рабочую папку после закрытия приложения.";
    }
    private async Task PickDatasets()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions { Title = "Добавить датасеты", AllowMultiple = true,
            FileTypeFilter = new[] { new FilePickerFileType("Датасеты") { Patterns = new[] { "*.txt", "*.json", "*.jsonl" } } } });
        var paths = files.Select(x => x.TryGetLocalPath()).Where(x => x is not null).Cast<string>().ToArray();
        if (paths.Length == 0) throw new OperationCanceledException("Выбор датасета отменён.");
        var combined = _datasetPaths.Concat(paths).Distinct().ToArray();
        var examples = await Task.Run(() => Dataset.LoadManyTraining(combined, _windowLife.Token), _windowLife.Token);
        _windowLife.Token.ThrowIfCancellationRequested();
        if (examples.Length == 0) throw new InvalidDataException("В выбранных файлах нет обучающих примеров.");
        _datasetPaths = combined;
        _pathsLabel.Text = "Выбранные файлы (для разговорного/своего этапа): " + string.Join(", ", _datasetPaths.Select(Path.GetFileName));
        int count = examples.Length, required = examples.Max(Dataset.RequiredSequenceLength);
        _status.Text = $"Файлы проверены: {paths.Length}, примеров: {count}. Для целых диалогов нужна длина не меньше {required}. Нажмите дообучение, чтобы изменить веса.";
    }

    private async Task ClearChat()
    {
        if (!await Confirm("Очистить чат?", "Сообщения исчезнут из окна и не войдут в контекст нового разговора. " +
            "Сохранённая переписка останется на диске. Веса, очередь и режим автообучения не изменятся. " +
            "Уже поставленные в очередь примеры могут продолжить обучаться отдельно. Набранный черновик сохранится."))
            throw new OperationCanceledException();
        ClearConversation();
    }
    private void ClearConversation()
    {
        _windowLife.Token.ThrowIfCancellationRequested();
        if (_generating || _openingWorkspace || _modeApplying || _operationBusy)
            throw new InvalidOperationException("Дождитесь окончания текущего действия и остановите ответ перед очисткой.");
        // Commit a new conversation ID FIRST. A storage failure must leave the visible conversation intact.
        string id = ChatSessionState.StartNew(_workspace, _windowLife.Token);
        _conversationId = id; _history.Clear(); _messages.Children.Clear(); _followTail = true;
        _latest.IsVisible = false; _scroll.Offset = default;
        // Preserve draft, its exclusion, weights, worker and online queue. No learning command is sent.
        UpdateContextBudget(); UpdateState();
        _status.Text = "Контекст очищен. Начат новый разговор. Черновик, сохранённая переписка, веса и очередь обучения сохранены.";
        _input.Focus();
    }

    private async Task<bool> Confirm(string title, string detail)
    {
        var dialog = new Window { Title = title, Width = 480, SizeToContent = SizeToContent.Height,
            CanResize = false, WindowStartupLocation = WindowStartupLocation.CenterOwner, Background = Background };
        var yes = Button("Подтвердить"); var no = Button("Отмена");
        yes.Click += (_, _) => dialog.Close(true); no.Click += (_, _) => dialog.Close(false);
        dialog.Content = new Border { Padding = new Thickness(22), Child = Column(Text(title, 19), Text(detail), Row(yes, no)) };
        return await dialog.ShowDialog<bool>(this);
    }
    private async Task MaintainSnapshots()
    {
        string path = _workspace ?? throw new InvalidOperationException("Откройте рабочую папку модели.");
        if (_generating || _operationBusy) throw new InvalidOperationException("Сначала остановите ответ и большое обучение.");
        if (!await Confirm("Сохранить 8 последних снимков?", "Останутся активная модель и её цепочка отката, всего до 8 снимков. Старые снимки удалятся, переписка сохранится. Обучение будет остановлено.")) throw new OperationCanceledException();
        _windowLife.Token.ThrowIfCancellationRequested();
        _operationBusy = true; UpdateState(); MaintenanceResult? result = null; Exception? failure = null;
        try
        {
            _status.Text = "Останавливаю тренер перед обслуживанием…";
            if (_client is not null && _workerReady) await _client.Request("stop", new { });
            ++_epoch; ++_publication; CancelSnapshotLoad(); _workerReady = false;
            if (_client is not null) { await _client.DisposeAsync(); _client = null; }
            await _modelLoader.WaitAsync();
            try
            {
                _pending = null; _status.Text = "Проверяю цепочку отката и освобождаю место…";
                var lease = _uiLease ?? throw new InvalidOperationException("Рабочая папка не заблокирована приложением.");
                result = await Task.Run(() => WorkspaceMaintenance.PruneWithUiLease(path, lease, keep: 8, apply: true));
            }
            finally { _modelLoader.Release(); }
        }
        catch (Exception error) { failure = error; }
        finally
        {
            _operationBusy = false;
            if (!_closing && _client is null) try { await OpenWorkspace(path); } catch (Exception error) { failure ??= error; }
            UpdateState();
        }
        if (failure is not null) throw new IOException("Обслуживание не завершено: " + failure.Message, failure);
        _status.Text = $"Удалено снимков: {result!.Removed.Length}; освобождено {result.BytesFreed / 1048576.0:F1} МиБ; сохранено {result.Kept}. Автообучение выключено.";
    }
}
