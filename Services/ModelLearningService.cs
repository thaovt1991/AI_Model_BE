using System.Diagnostics;
using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// BƯỚC 2 trong pipeline "học ngầm": ĐIỀU PHỐI HUẤN LUYỆN LoRA.
///
/// === LoRA là gì? (giải thích ngắn cho người mới) ===
/// - Model .gguf gốc ~5GB, chứa hàng tỷ con số (trọng số / weights).
/// - Fine-tune toàn bộ = cực chậm, cần GPU mạnh.
/// - LoRA = chỉ học thêm một "lớp vá" nhỏ (~vài MB) đặt lên model gốc.
///   Ví dụ: model gốc biết tiếng Việt chung chung → LoRA học thêm phong cách/công ty của bạn.
///
/// === Mặc định trong project ===
/// - Learning:Enabled = false → KHÔNG train, chỉ thu thập dataset (học ngầm giai đoạn 1).
/// - Khi bật Enabled = true → gọi script Python train_lora.py tạo file adapter .gguf.
/// - Sau train xong → LlamaChatService nạp adapter và áp dụng khi chat.
///
/// === File quan trọng ===
/// - Learning/dataset.jsonl     — dữ liệu thu thập từ chat
/// - Learning/finetune.txt      — chuyển đổi sang format dễ đọc cho tool train
/// - Learning/Adapters/latest.gguf — adapter LoRA sau khi train
/// - Learning/state.json        — trạng thái job (đang train / lỗi / xong...)
/// </summary>
public sealed class ModelLearningService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly IConfiguration _configuration;
    private readonly ILogger<ModelLearningService> _logger;
    private readonly LearningDataCollectorService _collector;

    // Dùng IServiceProvider thay vì inject trực tiếp LlamaChatService — tránh vòng lặp DI
    // (LlamaChatService cũng cần ModelLearningService)
    private readonly LearningSettingsService _settings;
    private readonly IServiceProvider _serviceProvider;

    private readonly string _learningDir;
    private readonly string _statePath;
    private readonly string _adapterPath;

    // Đọc từ LearningSettingsService — có thể đổi realtime từ UI
    private readonly int _minSamplesForTraining;

    // Mỗi lần train chỉ lấy tối đa N mẫu gần nhất — tránh train quá lâu trên CPU
    private readonly int _maxSamplesPerTrain;

    // Sau lượt chat cuối, chờ N phút rồi mới train — tránh máy lag khi user đang dùng
    private readonly int _idleMinutesAfterChat;

    private readonly string _pythonPath;
    private readonly string _trainScriptPath;

    // Độ mạnh adapter khi áp dụng lên model (1.0 = 100%, 0.5 = nhẹ hơn)
    private readonly float _adapterScale;

    // Chỉ cho 1 job train chạy tại một thời điểm
    private readonly SemaphoreSlim _trainGate = new(1, 1);

    private LearningJobStatus _status = LearningJobStatus.Disabled;
    private int _trainedSamples;
    private int _adapterVersion;   // Tăng mỗi lần train thành công — theo dõi phiên bản adapter
    private DateTime? _lastTrainUtc;
    private string? _lastError;
    private IReadOnlyList<string> _lastLearnedTopics = [];
    private string? _lastTrainingMessage;

    public ModelLearningService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        LearningDataCollectorService collector,
        LearningSettingsService settings,
        IServiceProvider serviceProvider,
        ILogger<ModelLearningService> logger)
    {
        _configuration = configuration;
        _collector = collector;
        _settings = settings;
        _serviceProvider = serviceProvider;
        _logger = logger;

        _learningDir = Path.Combine(
            env.ContentRootPath,
            configuration.GetValue("Learning:Directory", "Learning") ?? "Learning");
        _statePath = Path.Combine(_learningDir, "state.json");
        _adapterPath = ResolvePath(configuration["Learning:AdapterPath"] ?? "Learning/Adapters/latest.gguf", env.ContentRootPath);
        _minSamplesForTraining = configuration.GetValue("Learning:MinSamplesForTraining", 20);
        _maxSamplesPerTrain = configuration.GetValue("Learning:MaxSamplesPerTrain", 200);
        _idleMinutesAfterChat = configuration.GetValue("Learning:IdleMinutesAfterLastChat", 30);
        _pythonPath = configuration["Learning:PythonPath"] ?? "python";
        _trainScriptPath = ResolvePath(
            configuration["Learning:TrainScriptPath"] ?? "Learning/train_lora.py",
            env.ContentRootPath);
        _adapterScale = configuration.GetValue("Learning:AdapterScale", 1.0f);

        Directory.CreateDirectory(Path.GetDirectoryName(_adapterPath)!);
        LoadState();
        RefreshStatus();
    }

    public bool TrainingEnabled => _settings.TrainingEnabled;
    public float AdapterScale => _adapterScale;
    public string AdapterPath => _adapterPath;
    public bool AdapterExists => File.Exists(_adapterPath);

    /// <summary>API GET /api/ai/learning/status — xem tiến độ học ngầm.</summary>
    public LearningStatusResponse GetStatus()
    {
        _collector.FlushPendingSamples();

        return new LearningStatusResponse(
            Status: _status,
            TotalSamples: _collector.TotalSamples,
            PendingSamples: _collector.PendingSamples,
            TrainedSamples: _trainedSamples,
            TrainingEnabled: _settings.TrainingEnabled,
            CollectDataEnabled: _collector.CollectDataEnabled,
            AdapterPath: AdapterExists ? _adapterPath : null,
            AdapterVersion: _adapterVersion,
            LastError: _lastError,
            LastTrainUtc: _lastTrainUtc,
            LearnedTopics: _lastLearnedTopics,
            LastTrainingMessage: _lastTrainingMessage);
    }

    /// <summary>
    /// Thử chạy train LoRA — gọi từ background worker hoặc API POST /api/ai/learning/train.
    /// Kiểm tra đủ điều kiện trước khi chạy process Python.
    /// </summary>
    public async Task<LearningTrainResponse> TryTrainInBackgroundAsync(CancellationToken cancellationToken = default)
    {
        if (!_settings.TrainingEnabled)
        {
            return new LearningTrainResponse(
                false,
                "Huấn luyện đang tắt. Bật trong Thiết lập → Học model (LoRA).");
        }

        _collector.FlushPendingSamples();

        if (_collector.TotalSamples < _minSamplesForTraining)
        {
            _status = LearningJobStatus.Collecting;
            PersistState();
            return new LearningTrainResponse(
                false,
                $"Chưa đủ mẫu để train ({_collector.TotalSamples}/{_minSamplesForTraining}).");
        }

        // Chờ user ngừng chat một lúc — train tốn CPU/RAM, không nên chạy song song chat
        if (_collector.LastChatUtc is { } lastChat &&
            DateTime.UtcNow - lastChat < TimeSpan.FromMinutes(_idleMinutesAfterChat))
        {
            _status = LearningJobStatus.Queued;
            PersistState();
            return new LearningTrainResponse(false, "Đang chờ hệ thống rảnh sau lượt chat gần nhất.");
        }

        if (!await _trainGate.WaitAsync(0, cancellationToken))
        {
            return new LearningTrainResponse(false, "Đang có job train khác chạy.");
        }

        try
        {
            _status = LearningJobStatus.Training;
            _lastError = null;
            PersistState();

            var trained = await RunTrainingProcessAsync(cancellationToken);
            if (!trained)
            {
                _status = LearningJobStatus.Failed;
                PersistState();
                return new LearningTrainResponse(false, _lastError ?? "Train thất bại.");
            }

            _trainedSamples = Math.Min(_collector.TotalSamples, _maxSamplesPerTrain);
            _adapterVersion++;
            _lastTrainUtc = DateTime.UtcNow;
            _status = LearningJobStatus.Ready;
            _lastError = null;

            var trainedBatch = _collector.ReadRecentSamples(_maxSamplesPerTrain);
            _lastLearnedTopics = SummarizeLearnedTopics(trainedBatch);
            _lastTrainingMessage = BuildTrainingMessage(_lastLearnedTopics, trainedBatch.Count);

            PersistState();

            // Bước 3: báo LlamaChatService nạp adapter mới — model chat sẽ "thông minh hơn" theo dữ liệu đã học
            await _serviceProvider.GetRequiredService<LlamaChatService>().ReloadLearningAdapterAsync(cancellationToken);

            _logger.LogInformation(
                "Learning: train LoRA xong — adapter v{Version} tại {Path}",
                _adapterVersion,
                _adapterPath);

            return new LearningTrainResponse(
                true,
                $"Đã train adapter v{_adapterVersion}.",
                _lastTrainingMessage);
        }
        catch (Exception ex)
        {
            _status = LearningJobStatus.Failed;
            _lastError = ex.Message;
            PersistState();
            _logger.LogError(ex, "Learning: lỗi khi train");
            return new LearningTrainResponse(false, ex.Message);
        }
        finally
        {
            _trainGate.Release();
        }
    }

    /// <summary>
    /// Chạy script Python train_lora.py như một process riêng (không block UI/API lâu).
    /// Python xử lý phần nặng: đọc dataset, gọi llama.cpp hoặc HuggingFace PEFT.
    /// </summary>
    private async Task<bool> RunTrainingProcessAsync(CancellationToken cancellationToken)
    {
        if (!File.Exists(_trainScriptPath))
        {
            _lastError = $"Không tìm thấy script train: {_trainScriptPath}";
            return false;
        }

        var modelPath = _configuration["Llama:ModelPath"] ?? string.Empty;
        if (!Path.IsPathRooted(modelPath))
        {
            modelPath = Path.Combine(AppContext.BaseDirectory, modelPath);
        }

        // Chuyển JSONL → finetune.txt (format Instruction/Response dễ đọc cho tool train)
        var finetuneTxt = Path.Combine(_learningDir, "finetune.txt");
        ExportFinetuneText(finetuneTxt);

        var args =
            $"\"{_trainScriptPath}\" " +
            $"--dataset \"{_collector.DatasetPath}\" " +
            $"--finetune-text \"{finetuneTxt}\" " +
            $"--output \"{_adapterPath}\" " +
            $"--base-model \"{modelPath}\" " +
            $"--max-samples {_maxSamplesPerTrain}";

        var psi = new ProcessStartInfo
        {
            FileName = _pythonPath,
            Arguments = args,
            WorkingDirectory = _learningDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        // Truyền cấu hình thêm qua biến môi trường cho script Python
        psi.Environment["LEARNING_HF_MODEL"] = _configuration["Learning:HuggingFaceModelId"] ?? string.Empty;
        psi.Environment["LEARNING_LLAMA_CPP_BIN"] = _configuration["Learning:LlamaCppBinPath"] ?? string.Empty;

        using var process = Process.Start(psi);
        if (process is null)
        {
            _lastError = $"Không chạy được {_pythonPath}. Cài Python hoặc cập nhật Learning:PythonPath.";
            return false;
        }

        var stdout = await process.StandardOutput.ReadToEndAsync(cancellationToken);
        var stderr = await process.StandardError.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken);

        if (!string.IsNullOrWhiteSpace(stdout))
        {
            _logger.LogInformation("Learning train stdout: {Stdout}", stdout.Trim());
        }

        if (process.ExitCode != 0)
        {
            _lastError = string.IsNullOrWhiteSpace(stderr)
                ? $"Script train thoát với mã {process.ExitCode}."
                : stderr.Trim();
            _logger.LogWarning("Learning train stderr: {Stderr}", _lastError);
            return false;
        }

        if (!File.Exists(_adapterPath))
        {
            _lastError = "Script train xong nhưng chưa tạo file adapter .gguf.";
            return false;
        }

        return true;
    }

    /// <summary>
    /// Xuất dataset sang file text — nhiều công cụ train đọc format "### Instruction:" / "### Response:".
    /// </summary>
    private void ExportFinetuneText(string finetuneTxt)
    {
        var samples = _collector.ReadRecentSamples(_maxSamplesPerTrain);
        using var writer = new StreamWriter(finetuneTxt, false, System.Text.Encoding.UTF8);
        foreach (var sample in samples)
        {
            writer.WriteLine("### Instruction:");
            writer.WriteLine(sample.Instruction);
            writer.WriteLine();
            writer.WriteLine("### Response:");
            writer.WriteLine(sample.Output);
            writer.WriteLine();
        }
    }

    public LearningSettingsResponse GetSettingsResponse()
    {
        var status = GetStatus();
        var settings = _settings.Get();
        return new LearningSettingsResponse(
            settings.Enabled,
            settings.CollectData,
            status.Status,
            status.TotalSamples,
            status.PendingSamples,
            status.AdapterPath,
            status.AdapterVersion,
            status.LastError,
            status.LastTrainUtc,
            status.LearnedTopics,
            status.LastTrainingMessage);
    }

    /// <summary>Cập nhật trạng thái sau khi user đổi toggle trên UI.</summary>
    public void RefreshStatus()
    {
        if (!_settings.CollectDataEnabled && !_settings.TrainingEnabled)
        {
            _status = LearningJobStatus.Disabled;
        }
        else if (!_settings.TrainingEnabled)
        {
            _status = LearningJobStatus.Collecting;
        }
        else if (_collector.TotalSamples < _minSamplesForTraining)
        {
            _status = LearningJobStatus.Collecting;
        }
        else
        {
            _status = LearningJobStatus.Idle;
        }

        PersistState();
    }

    private void LoadState()
    {
        if (!File.Exists(_statePath))
        {
            return;
        }

        try
        {
            var json = File.ReadAllText(_statePath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.TryGetProperty("trainedSamples", out var trained))
            {
                _trainedSamples = trained.GetInt32();
            }

            if (root.TryGetProperty("adapterVersion", out var version))
            {
                _adapterVersion = version.GetInt32();
            }

            if (root.TryGetProperty("lastTrainUtc", out var lastTrain) &&
                lastTrain.ValueKind == JsonValueKind.String &&
                DateTime.TryParse(lastTrain.GetString(), out var parsed))
            {
                _lastTrainUtc = parsed;
            }

            if (root.TryGetProperty("lastError", out var err) && err.ValueKind == JsonValueKind.String)
            {
                _lastError = err.GetString();
            }

            if (root.TryGetProperty("lastLearnedTopics", out var topics) &&
                topics.ValueKind == JsonValueKind.Array)
            {
                _lastLearnedTopics = topics.EnumerateArray()
                    .Select(static t => t.GetString())
                    .Where(static t => !string.IsNullOrWhiteSpace(t))
                    .Select(static t => t!)
                    .ToList();
            }

            if (root.TryGetProperty("lastTrainingMessage", out var trainingMessage) &&
                trainingMessage.ValueKind == JsonValueKind.String)
            {
                _lastTrainingMessage = trainingMessage.GetString();
            }

            if (root.TryGetProperty("status", out var status) &&
                Enum.TryParse<LearningJobStatus>(status.GetString(), out var parsedStatus))
            {
                _status = parsedStatus;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không đọc được Learning/state.json");
        }
    }

    private void PersistState()
    {
        var state = new
        {
            status = _status.ToString(),
            trainedSamples = _trainedSamples,
            adapterVersion = _adapterVersion,
            lastTrainUtc = _lastTrainUtc,
            lastError = _lastError,
            adapterPath = _adapterPath,
            lastLearnedTopics = _lastLearnedTopics,
            lastTrainingMessage = _lastTrainingMessage
        };

        File.WriteAllText(_statePath, JsonSerializer.Serialize(state, JsonOptions));
    }

    private static string ResolvePath(string path, string contentRoot)
    {
        return Path.IsPathRooted(path) ? path : Path.Combine(contentRoot, path);
    }

    private static IReadOnlyList<string> SummarizeLearnedTopics(IReadOnlyList<LearningSample> samples)
    {
        var topics = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var sample in samples.Reverse())
        {
            var topic = ExtractTopic(sample.Instruction);
            if (topic is null || !seen.Add(topic))
            {
                continue;
            }

            topics.Add(topic);
            if (topics.Count >= 5)
            {
                break;
            }
        }

        return topics;
    }

    private static string? ExtractTopic(string instruction)
    {
        var text = instruction.Trim().Replace('\n', ' ').Replace('\r', ' ');
        while (text.Contains("  ", StringComparison.Ordinal))
        {
            text = text.Replace("  ", " ", StringComparison.Ordinal);
        }

        if (text.Length < 4)
        {
            return null;
        }

        if (text.Equals("ping", StringComparison.OrdinalIgnoreCase) ||
            text.Equals("hi", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        const int maxLen = 48;
        return text.Length > maxLen ? $"{text[..maxLen].TrimEnd()}…" : text;
    }

    private static string BuildTrainingMessage(IReadOnlyList<string> topics, int sampleCount)
    {
        if (topics.Count == 0)
        {
            return $"Đã học xong từ {sampleCount} mẫu hội thoại gần đây.";
        }

        if (topics.Count == 1)
        {
            return $"đã học kiến thức mới về: {topics[0]}";
        }

        var preview = string.Join(", ", topics.Take(4));
        if (topics.Count > 4)
        {
            preview += $" và {topics.Count - 4} chủ đề khác";
        }

        return $"đã học kiến thức mới về: {preview}";
    }
}
