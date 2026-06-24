using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// BƯỚC 1 trong pipeline "học ngầm": THU THẬP DỮ LIỆU từ chat.
///
/// === AI là gì trong project này? ===
/// - File .gguf là "bộ não" đã được huấn luyện sẵn (giống sách giáo khoa khổng lồ).
/// - Khi user chat, ta KHÔNG sửa bộ não ngay lập tức — ta chỉ ghi lại câu hỏi + câu trả lời.
/// - Sau này (khi bật Learning:Enabled), những cặp hỏi/đáp này dùng để "dạy thêm" model
///   bằng kỹ thuật LoRA (chỉ sửa một phần nhỏ trọng số, không train lại cả file 5GB).
///
/// === Luồng dữ liệu ===
/// User chat → LlamaChatService.RememberTurn → EnqueueSample (hàng đợi RAM)
/// → Background worker gọi FlushPendingSamples → ghi file Learning/dataset.jsonl
///
/// File dataset.jsonl: mỗi dòng là 1 mẫu JSON (instruction = câu hỏi, output = câu trả lời).
/// </summary>
public sealed class LearningDataCollectorService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private readonly ILogger<LearningDataCollectorService> _logger;

    // Thư mục gốc chứa mọi thứ liên quan học model: dataset, adapter, state...
    private readonly string _learningDir;

    // File chính lưu dữ liệu train — định dạng JSONL (JSON Lines): 1 dòng = 1 mẫu
    private readonly string _datasetPath;

    // Danh sách "dấu vân tay" các mẫu đã lưu — tránh ghi trùng cùng câu hỏi/trả lời
    private readonly string _seenHashesPath;

    private readonly LearningSettingsService _settings;

    // Lọc mẫu quá ngắn / vô nghĩa (ví dụ user chỉ gõ "hi")
    private readonly int _minInstructionLength;
    private readonly int _minOutputLength;

    // Khóa để nhiều thread không ghi file cùng lúc (chat + background worker)
    private readonly object _fileLock = new();

    // Bộ nhớ các hash đã thấy — load từ seen_hashes.txt khi khởi động
    private readonly HashSet<string> _seenHashes = new(StringComparer.Ordinal);

    // Hàng đợi tạm trong RAM — ghi xuống đĩa theo lô (flush) để không chậm mỗi lần chat
    private readonly ConcurrentQueue<LearningSample> _pending = new();

    private int _totalSamples;      // Tổng số mẫu đã có trong dataset.jsonl
    private int _pendingSamples;    // Số mẫu đang chờ flush
    private DateTime? _lastSampleUtc;
    private DateTime? _lastChatUtc; // Dùng để biết user vừa chat — tránh train khi đang bận

    public LearningDataCollectorService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        LearningSettingsService settings,
        ILogger<LearningDataCollectorService> logger)
    {
        _logger = logger;
        _settings = settings;
        _learningDir = Path.Combine(
            env.ContentRootPath,
            configuration.GetValue("Learning:Directory", "Learning") ?? "Learning");
        _datasetPath = Path.Combine(_learningDir, "dataset.jsonl");
        _seenHashesPath = Path.Combine(_learningDir, "seen_hashes.txt");
        _minInstructionLength = configuration.GetValue("Learning:MinInstructionLength", 4);
        _minOutputLength = configuration.GetValue("Learning:MinOutputLength", 8);

        Directory.CreateDirectory(_learningDir);
        Directory.CreateDirectory(Path.Combine(_learningDir, "Adapters"));
        LoadExistingState();
    }

    public bool CollectDataEnabled => _settings.CollectDataEnabled;
    public int TotalSamples => _totalSamples;
    public int PendingSamples => _pendingSamples;
    public DateTime? LastSampleUtc => _lastSampleUtc;
    public DateTime? LastChatUtc => _lastChatUtc;
    public string DatasetPath => _datasetPath;
    public string LearningDirectory => _learningDir;

    /// <summary>Ghi nhận "vừa có người chat" — background train sẽ chờ máy rảnh.</summary>
    public void NotifyChatActivity() => _lastChatUtc = DateTime.UtcNow;

    /// <summary>
    /// Đưa 1 cặp hỏi/đáp vào hàng đợi học ngầm.
    /// Gọi từ LlamaChatService sau mỗi lượt chat thành công.
    /// </summary>
    public void EnqueueSample(string? profileId, string instruction, string output)
    {
        NotifyChatActivity();

        if (!_settings.CollectDataEnabled)
        {
            return;
        }

        instruction = instruction.Trim();
        output = output.Trim();

        // Bỏ qua tin nhắn mock, quá ngắn, hoặc câu warmup ("ping")
        if (!IsTrainableSample(instruction, output))
        {
            return;
        }

        // Hash SHA256 = "dấu vân tay" nội dung — cùng câu hỏi + trả lời thì không lưu 2 lần
        var hash = ComputeHash(instruction, output);
        lock (_fileLock)
        {
            if (!_seenHashes.Add(hash))
            {
                return;
            }
        }

        var sample = new LearningSample(instruction, output, profileId, DateTime.UtcNow);
        _pending.Enqueue(sample);
        Interlocked.Increment(ref _pendingSamples);
    }

    /// <summary>
    /// Ghi các mẫu trong hàng đợi RAM xuống file dataset.jsonl.
    /// Background worker gọi định kỳ (mỗi 15 phút) — user không cảm nhận lag khi chat.
    /// </summary>
    public int FlushPendingSamples()
    {
        if (_pending.IsEmpty)
        {
            return 0;
        }

        var flushed = 0;
        lock (_fileLock)
        {
            // FileMode.Append = thêm vào cuối file, không xóa dữ liệu cũ
            using var stream = new FileStream(_datasetPath, FileMode.Append, FileAccess.Write, FileShare.Read);
            using var writer = new StreamWriter(stream, Encoding.UTF8);

            while (_pending.TryDequeue(out var sample))
            {
                writer.WriteLine(JsonSerializer.Serialize(sample));
                _seenHashes.Add(ComputeHash(sample.Instruction, sample.Output));
                flushed++;
                _totalSamples++;
                _lastSampleUtc = sample.CreatedAt;
            }

            PersistSeenHashes();
        }

        Interlocked.Add(ref _pendingSamples, -flushed);
        if (flushed > 0)
        {
            _logger.LogDebug("Learning: đã ghi {Count} mẫu vào dataset", flushed);
        }

        return flushed;
    }

    /// <summary>Đọc N mẫu gần nhất — dùng khi export sang định dạng train (finetune.txt).</summary>
    public IReadOnlyList<LearningSample> ReadRecentSamples(int maxSamples)
    {
        if (!File.Exists(_datasetPath))
        {
            return [];
        }

        var lines = File.ReadAllLines(_datasetPath)
            .Where(static line => !string.IsNullOrWhiteSpace(line))
            .TakeLast(maxSamples)
            .ToList();

        var samples = new List<LearningSample>(lines.Count);
        foreach (var line in lines)
        {
            try
            {
                var sample = JsonSerializer.Deserialize<LearningSample>(line, JsonOptions);
                if (sample is not null)
                {
                    samples.Add(sample);
                }
            }
            catch
            {
                // Dòng JSON hỏng — bỏ qua, không làm crash app
            }
        }

        return samples;
    }

    public void MarkSamplesTrained(int count)
    {
        Interlocked.Add(ref _pendingSamples, -count);
        if (_pendingSamples < 0)
        {
            Interlocked.Exchange(ref _pendingSamples, 0);
        }
    }

    /// <summary>Kiểm tra mẫu có đủ chất lượng để đưa vào dataset train không.</summary>
    private bool IsTrainableSample(string instruction, string output)
    {
        if (instruction.Length < _minInstructionLength || output.Length < _minOutputLength)
        {
            return false;
        }

        // Phản hồi mock = chưa có model thật → không học được gì hữu ích
        if (output.StartsWith("[Chế độ mock", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        // Câu warmup khi Backend khởi động — không phải dữ liệu user
        if (instruction.Equals("ping", StringComparison.OrdinalIgnoreCase) ||
            instruction.Equals("hi", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return true;
    }

    /// <summary>Khi restart Backend: đếm lại số mẫu và load hash đã lưu.</summary>
    private void LoadExistingState()
    {
        if (File.Exists(_seenHashesPath))
        {
            foreach (var line in File.ReadAllLines(_seenHashesPath))
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    _seenHashes.Add(line.Trim());
                }
            }
        }

        if (!File.Exists(_datasetPath))
        {
            return;
        }

        var lineCount = File.ReadLines(_datasetPath).Count(static line => !string.IsNullOrWhiteSpace(line));
        _totalSamples = lineCount;
    }

    private void PersistSeenHashes()
    {
        File.WriteAllLines(_seenHashesPath, _seenHashes);
    }

    private static string ComputeHash(string instruction, string output)
    {
        var bytes = Encoding.UTF8.GetBytes($"{instruction}\n---\n{output}");
        return Convert.ToHexString(SHA256.HashData(bytes));
    }
}
