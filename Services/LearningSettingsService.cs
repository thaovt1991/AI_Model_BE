using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Cài đặt học model — đọc/ghi Learning/settings.json, có thể đổi từ UI mà không sửa appsettings.
/// Giá trị appsettings chỉ là mặc định lần đầu (chưa có file settings).
/// </summary>
public sealed class LearningSettingsService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly ILogger<LearningSettingsService> _logger;
    private readonly string _settingsPath;
    private readonly object _lock = new();
    private bool _enabled;
    private bool _collectData;

    public LearningSettingsService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        ILogger<LearningSettingsService> logger)
    {
        _logger = logger;
        var learningDir = Path.Combine(
            env.ContentRootPath,
            configuration.GetValue("Learning:Directory", "Learning") ?? "Learning");
        Directory.CreateDirectory(learningDir);
        _settingsPath = Path.Combine(learningDir, "settings.json");

        _enabled = configuration.GetValue("Learning:Enabled", false);
        _collectData = configuration.GetValue("Learning:CollectData", true);
        LoadFromDisk();
    }

    public bool TrainingEnabled
    {
        get
        {
            lock (_lock)
            {
                return _enabled;
            }
        }
    }

    public bool CollectDataEnabled
    {
        get
        {
            lock (_lock)
            {
                return _collectData;
            }
        }
    }

    public LearningSettingsDto Get()
    {
        lock (_lock)
        {
            return new LearningSettingsDto(_enabled, _collectData);
        }
    }

    public LearningSettingsDto Update(bool? enabled, bool? collectData)
    {
        lock (_lock)
        {
            if (enabled.HasValue)
            {
                _enabled = enabled.Value;
            }

            if (collectData.HasValue)
            {
                _collectData = collectData.Value;
            }

            SaveToDisk();
            _logger.LogInformation(
                "Learning settings cập nhật — Enabled={Enabled}, CollectData={CollectData}",
                _enabled,
                _collectData);

            return new LearningSettingsDto(_enabled, _collectData);
        }
    }

    private void LoadFromDisk()
    {
        if (!File.Exists(_settingsPath))
        {
            SaveToDisk();
            return;
        }

        try
        {
            var json = File.ReadAllText(_settingsPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("enabled", out var enabled))
            {
                _enabled = enabled.GetBoolean();
            }

            if (root.TryGetProperty("collectData", out var collect))
            {
                _collectData = collect.GetBoolean();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không đọc được {Path}, dùng giá trị appsettings.", _settingsPath);
        }
    }

    private void SaveToDisk()
    {
        var payload = new { enabled = _enabled, collectData = _collectData, updatedAt = DateTime.UtcNow };
        File.WriteAllText(_settingsPath, JsonSerializer.Serialize(payload, JsonOptions));
    }
}
