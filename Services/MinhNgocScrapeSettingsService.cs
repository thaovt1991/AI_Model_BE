using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Cài đặt cào Minh Ngọc — mặc định từ appsettings, ghi đè qua Data/MinhNgoc/scrape-settings.json hoặc API.
/// </summary>
public sealed class MinhNgocScrapeSettingsService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly ILogger<MinhNgocScrapeSettingsService> _logger;
    private readonly string _settingsPath;
    private readonly object _lock = new();
    private MinhNgocScrapeSettingsDto _current;

    public MinhNgocScrapeSettingsService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        ILogger<MinhNgocScrapeSettingsService> logger)
    {
        _logger = logger;
        var dir = Path.Combine(env.ContentRootPath, "Data", "MinhNgoc");
        Directory.CreateDirectory(dir);
        _settingsPath = Path.Combine(dir, "scrape-settings.json");

        _current = ReadDefaults(configuration);
        LoadFromDisk();
    }

    public MinhNgocScrapeSettingsDto Get()
    {
        lock (_lock)
        {
            return _current;
        }
    }

    public MinhNgocScrapeSettingsResponse GetResponse() =>
        ToResponse(Get());

    public MinhNgocScrapeSettingsResponse Update(UpdateMinhNgocScrapeSettingsRequest request)
    {
        lock (_lock)
        {
            _current = Normalize(new MinhNgocScrapeSettingsDto
            {
                ScrapingEnabled = request.ScrapingEnabled ?? _current.ScrapingEnabled,
                RequestDelayMs = request.RequestDelayMs ?? _current.RequestDelayMs,
                CacheMinutes = request.CacheMinutes ?? _current.CacheMinutes,
                MaxHistoryFetchesPerRun = request.MaxHistoryFetchesPerRun ?? _current.MaxHistoryFetchesPerRun,
                MaxLatestProbeAttempts = request.MaxLatestProbeAttempts ?? _current.MaxLatestProbeAttempts,
                PreferLocalDataFirst = request.PreferLocalDataFirst ?? _current.PreferLocalDataFirst,
            });

            SaveToDisk();
            return ToResponse(_current);
        }
    }

    private static MinhNgocScrapeSettingsDto ReadDefaults(IConfiguration configuration)
    {
        var section = configuration.GetSection("MinhNgoc");
        return Normalize(new MinhNgocScrapeSettingsDto
        {
            ScrapingEnabled = section.GetValue("ScrapingEnabled", true),
            RequestDelayMs = section.GetValue("RequestDelayMs", 800),
            CacheMinutes = section.GetValue("CacheMinutes", 30),
            MaxHistoryFetchesPerRun = section.GetValue("MaxHistoryFetchesPerRun", 50),
            MaxLatestProbeAttempts = section.GetValue("MaxLatestProbeAttempts", 3),
            PreferLocalDataFirst = section.GetValue("PreferLocalDataFirst", true),
        });
    }

    private void LoadFromDisk()
    {
        if (!File.Exists(_settingsPath))
        {
            return;
        }

        try
        {
            var json = File.ReadAllText(_settingsPath);
            var loaded = JsonSerializer.Deserialize<MinhNgocScrapeSettingsDto>(json, JsonOptions);
            if (loaded is not null)
            {
                lock (_lock)
                {
                    _current = Normalize(loaded);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không đọc được {Path}, dùng mặc định appsettings", _settingsPath);
        }
    }

    private void SaveToDisk()
    {
        try
        {
            var json = JsonSerializer.Serialize(_current, JsonOptions);
            File.WriteAllText(_settingsPath, json);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không lưu được cài đặt Minh Ngọc");
        }
    }

    private static MinhNgocScrapeSettingsDto Normalize(MinhNgocScrapeSettingsDto raw) =>
        new()
        {
            ScrapingEnabled = raw.ScrapingEnabled,
            RequestDelayMs = Math.Clamp(raw.RequestDelayMs, 200, 10_000),
            CacheMinutes = Math.Clamp(raw.CacheMinutes, 1, 24 * 60),
            MaxHistoryFetchesPerRun = Math.Clamp(raw.MaxHistoryFetchesPerRun, 0, 100),
            MaxLatestProbeAttempts = Math.Clamp(raw.MaxLatestProbeAttempts, 1, 14),
            PreferLocalDataFirst = raw.PreferLocalDataFirst,
        };

    private static MinhNgocScrapeSettingsResponse ToResponse(MinhNgocScrapeSettingsDto dto) =>
        new(
            dto.ScrapingEnabled,
            dto.RequestDelayMs,
            dto.CacheMinutes,
            dto.MaxHistoryFetchesPerRun,
            dto.MaxLatestProbeAttempts,
            dto.PreferLocalDataFirst);
}
