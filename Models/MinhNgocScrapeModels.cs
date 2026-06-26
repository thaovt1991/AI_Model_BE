namespace AI_Model_BE.Models;

/// <summary>Cài đặt cào dữ liệu minhngoc.net.vn — giảm tải lên server đích.</summary>
public sealed class MinhNgocScrapeSettingsDto
{
    public bool ScrapingEnabled { get; init; } = true;

    /// <summary>Khoảng cách tối thiểu giữa hai HTTP request (ms).</summary>
    public int RequestDelayMs { get; init; } = 800;

    /// <summary>Cache HTML theo URL (phút).</summary>
    public int CacheMinutes { get; init; } = 30;

    /// <summary>Số request theo ngày tối đa mỗi lần gom lịch sử (ngoài trang list).</summary>
    public int MaxHistoryFetchesPerRun { get; init; } = 50;

    /// <summary>Số lần thử URL khi lấy kết quả mới nhất (XSMB theo ngày).</summary>
    public int MaxLatestProbeAttempts { get; init; } = 3;

    /// <summary>Ưu tiên file local khi gom lịch sử lúc dự đoán (không áp dụng /lotto/latest).</summary>
    public bool PreferLocalDataFirst { get; init; } = true;
}

public record UpdateMinhNgocScrapeSettingsRequest(
    bool? ScrapingEnabled = null,
    int? RequestDelayMs = null,
    int? CacheMinutes = null,
    int? MaxHistoryFetchesPerRun = null,
    int? MaxLatestProbeAttempts = null,
    bool? PreferLocalDataFirst = null);

public record MinhNgocScrapeSettingsResponse(
    bool ScrapingEnabled,
    int RequestDelayMs,
    int CacheMinutes,
    int MaxHistoryFetchesPerRun,
    int MaxLatestProbeAttempts,
    bool PreferLocalDataFirst);
