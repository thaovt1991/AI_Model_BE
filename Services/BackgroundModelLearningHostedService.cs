using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// BƯỚC "đồng hồ báo thức" của học ngầm — chạy nền suốt đời Backend.
///
/// BackgroundService = worker ASP.NET Core tự chạy khi app start, không cần user gọi API.
///
/// Mỗi N phút (Learning:CheckIntervalMinutes, mặc định 15):
///   1. Flush dataset từ RAM → file (LearningDataCollectorService)
///   2. Nếu Learning:Enabled = false → DỪNG (chỉ thu thập, không train)
///   3. Nếu Enabled = true → thử train LoRA khi đủ mẫu + máy rảnh
///
/// User chat bình thường — worker chạy "âm thầm" phía sau.
/// </summary>
public sealed class BackgroundModelLearningHostedService : BackgroundService
{
    private readonly ILogger<BackgroundModelLearningHostedService> _logger;
    private readonly IConfiguration _configuration;
    private readonly LearningDataCollectorService _collector;
    private readonly ModelLearningService _learning;

    public BackgroundModelLearningHostedService(
        ILogger<BackgroundModelLearningHostedService> logger,
        IConfiguration configuration,
        LearningDataCollectorService collector,
        ModelLearningService learning)
    {
        _logger = logger;
        _configuration = configuration;
        _collector = collector;
        _learning = learning;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var intervalMinutes = _configuration.GetValue("Learning:CheckIntervalMinutes", 15);
        var collectOnly = !_learning.TrainingEnabled;

        if (collectOnly)
        {
            _logger.LogInformation(
                "Learning nền: chỉ thu thập dữ liệu ngầm (Learning:Enabled = false). Kiểm tra mỗi {Minutes} phút.",
                intervalMinutes);
        }
        else
        {
            _logger.LogInformation(
                "Learning nền: thu thập + train LoRA khi đủ mẫu và máy rảnh. Kiểm tra mỗi {Minutes} phút.",
                intervalMinutes);
        }

        // PeriodicTimer = hẹn giờ lặp lại — nhẹ hơn Thread.Sleep trong vòng while
        using var timer = new PeriodicTimer(TimeSpan.FromMinutes(intervalMinutes));

        while (await timer.WaitForNextTickAsync(stoppingToken))
        {
            try
            {
                // Luôn flush — kể cả khi chưa bật train, để dataset.jsonl được cập nhật
                var flushed = _collector.FlushPendingSamples();
                if (flushed > 0)
                {
                    _logger.LogDebug("Learning: flush {Count} mẫu — tổng {Total}", flushed, _collector.TotalSamples);
                }

                // Học ngầm giai đoạn 1: chỉ gom dữ liệu, không train model
                if (!_learning.TrainingEnabled)
                {
                    continue;
                }

                // Học ngầm giai đoạn 2: thử train (có thể trả về false nếu chưa đủ điều kiện)
                var result = await _learning.TryTrainInBackgroundAsync(stoppingToken);
                if (result.Started)
                {
                    _logger.LogInformation("Learning nền: {Message}", result.Message);
                }
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break; // Backend đang tắt — thoát worker
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Learning nền: lỗi vòng lặp");
            }
        }
    }
}
