namespace AI_Model_BE.Models;

/// <summary>Trạng thái job học model — dùng cho API /api/ai/learning/status.</summary>
public enum LearningJobStatus
{
    /// <summary>Learning:Enabled = false — không train, có thể vẫn thu thập dữ liệu.</summary>
    Disabled,

    /// <summary>Đang gom mẫu chat, chưa đủ số lượng để train.</summary>
    Collecting,

    /// <summary>Đủ mẫu, chờ điều kiện (máy rảnh / lệnh train).</summary>
    Idle,

    /// <summary>Muốn train nhưng user vừa chat — chờ idle.</summary>
    Queued,

    /// <summary>Đang chạy script Python train LoRA.</summary>
    Training,

    /// <summary>Train xong, adapter .gguf sẵn sàng dùng.</summary>
    Ready,

    /// <summary>Train lỗi — xem LastError.</summary>
    Failed
}

/// <summary>
/// Một mẫu dữ liệu huấn luyện = 1 cặp câu hỏi + câu trả lời từ chat thật.
/// Instruction ≈ câu user hỏi; Output ≈ câu AI trả lời.
/// </summary>
public record LearningSample(
    string Instruction,
    string Output,
    string? ProfileId,
    DateTime CreatedAt);

public record LearningState(
    LearningJobStatus Status,
    int TotalSamples,
    int PendingSamples,
    int TrainedSamples,
    DateTime? LastSampleUtc,
    DateTime? LastTrainUtc,
    DateTime? LastChatUtc,
    string? AdapterPath,
    int AdapterVersion,
    string? LastError,
    bool TrainingEnabled,
    bool CollectDataEnabled);

/// <summary>DTO trả về Frontend / Swagger khi gọi GET learning/status.</summary>
public record LearningStatusResponse(
    LearningJobStatus Status,
    int TotalSamples,
    int PendingSamples,
    int TrainedSamples,
    bool TrainingEnabled,
    bool CollectDataEnabled,
    string? AdapterPath,
    int AdapterVersion,
    string? LastError,
    DateTime? LastTrainUtc,
    IReadOnlyList<string> LearnedTopics,
    string? LastTrainingMessage);

public record LearningTrainResponse(bool Started, string Message, string? TrainingMessage = null);

/// <summary>Cài đặt học model — có thể đổi từ UI, lưu file Learning/settings.json.</summary>
public record LearningSettingsDto(bool Enabled, bool CollectData);

public record UpdateLearningSettingsRequest(bool? Enabled = null, bool? CollectData = null);

/// <summary>Settings + trạng thái tổng hợp cho tab Thiết lập trên UI.</summary>
public record LearningSettingsResponse(
    bool Enabled,
    bool CollectData,
    LearningJobStatus Status,
    int TotalSamples,
    int PendingSamples,
    string? AdapterPath,
    int AdapterVersion,
    string? LastError,
    DateTime? LastTrainUtc,
    IReadOnlyList<string> LearnedTopics,
    string? LastTrainingMessage);
