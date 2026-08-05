namespace AI_Model_BE.Models;

public record PredictRequest(float Size, float Bedrooms);

public record PredictResponse(
    float PredictedPrice,
    string Currency,
    string Message);

public record ChatRequest(
    string Message,
    bool Stream = false,
    Guid[]? DocumentIds = null,
    string? ConversationId = null,
    string? ProfileId = null,
    // null = tự động (AutoDetect); true = luôn research mạng; false = tắt research.
    bool? EnableWebSearch = null,
    // true = Deep Research (nhiều truy vấn phụ + nhiều nguồn + citation).
    bool DeepResearch = false);

public record ChatResponse(
    string Reply,
    bool IsMock = false,
    int HistoryTurns = 0,
    bool UsedWebResearch = false,
    int WebSourceCount = 0,
    bool UsedDeepResearch = false,
    IReadOnlyList<CitationSource>? Citations = null);
