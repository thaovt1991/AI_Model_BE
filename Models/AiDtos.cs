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
    string? ProfileId = null);

public record ChatResponse(string Reply, bool IsMock = false, int HistoryTurns = 0);
