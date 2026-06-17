namespace AI_Model_BE.Models;

public record PredictRequest(float Size, float Bedrooms);

public record PredictResponse(
    float PredictedPrice,
    string Currency,
    string Message);

public record ChatRequest(string Message, bool Stream = false);

public record ChatResponse(string Reply, bool IsMock = false);
