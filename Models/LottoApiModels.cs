namespace AI_Model_BE.Models;

public record LottoRunResponse(
    string GameKind,
    string? DaiCode,
    DaiPredictionResult? Single,
    IReadOnlyList<DaiPredictionResult>? AllDais);
