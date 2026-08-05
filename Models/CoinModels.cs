namespace AI_Model_BE.Models;

/// <summary>
/// Một nến OHLCV (Open-High-Low-Close-Volume) — đơn vị thời gian do Interval quyết định.
/// </summary>
public sealed class CoinCandle
{
    public DateTime OpenTimeUtc { get; init; }
    public double Open { get; init; }
    public double High { get; init; }
    public double Low { get; init; }
    public double Close { get; init; }
    public double Volume { get; init; }
}

public record CoinInfoDto(
    string Symbol,
    string Name,
    string Quote,
    string BinancePair,
    string Description);

public record CoinRunRequest(
    string Symbol,
    /// <summary>1h | 4h | 1d — khung thời gian nến.</summary>
    string Interval = "1h",
    /// <summary>Số nến lịch sử lấy từ Binance (tối đa ~1000).</summary>
    int Lookback = 500);

public record CoinForecastResponse(
    string Symbol,
    string Name,
    string Interval,
    string Algorithm,
    DateTime LastCandleUtc,
    double LastClose,
    double PredictedClose,
    double PredictedChangePct,
    string Direction,
    double Confidence,
    double Support,
    double Resistance,
    double Rsi14,
    double MacdHistogram,
    int CandlesUsed,
    string Message,
    IReadOnlyList<CoinCandleDto>? RecentCandles = null,
    CoinModelBreakdown? Breakdown = null);

public record CoinCandleDto(
    DateTime OpenTimeUtc,
    double Open,
    double High,
    double Low,
    double Close,
    double Volume);

/// <summary>Chi tiết từng nhánh ensemble để UI giải thích.</summary>
public record CoinModelBreakdown(
    double LightGbmPredictedClose,
    double SsaPredictedClose,
    double LightGbmWeight,
    double SsaWeight,
    string LightGbmNote,
    string SsaNote);
