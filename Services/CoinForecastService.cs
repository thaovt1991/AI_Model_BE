using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// COIN FORECAST ORCHESTRATOR — Ensemble LightGBM + SSA
/// =====================================================================
/// Pipeline:
///  1) Tải nến Binance (CoinMarketDataService)
///  2) LightGBM dự đoán giá kỳ tới từ technical indicators  ← trọng số chính (~0.7)
///  3) SSA dự đoán xu hướng log-price                       ← trọng số phụ (~0.3)
///  4) Blend → hướng Tăng/Giảm/Đi ngang + confidence + hỗ trợ/kháng cự Bollinger
///
/// Đây là kiến trúc “competition-style”: GBDT tabular + classical time-series,
/// thường mạnh hơn dùng một mình LSTM/SSA trên chuỗi ngắn khi không có GPU.
/// </summary>
public sealed class CoinForecastService
{
    private readonly CoinMarketDataService _market;
    private readonly CoinLightGbmPredictor _lightGbm;
    private readonly CoinSsaTrendPredictor _ssa;
    private readonly IConfiguration _configuration;
    private readonly ILogger<CoinForecastService> _logger;

    public CoinForecastService(
        CoinMarketDataService market,
        CoinLightGbmPredictor lightGbm,
        CoinSsaTrendPredictor ssa,
        IConfiguration configuration,
        ILogger<CoinForecastService> logger)
    {
        _market = market;
        _lightGbm = lightGbm;
        _ssa = ssa;
        _configuration = configuration;
        _logger = logger;
    }

    public Task<IReadOnlyList<CoinInfoDto>> GetCatalogAsync(CancellationToken cancellationToken = default) =>
        _market.GetCatalogAsync(cancellationToken);

    public async Task<CoinForecastResponse> RunAsync(
        CoinRunRequest request,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(request.Symbol))
        {
            throw new ArgumentException("Symbol bắt buộc (BTC, ETH, ...).");
        }

        var info = await _market.ResolveAsync(request.Symbol, cancellationToken);
        var interval = _market.NormalizeInterval(request.Interval);
        var lookback = request.Lookback > 0
            ? request.Lookback
            : _configuration.GetValue("Coin:DefaultLookback", 500);

        var candles = await _market.GetKlinesAsync(
            info.BinancePair, interval, lookback, cancellationToken);

        var last = candles[^1];
        var latestFeatures = CoinTechnicalFeatureBuilder.BuildLatest(candles);

        // --- Nhánh 1: LightGBM ---
        var lgbm = _lightGbm.PredictNextClose(candles);

        // --- Nhánh 2: SSA ---
        var ssa = _ssa.PredictNextClose(candles);

        var wLgbm = _configuration.GetValue("Coin:LightGbmWeight", 0.70);
        var wSsa = _configuration.GetValue("Coin:SsaWeight", 0.30);

        double predicted;
        string algorithm;
        double usedLgbmW = 0, usedSsaW = 0;

        if (lgbm.Success && ssa.Success)
        {
            // Chuẩn hóa trọng số
            var sum = wLgbm + wSsa;
            usedLgbmW = wLgbm / sum;
            usedSsaW = wSsa / sum;
            predicted = usedLgbmW * lgbm.PredictedClose + usedSsaW * ssa.PredictedClose;
            algorithm = "Ensemble FastTree(GBDT) + SSA";
        }
        else if (lgbm.Success)
        {
            predicted = lgbm.PredictedClose;
            usedLgbmW = 1;
            algorithm = "FastTree GBDT (technical indicators)";
        }
        else if (ssa.Success)
        {
            predicted = ssa.PredictedClose;
            usedSsaW = 1;
            algorithm = "SSA (trend)";
        }
        else
        {
            throw new InvalidOperationException(
                $"Không chạy được mô hình. LightGBM: {lgbm.Note}; SSA: {ssa.Note}");
        }

        var changePct = (predicted - last.Close) / last.Close * 100.0;
        var flatThreshold = _configuration.GetValue("Coin:FlatThresholdPct", 0.15);
        var direction = changePct > flatThreshold
            ? "Tăng"
            : changePct < -flatThreshold
                ? "Giảm"
                : "Đi ngang";

        // Confidence: đồng thuận 2 model + độ mạnh RSI/MACD
        var confidence = EstimateConfidence(
            lgbm.Success, ssa.Success,
            lgbm.Success ? lgbm.PredictedClose : predicted,
            ssa.Success ? ssa.PredictedClose : predicted,
            last.Close,
            latestFeatures.Rsi14,
            latestFeatures.MacdHist);

        // Hỗ trợ / kháng cự gần đúng từ Bollinger position & range
        var support = last.Close * (1 - Math.Max(0.005, latestFeatures.BbWidth * 0.5));
        var resistance = last.Close * (1 + Math.Max(0.005, latestFeatures.BbWidth * 0.5));
        if (latestFeatures.BbPosition is > 0 and < 1)
        {
            // Ước lượng band từ vị trí giá trong dải
            var half = last.Close * Math.Max(0.008, latestFeatures.BbWidth / 2);
            support = last.Close - half;
            resistance = last.Close + half;
        }

        var recent = candles
            .TakeLast(96)
            .Select(c => new CoinCandleDto(c.OpenTimeUtc, c.Open, c.High, c.Low, c.Close, c.Volume))
            .ToList();

        var message =
            $"Dự đoán nến {interval} kế tiếp cho {info.Symbol}/USDT bằng {algorithm}. " +
            "Đây là mô hình thống kê ngắn hạn — không phải lời khuyên đầu tư.";

        _logger.LogInformation(
            "Coin forecast {Symbol} {Interval}: last={Last}, pred={Pred}, dir={Dir}, conf={Conf:P0}",
            info.Symbol,
            interval,
            last.Close,
            predicted,
            direction,
            confidence);

        return new CoinForecastResponse(
            Symbol: info.Symbol,
            Name: info.Name,
            Interval: interval,
            Algorithm: algorithm,
            LastCandleUtc: last.OpenTimeUtc,
            LastClose: RoundPrice(last.Close),
            PredictedClose: RoundPrice(predicted),
            PredictedChangePct: Math.Round(changePct, 3),
            Direction: direction,
            Confidence: Math.Round(confidence, 3),
            Support: RoundPrice(support),
            Resistance: RoundPrice(resistance),
            Rsi14: Math.Round(latestFeatures.Rsi14, 2),
            MacdHistogram: Math.Round(latestFeatures.MacdHist, 6),
            CandlesUsed: candles.Count,
            Message: message,
            RecentCandles: recent,
            Breakdown: new CoinModelBreakdown(
                LightGbmPredictedClose: RoundPrice(lgbm.Success ? lgbm.PredictedClose : 0),
                SsaPredictedClose: RoundPrice(ssa.Success ? ssa.PredictedClose : 0),
                LightGbmWeight: Math.Round(usedLgbmW, 2),
                SsaWeight: Math.Round(usedSsaW, 2),
                LightGbmNote: lgbm.Note,
                SsaNote: ssa.Note));
    }

    private static double EstimateConfidence(
        bool hasLgbm,
        bool hasSsa,
        double lgbmPrice,
        double ssaPrice,
        double lastClose,
        float rsi,
        float macdHist)
    {
        double score = 0.45; // baseline

        if (hasLgbm && hasSsa)
        {
            var agreeDir =
                Math.Sign(lgbmPrice - lastClose) == Math.Sign(ssaPrice - lastClose);
            score += agreeDir ? 0.25 : 0.05;

            var relGap = Math.Abs(lgbmPrice - ssaPrice) / lastClose;
            score += relGap < 0.01 ? 0.12 : relGap < 0.03 ? 0.06 : 0;
        }
        else
        {
            score += 0.08; // một model vẫn chạy
        }

        // RSI cực đoan → tín hiệu rõ hơn (nhưng cũng rủi ro đảo chiều)
        if (rsi is >= 65 or <= 35)
        {
            score += 0.06;
        }

        if (Math.Abs(macdHist) > 0)
        {
            score += 0.04;
        }

        return Math.Clamp(score, 0.35, 0.92);
    }

    private static double RoundPrice(double price) =>
        price >= 1000 ? Math.Round(price, 2) :
        price >= 1 ? Math.Round(price, 4) :
        Math.Round(price, 6);
}
