using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// SSA (Singular Spectrum Analysis) — nhánh xu hướng
/// =====================================================================
/// Dùng cùng họ thuật toán với xổ số (ML.NET ForecastingBySsa).
/// SSA tách nhiễu / chu kỳ ngắn khỏi xu hướng — bổ sung tốt cho LightGBM
/// (LightGBM giỏi pattern cục bộ; SSA giỏi hướng trend trung hạn).
/// </summary>
public sealed class CoinSsaTrendPredictor
{
    private readonly ILogger<CoinSsaTrendPredictor> _logger;

    public CoinSsaTrendPredictor(ILogger<CoinSsaTrendPredictor> logger)
    {
        _logger = logger;
    }

    public sealed class PredictResult
    {
        public double PredictedClose { get; init; }
        public string Note { get; init; } = string.Empty;
        public bool Success { get; init; }
    }

    private sealed class ClosePoint
    {
        public float Close { get; set; }
    }

    private sealed class CloseForecast
    {
        public float[] Forecast { get; set; } = [];
    }

    public PredictResult PredictNextClose(IReadOnlyList<CoinCandle> candles)
    {
        try
        {
            // Lấy log-price để SSA ổn định hơn trên asset biến động mạnh
            var series = candles
                .Select(c => new ClosePoint { Close = (float)Math.Log(Math.Max(c.Close, 1e-12)) })
                .ToList();

            if (series.Count < 48)
            {
                return new PredictResult
                {
                    Success = false,
                    Note = $"Không đủ nến cho SSA ({series.Count}/48)."
                };
            }

            var ml = new MLContext(seed: 7);
            var data = ml.Data.LoadFromEnumerable(series);

            var windowSize = Math.Clamp(series.Count / 10, 5, 24);
            var seriesLength = Math.Min(series.Count, Math.Max(windowSize * 3, 48));
            var trainSize = series.Count;
            var rank = Math.Min(8, windowSize - 1);

            var pipeline = ml.Forecasting.ForecastBySsa(
                outputColumnName: nameof(CloseForecast.Forecast),
                inputColumnName: nameof(ClosePoint.Close),
                windowSize: windowSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: 1,
                rank: rank);

            var model = pipeline.Fit(data);
            var engine = model.CreateTimeSeriesEngine<ClosePoint, CloseForecast>(ml);
            var forecast = engine.Predict();

            if (forecast.Forecast is null || forecast.Forecast.Length == 0)
            {
                return new PredictResult { Success = false, Note = "SSA không trả forecast." };
            }

            var logPred = forecast.Forecast[0];
            var predictedClose = Math.Exp(logPred);
            var last = candles[^1].Close;

            // Chặn dự báo SSA lệch quá xa (±20%)
            if (predictedClose > last * 1.2)
            {
                predictedClose = last * 1.2;
            }
            else if (predictedClose < last * 0.8)
            {
                predictedClose = last * 0.8;
            }

            _logger.LogInformation(
                "SSA coin: last={Last:F4}, pred={Pred:F4}, window={W}, rank={R}",
                last,
                predictedClose,
                windowSize,
                rank);

            return new PredictResult
            {
                Success = true,
                PredictedClose = predictedClose,
                Note = $"SSA trên log-price (window={windowSize}, rank={rank})."
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "SSA coin thất bại");
            return new PredictResult
            {
                Success = false,
                Note = "SSA lỗi: " + ex.Message
            };
        }
    }
}
