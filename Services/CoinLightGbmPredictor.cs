using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// GRADIENT BOOSTING (FastTree / họ LightGBM–XGBoost) — thuật toán chính
/// =====================================================================
/// Vì sao chọn?
/// - Gradient Boosted Decision Trees (GBDT) thường đứng top Kaggle cho dữ liệu
///   bảng (tabular) + dự đoán ngắn hạn crypto khi kết hợp technical indicators.
/// - ML.NET cung cấp FastTree (MART) — cùng họ với LightGBM/XGBoost, chạy tốt trên CPU.
/// - Phù hợp vài trăm–vài nghìn nến; ít overfit hơn DNN khi thiếu GPU/data lớn.
///
/// Mục tiêu học: NextLogReturn = ln(close[t+1]/close[t])
/// Dự đoán giá: close_hat = close * exp(predicted_log_return)
/// </summary>
public sealed class CoinLightGbmPredictor
{
    private readonly ILogger<CoinLightGbmPredictor> _logger;

    public CoinLightGbmPredictor(ILogger<CoinLightGbmPredictor> logger)
    {
        _logger = logger;
    }

    public sealed class PredictResult
    {
        public double PredictedClose { get; init; }
        public double PredictedLogReturn { get; init; }
        public string Note { get; init; } = string.Empty;
        public bool Success { get; init; }
    }

    public PredictResult PredictNextClose(IReadOnlyList<CoinCandle> candles)
    {
        try
        {
            var labeled = CoinTechnicalFeatureBuilder.BuildLabeled(candles);
            if (labeled.Count < 60)
            {
                return new PredictResult
                {
                    Success = false,
                    Note = $"Không đủ mẫu train GBDT ({labeled.Count}/60)."
                };
            }

            var latest = CoinTechnicalFeatureBuilder.BuildLatest(candles);
            var ml = new MLContext(seed: 42);

            // Train trên ~85% lịch sử; phần còn lại không dùng để tránh overfit quá mức
            var splitIndex = Math.Max(40, (int)(labeled.Count * 0.85));
            var trainRows = labeled.Take(splitIndex).ToList();
            var trainData = ml.Data.LoadFromEnumerable(trainRows);

            var featureColumns = new[]
            {
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Return1),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Return3),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Return6),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Return12),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Volatility6),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Volatility12),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Rsi14),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.Macd),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.MacdSignal),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.MacdHist),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.SmaRatio7),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.SmaRatio21),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.EmaRatio12),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.BbWidth),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.BbPosition),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.VolumeChange),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.HighLowRange),
                nameof(CoinTechnicalFeatureBuilder.FeatureRow.CloseLocation),
            };

            // FastTree = MART gradient boosting (họ LightGBM/XGBoost) — top-tier tabular
            var pipeline = ml.Transforms
                .Concatenate("Features", featureColumns)
                .Append(ml.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
                {
                    LabelColumnName = nameof(CoinTechnicalFeatureBuilder.FeatureRow.NextLogReturn),
                    FeatureColumnName = "Features",
                    NumberOfLeaves = 31,
                    NumberOfTrees = 150,
                    MinimumExampleCountPerLeaf = 10,
                    LearningRate = 0.05
                }));

            var model = pipeline.Fit(trainData);
            var engine = ml.Model.CreatePredictionEngine<
                CoinTechnicalFeatureBuilder.FeatureRow, LogReturnPrediction>(model);

            var pred = engine.Predict(latest);
            var logRet = Clamp(pred.Score, -0.15, 0.15); // giới hạn sốc ±15%/nến
            var predictedClose = latest.Close * Math.Exp(logRet);

            _logger.LogInformation(
                "GBDT coin: last={Last:F4}, logRet={Ret:F5}, pred={Pred:F4} (train={Train})",
                latest.Close,
                logRet,
                predictedClose,
                trainRows.Count);

            return new PredictResult
            {
                Success = true,
                PredictedClose = predictedClose,
                PredictedLogReturn = logRet,
                Note = $"FastTree (GBDT/LightGBM-family) trên {trainRows.Count} mẫu technical indicators."
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "GBDT coin thất bại");
            return new PredictResult
            {
                Success = false,
                Note = "GBDT lỗi: " + ex.Message
            };
        }
    }

    private static double Clamp(double v, double min, double max) =>
        Math.Max(min, Math.Min(max, v));

    private sealed class LogReturnPrediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
