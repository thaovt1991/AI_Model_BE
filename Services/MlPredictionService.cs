using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AI_Model_BE.Services;

/// <summary>
/// Huấn luyện và lưu model ML.NET giả lập khi khởi động,
/// để API /api/ai/predict chạy được ngay không cần file model có sẵn.
/// </summary>
public sealed class MlPredictionService
{
    private readonly ILogger<MlPredictionService> _logger;
    private readonly string _modelPath;
    private readonly object _lock = new();
    private PredictionEngine<HouseData, HousePrediction>? _predictionEngine;

    public MlPredictionService(IWebHostEnvironment env, ILogger<MlPredictionService> logger)
    {
        _logger = logger;
        var modelsDir = Path.Combine(env.ContentRootPath, "Models");
        Directory.CreateDirectory(modelsDir);
        _modelPath = Path.Combine(modelsDir, "house-price-model.zip");
    }

    public void EnsureModelReady()
    {
        lock (_lock)
        {
            if (_predictionEngine is not null)
            {
                return;
            }

            if (!File.Exists(_modelPath))
            {
                _logger.LogInformation("Chưa có file model — đang huấn luyện model giả lập ML.NET...");
                TrainAndSaveMockModel();
            }

            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(_modelPath, out _);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
            _logger.LogInformation("Model ML.NET đã sẵn sàng tại {Path}", _modelPath);
        }
    }

    public PredictResponse Predict(PredictRequest request)
    {
        EnsureModelReady();

        var input = new HouseData
        {
            Size = request.Size,
            Bedrooms = request.Bedrooms,
            Price = 0
        };

        var prediction = _predictionEngine!.Predict(input);
        var price = Math.Max(0, prediction.PredictedPrice);

        return new PredictResponse(
            PredictedPrice: (float)Math.Round(price, 2),
            Currency: "USD",
            Message: $"Dự đoán giá nhà {request.Size}m², {request.Bedrooms} phòng ngủ.");
    }

    private void TrainAndSaveMockModel()
    {
        var mlContext = new MLContext(seed: 42);

        // Dữ liệu mẫu: giá ≈ 1200 * diện tích + 25000 * số phòng
        var samples = new List<HouseData>
        {
            new() { Size = 50, Bedrooms = 1, Price = 85000 },
            new() { Size = 65, Bedrooms = 2, Price = 128000 },
            new() { Size = 80, Bedrooms = 2, Price = 146000 },
            new() { Size = 95, Bedrooms = 3, Price = 189000 },
            new() { Size = 110, Bedrooms = 3, Price = 207000 },
            new() { Size = 130, Bedrooms = 4, Price = 256000 },
            new() { Size = 150, Bedrooms = 4, Price = 280000 },
            new() { Size = 180, Bedrooms = 5, Price = 341000 },
        };

        var dataView = mlContext.Data.LoadFromEnumerable(samples);

        var pipeline = mlContext.Transforms
            .Concatenate("Features", nameof(HouseData.Size), nameof(HouseData.Bedrooms))
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: nameof(HouseData.Price),
                featureColumnName: "Features"));

        var model = pipeline.Fit(dataView);
        mlContext.Model.Save(model, dataView.Schema, _modelPath);
    }
}
