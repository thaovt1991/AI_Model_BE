// Import các model DTO (request/response) và schema dữ liệu cho ML.NET
using AI_Model_BE.Models;
using Microsoft.ML;           // Thư viện ML.NET chính
using Microsoft.ML.Data;      // Attribute như [LoadColumn], [ColumnName]

namespace AI_Model_BE.Services;

/// <summary>
/// Service xử lý dự đoán giá nhà bằng ML.NET.
/// Được gọi từ AiController endpoint POST /api/ai/predict.
/// </summary>
public sealed class MlPredictionService
{
    // Ghi log ra console khi huấn luyện / nạp model
    private readonly ILogger<MlPredictionService> _logger;

    // Đường dẫn file model đã lưu (house-price-model.zip)
    private readonly string _modelPath;

    // Khóa đồng bộ: tránh 2 thread cùng huấn luyện/nạp model một lúc
    private readonly object _lock = new();

    // Engine dùng để gọi Predict() — null cho đến khi model được nạp xong
    private PredictionEngine<HouseData, HousePrediction>? _predictionEngine;

    /// <summary>
    /// Constructor — ASP.NET Core tự inject IWebHostEnvironment và ILogger.
    /// </summary>
    public MlPredictionService(IWebHostEnvironment env, ILogger<MlPredictionService> logger)
    {
        _logger = logger;

        // Tạo đường dẫn thư mục Models/ ngay trong project Backend
        var modelsDir = Path.Combine(env.ContentRootPath, "Models");

        // Tạo thư mục nếu chưa có (không lỗi nếu đã tồn tại)
        Directory.CreateDirectory(modelsDir);

        // File model sẽ được lưu tại đây sau lần huấn luyện đầu tiên
        _modelPath = Path.Combine(modelsDir, "house-price-model.zip");
    }

    /// <summary>
    /// Đảm bảo model đã sẵn sàng trước khi dự đoán.
    /// Gọi từ Program.cs khi Backend khởi động và từ Predict().
    /// </summary>
    public void EnsureModelReady()
    {
        // lock: chỉ 1 thread được vào khối code bên trong
        lock (_lock)
        {
            // Nếu engine đã nạp rồi thì không làm gì thêm
            if (_predictionEngine is not null)
            {
                return;
            }

            // Chưa có file .zip → huấn luyện model giả lập từ dữ liệu mẫu
            if (!File.Exists(_modelPath))
            {
                _logger.LogInformation("Chưa có file model — đang huấn luyện model giả lập ML.NET...");
                TrainAndSaveMockModel();
            }

            // MLContext: "bộ não" của ML.NET, seed cố định để kết quả tái lập được
            var mlContext = new MLContext(seed: 42);

            // Đọc model từ file .zip vừa có (hoặc vừa train xong)
            var model = mlContext.Model.Load(_modelPath, out _);

            // Tạo PredictionEngine: nhận HouseData → trả HousePrediction
            _predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);

            _logger.LogInformation("Model ML.NET đã sẵn sàng tại {Path}", _modelPath);
        }
    }

    /// <summary>
    /// Dự đoán giá nhà từ diện tích và số phòng ngủ.
    /// Input: PredictRequest từ Frontend (JSON { size, bedrooms }).
    /// Output: PredictResponse gửi về Frontend.
    /// </summary>
    public PredictResponse Predict(PredictRequest request)
    {
        // Đảm bảo model đã nạp (an toàn nếu EnsureModelReady chưa chạy lúc startup)
        EnsureModelReady();

        // Map request API sang định dạng ML.NET cần
        var input = new HouseData
        {
            Size = request.Size,           // Diện tích m²
            Bedrooms = request.Bedrooms,     // Số phòng ngủ
            Price = 0                        // Khi predict không cần giá thật, để 0
        };

        // Gọi model: trả về HousePrediction với PredictedPrice
        var prediction = _predictionEngine!.Predict(input);

        // Đảm bảo giá không âm (model regression đôi khi trả số âm)
        var price = Math.Max(0, prediction.PredictedPrice);

        // Trả response JSON cho Frontend
        return new PredictResponse(
            PredictedPrice: (float)Math.Round(price, 2),
            Currency: "USD",
            Message: $"Dự đoán giá nhà {request.Size}m², {request.Bedrooms} phòng ngủ.");
    }

    /// <summary>
    /// Huấn luyện model regression đơn giản và lưu ra file .zip.
    /// Chỉ chạy lần đầu khi chưa có house-price-model.zip.
    /// </summary>
    private void TrainAndSaveMockModel()
    {
        var mlContext = new MLContext(seed: 42);

        // Dữ liệu mẫu: quy luật gần đúng giá ≈ 1200 * diện tích + 25000 * số phòng
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

        // Chuyển List C# thành IDataView — định dạng bảng dữ liệu của ML.NET
        var dataView = mlContext.Data.LoadFromEnumerable(samples);

        // Pipeline: bước 1 gộp cột Size + Bedrooms thành vector "Features"
        //           bước 2 huấn luyện SDCA regression để dự đoán cột Price
        var pipeline = mlContext.Transforms
            .Concatenate("Features", nameof(HouseData.Size), nameof(HouseData.Bedrooms))
            .Append(mlContext.Regression.Trainers.Sdca(
                labelColumnName: nameof(HouseData.Price),    // Cột cần dự đoán
                featureColumnName: "Features"));              // Cột đầu vào

        // Fit = huấn luyện model trên dữ liệu mẫu
        var model = pipeline.Fit(dataView);

        // Lưu model + schema ra file zip để lần sau load nhanh, không train lại
        mlContext.Model.Save(model, dataView.Schema, _modelPath);
    }
}
