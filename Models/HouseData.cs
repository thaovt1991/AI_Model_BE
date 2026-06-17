using Microsoft.ML.Data;

namespace AI_Model_BE.Models;

/// <summary>Dữ liệu huấn luyện / đầu vào dự đoán giá nhà.</summary>
public class HouseData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1)]
    public float Bedrooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }
}

/// <summary>Kết quả dự đoán từ ML.NET pipeline.</summary>
public class HousePrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}
