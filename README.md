# AI_Model_BE - Backend .NET 8 Web API

Backend này cung cấp 2 nhóm API AI chạy local:

- `POST /api/ai/predict`: dự đoán giá nhà bằng ML.NET.
- `POST /api/ai/chat`: trả lời chat bằng LLamaSharp.
- `POST /api/ai/chat/stream`: trả token từng phần để Frontend hiển thị realtime.

## Các thư viện chính

### ASP.NET Core Web API

Đây là framework Backend của .NET dùng để tạo REST API. Trong dự án này, `AiController` nhận request từ Angular, gọi service xử lý AI, rồi trả response JSON hoặc stream text.

File liên quan:

- `Program.cs`: đăng ký service, bật CORS, map controller.
- `Controllers/AiController.cs`: định nghĩa các endpoint `/api/ai/...`.

### Microsoft.ML

`Microsoft.ML` là thư viện ML.NET chính thức của Microsoft. Nó cho phép huấn luyện và chạy mô hình học máy truyền thống ngay trong .NET, không cần Python runtime.

Trong dự án này, ML.NET được dùng để minh họa bài toán regression: dự đoán giá nhà dựa trên:

- `Size`: diện tích.
- `Bedrooms`: số phòng ngủ.

File liên quan:

- `Services/MlPredictionService.cs`: tự huấn luyện model giả lập khi Backend chạy lần đầu.
- `Models/HouseData.cs`: schema input và output cho ML.NET.
- `Models/house-price-model.zip`: file model được sinh tự động.

Luồng xử lý:

1. Backend start.
2. `MlPredictionService.EnsureModelReady()` kiểm tra file model.
3. Nếu chưa có model, service tạo dữ liệu mẫu và huấn luyện model.
4. Khi Angular gọi `/api/ai/predict`, service dùng `PredictionEngine` để dự đoán.

### LLamaSharp

`LLamaSharp` là thư viện .NET wrapper cho llama.cpp. Nó giúp ứng dụng .NET nạp và chạy các mô hình LLM local định dạng `.gguf`.

Trong dự án này, LLamaSharp xử lý chatbot local. Khi có file `.gguf`, Backend sẽ nạp model và sinh câu trả lời. Khi chưa có model, service trả phản hồi mock để bạn vẫn test được UI.

File liên quan:

- `Services/LlamaChatService.cs`: nạp model, tạo context, sinh text.
- `appsettings.json`: cấu hình đường dẫn model và thông số inference.
- `Models/llama-model.gguf`: vị trí gợi ý để đặt model local.

### LLamaSharp.Backend.Cpu

Đây là backend runtime CPU cho LLamaSharp. Nó cho phép chạy LLM trên CPU, phù hợp để demo local/offline mà không cần GPU.

Nếu sau này muốn chạy nhanh hơn bằng GPU, bạn có thể thay backend CPU bằng backend CUDA/Vulkan phù hợp phần cứng.

## Cấu hình model `.gguf`

Trong `appsettings.json`:

```json
"Llama": {
  "ModelPath": "D:\\Job\\Model AI\\Model AI Local\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
  "ContextSize": 2048,
  "GpuLayerCount": 0,
  "MaxTokens": 512
}
```

Ý nghĩa:

- `ModelPath`: đường dẫn file `.gguf`. Có thể dùng đường dẫn tuyệt đối như ví dụ trên, hoặc đường dẫn tương đối như `Models/llama-model.gguf`.
- `ContextSize`: số token ngữ cảnh model có thể nhớ trong một lượt xử lý.
- `GpuLayerCount`: số layer đẩy lên GPU. Với `LLamaSharp.Backend.Cpu`, để `0`.
- `MaxTokens`: số token tối đa model sinh ra cho mỗi câu trả lời.

## CORS

Trong `Program.cs`, CORS cho phép Angular dev server gọi API:

```csharp
policy.WithOrigins("http://localhost:4200")
    .AllowAnyHeader()
    .AllowAnyMethod();
```

Nếu đổi port Frontend, cần sửa lại origin này.

## Chạy Backend

```powershell
cd "D:\Job\Model AI\AI_Model_BE"
dotnet run
```

Backend mặc định chạy tại:

```text
http://localhost:5296
```

## Xử lý lỗi port 5296 đang được sử dụng

Nếu chạy Backend và gặp lỗi:

```text
System.IO.IOException: Failed to bind to address http://127.0.0.1:5296: address already in use.
```

Nghĩa là đang có một process Backend cũ hoặc ứng dụng khác chiếm port `5296`.

Kiểm tra process đang giữ port:

```powershell
netstat -ano | Select-String ':5296'
```

Kết quả thường có dạng:

```text
TCP    127.0.0.1:5296    0.0.0.0:0    LISTENING    21920
```

Số cuối cùng là `PID`. Dừng process đó:

```powershell
Stop-Process -Id 21920 -Force
```

Sau đó chạy lại Backend:

```powershell
dotnet run --urls "http://localhost:5296"
```

Nếu đang mở nhiều terminal chạy `dotnet run`, hãy dừng các terminal Backend cũ trước để tránh chạy trùng instance.

