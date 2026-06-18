// Cho phép yield return trong async method (ChatStreamAsync)
using System.Runtime.CompilerServices;
using System.Text;
using AI_Model_BE.Models;
using LLama;              // Wrapper .NET cho llama.cpp
using LLama.Common;       // ModelParams, InferenceParams, InteractiveExecutor

namespace AI_Model_BE.Services;

/// <summary>
/// Service chat với LLM local qua file .gguf (LLamaSharp).
/// Được gọi từ AiController: POST /api/ai/chat và /api/ai/chat/stream.
/// </summary>
public sealed class LlamaChatService : IDisposable
{
    // Đọc cấu hình từ appsettings.json (Llama:ModelPath, ContextSize, ...)
    private readonly IConfiguration _configuration;
    private readonly ILogger<LlamaChatService> _logger;

    // Semaphore = cổng chỉ cho 1 request chat vào lúc một (model local chạy chậm, tránh race)
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    // Trọng số model đã load từ file .gguf
    private LLamaWeights? _weights;

    // Context = "bộ nhớ" của model trong một phiên suy luận
    private LLamaContext? _context;

    // Executor thực hiện sinh text từ prompt
    private InteractiveExecutor? _executor;

    // Đã thử khởi tạo model chưa (dù thành công hay thất bại)
    private bool _initialized;

    // true = có file .gguf và load OK; false = dùng phản hồi mock
    private bool _modelAvailable;

    public LlamaChatService(IConfiguration configuration, ILogger<LlamaChatService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    /// <summary>
    /// Chat trả về toàn bộ câu trả lời một lần (JSON).
    /// Frontend có thể dùng endpoint này thay vì stream.
    /// </summary>
    public async Task<ChatResponse> ChatAsync(string message, CancellationToken cancellationToken = default)
    {
        // Kiểm tra input rỗng
        if (string.IsNullOrWhiteSpace(message))
        {
            return new ChatResponse("Vui lòng nhập câu hỏi.", IsMock: true);
        }

        // Lần đầu gọi sẽ load file .gguf (có thể mất vài chục giây với model lớn)
        await EnsureInitializedAsync(cancellationToken);

        // Không có model → trả mock để UI vẫn test được
        if (!_modelAvailable || _executor is null)
        {
            return new ChatResponse(BuildMockReply(message), IsMock: true);
        }

        // Chờ lượt (nếu có request khác đang chạy)
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            // Ghép prompt theo format model (system + user + assistant)
            var prompt = BuildPrompt(message);

            // Tham số sinh text: max token, dấu hiệu dừng (anti-prompt)
            var inferenceParams = CreateInferenceParams();

            // Gom từng token model sinh ra thành một chuỗi hoàn chỉnh
            var replyBuilder = new StringBuilder();

            // InferAsync trả IAsyncEnumerable — mỗi lần lặp nhận 1 token (từ/cụm từ)
            await foreach (var token in _executor.InferAsync(prompt, inferenceParams)
                               .WithCancellation(cancellationToken))
            {
                replyBuilder.Append(token);
            }

            var reply = replyBuilder.ToString().Trim();
            return new ChatResponse(
                string.IsNullOrWhiteSpace(reply) ? "Mô hình không trả về nội dung." : reply);
        }
        finally
        {
            // Luôn mở cổng cho request tiếp theo
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Chat stream — trả từng token cho Controller ghi ra Response (text/plain).
    /// Frontend đọc bằng fetch + ReadableStream.
    /// </summary>
    public async IAsyncEnumerable<string> ChatStreamAsync(
        string message,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            yield return "Vui lòng nhập câu hỏi.";
            yield break; // Thoát generator
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null)
        {
            yield return BuildMockReply(message);
            yield break;
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var prompt = BuildPrompt(message);
            var inferenceParams = CreateInferenceParams();

            // yield return: gửi từng token ngay khi model sinh ra (realtime)
            await foreach (var token in _executor.InferAsync(prompt, inferenceParams)
                               .WithCancellation(cancellationToken))
            {
                yield return token;
            }
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Khởi tạo model LLM một lần duy nhất (lazy load).
    /// </summary>
    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        // Đã init rồi thì bỏ qua
        if (_initialized)
        {
            return;
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            // Double-check: thread khác có thể đã init trong lúc chờ lock
            if (_initialized)
            {
                return;
            }

            // Đọc đường dẫn file .gguf từ appsettings.json
            var modelPath = _configuration["Llama:ModelPath"] ?? "Models/llama-model.gguf";

            // Nếu là đường dẫn tương đối → ghép với thư mục chạy app
            if (!Path.IsPathRooted(modelPath))
            {
                modelPath = Path.Combine(AppContext.BaseDirectory, modelPath);
            }

            // Không tìm thấy file → chuyển sang chế độ mock
            if (!File.Exists(modelPath))
            {
                _logger.LogWarning(
                    "Không tìm thấy file .gguf tại {Path}. Chat sẽ dùng phản hồi mock.",
                    modelPath);
                _modelAvailable = false;
                _initialized = true;
                return;
            }

            // Số token ngữ cảnh model nhớ được (càng lớn càng tốn RAM)
            var contextSize = _configuration.GetValue("Llama:ContextSize", 2048);

            // Số layer chạy trên GPU (0 = chạy CPU hoàn toàn)
            var gpuLayers = _configuration.GetValue("Llama:GpuLayerCount", 0);

            // Cấu hình nạp model
            var parameters = new ModelParams(modelPath)
            {
                ContextSize = (uint)contextSize,
                GpuLayerCount = gpuLayers
            };

            // Bước 1: đọc file .gguf vào RAM (bước nặng nhất)
            _weights = await LLamaWeights.LoadFromFileAsync(parameters, cancellationToken);

            // Bước 2: tạo context suy luận từ weights
            _context = _weights.CreateContext(parameters);

            // Bước 3: executor dùng để gọi InferAsync
            _executor = new InteractiveExecutor(_context);

            _modelAvailable = true;
            _initialized = true;

            _logger.LogInformation("LLamaSharp đã nạp model: {Path}", modelPath);
        }
        catch (Exception ex)
        {
            // Lỗi load (thiếu RAM, file hỏng, ...) → fallback mock
            _logger.LogError(ex, "Không thể khởi tạo LLamaSharp.");
            _modelAvailable = false;
            _initialized = true;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Tham số khi model sinh câu trả lời.
    /// </summary>
    private InferenceParams CreateInferenceParams()
    {
        var maxTokens = _configuration.GetValue("Llama:MaxTokens", 512);

        return new InferenceParams
        {
            MaxTokens = maxTokens, // Giới hạn độ dài câu trả lời
            // Khi model gặp chuỗi này thì dừng sinh (token kết thúc của Llama/Phi)
            AntiPrompts = ["<|eot_id|>", "<|end_of_text|>"]
        };
    }

    /// <summary>
    /// Ghép prompt theo format chat template.
    /// Model càng đúng format thì câu trả lời càng ổn định.
    /// </summary>
    private static string BuildPrompt(string userMessage)
    {
        return "<|system|>\n" +
               "Bạn là trợ lý AI chạy local, trả lời ngắn gọn bằng tiếng Việt.<|end|>\n" +
               "<|user|>\n" +
               $"{userMessage.Trim()}<|end|>\n" +
               "<|assistant|>\n";
    }

    /// <summary>
    /// Phản hồi giả khi chưa có file .gguf — để Frontend vẫn test luồng UI.
    /// </summary>
    private static string BuildMockReply(string message) =>
        $"[Chế độ mock — chưa có file .gguf]\n" +
        $"Bạn hỏi: \"{message.Trim()}\"\n" +
        "Hãy đặt file model vào thư mục Models/ và cập nhật Llama:ModelPath trong appsettings.json.";

    /// <summary>
    /// Giải phóng RAM khi app tắt (ASP.NET gọi khi shutdown Singleton).
    /// </summary>
    public void Dispose()
    {
        _context?.Dispose();
        _weights?.Dispose();
        _semaphore.Dispose();
    }
}
