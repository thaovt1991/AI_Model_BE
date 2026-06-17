using System.Runtime.CompilerServices;
using System.Text;
using AI_Model_BE.Models;
using LLama;
using LLama.Common;

namespace AI_Model_BE.Services;

/// <summary>
/// Gọi LLamaSharp với file .gguf cấu hình trong appsettings.json.
/// Nếu chưa có model, trả về phản hồi mock để FE vẫn test được luồng chat.
/// </summary>
public sealed class LlamaChatService : IDisposable
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<LlamaChatService> _logger;
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    private LLamaWeights? _weights;
    private LLamaContext? _context;
    private InteractiveExecutor? _executor;
    private bool _initialized;
    private bool _modelAvailable;

    public LlamaChatService(IConfiguration configuration, ILogger<LlamaChatService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public async Task<ChatResponse> ChatAsync(string message, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return new ChatResponse("Vui lòng nhập câu hỏi.", IsMock: true);
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null)
        {
            return new ChatResponse(BuildMockReply(message), IsMock: true);
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var prompt = BuildPrompt(message);
            var inferenceParams = CreateInferenceParams();
            var replyBuilder = new StringBuilder();

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
            _semaphore.Release();
        }
    }

    public async IAsyncEnumerable<string> ChatStreamAsync(
        string message,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            yield return "Vui lòng nhập câu hỏi.";
            yield break;
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

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_initialized)
        {
            return;
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
            {
                return;
            }

            var modelPath = _configuration["Llama:ModelPath"] ?? "Models/llama-model.gguf";
            if (!Path.IsPathRooted(modelPath))
            {
                modelPath = Path.Combine(AppContext.BaseDirectory, modelPath);
            }

            if (!File.Exists(modelPath))
            {
                _logger.LogWarning(
                    "Không tìm thấy file .gguf tại {Path}. Chat sẽ dùng phản hồi mock.",
                    modelPath);
                _modelAvailable = false;
                _initialized = true;
                return;
            }

            var contextSize = _configuration.GetValue("Llama:ContextSize", 2048);
            var gpuLayers = _configuration.GetValue("Llama:GpuLayerCount", 0);

            var parameters = new ModelParams(modelPath)
            {
                ContextSize = (uint)contextSize,
                GpuLayerCount = gpuLayers
            };

            _weights = await LLamaWeights.LoadFromFileAsync(parameters, cancellationToken);
            _context = _weights.CreateContext(parameters);
            _executor = new InteractiveExecutor(_context);
            _modelAvailable = true;
            _initialized = true;

            _logger.LogInformation("LLamaSharp đã nạp model: {Path}", modelPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Không thể khởi tạo LLamaSharp.");
            _modelAvailable = false;
            _initialized = true;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    private InferenceParams CreateInferenceParams()
    {
        var maxTokens = _configuration.GetValue("Llama:MaxTokens", 512);

        return new InferenceParams
        {
            MaxTokens = maxTokens,
            AntiPrompts = ["<|eot_id|>", "<|end_of_text|>"]
        };
    }

    private static string BuildPrompt(string userMessage) =>
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
        "Bạn là trợ lý AI chạy local, trả lời ngắn gọn bằng tiếng Việt." +
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
        $"{userMessage.Trim()}" +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    private static string BuildMockReply(string message) =>
        $"[Chế độ mock — chưa có file .gguf]\n" +
        $"Bạn hỏi: \"{message.Trim()}\"\n" +
        "Hãy đặt file model vào thư mục Models/ và cập nhật Llama:ModelPath trong appsettings.json.";

    public void Dispose()
    {
        _context?.Dispose();
        _weights?.Dispose();
        _semaphore.Dispose();
    }
}
