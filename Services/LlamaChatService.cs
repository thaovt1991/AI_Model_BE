// Cho phép yield return trong async method (ChatStreamAsync)
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using AI_Model_BE.Models;
using LLama;              // Wrapper .NET cho llama.cpp — nạp file .gguf
using LLama.Common;       // ModelParams, InferenceParams, StatelessExecutor

namespace AI_Model_BE.Services;

/// <summary>
/// Service chat với LLM local qua file .gguf (LLamaSharp).
/// Được gọi từ AiController: POST /api/ai/chat và /api/ai/chat/stream.
///
/// Luồng khi user hỏi kèm tài liệu:
///   1. DocumentKnowledgeService.BuildContextForQuery → tìm đoạn text liên quan (RAG)
///   2. BuildMessages → ghép system + user (có ngữ cảnh tài liệu)
///   3. BuildPrompt → dùng LLamaTemplate (đúng format Meta-Llama-3)
///   4. StatelessExecutor.InferAsync → sinh token từng phần
/// </summary>
public sealed class LlamaChatService : IDisposable
{
    // Đọc cấu hình từ appsettings.json (Llama:ModelPath, ContextSize, MaxTokens...)
    private readonly IConfiguration _configuration;
    private readonly ILogger<LlamaChatService> _logger;

    // Service quản lý tài liệu nội bộ — inject context vào prompt khi chat
    private readonly DocumentKnowledgeService _documentService;

    // Semaphore = cổng chỉ cho 1 request chat vào lúc một.
    // Model local chạy chậm trên CPU; nếu 2 request cùng lúc dễ race / tốn RAM gấp đôi.
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    // Trọng số model đã load từ file .gguf (bước nặng nhất — vài GB RAM)
    private LLamaWeights? _weights;

    // Tham số context/threads — dùng lại khi tạo StatelessExecutor
    private ModelParams? _modelParams;

    // StatelessExecutor: mỗi câu hỏi độc lập, KHÔNG nhớ lịch sử chat trước.
    // (InteractiveExecutor cũ tích lũy context → chậm dần / treo sau vài lượt)
    private StatelessExecutor? _executor;

    // Đã thử khởi tạo model chưa (dù thành công hay thất bại)
    private bool _initialized;

    // true = có file .gguf và load OK; false = dùng phản hồi mock
    private bool _modelAvailable;

    public LlamaChatService(
        IConfiguration configuration,
        DocumentKnowledgeService documentService,
        ILogger<LlamaChatService> logger)
    {
        _configuration = configuration;
        _documentService = documentService;
        _logger = logger;
    }

    /// <summary>
    /// Nạp model ngay khi Backend khởi động (Program.cs gọi WarmUpOnStart).
    ///
    /// Vì sao cần warmup?
    /// - Load file .gguf lần đầu mất 30–60 giây.
    /// - Nếu không warmup, user phải chờ ở tin nhắn chat đầu tiên.
    /// - Warmup chạy 1 token "ping" để llama.cpp khởi tạo sẵn pipeline suy luận.
    /// </summary>
    public async Task WarmUpAsync(CancellationToken cancellationToken = default)
    {
        var total = Stopwatch.StartNew();
        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            _logger.LogWarning("Warmup bỏ qua — chưa có model .gguf.");
            return;
        }

        // Chỉ sinh 1 token — đủ để "làm nóng" model, không tốn thời gian
        var inference = CreateInferenceParams();
        inference.MaxTokens = 1;

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var prompt = BuildPrompt("Trả lời ngắn.", "ping");
            await foreach (var _ in _executor.InferAsync(prompt, inference, cancellationToken))
            {
                break; // Nhận token đầu tiên rồi dừng
            }
        }
        finally
        {
            _semaphore.Release();
        }

        _logger.LogInformation("LLM warmup xong sau {Ms} ms", total.ElapsedMilliseconds);
    }

    /// <summary>
    /// Chat trả về toàn bộ câu trả lời một lần (JSON).
    /// Frontend có thể dùng endpoint này thay vì stream.
    /// </summary>
    public async Task<ChatResponse> ChatAsync(
        string message,
        IReadOnlyList<Guid>? documentIds = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return new ChatResponse("Vui lòng nhập câu hỏi.", IsMock: true);
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            return new ChatResponse(BuildMockReply(message, documentIds), IsMock: true);
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var sw = Stopwatch.StartNew();

            // Bước 1: Lấy ngữ cảnh từ tài liệu (nếu user đã chọn file trên UI)
            var (systemPrompt, userContent) = BuildMessages(message, documentIds);

            // Bước 2: Ghép prompt đúng template Llama 3
            var prompt = BuildPrompt(systemPrompt, userContent);

            // Bước 3: Gom từng token model sinh ra thành chuỗi hoàn chỉnh
            var replyBuilder = new StringBuilder();
            await foreach (var token in _executor.InferAsync(prompt, CreateInferenceParams(), cancellationToken))
            {
                replyBuilder.Append(token);
            }

            var reply = replyBuilder.ToString().Trim();
            _logger.LogInformation(
                "Chat xong sau {Ms} ms (docs={DocCount}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                documentIds?.Count ?? 0,
                prompt.Length);

            return new ChatResponse(
                string.IsNullOrWhiteSpace(reply) ? "Mô hình không trả về nội dung." : reply);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Chat stream — trả từng token cho Controller ghi ra Response (text/plain).
    /// Frontend đọc bằng fetch + ReadableStream (ai.service.ts → streamChat).
    /// </summary>
    public async IAsyncEnumerable<string> ChatStreamAsync(
        string message,
        IReadOnlyList<Guid>? documentIds = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            yield return "Vui lòng nhập câu hỏi.";
            yield break; // Thoát generator — không sinh thêm token
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            yield return BuildMockReply(message, documentIds);
            yield break;
        }

        await _semaphore.WaitAsync(cancellationToken);
        var sw = Stopwatch.StartNew();
        var tokenCount = 0;
        var promptLen = 0;

        try
        {
            var (systemPrompt, userContent) = BuildMessages(message, documentIds);
            var prompt = BuildPrompt(systemPrompt, userContent);
            promptLen = prompt.Length;

            // yield return: gửi từng token ngay khi model sinh ra (realtime trên UI)
            await foreach (var token in _executor.InferAsync(prompt, CreateInferenceParams(), cancellationToken))
            {
                tokenCount++;
                yield return token;
            }
        }
        finally
        {
            _semaphore.Release();
            _logger.LogInformation(
                "Stream chat xong sau {Ms} ms — {Tokens} token (docs={DocCount}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                tokenCount,
                documentIds?.Count ?? 0,
                promptLen);
        }
    }

    /// <summary>
    /// Khởi tạo model LLM một lần duy nhất (lazy load — hoặc gọi sớm qua WarmUpAsync).
    /// </summary>
    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
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

            var sw = Stopwatch.StartNew();

            // Đọc đường dẫn file .gguf từ appsettings.json → Llama:ModelPath
            var modelPath = _configuration["Llama:ModelPath"] ?? "Models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
            var absolutePath = Path.GetFullPath(modelPath); // Chuyển đổi thành đường dẫn đầy đủ
            _logger.LogInformation($"--- DEBUG PATH ---");
            _logger.LogInformation($"Configured path: {modelPath}");
            _logger.LogInformation($"Absolute path: {absolutePath}");
            _logger.LogInformation($"File exists: {File.Exists(absolutePath)}");

            // Nếu là đường dẫn tương đối → ghép với thư mục chạy app
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

            // ContextSize: số token tối đa model nhớ (càng lớn càng tốn RAM)
            var contextSize = _configuration.GetValue("Llama:ContextSize", 2048);

            // GpuLayerCount: số layer chạy trên GPU (0 = CPU hoàn toàn — chậm nhưng không cần GPU)
            var gpuLayers = _configuration.GetValue("Llama:GpuLayerCount", 0);

            // Threads: số luồng CPU (0 = tự dùng hết core máy)
            var threads = _configuration.GetValue("Llama:Threads", 0);

            _modelParams = new ModelParams(modelPath)
            {
                ContextSize = (uint)contextSize,
                GpuLayerCount = gpuLayers,
                Threads = threads,
                BatchThreads = threads
            };

            // Bước nặng nhất: đọc file .gguf (4–5 GB) vào RAM
            _weights = await LLamaWeights.LoadFromFileAsync(_modelParams, cancellationToken);

            // StatelessExecutor: mỗi InferAsync là phiên mới — không tích lũy context cũ
            _executor = new StatelessExecutor(_weights, _modelParams, _logger)
            {
                // false vì ta tự build prompt bằng LLamaTemplate (đúng format model)
                ApplyTemplate = false
            };

            _modelAvailable = true;
            _initialized = true;

            _logger.LogInformation(
                "LLamaSharp nạp model sau {Ms} ms — CPU threads={Threads}, context={Context}",
                sw.ElapsedMilliseconds,
                threads == 0 ? Environment.ProcessorCount : threads,
                contextSize);
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

    /// <summary>
    /// Ghép prompt theo chat template gốc của model (Meta-Llama-3).
    ///
    /// KHÔNG dùng format cũ như &lt;|system|&gt; ... &lt;|end|&gt; — sai với Llama 3.
    /// LLamaTemplate đọc template từ metadata trong file .gguf.
    ///
    /// QUAN TRỌNG: template.Apply() trả về ReadOnlySpan&lt;byte&gt; (bytes UTF-8).
    /// KHÔNG được gọi .ToString() trên span — sẽ ra rác kiểu "System.ReadOnlySpan..."
    /// và model trả lời linh tinh. Phải decode bằng LLamaTemplate.Encoding.GetString().
    /// </summary>
    private string BuildPrompt(string systemPrompt, string userContent)
    {
        var template = new LLamaTemplate(_weights!, strict: false)
        {
            AddAssistant = true // Thêm header assistant để model biết bắt đầu trả lời
        };
        template.Add("system", systemPrompt);
        template.Add("user", userContent);

        var promptBytes = template.Apply();
        return LLamaTemplate.Encoding.GetString(promptBytes);
    }

    /// <summary>Tham số khi model sinh câu trả lời.</summary>
    private InferenceParams CreateInferenceParams()
    {
        // MaxTokens càng nhỏ → câu trả lời ngắn hơn, nhanh hơn trên CPU
        var maxTokens = _configuration.GetValue("Llama:MaxTokens", 256);

        return new InferenceParams
        {
            MaxTokens = maxTokens,
            // Khi model gặp chuỗi này thì dừng sinh (token kết thúc của Llama 3)
            AntiPrompts = ["<|eot_id|>", "<|end_of_text|>"]
        };
    }

    /// <summary>
    /// Tách system prompt và nội dung user.
    /// Nếu có documentIds → BuildContextForQuery inject đoạn text tài liệu vào user message.
    /// </summary>
    private (string systemPrompt, string userContent) BuildMessages(
        string userMessage,
        IReadOnlyList<Guid>? documentIds)
    {
        var documentContext = _documentService.BuildContextForQuery(userMessage, documentIds);

        var systemPrompt = string.IsNullOrWhiteSpace(documentContext)
            ? "Bạn là trợ lý AI chạy local. Trả lời ngắn gọn, đúng trọng tâm bằng tiếng Việt."
            : "Bạn là trợ lý AI phân tích tài liệu nội bộ. Chỉ trả lời dựa trên tài liệu được cung cấp. " +
              "Nếu thông tin không có trong tài liệu, hãy nói rõ là không tìm thấy trong tài liệu nội bộ. " +
              "Trả lời ngắn gọn bằng tiếng Việt.";

        var userContent = string.IsNullOrWhiteSpace(documentContext)
            ? userMessage.Trim()
            : $"{documentContext}\n\nCâu hỏi: {userMessage.Trim()}";

        return (systemPrompt, userContent);
    }

    /// <summary>Phản hồi giả khi chưa có file .gguf — để UI vẫn test được.</summary>
    private string BuildMockReply(string message, IReadOnlyList<Guid>? documentIds)
    {
        var context = _documentService.BuildContextForQuery(message, documentIds);
        if (!string.IsNullOrWhiteSpace(context))
        {
            return $"[Chế độ mock — chưa có LLM .gguf]\n" +
                   $"Đã tìm thấy ngữ cảnh từ tài liệu nội bộ:\n\n{context}\n\n" +
                   $"Câu hỏi của bạn: \"{message.Trim()}\"\n" +
                   "Hãy cấu hình file .gguf để AI trả lời phân tích đầy đủ.";
        }

        return $"[Chế độ mock — chưa có file .gguf]\n" +
               $"Bạn hỏi: \"{message.Trim()}\"\n" +
               "Hãy đặt file model vào thư mục Models/ và cập nhật Llama:ModelPath trong appsettings.json.";
    }

    /// <summary>Giải phóng RAM khi app tắt (ASP.NET gọi khi shutdown Singleton).</summary>
    public void Dispose()
    {
        _executor?.Context.Dispose();
        _weights?.Dispose();
        _semaphore.Dispose();
    }
}
