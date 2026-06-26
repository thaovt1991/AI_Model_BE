using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using AI_Model_BE.Models;
using LLama;
using LLama.Common;
using LLama.Native;

namespace AI_Model_BE.Services;

/// <summary>
/// Service chat với LLM local qua file .gguf (LLamaSharp).
/// Có bộ nhớ hội thoại theo profileId (lưu đĩa) + RAG tài liệu nội bộ.
/// </summary>
public sealed class LlamaChatService : IDisposable
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<LlamaChatService> _logger;
    private readonly DocumentKnowledgeService _documentService;
    private readonly ChatMemoryService _chatMemory;
    private readonly ChatProfileSettingsService _profileSettings;
    // --- Học model ngầm (LoRA) ---
    // _learningCollector: ghi cặp hỏi/đáp vào dataset train sau mỗi lượt chat
    // _modelLearning: biết đường dẫn adapter .gguf sau khi train xong
    private readonly LearningDataCollectorService _learningCollector;
    private readonly ModelLearningService _modelLearning;
    private readonly LottoChatIntentService _lottoChat;
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    private LLamaWeights? _weights;
    private ModelParams? _modelParams;
    private StatelessExecutor? _executor;
    // LoRA adapter = "lớp vá" nhỏ học từ chat — gắn lên model .gguf gốc khi infer
    private LoraAdapter? _loraAdapter;
    private string? _loadedAdapterPath;
    private bool _initialized;
    private bool _modelAvailable;

    public LlamaChatService(
        IConfiguration configuration,
        DocumentKnowledgeService documentService,
        ChatMemoryService chatMemory,
        ChatProfileSettingsService profileSettings,
        LearningDataCollectorService learningCollector,
        ModelLearningService modelLearning,
        LottoChatIntentService lottoChat,
        ILogger<LlamaChatService> logger)
    {
        _configuration = configuration;
        _documentService = documentService;
        _chatMemory = chatMemory;
        _profileSettings = profileSettings;
        _learningCollector = learningCollector;
        _modelLearning = modelLearning;
        _lottoChat = lottoChat;
        _logger = logger;
    }

    public async Task WarmUpAsync(CancellationToken cancellationToken = default)
    {
        var total = Stopwatch.StartNew();
        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            _logger.LogWarning("Warmup bỏ qua — chưa có model .gguf.");
            return;
        }

        var inference = CreateInferenceParams();
        inference.MaxTokens = 1;

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var prompt = BuildPrompt("Trả lời ngắn.", [], "ping");
            ApplyLearningAdapter();
            await foreach (var _ in _executor.InferAsync(prompt, inference, cancellationToken))
            {
                break;
            }
        }
        finally
        {
            _semaphore.Release();
        }

        _logger.LogInformation("LLM warmup xong sau {Ms} ms", total.ElapsedMilliseconds);
    }

    public async Task<ChatResponse> ChatAsync(
        string message,
        IReadOnlyList<Guid>? documentIds = null,
        string? conversationId = null,
        string? profileId = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return new ChatResponse("Vui lòng nhập câu hỏi.", IsMock: true);
        }

        var userMessage = message.Trim();
        var history = _chatMemory.GetHistoryForPrompt(profileId, conversationId);

        var lottoReply = await _lottoChat.TryBuildReplyAsync(userMessage, cancellationToken);
        if (lottoReply is not null)
        {
            RememberTurn(profileId, conversationId, userMessage, lottoReply);
            return new ChatResponse(
                lottoReply,
                HistoryTurns: _chatMemory.GetHistoryForPrompt(profileId, conversationId).Count);
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            var mockReply = BuildMockReply(userMessage, documentIds);
            RememberTurn(profileId, conversationId, userMessage, mockReply);
            return new ChatResponse(
                mockReply,
                IsMock: true,
                HistoryTurns: _chatMemory.GetHistoryForPrompt(profileId, conversationId).Count);
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var sw = Stopwatch.StartNew();
            var (systemPrompt, userContent) = BuildMessages(
                userMessage, documentIds, history.Count > 0, profileId);
            var prompt = BuildPrompt(systemPrompt, history, userContent);
            ApplyLearningAdapter();

            var replyBuilder = new StringBuilder();
            await foreach (var token in _executor.InferAsync(prompt, CreateInferenceParams(), cancellationToken))
            {
                replyBuilder.Append(token);
            }

            var reply = replyBuilder.ToString().Trim();
            if (string.IsNullOrWhiteSpace(reply))
            {
                reply = "Mô hình không trả về nội dung.";
            }

            RememberTurn(profileId, conversationId, userMessage, reply);

            _logger.LogInformation(
                "Chat xong sau {Ms} ms (docs={DocCount}, history={History}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                documentIds?.Count ?? 0,
                history.Count,
                prompt.Length);

            return new ChatResponse(reply, HistoryTurns: _chatMemory.GetHistoryForPrompt(profileId, conversationId).Count);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async IAsyncEnumerable<string> ChatStreamAsync(
        string message,
        IReadOnlyList<Guid>? documentIds = null,
        string? conversationId = null,
        string? profileId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            yield return "Vui lòng nhập câu hỏi.";
            yield break;
        }

        var userMessage = message.Trim();
        var history = _chatMemory.GetHistoryForPrompt(profileId, conversationId);

        var lottoReply = await _lottoChat.TryBuildReplyAsync(userMessage, cancellationToken);
        if (lottoReply is not null)
        {
            RememberTurn(profileId, conversationId, userMessage, lottoReply);
            yield return lottoReply;
            yield break;
        }

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            var mockReply = BuildMockReply(userMessage, documentIds);
            RememberTurn(profileId, conversationId, userMessage, mockReply);
            yield return mockReply;
            yield break;
        }

        await _semaphore.WaitAsync(cancellationToken);
        var sw = Stopwatch.StartNew();
        var tokenCount = 0;
        var promptLen = 0;
        var replyBuilder = new StringBuilder();

        try
        {
            var (systemPrompt, userContent) = BuildMessages(
                userMessage, documentIds, history.Count > 0, profileId);
            var prompt = BuildPrompt(systemPrompt, history, userContent);
            promptLen = prompt.Length;
            ApplyLearningAdapter();

            await foreach (var token in _executor.InferAsync(prompt, CreateInferenceParams(), cancellationToken))
            {
                tokenCount++;
                replyBuilder.Append(token);
                yield return token;
            }
        }
        finally
        {
            _semaphore.Release();
            RememberTurn(profileId, conversationId, userMessage, replyBuilder.ToString().Trim());
            _logger.LogInformation(
                "Stream chat xong sau {Ms} ms — {Tokens} token (docs={DocCount}, history={History}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                tokenCount,
                documentIds?.Count ?? 0,
                history.Count,
                promptLen);
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

            var sw = Stopwatch.StartNew();
            var modelPath = _configuration["Llama:ModelPath"] ?? "Models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

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
            var threads = _configuration.GetValue("Llama:Threads", 0);

            _modelParams = new ModelParams(modelPath)
            {
                ContextSize = (uint)contextSize,
                GpuLayerCount = gpuLayers,
                Threads = threads,
                BatchThreads = threads
            };

            _weights = await LLamaWeights.LoadFromFileAsync(_modelParams, cancellationToken);
            _executor = new StatelessExecutor(_weights, _modelParams, _logger)
            {
                ApplyTemplate = false
            };

            _modelAvailable = true;
            _initialized = true;
            TryLoadLearningAdapter();

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

    private string BuildPrompt(string systemPrompt, IReadOnlyList<ChatTurn> history, string userContent)
    {
        var template = new LLamaTemplate(_weights!, strict: false)
        {
            AddAssistant = true
        };
        template.Add("system", systemPrompt);

        foreach (var turn in history)
        {
            template.Add("user", turn.UserMessage);
            template.Add("assistant", turn.AssistantReply);
        }

        template.Add("user", userContent);
        return LLamaTemplate.Encoding.GetString(template.Apply());
    }

    private InferenceParams CreateInferenceParams()
    {
        var maxTokens = _configuration.GetValue("Llama:MaxTokens", 256);

        return new InferenceParams
        {
            MaxTokens = maxTokens,
            AntiPrompts = ["<|eot_id|>", "<|end_of_text|>"]
        };
    }

    private (string systemPrompt, string userContent) BuildMessages(
        string userMessage,
        IReadOnlyList<Guid>? documentIds,
        bool hasHistory,
        string? profileId)
    {
        var documentContext = _documentService.BuildContextForQuery(userMessage, documentIds);
        var aiName = _profileSettings.GetAiName(profileId);

        string systemPrompt;
        if (string.IsNullOrWhiteSpace(documentContext))
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI chạy local. Trả lời ngắn gọn, đúng trọng tâm bằng tiếng Việt."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI chạy local. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Trả lời ngắn gọn, đúng trọng tâm bằng tiếng Việt.";
        }
        else
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI phân tích tài liệu nội bộ. Chỉ trả lời dựa trên tài liệu được cung cấp. " +
                  "Nếu thông tin không có trong tài liệu, hãy nói rõ là không tìm thấy trong tài liệu nội bộ. " +
                  "Trả lời ngắn gọn bằng tiếng Việt."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI phân tích tài liệu nội bộ. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Chỉ trả lời dựa trên tài liệu được cung cấp. Nếu thông tin không có trong tài liệu, hãy nói rõ là không tìm thấy trong tài liệu nội bộ. " +
                  "Trả lời ngắn gọn bằng tiếng Việt.";
        }

        if (hasHistory)
        {
            systemPrompt += " Nhớ ngữ cảnh các lượt hội thoại trước trong phiên này để trả lời nhất quán.";
        }

        if (_lottoChat.IsLottoRelated(userMessage))
        {
            systemPrompt += " Người dùng có thể hỏi xổ số — gợi ý họ dùng câu như: \"Dự đoán XSMB\", \"Lô gan TP HCM\", \"Lịch sử miền Nam 7 ngày\".";
        }

        var userContent = string.IsNullOrWhiteSpace(documentContext)
            ? userMessage
            : $"{documentContext}\n\nCâu hỏi: {userMessage}";

        return (systemPrompt, userContent);
    }

    private void RememberTurn(
        string? profileId,
        string? conversationId,
        string userMessage,
        string assistantReply)
    {
        if (string.IsNullOrWhiteSpace(assistantReply))
        {
            return;
        }

        // 1) Nhớ hội thoại theo cuộc chat hiện tại (profileId + conversationId)
        _chatMemory.Append(profileId, conversationId, userMessage, assistantReply);

        // 2) Học ngầm: lưu cặp hỏi/đáp làm dữ liệu train LoRA (ghi file sau, không lag chat)
        _learningCollector.EnqueueSample(profileId, userMessage, assistantReply);
    }

    /// <summary>
    /// Gọi sau khi train LoRA xong — nạp lại file adapter mới vào model đang chạy.
    /// </summary>
    public async Task ReloadLearningAdapterAsync(CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            _loraAdapter?.Unload();
            _loraAdapter = null;
            _loadedAdapterPath = null;
            TryLoadLearningAdapter();
            ApplyLearningAdapter();
            _logger.LogInformation("Đã nạp lại LoRA adapter sau train.");
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Đọc file LoRA .gguf từ đĩa và gắn vào model gốc (chưa áp dụng lên context — gọi ApplyLearningAdapter sau).
    /// </summary>
    private void TryLoadLearningAdapter()
    {
        if (_weights is null || !_modelLearning.AdapterExists)
        {
            return;
        }

        var adapterPath = Path.GetFullPath(_modelLearning.AdapterPath);
        if (IsPlaceholderAdapter(adapterPath))
        {
            return;
        }

        if (_loadedAdapterPath == adapterPath && _loraAdapter is not null)
        {
            return;
        }

        try
        {
            _loraAdapter?.Unload();
            _loraAdapter = _weights.NativeHandle.LoadLoraFromFile(adapterPath);
            _loadedAdapterPath = adapterPath;
            _logger.LogInformation("Đã load LoRA adapter: {Path}", adapterPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không load được LoRA adapter tại {Path}", adapterPath);
            _loraAdapter = null;
            _loadedAdapterPath = null;
        }
    }

    /// <summary>
    /// File placeholder = JSON giả (khi chưa cài tool train) — không phải LoRA thật, bỏ qua.
    /// </summary>
    private static bool IsPlaceholderAdapter(string adapterPath)
    {
        try
        {
            var head = File.ReadAllText(adapterPath, Encoding.UTF8).TrimStart();
            return head.StartsWith('{') && head.Contains("learning-placeholder", StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Áp dụng LoRA lên context suy luận — model trả lời theo kiến thức đã học thêm.
    /// AdapterScale (appsettings) điều chỉnh độ mạnh: 1.0 = học full, 0.5 = ảnh hưởng nhẹ hơn.
    /// </summary>
    private void ApplyLearningAdapter()
    {
        var adapter = _loraAdapter;
        if (_executor is null || adapter is null)
        {
            return;
        }

        try
        {
            _executor.Context.NativeHandle.SetLoraAdapters(
                [(adapter, _modelLearning.AdapterScale)]);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không áp dụng được LoRA adapter lên context.");
        }
    }

    private string BuildMockReply(string message, IReadOnlyList<Guid>? documentIds)
    {
        var context = _documentService.BuildContextForQuery(message, documentIds);
        if (!string.IsNullOrWhiteSpace(context))
        {
            return $"[Chế độ mock — chưa có LLM .gguf]\n" +
                   $"Đã tìm thấy ngữ cảnh từ tài liệu nội bộ:\n\n{context}\n\n" +
                   $"Câu hỏi của bạn: \"{message}\"\n" +
                   "Hãy cấu hình file .gguf để AI trả lời phân tích đầy đủ.";
        }

        return $"[Chế độ mock — chưa có file .gguf]\n" +
               $"Bạn hỏi: \"{message}\"\n" +
               "Hãy đặt file model vào thư mục Models/ hoặc hỏi xổ số: \"Dự đoán XSMB\", \"Lô gan TP HCM\", \"Lịch sử miền Nam 7 ngày\".";
    }

    public void Dispose()
    {
        _loraAdapter?.Unload();
        _executor?.Context.Dispose();
        _weights?.Dispose();
        _semaphore.Dispose();
    }
}
