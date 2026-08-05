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
/// Có bộ nhớ hội thoại theo profileId (lưu đĩa) + RAG tài liệu nội bộ + research mạng.
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
    private readonly WebResearchService _webResearch;
    private readonly DeepResearchService _deepResearch;
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
        WebResearchService webResearch,
        DeepResearchService deepResearch,
        ILogger<LlamaChatService> logger)
    {
        _configuration = configuration;
        _documentService = documentService;
        _chatMemory = chatMemory;
        _profileSettings = profileSettings;
        _learningCollector = learningCollector;
        _modelLearning = modelLearning;
        _lottoChat = lottoChat;
        _webResearch = webResearch;
        _deepResearch = deepResearch;
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
        bool? enableWebSearch = null,
        bool deepResearch = false,
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

        // Hybrid RAG tài liệu + research/deep research mạng
        var docRetrieval = _documentService.Retrieve(userMessage, documentIds);
        var researchBundle = await GatherResearchAsync(
            userMessage, enableWebSearch, deepResearch, cancellationToken);
        var allCitations = MergeCitations(docRetrieval.Citations, researchBundle.Citations);

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            var mockReply = BuildMockReply(userMessage, docRetrieval, researchBundle);
            RememberTurn(profileId, conversationId, userMessage, mockReply);
            return new ChatResponse(
                mockReply,
                IsMock: true,
                HistoryTurns: _chatMemory.GetHistoryForPrompt(profileId, conversationId).Count,
                UsedWebResearch: researchBundle.HasWeb,
                WebSourceCount: researchBundle.WebSourceCount,
                UsedDeepResearch: researchBundle.IsDeep,
                Citations: allCitations);
        }

        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            var sw = Stopwatch.StartNew();
            var (systemPrompt, userContent) = BuildMessages(
                userMessage,
                docRetrieval.ContextBlock,
                history.Count > 0,
                profileId,
                OffsetCitationMarkers(researchBundle.ContextBlock, docRetrieval.Citations.Count),
                researchBundle.IsDeep);
            var prompt = BuildPrompt(systemPrompt, history, userContent);
            ApplyLearningAdapter();

            var replyBuilder = new StringBuilder();
            await foreach (var token in _executor.InferAsync(
                               prompt,
                               CreateInferenceParams(researchBundle.HasWeb, researchBundle.IsDeep),
                               cancellationToken))
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
                "Chat xong sau {Ms} ms (docs={DocCount}, web={WebCount}, deep={Deep}, history={History}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                docRetrieval.Citations.Count,
                researchBundle.WebSourceCount,
                researchBundle.IsDeep,
                history.Count,
                prompt.Length);

            return new ChatResponse(
                reply,
                HistoryTurns: _chatMemory.GetHistoryForPrompt(profileId, conversationId).Count,
                UsedWebResearch: researchBundle.HasWeb,
                WebSourceCount: researchBundle.WebSourceCount,
                UsedDeepResearch: researchBundle.IsDeep,
                Citations: allCitations);
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
        bool? enableWebSearch = null,
        bool deepResearch = false,
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

        // --- Phase RAG tài liệu (nhanh, local) ---
        yield return ChatStreamMetaCodec.Status("rag", "Đang tìm trong tài liệu nội bộ (Hybrid RAG)...");
        var docRetrieval = _documentService.Retrieve(userMessage, documentIds);
        if (docRetrieval.HasResults)
        {
            yield return ChatStreamMetaCodec.Status(
                "rag",
                $"Đã chọn {docRetrieval.Citations.Count} đoạn tài liệu liên quan.");
        }

        // --- Phase research mạng ---
        var wantDeep = deepResearch && _deepResearch.IsEnabled;
        var wantWeb = wantDeep || _webResearch.ShouldResearch(userMessage, enableWebSearch);

        if (wantDeep)
        {
            yield return ChatStreamMetaCodec.Status(
                "deep_research",
                "Deep Research: đang chạy nhiều truy vấn phụ trên mạng...");
        }
        else if (wantWeb)
        {
            yield return ChatStreamMetaCodec.Status("research", "Đang tìm kiếm trên mạng...");
        }

        var researchBundle = await GatherResearchAsync(
            userMessage, enableWebSearch, deepResearch, cancellationToken);

        if (wantWeb || wantDeep)
        {
            yield return researchBundle.HasWeb
                ? ChatStreamMetaCodec.Status(
                    wantDeep ? "deep_research" : "research",
                    $"Đã có {researchBundle.WebSourceCount} nguồn web. Đang suy luận...")
                : ChatStreamMetaCodec.Status(
                    wantDeep ? "deep_research" : "research",
                    "Không lấy được nguồn mạng đáng tin — trả lời theo kiến thức/tài liệu sẵn có.");
        }

        var allCitations = MergeCitations(docRetrieval.Citations, researchBundle.Citations);
        if (allCitations.Count > 0)
        {
            // Gửi citation cho UI TRƯỚC khi stream token — bubble hiện nguồn ngay
            yield return ChatStreamMetaCodec.Citations(allCitations);
        }

        yield return ChatStreamMetaCodec.Status("thinking", "Đang sinh câu trả lời...");

        await EnsureInitializedAsync(cancellationToken);

        if (!_modelAvailable || _executor is null || _weights is null)
        {
            var mockReply = BuildMockReply(userMessage, docRetrieval, researchBundle);
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
                userMessage,
                docRetrieval.ContextBlock,
                history.Count > 0,
                profileId,
                OffsetCitationMarkers(researchBundle.ContextBlock, docRetrieval.Citations.Count),
                researchBundle.IsDeep);
            var prompt = BuildPrompt(systemPrompt, history, userContent);
            promptLen = prompt.Length;
            ApplyLearningAdapter();

            await foreach (var token in _executor.InferAsync(
                               prompt,
                               CreateInferenceParams(researchBundle.HasWeb, researchBundle.IsDeep),
                               cancellationToken))
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
                "Stream chat xong sau {Ms} ms — {Tokens} token (docs={DocCount}, web={WebCount}, deep={Deep}, history={History}, promptLen={PromptLen})",
                sw.ElapsedMilliseconds,
                tokenCount,
                docRetrieval.Citations.Count,
                researchBundle.WebSourceCount,
                researchBundle.IsDeep,
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

    private InferenceParams CreateInferenceParams(bool usedWebResearch = false, bool deepResearch = false)
    {
        var maxTokens = _configuration.GetValue("Llama:MaxTokens", 256);
        if (deepResearch)
        {
            maxTokens = Math.Max(maxTokens, _configuration.GetValue("DeepResearch:MaxTokens", 512));
        }
        else if (usedWebResearch)
        {
            maxTokens = Math.Max(maxTokens, _configuration.GetValue("WebResearch:MaxTokens", 384));
        }

        return new InferenceParams
        {
            MaxTokens = maxTokens,
            AntiPrompts = ["<|eot_id|>", "<|end_of_text|>"]
        };
    }

    /// <summary>
    /// Gom research thường hoặc Deep Research thành 1 bundle context + citations.
    /// Deep Research luôn bật web; research thường tôn trọng EnableWebSearch/AutoDetect.
    /// </summary>
    private async Task<ResearchBundle> GatherResearchAsync(
        string userMessage,
        bool? enableWebSearch,
        bool deepResearch,
        CancellationToken cancellationToken)
    {
        try
        {
            if (deepResearch && _deepResearch.IsEnabled)
            {
                var deep = await _deepResearch.ResearchAsync(userMessage, cancellationToken);
                return new ResearchBundle(
                    deep.HasResults,
                    IsDeep: true,
                    deep.Citations.Count,
                    deep.ContextBlock,
                    deep.Citations);
            }

            if (!_webResearch.ShouldResearch(userMessage, enableWebSearch))
            {
                return ResearchBundle.Empty;
            }

            var normal = await _webResearch.ResearchAsync(userMessage, cancellationToken);
            var citations = normal.ToCitations();
            return new ResearchBundle(
                normal.HasResults,
                IsDeep: false,
                citations.Count,
                normal.ContextBlock,
                citations);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Gather research thất bại cho: {Message}", userMessage);
            return ResearchBundle.Empty;
        }
    }

    /// <summary>
    /// Gộp citation tài liệu + web, đánh lại Id liên tục [1]..[N] để khớp prompt.
    /// </summary>
    private static IReadOnlyList<CitationSource> MergeCitations(
        IReadOnlyList<CitationSource> docs,
        IReadOnlyList<CitationSource> web)
    {
        var merged = new List<CitationSource>();
        var id = 1;
        foreach (var c in docs)
        {
            merged.Add(CloneCitation(c, id++));
        }

        foreach (var c in web)
        {
            merged.Add(CloneCitation(c, id++));
        }

        return merged;
    }

    private static CitationSource CloneCitation(CitationSource c, int id) => new()
    {
        Id = id,
        Kind = c.Kind,
        Title = c.Title,
        Url = c.Url,
        FileName = c.FileName,
        ChunkIndex = c.ChunkIndex,
        Snippet = c.Snippet,
        Score = c.Score,
        SourceLabel = c.SourceLabel
    };

    /// <summary>
    /// Tài liệu dùng [1]..[D]; web vốn cũng bắt đầu từ [1] → lệch số.
    /// Cộng offset = số citation tài liệu để khớp danh sách MergeCitations trên UI.
    /// </summary>
    private static string OffsetCitationMarkers(string? context, int offset)
    {
        if (string.IsNullOrWhiteSpace(context) || offset <= 0)
        {
            return context ?? string.Empty;
        }

        return System.Text.RegularExpressions.Regex.Replace(
            context,
            @"\[(\d+)\]",
            m => $"[{int.Parse(m.Groups[1].Value) + offset}]");
    }

    private (string systemPrompt, string userContent) BuildMessages(
        string userMessage,
        string? documentContext,
        bool hasHistory,
        string? profileId,
        string? webResearchContext = null,
        bool isDeepResearch = false)
    {
        var aiName = _profileSettings.GetAiName(profileId);
        var hasDocs = !string.IsNullOrWhiteSpace(documentContext);
        var hasWeb = !string.IsNullOrWhiteSpace(webResearchContext);

        string systemPrompt;
        if (isDeepResearch && hasWeb)
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI nghiên cứu sâu. Tổng hợp nhiều nguồn web, nêu điểm đồng thuận/mâu thuẫn, " +
                  "trích dẫn bằng [1], [2]... Trả lời có cấu trúc (tóm tắt → chi tiết → kết luận). Tiếng Việt. Không bịa số liệu."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI nghiên cứu sâu. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Tổng hợp nhiều nguồn web, nêu điểm đồng thuận/mâu thuẫn, trích dẫn bằng [1], [2]... " +
                  "Trả lời có cấu trúc. Tiếng Việt. Không bịa số liệu.";
        }
        else if (hasDocs && hasWeb)
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI thông minh. Ưu tiên tài liệu nội bộ; bổ sung bằng kết quả tìm kiếm web khi cần. " +
                  "Trích dẫn [n] khi dùng nguồn. Trả lời rõ ràng bằng tiếng Việt."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI thông minh. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Ưu tiên tài liệu nội bộ; bổ sung bằng web. Trích dẫn [n] khi dùng nguồn. Tiếng Việt.";
        }
        else if (hasDocs)
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI phân tích tài liệu nội bộ (Hybrid RAG). Chỉ trả lời dựa trên tài liệu được cung cấp. " +
                  "Trích dẫn [n]. Nếu không có trong tài liệu, nói rõ. Tiếng Việt."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI phân tích tài liệu nội bộ. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Chỉ trả lời dựa trên tài liệu. Trích dẫn [n]. Tiếng Việt.";
        }
        else if (hasWeb)
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI có research mạng. Dựa vào kết quả web, nêu nguồn [n], không bịa số liệu. Tiếng Việt."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI có research mạng. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Dựa vào kết quả web, nêu nguồn [n], không bịa số liệu. Tiếng Việt.";
        }
        else
        {
            systemPrompt = string.IsNullOrWhiteSpace(aiName)
                ? "Bạn là trợ lý AI chạy local, thông minh và hữu ích. Suy nghĩ từng bước khi cần, trả lời đúng trọng tâm bằng tiếng Việt. " +
                  "Nếu thiếu dữ liệu thời sự/cập nhật, hãy nói rõ giới hạn kiến thức."
                : $"Bạn tên là {aiName}. Bạn là trợ lý AI chạy local, thông minh và hữu ích. Khi được hỏi tên, hãy trả lời là {aiName}. " +
                  "Suy nghĩ từng bước khi cần, trả lời đúng trọng tâm bằng tiếng Việt. " +
                  "Nếu thiếu dữ liệu thời sự/cập nhật, hãy nói rõ giới hạn kiến thức.";
        }

        if (hasHistory)
        {
            systemPrompt += " Nhớ ngữ cảnh các lượt hội thoại trước trong phiên này để trả lời nhất quán.";
        }

        if (_lottoChat.IsLottoRelated(userMessage))
        {
            systemPrompt += " Người dùng có thể hỏi xổ số — gợi ý họ dùng câu như: \"Dự đoán XSMB\", \"Lô gan TP HCM\", \"Lịch sử miền Nam 7 ngày\".";
        }

        var parts = new List<string>();
        if (hasDocs)
        {
            parts.Add(documentContext!);
        }

        if (hasWeb)
        {
            parts.Add(webResearchContext!);
        }

        var userContent = parts.Count == 0
            ? userMessage
            : $"{string.Join("\n\n", parts)}\n\nCâu hỏi: {userMessage}";

        return (systemPrompt, userContent);
    }

    private sealed record ResearchBundle(
        bool HasWeb,
        bool IsDeep,
        int WebSourceCount,
        string ContextBlock,
        IReadOnlyList<CitationSource> Citations)
    {
        public static ResearchBundle Empty { get; } =
            new(false, false, 0, string.Empty, Array.Empty<CitationSource>());
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

    private string BuildMockReply(
        string message,
        DocumentRetrievalResult docRetrieval,
        ResearchBundle research)
    {
        var context = docRetrieval.ContextBlock;
        var web = research.HasWeb ? research.ContextBlock : null;

        if (!string.IsNullOrWhiteSpace(context) || !string.IsNullOrWhiteSpace(web))
        {
            var sb = new StringBuilder();
            sb.AppendLine("[Chế độ mock — chưa có LLM .gguf]");
            if (!string.IsNullOrWhiteSpace(context))
            {
                sb.AppendLine("Ngữ cảnh Hybrid RAG (tài liệu):");
                sb.AppendLine(context);
                sb.AppendLine();
            }

            if (!string.IsNullOrWhiteSpace(web))
            {
                sb.AppendLine(research.IsDeep ? "Deep Research:" : "Research mạng:");
                sb.AppendLine(web);
                sb.AppendLine();
            }

            sb.AppendLine($"Câu hỏi của bạn: \"{message}\"");
            sb.Append("Hãy cấu hình file .gguf để AI phân tích đầy đủ.");
            return sb.ToString();
        }

        return $"[Chế độ mock — chưa có file .gguf]\n" +
               $"Bạn hỏi: \"{message}\"\n" +
               "Hãy đặt file model vào thư mục Models/. Bật Research / Deep Research trên UI khi có model.";
    }

    public void Dispose()
    {
        _loraAdapter?.Unload();
        _executor?.Context.Dispose();
        _weights?.Dispose();
        _semaphore.Dispose();
    }
}
