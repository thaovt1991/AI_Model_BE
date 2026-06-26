using AI_Model_BE.Models;
using AI_Model_BE.Services;
using Microsoft.AspNetCore.Mvc;

namespace AI_Model_BE.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AiController : ControllerBase
{
    private readonly MlPredictionService _mlService;
    private readonly LlamaChatService _llamaService;
    private readonly DocumentKnowledgeService _documentService;
    private readonly ChatMemoryService _chatMemory;
    private readonly ChatProfileSettingsService _profileSettings;
    private readonly ModelLearningService _modelLearning;
    private readonly LearningSettingsService _learningSettings;
    private readonly LottoForecastService _lottoForecast;
    private readonly MinhNgocScrapeSettingsService _minhNgocScrapeSettings;

    public AiController(
        MlPredictionService mlService,
        LlamaChatService llamaService,
        DocumentKnowledgeService documentService,
        ChatMemoryService chatMemory,
        ChatProfileSettingsService profileSettings,
        ModelLearningService modelLearning,
        LearningSettingsService learningSettings,
        LottoForecastService lottoForecast,
        MinhNgocScrapeSettingsService minhNgocScrapeSettings)
    {
        _mlService = mlService;
        _llamaService = llamaService;
        _documentService = documentService;
        _chatMemory = chatMemory;
        _profileSettings = profileSettings;
        _modelLearning = modelLearning;
        _learningSettings = learningSettings;
        _lottoForecast = lottoForecast;
        _minhNgocScrapeSettings = minhNgocScrapeSettings;
    }

    /// <summary>GET /api/ai/health — kiểm tra nhanh Backend đã chạy OK.</summary>
    [HttpGet("health")]
    [ProducesResponseType(typeof(HealthResponse), StatusCodes.Status200OK)]
    public ActionResult<HealthResponse> Health()
    {
        return Ok(new HealthResponse(
            Status: "ok",
            Service: "AI_Model_BE",
            Environment: Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production",
            UtcTime: DateTime.UtcNow));
    }

    [HttpPost("predict")]
    public ActionResult<PredictResponse> Predict([FromBody] PredictRequest request)
    {
        if (request.Size <= 0 || request.Bedrooms <= 0)
        {
            return BadRequest(new { message = "Diện tích và số phòng phải lớn hơn 0." });
        }

        var result = _mlService.Predict(request);
        return Ok(result);
    }

    /// <summary>GET /api/ai/lotto/games — danh mục loại xổ số và đài.</summary>
    [HttpGet("lotto/games")]
    public ActionResult<IReadOnlyList<LottoGameInfoDto>> GetLottoGames() =>
        Ok(_lottoForecast.GetGameCatalog());

    /// <summary>GET /api/ai/lotto/latest — kỳ quay gần nhất của đài.</summary>
    [HttpGet("lotto/latest")]
    public async Task<ActionResult<LottoLatestResponse>> GetLottoLatest(
        [FromQuery] string gameKind,
        [FromQuery] string? daiCode,
        CancellationToken cancellationToken)
    {
        try
        {
            var latest = await _lottoForecast.GetLatestForApiAsync(gameKind, daiCode, cancellationToken);
            return Ok(new LottoLatestResponse(gameKind, daiCode, latest));
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>GET /api/ai/lotto/records — lịch sử theo loại/đài (file mẫu).</summary>
    [HttpGet("lotto/records")]
    public async Task<ActionResult<IReadOnlyList<LotteryRecord>>> GetLottoRecords(
        [FromQuery] string gameKind,
        [FromQuery] string? daiCode,
        CancellationToken cancellationToken,
        [FromServices] LotteryRecordLoader loader)
    {
        try
        {
            var data = await loader.LoadDatasetAsync(gameKind, daiCode, cancellationToken);
            return Ok(data);
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>GET /api/ai/lotto/history — lịch sử Minh Ngọc theo ngày/đài.</summary>
    [HttpGet("lotto/history")]
    public async Task<ActionResult<LottoHistoryResponse>> GetLottoHistory(
        [FromQuery] string gameKind,
        [FromQuery] string? daiCode,
        [FromQuery] string? fromDate,
        [FromQuery] string? toDate,
        [FromQuery] string? date,
        CancellationToken cancellationToken)
    {
        try
        {
            DateTime? from = ParseViDate(fromDate);
            DateTime? to = ParseViDate(toDate);
            DateTime? single = ParseViDate(date);
            return Ok(await _lottoForecast.GetHistoryAsync(
                gameKind, daiCode, from, to, single, cancellationToken));
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>GET /api/ai/lotto/lo-gan — thống kê lô gan từ lịch sử.</summary>
    [HttpGet("lotto/lo-gan")]
    public async Task<ActionResult<LottoLoGanResponse>> GetLottoLoGan(
        [FromQuery] string gameKind,
        [FromQuery] string? daiCode,
        [FromQuery] int top = 30,
        CancellationToken cancellationToken = default)
    {
        try
        {
            return Ok(await _lottoForecast.GetLoGanAsync(gameKind, daiCode, top, cancellationToken));
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    private static DateTime? ParseViDate(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        string[] formats = ["dd/MM/yyyy", "yyyy-MM-dd", "dd-MM-yyyy"];
        foreach (var fmt in formats)
        {
            if (DateTime.TryParseExact(text, fmt, null, System.Globalization.DateTimeStyles.None, out var d))
            {
                return d;
            }
        }

        return DateTime.TryParse(text, out var parsed) ? parsed : null;
    }

    /// <summary>GET /api/ai/lotto/scrape-settings — cài đặt cào Minh Ngọc.</summary>
    [HttpGet("lotto/scrape-settings")]
    public ActionResult<MinhNgocScrapeSettingsResponse> GetMinhNgocScrapeSettings() =>
        Ok(_minhNgocScrapeSettings.GetResponse());

    /// <summary>PUT /api/ai/lotto/scrape-settings — cập nhật cài đặt cào (áp dụng ngay, lưu file).</summary>
    [HttpPut("lotto/scrape-settings")]
    public ActionResult<MinhNgocScrapeSettingsResponse> UpdateMinhNgocScrapeSettings(
        [FromBody] UpdateMinhNgocScrapeSettingsRequest request) =>
        Ok(_minhNgocScrapeSettings.Update(request));

    /// <summary>POST /api/ai/lotto/run — huấn luyện SSA và dự đoán.</summary>
    [HttpPost("lotto/run")]
    public async Task<ActionResult<LottoRunResponse>> RunLottoForecast(
        [FromBody] LottoRunRequest request,
        CancellationToken cancellationToken)
    {
        try
        {
            return Ok(await _lottoForecast.RunForecastAsync(request, cancellationToken));
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>
    /// API nhận file upload từ Frontend.
    ///
    /// IFormFile: ASP.NET Core tự bind file từ HTTP request multipart/form-data.
    /// Frontend gửi FormData với field tên "file" → Backend nhận qua tham số IFormFile file.
    ///
    /// RequestSizeLimit(20MB): giới hạn file tối đa 20 megabyte.
    /// </summary>
    [HttpPost("documents/upload")]
    [RequestSizeLimit(20 * 1024 * 1024)]
    public async Task<ActionResult<DocumentUploadResponse>> UploadDocument(
        IFormFile file,  // File user chọn trên UI — chưa lưu ổ đĩa, chỉ nằm trong request
        CancellationToken cancellationToken)
    {
        if (file is null || file.Length == 0)
        {
            return BadRequest(new { message = "Vui lòng chọn file để upload." });
        }

        try
        {
            // Chuyển file sang DocumentKnowledgeService.UploadAsync để ghi xuống ổ đĩa
            var result = await _documentService.UploadAsync(file, cancellationToken);
            return Ok(result);
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>GET /api/ai/documents — trả danh sách file đã lưu (đọc từ RAM, RAM load từ index.json).</summary>
    [HttpGet("documents")]
    public ActionResult<IReadOnlyList<DocumentInfo>> ListDocuments()
    {
        return Ok(_documentService.GetAll());
    }

    /// <summary>
    /// DELETE /api/ai/documents/{id} — xóa file khỏi ổ đĩa và RAM.
    /// Frontend gọi khi user bấm nút × trên danh sách tài liệu.
    /// </summary>
    [HttpDelete("documents/{id:guid}")]
    public IActionResult DeleteDocument(Guid id)
    {
        return _documentService.Delete(id) ? NoContent() : NotFound();
    }

    [HttpPost("chat")]
    public async Task<ActionResult<ChatResponse>> Chat(
        [FromBody] ChatRequest request,
        CancellationToken cancellationToken)
    {
        var reply = await _llamaService.ChatAsync(
            request.Message,
            request.DocumentIds,
            request.ConversationId,
            request.ProfileId,
            cancellationToken);
        return Ok(reply);
    }

    [HttpPost("chat/stream")]
    public async Task ChatStream(
        [FromBody] ChatRequest request,
        CancellationToken cancellationToken)
    {
        Response.ContentType = "text/plain; charset=utf-8";

        await foreach (var token in _llamaService.ChatStreamAsync(
                           request.Message,
                           request.DocumentIds,
                           request.ConversationId,
                           request.ProfileId,
                           cancellationToken))
        {
            await Response.WriteAsync(token, cancellationToken);
            await Response.Body.FlushAsync(cancellationToken);
        }
    }

    /// <summary>GET /api/ai/learning/settings — cài đặt học model + trạng thái (cho UI Thiết lập).</summary>
    [HttpGet("learning/settings")]
    public ActionResult<LearningSettingsResponse> GetLearningSettings()
    {
        return Ok(_modelLearning.GetSettingsResponse());
    }

    /// <summary>PUT /api/ai/learning/settings — bật/tắt học vào model hoặc thu thập dữ liệu.</summary>
    [HttpPut("learning/settings")]
    public ActionResult<LearningSettingsResponse> UpdateLearningSettings(
        [FromBody] UpdateLearningSettingsRequest request)
    {
        _learningSettings.Update(request.Enabled, request.CollectData);
        _modelLearning.RefreshStatus();
        return Ok(_modelLearning.GetSettingsResponse());
    }

    /// <summary>
    /// GET /api/ai/learning/status — xem học ngầm đang ở giai đoạn nào.
    /// totalSamples = số cặp hỏi/đáp đã gom; status = Collecting khi chưa bật train.
    /// </summary>
    [HttpGet("learning/status")]
    public ActionResult<LearningStatusResponse> GetLearningStatus()
    {
        return Ok(_modelLearning.GetStatus());
    }

    /// <summary>
    /// POST /api/ai/learning/train — ép train ngay (cần Learning:Enabled = true trong appsettings).
    /// Thường không cần gọi tay — BackgroundModelLearningHostedService tự train khi máy rảnh.
    /// </summary>
    [HttpPost("learning/train")]
    public async Task<ActionResult<LearningTrainResponse>> TrainLearning(CancellationToken cancellationToken)
    {
        var result = await _modelLearning.TryTrainInBackgroundAsync(cancellationToken);
        return result.Started ? Ok(result) : BadRequest(result);
    }

    /// <summary>Lấy cài đặt profile (tên AI, ...).</summary>
    [HttpGet("chat/profile")]
    public ActionResult<ChatProfileSettingsResponse> GetChatProfile([FromQuery] string? profileId)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            return BadRequest(new { message = "Thiếu profileId." });
        }

        return Ok(_profileSettings.Get(profileId));
    }

    /// <summary>Cập nhật tên AI cho profile.</summary>
    [HttpPut("chat/profile")]
    public ActionResult<ChatProfileSettingsResponse> UpdateChatProfile(
        [FromQuery] string? profileId,
        [FromBody] UpdateChatProfileSettingsRequest request)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            return BadRequest(new { message = "Thiếu profileId." });
        }

        try
        {
            return Ok(_profileSettings.Update(profileId, request.AiName));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    /// <summary>Lấy toàn bộ lịch sử đã lưu (khôi phục UI sau khi mở lại trang).</summary>
    [HttpGet("chat/memory")]
    public ActionResult<IReadOnlyList<ChatTurn>> GetChatMemory(
        [FromQuery] string? profileId,
        [FromQuery] string? conversationId)
    {
        var turns = _chatMemory.GetAllStored(profileId, conversationId);
        return Ok(turns);
    }

    /// <summary>Xóa toàn bộ trí nhớ hội thoại của profile (mọi cuộc chat).</summary>
    [HttpDelete("chat/memory")]
    public IActionResult ClearChatMemory([FromQuery] string? profileId)
    {
        return _chatMemory.ClearProfile(profileId) ? NoContent() : NotFound();
    }
}
