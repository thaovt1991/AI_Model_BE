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

    public AiController(
        MlPredictionService mlService,
        LlamaChatService llamaService,
        DocumentKnowledgeService documentService)
    {
        _mlService = mlService;
        _llamaService = llamaService;
        _documentService = documentService;
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
        var reply = await _llamaService.ChatAsync(request.Message, request.DocumentIds, cancellationToken);
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
                           cancellationToken))
        {
            await Response.WriteAsync(token, cancellationToken);
            await Response.Body.FlushAsync(cancellationToken);
        }
    }
}
