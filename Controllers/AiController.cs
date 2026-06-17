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

    public AiController(MlPredictionService mlService, LlamaChatService llamaService)
    {
        _mlService = mlService;
        _llamaService = llamaService;
    }

    /// <summary>POST /api/ai/predict — FE gửi diện tích & số phòng, BE trả giá dự đoán.</summary>
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

    /// <summary>POST /api/ai/chat — FE gửi câu hỏi, BE trả chuỗi phản hồi đầy đủ.</summary>
    [HttpPost("chat")]
    public async Task<ActionResult<ChatResponse>> Chat(
        [FromBody] ChatRequest request,
        CancellationToken cancellationToken)
    {
        var reply = await _llamaService.ChatAsync(request.Message, cancellationToken);
        return Ok(reply);
    }

    /// <summary>POST /api/ai/chat/stream — stream token từng phần dạng text/plain cho Frontend đọc realtime.</summary>
    [HttpPost("chat/stream")]
    public async Task ChatStream(
        [FromBody] ChatRequest request,
        CancellationToken cancellationToken)
    {
        Response.ContentType = "text/plain; charset=utf-8";

        await foreach (var token in _llamaService.ChatStreamAsync(request.Message, cancellationToken))
        {
            await Response.WriteAsync(token, cancellationToken);
            await Response.Body.FlushAsync(cancellationToken);
        }
    }
}
