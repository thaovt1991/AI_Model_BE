using System.Text.Json.Serialization;

namespace AI_Model_BE.Models;

/// <summary>
/// Một nguồn trích dẫn (citation) — tài liệu nội bộ hoặc trang web.
/// Frontend dùng để hiện danh sách nguồn dưới bubble trả lời.
/// </summary>
public sealed class CitationSource
{
    /// <summary>Số thứ tự hiển thị [1], [2], ... — model được nhắc nêu số này khi trích.</summary>
    public int Id { get; init; }

    /// <summary>"document" | "web"</summary>
    public string Kind { get; init; } = "web";

    public string Title { get; init; } = string.Empty;

    /// <summary>URL trang web; null nếu là file nội bộ.</summary>
    public string? Url { get; init; }

    /// <summary>Tên file tài liệu (khi Kind=document).</summary>
    public string? FileName { get; init; }

    /// <summary>Đoạn / index chunk trong file (1-based cho người đọc).</summary>
    public int? ChunkIndex { get; init; }

    /// <summary>Đoạn trích ngắn để user xem nhanh.</summary>
    public string Snippet { get; init; } = string.Empty;

    /// <summary>Điểm liên quan 0..1 (càng cao càng khớp câu hỏi).</summary>
    public double Score { get; init; }

    public string SourceLabel { get; init; } = string.Empty;
}

/// <summary>
/// Gói meta gửi xen kẽ trong stream chat.
/// Frontend bắt chuỗi [[AI_META]]...[[/AI_META]] rồi parse JSON này.
/// </summary>
public sealed class ChatStreamMeta
{
    /// <summary>"status" | "citations" | "phase"</summary>
    [JsonPropertyName("type")]
    public string Type { get; init; } = "status";

    /// <summary>research | deep_research | rag | thinking | done</summary>
    [JsonPropertyName("phase")]
    public string? Phase { get; init; }

    [JsonPropertyName("message")]
    public string? Message { get; init; }

    [JsonPropertyName("items")]
    public List<CitationSource>? Items { get; init; }
}
