namespace AI_Model_BE.Models;

public record DocumentUploadResponse(
    Guid Id,
    string FileName,
    long SizeBytes,
    int ChunkCount,
    string Preview);

public record DocumentInfo(
    Guid Id,
    string FileName,
    long SizeBytes,
    int ChunkCount,
    string Preview,
    DateTime UploadedAt);

public sealed class DocumentChunk
{
    public Guid DocumentId { get; init; }
    public string FileName { get; init; } = string.Empty;
    public int Index { get; init; }
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Embedding thưa đã tính sẵn lúc upload (LocalTextEmbedding).
    /// Key = chỉ số chiều dạng chuỗi, Value = trọng số.
    /// Null với tài liệu cũ → sẽ embed lại lúc load / retrieve.
    /// </summary>
    public Dictionary<string, float>? EmbeddingSparse { get; set; }

    /// <summary>Không serialize — vector đầy đủ giữ trong RAM để so cosine nhanh.</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public float[]? Embedding { get; set; }
}

/// <summary>Kết quả Hybrid RAG: context đưa vào LLM + danh sách citation cho UI.</summary>
public sealed class DocumentRetrievalResult
{
    public string ContextBlock { get; init; } = string.Empty;
    public IReadOnlyList<CitationSource> Citations { get; init; } = [];
    public bool HasResults => Citations.Count > 0 || !string.IsNullOrWhiteSpace(ContextBlock);
}

