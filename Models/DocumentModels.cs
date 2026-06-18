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
}
