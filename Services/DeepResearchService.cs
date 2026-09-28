using System.Text;
using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// DEEP RESEARCH — nghiên cứu sâu nhiều góc độ (không chỉ 1 lần search)
/// =====================================================================
///
/// Quy trình:
///  1) Sinh nhiều truy vấn phụ từ câu hỏi gốc (heuristic tiếng Việt/Anh).
///  2) Gọi WebResearchService cho từng truy vấn.
///  3) Gộp + loại trùng nguồn, xếp hạng theo độ phủ.
///  4) Đánh số citation [1]..[N] để LLM và UI dùng chung.
///
/// Khác research thường: rộng hơn, tốn thời gian/mạng hơn, câu trả lời giàu nguồn hơn.
/// </summary>
public sealed class DeepResearchService
{
    private readonly WebResearchService _webResearch;
    private readonly IConfiguration _configuration;
    private readonly ILogger<DeepResearchService> _logger;

    public DeepResearchService(
        WebResearchService webResearch,
        IConfiguration configuration,
        ILogger<DeepResearchService> logger)
    {
        _webResearch = webResearch;
        _configuration = configuration;
        _logger = logger;
    }

    public bool IsEnabled => _configuration.GetValue("DeepResearch:Enabled", true);

    /// <summary>
    /// Chạy deep research. Trả về context block + citations đã đánh số.
    /// </summary>
    public async Task<DeepResearchResult> ResearchAsync(
        string userQuestion,
        CancellationToken cancellationToken = default)
    {
        if (!IsEnabled || string.IsNullOrWhiteSpace(userQuestion))
        {
            return DeepResearchResult.Empty;
        }

        var maxSubQueries = _configuration.GetValue("DeepResearch:MaxSubQueries", 4);
        var maxSources = _configuration.GetValue("DeepResearch:MaxSources", 8);
        var maxContextChars = _configuration.GetValue("DeepResearch:MaxContextChars", 4500);

        // Bước 1: sinh truy vấn phụ
        var subQueries = ExpandQueries(userQuestion.Trim(), maxSubQueries);
        _logger.LogInformation(
            "Deep Research: {Count} truy vấn phụ cho \"{Q}\"",
            subQueries.Count,
            userQuestion);

        // Bước 2: tối đa 2 truy vấn cùng lúc — nhanh hơn tuần tự, vẫn nhẹ với DuckDuckGo
        var merged = new List<WebFinding>();
        await Parallel.ForEachAsync(
            subQueries,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = 2,
                CancellationToken = cancellationToken
            },
            async (q, ct) =>
            {
                try
                {
                    var partial = await _webResearch.ResearchAsync(q, ct);
                    if (!partial.HasResults)
                    {
                        return;
                    }

                    lock (merged)
                    {
                        merged.AddRange(partial.Findings);
                    }
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Deep Research lỗi ở sub-query: {Q}", q);
                }
            });

        // Bước 3: dedupe theo URL/title
        var unique = Deduplicate(merged).Take(maxSources).ToList();
        if (unique.Count == 0)
        {
            return DeepResearchResult.Empty;
        }

        // Bước 4: đánh số citation + ghép context
        var citations = new List<CitationSource>();
        var sb = new StringBuilder();
        sb.AppendLine("=== DEEP RESEARCH ===");
        sb.AppendLine($"Câu hỏi: {userQuestion}");
        sb.AppendLine("Câu đầu là kết luận. Trích [n] một lần. Không bịa số liệu ngoài các đoạn dưới.");
        sb.AppendLine();

        for (var i = 0; i < unique.Count; i++)
        {
            var f = unique[i];
            var id = i + 1;
            var snippet = string.IsNullOrWhiteSpace(f.PageExcerpt) ? f.Snippet : f.PageExcerpt;
            if (snippet.Length > 500)
            {
                snippet = snippet[..499] + "…";
            }

            citations.Add(new CitationSource
            {
                Id = id,
                Kind = "web",
                Title = f.Title,
                Url = f.Url,
                Snippet = Truncate(f.Snippet, 220),
                Score = Math.Round(1.0 - i * 0.05, 2),
                SourceLabel = f.Source
            });

            var block = new StringBuilder();
            block.AppendLine($"[{id}] {f.Title} ({f.Source})");
            if (!string.IsNullOrWhiteSpace(f.Url))
            {
                block.AppendLine($"URL: {f.Url}");
            }

            block.AppendLine($"Nội dung: {snippet}");
            block.AppendLine();

            if (sb.Length + block.Length > maxContextChars)
            {
                break;
            }

            sb.Append(block);
        }

        sb.AppendLine("=== HẾT DEEP RESEARCH ===");
        sb.AppendLine("Tổng hợp khách quan, nêu điểm đồng thuận / mâu thuẫn giữa các nguồn. Không bịa số liệu.");

        return new DeepResearchResult(true, citations, sb.ToString().Trim(), subQueries);
    }

    /// <summary>
    /// Sinh truy vấn phụ bằng heuristic (không gọi LLM — tránh vòng chờ kép).
    /// Ví dụ: "giá vàng hôm nay" → thêm "giá vàng Việt Nam", "giá vàng SJC mới nhất"...
    /// </summary>
    public static IReadOnlyList<string> ExpandQueries(string question, int maxCount)
    {
        var list = new List<string> { question };
        var q = question.Trim();

        void Add(string s)
        {
            if (list.Count >= maxCount)
            {
                return;
            }

            s = s.Trim();
            if (s.Length < 3)
            {
                return;
            }

            if (list.Any(x => string.Equals(x, s, StringComparison.OrdinalIgnoreCase)))
            {
                return;
            }

            list.Add(s);
        }

        Add($"{q} tổng quan");
        Add($"{q} mới nhất");
        Add($"{q} Việt Nam");

        if (q.Contains("giá", StringComparison.OrdinalIgnoreCase) ||
            q.Contains("price", StringComparison.OrdinalIgnoreCase))
        {
            Add($"{q} hôm nay cập nhật");
        }

        if (q.Contains("tin", StringComparison.OrdinalIgnoreCase) ||
            q.Contains("news", StringComparison.OrdinalIgnoreCase))
        {
            Add($"{q} thời sự");
        }

        if (q.Contains("là gì", StringComparison.OrdinalIgnoreCase) ||
            q.Contains("what is", StringComparison.OrdinalIgnoreCase))
        {
            Add($"{q} định nghĩa giải thích");
        }

        if (q.Contains("so sánh", StringComparison.OrdinalIgnoreCase) ||
            q.Contains("vs", StringComparison.OrdinalIgnoreCase))
        {
            Add($"{q} ưu nhược điểm");
        }

        // Biến thể bỏ dấu hỏi / rút gọn
        var shortQ = q.TrimEnd('?', '？', '.', '!');
        if (!string.Equals(shortQ, q, StringComparison.Ordinal))
        {
            Add(shortQ);
        }

        return list.Take(Math.Max(1, maxCount)).ToList();
    }

    private static List<WebFinding> Deduplicate(List<WebFinding> findings)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new List<WebFinding>();
        foreach (var f in findings)
        {
            var key = !string.IsNullOrWhiteSpace(f.Url)
                ? f.Url!
                : $"{f.Title}|{f.Snippet}";
            if (!seen.Add(key))
            {
                continue;
            }

            result.Add(f);
        }

        return result;
    }

    private static string Truncate(string text, int max)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= max)
        {
            return text;
        }

        return text[..(max - 1)].TrimEnd() + "…";
    }
}

public sealed record DeepResearchResult(
    bool HasResults,
    IReadOnlyList<CitationSource> Citations,
    string ContextBlock,
    IReadOnlyList<string> SubQueries)
{
    public static DeepResearchResult Empty { get; } =
        new(false, Array.Empty<CitationSource>(), string.Empty, Array.Empty<string>());
}

/// <summary>
/// Helper đóng gói / giải mã meta trong stream text/plain.
/// </summary>
public static class ChatStreamMetaCodec
{
    public const string Begin = "[[AI_META]]";
    public const string End = "[[/AI_META]]";

    public static string Encode(ChatStreamMeta meta)
    {
        // camelCase để Frontend parse thẳng không cần map PascalCase
        var json = JsonSerializer.Serialize(meta, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        return $"{Begin}{json}{End}\n";
    }

    public static string Status(string phase, string message) =>
        Encode(new ChatStreamMeta { Type = "status", Phase = phase, Message = message });

    public static string Citations(IReadOnlyList<CitationSource> items) =>
        Encode(new ChatStreamMeta
        {
            Type = "citations",
            Phase = "sources",
            Items = items.ToList()
        });
}
