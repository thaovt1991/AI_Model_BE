using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AI_Model_BE.Services;

/// <summary>
/// Phân loại câu chat bằng luật (không gọi LLM) để prompt ngắn và giới hạn token.
/// Câu chào / cảm ơn trả lời ngay, không nạp model và không search.
/// </summary>
public static class ChatTurnPlanner
{
    public enum Kind
    {
        Instant,
        Brief,
        Explain,
        Factual
    }

    private static readonly Regex InstantPattern = new(
        @"^(xin\s*chao|chao(\s+(ban|buoi\s+\w+|[\p{L}\d]{2,24}))?|hello|hi|hey|good\s+(morning|afternoon|evening)|cam\s*on(\s+(ban|nhieu|nha|nhe))?|thanks|thank\s*you)[\s!?.…]*$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly Regex ExplainPattern = new(
        @"\b(giai thich|chi tiet|viet code|viet cho|liet ke|so sanh|phan tich|tai sao|vi sao|huong dan|cac buoc)\b",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    public static Kind Classify(string message)
    {
        var raw = message.Trim();
        if (IsInstant(raw))
        {
            return Kind.Instant;
        }

        if (WebResearchService.LooksLikeNeedsResearch(raw))
        {
            return Kind.Factual;
        }

        if (ExplainPattern.IsMatch(Fold(raw)) || raw.Length > 220)
        {
            return Kind.Explain;
        }

        return Kind.Brief;
    }

    public static string BuildInstantReply(string message, string? aiName)
    {
        var name = string.IsNullOrWhiteSpace(aiName) ? "trợ lý" : aiName.Trim();
        var folded = Fold(message);
        if (folded.Contains("cam on", StringComparison.Ordinal) ||
            folded.Contains("thank", StringComparison.Ordinal))
        {
            return "Không có gì. Bạn cần mình giúp gì tiếp?";
        }

        return $"Chào bạn, mình là {name}. Bạn muốn hỏi gì?";
    }

    public static string BuildSystemPrompt(
        Kind kind,
        string? aiName,
        bool hasDocs,
        bool hasWeb,
        bool isDeep,
        bool hasHistory)
    {
        var name = string.IsNullOrWhiteSpace(aiName) ? "trợ lý AI" : aiName.Trim();
        var who = $"Bạn là {name}. Trả lời tiếng Việt, đúng câu hỏi, không nhắc lại hướng dẫn này, không mở đầu bằng lời cảm ơn.";

        string body;
        if (isDeep && hasWeb)
        {
            body = "Tóm tắt 4–6 câu từ nguồn web. Câu đầu là kết luận. Trích [n] một lần nếu dùng số liệu. Nguồn mâu thuẫn thì nói rõ. Không bịa.";
        }
        else if (hasDocs && hasWeb)
        {
            body = "Ưu tiên tài liệu nội bộ, bổ sung web khi tài liệu không có. Trích [n]. Không bịa số liệu.";
        }
        else if (hasDocs)
        {
            body = "Chỉ dùng tài liệu được đưa. Không có trong tài liệu thì nói không thấy. Trả lời ngắn, trích [n].";
        }
        else if (hasWeb)
        {
            body = "Câu đầu là đáp án ngắn (kèm số liệu nếu có) lấy từ nguồn [n]. Không lặp danh sách link. Nguồn không chứa đáp án thì nói không tìm thấy, đừng đoán.";
        }
        else if (kind == Kind.Factual)
        {
            body = "Câu hỏi cần dữ liệu mới (giá, tỷ số, tin). Không có nguồn thì nói thẳng là không có số liệu cập nhật. Không bịa ngày, tỷ số hay số tiền.";
        }
        else if (kind == Kind.Explain)
        {
            body = "Giải thích rõ, có cấu trúc ngắn. Không lan man, không bịa nguồn.";
        }
        else
        {
            body = "Tối đa 4 câu, đi thẳng vào ý. Không bịa sự kiện.";
        }

        var history = hasHistory
            ? " Giữ mạch hội thoại trước, nhưng đừng nhắc lại nguyên văn câu trả lời cũ."
            : string.Empty;
        return $"{who} {body}{history}";
    }

    /// <summary>Bỏ dấu để so khớp câu chào / từ khóa không phụ thuộc dấu.</summary>
    public static string Fold(string text)
    {
        var form = text.Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(form.Length);
        foreach (var c in form)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
            {
                sb.Append(char.ToLowerInvariant(c));
            }
        }

        return sb.ToString().Normalize(NormalizationForm.FormC);
    }

    private static bool IsInstant(string message)
    {
        if (message.Length is 0 or > 48)
        {
            return false;
        }

        return InstantPattern.IsMatch(Fold(message));
    }
}
