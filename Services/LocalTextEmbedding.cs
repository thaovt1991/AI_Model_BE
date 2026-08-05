using System.Text;
using System.Text.RegularExpressions;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// EMBEDDING LOCAL (không cần model AI riêng, không cần GPU)
/// =====================================================================
///
/// Ý tưởng:
/// - Biến mỗi đoạn văn bản thành 1 "vector" số thực cố định chiều (ví dụ 384 số).
/// - Hai đoạn gần nghĩa → 2 vector gần nhau (cosine similarity cao).
/// - Dùng Feature Hashing: băm token/bigram vào các ô của vector (nhanh, không lưu từ điển).
///
/// Đây KHÔNG phải embedding transformer (như OpenAI/sentence-transformers),
/// nhưng tốt hơn hẳn đếm từ khóa thô, chạy 100% offline trong .NET.
///
/// Kết hợp với BM25 trong DocumentKnowledgeService = Hybrid RAG.
/// </summary>
public static class LocalTextEmbedding
{
    /// <summary>Số chiều vector. Càng lớn càng chi tiết nhưng tốn RAM hơn.</summary>
    public const int Dimensions = 384;

    private static readonly Regex TokenRegex = new(
        @"[\p{L}\p{N}]{2,}",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Biến text → vector đã chuẩn hóa L2 (độ dài = 1).
    /// Cosine(a,b) khi đã L2-norm = tích vô hướng a·b (nhanh).
    /// </summary>
    public static float[] Embed(string? text)
    {
        var vector = new float[Dimensions];
        if (string.IsNullOrWhiteSpace(text))
        {
            return vector;
        }

        var tokens = Tokenize(text);
        if (tokens.Count == 0)
        {
            return vector;
        }

        // --- Unigram: từng từ ---
        foreach (var token in tokens)
        {
            Accumulate(vector, token, weight: 1f);
        }

        // --- Bigram: cặp từ liền kề (bắt được cụm "học máy", "giá vàng"...) ---
        for (var i = 0; i < tokens.Count - 1; i++)
        {
            Accumulate(vector, tokens[i] + "_" + tokens[i + 1], weight: 0.7f);
        }

        // --- Character trigram: giúp khớp lỗi gõ / biến thể tiếng Việt nhẹ ---
        foreach (var token in tokens)
        {
            if (token.Length < 3)
            {
                continue;
            }

            for (var i = 0; i <= token.Length - 3; i++)
            {
                Accumulate(vector, "#" + token.Substring(i, 3), weight: 0.25f);
            }
        }

        L2Normalize(vector);
        return vector;
    }

    /// <summary>
    /// Cosine similarity ∈ [-1, 1]. Với vector đã L2-norm → chỉ cần dot product.
    /// 1 = giống hệt hướng, 0 = không liên quan, âm = ngược hướng (hiếm với text dương).
    /// </summary>
    public static float Cosine(float[] a, float[] b)
    {
        if (a.Length != b.Length || a.Length == 0)
        {
            return 0f;
        }

        double sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return (float)sum;
    }

    /// <summary>BM25 cổ điển — đo mức "từ khóa quan trọng" xuất hiện trong chunk.</summary>
    public static float Bm25(
        IReadOnlyList<string> queryTokens,
        IReadOnlyDictionary<string, int> docTermFreq,
        int docLength,
        double avgDocLength,
        IReadOnlyDictionary<string, int> docFreq,
        int totalDocs,
        double k1 = 1.5,
        double b = 0.75)
    {
        if (queryTokens.Count == 0 || totalDocs <= 0 || avgDocLength <= 0)
        {
            return 0f;
        }

        double score = 0;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var term in queryTokens)
        {
            if (!seen.Add(term))
            {
                continue; // mỗi term chỉ tính 1 lần trong query
            }

            if (!docTermFreq.TryGetValue(term, out var tf) || tf <= 0)
            {
                continue;
            }

            docFreq.TryGetValue(term, out var df);
            // IDF Robertson: log(1 + (N - df + 0.5) / (df + 0.5))
            var idf = Math.Log(1.0 + (totalDocs - df + 0.5) / (df + 0.5));
            var tfNorm = (tf * (k1 + 1)) /
                         (tf + k1 * (1 - b + b * docLength / avgDocLength));
            score += idf * tfNorm;
        }

        return (float)score;
    }

    public static List<string> Tokenize(string text)
    {
        var list = new List<string>();
        foreach (Match m in TokenRegex.Matches(text.ToLowerInvariant()))
        {
            list.Add(m.Value);
        }

        return list;
    }

    public static Dictionary<string, int> BuildTermFrequency(IReadOnlyList<string> tokens)
    {
        var tf = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var t in tokens)
        {
            tf[t] = tf.TryGetValue(t, out var c) ? c + 1 : 1;
        }

        return tf;
    }

    /// <summary>
    /// Feature hashing: hash(token) → index trong [0, Dimensions).
    /// Dùng dấu +/- theo bit để giảm collision bias (signed hashing).
    /// </summary>
    private static void Accumulate(float[] vector, string feature, float weight)
    {
        var hash = StableHash(feature);
        var index = (hash & 0x7FFFFFFF) % Dimensions;
        var sign = (hash & 1) == 0 ? 1f : -1f;
        vector[index] += sign * weight;
    }

    /// <summary>Hash ổn định giữa các lần chạy (không dùng string.GetHashCode — đổi theo process).</summary>
    private static int StableHash(string value)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (var c in value)
            {
                hash ^= c;
                hash *= 16777619;
            }

            return (int)hash;
        }
    }

    private static void L2Normalize(float[] vector)
    {
        double sumSq = 0;
        foreach (var v in vector)
        {
            sumSq += v * v;
        }

        if (sumSq <= 1e-12)
        {
            return;
        }

        var inv = (float)(1.0 / Math.Sqrt(sumSq));
        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] *= inv;
        }
    }

    /// <summary>Nén vector thưa để lưu JSON nhẹ hơn (chỉ giữ ô khác 0 đáng kể).</summary>
    public static Dictionary<string, float> ToSparse(float[] vector, float epsilon = 1e-4f)
    {
        var sparse = new Dictionary<string, float>();
        for (var i = 0; i < vector.Length; i++)
        {
            if (Math.Abs(vector[i]) >= epsilon)
            {
                sparse[i.ToString()] = vector[i];
            }
        }

        return sparse;
    }

    public static float[] FromSparse(Dictionary<string, float>? sparse)
    {
        var vector = new float[Dimensions];
        if (sparse is null || sparse.Count == 0)
        {
            return vector;
        }

        foreach (var (key, value) in sparse)
        {
            if (int.TryParse(key, out var index) && index >= 0 && index < Dimensions)
            {
                vector[index] = value;
            }
        }

        return vector;
    }
}
