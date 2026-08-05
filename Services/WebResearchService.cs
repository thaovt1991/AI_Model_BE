using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using AI_Model_BE.Models;
using HtmlAgilityPack;

namespace AI_Model_BE.Services;

/// <summary>
/// Research dữ liệu mạng (không cần API key):
/// DuckDuckGo Instant Answer + HTML search + Wikipedia + đọc tóm tắt trang nguồn.
/// </summary>
public sealed class WebResearchService
{
    public const string HttpClientName = nameof(WebResearchService);

    private static readonly Regex NeedsResearchRegex = new(
        @"\b(tin\s*tức|thời\s*sự|hôm\s*nay|hiện\s*nay|mới\s*nhất|cập\s*nhật|" +
        @"năm\s*20\d{2}|202[4-9]|203\d|giá\s*(vàng|bitcoin|btc|usd|đô|xăng)|" +
        @"thời\s*tiết|tỷ\s*giá|chứng\s*khoán|bóng\s*đá|kết\s*quả|" +
        @"who\s+is|what\s+is|latest|news|current|today|research|" +
        @"tìm\s*kiếm|tra\s*cứu|nghiên\s*cứu|search|web|internet|mạng)\b",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly Regex StripTagsRegex = new("<[^>]+>", RegexOptions.Compiled);
    private static readonly Regex MultiSpaceRegex = new(@"\s{2,}", RegexOptions.Compiled);

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly IConfiguration _configuration;
    private readonly ILogger<WebResearchService> _logger;

    public WebResearchService(
        IHttpClientFactory httpClientFactory,
        IConfiguration configuration,
        ILogger<WebResearchService> logger)
    {
        _httpClientFactory = httpClientFactory;
        _configuration = configuration;
        _logger = logger;
    }

    public bool IsEnabled => _configuration.GetValue("WebResearch:Enabled", true);

    public bool AutoDetect => _configuration.GetValue("WebResearch:AutoDetect", true);

    /// <summary>
    /// Quyết định có research mạng hay không:
    /// - enableWebSearch=true → luôn research (nếu Enabled)
    /// - enableWebSearch=null → AutoDetect theo từ khóa
    /// - enableWebSearch=false → tắt
    /// </summary>
    public bool ShouldResearch(string userMessage, bool? enableWebSearch)
    {
        if (!IsEnabled)
        {
            return false;
        }

        if (enableWebSearch == false)
        {
            return false;
        }

        if (enableWebSearch == true)
        {
            return true;
        }

        return AutoDetect && LooksLikeNeedsResearch(userMessage);
    }

    public static bool LooksLikeNeedsResearch(string message)
    {
        if (string.IsNullOrWhiteSpace(message) || message.Length < 4)
        {
            return false;
        }

        return NeedsResearchRegex.IsMatch(message);
    }

    public async Task<WebResearchResult> ResearchAsync(
        string query,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            return WebResearchResult.Empty;
        }

        var maxResults = _configuration.GetValue("WebResearch:MaxResults", 5);
        var maxSnippetChars = _configuration.GetValue("WebResearch:MaxSnippetChars", 400);
        var maxContextChars = _configuration.GetValue("WebResearch:MaxContextChars", 3200);
        var fetchPages = _configuration.GetValue("WebResearch:FetchPageContent", true);
        var maxPages = _configuration.GetValue("WebResearch:MaxPagesToFetch", 2);

        var findings = new List<WebFinding>();
        var q = query.Trim();

        try
        {
            findings.AddRange(await SearchDuckDuckGoInstantAsync(q, maxSnippetChars, cancellationToken));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "DuckDuckGo Instant Answer lỗi cho query: {Query}", q);
        }

        try
        {
            findings.AddRange(await SearchDuckDuckGoHtmlAsync(q, maxResults, maxSnippetChars, cancellationToken));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "DuckDuckGo HTML search lỗi cho query: {Query}", q);
        }

        try
        {
            findings.AddRange(await SearchWikipediaAsync(q, maxResults: 2, maxSnippetChars, cancellationToken));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Wikipedia search lỗi cho query: {Query}", q);
        }

        findings = Deduplicate(findings).Take(maxResults).ToList();

        if (fetchPages && findings.Count > 0)
        {
            var pageTargets = findings
                .Where(f => !string.IsNullOrWhiteSpace(f.Url) && IsHttpUrl(f.Url!))
                .Take(maxPages)
                .ToList();

            foreach (var item in pageTargets)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    var pageText = await FetchPageSnippetAsync(item.Url!, maxSnippetChars * 2, cancellationToken);
                    if (!string.IsNullOrWhiteSpace(pageText))
                    {
                        item.PageExcerpt = pageText;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Không đọc được trang {Url}", item.Url);
                }
            }
        }

        if (findings.Count == 0)
        {
            _logger.LogInformation("Web research không có kết quả cho: {Query}", q);
            return WebResearchResult.Empty;
        }

        var context = BuildContextBlock(q, findings, maxContextChars);
        _logger.LogInformation(
            "Web research xong: {Count} nguồn cho query \"{Query}\" (context={Chars} chars)",
            findings.Count,
            q,
            context.Length);

        return new WebResearchResult(true, findings, context);
    }

    private async Task<List<WebFinding>> SearchDuckDuckGoInstantAsync(
        string query,
        int maxSnippetChars,
        CancellationToken cancellationToken)
    {
        var client = _httpClientFactory.CreateClient(HttpClientName);
        var url =
            $"https://api.duckduckgo.com/?q={Uri.EscapeDataString(query)}&format=json&no_html=1&skip_disambig=1";

        using var response = await client.GetAsync(url, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return [];
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var root = doc.RootElement;
        var list = new List<WebFinding>();

        var abstractText = root.TryGetProperty("AbstractText", out var abs) ? abs.GetString() : null;
        var abstractUrl = root.TryGetProperty("AbstractURL", out var absUrl) ? absUrl.GetString() : null;
        var heading = root.TryGetProperty("Heading", out var h) ? h.GetString() : null;

        if (!string.IsNullOrWhiteSpace(abstractText))
        {
            list.Add(new WebFinding(
                heading ?? "DuckDuckGo",
                Truncate(CleanText(abstractText), maxSnippetChars),
                abstractUrl,
                "DuckDuckGo"));
        }

        if (root.TryGetProperty("RelatedTopics", out var related) && related.ValueKind == JsonValueKind.Array)
        {
            foreach (var topic in related.EnumerateArray())
            {
                if (list.Count >= 3)
                {
                    break;
                }

                if (topic.TryGetProperty("Text", out var textProp))
                {
                    var text = textProp.GetString();
                    var link = topic.TryGetProperty("FirstURL", out var firstUrl)
                        ? firstUrl.GetString()
                        : null;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        list.Add(new WebFinding(
                            Truncate(CleanText(text), 80),
                            Truncate(CleanText(text), maxSnippetChars),
                            link,
                            "DuckDuckGo"));
                    }
                }
            }
        }

        return list;
    }

    private async Task<List<WebFinding>> SearchDuckDuckGoHtmlAsync(
        string query,
        int maxResults,
        int maxSnippetChars,
        CancellationToken cancellationToken)
    {
        var client = _httpClientFactory.CreateClient(HttpClientName);
        var url = $"https://html.duckduckgo.com/html/?q={Uri.EscapeDataString(query)}";

        using var response = await client.GetAsync(url, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return [];
        }

        var html = await response.Content.ReadAsStringAsync(cancellationToken);
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        var results = new List<WebFinding>();
        var nodes = doc.DocumentNode.SelectNodes("//div[contains(@class,'result')]")
                    ?? doc.DocumentNode.SelectNodes("//div[contains(@class,'web-result')]");

        if (nodes is null)
        {
            return results;
        }

        foreach (var node in nodes)
        {
            if (results.Count >= maxResults)
            {
                break;
            }

            var titleNode = node.SelectSingleNode(".//a[contains(@class,'result__a')]")
                            ?? node.SelectSingleNode(".//a[@href]");
            var snippetNode = node.SelectSingleNode(".//*[contains(@class,'result__snippet')]")
                              ?? node.SelectSingleNode(".//td[contains(@class,'result-snippet')]");

            var title = CleanText(titleNode?.InnerText ?? string.Empty);
            var href = titleNode?.GetAttributeValue("href", null);
            var snippet = CleanText(snippetNode?.InnerText ?? string.Empty);

            href = NormalizeDuckDuckGoUrl(href);

            if (string.IsNullOrWhiteSpace(title) && string.IsNullOrWhiteSpace(snippet))
            {
                continue;
            }

            results.Add(new WebFinding(
                string.IsNullOrWhiteSpace(title) ? "Kết quả web" : Truncate(title, 120),
                Truncate(snippet, maxSnippetChars),
                href,
                "DuckDuckGo"));
        }

        return results;
    }

    private async Task<List<WebFinding>> SearchWikipediaAsync(
        string query,
        int maxResults,
        int maxSnippetChars,
        CancellationToken cancellationToken)
    {
        var client = _httpClientFactory.CreateClient(HttpClientName);
        var list = new List<WebFinding>();

        // Ưu tiên Wikipedia tiếng Việt, fallback English
        foreach (var lang in new[] { "vi", "en" })
        {
            if (list.Count >= maxResults)
            {
                break;
            }

            var searchUrl =
                $"https://{lang}.wikipedia.org/w/api.php?action=opensearch&search={Uri.EscapeDataString(query)}" +
                $"&limit={maxResults}&namespace=0&format=json";

            using var searchResponse = await client.GetAsync(searchUrl, cancellationToken);
            if (!searchResponse.IsSuccessStatusCode)
            {
                continue;
            }

            await using var stream = await searchResponse.Content.ReadAsStreamAsync(cancellationToken);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
            if (doc.RootElement.ValueKind != JsonValueKind.Array || doc.RootElement.GetArrayLength() < 4)
            {
                continue;
            }

            var titles = doc.RootElement[1];
            var urls = doc.RootElement[3];
            if (titles.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            for (var i = 0; i < titles.GetArrayLength() && list.Count < maxResults; i++)
            {
                var title = titles[i].GetString();
                if (string.IsNullOrWhiteSpace(title))
                {
                    continue;
                }

                var pageUrl = urls.ValueKind == JsonValueKind.Array && i < urls.GetArrayLength()
                    ? urls[i].GetString()
                    : $"https://{lang}.wikipedia.org/wiki/{Uri.EscapeDataString(title.Replace(' ', '_'))}";

                var summary = await FetchWikipediaSummaryAsync(client, lang, title, maxSnippetChars, cancellationToken);
                list.Add(new WebFinding(
                    title,
                    summary ?? $"Bài Wikipedia ({lang}): {title}",
                    pageUrl,
                    $"Wikipedia ({lang})"));
            }
        }

        return list;
    }

    private static async Task<string?> FetchWikipediaSummaryAsync(
        HttpClient client,
        string lang,
        string title,
        int maxSnippetChars,
        CancellationToken cancellationToken)
    {
        var url =
            $"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{Uri.EscapeDataString(title.Replace(' ', '_'))}";

        using var response = await client.GetAsync(url, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            return null;
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        var extract = doc.RootElement.TryGetProperty("extract", out var e) ? e.GetString() : null;
        return string.IsNullOrWhiteSpace(extract) ? null : Truncate(CleanText(extract), maxSnippetChars);
    }

    private async Task<string?> FetchPageSnippetAsync(
        string pageUrl,
        int maxChars,
        CancellationToken cancellationToken)
    {
        if (!IsHttpUrl(pageUrl))
        {
            return null;
        }

        // Tránh một số domain nặng / ít hữu ích
        if (pageUrl.Contains("youtube.com", StringComparison.OrdinalIgnoreCase) ||
            pageUrl.Contains("facebook.com", StringComparison.OrdinalIgnoreCase) ||
            pageUrl.Contains("twitter.com", StringComparison.OrdinalIgnoreCase) ||
            pageUrl.Contains("x.com/", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var client = _httpClientFactory.CreateClient(HttpClientName);
        using var request = new HttpRequestMessage(HttpMethod.Get, pageUrl);
        request.Headers.Accept.ParseAdd("text/html,application/xhtml+xml");

        using var response = await client.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            return null;
        }

        var contentType = response.Content.Headers.ContentType?.MediaType ?? string.Empty;
        if (!contentType.Contains("html", StringComparison.OrdinalIgnoreCase) &&
            !contentType.Contains("text", StringComparison.OrdinalIgnoreCase) &&
            !string.IsNullOrEmpty(contentType))
        {
            return null;
        }

        var html = await response.Content.ReadAsStringAsync(cancellationToken);
        if (html.Length > 1_500_000)
        {
            html = html[..1_500_000];
        }

        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        foreach (var node in doc.DocumentNode.SelectNodes("//script|//style|//noscript|//nav|//footer|//header")
                     ?? Enumerable.Empty<HtmlNode>())
        {
            node.Remove();
        }

        var main = doc.DocumentNode.SelectSingleNode("//article")
                   ?? doc.DocumentNode.SelectSingleNode("//main")
                   ?? doc.DocumentNode.SelectSingleNode("//*[@id='content']")
                   ?? doc.DocumentNode.SelectSingleNode("//body");

        var text = CleanText(main?.InnerText ?? string.Empty);
        return string.IsNullOrWhiteSpace(text) ? null : Truncate(text, maxChars);
    }

    private static string BuildContextBlock(string query, IReadOnlyList<WebFinding> findings, int maxContextChars)
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== KẾT QUẢ TÌM KIẾM WEB (cập nhật từ mạng) ===");
        sb.AppendLine($"Truy vấn: {query}");
        sb.AppendLine($"Thời điểm: {DateTimeOffset.Now:yyyy-MM-dd HH:mm} (local)");
        sb.AppendLine();

        for (var i = 0; i < findings.Count; i++)
        {
            var f = findings[i];
            var block = new StringBuilder();
            block.AppendLine($"[{i + 1}] {f.Title} ({f.Source})");
            if (!string.IsNullOrWhiteSpace(f.Url))
            {
                block.AppendLine($"URL: {f.Url}");
            }

            if (!string.IsNullOrWhiteSpace(f.Snippet))
            {
                block.AppendLine($"Tóm tắt: {f.Snippet}");
            }

            if (!string.IsNullOrWhiteSpace(f.PageExcerpt))
            {
                block.AppendLine($"Nội dung trang: {f.PageExcerpt}");
            }

            block.AppendLine();

            if (sb.Length + block.Length > maxContextChars)
            {
                break;
            }

            sb.Append(block);
        }

        sb.AppendLine("=== HẾT KẾT QUẢ WEB ===");
        sb.AppendLine("Hãy ưu tiên thông tin trên khi trả lời. Nêu nguồn nếu có thể. Nếu dữ liệu mâu thuẫn hoặc thiếu, nói rõ.");
        return sb.ToString().Trim();
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

    private static string? NormalizeDuckDuckGoUrl(string? href)
    {
        if (string.IsNullOrWhiteSpace(href))
        {
            return null;
        }

        // DDG thường bọc: //duckduckgo.com/l/?uddg=<encoded>&...
        if (href.Contains("uddg=", StringComparison.OrdinalIgnoreCase))
        {
            var match = Regex.Match(href, @"[?&]uddg=([^&]+)", RegexOptions.IgnoreCase);
            if (match.Success)
            {
                return WebUtility.UrlDecode(match.Groups[1].Value);
            }
        }

        if (href.StartsWith("//", StringComparison.Ordinal))
        {
            return "https:" + href;
        }

        return href;
    }

    private static bool IsHttpUrl(string url) =>
        Uri.TryCreate(url, UriKind.Absolute, out var uri) &&
        (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);

    private static string CleanText(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return string.Empty;
        }

        var text = WebUtility.HtmlDecode(input);
        text = StripTagsRegex.Replace(text, " ");
        text = text.Replace('\u00a0', ' ');
        text = MultiSpaceRegex.Replace(text, " ").Trim();
        return text;
    }

    private static string Truncate(string text, int maxChars)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxChars)
        {
            return text;
        }

        return text[..(maxChars - 1)].TrimEnd() + "…";
    }
}

public sealed class WebFinding
{
    public WebFinding(string title, string snippet, string? url, string source)
    {
        Title = title;
        Snippet = snippet;
        Url = url;
        Source = source;
    }

    public string Title { get; }
    public string Snippet { get; }
    public string? Url { get; }
    public string Source { get; }
    public string? PageExcerpt { get; set; }
}

public sealed record WebResearchResult(
    bool HasResults,
    IReadOnlyList<WebFinding> Findings,
    string ContextBlock)
{
    public static WebResearchResult Empty { get; } = new(false, Array.Empty<WebFinding>(), string.Empty);

    /// <summary>Đổi findings → CitationSource đánh số [1]..[N] cho UI / prompt.</summary>
    public IReadOnlyList<CitationSource> ToCitations(int startId = 1)
    {
        if (!HasResults)
        {
            return [];
        }

        var list = new List<CitationSource>();
        for (var i = 0; i < Findings.Count; i++)
        {
            var f = Findings[i];
            list.Add(new CitationSource
            {
                Id = startId + i,
                Kind = "web",
                Title = f.Title,
                Url = f.Url,
                Snippet = f.Snippet.Length > 220 ? f.Snippet[..219] + "…" : f.Snippet,
                Score = Math.Round(1.0 - i * 0.08, 2),
                SourceLabel = f.Source
            });
        }

        return list;
    }
}
