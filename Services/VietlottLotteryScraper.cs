using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using AI_Model_BE.Models;
using HtmlAgilityPack;

namespace AI_Model_BE.Services;

/// <summary>Cào kết quả Vietlott Mega 6/45 &amp; Power 6/55 từ vietlott.vn (AjaxPro).</summary>
public sealed class VietlottLotteryScraper
{
    private static readonly CultureInfo ViCulture = CultureInfo.GetCultureInfo("vi-VN");

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly MinhNgocScrapeSettingsService _scrapeSettings;
    private readonly ILogger<VietlottLotteryScraper> _logger;
    private readonly SemaphoreSlim _requestGate = new(1, 1);

    public VietlottLotteryScraper(
        IHttpClientFactory httpClientFactory,
        MinhNgocScrapeSettingsService scrapeSettings,
        ILogger<VietlottLotteryScraper> logger)
    {
        _httpClientFactory = httpClientFactory;
        _scrapeSettings = scrapeSettings;
        _logger = logger;
    }

    public async Task<LotteryRecord?> FetchLatestAsync(
        string gameKind,
        CancellationToken cancellationToken = default)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            return null;
        }

        var profile = ResolveProfile(gameKind);
        var rows = await FetchPageAsync(profile, 0, cancellationToken);
        return rows.Count == 0 ? null : rows[^1];
    }

    public async Task<List<LotteryRecord>> FetchHistoryAsync(
        string gameKind,
        int minDraws,
        CancellationToken cancellationToken = default)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            return [];
        }

        var profile = ResolveProfile(gameKind);
        var merged = new Dictionary<DateTime, LotteryRecord>();
        var maxPages = Math.Max(1, Math.Min(settings.MaxHistoryFetchesPerRun, 50));

        for (var pageIndex = 0; pageIndex < maxPages; pageIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var rows = await FetchPageAsync(profile, pageIndex, cancellationToken);
            if (rows.Count == 0)
            {
                break;
            }

            foreach (var row in rows)
            {
                merged[row.NgayQuay.Date] = row;
            }

            if (merged.Count >= minDraws)
            {
                break;
            }
        }

        var ordered = merged.Values
            .OrderBy(static r => r.NgayQuay)
            .ThenBy(static r => r.KyQuay)
            .ToList();

        for (var i = 0; i < ordered.Count; i++)
        {
            ordered[i].KyQuay = i + 1;
        }

        _logger.LogInformation(
            "Vietlott {Game}: {Count} kỳ, mới nhất {Date:dd/MM/yyyy} #{Ky}",
            gameKind,
            ordered.Count,
            ordered.LastOrDefault()?.NgayQuay,
            ordered.LastOrDefault()?.KyQuay);

        return ordered;
    }

    private async Task<List<LotteryRecord>> FetchPageAsync(
        VietlottProfile profile,
        int pageIndex,
        CancellationToken cancellationToken)
    {
        await _requestGate.WaitAsync(cancellationToken);
        try
        {
            var settings = _scrapeSettings.Get();
            if (settings.RequestDelayMs > 0)
            {
                await Task.Delay(settings.RequestDelayMs, cancellationToken);
            }

            var client = _httpClientFactory.CreateClient(nameof(VietlottLotteryScraper));
            using var request = new HttpRequestMessage(HttpMethod.Post, profile.ApiUrl)
            {
                Content = new StringContent(BuildRequestBody(profile, pageIndex), Encoding.UTF8, "text/plain"),
            };

            foreach (var (key, value) in BuildHeaders(profile))
            {
                request.Headers.TryAddWithoutValidation(key, value);
            }

            using var response = await client.SendAsync(request, cancellationToken);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync(cancellationToken);
            return ParseResponseHtml(json, profile.DaiName);
        }
        finally
        {
            _requestGate.Release();
        }
    }

    internal static List<LotteryRecord> ParseResponseHtml(string json, string daiName)
    {
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("value", out var value))
        {
            return [];
        }

        if (value.TryGetProperty("Error", out var err) && err.GetBoolean())
        {
            return [];
        }

        if (!value.TryGetProperty("HtmlContent", out var htmlNode))
        {
            return [];
        }

        var html = htmlNode.GetString();
        if (string.IsNullOrWhiteSpace(html))
        {
            return [];
        }

        return ParseResultTable(html, daiName);
    }

    internal static List<LotteryRecord> ParseResultTable(string html, string daiName)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        var records = new List<LotteryRecord>();
        foreach (var row in doc.DocumentNode.SelectNodes("//table//tbody//tr") ?? Enumerable.Empty<HtmlNode>())
        {
            var cells = row.SelectNodes("./td");
            if (cells is null || cells.Count < 3)
            {
                continue;
            }

            var dateText = HtmlEntity.DeEntitize(cells[0].InnerText).Trim();
            if (!TryParseDate(dateText, out var drawDate))
            {
                continue;
            }

            var kyText = HtmlEntity.DeEntitize(cells[1].InnerText).Trim();
            var kyMatch = Regex.Match(kyText, @"(\d+)");
            if (!kyMatch.Success)
            {
                continue;
            }

            var numbers = cells[2]
                .SelectNodes(".//span[contains(@class,'bong_tron') and not(contains(@class,'sperator'))]")
                ?.Select(static span => int.TryParse(span.InnerText.Trim(), out var n) ? n : -1)
                .Where(static n => n > 0)
                .ToList() ?? [];

            if (numbers.Count < 6)
            {
                continue;
            }

            records.Add(new LotteryRecord
            {
                Dai = daiName,
                NgayQuay = drawDate,
                KyQuay = int.Parse(kyMatch.Groups[1].Value, ViCulture),
                CacSoDaVe = numbers,
            });
        }

        return records
            .OrderBy(static r => r.NgayQuay)
            .ThenBy(static r => r.KyQuay)
            .ToList();
    }

    private static bool TryParseDate(string text, out DateTime drawDate)
    {
        drawDate = default;
        var match = Regex.Match(text, @"(\d{1,2})/(\d{1,2})/(\d{4})");
        if (!match.Success)
        {
            return false;
        }

        drawDate = new DateTime(
            int.Parse(match.Groups[3].Value, ViCulture),
            int.Parse(match.Groups[2].Value, ViCulture),
            int.Parse(match.Groups[1].Value, ViCulture));
        return true;
    }

    private static string BuildRequestBody(VietlottProfile profile, int pageIndex)
    {
        var body = new
        {
            ORenderInfo = new
            {
                SiteId = "main.frontend.vi",
                SiteAlias = "main.vi",
                UserSessionId = "",
                SiteLang = "vi",
                IsPageDesign = false,
                ExtraParam1 = "",
                ExtraParam2 = "",
                ExtraParam3 = "",
                SiteURL = "",
                WebPage = (string?)null,
                SiteName = "Vietlott",
                OrgPageAlias = (string?)null,
                PageAlias = (string?)null,
                RefKey = (string?)null,
                FullPageAlias = (string?)null,
                System = 1,
            },
            Key = profile.RequestKey,
            GameDrawId = "",
            ArrayNumbers = BuildEmptyArray(profile.ArrayRows, 18),
            CheckMulti = false,
            PageIndex = pageIndex,
        };

        return JsonSerializer.Serialize(body);
    }

    private static List<List<string>> BuildEmptyArray(int rows, int cols)
    {
        var result = new List<List<string>>(rows);
        for (var i = 0; i < rows; i++)
        {
            var row = new List<string>(cols);
            for (var j = 0; j < cols; j++)
            {
                row.Add("");
            }

            result.Add(row);
        }

        return result;
    }

    private static Dictionary<string, string> BuildHeaders(VietlottProfile profile) => new()
    {
        ["Accept"] = "*/*",
        ["Accept-Language"] = "vi-VN,vi;q=0.9,en;q=0.8",
        ["Content-Type"] = "text/plain; charset=utf-8",
        ["X-AjaxPro-Method"] = "ServerSideDrawResult",
        ["X-Requested-With"] = "XMLHttpRequest",
        ["Origin"] = "https://vietlott.vn",
        ["Referer"] = profile.RefererUrl,
    };

    private static VietlottProfile ResolveProfile(string gameKind) => gameKind switch
    {
        LottoGameKinds.Vietlott645 => new VietlottProfile(
            LottoGameKinds.Vietlott645,
            "Vietlott Mega 6/45",
            "https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.Game645CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
            "8290fce2",
            6,
            "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-645"),
        LottoGameKinds.Vietlott655 => new VietlottProfile(
            LottoGameKinds.Vietlott655,
            "Vietlott Power 6/55",
            "https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.Game655CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
            "23bbd667",
            5,
            "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-power655"),
        _ => throw new ArgumentException($"Không hỗ trợ Vietlott: {gameKind}"),
    };

    private sealed record VietlottProfile(
        string GameKind,
        string DaiName,
        string ApiUrl,
        string RequestKey,
        int ArrayRows,
        string RefererUrl);
}
