using System.Globalization;
using System.Net;
using System.Text.RegularExpressions;
using AI_Model_BE.Models;
using HtmlAgilityPack;
using Microsoft.Extensions.Caching.Memory;

namespace AI_Model_BE.Services;

/// <summary>Cào dữ liệu kết quả xổ số từ minhngoc.net.vn — có cache, delay và giới hạn request.</summary>
public sealed class MinhNgocLotteryScraper
{
    private static readonly CultureInfo ViCulture = CultureInfo.GetCultureInfo("vi-VN");

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly MinhNgocScrapeSettingsService _scrapeSettings;
    private readonly IMemoryCache _cache;
    private readonly ILogger<MinhNgocLotteryScraper> _logger;
    private readonly SemaphoreSlim _requestGate = new(1, 1);

    public MinhNgocLotteryScraper(
        IHttpClientFactory httpClientFactory,
        MinhNgocScrapeSettingsService scrapeSettings,
        IMemoryCache cache,
        ILogger<MinhNgocLotteryScraper> logger)
    {
        _httpClientFactory = httpClientFactory;
        _scrapeSettings = scrapeSettings;
        _cache = cache;
        _logger = logger;
    }

    public static string BuildDaiListUrl(string gameKind, string daiCode) =>
        $"https://www.minhngoc.net.vn/ket-qua-xo-so/{ResolveRegion(gameKind)}/{LottoGameCatalog.ResolveMinhNgocSlug(gameKind, daiCode)}.html";

    public static string BuildDaiDateUrl(string gameKind, string daiCode, DateTime date) =>
        $"https://www.minhngoc.net.vn/ket-qua-xo-so/{ResolveRegion(gameKind)}/{LottoGameCatalog.ResolveMinhNgocSlug(gameKind, daiCode)}/{date:dd-MM-yyyy}.html";

    /// <summary>Tải HTML và parse tất cả kỳ quay có trong trang.</summary>
    public async Task<IReadOnlyList<LotteryRecord>> FetchLatestDataFromMinhNgocAsync(
        string url,
        string expectedDaiName,
        CancellationToken cancellationToken = default)
    {
        var html = await DownloadHtmlAsync(url, cancellationToken);
        return ParseRecordsFromHtml(html, expectedDaiName);
    }

    /// <summary>
    /// Thu thập lịch sử đài: trang danh sách + lùi theo lịch quay (giới hạn số request theo cài đặt).
    /// </summary>
    public async Task<List<LotteryRecord>> FetchDaiHistoryAsync(
        string gameKind,
        string daiCode,
        string daiName,
        int minDraws,
        CancellationToken cancellationToken = default)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            return [];
        }

        var merged = new Dictionary<DateTime, LotteryRecord>();

        var listUrl = BuildDaiListUrl(gameKind, daiCode);
        foreach (var row in await FetchLatestDataFromMinhNgocAsync(listUrl, daiName, cancellationToken))
        {
            merged[row.NgayQuay.Date] = row;
        }

        if (merged.Count >= minDraws)
        {
            return OrderAndAssignKyQuay(merged);
        }

        var anchor = merged.Count > 0 ? merged.Keys.Max() : DateTime.Today;
        var dateFetchCount = 0;

        foreach (var drawDate in LottoDrawSchedule.EnumeratePreviousDrawDates(anchor, daiCode, gameKind, minDraws + 40))
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (merged.ContainsKey(drawDate))
            {
                continue;
            }

            if (dateFetchCount >= settings.MaxHistoryFetchesPerRun)
            {
                _logger.LogDebug(
                    "Đạt giới hạn {Max} request lịch sử/đài cho {Dai}",
                    settings.MaxHistoryFetchesPerRun,
                    daiCode);
                break;
            }

            var dateUrl = BuildDaiDateUrl(gameKind, daiCode, drawDate);
            dateFetchCount++;
            try
            {
                var rows = await FetchLatestDataFromMinhNgocAsync(dateUrl, daiName, cancellationToken);
                foreach (var row in rows)
                {
                    merged[row.NgayQuay.Date] = row;
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Bỏ qua ngày {Date} đài {Dai}", drawDate, daiCode);
            }

            if (merged.Count >= minDraws)
            {
                break;
            }
        }

        return OrderAndAssignKyQuay(merged);
    }

    /// <summary>Lịch sử XSMB — chỉ các kỳ do đài được chọn phát hành.</summary>
    public async Task<List<LotteryRecord>> FetchMienBacHistoryAsync(
        string daiCode,
        string daiName,
        int minDraws,
        CancellationToken cancellationToken = default)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            return [];
        }

        var merged = await CollectMienBacRecordsForDaiAsync(
            daiCode,
            minDraws,
            settings.MaxHistoryFetchesPerRun,
            cancellationToken);

        return OrderAndAssignKyQuay(FilterRecordsForDai(merged, daiName));
    }

    /// <summary>Kết quả mới nhất của riêng một đài XSMB (theo lịch phát hành).</summary>
    public async Task<LotteryRecord?> FetchLatestMienBacRecordForDaiAsync(
        string daiCode,
        string daiName,
        CancellationToken cancellationToken = default)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            return null;
        }

        var merged = await CollectMienBacRecordsForDaiAsync(
            daiCode,
            minTarget: 1,
            maxDateFetches: settings.MaxLatestProbeAttempts,
            cancellationToken);

        var latest = FilterRecordsForDai(merged, daiName)
            .OrderBy(r => r.NgayQuay)
            .LastOrDefault();

        if (latest is not null)
        {
            var ordered = FilterRecordsForDai(merged, daiName);
            latest.KyQuay = ordered.Count;
        }

        return latest;
    }

    private async Task<Dictionary<DateTime, LotteryRecord>> CollectMienBacRecordsForDaiAsync(
        string daiCode,
        int minTarget,
        int maxDateFetches,
        CancellationToken cancellationToken)
    {
        var merged = new Dictionary<DateTime, LotteryRecord>();
        const string listUrl = "https://www.minhngoc.net.vn/ket-qua-xo-so/mien-bac.html";

        foreach (var row in await FetchMienBacRowsFromUrlAsync(listUrl, cancellationToken))
        {
            if (MienBacDaiSchedule.IsDaiDrawDay(row.NgayQuay, daiCode))
            {
                merged[row.NgayQuay.Date] = row;
            }
        }

        if (CountForDai(merged, daiCode) >= minTarget)
        {
            return merged;
        }

        var dateFetchCount = 0;
        foreach (var drawDate in MienBacDaiSchedule.EnumerateDrawDatesForDai(
                     daiCode,
                     DateTime.Today,
                     Math.Max(maxDateFetches * 2, 14)))
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (merged.ContainsKey(drawDate.Date))
            {
                continue;
            }

            if (dateFetchCount >= maxDateFetches)
            {
                break;
            }

            dateFetchCount++;
            var dateUrl = $"https://www.minhngoc.net.vn/ket-qua-xo-so/mien-bac/{drawDate:dd-MM-yyyy}.html";
            try
            {
                foreach (var row in await FetchMienBacRowsFromUrlAsync(dateUrl, cancellationToken))
                {
                    if (MienBacDaiSchedule.IsDaiDrawDay(row.NgayQuay, daiCode))
                    {
                        merged[row.NgayQuay.Date] = row;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Bỏ qua ngày XSMB {Date} đài {Dai}", drawDate, daiCode);
            }

            if (CountForDai(merged, daiCode) >= minTarget)
            {
                break;
            }
        }

        return merged;
    }

    private static int CountForDai(Dictionary<DateTime, LotteryRecord> merged, string daiCode) =>
        merged.Values.Count(r => MienBacDaiSchedule.IsDaiDrawDay(r.NgayQuay, daiCode));

    private static List<LotteryRecord> FilterRecordsForDai(
        Dictionary<DateTime, LotteryRecord> merged,
        string daiName)
    {
        var key = LotteryRecordNormalizer.NormalizeDaiKey(daiName);
        return merged.Values
            .Where(r => LotteryRecordNormalizer.NormalizeDaiKey(r.Dai) == key)
            .OrderBy(r => r.NgayQuay)
            .ThenBy(r => r.KyQuay)
            .ToList();
    }

    private static List<LotteryRecord> OrderAndAssignKyQuay(List<LotteryRecord> records)
    {
        for (var i = 0; i < records.Count; i++)
        {
            records[i].KyQuay = i + 1;
        }

        return records;
    }

    private static List<LotteryRecord> OrderAndAssignKyQuay(Dictionary<DateTime, LotteryRecord> merged) =>
        OrderAndAssignKyQuay(merged.Values.OrderBy(r => r.NgayQuay).ThenBy(r => r.KyQuay).ToList());

    private async Task<IReadOnlyList<LotteryRecord>> FetchMienBacRowsFromUrlAsync(
        string url,
        CancellationToken cancellationToken)
    {
        var placeholderDai = MienBacDaiSchedule.ResolveDaiForDate(DateTime.Today).Name;
        var rows = await FetchLatestDataFromMinhNgocAsync(url, placeholderDai, cancellationToken);
        var result = new List<LotteryRecord>();

        foreach (var row in rows)
        {
            row.Dai = MienBacDaiSchedule.ResolveDaiForDate(row.NgayQuay).Name;
            if (row.GiaiDacBietSo.Count >= 5 && row.TatCaLoVe.Count == 27)
            {
                result.Add(row);
            }
        }

        return result;
    }

    private async Task<string> DownloadHtmlAsync(string url, CancellationToken cancellationToken)
    {
        var settings = _scrapeSettings.Get();
        if (!settings.ScrapingEnabled)
        {
            throw new InvalidOperationException("Cào Minh Ngọc đang tắt trong cài đặt.");
        }

        var cacheKey = $"minhngoc:{url}";
        if (_cache.TryGetValue(cacheKey, out string? cached) && cached is not null)
        {
            return cached;
        }

        await _requestGate.WaitAsync(cancellationToken);
        try
        {
            if (_cache.TryGetValue(cacheKey, out cached) && cached is not null)
            {
                return cached;
            }

            if (settings.RequestDelayMs > 0)
            {
                await Task.Delay(settings.RequestDelayMs, cancellationToken);
            }

            var client = _httpClientFactory.CreateClient(nameof(MinhNgocLotteryScraper));
            using var response = await client.GetAsync(url, cancellationToken);
            response.EnsureSuccessStatusCode();
            var html = await response.Content.ReadAsStringAsync(cancellationToken);

            _cache.Set(
                cacheKey,
                html,
                new MemoryCacheEntryOptions
                {
                    AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(settings.CacheMinutes),
                });

            return html;
        }
        finally
        {
            _requestGate.Release();
        }
    }

    internal static List<LotteryRecord> ParseRecordsFromHtml(string html, string expectedDaiName)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(html);

        var records = new List<LotteryRecord>();

        var tables = doc.DocumentNode.SelectNodes(
            "//table[contains(@class,'bkqtinhmiennam') or contains(@class,'bkqtinhmientrung')]");
        if (tables is not null)
        {
            foreach (var table in tables)
            {
                var record = ParseSingleDrawTable(table, expectedDaiName);
                if (record is not null)
                {
                    records.Add(record);
                }
            }
        }

        if (records.Count == 0)
        {
            var mbTables = doc.DocumentNode.SelectNodes("//table[contains(@class,'bkqtinhmienbac')]");
            if (mbTables is not null)
            {
                foreach (var table in mbTables)
                {
                    var mb = ParseMienBacDrawTable(table, expectedDaiName);
                    if (mb is not null)
                    {
                        records.Add(mb);
                    }
                }
            }
        }

        return records
            .OrderBy(r => r.NgayQuay)
            .ThenBy(r => r.KyQuay)
            .ToList();
    }

    private static LotteryRecord? ParseSingleDrawTable(HtmlNode table, string expectedDaiName)
    {
        var dateNode = table.SelectSingleNode(".//div[contains(@class,'ngay')]//a")
            ?? table.SelectSingleNode(".//div[contains(@class,'ngay')]");
        if (dateNode is null)
        {
            return null;
        }

        var dateText = WebUtility.HtmlDecode(StripTags(dateNode.InnerText));
        var dateMatch = Regex.Match(dateText, @"(\d{1,2})/(\d{1,2})/(\d{4})");
        if (!dateMatch.Success)
        {
            return null;
        }

        var drawDate = new DateTime(
            int.Parse(dateMatch.Groups[3].Value, ViCulture),
            int.Parse(dateMatch.Groups[2].Value, ViCulture),
            int.Parse(dateMatch.Groups[1].Value, ViCulture));

        var loaiveNode = table.SelectSingleNode(".//span[contains(@class,'loaive')]");
        var kyQuay = 0;
        if (loaiveNode is not null)
        {
            var kyMatch = Regex.Match(WebUtility.HtmlDecode(loaiveNode.InnerText), @"(\d+)");
            if (kyMatch.Success)
            {
                kyQuay = int.Parse(kyMatch.Groups[1].Value, ViCulture);
            }
        }

        var loto = ExtractLotoNumbers(table);
        var giaiDb = ExtractGiaiDacBietDigits(table);
        if (loto.Count == 0 && giaiDb.Count == 0)
        {
            return null;
        }

        var tatCaLo = new HashSet<int>(loto);
        foreach (var lo in ExtractGiaiDacBietLoto(giaiDb))
        {
            tatCaLo.Add(lo);
        }

        return new LotteryRecord
        {
            Dai = expectedDaiName,
            NgayQuay = drawDate,
            KyQuay = kyQuay,
            CacSoDaVe = NormalizeSixLoto(loto),
            GiaiDacBietSo = giaiDb,
            TatCaLoVe = tatCaLo.OrderBy(static n => n).ToList()
        };
    }

    private static LotteryRecord? ParseMienBacDrawTable(HtmlNode table, string expectedDaiName)
    {
        var drawDate = ParseMienBacDrawDate(table);
        if (drawDate is null)
        {
            return null;
        }

        var giaiDb = ExtractMienBacGiaiDbDigits(table);
        if (giaiDb.Count < 5)
        {
            return null;
        }

        var tatCaLo = ExtractMienBacAllLoto(table);
        if (tatCaLo.Count != 27)
        {
            return null;
        }

        return new LotteryRecord
        {
            Dai = expectedDaiName,
            NgayQuay = drawDate.Value,
            CacSoDaVe = giaiDb,
            GiaiDacBietSo = giaiDb,
            TatCaLoVe = tatCaLo
        };
    }

    private static DateTime? ParseMienBacDrawDate(HtmlNode table)
    {
        var dateNode = table.SelectSingleNode(".//td[contains(@class,'ngay')]//a")
            ?? table.SelectSingleNode(".//td[contains(@class,'ngay')]");
        if (dateNode is null)
        {
            return null;
        }

        var m = Regex.Match(WebUtility.HtmlDecode(dateNode.InnerText), @"(\d{1,2})/(\d{1,2})/(\d{4})");
        if (!m.Success)
        {
            return null;
        }

        return new DateTime(
            int.Parse(m.Groups[3].Value, ViCulture),
            int.Parse(m.Groups[2].Value, ViCulture),
            int.Parse(m.Groups[1].Value, ViCulture));
    }

    private static List<int> ExtractMienBacGiaiDbDigits(HtmlNode table)
    {
        var cell = SelectMienBacPrizeCell(table, "giaidb");
        var div = cell?.SelectSingleNode("./div");
        if (div is null)
        {
            return [];
        }

        return WebUtility.HtmlDecode(div.InnerText).Trim()
            .Where(char.IsDigit)
            .Take(5)
            .Select(static c => c - '0')
            .ToList();
    }

    /// <summary>XSMB: đúng 27 lô — 2 số cuối của mỗi giải (27 giải).</summary>
    private static List<int> ExtractMienBacAllLoto(HtmlNode table)
    {
        (string Class, int ExpectedDivs, int DigitLength)[] prizes =
        [
            ("giaidb", 1, 5),
            ("giai1", 1, 5),
            ("giai2", 2, 5),
            ("giai3", 6, 5),
            ("giai4", 4, 4),
            ("giai5", 6, 4),
            ("giai6", 3, 3),
            ("giai7", 4, 2),
        ];

        var loto = new List<int>(27);
        foreach (var (cls, expectedDivs, digitLength) in prizes)
        {
            var cell = SelectMienBacPrizeCell(table, cls);
            if (cell is null)
            {
                return [];
            }

            var numbers = ExtractDivNumbers(cell, expectedDivs, digitLength);
            if (numbers.Count != expectedDivs)
            {
                return [];
            }

            loto.AddRange(numbers.Select(n => ToLo2Digits(n, digitLength)));
        }

        return loto;
    }

    private static HtmlNode? SelectMienBacPrizeCell(HtmlNode table, string prizeClass)
    {
        var cells = table.SelectNodes($".//td[contains(@class,'{prizeClass}')]");
        if (cells is null)
        {
            return null;
        }

        foreach (var cell in cells)
        {
            var cls = cell.GetAttributeValue("class", string.Empty);
            if (cls.Contains($"{prizeClass}l", StringComparison.Ordinal))
            {
                continue;
            }

            if (Regex.IsMatch(cls, $@"\b{Regex.Escape(prizeClass)}\b"))
            {
                return cell;
            }
        }

        return null;
    }

    private static List<int> ExtractDivNumbers(HtmlNode cell, int expectedCount, int digitLength)
    {
        var divs = cell.SelectNodes("./div");
        if (divs is null || divs.Count < expectedCount)
        {
            return [];
        }

        var numbers = new List<int>(expectedCount);
        foreach (var div in divs)
        {
            var text = WebUtility.HtmlDecode(div.InnerText).Trim();
            if (!Regex.IsMatch(text, @"^\d+$") || text.Length != digitLength)
            {
                continue;
            }

            if (int.TryParse(text, out var raw))
            {
                numbers.Add(raw);
            }
        }

        return numbers.Count == expectedCount ? numbers : [];
    }

    private static int ToLo2Digits(int raw, int digitLength)
    {
        var text = raw.ToString().PadLeft(digitLength, '0');
        return int.Parse(text[^2..], ViCulture);
    }

    private static List<int> ExtractLotoNumbers(HtmlNode table)
    {
        var loto = new HashSet<int>();
        string[] prizeClasses =
        [
            "giai8", "giai7", "giai6", "giai5", "giai4", "giai3", "giai2", "giai1"
        ];

        foreach (var cls in prizeClasses)
        {
            var cells = table.SelectNodes($".//td[contains(@class,'{cls}')]");
            if (cells is null)
            {
                continue;
            }

            foreach (var cell in cells)
            {
                var divs = cell.SelectNodes(".//div");
                if (divs is null)
                {
                    continue;
                }

                foreach (var div in divs)
                {
                    var text = WebUtility.HtmlDecode(div.InnerText).Trim();
                    if (!Regex.IsMatch(text, @"^\d+$"))
                    {
                        continue;
                    }

                    if (!int.TryParse(text, out var raw))
                    {
                        continue;
                    }

                    loto.Add(raw % 100);
                }
            }
        }

        return loto.OrderBy(n => n).ToList();
    }

    private static IEnumerable<int> ExtractGiaiDacBietLoto(IReadOnlyList<int> giaiDb)
    {
        if (giaiDb.Count < 2)
        {
            yield break;
        }

        var text = string.Concat(giaiDb.Select(static d => d.ToString()));
        if (text.Length >= 2 && int.TryParse(text[^2..], out var lo2))
        {
            yield return lo2;
        }

        if (text.Length >= 4)
        {
            if (int.TryParse(text[^4..^2], out var lo1))
            {
                yield return lo1;
            }
        }
    }

    private static List<int> ExtractGiaiDacBietDigits(HtmlNode table)
    {
        var cell = table.SelectSingleNode(".//td[contains(@class,'giaidb')]//div");
        if (cell is null)
        {
            return [];
        }

        var digits = WebUtility.HtmlDecode(cell.InnerText).Trim()
            .Where(char.IsDigit)
            .Select(c => c - '0')
            .ToList();

        return digits.Count >= 6 ? digits.Take(6).ToList() : digits;
    }

    /// <summary>Chuẩn hóa 6 số lô đại diện mỗi kỳ (00–99) cho chuỗi SSA.</summary>
    internal static List<int> NormalizeSixLoto(IReadOnlyList<int> lotoPool)
    {
        var result = lotoPool.Distinct().OrderBy(n => n).Take(6).ToList();
        var seed = 0;
        while (result.Count < 6 && seed <= 99)
        {
            if (!result.Contains(seed))
            {
                result.Add(seed);
            }

            seed++;
        }

        return result;
    }

    private static string StripTags(string html) =>
        Regex.Replace(WebUtility.HtmlDecode(html), "<.*?>", string.Empty).Trim();

    private static string ResolveRegion(string gameKind) => gameKind switch
    {
        LottoGameKinds.XsMienNam => "mien-nam",
        LottoGameKinds.XsMienTrung => "mien-trung",
        LottoGameKinds.XsMienBac => "mien-bac",
        _ => "mien-nam"
    };
}
