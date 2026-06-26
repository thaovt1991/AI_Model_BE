using AI_Model_BE.Models;



namespace AI_Model_BE.Services;



/// <summary>

/// API dự đoán xổ số — ủy quyền ML.NET SSA + dữ liệu Minh Ngọc cho MlPredictionService.

/// </summary>

public sealed class LottoForecastService

{

    private readonly ILogger<LottoForecastService> _logger;

    private readonly LotteryRecordLoader _loader;

    private readonly MlPredictionService _mlPrediction;

    private readonly MinhNgocLotteryScraper _scraper;

    private readonly MinhNgocScrapeSettingsService _scrapeSettings;



    public LottoForecastService(

        LotteryRecordLoader loader,

        MlPredictionService mlPrediction,

        MinhNgocLotteryScraper scraper,

        MinhNgocScrapeSettingsService scrapeSettings,

        ILogger<LottoForecastService> logger)

    {

        _loader = loader;

        _mlPrediction = mlPrediction;

        _scraper = scraper;

        _scrapeSettings = scrapeSettings;

        _logger = logger;

    }



    public IReadOnlyList<LottoGameInfoDto> GetGameCatalog() => LottoGameCatalog.GetAllForApi();



    public LotteryRecord? GetLatestResultByDai(string dai, IReadOnlyList<LotteryRecord> allData)

    {

        var records = LotteryRecordNormalizer.FilterByDai(allData, dai);

        return records.Count == 0 ? null : records[^1];

    }



    public async Task<DaiPredictionResult> TrainAndPredictForDaiAsync(

        string dai,

        IReadOnlyList<LotteryRecord> allData,

        LottoGameProfile profile,

        string gameKind,

        string? daiCode,

        CancellationToken cancellationToken = default)

    {

        _ = allData;

        _ = profile;

        _ = dai;



        var output = await _mlPrediction.PredictLotteryForDaiAsync(gameKind, daiCode, cancellationToken);

        return MapToDaiResult(output, profile);

    }



    public async Task<List<DaiPredictionResult>> PredictDaisAsync(
        IReadOnlyList<string> daiCodes,
        LottoGameProfile profile,
        string gameKind,
        CancellationToken cancellationToken = default)
    {
        var results = new List<DaiPredictionResult>(daiCodes.Count);
        foreach (var code in daiCodes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var daiProfile = LottoGameCatalog.GetProfile(gameKind, code);
            var output = await _mlPrediction.PredictLotteryForDaiAsync(gameKind, code, cancellationToken);
            results.Add(MapToDaiResult(output, daiProfile));
        }

        return results;
    }

    public async Task<List<DaiPredictionResult>> PredictAllDaisAsync(
        IReadOnlyList<LotteryRecord> dataset,
        LottoGameProfile profile,
        string gameKind,
        CancellationToken cancellationToken = default)
    {
        _ = dataset;
        _ = profile;

        var daiCodes = LottoGameCatalog.GetDaiList(gameKind)
            .Select(d => d.Code)
            .ToList();

        return await PredictDaisAsync(daiCodes, profile, gameKind, cancellationToken);
    }

    public async Task<LottoRunResponse> RunForecastAsync(
        LottoRunRequest request,
        CancellationToken cancellationToken = default)
    {
        var profile = LottoGameCatalog.GetProfileByKind(request.GameKind);

        if (!profile.RequiresDai)
        {
            var output = await _mlPrediction.PredictLotteryForDaiAsync(
                request.GameKind,
                null,
                cancellationToken);
            var single = MapToDaiResult(output, profile);
            return new LottoRunResponse(request.GameKind, null, single, null);
        }

        var daiCodes = ResolvePredictDaiCodes(request);
        if (daiCodes.Count == 0)
        {
            throw new ArgumentException("Cần chọn ít nhất một đài để dự đoán.");
        }

        if (daiCodes.Count > 1 || request.PredictAllDais || request.DaiCodes is { Count: > 0 })
        {
            var all = await PredictDaisAsync(daiCodes, profile, request.GameKind, cancellationToken);
            var single = all.Count == 1 ? all[0] : null;
            var multi = all.Count > 1 ? all : null;
            return new LottoRunResponse(
                request.GameKind,
                daiCodes.Count == 1 ? daiCodes[0] : null,
                single,
                multi);
        }

        var effectiveDaiCode = daiCodes[0];
        var singleProfile = LottoGameCatalog.GetProfile(request.GameKind, effectiveDaiCode);
        var singleOutput = await _mlPrediction.PredictLotteryForDaiAsync(
            request.GameKind,
            effectiveDaiCode,
            cancellationToken);
        var singleResult = MapToDaiResult(singleOutput, singleProfile);
        return new LottoRunResponse(request.GameKind, effectiveDaiCode, singleResult, null);
    }

    private static List<string> ResolvePredictDaiCodes(LottoRunRequest request)
    {
        if (request.PredictAllDais)
        {
            return LottoGameCatalog.GetDaiList(request.GameKind)
                .Select(d => d.Code)
                .ToList();
        }

        if (request.DaiCodes is { Count: > 0 })
        {
            return request.DaiCodes
                .Where(static c => !string.IsNullOrWhiteSpace(c))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        if (!string.IsNullOrWhiteSpace(request.DaiCode))
        {
            return [request.DaiCode];
        }

        var fallback = ResolveDefaultDaiCode(request.GameKind);
        return string.IsNullOrWhiteSpace(fallback) ? [] : [fallback];
    }



    public async Task<LottoHistoryResponse> GetHistoryAsync(

        string gameKind,

        string? daiCode,

        DateTime? fromDate,

        DateTime? toDate,

        DateTime? singleDate,

        CancellationToken cancellationToken = default)

    {

        var effectiveDai = daiCode ?? ResolveDefaultDaiCode(gameKind);

        var daiName = ResolveDaiDisplayName(gameKind, effectiveDai);

        var all = await _mlPrediction.LoadHistoryForApiAsync(gameKind, effectiveDai, cancellationToken);



        IEnumerable<LotteryRecord> filtered = all;

        if (singleDate.HasValue)

        {

            var d = singleDate.Value.Date;

            filtered = all.Where(r => r.NgayQuay.Date == d);

        }

        else

        {

            if (fromDate.HasValue)

            {

                filtered = filtered.Where(r => r.NgayQuay.Date >= fromDate.Value.Date);

            }

            if (toDate.HasValue)

            {

                filtered = filtered.Where(r => r.NgayQuay.Date <= toDate.Value.Date);

            }

        }



        var records = filtered.OrderByDescending(static r => r.NgayQuay).ToList();



        if (singleDate.HasValue && records.Count == 0)

        {

            records = await TryFetchSingleDateAsync(gameKind, effectiveDai, daiName, singleDate.Value, cancellationToken);

        }



        return new LottoHistoryResponse

        {

            GameKind = gameKind,

            DaiCode = effectiveDai,

            DaiTen = daiName,

            TuNgay = fromDate,

            DenNgay = toDate,

            TongKy = records.Count,

            Records = records

        };

    }



    public async Task<LottoLoGanResponse> GetLoGanAsync(

        string gameKind,

        string? daiCode,

        int top,

        CancellationToken cancellationToken = default)

    {

        var effectiveDai = daiCode ?? ResolveDefaultDaiCode(gameKind);

        var daiName = ResolveDaiDisplayName(gameKind, effectiveDai);

        var history = await _mlPrediction.LoadHistoryForApiAsync(gameKind, effectiveDai, cancellationToken);

        var ordered = history.OrderBy(static r => r.NgayQuay).ToList();

        var items = LottoLotoAnalyzer.ComputeLoGan(ordered, gameKind, Math.Clamp(top, 5, 100));



        return new LottoLoGanResponse

        {

            GameKind = gameKind,

            DaiCode = effectiveDai,

            DaiTen = daiName,

            SoKyPhanTich = ordered.Count,

            Items = items

        };

    }



    private async Task<List<LotteryRecord>> TryFetchSingleDateAsync(

        string gameKind,

        string? daiCode,

        string daiName,

        DateTime date,

        CancellationToken cancellationToken)

    {

        try

        {

            if (gameKind == LottoGameKinds.XsMienBac)

            {

                var url = $"https://www.minhngoc.net.vn/ket-qua-xo-so/mien-bac/{date:dd-MM-yyyy}.html";

                var rows = await _scraper.FetchLatestDataFromMinhNgocAsync(url, daiName, cancellationToken);

                return rows.Where(r => r.NgayQuay.Date == date.Date).ToList();

            }



            if (!string.IsNullOrWhiteSpace(daiCode))

            {

                var url = MinhNgocLotteryScraper.BuildDaiDateUrl(gameKind, daiCode, date);

                var rows = await _scraper.FetchLatestDataFromMinhNgocAsync(url, daiName, cancellationToken);

                return rows.Where(r => r.NgayQuay.Date == date.Date).ToList();

            }

        }

        catch (Exception ex)

        {

            _logger.LogDebug(ex, "Không tải được kết quả ngày {Date}", date);

        }



        return [];

    }



    private static string? ResolveDefaultDaiCode(string gameKind) =>

        gameKind == LottoGameKinds.XsMienBac

            ? LottoGameCatalog.ResolveMienBacDaiCodeForDate(DateTime.Today)

            : null;



    public async Task<LotteryRecord?> GetLatestForApiAsync(

        string gameKind,

        string? daiCode,

        CancellationToken cancellationToken = default)

    {

        var settings = _scrapeSettings.Get();

        var effectiveDaiCode = daiCode ?? ResolveDefaultDaiCode(gameKind);

        var daiName = ResolveDaiDisplayName(gameKind, effectiveDaiCode);



        if (gameKind == LottoGameKinds.XsMienBac)

        {

            if (settings.ScrapingEnabled)

            {

                try

                {

                    var scraped = await _scraper.FetchLatestMienBacRecordForDaiAsync(
                        effectiveDaiCode!,
                        daiName,
                        cancellationToken);

                    if (scraped is not null)

                    {

                        return scraped;

                    }

                }

                catch (Exception ex)

                {

                    _logger.LogDebug(ex, "Không scrape được XSMB latest");

                }

            }



            return await TryGetLatestFromLocalAsync(

                LottoGameKinds.XsMienBac,

                effectiveDaiCode,

                daiName,

                cancellationToken);

        }



        if (settings.ScrapingEnabled && !string.IsNullOrWhiteSpace(effectiveDaiCode))

        {

            try

            {

                var url = MinhNgocLotteryScraper.BuildDaiListUrl(gameKind, effectiveDaiCode);

                var scraped = await _scraper.FetchLatestDataFromMinhNgocAsync(url, daiName, cancellationToken);
                var daiKey = LotteryRecordNormalizer.NormalizeDaiKey(daiName);
                var latest = scraped
                    .Where(r => LotteryRecordNormalizer.NormalizeDaiKey(r.Dai) == daiKey)
                    .OrderBy(r => r.NgayQuay)
                    .LastOrDefault()
                    ?? scraped.OrderBy(r => r.NgayQuay).LastOrDefault();

                if (latest is not null)
                {
                    return latest;
                }

            }

            catch (Exception ex)

            {

                _logger.LogDebug(ex, "Fallback local latest cho đài {Dai}", daiName);

            }

        }



        return await TryGetLatestFromLocalAsync(gameKind, effectiveDaiCode, daiName, cancellationToken);

    }



    private async Task<LotteryRecord?> TryGetLatestFromLocalAsync(

        string gameKind,

        string? daiCode,

        string daiName,

        CancellationToken cancellationToken)

    {

        var dataset = await _loader.LoadDatasetAsync(gameKind, daiCode, cancellationToken);

        if (dataset.Count == 0)

        {

            return null;

        }



        return GetLatestResultByDai(daiName, dataset)

            ?? dataset.OrderBy(r => r.NgayQuay).LastOrDefault();

    }



    private static DaiPredictionResult MapToDaiResult(

        PredictionOutputDto output,

        LottoGameProfile profile)

    {

        var result = new DaiPredictionResult

        {

            Dai = output.TenDai,

            KyQuayTruoc = output.KyQuayTruoc,

            KetQuaKyTruoc = output.KetQuaKyTruoc.ToList(),

            KyQuayDuDoan = output.KyQuayDuDoan,

            CacSoDuDoanLamTron = output.CacSoDuDoan.ToList(),

            CacSoDuDoanKyTiep = output.CacSoDuDoan.Select(static n => (float)n).ToList(),

            ThongBaoLoi = output.ThongBaoLoi,

            KetQuaTruocDinhDang = IsVietlottProfile(profile)
                ? LottoDisplayFormatter.Format(profile, output.KetQuaKyTruoc, ResolveVietlottPowerFromNumbers(output.KetQuaKyTruoc))
                : LottoDisplayFormatter.Format(profile, output.KetQuaKyTruoc, null),

            KetQuaDuDoanDinhDang = IsVietlottProfile(profile)
                ? LottoDisplayFormatter.Format(profile, output.CacSoDuDoan, ResolveVietlottPower(output))
                : output.DuDoanGiaiDacBiet.Count >= 6
                    ? LottoDisplayFormatter.Format(
                        profile with { ParseMode = LottoParseMode.SouthernSpecialPrizeSixDigits },
                        output.DuDoanGiaiDacBiet,
                        null)
                    : LottoDisplayFormatter.Format(profile, output.CacSoDuDoan, null),

            KetQuaGiaiDacBietTruoc = output.KetQuaGiaiDacBietTruoc.ToList(),

            TatCaLoVeKyTruoc = output.TatCaLoVeKyTruoc.ToList(),

            DuDoanGiaiDacBiet = output.DuDoanGiaiDacBiet.ToList(),

            GiaiDacBietTruocDinhDang = output.GiaiDacBietTruocDinhDang,

            GiaiDacBietDuDoanDinhDang = output.GiaiDacBietDuDoanDinhDang,

            SoKyDaHoc = output.SoKyDaHoc,

            DuDoanLo = output.DuDoanLo

        };



        if (DateTime.TryParseExact(

                output.NgayQuayKyTruoc,

                "dd/MM/yyyy",

                null,

                System.Globalization.DateTimeStyles.None,

                out var prevDate))

        {

            result.NgayQuayTruoc = prevDate;

        }



        if (DateTime.TryParseExact(

                output.NgayQuayDuDoan,

                "dd/MM/yyyy",

                null,

                System.Globalization.DateTimeStyles.None,

                out var nextDate))

        {

            result.NgayQuayDuKien = nextDate;

        }



        return result;

    }



    private static string ResolveDaiDisplayName(string gameKind, string? daiCode)

    {

        if (!string.IsNullOrWhiteSpace(daiCode))

        {

            return LottoGameCatalog.ResolveDai(gameKind, daiCode)?.Name ?? daiCode;

        }



        if (gameKind == LottoGameKinds.XsMienBac)

        {

            return MienBacDaiSchedule.ResolveDaiForDate(DateTime.Today).Name;

        }



        return gameKind switch

        {

            LottoGameKinds.XsMienBac => "Miền Bắc",

            LottoGameKinds.Vietlott645 => "Vietlott Mega 6/45",

            LottoGameKinds.Vietlott655 => "Vietlott Power 6/55",

            _ => "Chung"

        };

    }

    private static bool IsVietlottProfile(LottoGameProfile profile) =>
        profile.ParseMode is LottoParseMode.Vietlott645 or LottoParseMode.Vietlott655;

    private static int? ResolveVietlottPower(PredictionOutputDto output) =>
        ResolveVietlottPowerFromNumbers(output.CacSoDuDoan);

    private static int? ResolveVietlottPowerFromNumbers(IReadOnlyList<int> numbers) =>
        numbers.Count >= 7 ? numbers[^1] : null;

}


