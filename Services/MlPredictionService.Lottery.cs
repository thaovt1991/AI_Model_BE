using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Phần dự đoán xổ số ML.NET (ForecastingBySsa) — tích hợp minhngoc.net.vn.
/// </summary>
public sealed partial class MlPredictionService
{
    private const int MinHistoryDraws = 20;

    private readonly MinhNgocLotteryScraper _minhNgocScraper;
    private readonly VietlottLotteryScraper _vietlottScraper;
    private readonly LotteryRecordLoader _lotteryLoader;
    private readonly MinhNgocScrapeSettingsService _scrapeSettings;

    /// <summary>
    /// Cào kết quả từ URL minhngoc.net.vn (trang đài hoặc trang theo ngày).
    /// </summary>
    public Task<IReadOnlyList<LotteryRecord>> FetchLatestDataFromMinhNgocAsync(
        string url,
        string daiName,
        CancellationToken cancellationToken = default) =>
        _minhNgocScraper.FetchLatestDataFromMinhNgocAsync(url, daiName, cancellationToken);

    /// <summary>
    /// Huấn luyện SSA và dự đoán kỳ tiếp theo cho một đài — không trộn dữ liệu đài khác.
    /// </summary>
    public async Task<PredictionOutputDto> PredictLotteryForDaiAsync(
        string gameKind,
        string? daiCode,
        CancellationToken cancellationToken = default)
    {
        var profile = daiCode is null
            ? LottoGameCatalog.GetProfileByKind(gameKind)
            : LottoGameCatalog.GetProfile(gameKind, daiCode);

        var daiName = ResolveDaiName(gameKind, daiCode);
        var history = await LoadDaiHistoryAsync(gameKind, daiCode, daiName, cancellationToken);

        if (history.Count == 0)
        {
            return Fail(daiName, "Không có dữ liệu lịch sử từ Minh Ngọc hoặc file mẫu.");
        }

        // Bắt buộc sắp xếp tăng dần — phần tử cuối là kỳ gần nhất (mốc forecast)
        var sortedData = history
            .Where(r => LotteryRecordNormalizer.NormalizeDaiKey(r.Dai) ==
                        LotteryRecordNormalizer.NormalizeDaiKey(daiName))
            .OrderBy(x => x.NgayQuay)
            .ThenBy(x => x.KyQuay)
            .ToList();

        for (var i = 0; i < sortedData.Count; i++)
        {
            sortedData[i].KyQuay = i + 1;
        }

        var latestRecord = sortedData.LastOrDefault();
        if (latestRecord is null)
        {
            return Fail(daiName, "Không lọc được dữ liệu đài.");
        }

        if (sortedData.Count < MinHistoryDraws)
        {
            return Fail(
                daiName,
                $"Cần ít nhất {MinHistoryDraws} kỳ (hiện có {sortedData.Count}).",
                latestRecord,
                gameKind);
        }

        try
        {
            var historicalDates = sortedData.Select(static r => r.NgayQuay).ToList();
            var giaiDbTruoc = ResolveGiaiDacBietTruoc(latestRecord, gameKind);
            var giaiDbDigitCount = gameKind switch
            {
                LottoGameKinds.XsMienBac => 5,
                LottoGameKinds.XsMienNam or LottoGameKinds.XsMienTrung => 6,
                _ => 0
            };

            List<int> predictions;
            if (IsVietlott(gameKind))
            {
                var trained = await Task.Run(
                    () => LotterySsaTrainer.TrainSlots(sortedData, profile),
                    cancellationToken);
                var (_, rounded) = LotterySsaTrainer.ForecastSlots(trained, profile);
                predictions = rounded
                    .Select(n => ClampLotto(n, profile.MinNumber, profile.MaxNumber))
                    .Take(profile.PredictCount)
                    .ToList();
            }
            else if (IsSouthernRegion(gameKind))
            {
                predictions = [];
            }
            else
            {
                predictions = [];
            }

            List<int> giaiDbDuDoan = [];
            if (giaiDbDigitCount > 0)
            {
                (_, giaiDbDuDoan) = LotterySsaTrainer.ForecastGiaiDacBiet(sortedData, giaiDbDigitCount);
                if (giaiDbDuDoan.Count == 0 &&
                    giaiDbTruoc.Count >= giaiDbDigitCount &&
                    gameKind == LottoGameKinds.XsMienBac)
                {
                    giaiDbDuDoan = giaiDbTruoc.Take(giaiDbDigitCount).ToList();
                }

                if (IsSouthernRegion(gameKind) && giaiDbDuDoan.Count < giaiDbDigitCount)
                {
                    var dbReady = LotteryRecordNormalizer.CountWithGiaiDacBiet(sortedData, giaiDbDigitCount);
                    return Fail(
                        daiName,
                        $"Không dự đoán được giải ĐB: cần ≥12 kỳ có đủ {giaiDbDigitCount} chữ số giải ĐB (hiện {dbReady}). " +
                        "Bật cào Minh Ngọc để lấy lịch sử giải đặc biệt.",
                        latestRecord,
                        gameKind);
                }

                if (gameKind == LottoGameKinds.XsMienBac || IsSouthernRegion(gameKind))
                {
                    predictions = giaiDbDuDoan.ToList();
                }
            }

            var loExtras = IsVietlott(gameKind)
                ? null
                : TrimLotoExtras(
                    LottoLotoPredictor.Predict(sortedData, gameKind),
                    gameKind);

            var nextDrawDate = LottoDrawSchedule.ComputeNextDrawDate(
                latestRecord.NgayQuay,
                daiCode ?? (gameKind == LottoGameKinds.XsMienBac
                    ? LottoGameCatalog.ResolveMienBacDaiCodeForDate(latestRecord.NgayQuay.AddDays(1))
                    : null),
                gameKind,
                historicalDates);

            var dbParseMode = gameKind switch
            {
                LottoGameKinds.XsMienBac => LottoParseMode.SpecialPrizeFiveDigits,
                LottoGameKinds.XsMienNam or LottoGameKinds.XsMienTrung => LottoParseMode.SouthernSpecialPrizeSixDigits,
                _ => profile.ParseMode
            };

            return new PredictionOutputDto
            {
                TenDai = daiName,
                KyQuayTruoc = latestRecord.KyQuay,
                NgayQuayKyTruoc = latestRecord.NgayQuay.ToString("dd/MM/yyyy", ViCulture),
                KetQuaKyTruoc = latestRecord.CacSoDaVe.ToList(),
                KyQuayDuDoan = latestRecord.KyQuay + 1,
                NgayQuayDuDoan = nextDrawDate.ToString("dd/MM/yyyy", ViCulture),
                CacSoDuDoan = predictions,
                KetQuaGiaiDacBietTruoc = giaiDbTruoc,
                TatCaLoVeKyTruoc = latestRecord.TatCaLoVe.ToList(),
                DuDoanGiaiDacBiet = giaiDbDuDoan,
                GiaiDacBietTruocDinhDang = giaiDbDigitCount > 0
                    ? LottoDisplayFormatter.Format(profile with { ParseMode = dbParseMode }, giaiDbTruoc, null)
                    : string.Empty,
                GiaiDacBietDuDoanDinhDang = giaiDbDigitCount > 0 && giaiDbDuDoan.Count > 0
                    ? LottoDisplayFormatter.Format(profile with { ParseMode = dbParseMode }, giaiDbDuDoan, null)
                    : string.Empty,
                DuDoanLo = loExtras,
                SoKyDaHoc = sortedData.Count
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Lỗi SSA đài {Dai}", daiName);
            return Fail(daiName, ex.Message, latestRecord, gameKind);
        }
    }

    /// <summary>Ép float SSA về số nguyên lô 00–99.</summary>
    public static int ClampLotto(float raw, int min, int max)
    {
        var rounded = (int)MathF.Round(raw);
        if (rounded < 0)
        {
            rounded = (rounded % 100 + 100) % 100;
        }

        if (max <= 99)
        {
            return Math.Clamp(rounded % 100, 0, 99);
        }

        return Math.Clamp(rounded, min, max);
    }

    private async Task<List<LotteryRecord>> LoadDaiHistoryAsync(
        string gameKind,
        string? daiCode,
        string daiName,
        CancellationToken cancellationToken)
    {
        List<LotteryRecord> scraped = [];

        if (!string.IsNullOrWhiteSpace(daiCode) && profileUsesMinhNgoc(gameKind))
        {
            try
            {
                scraped = await _minhNgocScraper.FetchDaiHistoryAsync(
                    gameKind,
                    daiCode,
                    daiName,
                    MinHistoryDraws + 30,
                    cancellationToken);

                _logger.LogInformation(
                    "Minh Ngọc: đài {Dai} — {Count} kỳ, mới nhất {Date:dd/MM/yyyy}",
                    daiName,
                    scraped.Count,
                    scraped.LastOrDefault()?.NgayQuay);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Không cào được Minh Ngọc cho đài {Dai}", daiName);
            }
        }
        else if (gameKind == LottoGameKinds.XsMienBac)
        {
            try
            {
                scraped = await _minhNgocScraper.FetchMienBacHistoryAsync(
                    daiCode!,
                    daiName,
                    MinHistoryDraws + 10,
                    cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Không cào được XSMB từ Minh Ngọc");
            }
        }
        else if (IsVietlott(gameKind))
        {
            try
            {
                scraped = await _vietlottScraper.FetchHistoryAsync(
                    gameKind,
                    MinHistoryDraws + 10,
                    cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Không cào được Vietlott {Game}", gameKind);
            }
        }

        var local = await _lotteryLoader.LoadDatasetAsync(gameKind, daiCode, cancellationToken);
        var localDai = local
            .Where(r => LotteryRecordNormalizer.NormalizeDaiKey(r.Dai) ==
                        LotteryRecordNormalizer.NormalizeDaiKey(daiName))
            .ToList();

        var merged = LotteryRecordNormalizer.MergeDaiHistory(scraped, localDai);
        var giaiDbDigits = gameKind switch
        {
            LottoGameKinds.XsMienBac => 5,
            LottoGameKinds.XsMienNam or LottoGameKinds.XsMienTrung => 6,
            _ => 0
        };
        var dbReadyCount = giaiDbDigits > 0
            ? LotteryRecordNormalizer.CountWithGiaiDacBiet(merged, giaiDbDigits)
            : merged.Count;

        if (_scrapeSettings.Get().PreferLocalDataFirst &&
            localDai.Count >= MinHistoryDraws &&
            (giaiDbDigits == 0 || dbReadyCount >= MinHistoryDraws))
        {
            for (var i = 0; i < merged.Count; i++)
            {
                merged[i].KyQuay = i + 1;
            }

            return merged;
        }

        for (var i = 0; i < merged.Count; i++)
        {
            merged[i].KyQuay = i + 1;
        }

        _logger.LogInformation(
            "Lịch sử đài {Dai}: {Total} kỳ, {DbReady} kỳ có giải ĐB ({Digits} chữ số)",
            daiName,
            merged.Count,
            dbReadyCount,
            giaiDbDigits);

        return merged;
    }

    private static bool profileUsesMinhNgoc(string gameKind) =>
        gameKind is LottoGameKinds.XsMienNam or LottoGameKinds.XsMienTrung;

    private static string ResolveDaiName(string gameKind, string? daiCode)
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

    private static PredictionOutputDto Fail(
        string daiName,
        string message,
        LotteryRecord? latest = null,
        string? gameKind = null)
    {
        var dto = new PredictionOutputDto
        {
            TenDai = daiName,
            ThongBaoLoi = message
        };

        if (latest is null)
        {
            return dto;
        }

        dto.KyQuayTruoc = latest.KyQuay;
        dto.NgayQuayKyTruoc = latest.NgayQuay.ToString("dd/MM/yyyy", ViCulture);
        dto.KetQuaKyTruoc = latest.CacSoDaVe.ToList();
        dto.KetQuaGiaiDacBietTruoc = ResolveGiaiDacBietTruoc(latest, gameKind);
        dto.TatCaLoVeKyTruoc = latest.TatCaLoVe.ToList();
        return dto;
    }

    private static List<int> ResolveGiaiDacBietTruoc(LotteryRecord record, string? gameKind)
    {
        if (record.GiaiDacBietSo.Count > 0)
        {
            return record.GiaiDacBietSo.ToList();
        }

        if (gameKind == LottoGameKinds.XsMienBac && record.CacSoDaVe.Count >= 5)
        {
            return record.CacSoDaVe.Take(5).ToList();
        }

        return [];
    }

    private static readonly System.Globalization.CultureInfo ViCulture =
        System.Globalization.CultureInfo.GetCultureInfo("vi-VN");
}
