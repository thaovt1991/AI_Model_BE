using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI_Model_BE.Services;

/// <summary>Dự đoán bạch thủ, song thủ, xiên 2/3/4 từ chuỗi lô lịch sử + SSA.</summary>
internal static class LottoLotoPredictor
{
    private const int MinPoints = 12;

    public static LottoLotoExtras Predict(
        IReadOnlyList<LotteryRecord> history,
        string gameKind,
        string? daiCode = null)
    {
        if (history.Count < MinPoints)
        {
            return new LottoLotoExtras();
        }

        var ordered = history.OrderBy(static r => r.NgayQuay).ToList();
        var dbTailSeries = ordered.Select(r => LottoLotoAnalyzer.GetSpecialPrizeLo2(r, gameKind)).ToList();
        var dbHeadSeries = ordered.Select(r => LottoLotoAnalyzer.GetSpecialPrizeHeadLo2(r, gameKind)).ToList();
        var dbMidSeries = ordered.Select(r => LottoLotoAnalyzer.GetSpecialPrizeMidLo2(r, gameKind)).ToList();
        var repLoSeries = ordered.Select(r => LottoLotoAnalyzer.GetDrawRepresentativeLo(r, gameKind)).ToList();

        var seriesKey = BuildSeriesKey(ordered, gameKind, daiCode);
        var bachThu = ResolveBachThu(dbTailSeries, repLoSeries, dbMidSeries, seriesKey);
        var songThu = BuildSongThu(bachThu, ordered, gameKind, dbHeadSeries, repLoSeries, dbMidSeries, seriesKey);
        var xien2 = BuildXien2(ordered, gameKind, bachThu, songThu, dbTailSeries, seriesKey);
        var candidates = BuildCandidatePool(ordered, gameKind, bachThu, songThu, xien2, seriesKey);

        return new LottoLotoExtras
        {
            BachThuLo = bachThu,
            SongThuLo = songThu,
            Xien2 = xien2,
            Xien3 = candidates.Take(3).ToList(),
            Xien4 = candidates.Take(4).ToList()
        };
    }

    private static int ResolveBachThu(
        IReadOnlyList<int> dbTailSeries,
        IReadOnlyList<int> repLoSeries,
        IReadOnlyList<int> dbMidSeries,
        int seriesKey)
    {
        var fromTail = ForecastMod100(dbTailSeries, seriesKey);
        var fromRep = ForecastMod100(repLoSeries, seriesKey ^ 0x5A3C);
        var fromMid = ForecastMod100(dbMidSeries, seriesKey ^ 0x2F19);

        var blended = (fromTail + fromRep + fromMid) / 3;
        if (fromRep != fromTail || fromMid != fromTail)
        {
            return NormalizeLo(blended + (fromTail % 7));
        }

        return fromTail;
    }

    private static List<int> BuildSongThu(
        int bachThu,
        IReadOnlyList<LotteryRecord> ordered,
        string gameKind,
        IReadOnlyList<int> dbHeadSeries,
        IReadOnlyList<int> repLoSeries,
        IReadOnlyList<int> dbMidSeries,
        int seriesKey)
    {
        var reversed = ReverseLo(bachThu);
        var picks = new List<int> { bachThu };

        void TrySecond(int value)
        {
            value = NormalizeLo(value);
            if (value != bachThu && value != reversed && !picks.Contains(value))
            {
                picks.Add(value);
            }
        }

        TrySecond(ForecastMod100(dbHeadSeries, seriesKey ^ 0x7E11));
        TrySecond(ForecastMod100(repLoSeries, seriesKey ^ 0x9B2F, skipLast: 1));
        TrySecond(ForecastMod100(dbMidSeries, seriesKey ^ 0xB812, skipLast: 1));

        foreach (var lo in LottoLotoAnalyzer.ComputeLoGan(ordered, gameKind, 5)
                     .OrderByDescending(static g => g.SoNgayChuaVe)
                     .Select(static g => g.So))
        {
            TrySecond(lo);
            if (picks.Count >= 2)
            {
                break;
            }
        }

        foreach (var lo in LottoLotoAnalyzer.TopFrequency(ordered, gameKind, 20, 5).Select(static x => x.So))
        {
            TrySecond(lo);
            if (picks.Count >= 2)
            {
                break;
            }
        }

        TrySecond(reversed);

        while (picks.Count < 2)
        {
            TrySecond((bachThu + 11 + picks.Count) % 100);
        }

        return picks.Take(2).ToList();
    }

    private static List<int> BuildXien2(
        IReadOnlyList<LotteryRecord> ordered,
        string gameKind,
        int bachThu,
        IReadOnlyList<int> songThu,
        IReadOnlyList<int> dbTailSeries,
        int seriesKey)
    {
        var exclude = new HashSet<int>(songThu) { bachThu, ReverseLo(bachThu) };
        var picks = new List<int>();

        void TryAdd(int value)
        {
            value = NormalizeLo(value);
            if (value < 0 || picks.Contains(value))
            {
                return;
            }

            picks.Add(value);
        }

        foreach (var lo in LottoLotoAnalyzer.TopFrequency(ordered, gameKind, 30, 12).Select(static x => x.So))
        {
            if (!exclude.Contains(lo))
            {
                TryAdd(lo);
            }

            if (picks.Count >= 2)
            {
                return picks.Take(2).ToList();
            }
        }

        foreach (var lo in LottoLotoAnalyzer.ComputeLoGan(ordered, gameKind, 15)
                     .OrderByDescending(static g => g.SoNgayChuaVe)
                     .Select(static g => g.So))
        {
            if (!exclude.Contains(lo))
            {
                TryAdd(lo);
            }

            if (picks.Count >= 2)
            {
                return picks.Take(2).ToList();
            }
        }

        TryAdd(ForecastMod100(dbTailSeries, seriesKey ^ 0xD1CE, skipLast: 1));
        TryAdd(ForecastMod100(dbTailSeries, seriesKey ^ 0xE2AF, skipLast: 2));

        while (picks.Count < 2)
        {
            TryAdd((picks.Count == 0 ? bachThu + 19 : picks[0] + 23) % 100);
        }

        return picks.Take(2).ToList();
    }

    private static List<int> BuildCandidatePool(
        IReadOnlyList<LotteryRecord> ordered,
        string gameKind,
        int bachThu,
        IReadOnlyList<int> songThu,
        IReadOnlyList<int> xien2,
        int seriesKey)
    {
        var pool = new List<int>();
        foreach (var lo in xien2.Concat(songThu))
        {
            TryAddUnique(pool, lo);
        }

        TryAddUnique(pool, bachThu);

        var hot = LottoLotoAnalyzer.TopFrequency(ordered, gameKind, 30, 10)
            .Select(static x => x.So);
        foreach (var lo in hot)
        {
            TryAddUnique(pool, lo);
        }

        var gan = LottoLotoAnalyzer.ComputeLoGan(ordered, gameKind, 15)
            .OrderByDescending(static g => g.SoNgayChuaVe)
            .Select(static g => g.So);
        foreach (var lo in gan)
        {
            TryAddUnique(pool, lo);
        }

        var tailSeries = ordered
            .Select(r => LottoLotoAnalyzer.GetSpecialPrizeLo2(r, gameKind))
            .ToList();
        for (var offset = 0; offset < 3; offset++)
        {
            TryAddUnique(pool, ForecastMod100(tailSeries, seriesKey ^ (offset * 31), skipLast: offset));
        }

        while (pool.Count < 4)
        {
            var next = (pool[^1] + 13) % 100;
            if (!TryAddUnique(pool, next))
            {
                break;
            }
        }

        return pool;
    }

    private static bool TryAddUnique(List<int> pool, int value)
    {
        value = NormalizeLo(value);
        if (pool.Contains(value))
        {
            return false;
        }

        pool.Add(value);
        return true;
    }

    private static int ForecastMod100(IReadOnlyList<int> series, int seriesKey, int skipLast = 0)
    {
        var usable = series.Take(series.Count - skipLast).ToList();
        if (usable.Count < MinPoints)
        {
            return usable.Count > 0 ? usable[^1] : 0;
        }

        try
        {
            var mlContext = new MLContext(seed: seriesKey);
            var points = usable.Select(v => new LottoSsaPoint { GiaTri = v }).ToList();
            var effectiveSeriesLength = Math.Min(24, points.Count);
            var effectiveTrainSize = Math.Min(18, Math.Max(2, points.Count - 1));
            var dataView = mlContext.Data.LoadFromEnumerable(points);

            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(LottoSsaForecast.DuBao),
                inputColumnName: nameof(LottoSsaPoint.GiaTri),
                windowSize: 5,
                seriesLength: effectiveSeriesLength,
                trainSize: effectiveTrainSize,
                horizon: 1);

            var model = pipeline.Fit(dataView);
            var engine = model.CreateTimeSeriesEngine<LottoSsaPoint, LottoSsaForecast>(mlContext);
            var forecast = engine.Predict();
            var value = forecast.DuBao is { Length: > 0 } ? forecast.DuBao[0] : usable[^1];
            return NormalizeLo((int)MathF.Round(value));
        }
        catch
        {
            return usable[^1];
        }
    }

    private static int BuildSeriesKey(
        IReadOnlyList<LotteryRecord> ordered,
        string gameKind,
        string? daiCode)
    {
        var latest = ordered[^1];
        var hash = new HashCode();
        hash.Add(gameKind);
        hash.Add(daiCode ?? latest.Dai);
        hash.Add(latest.Dai);
        hash.Add(latest.NgayQuay.Date);
        hash.Add(latest.KyQuay);
        hash.Add(ordered.Count);
        return Math.Abs(hash.ToHashCode()) % 10_000 + 42;
    }

    private static int NormalizeLo(int value) => ((value % 100) + 100) % 100;

    private static int ReverseLo(int lo)
    {
        var tens = lo / 10;
        var units = lo % 10;
        return units * 10 + tens;
    }
}
