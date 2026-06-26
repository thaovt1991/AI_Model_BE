using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI_Model_BE.Services;

/// <summary>Dự đoán bạch thủ, song thủ, xiên 2/3/4 từ chuỗi lô lịch sử + SSA.</summary>
internal static class LottoLotoPredictor
{
    private const int MinPoints = 12;

    public static LottoLotoExtras Predict(IReadOnlyList<LotteryRecord> history, string gameKind)
    {
        if (history.Count < MinPoints)
        {
            return new LottoLotoExtras();
        }

        var ordered = history.OrderBy(static r => r.NgayQuay).ToList();
        var dbLoSeries = ordered.Select(r => LottoLotoAnalyzer.GetSpecialPrizeLo2(r, gameKind)).ToList();
        var bachThu = ForecastMod100(dbLoSeries);

        var songThu = BuildSongThu(bachThu, dbLoSeries);
        var candidates = BuildCandidatePool(ordered, gameKind, bachThu);

        return new LottoLotoExtras
        {
            BachThuLo = bachThu,
            SongThuLo = songThu,
            Xien2 = candidates.Take(2).ToList(),
            Xien3 = candidates.Take(3).ToList(),
            Xien4 = candidates.Take(4).ToList()
        };
    }

    private static List<int> BuildSongThu(int bachThu, List<int> dbLoSeries)
    {
        var reversed = ReverseLo(bachThu);
        var second = ForecastMod100(dbLoSeries, skipLast: 1);
        if (second == bachThu || second == reversed)
        {
            second = (bachThu + 11) % 100;
        }

        var result = new List<int> { bachThu };
        if (reversed != bachThu)
        {
            result.Add(reversed);
        }
        else if (!result.Contains(second))
        {
            result.Add(second);
        }

        while (result.Count < 2)
        {
            var filler = (result[^1] + 7) % 100;
            if (!result.Contains(filler))
            {
                result.Add(filler);
            }
            else
            {
                break;
            }
        }

        return result.Take(2).ToList();
    }

    private static List<int> BuildCandidatePool(
        IReadOnlyList<LotteryRecord> ordered,
        string gameKind,
        int seed)
    {
        var pool = new List<int> { seed };
        var reversed = ReverseLo(seed);
        if (!pool.Contains(reversed))
        {
            pool.Add(reversed);
        }

        var hot = LottoLotoAnalyzer.TopFrequency(ordered, gameKind, 30, 10)
            .Select(static x => x.So);
        foreach (var lo in hot)
        {
            if (!pool.Contains(lo))
            {
                pool.Add(lo);
            }
        }

        var gan = LottoLotoAnalyzer.ComputeLoGan(ordered, gameKind, 15)
            .OrderByDescending(static g => g.SoNgayChuaVe)
            .Select(static g => g.So);
        foreach (var lo in gan)
        {
            if (!pool.Contains(lo))
            {
                pool.Add(lo);
            }
        }

        var tailSeries = ordered
            .Select(r => LottoLotoAnalyzer.GetSpecialPrizeLo2(r, gameKind))
            .ToList();
        for (var offset = 0; offset < 3; offset++)
        {
            var forecast = ForecastMod100(tailSeries, skipLast: offset);
            if (!pool.Contains(forecast))
            {
                pool.Add(forecast);
            }
        }

        while (pool.Count < 4)
        {
            var next = (pool[^1] + 13) % 100;
            if (!pool.Contains(next))
            {
                pool.Add(next);
            }
            else
            {
                break;
            }
        }

        return pool;
    }

    private static int ForecastMod100(IReadOnlyList<int> series, int skipLast = 0)
    {
        var usable = series.Take(series.Count - skipLast).ToList();
        if (usable.Count < MinPoints)
        {
            return usable.Count > 0 ? usable[^1] : 0;
        }

        try
        {
            var mlContext = new MLContext(seed: 42);
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
            return Math.Clamp((int)MathF.Round(value) % 100, 0, 99);
        }
        catch
        {
            return usable[^1];
        }
    }

    private static int ReverseLo(int lo)
    {
        var tens = lo / 10;
        var units = lo % 10;
        return units * 10 + tens;
    }
}
