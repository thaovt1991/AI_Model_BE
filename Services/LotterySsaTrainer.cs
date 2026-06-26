using AI_Model_BE.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI_Model_BE.Services;

/// <summary>
/// Huấn luyện ForecastingBySsa cho một chuỗi thời gian của đài.
/// windowSize / seriesLength / trainSize điều khiển độ sâu phân tích SSA.
/// </summary>
internal static class LotterySsaTrainer
{
    // windowSize: cửa sổ trượt khi tách thành phần chuỗi (càng lớn càng mượt, cần nhiều điểm hơn)
    private const int WindowSize = 5;

    // seriesLength: số điểm gần nhất đưa vào mô hình SSA
    private const int SeriesLength = 24;

    // trainSize: số điểm dùng fit trong cửa sổ train (phải &lt; seriesLength)
    private const int TrainSize = 18;

    private const int Horizon = 1;
    private const int MinPoints = 20;
    private const int MinPointsGiaiDb = 12;

    public static SsaDaiTrainResult TrainSlots(IReadOnlyList<LotteryRecord> daiRecords, LottoGameProfile profile)
    {
        if (daiRecords.Count < MinPoints)
        {
            throw new InvalidOperationException(
                $"Đài {daiRecords.FirstOrDefault()?.Dai}: cần ít nhất {MinPoints} kỳ (có {daiRecords.Count}).");
        }

        var slotCount = ResolveSlotCount(daiRecords, profile);
        var mlContext = new MLContext(seed: 42);
        var engines = new Dictionary<int, TimeSeriesPredictionEngine<LottoSsaPoint, LottoSsaForecast>>();
        var frequency = BuildFrequency(daiRecords);

        for (var slot = 0; slot < slotCount; slot++)
        {
            var series = BuildSlotSeries(daiRecords, slot);
            if (series.Count < MinPoints)
            {
                continue;
            }

            var effectiveSeriesLength = Math.Min(SeriesLength, series.Count);
            var effectiveTrainSize = Math.Min(TrainSize, Math.Max(2, series.Count - 1));

            var dataView = mlContext.Data.LoadFromEnumerable(series);

        // Pipeline SSA — horizon = 1: nhảy đúng 1 bước sau điểm cuối chuỗi đã sắp xếp tăng dần theo ngày
            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(LottoSsaForecast.DuBao),
                inputColumnName: nameof(LottoSsaPoint.GiaTri),
                windowSize: WindowSize,
                seriesLength: effectiveSeriesLength,
                trainSize: effectiveTrainSize,
                horizon: Horizon);

            var model = pipeline.Fit(dataView);
            engines[slot] = model.CreateTimeSeriesEngine<LottoSsaPoint, LottoSsaForecast>(mlContext);
        }

        if (engines.Count == 0)
        {
            throw new InvalidOperationException("Không tạo được engine SSA cho đài này.");
        }

        return new SsaDaiTrainResult(engines, frequency, slotCount);
    }

    public static (List<float> Raw, List<int> Rounded) ForecastSlots(
        SsaDaiTrainResult trained,
        LottoGameProfile profile)
    {
        var raw = new List<float>();
        var rounded = new List<int>();

        foreach (var slot in Enumerable.Range(0, trained.SlotCount).OrderBy(static i => i))
        {
            if (!trained.Engines.TryGetValue(slot, out var engine))
            {
                continue;
            }

            var forecast = engine.Predict();
            var value = forecast.DuBao is { Length: > 0 } ? forecast.DuBao[0] : 0f;
            raw.Add(value);
            rounded.Add(Clamp(value, profile.MinNumber, profile.MaxNumber));
        }

        ApplyProfileRules(raw, rounded, trained, profile);
        return (raw, rounded);
    }

    /// <summary>SSA riêng cho 6 chữ số giải đặc biệt (MN/MT) — mỗi digit là một slot 0–9.</summary>
    public static (List<float> Raw, List<int> Rounded) ForecastGiaiDacBiet(
        IReadOnlyList<LotteryRecord> daiRecords,
        int digitCount = 6)
    {
        var usable = daiRecords
            .Where(r => r.GiaiDacBietSo.Count >= digitCount)
            .ToList();

        if (usable.Count < MinPointsGiaiDb)
        {
            return ([], []);
        }

        var mlContext = new MLContext(seed: 42);
        var raw = new List<float>();
        var rounded = new List<int>();

        for (var slot = 0; slot < digitCount; slot++)
        {
            var series = BuildGiaiDbSlotSeries(usable, slot);
            if (series.Count < MinPointsGiaiDb)
            {
                continue;
            }

            var effectiveSeriesLength = Math.Min(SeriesLength, series.Count);
            var effectiveTrainSize = Math.Min(TrainSize, Math.Max(2, series.Count - 1));
            var dataView = mlContext.Data.LoadFromEnumerable(series);

            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(LottoSsaForecast.DuBao),
                inputColumnName: nameof(LottoSsaPoint.GiaTri),
                windowSize: WindowSize,
                seriesLength: effectiveSeriesLength,
                trainSize: effectiveTrainSize,
                horizon: Horizon);

            var model = pipeline.Fit(dataView);
            var engine = model.CreateTimeSeriesEngine<LottoSsaPoint, LottoSsaForecast>(mlContext);
            var forecast = engine.Predict();
            var value = forecast.DuBao is { Length: > 0 } ? forecast.DuBao[0] : 0f;
            raw.Add(value);
            rounded.Add(Math.Clamp((int)MathF.Round(value), 0, 9));
        }

        while (rounded.Count < digitCount)
        {
            rounded.Add(0);
            raw.Add(0f);
        }

        return (raw, rounded.Take(digitCount).ToList());
    }

    private static List<LottoSsaPoint> BuildGiaiDbSlotSeries(IReadOnlyList<LotteryRecord> records, int slot)
    {
        var series = new List<LottoSsaPoint>();
        foreach (var draw in records)
        {
            if (draw.GiaiDacBietSo.Count == 0)
            {
                continue;
            }

            var value = slot < draw.GiaiDacBietSo.Count ? draw.GiaiDacBietSo[slot] : draw.GiaiDacBietSo[^1];
            series.Add(new LottoSsaPoint { GiaTri = value });
        }

        return series;
    }

    private static void ApplyProfileRules(
        List<float> raw,
        List<int> rounded,
        SsaDaiTrainResult trained,
        LottoGameProfile profile)
    {
        if (profile.ParseMode == LottoParseMode.Vietlott645)
        {
            var unique = PickUnique(rounded, 6, profile.MinNumber, profile.MaxNumber, trained.Frequency);
            rounded.Clear();
            rounded.AddRange(unique);
            while (raw.Count > rounded.Count)
            {
                raw.RemoveAt(raw.Count - 1);
            }
        }
        else if (profile.ParseMode == LottoParseMode.Vietlott655)
        {
            var unique = PickUnique(rounded, 7, profile.MinNumber, profile.MaxNumber, trained.Frequency);
            rounded.Clear();
            rounded.AddRange(unique);
        }
        else if (profile.ParseMode is LottoParseMode.SpecialPrizeFiveDigits or LottoParseMode.SouthernSpecialPrizeSixDigits)
        {
            for (var i = 0; i < rounded.Count; i++)
            {
                rounded[i] = Math.Clamp(rounded[i], 0, 9);
            }
        }
        else
        {
            var unique = PickUnique(rounded, profile.PredictCount, profile.MinNumber, profile.MaxNumber, trained.Frequency);
            rounded.Clear();
            rounded.AddRange(unique);
        }
    }

    private static int ResolveSlotCount(IReadOnlyList<LotteryRecord> records, LottoGameProfile profile)
    {
        if (profile.ParseMode == LottoParseMode.Vietlott655)
        {
            return 7;
        }

        if (profile.ParseMode == LottoParseMode.SpecialPrizeFiveDigits)
        {
            return 5;
        }

        return Math.Max(profile.PredictCount, records.Max(r => r.CacSoDaVe.Count));
    }

    private static Dictionary<int, int> BuildFrequency(IReadOnlyList<LotteryRecord> records)
    {
        var freq = new Dictionary<int, int>();
        foreach (var n in records.SelectMany(static r => r.CacSoDaVe))
        {
            freq[n] = freq.GetValueOrDefault(n) + 1;
        }

        return freq;
    }

    private static List<LottoSsaPoint> BuildSlotSeries(IReadOnlyList<LotteryRecord> records, int slot)
    {
        var series = new List<LottoSsaPoint>();
        foreach (var draw in records)
        {
            if (draw.CacSoDaVe.Count == 0)
            {
                continue;
            }

            var value = slot < draw.CacSoDaVe.Count ? draw.CacSoDaVe[slot] : draw.CacSoDaVe[^1];
            series.Add(new LottoSsaPoint { GiaTri = value });
        }

        return series;
    }

    private static List<int> PickUnique(
        List<int> seed,
        int count,
        int min,
        int max,
        Dictionary<int, int> frequency)
    {
        var result = new List<int>();
        foreach (var n in seed.Select(v => Math.Clamp(v, min, max)))
        {
            if (!result.Contains(n))
            {
                result.Add(n);
            }

            if (result.Count >= count)
            {
                return result;
            }
        }

        foreach (var (n, _) in frequency.OrderByDescending(static kv => kv.Value))
        {
            if (n < min || n > max || result.Contains(n))
            {
                continue;
            }

            result.Add(n);
            if (result.Count >= count)
            {
                break;
            }
        }

        for (var fallback = min; result.Count < count && fallback <= max; fallback++)
        {
            if (!result.Contains(fallback))
            {
                result.Add(fallback);
            }
        }

        return result;
    }

    private static int Clamp(float value, int min, int max) =>
        Math.Clamp((int)MathF.Round(value), min, max);
}

internal sealed record SsaDaiTrainResult(
    Dictionary<int, TimeSeriesPredictionEngine<LottoSsaPoint, LottoSsaForecast>> Engines,
    Dictionary<int, int> Frequency,
    int SlotCount);
