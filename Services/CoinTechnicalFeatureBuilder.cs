using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// =====================================================================
/// FEATURE ENGINEERING — chỉ báo kỹ thuật dùng cho LightGBM
/// =====================================================================
/// Các chỉ báo này là "ngôn ngữ" phổ biến nhất trong quantitative trading /
/// cuộc thi Kaggle crypto: RSI, MACD, SMA/EMA, Bollinger, returns, volume.
/// LightGBM học quan hệ phi tuyến giữa chúng → dự đoán log-return kỳ tới.
/// </summary>
public static class CoinTechnicalFeatureBuilder
{
    public sealed class FeatureRow
    {
        // --- Input features ---
        public float Return1 { get; set; }
        public float Return3 { get; set; }
        public float Return6 { get; set; }
        public float Return12 { get; set; }
        public float Volatility6 { get; set; }
        public float Volatility12 { get; set; }
        public float Rsi14 { get; set; }
        public float Macd { get; set; }
        public float MacdSignal { get; set; }
        public float MacdHist { get; set; }
        public float SmaRatio7 { get; set; }
        public float SmaRatio21 { get; set; }
        public float EmaRatio12 { get; set; }
        public float BbWidth { get; set; }
        public float BbPosition { get; set; }
        public float VolumeChange { get; set; }
        public float HighLowRange { get; set; }
        public float CloseLocation { get; set; }

        // --- Label: log(close[t+1] / close[t]) ---
        public float NextLogReturn { get; set; }

        // Không đưa vào train — chỉ để debug / suy luận cuối
        public double Close { get; set; }
        public DateTime OpenTimeUtc { get; set; }
    }

    /// <summary>
    /// Xây dataset có nhãn (trừ nến cuối — chưa có tương lai).
    /// </summary>
    public static List<FeatureRow> BuildLabeled(IReadOnlyList<CoinCandle> candles)
    {
        var rows = BuildAll(candles);
        // Bỏ hàng cuối (chưa có NextLogReturn thật) và các hàng warm-up NaN
        return rows
            .Where(r => !float.IsNaN(r.NextLogReturn) && IsFiniteFeatures(r))
            .ToList();
    }

    /// <summary>Hàng feature tại nến mới nhất — dùng để predict (NextLogReturn = NaN).</summary>
    public static FeatureRow BuildLatest(IReadOnlyList<CoinCandle> candles)
    {
        var rows = BuildAll(candles);
        if (rows.Count == 0)
        {
            throw new InvalidOperationException("Không đủ dữ liệu để tính chỉ báo kỹ thuật.");
        }

        return rows[^1];
    }

    public static List<FeatureRow> BuildAll(IReadOnlyList<CoinCandle> candles)
    {
        if (candles.Count < 40)
        {
            throw new InvalidOperationException("Cần tối thiểu ~40 nến để tính RSI/MACD/Bollinger.");
        }

        var closes = candles.Select(c => c.Close).ToArray();
        var volumes = candles.Select(c => c.Volume).ToArray();
        var sma7 = Sma(closes, 7);
        var sma21 = Sma(closes, 21);
        var ema12 = Ema(closes, 12);
        var ema26 = Ema(closes, 26);
        var rsi = Rsi(closes, 14);
        var (macd, signal, hist) = Macd(closes, 12, 26, 9);
        var (bbMid, bbUpper, bbLower) = Bollinger(closes, 20, 2.0);

        var rows = new List<FeatureRow>(candles.Count);
        for (var i = 0; i < candles.Count; i++)
        {
            var c = candles[i];
            var nextLog = i + 1 < candles.Count
                ? (float)Math.Log(Math.Max(candles[i + 1].Close, 1e-12) / Math.Max(c.Close, 1e-12))
                : float.NaN;

            rows.Add(new FeatureRow
            {
                Return1 = (float)LogReturn(closes, i, 1),
                Return3 = (float)LogReturn(closes, i, 3),
                Return6 = (float)LogReturn(closes, i, 6),
                Return12 = (float)LogReturn(closes, i, 12),
                Volatility6 = (float)Volatility(closes, i, 6),
                Volatility12 = (float)Volatility(closes, i, 12),
                Rsi14 = (float)rsi[i],
                Macd = (float)macd[i],
                MacdSignal = (float)signal[i],
                MacdHist = (float)hist[i],
                SmaRatio7 = sma7[i] > 0 ? (float)(c.Close / sma7[i] - 1) : 0f,
                SmaRatio21 = sma21[i] > 0 ? (float)(c.Close / sma21[i] - 1) : 0f,
                EmaRatio12 = ema12[i] > 0 ? (float)(c.Close / ema12[i] - 1) : 0f,
                BbWidth = bbMid[i] > 0 ? (float)((bbUpper[i] - bbLower[i]) / bbMid[i]) : 0f,
                BbPosition = (bbUpper[i] - bbLower[i]) > 1e-12
                    ? (float)((c.Close - bbLower[i]) / (bbUpper[i] - bbLower[i]))
                    : 0.5f,
                VolumeChange = i > 0 && volumes[i - 1] > 0
                    ? (float)(volumes[i] / volumes[i - 1] - 1)
                    : 0f,
                HighLowRange = c.Close > 0 ? (float)((c.High - c.Low) / c.Close) : 0f,
                CloseLocation = (c.High - c.Low) > 1e-12
                    ? (float)((c.Close - c.Low) / (c.High - c.Low))
                    : 0.5f,
                NextLogReturn = nextLog,
                Close = c.Close,
                OpenTimeUtc = c.OpenTimeUtc
            });
        }

        return rows;
    }

    private static bool IsFiniteFeatures(FeatureRow r) =>
        float.IsFinite(r.Return1) && float.IsFinite(r.Rsi14) && float.IsFinite(r.Macd) &&
        float.IsFinite(r.SmaRatio7) && float.IsFinite(r.BbWidth) && float.IsFinite(r.Volatility6);

    private static double LogReturn(double[] closes, int i, int lag)
    {
        if (i < lag || closes[i - lag] <= 0 || closes[i] <= 0)
        {
            return 0;
        }

        return Math.Log(closes[i] / closes[i - lag]);
    }

    private static double Volatility(double[] closes, int i, int window)
    {
        if (i < window)
        {
            return 0;
        }

        var rets = new List<double>(window);
        for (var j = i - window + 1; j <= i; j++)
        {
            if (j > 0 && closes[j - 1] > 0)
            {
                rets.Add(Math.Log(closes[j] / closes[j - 1]));
            }
        }

        if (rets.Count < 2)
        {
            return 0;
        }

        var mean = rets.Average();
        var varSum = rets.Sum(r => (r - mean) * (r - mean));
        return Math.Sqrt(varSum / (rets.Count - 1));
    }

    private static double[] Sma(double[] values, int period)
    {
        var result = new double[values.Length];
        double sum = 0;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
            if (i >= period)
            {
                sum -= values[i - period];
            }

            result[i] = i >= period - 1 ? sum / period : values[i];
        }

        return result;
    }

    private static double[] Ema(double[] values, int period)
    {
        var result = new double[values.Length];
        var k = 2.0 / (period + 1);
        result[0] = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            result[i] = values[i] * k + result[i - 1] * (1 - k);
        }

        return result;
    }

    private static double[] Rsi(double[] closes, int period)
    {
        var result = new double[closes.Length];
        if (closes.Length < 2)
        {
            return result;
        }

        double avgGain = 0, avgLoss = 0;
        for (var i = 1; i <= Math.Min(period, closes.Length - 1); i++)
        {
            var diff = closes[i] - closes[i - 1];
            if (diff >= 0)
            {
                avgGain += diff;
            }
            else
            {
                avgLoss -= diff;
            }
        }

        avgGain /= period;
        avgLoss /= period;
        result[period] = avgLoss < 1e-12 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

        for (var i = period + 1; i < closes.Length; i++)
        {
            var diff = closes[i] - closes[i - 1];
            var gain = diff > 0 ? diff : 0;
            var loss = diff < 0 ? -diff : 0;
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
            result[i] = avgLoss < 1e-12 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
        }

        // Warm-up: copy first valid
        for (var i = 0; i < period; i++)
        {
            result[i] = result[period];
        }

        return result;
    }

    private static (double[] macd, double[] signal, double[] hist) Macd(
        double[] closes, int fast, int slow, int signalPeriod)
    {
        var emaFast = Ema(closes, fast);
        var emaSlow = Ema(closes, slow);
        var macd = new double[closes.Length];
        for (var i = 0; i < closes.Length; i++)
        {
            macd[i] = emaFast[i] - emaSlow[i];
        }

        var signal = Ema(macd, signalPeriod);
        var hist = new double[closes.Length];
        for (var i = 0; i < closes.Length; i++)
        {
            hist[i] = macd[i] - signal[i];
        }

        return (macd, signal, hist);
    }

    private static (double[] mid, double[] upper, double[] lower) Bollinger(
        double[] closes, int period, double stdMult)
    {
        var mid = Sma(closes, period);
        var upper = new double[closes.Length];
        var lower = new double[closes.Length];

        for (var i = 0; i < closes.Length; i++)
        {
            if (i < period - 1)
            {
                upper[i] = mid[i];
                lower[i] = mid[i];
                continue;
            }

            double sumSq = 0;
            for (var j = i - period + 1; j <= i; j++)
            {
                var d = closes[j] - mid[i];
                sumSq += d * d;
            }

            var std = Math.Sqrt(sumSq / period);
            upper[i] = mid[i] + stdMult * std;
            lower[i] = mid[i] - stdMult * std;
        }

        return (mid, upper, lower);
    }
}
