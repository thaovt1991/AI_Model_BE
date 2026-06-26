using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Phân tích lô 00–99 từ lịch sử quay — lô gan, tần suất.</summary>
internal static class LottoLotoAnalyzer
{
    public static HashSet<int> ExtractAllLoto(LotteryRecord record, string gameKind)
    {
        if (record.TatCaLoVe.Count > 0)
        {
            return record.TatCaLoVe.ToHashSet();
        }

        if (gameKind == LottoGameKinds.XsMienBac)
        {
            return record.CacSoDaVe
                .Where(n => n is >= 0 and <= 99)
                .ToHashSet();
        }

        var pool = record.CacSoDaVe.Select(n => n % 100).ToHashSet();
        foreach (var digit in record.GiaiDacBietSo)
        {
            _ = digit;
        }

        if (record.GiaiDacBietSo.Count >= 2)
        {
            var db = string.Concat(record.GiaiDacBietSo.Select(static d => d.ToString()));
            if (db.Length >= 2)
            {
                pool.Add(ParseLo(db[^2..]));
            }
        }

        return pool;
    }

    public static int GetSpecialPrizeLo2(LotteryRecord record, string gameKind)
    {
        if (record.GiaiDacBietSo.Count >= 2)
        {
            var digits = string.Concat(record.GiaiDacBietSo.Select(static d => d.ToString()));
            return ParseLo(digits[^2..]);
        }

        if (gameKind == LottoGameKinds.XsMienBac && record.CacSoDaVe.Count >= 2)
        {
            return ParseLo(string.Concat(record.CacSoDaVe.Select(static d => d.ToString()))[^2..]);
        }

        var loto = ExtractAllLoto(record, gameKind);
        return loto.Count > 0 ? loto.Max() : 0;
    }

    public static List<LoGanItem> ComputeLoGan(
        IReadOnlyList<LotteryRecord> orderedHistory,
        string gameKind,
        int top = 30)
    {
        if (orderedHistory.Count == 0)
        {
            return [];
        }

        var lastSeen = new Dictionary<int, DateTime>();
        var maxGan = new Dictionary<int, int>();
        var currentGan = new Dictionary<int, int>();

        for (var n = 0; n <= 99; n++)
        {
            currentGan[n] = 0;
            maxGan[n] = 0;
        }

        foreach (var draw in orderedHistory.OrderBy(static r => r.NgayQuay))
        {
            var appeared = ExtractAllLoto(draw, gameKind);
            for (var n = 0; n <= 99; n++)
            {
                if (appeared.Contains(n))
                {
                    maxGan[n] = Math.Max(maxGan[n], currentGan[n]);
                    currentGan[n] = 0;
                    lastSeen[n] = draw.NgayQuay;
                }
                else
                {
                    currentGan[n]++;
                }
            }
        }

        var latestDate = orderedHistory.Max(static r => r.NgayQuay).Date;
        var items = Enumerable.Range(0, 100)
            .Select(n => new LoGanItem
            {
                So = n,
                SoNgayChuaVe = currentGan[n],
                GanMax = Math.Max(maxGan[n], currentGan[n]),
                NgayVeCuoi = lastSeen.GetValueOrDefault(n)
            })
            .OrderByDescending(static i => i.SoNgayChuaVe)
            .ThenBy(static i => i.So)
            .Take(top)
            .ToList();

        _ = latestDate;
        return items;
    }

    public static List<(int So, int TanSuat)> TopFrequency(
        IReadOnlyList<LotteryRecord> orderedHistory,
        string gameKind,
        int lastDraws = 30,
        int top = 20)
    {
        var slice = orderedHistory.OrderBy(static r => r.NgayQuay).TakeLast(lastDraws);
        var freq = new Dictionary<int, int>();
        foreach (var draw in slice)
        {
            foreach (var lo in ExtractAllLoto(draw, gameKind))
            {
                freq[lo] = freq.GetValueOrDefault(lo) + 1;
            }
        }

        return freq
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(top)
            .Select(static kv => (kv.Key, kv.Value))
            .ToList();
    }

    private static int ParseLo(string twoDigits) =>
        int.TryParse(twoDigits, out var n) ? Math.Clamp(n, 0, 99) : 0;
}
