using Microsoft.ML.Data;

namespace AI_Model_BE.Models;

/// <summary>DTO trả về Angular — kỳ trước + dự đoán kỳ tiếp (ngày dd/MM/yyyy).</summary>
public sealed class PredictionOutputDto
{
    public string TenDai { get; set; } = string.Empty;

    public int KyQuayTruoc { get; set; }

    public string NgayQuayKyTruoc { get; set; } = string.Empty;

    public List<int> KetQuaKyTruoc { get; set; } = [];

    public int KyQuayDuDoan { get; set; }

    public string NgayQuayDuDoan { get; set; } = string.Empty;

    public List<int> CacSoDuDoan { get; set; } = [];

    public List<int> KetQuaGiaiDacBietTruoc { get; set; } = [];

    public List<int> TatCaLoVeKyTruoc { get; set; } = [];

    public List<int> DuDoanGiaiDacBiet { get; set; } = [];

    public string GiaiDacBietTruocDinhDang { get; set; } = string.Empty;

    public string GiaiDacBietDuDoanDinhDang { get; set; } = string.Empty;

    public int SoKyDaHoc { get; set; }

    public LottoLotoExtras? DuDoanLo { get; set; }

    public string? ThongBaoLoi { get; set; }

    public bool ThanhCong => string.IsNullOrEmpty(ThongBaoLoi);
}

/// <summary>Một kỳ quay — luôn gắn với một đài cụ thể (không trộn đài).</summary>
public sealed class LotteryRecord
{
    public string Dai { get; set; } = string.Empty;

    public DateTime NgayQuay { get; set; }

    /// <summary>Số thứ tự tăng dần theo thời gian của riêng đài đó.</summary>
    public int KyQuay { get; set; }

    public List<int> CacSoDaVe { get; set; } = [];

    /// <summary>Giải đặc biệt — 6 chữ số (MN/MT) hoặc 5 chữ số (MB), mỗi phần tử là 1 digit 0–9.</summary>
    public List<int> GiaiDacBietSo { get; set; } = [];

    /// <summary>Tất cả lô 00–99 xuất hiện trong kỳ (2 số cuối mỗi giải).</summary>
    public List<int> TatCaLoVe { get; set; } = [];
}

/// <summary>Kết quả dự đoán cho một đài — có kỳ trước để đối chiếu.</summary>
public sealed class DaiPredictionResult
{
    public string Dai { get; set; } = string.Empty;

    public int KyQuayTruoc { get; set; }

    public DateTime NgayQuayTruoc { get; set; }

    /// <summary>Kết quả kỳ gần nhất (hiển thị trước khi xem dự đoán).</summary>
    public List<int> KetQuaKyTruoc { get; set; } = [];

    public int KyQuayDuDoan { get; set; }

    public DateTime? NgayQuayDuKien { get; set; }

    /// <summary>Giá trị forecast thô từ ML.NET SSA (theo từng vị trí số).</summary>
    public List<float> CacSoDuDoanKyTiep { get; set; } = [];

    /// <summary>Số đã làm tròn/clamp để hiển thị.</summary>
    public List<int> CacSoDuDoanLamTron { get; set; } = [];

    public string KetQuaTruocDinhDang { get; set; } = string.Empty;

    public string KetQuaDuDoanDinhDang { get; set; } = string.Empty;

    public List<int> KetQuaGiaiDacBietTruoc { get; set; } = [];

    public List<int> TatCaLoVeKyTruoc { get; set; } = [];

    public List<int> DuDoanGiaiDacBiet { get; set; } = [];

    public string GiaiDacBietTruocDinhDang { get; set; } = string.Empty;

    public string GiaiDacBietDuDoanDinhDang { get; set; } = string.Empty;

    public int SoKyDaHoc { get; set; }

    public LottoLotoExtras? DuDoanLo { get; set; }

    public string? ThongBaoLoi { get; set; }

    public bool ThanhCong => string.IsNullOrEmpty(ThongBaoLoi);
}

/// <summary>Bạch thủ / song thủ / xiên — Miền Bắc đầy đủ; MN/MT chỉ bạch thủ.</summary>
public sealed class LottoLotoExtras
{
    public int? BachThuLo { get; set; }

    public List<int> SongThuLo { get; set; } = [];

    public List<int> Xien2 { get; set; } = [];

    public List<int> Xien3 { get; set; } = [];

    public List<int> Xien4 { get; set; } = [];

    public string BachThuLoDinhDang => BachThuLo.HasValue ? BachThuLo.Value.ToString("00") : string.Empty;

    public string SongThuLoDinhDang =>
        SongThuLo.Count > 0
            ? string.Join(" · ", SongThuLo.Select(static n => n.ToString("00")))
            : string.Empty;

    public string Xien2DinhDang => FormatXien(Xien2);

    public string Xien3DinhDang => FormatXien(Xien3);

    public string Xien4DinhDang => FormatXien(Xien4);

    private static string FormatXien(IReadOnlyList<int> nums) =>
        nums.Count > 0
            ? string.Join(" · ", nums.Select(static n => n.ToString("00")))
            : string.Empty;
}

public sealed class LoGanItem
{
    public int So { get; set; }

    public int SoNgayChuaVe { get; set; }

    public int GanMax { get; set; }

    public DateTime? NgayVeCuoi { get; set; }

    public string SoDinhDang => So.ToString("00");
}

public sealed class LottoHistoryResponse
{
    public string GameKind { get; set; } = string.Empty;

    public string? DaiCode { get; set; }

    public string DaiTen { get; set; } = string.Empty;

    public DateTime? TuNgay { get; set; }

    public DateTime? DenNgay { get; set; }

    public int TongKy { get; set; }

    public List<LotteryRecord> Records { get; set; } = [];
}

public sealed class LottoLoGanResponse
{
    public string GameKind { get; set; } = string.Empty;

    public string? DaiCode { get; set; }

    public string DaiTen { get; set; } = string.Empty;

    public int SoKyPhanTich { get; set; }

    public List<LoGanItem> Items { get; set; } = [];
}

public sealed class LottoSsaPoint
{
    public float GiaTri { get; set; }
}

public sealed class LottoSsaForecast
{
    [VectorType(1)]
    public float[] DuBao { get; set; } = [];
}

public record LottoLatestResponse(
    string GameKind,
    string? DaiCode,
    LotteryRecord? Latest);

public static class LotteryRecordNormalizer
{
    public static string NormalizeDaiKey(string dai) => dai.Trim().ToLowerInvariant();

    /// <summary>Gán KyQuay 1..n theo NgayQuay cho từng đài (nếu file nguồn thiếu).</summary>
    public static List<LotteryRecord> EnsureKyQuayPerDai(IEnumerable<LotteryRecord> records)
    {
        return records
            .GroupBy(r => NormalizeDaiKey(r.Dai), StringComparer.OrdinalIgnoreCase)
            .SelectMany(group =>
            {
                var ordered = group.OrderBy(r => r.NgayQuay).ThenBy(r => r.KyQuay).ToList();
                var ky = 1;
                foreach (var row in ordered)
                {
                    if (row.KyQuay <= 0)
                    {
                        row.KyQuay = ky;
                    }

                    ky = Math.Max(ky, row.KyQuay) + 1;
                }

                return ordered;
            })
            .ToList();
    }

    public static List<LotteryRecord> FilterByDai(IEnumerable<LotteryRecord> allData, string dai)
    {
        var key = NormalizeDaiKey(dai);
        // Sắp xếp tăng dần — phần tử cuối là kỳ gần nhất cho SSA
        return allData
            .Where(r => NormalizeDaiKey(r.Dai) == key)
            .OrderBy(r => r.NgayQuay)
            .ThenBy(r => r.KyQuay)
            .ToList();
    }

    public static int CountWithGiaiDacBiet(IEnumerable<LotteryRecord> records, int minDigits) =>
        records.Count(r => r.GiaiDacBietSo.Count >= minDigits);

    /// <summary>Gộp scrape + local — ưu tiên bản ghi có giải ĐB đủ chữ số.</summary>
    public static List<LotteryRecord> MergeDaiHistory(
        IEnumerable<LotteryRecord> scraped,
        IEnumerable<LotteryRecord> local)
    {
        return scraped
            .Concat(local)
            .GroupBy(r => r.NgayQuay.Date)
            .Select(MergeSameDayRecords)
            .OrderBy(r => r.NgayQuay)
            .ThenBy(r => r.KyQuay)
            .ToList();
    }

    private static LotteryRecord MergeSameDayRecords(IGrouping<DateTime, LotteryRecord> group)
    {
        var candidates = group.ToList();
        var best = candidates
            .OrderByDescending(r => r.GiaiDacBietSo.Count)
            .ThenByDescending(r => r.TatCaLoVe.Count)
            .ThenByDescending(r => r.CacSoDaVe.Count)
            .First();

        var withDb = candidates
            .Where(r => r.GiaiDacBietSo.Count >= 5)
            .OrderByDescending(r => r.GiaiDacBietSo.Count)
            .FirstOrDefault();

        if (withDb is null || best.GiaiDacBietSo.Count >= withDb.GiaiDacBietSo.Count)
        {
            return best;
        }

        best.GiaiDacBietSo = withDb.GiaiDacBietSo.ToList();
        if (best.TatCaLoVe.Count == 0 && withDb.TatCaLoVe.Count > 0)
        {
            best.TatCaLoVe = withDb.TatCaLoVe.ToList();
        }

        return best;
    }
}

public static class LottoDisplayFormatter
{
    public static string FormatGiaiDacBiet(IReadOnlyList<int> digits, string gameKind)
    {
        if (digits.Count == 0)
        {
            return "—";
        }

        return string.Concat(digits.Select(static d => d.ToString()));
    }

    /// <summary>Hiển thị lô về — các số 00–99, sắp xếp tăng dần, mỗi số 2 chữ số.</summary>
    public static string FormatLoVe(IReadOnlyList<int> tatCaLoVe)
    {
        if (tatCaLoVe.Count == 0)
        {
            return "—";
        }

        return string.Join(" · ", tatCaLoVe.Distinct().OrderBy(static n => n).Select(static n => n.ToString("00")));
    }

    public static int CountLoVe(IReadOnlyList<int> tatCaLoVe) =>
        tatCaLoVe.Count;

    public static string Format(LottoGameProfile profile, IReadOnlyList<int> numbers, int? power = null) =>
        profile.ParseMode switch
        {
            LottoParseMode.SpecialPrizeFiveDigits => string.Concat(numbers.Select(static n => n.ToString())),
            LottoParseMode.SouthernSpecialPrizeSixDigits => string.Concat(numbers.Select(static n => n.ToString())),
            LottoParseMode.Vietlott645 => string.Join(" - ", numbers.OrderBy(static n => n).Select(static n => n.ToString("00"))),
            LottoParseMode.Vietlott655 when power.HasValue =>
                string.Join(" - ", numbers.Take(6).OrderBy(static n => n).Select(static n => n.ToString("00")))
                + $" | Power: {power.Value:00}",
            LottoParseMode.Vietlott655 => string.Join(" - ", numbers.Select(static n => n.ToString("00"))),
            _ => string.Join(" · ", numbers.Select(static n => n.ToString("00")))
        };
}
