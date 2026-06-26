using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Nhận diện câu hỏi xổ số trong chat và trả kết quả ML.NET / lịch sử / lô gan.</summary>
public sealed class LottoChatIntentService
{
    private static readonly CultureInfo ViCulture = CultureInfo.GetCultureInfo("vi-VN");

    private readonly LottoForecastService _forecast;
    private readonly ILogger<LottoChatIntentService> _logger;

    public LottoChatIntentService(LottoForecastService forecast, ILogger<LottoChatIntentService> logger)
    {
        _forecast = forecast;
        _logger = logger;
    }

    public bool IsLottoRelated(string message) =>
        ResolveIntent(message) is not LottoChatIntent.None;

    public async Task<string?> TryBuildReplyAsync(string message, CancellationToken cancellationToken = default)
    {
        var intent = ResolveIntent(message);
        if (intent == LottoChatIntent.None)
        {
            return null;
        }

        var (gameKind, daiCode) = ResolveGameAndDai(message);
        if (gameKind is null)
        {
            return "Bạn muốn hỏi xổ số miền nào? Ví dụ:\n" +
                   "• \"Dự đoán XSMB Hà Nội\"\n" +
                   "• \"Dự đoán giải đặc biệt TP HCM\"\n" +
                   "• \"Lô gan đài Đà Nẵng\"\n" +
                   "• \"Lịch sử xổ số miền Nam 7 ngày\"";
        }

        try
        {
            return intent switch
            {
                LottoChatIntent.Predict => await BuildPredictReplyAsync(gameKind, daiCode, cancellationToken),
                LottoChatIntent.LoGan => await BuildLoGanReplyAsync(gameKind, daiCode, cancellationToken),
                LottoChatIntent.History => await BuildHistoryReplyAsync(gameKind, daiCode, message, cancellationToken),
                LottoChatIntent.Latest => await BuildLatestReplyAsync(gameKind, daiCode, cancellationToken),
                _ => null
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Lotto chat intent failed");
            return $"Không lấy được dữ liệu xổ số: {ex.Message}";
        }
    }

    private async Task<string> BuildPredictReplyAsync(
        string gameKind,
        string? daiCode,
        CancellationToken cancellationToken)
    {
        var response = await _forecast.RunForecastAsync(
            new LottoRunRequest(gameKind, daiCode, PredictAllDais: false),
            cancellationToken);

        if (response.Single is null)
        {
            return "Không có kết quả dự đoán.";
        }

        return FormatPrediction(response.Single, gameKind);
    }

    private async Task<string> BuildLoGanReplyAsync(
        string gameKind,
        string? daiCode,
        CancellationToken cancellationToken)
    {
        var gan = await _forecast.GetLoGanAsync(gameKind, daiCode, 15, cancellationToken);
        var sb = new StringBuilder();
        sb.AppendLine($"Lô gan — {gan.DaiTen} ({gan.SoKyPhanTich} kỳ phân tích)");
        sb.AppendLine();
        foreach (var item in gan.Items.Take(10))
        {
            var last = item.NgayVeCuoi?.ToString("dd/MM/yyyy", ViCulture) ?? "chưa từng";
            sb.AppendLine($"• {item.SoDinhDang}: {item.SoNgayChuaVe} ngày chưa về (gan max {item.GanMax}, về gần nhất {last})");
        }

        sb.AppendLine();
        sb.AppendLine("(Chỉ tham khảo — tính từ lịch sử Minh Ngọc.)");
        return sb.ToString().Trim();
    }

    private async Task<string> BuildHistoryReplyAsync(
        string gameKind,
        string? daiCode,
        string message,
        CancellationToken cancellationToken)
    {
        var days = ParseHistoryDays(message) ?? 7;
        var to = DateTime.Today;
        var from = to.AddDays(-days);
        var hist = await _forecast.GetHistoryAsync(gameKind, daiCode, from, to, null, cancellationToken);

        if (hist.Records.Count == 0)
        {
            return $"Không có lịch sử {hist.DaiTen} trong {days} ngày gần đây.";
        }

        var sb = new StringBuilder();
        sb.AppendLine($"Lịch sử {hist.DaiTen} — {hist.TongKy} kỳ ({from:dd/MM/yyyy} → {to:dd/MM/yyyy})");
        sb.AppendLine();
        foreach (var row in hist.Records.Take(10))
        {
            var db = LottoDisplayFormatter.FormatGiaiDacBiet(row.GiaiDacBietSo, gameKind);
            var lo = LottoDisplayFormatter.FormatLoVe(row.TatCaLoVe);
            var count = LottoDisplayFormatter.CountLoVe(row.TatCaLoVe);
            sb.AppendLine($"• {row.NgayQuay:dd/MM/yyyy} — ĐB: {db} | Lô về ({count}): {lo}");
        }

        if (hist.TongKy > 10)
        {
            sb.AppendLine($"... và {hist.TongKy - 10} kỳ khác (xem tab Lịch sử trên UI).");
        }

        return sb.ToString().Trim();
    }

    private async Task<string> BuildLatestReplyAsync(
        string gameKind,
        string? daiCode,
        CancellationToken cancellationToken)
    {
        var latest = await _forecast.GetLatestForApiAsync(gameKind, daiCode, cancellationToken);
        if (latest is null)
        {
            return "Chưa có kết quả kỳ gần nhất.";
        }

        var db = LottoDisplayFormatter.FormatGiaiDacBiet(latest.GiaiDacBietSo, gameKind);
        var lo = LottoDisplayFormatter.FormatLoVe(latest.TatCaLoVe);
        return $"Kết quả mới nhất — {latest.Dai} ({latest.NgayQuay:dd/MM/yyyy})\n" +
               $"Giải ĐB: {db}\n" +
               $"Lô về ({LottoDisplayFormatter.CountLoVe(latest.TatCaLoVe)} số): {lo}";
    }

    public static string FormatPrediction(DaiPredictionResult r, string gameKind)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Dự đoán — {r.Dai} (kỳ #{r.KyQuayDuDoan})");
        if (r.NgayQuayDuKien.HasValue)
        {
            sb.AppendLine($"Ngày dự kiến: {r.NgayQuayDuKien:dd/MM/yyyy}");
        }

        if (!string.IsNullOrWhiteSpace(r.GiaiDacBietDuDoanDinhDang))
        {
            sb.AppendLine($"Giải ĐB: {r.GiaiDacBietDuDoanDinhDang}");
        }

        if (r.DuDoanLo?.BachThuLo is not null)
        {
            sb.AppendLine($"Bạch thủ lô: {r.DuDoanLo.BachThuLoDinhDang}");
        }

        if (gameKind == LottoGameKinds.XsMienBac && r.DuDoanLo is not null)
        {
            if (!string.IsNullOrWhiteSpace(r.DuDoanLo.SongThuLoDinhDang))
            {
                sb.AppendLine($"Song thủ lô: {r.DuDoanLo.SongThuLoDinhDang}");
            }

            if (!string.IsNullOrWhiteSpace(r.DuDoanLo.Xien2DinhDang))
            {
                sb.AppendLine($"Xiên 2: {r.DuDoanLo.Xien2DinhDang}");
            }

            if (!string.IsNullOrWhiteSpace(r.DuDoanLo.Xien3DinhDang))
            {
                sb.AppendLine($"Xiên 3: {r.DuDoanLo.Xien3DinhDang}");
            }

            if (!string.IsNullOrWhiteSpace(r.DuDoanLo.Xien4DinhDang))
            {
                sb.AppendLine($"Xiên 4: {r.DuDoanLo.Xien4DinhDang}");
            }
        }

        if (!string.IsNullOrWhiteSpace(r.ThongBaoLoi))
        {
            sb.AppendLine($"⚠ {r.ThongBaoLoi}");
        }
        else
        {
            sb.AppendLine($"(SSA — học {r.SoKyDaHoc} kỳ. Chỉ tham khảo.)");
        }

        return sb.ToString().Trim();
    }

    private static LottoChatIntent ResolveIntent(string message)
    {
        var m = message.ToLowerInvariant();
        if (!ContainsAny(m, "xổ số", "xo so", "xsmb", "xsmn", "xsmt", "vietlott", "lô", "lo ", "đài", "dai "))
        {
            if (!ContainsAny(m, "dự đoán", "du doan", "dự báo"))
            {
                return LottoChatIntent.None;
            }
        }

        if (ContainsAny(m, "lô gan", "lo gan", "gan cực", "gan cuc", "số gan", "so gan"))
        {
            return LottoChatIntent.LoGan;
        }

        if (ContainsAny(m, "lịch sử", "lich su", "kết quả các ngày", "ket qua cac ngay", "xem ngày", "tra cứu"))
        {
            return LottoChatIntent.History;
        }

        if (ContainsAny(m, "mới nhất", "moi nhat", "kỳ trước", "ky truoc", "hôm qua", "hom qua", "vừa quay"))
        {
            return LottoChatIntent.Latest;
        }

        if (ContainsAny(m, "dự đoán", "du doan", "dự báo", "du bao", "soi cầu", "gợi ý", "goi y"))
        {
            return LottoChatIntent.Predict;
        }

        if (ContainsAny(m, "xổ số", "xo so", "xsmb", "xsmn", "xsmt"))
        {
            return LottoChatIntent.Predict;
        }

        return LottoChatIntent.None;
    }

    private static (string? GameKind, string? DaiCode) ResolveGameAndDai(string message)
    {
        var normalized = Normalize(message);

        string? gameKind = null;
        if (ContainsAny(normalized, "miền bắc", "mien bac", "xsmb", "hà nội", "ha noi"))
        {
            gameKind = LottoGameKinds.XsMienBac;
        }
        else if (ContainsAny(normalized, "miền nam", "mien nam", "xsmn"))
        {
            gameKind = LottoGameKinds.XsMienNam;
        }
        else if (ContainsAny(normalized, "miền trung", "mien trung", "xsmt"))
        {
            gameKind = LottoGameKinds.XsMienTrung;
        }

        string? daiCode = null;
        foreach (var profile in LottoGameCatalog.All.Where(p => p.RequiresDai))
        {
            foreach (var dai in LottoGameCatalog.GetDaiList(profile.Kind))
            {
                var nameKey = Normalize(dai.Name);
                var codeKey = Normalize(dai.Code.Replace('-', ' '));
                if (normalized.Contains(nameKey, StringComparison.Ordinal) ||
                    normalized.Contains(codeKey, StringComparison.Ordinal))
                {
                    gameKind ??= profile.Kind;
                    daiCode = dai.Code;
                    break;
                }
            }
        }

        if (gameKind is null)
        {
            gameKind = LottoGameKinds.XsMienBac;
        }

        if (daiCode is null && gameKind == LottoGameKinds.XsMienBac)
        {
            daiCode = LottoGameCatalog.ResolveMienBacDaiCodeForDate(DateTime.Today);
        }

        return (gameKind, daiCode);
    }

    private static int? ParseHistoryDays(string message)
    {
        var m = Regex.Match(message, @"(\d+)\s*(ngày|ngay|ky|kỳ)");
        if (m.Success && int.TryParse(m.Groups[1].Value, out var days))
        {
            return Math.Clamp(days, 1, 60);
        }

        return null;
    }

    private static bool ContainsAny(string haystack, params string[] needles)
    {
        foreach (var n in needles)
        {
            if (haystack.Contains(n, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static string Normalize(string text) =>
        Regex.Replace(text.ToLowerInvariant(), @"\s+", " ").Trim();

    private enum LottoChatIntent
    {
        None,
        Predict,
        LoGan,
        History,
        Latest
    }
}
