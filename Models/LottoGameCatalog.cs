namespace AI_Model_BE.Models;

/// <summary>Loại hình xổ số hỗ trợ dự đoán SSA.</summary>
public static class LottoGameKinds
{
    public const string XsMienBac = "xs-mien-bac";
    public const string XsMienNam = "xs-mien-nam";
    public const string XsMienTrung = "xs-mien-trung";
    public const string Vietlott645 = "vietlott-645";
    public const string Vietlott655 = "vietlott-655";
}

public enum LottoParseMode
{
    TwoDigitPool,
    SpecialPrizeFiveDigits,
    SouthernSpecialPrizeSixDigits,
    Vietlott645,
    Vietlott655
}

public sealed record LottoDaiInfo(
    string Code,
    string Name,
    string RegionLabel,
    string MinhNgocSlug);

public sealed record LottoGameProfile(
    string Kind,
    string Title,
    string Subtitle,
    string Description,
    LottoParseMode ParseMode,
    int MinNumber,
    int MaxNumber,
    int PredictCount,
    bool RequiresDai,
    string SampleFileName,
    string ResultLabel);

public sealed record LottoGameInfoDto(
    string Kind,
    string Title,
    string Subtitle,
    string Description,
    bool RequiresDai,
    IReadOnlyList<LottoDaiInfo> DaiList);

public record LottoRunRequest(
    string GameKind,
    string? DaiCode = null,
    IReadOnlyList<string>? DaiCodes = null,
    bool UseSampleIfEmpty = true,
    bool PredictAllDais = false);

public static class LottoGameCatalog
{
    public static IReadOnlyList<LottoGameProfile> All { get; } =
    [
        new(
            LottoGameKinds.XsMienBac,
            "XS Kiến thiết — Miền Bắc",
            "Giải ĐB + lô",
            "Dự báo giải ĐB, bạch thủ, song thủ, xiên 2/3/4 theo đài phát hành XSMB.",
            LottoParseMode.SpecialPrizeFiveDigits,
            0, 9, 5, true,
            "xs-mien-bac.json",
            "Giải ĐB dự kiến"),
        new(
            LottoGameKinds.XsMienNam,
            "XS Kiến thiết — Miền Nam",
            "Giải ĐB + lô",
            "Dự báo giải ĐB và bạch thủ lô theo từng đài phát hành.",
            LottoParseMode.TwoDigitPool,
            0, 99, 1, true,
            "xs-mien-nam-{dai}.json",
            "Giải ĐB dự kiến"),
        new(
            LottoGameKinds.XsMienTrung,
            "XS Kiến thiết — Miền Trung",
            "Giải ĐB + lô",
            "Dự báo giải ĐB và bạch thủ lô theo từng đài phát hành.",
            LottoParseMode.TwoDigitPool,
            0, 99, 1, true,
            "xs-mien-trung-{dai}.json",
            "Giải ĐB dự kiến"),
        new(
            LottoGameKinds.Vietlott645,
            "Vietlott Mega 6/45",
            "6 số từ 1 đến 45",
            "Dự báo 6 số Vietlott Mega 6/45.",
            LottoParseMode.Vietlott645,
            1, 45, 6, false,
            "vietlott-645.json",
            "Bộ số 6/45"),
        new(
            LottoGameKinds.Vietlott655,
            "Vietlott Power 6/55",
            "6 số chính + 1 số phụ",
            "Dự báo 6 số chính và 1 số Power.",
            LottoParseMode.Vietlott655,
            1, 55, 7, false,
            "vietlott-655.json",
            "Bộ số 6/55 + Power")
    ];

    /// <summary>6 đài phát hành XSMB — slug minhngoc.</summary>
    private static readonly IReadOnlyList<LottoDaiInfo> MienBacDai =
    [
        new("ha-noi", "Hà Nội", "Miền Bắc", "ha-noi"),
        new("quang-ninh", "Quảng Ninh", "Miền Bắc", "quang-ninh"),
        new("bac-ninh", "Bắc Ninh", "Miền Bắc", "bac-ninh"),
        new("hai-phong", "Hải Phòng", "Miền Bắc", "hai-phong"),
        new("nam-dinh", "Nam Định", "Miền Bắc", "nam-dinh"),
        new("thai-binh", "Thái Bình", "Miền Bắc", "thai-binh"),
    ];

    private static readonly Dictionary<DayOfWeek, string> MienBacWeekdayDai = new()
    {
        [DayOfWeek.Monday] = "ha-noi",
        [DayOfWeek.Tuesday] = "quang-ninh",
        [DayOfWeek.Wednesday] = "bac-ninh",
        [DayOfWeek.Thursday] = "ha-noi",
        [DayOfWeek.Friday] = "hai-phong",
        [DayOfWeek.Saturday] = "nam-dinh",
        [DayOfWeek.Sunday] = "thai-binh",
    };

    public static string ResolveMienBacDaiCodeForDate(DateTime date) =>
        MienBacWeekdayDai.TryGetValue(date.DayOfWeek, out var code) ? code : "ha-noi";

    /// <summary>21 đài Miền Nam — slug khớp minhngoc.net.vn.</summary>
    private static readonly IReadOnlyList<LottoDaiInfo> MienNamDai =
    [
        new("hcm", "TP. Hồ Chí Minh", "Miền Nam", "tp-hcm"),
        new("dong-thap", "Đồng Tháp", "Miền Nam", "dong-thap"),
        new("ca-mau", "Cà Mau", "Miền Nam", "ca-mau"),
        new("ben-tre", "Bến Tre", "Miền Nam", "ben-tre"),
        new("vung-tau", "Vũng Tàu", "Miền Nam", "vung-tau"),
        new("bac-lieu", "Bạc Liêu", "Miền Nam", "bac-lieu"),
        new("dong-nai", "Đồng Nai", "Miền Nam", "dong-nai"),
        new("can-tho", "Cần Thơ", "Miền Nam", "can-tho"),
        new("soc-trang", "Sóc Trăng", "Miền Nam", "soc-trang"),
        new("tay-ninh", "Tây Ninh", "Miền Nam", "tay-ninh"),
        new("an-giang", "An Giang", "Miền Nam", "an-giang"),
        new("binh-thuan", "Bình Thuận", "Miền Nam", "binh-thuan"),
        new("vinh-long", "Vĩnh Long", "Miền Nam", "vinh-long"),
        new("binh-duong", "Bình Dương", "Miền Nam", "binh-duong"),
        new("tra-vinh", "Trà Vinh", "Miền Nam", "tra-vinh"),
        new("long-an", "Long An", "Miền Nam", "long-an"),
        new("binh-phuoc", "Bình Phước", "Miền Nam", "binh-phuoc"),
        new("hau-giang", "Hậu Giang", "Miền Nam", "hau-giang"),
        new("tien-giang", "Tiền Giang", "Miền Nam", "tien-giang"),
        new("kien-giang", "Kiên Giang", "Miền Nam", "kien-giang"),
        new("da-lat", "Đà Lạt", "Miền Nam", "da-lat"),
    ];

    /// <summary>14 đài Miền Trung — slug khớp minhngoc.net.vn.</summary>
    private static readonly IReadOnlyList<LottoDaiInfo> MienTrungDai =
    [
        new("phu-yen", "Phú Yên", "Miền Trung", "phu-yen"),
        new("hue", "Thừa Thiên Huế", "Miền Trung", "thua-thien-hue"),
        new("dak-lak", "Đắk Lắk", "Miền Trung", "dak-lak"),
        new("quang-nam", "Quảng Nam", "Miền Trung", "quang-nam"),
        new("da-nang", "Đà Nẵng", "Miền Trung", "da-nang"),
        new("khanh-hoa", "Khánh Hòa", "Miền Trung", "khanh-hoa"),
        new("binh-dinh", "Bình Định", "Miền Trung", "binh-dinh"),
        new("quang-tri", "Quảng Trị", "Miền Trung", "quang-tri"),
        new("quang-binh", "Quảng Bình", "Miền Trung", "quang-binh"),
        new("gia-lai", "Gia Lai", "Miền Trung", "gia-lai"),
        new("ninh-thuan", "Ninh Thuận", "Miền Trung", "ninh-thuan"),
        new("dak-nong", "Đắk Nông", "Miền Trung", "dak-nong"),
        new("quang-ngai", "Quảng Ngãi", "Miền Trung", "quang-ngai"),
        new("kon-tum", "Kon Tum", "Miền Trung", "kon-tum"),
    ];

    public static LottoGameProfile GetProfileByKind(string gameKind) =>
        All.FirstOrDefault(p => p.Kind == gameKind)
        ?? throw new ArgumentException($"Loại xổ số không hỗ trợ: {gameKind}");

    public static LottoGameProfile GetProfile(string gameKind, string? daiCode = null)
    {
        var profile = GetProfileByKind(gameKind);

        if (profile.RequiresDai && string.IsNullOrWhiteSpace(daiCode))
        {
            if (gameKind == LottoGameKinds.XsMienBac)
            {
                daiCode = ResolveMienBacDaiCodeForDate(DateTime.Today);
            }
            else
            {
                throw new ArgumentException("XS Miền Nam/Trung cần chọn mã đài (daiCode).");
            }
        }

        if (profile.RequiresDai && ResolveDai(gameKind, daiCode) is null)
        {
            throw new ArgumentException($"Đài không hợp lệ: {daiCode}");
        }

        return profile;
    }

    public static IReadOnlyList<LottoDaiInfo> GetDaiList(string gameKind) => gameKind switch
    {
        LottoGameKinds.XsMienNam => MienNamDai,
        LottoGameKinds.XsMienTrung => MienTrungDai,
        LottoGameKinds.XsMienBac => MienBacDai,
        _ => []
    };

    public static LottoDaiInfo? ResolveDai(string gameKind, string? daiCode)
    {
        if (string.IsNullOrWhiteSpace(daiCode))
        {
            return gameKind == LottoGameKinds.XsMienBac
                ? MienBacDai.FirstOrDefault(d =>
                    d.Code.Equals(ResolveMienBacDaiCodeForDate(DateTime.Today),
                        StringComparison.OrdinalIgnoreCase))
                : null;
        }

        if (gameKind == LottoGameKinds.XsMienBac)
        {
            return MienBacDai.FirstOrDefault(d =>
                d.Code.Equals(daiCode, StringComparison.OrdinalIgnoreCase));
        }

        return GetDaiList(gameKind).FirstOrDefault(d =>
            d.Code.Equals(daiCode, StringComparison.OrdinalIgnoreCase));
    }

    public static string ResolveMinhNgocSlug(string gameKind, string? daiCode)
    {
        var dai = ResolveDai(gameKind, daiCode);
        return dai?.MinhNgocSlug ?? daiCode ?? string.Empty;
    }

    public static string BuildSampleFileName(LottoGameProfile profile, string? daiCode) =>
        profile.SampleFileName.Replace("{dai}", daiCode ?? string.Empty, StringComparison.Ordinal);

    public static string BuildModelKey(string gameKind, string? daiCode) =>
        string.IsNullOrWhiteSpace(daiCode) ? gameKind : $"{gameKind}:{daiCode}";

    public static IReadOnlyList<LottoGameInfoDto> GetAllForApi() =>
        All.Select(p => new LottoGameInfoDto(
            p.Kind,
            p.Title,
            p.Subtitle,
            p.Description,
            p.RequiresDai,
            p.RequiresDai ? GetDaiList(p.Kind) : [])).ToList();
}
