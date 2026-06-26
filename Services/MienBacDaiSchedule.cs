using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Đài phát hành XSMB theo ngày — minhngoc.net.vn.</summary>
internal static class MienBacDaiSchedule
{
    public static IReadOnlyList<LottoDaiInfo> GetAllDais() =>
        LottoGameCatalog.GetDaiList(LottoGameKinds.XsMienBac);

    public static LottoDaiInfo? ResolveDai(string? daiCode) =>
        LottoGameCatalog.ResolveDai(LottoGameKinds.XsMienBac, daiCode);

    public static string ResolveDaiCodeForDate(DateTime date) =>
        LottoGameCatalog.ResolveMienBacDaiCodeForDate(date);

    public static LottoDaiInfo ResolveDaiForDate(DateTime date)
    {
        var code = ResolveDaiCodeForDate(date);
        return ResolveDai(code) ?? GetAllDais()[0];
    }

    public static bool IsDaiDrawDay(DateTime date, string daiCode) =>
        ResolveDaiCodeForDate(date).Equals(daiCode, StringComparison.OrdinalIgnoreCase);

    /// <summary>Các ngày quay gần nhất của riêng một đài XSMB (lùi từ start).</summary>
    public static IEnumerable<DateTime> EnumerateDrawDatesForDai(string daiCode, DateTime start, int maxCount)
    {
        var cursor = start.Date;
        var yielded = 0;
        while (yielded < maxCount && cursor.Year >= 2005)
        {
            if (IsDaiDrawDay(cursor, daiCode))
            {
                yield return cursor;
                yielded++;
            }

            cursor = cursor.AddDays(-1);
        }
    }
}
