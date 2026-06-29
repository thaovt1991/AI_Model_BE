using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Lịch quay chính thức theo minhngoc.net.vn — tính ngày kỳ tiếp theo.</summary>
internal static class LottoDrawSchedule
{
    private static readonly Dictionary<string, DayOfWeek[]> DrawWeekdays =
        new(StringComparer.OrdinalIgnoreCase)
        {
            // Miền Nam
            ["hcm"] = [DayOfWeek.Monday, DayOfWeek.Saturday],
            ["dong-thap"] = [DayOfWeek.Monday],
            ["ca-mau"] = [DayOfWeek.Monday],
            ["ben-tre"] = [DayOfWeek.Tuesday],
            ["vung-tau"] = [DayOfWeek.Tuesday],
            ["bac-lieu"] = [DayOfWeek.Tuesday],
            ["dong-nai"] = [DayOfWeek.Wednesday],
            ["can-tho"] = [DayOfWeek.Wednesday],
            ["soc-trang"] = [DayOfWeek.Wednesday],
            ["tay-ninh"] = [DayOfWeek.Thursday],
            ["an-giang"] = [DayOfWeek.Thursday],
            ["binh-thuan"] = [DayOfWeek.Thursday],
            ["vinh-long"] = [DayOfWeek.Friday],
            ["binh-duong"] = [DayOfWeek.Friday],
            ["tra-vinh"] = [DayOfWeek.Friday],
            ["long-an"] = [DayOfWeek.Saturday],
            ["binh-phuoc"] = [DayOfWeek.Saturday],
            ["hau-giang"] = [DayOfWeek.Saturday],
            ["tien-giang"] = [DayOfWeek.Sunday],
            ["kien-giang"] = [DayOfWeek.Sunday],
            ["da-lat"] = [DayOfWeek.Sunday],
            // Miền Trung
            ["phu-yen"] = [DayOfWeek.Monday],
            ["hue"] = [DayOfWeek.Monday, DayOfWeek.Sunday],
            ["dak-lak"] = [DayOfWeek.Tuesday],
            ["quang-nam"] = [DayOfWeek.Tuesday],
            ["da-nang"] = [DayOfWeek.Wednesday, DayOfWeek.Saturday],
            ["khanh-hoa"] = [DayOfWeek.Wednesday, DayOfWeek.Sunday],
            ["binh-dinh"] = [DayOfWeek.Thursday],
            ["quang-tri"] = [DayOfWeek.Thursday],
            ["quang-binh"] = [DayOfWeek.Thursday],
            ["gia-lai"] = [DayOfWeek.Friday],
            ["ninh-thuan"] = [DayOfWeek.Friday],
            ["dak-nong"] = [DayOfWeek.Saturday],
            ["quang-ngai"] = [DayOfWeek.Saturday],
            ["kon-tum"] = [DayOfWeek.Sunday],
        };

    public static bool IsDrawDay(DateTime date, string? daiCode, string gameKind)
    {
        if (gameKind == LottoGameKinds.XsMienBac)
        {
            return true;
        }

        if (string.IsNullOrWhiteSpace(daiCode))
        {
            return false;
        }

        if (!DrawWeekdays.TryGetValue(daiCode, out var days))
        {
            return date.DayOfWeek is not DayOfWeek.Sunday;
        }

        return days.Contains(date.DayOfWeek);
    }

    /// <summary>
    /// Ngày quay kế tiếp sau kỳ gần nhất.
    /// Duyệt từng ngày sau lastDrawDate; ưu tiên thứ trong lịch đài và khớp lịch sử thực tế (nếu có).
    /// </summary>
    public static DateTime ComputeNextDrawDate(
        DateTime lastDrawDate,
        string? daiCode,
        string gameKind,
        IReadOnlyList<DateTime>? historicalDates = null)
    {
        if (gameKind == LottoGameKinds.XsMienBac)
        {
            if (!string.IsNullOrWhiteSpace(daiCode))
            {
                var mbNext = lastDrawDate.Date.AddDays(1);
                for (var i = 0; i < 14; i++)
                {
                    if (MienBacDaiSchedule.IsDaiDrawDay(mbNext, daiCode))
                    {
                        return mbNext;
                    }

                    mbNext = mbNext.AddDays(1);
                }
            }

            return lastDrawDate.Date.AddDays(1);
        }

        if (gameKind is LottoGameKinds.Vietlott645 or LottoGameKinds.Vietlott655)
        {
            var vltNext = lastDrawDate.Date.AddDays(1);
            for (var i = 0; i < 14; i++)
            {
                if (IsVietlottDrawDay(vltNext, gameKind))
                {
                    return vltNext;
                }

                vltNext = vltNext.AddDays(1);
            }

            return lastDrawDate.Date.AddDays(2);
        }

        var histWeekdays = historicalDates?
            .Select(d => d.DayOfWeek)
            .Distinct()
            .ToHashSet() ?? [];

        var candidate = lastDrawDate.Date.AddDays(1);
        for (var i = 0; i < 14; i++)
        {
            if (!IsDrawDay(candidate, daiCode, gameKind))
            {
                candidate = candidate.AddDays(1);
                continue;
            }

            if (histWeekdays.Count >= 2 && !histWeekdays.Contains(candidate.DayOfWeek))
            {
                candidate = candidate.AddDays(1);
                continue;
            }

            return candidate;
        }

        return FindNextFromHistory(lastDrawDate, historicalDates)
            ?? lastDrawDate.Date.AddDays(7);
    }

    private static bool IsVietlottDrawDay(DateTime date, string gameKind) =>
        gameKind switch
        {
            LottoGameKinds.Vietlott645 => date.DayOfWeek is DayOfWeek.Wednesday or DayOfWeek.Friday or DayOfWeek.Sunday,
            LottoGameKinds.Vietlott655 => date.DayOfWeek is DayOfWeek.Tuesday or DayOfWeek.Thursday or DayOfWeek.Saturday,
            _ => false,
        };

    private static DateTime? FindNextFromHistory(
        DateTime lastDrawDate,
        IReadOnlyList<DateTime>? historicalDates)
    {
        if (historicalDates is null || historicalDates.Count < 2)
        {
            return null;
        }

        var ordered = historicalDates.OrderBy(d => d).ToList();
        var gaps = new List<int>();
        for (var i = 1; i < ordered.Count; i++)
        {
            gaps.Add((ordered[i] - ordered[i - 1]).Days);
        }

        var typicalGap = gaps.OrderBy(g => g).ElementAt(gaps.Count / 2);
        if (typicalGap <= 0)
        {
            typicalGap = 7;
        }

        return lastDrawDate.Date.AddDays(typicalGap);
    }

    public static IEnumerable<DateTime> EnumeratePreviousDrawDates(
        DateTime startFrom,
        string? daiCode,
        string gameKind,
        int maxCount)
    {
        var cursor = startFrom.Date;
        var yielded = 0;

        while (yielded < maxCount && cursor.Year >= 2005)
        {
            if (IsDrawDay(cursor, daiCode, gameKind))
            {
                yield return cursor;
                yielded++;
            }

            cursor = cursor.AddDays(-1);
        }
    }
}
