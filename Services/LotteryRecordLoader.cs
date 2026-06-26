using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Nạp lịch sử kỳ quay — mỗi bản ghi gắn đúng một Dai.</summary>
public sealed class LotteryRecordLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private readonly string _lottoDataDir;
    private readonly ILogger<LotteryRecordLoader> _logger;

    public LotteryRecordLoader(IWebHostEnvironment env, ILogger<LotteryRecordLoader> logger)
    {
        _logger = logger;
        _lottoDataDir = Path.Combine(env.ContentRootPath, "Data", "Lotto");
        Directory.CreateDirectory(_lottoDataDir);
    }

    public async Task<IReadOnlyList<LotteryRecord>> LoadDatasetAsync(
        string gameKind,
        string? daiCode = null,
        CancellationToken cancellationToken = default)
    {
        if (!string.IsNullOrWhiteSpace(daiCode))
        {
            return await LoadSingleDaiAsync(gameKind, daiCode, cancellationToken);
        }

        var profile = LottoGameCatalog.GetProfileByKind(gameKind);
        var merged = new List<LotteryRecord>();

        foreach (var path in EnumerateRecordFiles(profile, gameKind))
        {
            merged.AddRange(await ReadRecordsFileAsync(path, cancellationToken));
        }

        if (merged.Count == 0 && profile.RequiresDai)
        {
            foreach (var dai in LottoGameCatalog.GetDaiList(gameKind))
            {
                var perDai = await LoadSingleDaiAsync(gameKind, dai.Code, cancellationToken);
                merged.AddRange(perDai);
            }
        }

        return LotteryRecordNormalizer.EnsureKyQuayPerDai(merged);
    }

    private async Task<IReadOnlyList<LotteryRecord>> LoadSingleDaiAsync(
        string gameKind,
        string daiCode,
        CancellationToken cancellationToken)
    {
        var profile = LottoGameCatalog.GetProfile(gameKind, daiCode);
        var daiInfo = LottoGameCatalog.ResolveDai(gameKind, daiCode)!;
        var path = Path.Combine(_lottoDataDir, $"records-{gameKind}-{daiCode}.json");

        if (File.Exists(path))
        {
            var records = await ReadRecordsFileAsync(path, cancellationToken);
            if (records.Count > 0)
            {
                return LotteryRecordNormalizer.EnsureKyQuayPerDai(records);
            }
        }

        var legacyPath = Path.Combine(_lottoDataDir, LottoGameCatalog.BuildSampleFileName(profile, daiCode));
        if (!File.Exists(legacyPath))
        {
            _logger.LogWarning("Không có dữ liệu cho đài {Dai} ({Path})", daiInfo.Name, path);
            return [];
        }

        await using var stream = File.OpenRead(legacyPath);
        var legacy = await JsonSerializer.DeserializeAsync<List<LegacyLottoRow>>(stream, JsonOptions, cancellationToken);
        var converted = (legacy ?? [])
            .Select(row => new LotteryRecord
            {
                Dai = daiInfo.Name,
                NgayQuay = row.NgayQuay,
                KyQuay = row.Id,
                CacSoDaVe = ParseLegacyNumbers(row.KetQua, profile)
            })
            .ToList();

        return LotteryRecordNormalizer.EnsureKyQuayPerDai(converted);
    }

    private IEnumerable<string> EnumerateRecordFiles(LottoGameProfile profile, string gameKind)
    {
        if (!profile.RequiresDai)
        {
            yield return Path.Combine(_lottoDataDir, $"records-{gameKind}.json");
            yield break;
        }

        foreach (var dai in LottoGameCatalog.GetDaiList(gameKind))
        {
            yield return Path.Combine(_lottoDataDir, $"records-{gameKind}-{dai.Code}.json");
        }
    }

    private async Task<List<LotteryRecord>> ReadRecordsFileAsync(string path, CancellationToken cancellationToken)
    {
        if (!File.Exists(path))
        {
            return [];
        }

        await using var stream = File.OpenRead(path);
        var batch = await JsonSerializer.DeserializeAsync<List<LotteryRecord>>(stream, JsonOptions, cancellationToken);
        return batch ?? [];
    }

    private static List<int> ParseLegacyNumbers(string ketQua, LottoGameProfile profile)
    {
        if (profile.ParseMode == LottoParseMode.SpecialPrizeFiveDigits)
        {
            return ketQua.Trim().Where(char.IsDigit).Select(c => c - '0').Take(5).ToList();
        }

        return ketQua
            .Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => int.TryParse(s, out var n) ? n : -1)
            .Where(n => n >= profile.MinNumber && n <= profile.MaxNumber)
            .ToList();
    }

    private sealed class LegacyLottoRow
    {
        public int Id { get; set; }
        public DateTime NgayQuay { get; set; }
        public string KetQua { get; set; } = string.Empty;
    }
}
