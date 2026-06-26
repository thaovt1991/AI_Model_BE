using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>Huấn luyện SSA và dự đoán kỳ tiếp theo — tách file partial.</summary>
public sealed partial class MlPredictionService
{
    public async Task<List<LotteryRecord>> LoadHistoryForApiAsync(
        string gameKind,
        string? daiCode,
        CancellationToken cancellationToken = default)
    {
        var daiName = ResolveDaiName(gameKind, daiCode);
        return await LoadDaiHistoryAsync(gameKind, daiCode, daiName, cancellationToken);
    }

    private static bool IsSouthernRegion(string gameKind) =>
        gameKind is LottoGameKinds.XsMienNam or LottoGameKinds.XsMienTrung;

    private static bool IsVietlott(string gameKind) =>
        gameKind is LottoGameKinds.Vietlott645 or LottoGameKinds.Vietlott655;

    private static LottoLotoExtras TrimLotoExtras(LottoLotoExtras full, string gameKind)
    {
        if (gameKind == LottoGameKinds.XsMienBac)
        {
            return full;
        }

        return new LottoLotoExtras
        {
            BachThuLo = full.BachThuLo
        };
    }
}
