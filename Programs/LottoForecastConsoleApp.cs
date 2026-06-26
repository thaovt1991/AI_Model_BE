using AI_Model_BE.Models;
using AI_Model_BE.Services;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

namespace AI_Model_BE.Programs;

public static class LottoForecastConsoleApp
{
    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || !args.Contains("lotto-demo", StringComparer.OrdinalIgnoreCase))
        {
            return 0;
        }

        Console.OutputEncoding = System.Text.Encoding.UTF8;
        var env = new HostEnv(Directory.GetCurrentDirectory());

        var services = new ServiceCollection();
        services.AddLogging();
        services.AddMemoryCache();
        services.AddSingleton<IConfiguration>(new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["MinhNgoc:ScrapingEnabled"] = "true",
            ["MinhNgoc:RequestDelayMs"] = "800",
        }).Build());
        services.AddHttpClient(nameof(MinhNgocLotteryScraper));
        services.AddSingleton<MinhNgocScrapeSettingsService>();
        services.AddSingleton<MinhNgocLotteryScraper>();
        services.AddSingleton<LotteryRecordLoader>();
        services.AddSingleton<MlPredictionService>(sp => new MlPredictionService(
            env,
            sp.GetRequiredService<ILogger<MlPredictionService>>(),
            sp.GetRequiredService<MinhNgocLotteryScraper>(),
            sp.GetRequiredService<LotteryRecordLoader>(),
            sp.GetRequiredService<MinhNgocScrapeSettingsService>()));
        services.AddSingleton<LottoForecastService>();

        var provider = services.BuildServiceProvider();
        var ml = provider.GetRequiredService<MlPredictionService>();

        var result = await ml.PredictLotteryForDaiAsync(
            LottoGameKinds.XsMienNam,
            "hcm");

        Console.WriteLine($"[{result.TenDai}] Kỳ trước #{result.KyQuayTruoc} ({result.NgayQuayKyTruoc}): {string.Join(" · ", result.KetQuaKyTruoc.Select(n => n.ToString("00")))}");
        Console.WriteLine($"  → Dự đoán #{result.KyQuayDuDoan} ({result.NgayQuayDuDoan}): {string.Join(" · ", result.CacSoDuDoan.Select(n => n.ToString("00")))}");

        return 0;
    }

    private sealed class HostEnv(string contentRoot) : IWebHostEnvironment
    {
        public string WebRootPath { get; set; } = contentRoot;
        public string EnvironmentName { get; set; } = "Development";
        public string ApplicationName { get; set; } = "LottoForecastDemo";
        public string ContentRootPath { get; set; } = contentRoot;
        public Microsoft.Extensions.FileProviders.IFileProvider WebRootFileProvider { get; set; } =
            new Microsoft.Extensions.FileProviders.PhysicalFileProvider(contentRoot);
        public Microsoft.Extensions.FileProviders.IFileProvider ContentRootFileProvider { get; set; } =
            new Microsoft.Extensions.FileProviders.PhysicalFileProvider(contentRoot);
    }
}
