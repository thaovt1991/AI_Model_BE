using System.Globalization;
using System.Text.Json;
using AI_Model_BE.Models;
using Microsoft.Extensions.Caching.Memory;

namespace AI_Model_BE.Services;

/// <summary>
/// Lấy dữ liệu nến OHLCV + danh mục coin từ Binance Public API (không cần API key).
/// Catalog: GET /api/v3/exchangeInfo → toàn bộ cặp *USDT đang TRADING.
/// </summary>
public sealed class CoinMarketDataService
{
    public const string HttpClientName = nameof(CoinMarketDataService);
    private const string CatalogCacheKey = "coin:catalog:usdt";

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly IMemoryCache _cache;
    private readonly IConfiguration _configuration;
    private readonly ILogger<CoinMarketDataService> _logger;

    /// <summary>Fallback khi Binance không gọi được — vài coin phổ biến.</summary>
    private static readonly IReadOnlyList<CoinInfoDto> FallbackCatalog =
    [
        new("BTC", "Bitcoin", "USDT", "BTCUSDT", "Coin lớn nhất thị trường."),
        new("ETH", "Ethereum", "USDT", "ETHUSDT", "Nền tảng smart contract."),
        new("BNB", "BNB", "USDT", "BNBUSDT", "Coin sàn Binance."),
        new("SOL", "Solana", "USDT", "SOLUSDT", "Layer-1 tốc độ cao."),
        new("XRP", "XRP", "USDT", "XRPUSDT", "Thanh toán xuyên biên giới."),
        new("DOGE", "Dogecoin", "USDT", "DOGEUSDT", "Meme coin thanh khoản lớn."),
        new("ADA", "Cardano", "USDT", "ADAUSDT", "Layer-1 proof-of-stake."),
        new("AVAX", "Avalanche", "USDT", "AVAXUSDT", "Layer-1 Avalanche."),
        new("DOT", "Polkadot", "USDT", "DOTUSDT", "Đa chuỗi Polkadot."),
        new("LINK", "Chainlink", "USDT", "LINKUSDT", "Oracle mạng."),
        new("MATIC", "Polygon", "USDT", "MATICUSDT", "Scaling Ethereum."),
        new("TRX", "TRON", "USDT", "TRXUSDT", "TRON network."),
        new("LTC", "Litecoin", "USDT", "LTCUSDT", "Litecoin."),
        new("ATOM", "Cosmos", "USDT", "ATOMUSDT", "Cosmos Hub."),
        new("UNI", "Uniswap", "USDT", "UNIUSDT", "DEX Uniswap."),
        new("NEAR", "NEAR", "USDT", "NEARUSDT", "NEAR Protocol."),
        new("APT", "Aptos", "USDT", "APTUSDT", "Aptos L1."),
        new("ARB", "Arbitrum", "USDT", "ARBUSDT", "Arbitrum L2."),
        new("OP", "Optimism", "USDT", "OPUSDT", "Optimism L2."),
        new("SUI", "Sui", "USDT", "SUIUSDT", "Sui L1."),
        new("PEPE", "Pepe", "USDT", "PEPEUSDT", "Meme coin Pepe."),
        new("SHIB", "Shiba Inu", "USDT", "SHIBUSDT", "Meme coin Shiba."),
        new("TON", "Toncoin", "USDT", "TONUSDT", "TON network."),
        new("FIL", "Filecoin", "USDT", "FILUSDT", "Lưu trữ phi tập trung."),
    ];

    /// <summary>Thứ tự ưu tiên hiện đầu danh sách (thanh khoản / phổ biến).</summary>
    private static readonly string[] PopularOrder =
    [
        "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "DOT", "LINK",
        "TRX", "TON", "SHIB", "PEPE", "LTC", "BCH", "NEAR", "APT", "ARB", "OP",
        "SUI", "UNI", "ATOM", "FIL", "ICP", "ETC", "XLM", "HBAR", "INJ", "RENDER"
    ];

    /// <summary>Tên đẹp cho một số base asset phổ biến (còn lại dùng chính symbol).</summary>
    private static readonly Dictionary<string, string> WellKnownNames = new(StringComparer.OrdinalIgnoreCase)
    {
        ["BTC"] = "Bitcoin",
        ["ETH"] = "Ethereum",
        ["BNB"] = "BNB",
        ["SOL"] = "Solana",
        ["XRP"] = "XRP",
        ["DOGE"] = "Dogecoin",
        ["ADA"] = "Cardano",
        ["AVAX"] = "Avalanche",
        ["DOT"] = "Polkadot",
        ["LINK"] = "Chainlink",
        ["MATIC"] = "Polygon",
        ["POL"] = "Polygon",
        ["TRX"] = "TRON",
        ["LTC"] = "Litecoin",
        ["ATOM"] = "Cosmos",
        ["UNI"] = "Uniswap",
        ["NEAR"] = "NEAR",
        ["APT"] = "Aptos",
        ["ARB"] = "Arbitrum",
        ["OP"] = "Optimism",
        ["SUI"] = "Sui",
        ["PEPE"] = "Pepe",
        ["SHIB"] = "Shiba Inu",
        ["TON"] = "Toncoin",
        ["FIL"] = "Filecoin",
        ["BCH"] = "Bitcoin Cash",
        ["ETC"] = "Ethereum Classic",
        ["XLM"] = "Stellar",
        ["HBAR"] = "Hedera",
        ["INJ"] = "Injective",
        ["RENDER"] = "Render",
        ["ICP"] = "Internet Computer",
        ["AAVE"] = "Aave",
        ["MKR"] = "Maker",
        ["CRV"] = "Curve",
        ["WLD"] = "Worldcoin",
        ["FET"] = "Fetch.ai",
        ["TAO"] = "Bittensor",
    };

    private static readonly HashSet<string> AllowedIntervals = new(StringComparer.OrdinalIgnoreCase)
    {
        "15m", "1h", "4h", "1d"
    };

    public CoinMarketDataService(
        IHttpClientFactory httpClientFactory,
        IMemoryCache cache,
        IConfiguration configuration,
        ILogger<CoinMarketDataService> logger)
    {
        _httpClientFactory = httpClientFactory;
        _cache = cache;
        _configuration = configuration;
        _logger = logger;
    }

    /// <summary>
    /// Danh mục đầy đủ cặp USDT spot đang TRADING trên Binance (cache dài).
    /// Thường vài trăm coin — Frontend search để lọc nhanh.
    /// </summary>
    public async Task<IReadOnlyList<CoinInfoDto>> GetCatalogAsync(CancellationToken cancellationToken = default)
    {
        if (_cache.TryGetValue(CatalogCacheKey, out IReadOnlyList<CoinInfoDto>? cached) &&
            cached is { Count: > 0 })
        {
            return cached;
        }

        try
        {
            var list = await FetchUsdtSpotCatalogAsync(cancellationToken);
            if (list.Count == 0)
            {
                _logger.LogWarning("Binance exchangeInfo trả 0 cặp USDT — dùng fallback.");
                return FallbackCatalog;
            }

            var cacheHours = _configuration.GetValue("Coin:CatalogCacheHours", 6);
            _cache.Set(CatalogCacheKey, list, TimeSpan.FromHours(Math.Clamp(cacheHours, 1, 24)));
            _logger.LogInformation("Đã nạp catalog Binance: {Count} cặp USDT", list.Count);
            return list;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không tải catalog Binance — dùng fallback {Count} coin", FallbackCatalog.Count);
            return FallbackCatalog;
        }
    }

    public async Task<CoinInfoDto> ResolveAsync(string symbol, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            throw new ArgumentException("Symbol bắt buộc.");
        }

        var key = symbol.Trim().ToUpperInvariant();
        // Cho phép nhập cả BTC hoặc BTCUSDT
        if (key.EndsWith("USDT", StringComparison.Ordinal) && key.Length > 4)
        {
            key = key[..^4];
        }

        var catalog = await GetCatalogAsync(cancellationToken);
        var hit = catalog.FirstOrDefault(c =>
            string.Equals(c.Symbol, key, StringComparison.OrdinalIgnoreCase));
        if (hit is not null)
        {
            return hit;
        }

        // Cho phép trade pair hợp lệ dù chưa có trong cache (ví dụ cache cũ)
        var pair = key + "USDT";
        return new CoinInfoDto(
            key,
            WellKnownNames.GetValueOrDefault(key, key),
            "USDT",
            pair,
            $"Cặp {pair} trên Binance spot.");
    }

    public string NormalizeInterval(string? interval)
    {
        var value = string.IsNullOrWhiteSpace(interval) ? "1h" : interval.Trim();
        if (!AllowedIntervals.Contains(value))
        {
            throw new ArgumentException($"Interval không hỗ trợ: {interval}. Dùng: 15m, 1h, 4h, 1d.");
        }

        return value.ToLowerInvariant() switch
        {
            "15m" => "15m",
            "1h" => "1h",
            "4h" => "4h",
            "1d" => "1d",
            _ => "1h"
        };
    }

    /// <summary>
    /// Tải nến từ Binance (có cache ngắn để tránh spam API).
    /// </summary>
    public async Task<IReadOnlyList<CoinCandle>> GetKlinesAsync(
        string binancePair,
        string interval,
        int limit,
        CancellationToken cancellationToken = default)
    {
        limit = Math.Clamp(limit, 50, 1000);
        var cacheMinutes = _configuration.GetValue("Coin:CacheMinutes", 2);
        var cacheKey = $"coin:klines:{binancePair}:{interval}:{limit}";

        if (_cache.TryGetValue(cacheKey, out IReadOnlyList<CoinCandle>? cached) && cached is not null)
        {
            return cached;
        }

        var baseUrl = _configuration["Coin:BinanceBaseUrl"] ?? "https://api.binance.com";
        var url =
            $"{baseUrl.TrimEnd('/')}/api/v3/klines?symbol={binancePair}&interval={interval}&limit={limit}";

        var client = _httpClientFactory.CreateClient(HttpClientName);
        using var response = await client.GetAsync(url, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            _logger.LogWarning("Binance klines lỗi {Status}: {Body}", response.StatusCode, body);
            throw new InvalidOperationException(
                $"Không lấy được dữ liệu Binance ({(int)response.StatusCode}). Kiểm tra mạng hoặc thử lại.");
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);

        var candles = new List<CoinCandle>();
        foreach (var row in doc.RootElement.EnumerateArray())
        {
            if (row.GetArrayLength() < 6)
            {
                continue;
            }

            candles.Add(new CoinCandle
            {
                OpenTimeUtc = DateTimeOffset.FromUnixTimeMilliseconds(row[0].GetInt64()).UtcDateTime,
                Open = ParseDouble(row[1]),
                High = ParseDouble(row[2]),
                Low = ParseDouble(row[3]),
                Close = ParseDouble(row[4]),
                Volume = ParseDouble(row[5])
            });
        }

        if (candles.Count < 40)
        {
            throw new InvalidOperationException("Dữ liệu nến quá ít để huấn luyện mô hình.");
        }

        _cache.Set(cacheKey, candles, TimeSpan.FromMinutes(Math.Max(1, cacheMinutes)));
        _logger.LogInformation(
            "Đã tải {Count} nến {Pair} {Interval} từ Binance",
            candles.Count,
            binancePair,
            interval);

        return candles;
    }

    private async Task<List<CoinInfoDto>> FetchUsdtSpotCatalogAsync(CancellationToken cancellationToken)
    {
        var baseUrl = _configuration["Coin:BinanceBaseUrl"] ?? "https://api.binance.com";
        var url = $"{baseUrl.TrimEnd('/')}/api/v3/exchangeInfo";
        var client = _httpClientFactory.CreateClient(HttpClientName);

        using var response = await client.GetAsync(url, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);

        if (!doc.RootElement.TryGetProperty("symbols", out var symbols) ||
            symbols.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var list = new List<CoinInfoDto>();
        foreach (var s in symbols.EnumerateArray())
        {
            var status = s.TryGetProperty("status", out var st) ? st.GetString() : null;
            var quote = s.TryGetProperty("quoteAsset", out var q) ? q.GetString() : null;
            var baseAsset = s.TryGetProperty("baseAsset", out var b) ? b.GetString() : null;
            var pair = s.TryGetProperty("symbol", out var sym) ? sym.GetString() : null;
            var isSpot = !s.TryGetProperty("isSpotTradingAllowed", out var spot) || spot.GetBoolean();

            if (!string.Equals(status, "TRADING", StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(quote, "USDT", StringComparison.OrdinalIgnoreCase) ||
                string.IsNullOrWhiteSpace(baseAsset) ||
                string.IsNullOrWhiteSpace(pair) ||
                !isSpot)
            {
                continue;
            }

            // Bỏ qua leverage tokens / odd suffixes phổ biến gây nhiễu
            if (baseAsset.EndsWith("UP", StringComparison.OrdinalIgnoreCase) ||
                baseAsset.EndsWith("DOWN", StringComparison.OrdinalIgnoreCase) ||
                baseAsset.EndsWith("BULL", StringComparison.OrdinalIgnoreCase) ||
                baseAsset.EndsWith("BEAR", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var name = WellKnownNames.GetValueOrDefault(baseAsset, baseAsset);
            list.Add(new CoinInfoDto(
                baseAsset.ToUpperInvariant(),
                name,
                "USDT",
                pair.ToUpperInvariant(),
                $"Cặp {pair} · Binance spot USDT."));
        }

        // Ưu tiên coin phổ biến → còn lại theo alphabet
        var popularIndex = PopularOrder
            .Select((sym, i) => (sym, i))
            .ToDictionary(x => x.sym, x => x.i, StringComparer.OrdinalIgnoreCase);

        return list
            .GroupBy(c => c.Symbol, StringComparer.OrdinalIgnoreCase)
            .Select(g => g.First())
            .OrderBy(c => popularIndex.TryGetValue(c.Symbol, out var idx) ? idx : 10_000)
            .ThenBy(c => c.Symbol, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static double ParseDouble(JsonElement el) =>
        double.Parse(el.GetString() ?? "0", CultureInfo.InvariantCulture);
}
