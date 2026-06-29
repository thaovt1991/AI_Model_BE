using AI_Model_BE.Programs;
using AI_Model_BE.Services;
using Microsoft.OpenApi.Models;

if (args.Contains("lotto-demo", StringComparer.OrdinalIgnoreCase))
{
    Environment.Exit(await LottoForecastConsoleApp.RunAsync(args));
}

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "AI Local API",
        Version = "v1",
        Description = "Backend ML.NET + LLamaSharp local. Dùng GET /api/ai/health để kiểm tra server OK."
    });
});

// CORS — đọc từ appsettings.json → Cors:AllowedOrigins
// Lưu ý: KHÔNG dùng AllowAnyOrigin() + AllowCredentials() cùng lúc — browser sẽ chặn.
var corsSection = builder.Configuration.GetSection("Cors");
var corsPolicyName = corsSection["PolicyName"] ?? "SpaClients";
var allowedOrigins = corsSection.GetSection("AllowedOrigins").Get<string[]>()
    ?.Where(static origin => !string.IsNullOrWhiteSpace(origin))
    .Select(static origin => origin.Trim().TrimEnd('/'))
    .Distinct(StringComparer.OrdinalIgnoreCase)
    .ToArray() ?? [];

builder.Services.AddCors(options =>
{
    options.AddPolicy(corsPolicyName, policy =>
    {
        //if (allowedOrigins.Length > 0)
        //{
        //    policy.WithOrigins(allowedOrigins);
        //}
        policy.AllowAnyOrigin()
            .AllowAnyHeader()
            .AllowAnyMethod()
            .WithExposedHeaders("Content-Type", "Cache-Control", "Connection");
    });
});

builder.Services.AddMemoryCache();
builder.Services.AddHttpClient(nameof(MinhNgocLotteryScraper), client =>
{
    client.Timeout = TimeSpan.FromSeconds(30);
    client.DefaultRequestHeaders.UserAgent.ParseAdd(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI_Model_BE/1.0");
    client.DefaultRequestHeaders.AcceptLanguage.ParseAdd("vi-VN,vi;q=0.9");
});
builder.Services.AddHttpClient(nameof(VietlottLotteryScraper), client =>
{
    client.Timeout = TimeSpan.FromSeconds(30);
    client.DefaultRequestHeaders.UserAgent.ParseAdd(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI_Model_BE/1.0");
    client.DefaultRequestHeaders.AcceptLanguage.ParseAdd("vi-VN,vi;q=0.9");
});
builder.Services.AddSingleton<MinhNgocScrapeSettingsService>();
builder.Services.AddSingleton<MinhNgocLotteryScraper>();
builder.Services.AddSingleton<VietlottLotteryScraper>();
builder.Services.AddSingleton<MlPredictionService>();
builder.Services.AddSingleton<LotteryRecordLoader>();
builder.Services.AddSingleton<LottoForecastService>();
builder.Services.AddSingleton<LottoChatIntentService>();

// === Học model ngầm (LoRA) ===
builder.Services.AddSingleton<LearningSettingsService>();
builder.Services.AddSingleton<LearningDataCollectorService>();
builder.Services.AddSingleton<ModelLearningService>();
builder.Services.AddHostedService<BackgroundModelLearningHostedService>();

builder.Services.AddSingleton<ChatMemoryService>();
builder.Services.AddSingleton<ChatProfileSettingsService>();
builder.Services.AddSingleton<DocumentKnowledgeService>();
builder.Services.AddSingleton<LlamaChatService>();               // Bước 3: chat + nạp adapter sau train

var app = builder.Build();

if (allowedOrigins.Length == 0)
{
    app.Logger.LogWarning(
        "Cors:AllowedOrigins dang rong — Frontend tren IIS se bi loi CORS. Them URL UI vao appsettings.");
}
else
{
    app.Logger.LogInformation(
        "CORS policy {Policy}: {Origins}",
        corsPolicyName,
        string.Join(", ", allowedOrigins));
}

if (builder.Configuration.GetValue("Llama:WarmUpOnStart", true))
{
    await app.Services.GetRequiredService<LlamaChatService>().WarmUpAsync();
}

app.UseRouting();
app.UseCors(corsPolicyName);

var swaggerEnabled = app.Configuration.GetValue("Swagger:Enabled", app.Environment.IsDevelopment());
if (swaggerEnabled)
{
    app.UseSwagger();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/swagger/v1/swagger.json", "AI Local API v1");
        options.RoutePrefix = "swagger";
    });
}

app.MapGet("/", () => Results.Redirect("/swagger")).ExcludeFromDescription();
app.MapControllers();

app.Run();
