using AI_Model_BE.Services;
using Microsoft.OpenApi.Models;

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
        if (allowedOrigins.Length > 0)
        {
            policy.WithOrigins(allowedOrigins);
        }

        policy.AllowAnyHeader()
            .AllowAnyMethod()
            .WithExposedHeaders("Content-Type", "Cache-Control", "Connection");
    });
});

builder.Services.AddSingleton<MlPredictionService>();
builder.Services.AddSingleton<DocumentKnowledgeService>();
builder.Services.AddSingleton<LlamaChatService>();

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
