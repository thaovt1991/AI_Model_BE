using AI_Model_BE.Services;
using Microsoft.OpenApi.Models;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();

// Swagger — mở /swagger để kiểm tra BE đang chạy và thử API
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "AI Local API",
        Version = "v1",
        Description = "Backend ML.NET + LLamaSharp local. Dùng GET /api/ai/health để kiểm tra server OK."
    });
});

// CORS: danh sách origin Frontend được gọi API — cấu hình trong appsettings.json → Cors:AllowedOrigins
var corsSection = builder.Configuration.GetSection("Cors") ;
var corsPolicyName = corsSection["PolicyName"] ?? "SpaClients";
var allowedOrigins = corsSection.GetSection("AllowedOrigins").Get<string[]>() ?? [];

builder.Services.AddCors(options =>
{
    options.AddPolicy(corsPolicyName, policy =>
    {
        options.AddPolicy(corsPolicyName,
         policy =>
         {
             policy.WithOrigins(allowedOrigins) // Địa chỉ của Frontend
                   .WithExposedHeaders("Content-Type", "Cache-Control", "Connection") // Quan trọng cho Streaming
                   .AllowAnyMethod()
                   .AllowCredentials(); // Rất quan trọng nếu bạn dùng Auth hoặc Streaming
         });
    });
});

builder.Services.AddSingleton<MlPredictionService>();
builder.Services.AddSingleton<DocumentKnowledgeService>();
builder.Services.AddSingleton<LlamaChatService>();

var app = builder.Build();

// Khởi tạo model ML.NET ngay khi BE start — tránh chờ train/load ở request đầu tiên
//app.Services.GetRequiredService<MlPredictionService>().EnsureModelReady(); //chua lam nen cmt

// Nạp LLM (.gguf) sớm khi Backend khởi động — tránh user chờ 30–60 giây ở tin nhắn chat đầu tiên.
// Tắt bằng cách đặt "Llama:WarmUpOnStart": false trong appsettings.json nếu muốn start nhanh hơn.
if (builder.Configuration.GetValue("Llama:WarmUpOnStart", true))
{
    await app.Services.GetRequiredService<LlamaChatService>().WarmUpAsync();
}

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
