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

// CORS: cho phép Angular dev server (port 4200) gọi API
builder.Services.AddCors(options =>
{
    options.AddPolicy("AngularDev", policy =>
    {
        policy.WithOrigins("http://localhost:4200", "http://127.0.0.1:4200")
            .AllowAnyHeader()
            .AllowAnyMethod();
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

app.UseCors("AngularDev");

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
