using AI_Model_BE.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

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
app.Services.GetRequiredService<MlPredictionService>().EnsureModelReady();

// Nạp LLM (.gguf) sớm khi Backend khởi động — tránh user chờ 30–60 giây ở tin nhắn chat đầu tiên.
// Tắt bằng cách đặt "Llama:WarmUpOnStart": false trong appsettings.json nếu muốn start nhanh hơn.
if (builder.Configuration.GetValue("Llama:WarmUpOnStart", true))
{
    await app.Services.GetRequiredService<LlamaChatService>().WarmUpAsync();
}

app.UseCors("AngularDev");

app.MapControllers();

app.Run();
