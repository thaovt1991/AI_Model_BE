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
builder.Services.AddSingleton<LlamaChatService>();

var app = builder.Build();

// Khởi tạo model ML.NET ngay khi BE start
app.Services.GetRequiredService<MlPredictionService>().EnsureModelReady();

app.UseCors("AngularDev");

app.MapControllers();

app.Run();
