using System.Collections.Concurrent;
using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Cài đặt theo profileId (trình duyệt) — ví dụ tên AI. Lưu ChatProfiles/{profileId}.json
/// </summary>
public sealed class ChatProfileSettingsService
{
    private const int MaxAiNameLength = 32;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly ILogger<ChatProfileSettingsService> _logger;
    private readonly string _profilesDir;
    private readonly object _fileLock = new();
    private readonly ConcurrentDictionary<string, ChatProfileSettings> _cache = new();

    public ChatProfileSettingsService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        ILogger<ChatProfileSettingsService> logger)
    {
        _logger = logger;
        _profilesDir = Path.Combine(
            env.ContentRootPath,
            configuration.GetValue("ChatProfile:Directory", "ChatProfiles") ?? "ChatProfiles");
        Directory.CreateDirectory(_profilesDir);
    }

    public ChatProfileSettingsResponse Get(string? profileId)
    {
        var settings = GetOrLoad(profileId);
        return ToResponse(profileId, settings);
    }

    public ChatProfileSettingsResponse Update(string? profileId, string? aiName)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            throw new InvalidOperationException("Thiếu profileId.");
        }

        var normalizedProfile = NormalizeProfileId(profileId);
        var normalizedName = NormalizeAiName(aiName);
        var settings = new ChatProfileSettings(normalizedName, DateTime.UtcNow);

        _cache[normalizedProfile] = settings;
        Persist(normalizedProfile, settings);

        _logger.LogInformation(
            "Profile {ProfileId} đặt tên AI: {AiName}",
            normalizedProfile,
            string.IsNullOrEmpty(normalizedName) ? "(mặc định)" : normalizedName);

        return ToResponse(normalizedProfile, settings);
    }

    public string? GetAiName(string? profileId)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            return null;
        }

        return GetOrLoad(profileId).AiName;
    }

    private ChatProfileSettings GetOrLoad(string? profileId)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            return ChatProfileSettings.Empty;
        }

        var key = NormalizeProfileId(profileId);
        return _cache.GetOrAdd(key, LoadFromDisk);
    }

    private ChatProfileSettings LoadFromDisk(string profileId)
    {
        var path = GetFilePath(profileId);
        if (!File.Exists(path))
        {
            return ChatProfileSettings.Empty;
        }

        try
        {
            var json = File.ReadAllText(path);
            var file = JsonSerializer.Deserialize<ChatProfileSettingsFile>(json, JsonOptions);
            var aiName = NormalizeAiName(file?.AiName);
            return new ChatProfileSettings(aiName, file?.UpdatedAt ?? DateTime.UtcNow);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không đọc được profile settings {Path}", path);
            return ChatProfileSettings.Empty;
        }
    }

    private void Persist(string profileId, ChatProfileSettings settings)
    {
        var path = GetFilePath(profileId);
        var file = new ChatProfileSettingsFile(settings.AiName, settings.UpdatedAt);

        lock (_fileLock)
        {
            File.WriteAllText(path, JsonSerializer.Serialize(file, JsonOptions));
        }
    }

    private string GetFilePath(string profileId) =>
        Path.Combine(_profilesDir, $"{SanitizeFileName(profileId)}.json");

    private static string NormalizeProfileId(string profileId) =>
        profileId.Trim().ToLowerInvariant();

    private static string SanitizeFileName(string segment) =>
        string.Concat(segment.Where(static c => char.IsLetterOrDigit(c) || c == '-'));

    private static string? NormalizeAiName(string? aiName)
    {
        if (string.IsNullOrWhiteSpace(aiName))
        {
            return null;
        }

        var trimmed = aiName.Trim();
        if (trimmed.Length > MaxAiNameLength)
        {
            trimmed = trimmed[..MaxAiNameLength];
        }

        return trimmed;
    }

    private static ChatProfileSettingsResponse ToResponse(string? profileId, ChatProfileSettings settings) =>
        new(profileId ?? string.Empty, settings.AiName);
}
