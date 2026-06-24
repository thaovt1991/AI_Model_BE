using System.Collections.Concurrent;
using System.Text.Json;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Bộ nhớ hội thoại theo profileId + conversationId (mỗi cuộc chat mới = file riêng).
/// Lưu: ChatMemory/{profileId}/{conversationId}.json
/// </summary>
public sealed class ChatMemoryService
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private readonly ILogger<ChatMemoryService> _logger;
    private readonly string _memoryDir;
    private readonly bool _persistToDisk;
    private readonly int _maxHistoryTurns;
    private readonly int _maxStoredTurns;
    private readonly ConcurrentDictionary<string, List<ChatTurn>> _sessions = new();
    private readonly object _fileLock = new();

    public ChatMemoryService(
        IWebHostEnvironment env,
        IConfiguration configuration,
        ILogger<ChatMemoryService> logger)
    {
        _logger = logger;
        _memoryDir = Path.Combine(
            env.ContentRootPath,
            configuration.GetValue("ChatMemory:Directory", "ChatMemory") ?? "ChatMemory");
        _persistToDisk = configuration.GetValue("ChatMemory:PersistToDisk", true);
        _maxHistoryTurns = configuration.GetValue(
            "ChatMemory:MaxHistoryTurns",
            configuration.GetValue("Llama:MaxHistoryTurns", 8));
        _maxStoredTurns = configuration.GetValue("ChatMemory:MaxStoredTurns", 100);

        if (_persistToDisk)
        {
            Directory.CreateDirectory(_memoryDir);
            _logger.LogInformation("Chat memory lưu tại {Dir} (tối đa {Stored} lượt, {Prompt} lượt trong prompt)",
                _memoryDir, _maxStoredTurns, _maxHistoryTurns);
        }
    }

    public static string ResolveMemoryKey(string? profileId, string? conversationId)
    {
        if (string.IsNullOrWhiteSpace(profileId) || string.IsNullOrWhiteSpace(conversationId))
        {
            return string.Empty;
        }

        return $"{NormalizeMemoryId(profileId)}/{NormalizeMemoryId(conversationId)}";
    }

    public IReadOnlyList<ChatTurn> GetHistoryForPrompt(string? profileId, string? conversationId)
    {
        var key = ResolveMemoryKey(profileId, conversationId);
        if (string.IsNullOrEmpty(key))
        {
            return [];
        }

        var turns = GetOrLoadSession(key);
        lock (turns)
        {
            if (turns.Count <= _maxHistoryTurns)
            {
                return turns.ToList();
            }

            return turns.Skip(turns.Count - _maxHistoryTurns).ToList();
        }
    }

    public IReadOnlyList<ChatTurn> GetAllStored(string? profileId, string? conversationId)
    {
        var key = ResolveMemoryKey(profileId, conversationId);
        if (string.IsNullOrEmpty(key))
        {
            return [];
        }

        var turns = GetOrLoadSession(key);
        lock (turns)
        {
            return turns.ToList();
        }
    }

    public void Append(string? profileId, string? conversationId, string userMessage, string assistantReply)
    {
        var key = ResolveMemoryKey(profileId, conversationId);
        if (string.IsNullOrEmpty(key) || string.IsNullOrWhiteSpace(assistantReply))
        {
            return;
        }

        var turns = GetOrLoadSession(key);
        lock (turns)
        {
            turns.Add(new ChatTurn(userMessage.Trim(), assistantReply.Trim()));

            while (turns.Count > _maxStoredTurns)
            {
                turns.RemoveAt(0);
            }

            Persist(key, turns);
        }

        _logger.LogDebug("Chat memory {MemoryKey}: {Count} lượt đã lưu", key, turns.Count);
    }

    /// <summary>Xóa toàn bộ hội thoại của profile (mọi cuộc chat).</summary>
    public bool ClearProfile(string? profileId)
    {
        if (string.IsNullOrWhiteSpace(profileId))
        {
            return false;
        }

        var normalizedProfile = NormalizeMemoryId(profileId);
        var removed = false;

        foreach (var key in _sessions.Keys.Where(k => k.StartsWith($"{normalizedProfile}/", StringComparison.Ordinal)).ToList())
        {
            _sessions.TryRemove(key, out _);
            removed = true;
        }

        if (!_persistToDisk)
        {
            return removed;
        }

        var profileDir = Path.Combine(_memoryDir, SanitizePathSegment(normalizedProfile));
        if (Directory.Exists(profileDir))
        {
            lock (_fileLock)
            {
                Directory.Delete(profileDir, recursive: true);
            }

            return true;
        }

        var legacyFile = Path.Combine(_memoryDir, $"{SanitizePathSegment(normalizedProfile)}.json");
        if (File.Exists(legacyFile))
        {
            lock (_fileLock)
            {
                File.Delete(legacyFile);
            }

            return true;
        }

        return removed;
    }

    public ChatMemoryInfo? GetInfo(string? profileId, string? conversationId)
    {
        var key = ResolveMemoryKey(profileId, conversationId);
        if (string.IsNullOrEmpty(key))
        {
            return null;
        }

        var turns = GetOrLoadSession(key);
        lock (turns)
        {
            var promptTurns = Math.Min(turns.Count, _maxHistoryTurns);
            return new ChatMemoryInfo(key, promptTurns, turns.Count);
        }
    }

    private List<ChatTurn> GetOrLoadSession(string key) =>
        _sessions.GetOrAdd(key, LoadFromDisk);

    private List<ChatTurn> LoadFromDisk(string key)
    {
        if (!_persistToDisk)
        {
            return [];
        }

        var path = GetFilePath(key);
        if (!File.Exists(path))
        {
            return [];
        }

        try
        {
            var json = File.ReadAllText(path);
            var file = JsonSerializer.Deserialize<ChatMemoryFile>(json, JsonOptions);
            return file?.Turns ?? [];
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không đọc được file memory {Path}", path);
            return [];
        }
    }

    private void Persist(string key, List<ChatTurn> turns)
    {
        if (!_persistToDisk)
        {
            return;
        }

        var path = GetFilePath(key);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var file = new ChatMemoryFile(key, turns.ToList(), DateTime.UtcNow);

        lock (_fileLock)
        {
            File.WriteAllText(path, JsonSerializer.Serialize(file, JsonOptions));
        }
    }

    private string GetFilePath(string key)
    {
        var parts = key.Split('/', 2, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2)
        {
            return Path.Combine(_memoryDir, $"{SanitizePathSegment(key)}.json");
        }

        return Path.Combine(
            _memoryDir,
            SanitizePathSegment(parts[0]),
            $"{SanitizePathSegment(parts[1])}.json");
    }

    private static string SanitizePathSegment(string segment) =>
        string.Concat(segment.Where(static c => char.IsLetterOrDigit(c) || c == '-'));

    private static string NormalizeMemoryId(string memoryId) =>
        memoryId.Trim().ToLowerInvariant();
}
