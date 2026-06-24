namespace AI_Model_BE.Models;

public record ChatTurn(string UserMessage, string AssistantReply);

public record ChatMemoryInfo(string MemoryId, int TurnCount, int StoredTurns);

/// <summary>File JSON lưu trên đĩa — bộ nhớ hội thoại vĩnh viễn (theo profileId).</summary>
public record ChatMemoryFile(string MemoryId, List<ChatTurn> Turns, DateTime UpdatedAt);

public record ChatProfileSettings(string? AiName, DateTime UpdatedAt)
{
    public static ChatProfileSettings Empty { get; } = new(null, DateTime.MinValue);
}

public record ChatProfileSettingsFile(string? AiName, DateTime UpdatedAt);

public record ChatProfileSettingsResponse(string ProfileId, string? AiName);

public record UpdateChatProfileSettingsRequest(string? AiName);
