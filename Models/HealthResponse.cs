namespace AI_Model_BE.Models;

public record HealthResponse(
    string Status,
    string Service,
    string Environment,
    DateTime UtcTime);
