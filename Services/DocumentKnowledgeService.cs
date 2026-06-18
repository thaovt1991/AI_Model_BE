using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using AI_Model_BE.Models;

namespace AI_Model_BE.Services;

/// <summary>
/// Service quản lý tài liệu nội bộ (RAG).
///
/// Cấu trúc thư mục trên ổ đĩa sau khi upload:
///   AI_Model_BE/Documents/
///   ├── index.json                    ← danh sách tất cả tài liệu (metadata)
///   └── {guid-không-dấu-gạch}/        ← mỗi file upload có 1 folder riêng
///       ├── ten-file-goc.pdf          ← bản sao file gốc user upload
///       └── chunks.json               ← text đã chia nhỏ để AI tìm kiếm
/// </summary>
public sealed class DocumentKnowledgeService
{
    private readonly ILogger<DocumentKnowledgeService> _logger;

    // Đường dẫn tuyệt đối tới thư mục Documents/ (ví dụ: D:\...\AI_Model_BE\Documents)
    private readonly string _documentsDir;

    private readonly int _chunkSize;       // Mỗi đoạn text dài bao nhiêu ký tự
    private readonly int _chunkOverlap;    // Đoạn sau chồng lên đoạn trước bao nhiêu ký tự
    private readonly int _maxChunksPerQuery; // Lấy tối đa bao nhiêu đoạn khi chat
    private readonly int _maxContextChars;   // Giới hạn tổng ký tự context gửi vào LLM

    // Lưu metadata tài liệu trong RAM — key là Guid id
    private readonly ConcurrentDictionary<Guid, DocumentInfo> _documents = new();

    // Lưu các đoạn text đã chia trong RAM — key là Guid id
    private readonly ConcurrentDictionary<Guid, List<DocumentChunk>> _chunks = new();

    // Khóa để tránh 2 thread cùng ghi file JSON một lúc
    private readonly object _persistLock = new();

    // Các phần mở rộng file được phép upload
    private static readonly HashSet<string> AllowedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".log",
        ".cs", ".ts", ".js", ".css", ".scss", ".yaml", ".yml", ".sql",
        ".pdf", ".docx", ".xlsx"
    };

    public DocumentKnowledgeService(IWebHostEnvironment env, IConfiguration config, ILogger<DocumentKnowledgeService> logger)
    {
        _logger = logger;

        // env.ContentRootPath = thư mục gốc project Backend (nơi có Program.cs)
        // Path.Combine ghép đường dẫn an toàn trên Windows/Linux
        _documentsDir = Path.Combine(env.ContentRootPath, "Documents");

        // Tạo thư mục Documents nếu chưa có (không lỗi nếu đã tồn tại)
        Directory.CreateDirectory(_documentsDir);

        _chunkSize = config.GetValue("Documents:ChunkSize", 600);
        _chunkOverlap = config.GetValue("Documents:ChunkOverlap", 80);
        _maxChunksPerQuery = config.GetValue("Documents:MaxChunksPerQuery", 3);
        _maxContextChars = config.GetValue("Documents:MaxContextChars", 2400);

        // Khi Backend khởi động: đọc lại file đã lưu từ lần chạy trước
        LoadIndexFromDisk();
    }

    /// <summary>Trả danh sách tài liệu cho Frontend hiển thị.</summary>
    public IReadOnlyList<DocumentInfo> GetAll() =>
        _documents.Values.OrderByDescending(d => d.UploadedAt).ToList();

    /// <summary>
    /// HÀM CHÍNH GHI FILE VÀO Ổ ĐĨA.
    ///
    /// IFormFile là đối tượng ASP.NET Core dùng để nhận file upload từ HTTP request.
    /// File lúc này còn nằm trong bộ nhớ/tạm của request — chưa lưu vĩnh viễn.
    /// Hàm này copy nó xuống ổ đĩa và xử lý text.
    /// </summary>
    public async Task<DocumentUploadResponse> UploadAsync(IFormFile file, CancellationToken cancellationToken = default)
    {
        // file.Length = kích thước file tính bằng byte
        if (file.Length == 0)
        {
            throw new InvalidOperationException("File rỗng.");
        }

        // Path.GetExtension("bao-cao.pdf") → ".pdf"
        var extension = Path.GetExtension(file.FileName);
        if (!AllowedExtensions.Contains(extension))
        {
            throw new InvalidOperationException(
                $"Định dạng {extension} chưa hỗ trợ. Hỗ trợ: {string.Join(", ", AllowedExtensions)}");
        }

        // --- BƯỚC 1: Tạo ID và thư mục riêng cho file này ---
        var id = Guid.NewGuid();
        // ToString("N") = bỏ dấu gạch ngang → a1b2c3d4e5f6... (dùng làm tên folder)
        // Kết quả: Documents/a1b2c3d4e5f6.../
        var docDir = Path.Combine(_documentsDir, id.ToString("N"));

        // Tạo folder con — tương đương mkdir trong terminal
        Directory.CreateDirectory(docDir);

        // --- BƯỚC 2: Ghi file gốc xuống ổ đĩa ---

        // Ghép đường dẫn đầy đủ: Documents/{id}/ten-file.pdf
        var savedPath = Path.Combine(docDir, file.FileName);

        // File.Create(path):
        //   - Tạo file mới trên ổ đĩa tại savedPath
        //   - Nếu file đã tồn tại → ghi đè
        //   - Trả về FileStream (luồng ghi)
        //
        // await using = tự đóng stream sau khi xong (giống using nhưng hỗ trợ async)
        await using (var stream = File.Create(savedPath))
        {
            // CopyToAsync: copy toàn bộ byte từ file upload (IFormFile) sang ổ đĩa (stream)
            // Đây là bước QUAN TRỌNG NHẤT — file thật sự được lưu tại đây
            await file.CopyToAsync(stream, cancellationToken);
        }
        // Sau dòng này, file gốc đã nằm trên ổ đĩa tại savedPath

        // --- BƯỚC 3: Đọc nội dung text từ file vừa lưu ---

        // DocumentTextExtractor có thể block CPU (PDF/Excel) — chạy trên thread pool
        // để không kẹt HTTP request thread trong lúc đọc file lớn.
        var text = await Task.Run(
            () => DocumentTextExtractor.Extract(savedPath),
            cancellationToken);

        // Nếu không đọc được text (file hỏng, PDF scan ảnh...) → xóa folder vừa tạo, báo lỗi
        if (string.IsNullOrWhiteSpace(text))
        {
            // recursive: true = xóa cả folder con bên trong
            Directory.Delete(docDir, recursive: true);
            throw new InvalidOperationException("Không trích xuất được nội dung text từ file.");
        }

        // --- BƯỚC 4: Chia text thành các đoạn nhỏ (chunk) cho AI tìm kiếm ---

        var chunks = SplitIntoChunks(id, file.FileName, text);

        // Preview = 200 ký tự đầu để hiển thị trên UI
        var preview = text.Length <= 200 ? text : text[..200] + "...";

        // --- BƯỚC 5: Lưu metadata vào RAM ---

        var info = new DocumentInfo(
            Id: id,
            FileName: file.FileName,
            SizeBytes: file.Length,
            ChunkCount: chunks.Count,
            Preview: preview,
            UploadedAt: DateTime.UtcNow);

        _documents[id] = info;   // Lưu thông tin tài liệu vào dictionary RAM
        _chunks[id] = chunks;    // Lưu các đoạn text vào dictionary RAM

        // --- BƯỚC 6: Ghi metadata ra file JSON trên ổ đĩa ---

        // Chỉ ghi JSON của file vừa upload — không ghi lại toàn bộ kho tài liệu.
        PersistIndex(id);

        _logger.LogInformation("Đã upload tài liệu {FileName} — {ChunkCount} chunks", file.FileName, chunks.Count);

        return new DocumentUploadResponse(id, file.FileName, file.Length, chunks.Count, preview);
    }

    /// <summary>
    /// Xóa tài liệu khỏi RAM và ổ đĩa.
    /// Gọi khi user bấm nút × trên Frontend.
    /// </summary>
    public bool Delete(Guid id)
    {
        // TryRemove: xóa khỏi dictionary RAM, trả false nếu id không tồn tại
        if (!_documents.TryRemove(id, out _))
        {
            return false;
        }

        _chunks.TryRemove(id, out _);

        // Ghép đường dẫn folder chứa file gốc + chunks.json
        var docDir = Path.Combine(_documentsDir, id.ToString("N"));

        // Directory.Exists: kiểm tra folder có tồn tại trên ổ đĩa không
        if (Directory.Exists(docDir))
        {
            // Directory.Delete recursive: xóa cả folder và mọi file bên trong
            // (file gốc .pdf/.docx + chunks.json đều bị xóa)
            Directory.Delete(docDir, recursive: true);
        }

        // Cập nhật lại index.json (bỏ tài liệu vừa xóa khỏi danh sách)
        PersistIndex(changedDocId: null);
        return true;
    }

    /// <summary>
    /// Tìm các đoạn text liên quan nhất với câu hỏi user, ghép thành context cho LLM.
    /// Không đọc/ghi file — chỉ làm việc với dữ liệu trong RAM (_chunks).
    /// </summary>
    public string BuildContextForQuery(string question, IReadOnlyList<Guid>? documentIds)
    {
        if (documentIds is null || documentIds.Count == 0)
        {
            return string.Empty;
        }

        var queryTokens = Tokenize(question);
        if (queryTokens.Count == 0)
        {
            return string.Empty;
        }

        var candidates = new List<(DocumentChunk Chunk, int Score)>();

        foreach (var docId in documentIds.Distinct())
        {
            if (!_chunks.TryGetValue(docId, out var docChunks))
            {
                continue;
            }

            foreach (var chunk in docChunks)
            {
                var score = ScoreChunk(queryTokens, chunk.Text);
                if (score > 0)
                {
                    candidates.Add((chunk, score));
                }
            }
        }

        // Không match từ khóa → lấy đoạn đầu tiên làm tổng quan
        if (candidates.Count == 0)
        {
            foreach (var docId in documentIds.Distinct())
            {
                if (_chunks.TryGetValue(docId, out var docChunks) && docChunks.Count > 0)
                {
                    candidates.Add((docChunks[0], 1));
                }
            }
        }

        var topChunks = candidates
            .OrderByDescending(c => c.Score)
            .Take(_maxChunksPerQuery)
            .Select(c => c.Chunk)
            .ToList();

        if (topChunks.Count == 0)
        {
            return string.Empty;
        }

        // Ghép các đoạn text làm context — giới hạn _maxContextChars để prompt không quá dài (LLM CPU chậm)
        var sb = new StringBuilder();
        sb.AppendLine("=== TÀI LIỆU NỘI BỘ (chỉ dùng thông tin dưới đây để trả lời) ===");

        foreach (var chunk in topChunks)
        {
            var block =
                $"[File: {chunk.FileName} | Đoạn {chunk.Index + 1}]\n{chunk.Text.Trim()}\n---\n";

            if (sb.Length + block.Length > _maxContextChars)
            {
                break;
            }

            sb.Append(block);
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Chia văn bản dài thành các đoạn nhỏ (chunk).
    /// Ví dụ: text 2000 ký tự, chunkSize=800 → ra 3 đoạn có phần chồng nhau (overlap).
    /// </summary>
    private List<DocumentChunk> SplitIntoChunks(Guid documentId, string fileName, string text)
    {
        text = Regex.Replace(text, @"\s+", " ").Trim();
        var chunks = new List<DocumentChunk>();
        var start = 0;
        var index = 0;

        while (start < text.Length)
        {
            var length = Math.Min(_chunkSize, text.Length - start);
            var chunkText = text.Substring(start, length);

            chunks.Add(new DocumentChunk
            {
                DocumentId = documentId,
                FileName = fileName,
                Index = index,
                Text = chunkText
            });

            if (start + length >= text.Length)
            {
                break;
            }

            // Nhảy tới vị trí tiếp theo, trừ overlap để 2 đoạn liền kề có phần chung
            start += _chunkSize - _chunkOverlap;
            index++;
        }

        return chunks;
    }

    private static HashSet<string> Tokenize(string text)
    {
        var tokens = Regex.Split(text.ToLowerInvariant(), @"[^\p{L}\p{N}]+")
            .Where(t => t.Length >= 2)
            .ToHashSet();
        return tokens;
    }

    private static int ScoreChunk(HashSet<string> queryTokens, string chunkText)
    {
        var chunkTokens = Tokenize(chunkText);
        return queryTokens.Count(t => chunkTokens.Contains(t));
    }

    /// <summary>
    /// ĐỌC FILE TỪ Ổ ĐĨA khi Backend khởi động.
    ///
    /// Vì sao cần hàm này?
    /// - Dữ liệu trong RAM (_documents, _chunks) mất khi tắt Backend.
    /// - Nhưng file trên ổ đĩa (index.json, chunks.json, file gốc) vẫn còn.
    /// - Hàm này đọc JSON từ ổ đĩa → nạp lại vào RAM.
    /// </summary>
    private void LoadIndexFromDisk()
    {
        // Đường dẫn file danh sách tài liệu
        var indexPath = Path.Combine(_documentsDir, "index.json");

        // File.Exists: kiểm tra file có tồn tại không (lần đầu chạy thì chưa có)
        if (!File.Exists(indexPath))
        {
            return; // Chưa upload gì → bỏ qua
        }

        try
        {
            // File.ReadAllText: đọc toàn bộ nội dung file JSON thành 1 chuỗi
            var json = File.ReadAllText(indexPath);

            // Deserialize: chuyển chuỗi JSON → List<DocumentInfo> trong C#
            var items = JsonSerializer.Deserialize<List<DocumentInfo>>(json) ?? [];

            foreach (var doc in items)
            {
                // Nạp metadata vào RAM
                _documents[doc.Id] = doc;

                // Đọc chunks.json của từng tài liệu
                var metaPath = Path.Combine(_documentsDir, doc.Id.ToString("N"), "chunks.json");
                if (File.Exists(metaPath))
                {
                    var chunkJson = File.ReadAllText(metaPath);
                    var chunks = JsonSerializer.Deserialize<List<DocumentChunk>>(chunkJson) ?? [];
                    _chunks[doc.Id] = chunks;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Không load được index tài liệu, bắt đầu với kho trống.");
        }
    }

    /// <summary>
    /// GHI METADATA RA FILE JSON TRÊN Ổ ĐĨA.
    ///
    /// Ghi 2 loại file:
    ///   1. Documents/index.json        — danh sách tất cả tài liệu
    ///   2. Documents/{id}/chunks.json — các đoạn text của từng tài liệu
    ///
    /// Lưu ý: file gốc (.pdf, .docx...) đã được ghi trong UploadAsync,
    ///        hàm này chỉ ghi thêm metadata JSON.
    /// </summary>
    /// <param name="changedDocId">Id tài liệu vừa thay đổi; null = chỉ cập nhật index.json (khi xóa).</param>
    private void PersistIndex(Guid? changedDocId = null)
    {
        // lock: chỉ 1 thread được ghi file tại 1 thời điểm (tránh ghi đè lẫn nhau)
        lock (_persistLock)
        {
            var indexPath = Path.Combine(_documentsDir, "index.json");
            var list = _documents.Values.OrderByDescending(d => d.UploadedAt).ToList();

            // File.WriteAllText: ghi danh sách metadata ra Documents/index.json
            File.WriteAllText(
                indexPath,
                JsonSerializer.Serialize(list, new JsonSerializerOptions { WriteIndented = true }));

            // Khi xóa tài liệu: chỉ cần cập nhật index.json (folder file gốc đã bị xóa)
            if (changedDocId is null)
            {
                return;
            }

            // Khi upload: chỉ ghi chunks.json của file vừa thêm — không ghi lại toàn bộ kho (nhanh hơn)
            if (_chunks.TryGetValue(changedDocId.Value, out var docChunks))
            {
                var metaPath = Path.Combine(_documentsDir, changedDocId.Value.ToString("N"), "chunks.json");
                File.WriteAllText(metaPath, JsonSerializer.Serialize(docChunks));
            }
        }
    }
}
