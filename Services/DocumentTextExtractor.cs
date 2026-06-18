using System.Text;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;
using UglyToad.PdfPig;
using WordTable = DocumentFormat.OpenXml.Wordprocessing.Table;
using WordTableCell = DocumentFormat.OpenXml.Wordprocessing.TableCell;
using WordTableRow = DocumentFormat.OpenXml.Wordprocessing.TableRow;
using WordParagraph = DocumentFormat.OpenXml.Wordprocessing.Paragraph;
using WordText = DocumentFormat.OpenXml.Wordprocessing.Text;

namespace AI_Model_BE.Services;

/// <summary>
/// Trích xuất text từ nhiều định dạng file phục vụ RAG nội bộ.
/// Mỗi hàm tương ứng một loại file — có comment chi tiết bên trong.
/// </summary>
public static class DocumentTextExtractor
{
    /// <summary>
    /// Phiên bản đồng bộ — gọi từ Task.Run trong UploadAsync để không block HTTP thread.
    /// Chọn hàm đọc theo phần mở rộng file (.pdf / .docx / .xlsx / .txt ...).
    /// </summary>
    public static string Extract(string path)
    {
        var extension = Path.GetExtension(path).ToLowerInvariant();

        return extension switch
        {
            // PDF: dùng thư viện PdfPig đọc từng trang
            ".pdf" => ExtractPdf(path),

            // Word (.docx): file ZIP chứa XML, dùng Open XML SDK
            ".docx" => ExtractDocx(path),

            // Excel (.xlsx): đọc từng sheet, từng dòng/cột
            ".xlsx" => ExtractXlsx(path),

            // HTML: bỏ thẻ, giữ text
            ".html" or ".htm" => StripHtmlTags(File.ReadAllText(path)),

            // Các file text thuần: đọc UTF-8 trực tiếp (.txt, .md, .csv, .json...)
            _ => File.ReadAllText(path, Encoding.UTF8)
        };
    }

    /// <summary>
    /// Wrapper async — chạy Extract trên thread pool (dùng khi cần await).
    /// </summary>
    public static Task<string> ExtractAsync(string path, CancellationToken cancellationToken = default) =>
        Task.Run(() => Extract(path), cancellationToken);

    /// <summary>
    /// Đọc file PDF bằng UglyToad.PdfPig.
    ///
    /// Cách hoạt động:
    /// 1. PdfDocument.Open(path) — mở file PDF (không cần Adobe).
    /// 2. GetPages() — lấy danh sách trang.
    /// 3. page.Text — PdfPig tự trích xuất text layer trên mỗi trang.
    ///
    /// Lưu ý:
    /// - PDF scan (ảnh) không có text layer → kết quả rỗng (cần OCR riêng).
    /// - PDF có bảng phức tạp có thể mất thứ tự đọc.
    /// </summary>
    private static string ExtractPdf(string path)
    {
        var sb = new StringBuilder();

        using (var document = PdfDocument.Open(path))
        {
            var pageNumber = 1;
            foreach (var page in document.GetPages())
            {
                var pageText = page.Text?.Trim();
                if (!string.IsNullOrWhiteSpace(pageText))
                {
                    sb.AppendLine($"[Trang {pageNumber}]");
                    sb.AppendLine(pageText);
                    sb.AppendLine();
                }

                pageNumber++;
            }
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Đọc file Word (.docx) bằng DocumentFormat.OpenXml.
    ///
    /// Cấu trúc .docx:
    /// - Thực chất là file ZIP.
    /// - Nội dung chính nằm trong word/document.xml.
    ///
    /// Cách hoạt động:
    /// 1. WordprocessingDocument.Open — mở package Open XML.
    /// 2. MainDocumentPart.Document.Body — thân tài liệu.
    /// 3. Duyệt Paragraph (w:p) → Text (w:t) trong từng đoạn.
    /// 4. Duyệt Table (w:tbl) → đọc text từng ô để không bỏ bảng.
    ///
    /// Không hỗ trợ .doc (Word 97-2003 binary) — chỉ .docx.
    /// </summary>
    private static string ExtractDocx(string path)
    {
        var sb = new StringBuilder();

        using (var document = WordprocessingDocument.Open(path, isEditable: false))
        {
            var body = document.MainDocumentPart?.Document?.Body;
            if (body is null)
            {
                return string.Empty;
            }

            foreach (var element in body.Elements())
            {
                switch (element)
                {
                    // Đoạn văn bản thường
                    case WordParagraph paragraph:
                        var paraText = string.Concat(paragraph.Descendants<WordText>().Select(t => t.Text)).Trim();
                        if (!string.IsNullOrWhiteSpace(paraText))
                        {
                            sb.AppendLine(paraText);
                        }
                        break;

                    // Bảng trong Word — đọc từng hàng, ghép ô bằng " | "
                    case WordTable table:
                        sb.AppendLine("[Bảng]");
                        foreach (var row in table.Elements<WordTableRow>())
                        {
                            var cells = row.Elements<WordTableCell>()
                                .Select(cell => string.Concat(cell.Descendants<WordText>().Select(t => t.Text)).Trim())
                                .Where(text => !string.IsNullOrWhiteSpace(text));
                            var rowText = string.Join(" | ", cells);
                            if (!string.IsNullOrWhiteSpace(rowText))
                            {
                                sb.AppendLine(rowText);
                            }
                        }
                        sb.AppendLine();
                        break;
                }
            }
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Đọc file Excel (.xlsx) bằng DocumentFormat.OpenXml.
    ///
    /// Cấu trúc .xlsx:
    /// - Cũng là file ZIP chứa XML.
    /// - Mỗi sheet là một worksheet riêng trong workbook.
    ///
    /// Cách hoạt động:
    /// 1. SpreadsheetDocument.Open — mở workbook.
    /// 2. WorkbookPart.Workbook.Sheets — liệt kê các sheet.
    /// 3. Với mỗi sheet: đọc SheetData → Row → Cell.
    /// 4. Cell có thể chứa số trực tiếp hoặc index vào SharedStringTable (chuỗi dùng chung).
    ///
    /// Không hỗ trợ .xls (Excel 97-2003) — chỉ .xlsx.
    /// </summary>
    private static string ExtractXlsx(string path)
    {
        var sb = new StringBuilder();

        using (var document = SpreadsheetDocument.Open(path, isEditable: false))
        {
            var workbookPart = document.WorkbookPart;
            if (workbookPart?.Workbook?.Sheets is null)
            {
                return string.Empty;
            }

            // Bảng chuỗi dùng chung — Excel lưu text dài ở đây, cell chỉ giữ index
            var sharedStrings = workbookPart.SharedStringTablePart?.SharedStringTable;

            foreach (var sheet in workbookPart.Workbook.Sheets.Elements<Sheet>())
            {
                if (sheet.Id?.Value is null || sheet.Name?.Value is null)
                {
                    continue;
                }

                var worksheetPart = (WorksheetPart)workbookPart.GetPartById(sheet.Id.Value);
                var worksheet = worksheetPart.Worksheet;
                if (worksheet is null)
                {
                    continue;
                }

                var sheetData = worksheet.Elements<SheetData>().FirstOrDefault();
                if (sheetData is null)
                {
                    continue;
                }

                sb.AppendLine($"[Sheet: {sheet.Name!.Value}]");

                foreach (var row in sheetData.Elements<Row>())
                {
                    var cellValues = row.Elements<Cell>()
                        .Select(cell => GetExcelCellValue(cell, sharedStrings))
                        .Where(value => !string.IsNullOrWhiteSpace(value));

                    var rowText = string.Join(" | ", cellValues);
                    if (!string.IsNullOrWhiteSpace(rowText))
                    {
                        sb.AppendLine(rowText);
                    }
                }

                sb.AppendLine();
            }
        }

        return sb.ToString().Trim();
    }

    /// <summary>
    /// Lấy giá trị hiển thị của một ô Excel.
    ///
    /// - DataType = SharedString → tra SharedStringTable bằng index.
    /// - DataType = Boolean → "TRUE"/"FALSE".
    /// - Còn lại → lấy InnerText (số, công thức đã tính, ...).
    /// </summary>
    private static string GetExcelCellValue(Cell cell, SharedStringTable? sharedStrings)
    {
        if (cell.CellValue is null)
        {
            return string.Empty;
        }

        var rawValue = cell.CellValue.InnerText;

        if (cell.DataType?.Value == CellValues.SharedString && sharedStrings is not null)
        {
            if (int.TryParse(rawValue, out var index) && index >= 0 && index < sharedStrings.Count())
            {
                return sharedStrings.ElementAt(index).InnerText?.Trim() ?? string.Empty;
            }
        }

        if (cell.DataType?.Value == CellValues.Boolean)
        {
            return rawValue == "1" ? "TRUE" : "FALSE";
        }

        return rawValue.Trim();
    }

    private static string StripHtmlTags(string html) =>
        Regex.Replace(html, "<[^>]+>", " ", RegexOptions.Singleline);
}
