Thư mục lưu tài liệu nội bộ upload qua API POST /api/ai/documents/upload.

Định dạng hỗ trợ:
- PDF (.pdf)     — UglyToad.PdfPig, đọc text từng trang
- Word (.docx)   — DocumentFormat.OpenXml, đọc đoạn văn + bảng
- Excel (.xlsx)  — DocumentFormat.OpenXml, đọc từng sheet/dòng
- Text (.txt, .md, .csv, .json, ...)

Không hỗ trợ: .doc (Word cũ), .xls (Excel cũ), PDF scan ảnh (cần OCR).

File gốc và chunks được lưu tại đây — không commit lên git.
