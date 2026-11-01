// Local PDFKit first-page extraction; no network, credentials or model.
import Foundation
import PDFKit

// Highlighted PDF objects can appear out of order in page.string. Retain the raw
// extraction and separately order glyphs by their visual line and x coordinate.
func readingOrder(_ page: PDFPage) -> String {
    let source = (page.string ?? "") as NSString
    var glyphs: [(String, CGRect, Int)] = []
    for i in 0..<min(source.length, page.numberOfCharacters) {
        let value = source.substring(with: NSRange(location: i, length: 1))
        let box = page.characterBounds(at: i)
        if value.rangeOfCharacter(from: .newlines) != nil || box.isEmpty || box.isNull { continue }
        glyphs.append((value, box, i))
    }
    glyphs.sort { $0.1.midY > $1.1.midY }
    var lines: [[(String, CGRect, Int)]] = []
    for glyph in glyphs {
        if let last = lines.last, let anchor = last.first,
           abs(anchor.1.midY - glyph.1.midY) <= 2.0 {
            lines[lines.count - 1].append(glyph)
        } else { lines.append([glyph]) }
    }
    return lines.map { line in
        let ordered = line.sorted { $0.1.minX == $1.1.minX ? $0.2 < $1.2 : $0.1.minX < $1.1.minX }
        return ordered.map { $0.0 }.joined()
    }.joined(separator: "\n")
}

guard CommandLine.arguments.count == 3 || (CommandLine.arguments.count == 4 && CommandLine.arguments[3] == "--all-pages") else {
    fputs("Usage: extract_detector_pdf_page.swift INPUT.pdf OUTPUT.json\n", stderr)
    exit(2)
}
guard let document = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])),
      let page = document.page(at: 0) else {
    fputs("Unreadable PDF first page\n", stderr)
    exit(1)
}
var result: [String: Any] = ["page_count": document.pageCount,
                           "page_extracted": 1, "text": page.string ?? ""]
if CommandLine.arguments.count == 4 {
    result["pages"] = (0..<document.pageCount).map { document.page(at: $0)?.string ?? "" }
    result["layout_pages"] = (0..<document.pageCount).map { document.page(at: $0).map(readingOrder) ?? "" }
}
do {
    let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
} catch {
    fputs("Cannot save extracted first page\n", stderr)
    exit(1)
}
