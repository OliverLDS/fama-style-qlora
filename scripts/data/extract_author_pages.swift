// Private, full-page native extraction with line geometry; no network or ML.
// Input [{paper_id,pdf}], output fresh JSONL. PDF coordinates are normalized.
import Foundation
import PDFKit
import CoreGraphics
import AppKit
import Vision

let args = CommandLine.arguments
if args.count != 3 { fatalError("Expected tasks JSON and fresh output JSONL") }
let tasks = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as! [[String:Any]]
if FileManager.default.fileExists(atPath: args[2]) { fatalError("Refusing to overwrite extraction") }
FileManager.default.createFile(atPath: args[2], contents: nil)
let output = try FileHandle(forWritingTo: URL(fileURLWithPath: args[2]))
for task in tasks {
    let name = task["paper_id"] as! String
    guard let doc = PDFDocument(url: URL(fileURLWithPath: task["pdf"] as! String)) else { fatalError("Unreadable PDF") }
    let numbers = task["pages"] as? [Int] ?? Array(1...doc.pageCount)
    for number in numbers {
        try autoreleasepool {
            let index = number - 1
            guard let page = doc.page(at:index) else { fatalError("Missing page") }
            let bounds = page.bounds(for:.mediaBox)
            var lines = [[String:Any]]()
            if (task["method"] as? String) == "vision" {
                // PDFKit applies page rotation before Vision sees the image.
                let size = task["render_width"] as? Double ?? 2400
                let preview = page.thumbnail(of:NSSize(width:size,height:size*2),for:.mediaBox)
                guard let bitmap = NSBitmapImageRep(data:preview.tiffRepresentation!), let image = bitmap.cgImage else { fatalError("No rendered page") }
                let regions = task["regions"] as? [[String:Double]] ?? [["x":0,"width":1]]
                for (column,region) in regions.enumerated() {
                    let x = region["x"]!, width = region["width"]!
                    let crop = CGRect(x:x*Double(image.width),y:0,width:width*Double(image.width),height:Double(image.height))
                    guard let part = image.cropping(to:crop) else { fatalError("Invalid region") }
                    let request = VNRecognizeTextRequest()
                    request.recognitionLevel = .accurate
                    request.recognitionLanguages = ["en-US"]
                    request.usesLanguageCorrection = false
                    request.usesCPUOnly = true
                    try VNImageRequestHandler(cgImage:part,options:[:]).perform([request])
                    for observation in request.results ?? [] {
                        guard let candidate = observation.topCandidates(1).first else { continue }
                        let b = observation.boundingBox
                        lines.append(["line_id":lines.count+1,"column":column+1,"text":candidate.string,
                                      "x":x+b.minX*width,"y":b.minY,"width":b.width*width,"height":b.height,
                                      "confidence":candidate.confidence])
                    }
                }
                let record:[String:Any] = ["paper_id":name,"pdf_page":number,"total_pdf_pages":doc.pageCount,
                    "method":"macOS Vision accurate English; CPU only; language correction off; PDFKit rotation applied",
                    "render_width":image.width,"render_height":image.height,"regions":regions,
                    "page_width":bounds.width,"page_height":bounds.height,"rotation":page.rotation,"lines":lines]
                output.write(try JSONSerialization.data(withJSONObject:record,options:[.sortedKeys]));output.write(Data([10]))
                output.synchronizeFile()
                print("OCR \(name): page \(number)");fflush(stdout)
                return
            }
            if let selection = page.selection(for:bounds) {
                for line in selection.selectionsByLine() {
                    let text = line.string ?? ""
                    if text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty { continue }
                    let b = line.bounds(for:page)
                    lines.append(["line_id":lines.count+1,"text":text,
                                  "x":(b.minX-bounds.minX)/bounds.width,"y":(b.minY-bounds.minY)/bounds.height,
                                  "width":b.width/bounds.width,"height":b.height/bounds.height,
                                  "confidence":NSNull()])
                }
            }
            let record:[String:Any] = ["paper_id":name,"pdf_page":index+1,"total_pdf_pages":doc.pageCount,
                "method":"PDFKit native text/line bounds; confidence not measured","text":page.string ?? "",
                "page_width":bounds.width,"page_height":bounds.height,"rotation":page.rotation,"lines":lines]
            output.write(try JSONSerialization.data(withJSONObject:record,options:[.sortedKeys]));output.write(Data([10]))
        }
    }
    output.synchronizeFile()
    print("Completed \(name): \(numbers.count) pages");fflush(stdout)
}
try output.close()
