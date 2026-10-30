// Uniform, CPU-only macOS Vision OCR for both labels. No network or model download.
// Input: [{paper_id,pdf,pages:[1-based page numbers]}]. Output is private JSONL.
import Foundation
import PDFKit
import Vision
import CoreGraphics

let args = CommandLine.arguments
if args.count != 3 { fatalError("Expected tasks JSON and fresh output JSONL") }
let tasks = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as! [[String:Any]]
if FileManager.default.fileExists(atPath: args[2]) { fatalError("Refusing to overwrite OCR output") }
FileManager.default.createFile(atPath: args[2], contents: nil)
let file = try FileHandle(forWritingTo: URL(fileURLWithPath: args[2]))
for task in tasks {
    let name = task["paper_id"] as! String
    guard let doc = PDFDocument(url: URL(fileURLWithPath: task["pdf"] as! String)) else { fatalError("Unreadable PDF") }
    let numbers = task["pages"] as! [Int]
    for number in numbers {
        try autoreleasepool {
            guard number > 0, number <= doc.pageCount, let page = doc.page(at: number - 1) else { fatalError("Invalid page") }
            let bounds = page.bounds(for: .mediaBox)
            let scale = 1800.0 / bounds.width
            let width = Int(bounds.width * scale), height = Int(bounds.height * scale)
            guard let context = CGContext(data:nil,width:width,height:height,bitsPerComponent:8,bytesPerRow:0,
                                          space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.noneSkipLast.rawValue) else { fatalError("No render context") }
            context.setFillColor(CGColor(gray:1,alpha:1));context.fill(CGRect(x:0,y:0,width:width,height:height))
            context.scaleBy(x:scale,y:scale);context.translateBy(x:-bounds.minX,y:-bounds.minY)
            page.draw(with:.mediaBox,to:context)
            guard let image = context.makeImage() else { fatalError("No rendered page") }
            let columns = task["columns"] as? Int ?? 1
            var lines = [[String:Any]]()
            var revision = 0
            for column in 0..<columns {
                let overlap = column > 0 ? (task["column_overlap"] as? Double ?? 0.0) : 0.0
                let x0 = Double(column) / Double(columns) - overlap
                let fraction = 1.0 / Double(columns) + overlap
                let crop = CGRect(x:Double(width)*x0,y:0,width:Double(width)*fraction,height:Double(height))
                guard let region = image.cropping(to:crop) else { fatalError("No column image") }
                let request = VNRecognizeTextRequest()
                request.recognitionLevel = .accurate
                request.recognitionLanguages = ["en-US"]
                request.usesLanguageCorrection = false
                request.usesCPUOnly = true
                try VNImageRequestHandler(cgImage:region,options:[:]).perform([request])
                revision = request.revision
                for observation in request.results ?? [] {
                    guard let text = observation.topCandidates(1).first else { continue }
                    let b = observation.boundingBox
                    lines.append(["line_id":lines.count+1,"column":column+1,"text":text.string,"confidence":text.confidence,
                                  "x":x0+b.minX*fraction,"y":b.minY,"width":b.width*fraction,"height":b.height])
                }
            }
            let record:[String:Any] = ["paper_id":name,"pdf_page":number,"total_pdf_pages":doc.pageCount,
                "method":"macOS Vision accurate English; CPU only; language correction off; width 1800 pixels",
                "vision_revision":revision,"columns":columns,"column_overlap":task["column_overlap"] ?? 0.0,"render_width":width,"render_height":height,"lines":lines]
            let data = try JSONSerialization.data(withJSONObject:record,options:[.sortedKeys])
            file.write(data);file.write(Data([10]));file.synchronizeFile()
            print("OCR \(name): page \(number) (\(lines.count) lines)")
            fflush(stdout)
        }
    }
}
try file.close()
