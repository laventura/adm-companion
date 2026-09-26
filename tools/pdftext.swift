// Print the text of pages [start, end) of a PDF, using macOS's built-in PDFKit.
import Foundation
import PDFKit

let args = CommandLine.arguments
guard args.count == 4, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])),
      let start = Int(args[2]), let end = Int(args[3]) else {
    FileHandle.standardError.write("usage: pdftext.swift <pdf> <start> <end>\n".data(using: .utf8)!)
    exit(1)
}
print("pages: \(doc.pageCount)")
for i in max(start, 0)..<min(end, doc.pageCount) {
    print("=== page \(i + 1)")
    print(doc.page(at: i)?.string ?? "")
}
