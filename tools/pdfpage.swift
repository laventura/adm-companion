// Render one page of a PDF to a PNG, using macOS's built-in PDFKit.
// Usage: swift tools/pdfpage.swift <pdf> <page index, 0-based> <out.png> [scale]
import AppKit
import PDFKit

let args = CommandLine.arguments
guard args.count >= 4, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])),
      let index = Int(args[2]), let page = doc.page(at: index) else {
    FileHandle.standardError.write("usage: pdfpage.swift <pdf> <page index> <out.png> [scale]\n".data(using: .utf8)!)
    exit(1)
}
let scale = args.count > 4 ? CGFloat(Double(args[4]) ?? 2) : 2
let box = page.bounds(for: .mediaBox)
let image = page.thumbnail(of: NSSize(width: box.width * scale, height: box.height * scale), for: .mediaBox)
guard let tiff = image.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else { exit(1) }
try png.write(to: URL(fileURLWithPath: args[3]))
