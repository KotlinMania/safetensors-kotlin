#if canImport(Testing)
import Testing
import Safetensors

@Suite("Safetensors Swift Export Tests")
struct SafetensorsExportTests {
    @Test("Safetensors swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        let dtype = Dtype.F32
        #expect(dtype.bitsize() == 32)
        let span = ByteSpan(start: 0, stop: 24)
        #expect(span.start == 0)
        #expect(span.stop == 24)
    }
}
#elseif canImport(XCTest)
import XCTest
import Safetensors

final class SafetensorsExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        let dtype = Dtype.F32
        XCTAssertEqual(dtype.bitsize(), 32)
        let span = ByteSpan(start: 0, stop: 24)
        XCTAssertEqual(span.start, 0)
        XCTAssertEqual(span.stop, 24)
    }
}
#endif
