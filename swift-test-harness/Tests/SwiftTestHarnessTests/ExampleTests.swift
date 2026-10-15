#if canImport(Testing)
import Testing
import Safetensors

@Suite("Safetensors Swift Export Tests")
struct SafetensorsExportTests {
    @Test("Safetensors swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "Safetensors swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import Safetensors

final class SafetensorsExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "Safetensors swift module imported cleanly")
    }
}
#endif
