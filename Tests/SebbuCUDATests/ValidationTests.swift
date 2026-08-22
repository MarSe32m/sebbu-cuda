import XCTest
@testable import SebbuCUDA

final class ValidationTests: XCTestCase {
    func testCheckedByteCount() throws {
        XCTAssertEqual(
            try cudaCheckedByteCount(count: 7, stride: 8),
            56
        )
        XCTAssertEqual(
            try cudaCheckedByteCount(count: 0, stride: 8),
            0
        )
        XCTAssertThrowsError(
            try cudaCheckedByteCount(count: -1, stride: 8)
        )
        XCTAssertThrowsError(
            try cudaCheckedByteCount(count: Int.max, stride: 2)
        )
    }

    func testTypedDevicePointerArithmeticUsesStride() {
        let pointer = CUDA.DevicePointer<UInt64>(rawValue: 0x1_000)
        XCTAssertEqual(pointer.advanced(by: 3).rawValue, 0x1_018)
        XCTAssertEqual(pointer.advanced(by: -2).rawValue, 0x0ff0)
        XCTAssertEqual(
            pointer.advanced(byByteCount: 7).rawValue,
            0x1_007
        )
    }

    func testForwardCompatibleErrorCode() {
        let futureCode = CUDA.Error.Code(rawValue: 12_345)
        XCTAssertEqual(futureCode.rawValue, 12_345)
        XCTAssertEqual(CUDA.Error.Code.invalidValue.rawValue, 1)
    }

    func testNegativeDeviceOrdinalIsAValidationError() {
        XCTAssertThrowsError(try CUDA.Device(ordinal: -1)) { error in
            XCTAssertEqual(
                error as? CUDA.ValidationError,
                .invalidDeviceOrdinal(-1)
            )
        }
    }

    func testPinnedMemoryFlagsFormAnOptionSet() {
        let flags: CUDA.PinnedMemoryFlags = [.portable, .deviceMapped]
        XCTAssertTrue(flags.contains(.portable))
        XCTAssertTrue(flags.contains(.deviceMapped))
        XCTAssertFalse(flags.contains(.writeCombined))
        XCTAssertEqual(flags.rawValue, 0x03)
    }

    func testKernelArgumentAddressesAreStableAndAligned() {
        var arguments = CUDA.KernelArguments()
        arguments.append(Int32(-7))
        arguments.append(UInt64(0xfeed_face_dead_beef))
        arguments.append(Float(1.25))
        arguments.append(
            CUDA.DevicePointer<Float>(rawValue: 0x1234_5678)
        )

        XCTAssertEqual(arguments.storage.count, 4)
        arguments.storage.withUnsafeArgumentPointers { pointers in
            XCTAssertNotNil(pointers)
            XCTAssertEqual(pointers![0]!.load(as: Int32.self), -7)
            XCTAssertEqual(
                pointers![1]!.load(as: UInt64.self),
                0xfeed_face_dead_beef
            )
            XCTAssertEqual(pointers![2]!.load(as: Float.self), 1.25)
            XCTAssertEqual(
                pointers![3]!.load(as: UInt64.self),
                0x1234_5678
            )

            XCTAssertEqual(
                Int(bitPattern: pointers![0]!) % MemoryLayout<Int32>.alignment,
                0
            )
            XCTAssertEqual(
                Int(bitPattern: pointers![1]!) % MemoryLayout<UInt64>.alignment,
                0
            )
        }
    }
}
