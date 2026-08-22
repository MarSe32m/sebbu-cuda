// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import Foundation
import XCTest
@testable import SebbuCUDA

final class CUDAIntegrationTests: XCTestCase {
    func testRepeatedInitializationAndDeviceProperties() throws {
        let device = try cudaDeviceOrSkip()
        try CUDA.initialize()
        try CUDA.initialize()

        XCTAssertFalse(try device.name.isEmpty)
        XCTAssertGreaterThan(try device.totalMemory, 0)
        XCTAssertGreaterThanOrEqual(try device.computeCapability.major, 1)
    }

    func testPrimaryContextRetentionPath() throws {
        let context = try cudaDeviceOrSkip().retainPrimaryContext()
        try context.withCurrent {
            let stream = try context.makeStream()
            try stream.synchronize()
        }
    }

    func testZeroSizedAllocationsAndRoundTrip() throws {
        let context = try cudaDeviceOrSkip().makeContext()
        try context.withCurrent {
            let empty: CUDA.DeviceBuffer<Float> = try context.allocate(count: 0)
            let emptyRaw = try context.allocate(byteCount: 0)
            XCTAssertEqual(empty.count, 0)
            XCTAssertEqual(empty.byteCount, 0)
            XCTAssertEqual(empty.pointer.rawValue, 0)
            XCTAssertEqual(emptyRaw.byteCount, 0)
            XCTAssertEqual(emptyRaw.pointer.rawValue, 0)

            XCTAssertThrowsError(
                try { let _ = try context.allocate(UInt8.self, count: -1) }()
            )
            XCTAssertThrowsError(try { let _ = try context.allocate(byteCount: -1) }())
            XCTAssertThrowsError(
                try { let _ = try context.allocate(UInt64.self, count: Int.max) }()
            )

            let values = (0..<257).map { UInt64($0) &* 17 }
            let buffer: CUDA.DeviceBuffer<UInt64> = try context.allocate(
                count: values.count
            )
            try buffer.copy(from: values)
            var result = [UInt64](repeating: 0, count: values.count)
            try buffer.copy(to: &result)
            XCTAssertEqual(result, values)
        }
    }

    func testChildrenRetainContext() throws {
        var context: CUDA.Context? = try cudaDeviceOrSkip().makeContext()
        weak let weakContext = context

        do {
            let buffer: CUDA.DeviceBuffer<UInt8> = try context!.allocate(count: 1)
            context = nil
            XCTAssertNotNil(weakContext)
            XCTAssertEqual(buffer.byteCount, 1)
        }

        XCTAssertNil(weakContext)
    }

    func testStreamRetainsContext() throws {
        var context: CUDA.Context? = try cudaDeviceOrSkip().makeContext()
        weak let weakContext = context

        do {
            let stream = try context!.makeStream()
            context = nil
            XCTAssertNotNil(weakContext)
            try stream.synchronize()
        }

        XCTAssertNil(weakContext)
    }

    func testKernelRetainsModule() throws {
        let context = try cudaDeviceOrSkip().makeContext()
        weak var weakModule: CUDA.Module?
        var kernel: CUDA.Kernel?

        do {
            let module = try context.loadModule(ptx: testSAXPYPTX)
            weakModule = module
            kernel = try module.kernel(named: "saxpy")
        }

        withExtendedLifetime(kernel) {
            XCTAssertNotNil(weakModule)
        }
        kernel = nil
        XCTAssertNil(weakModule)
    }

    func testLoadModuleFromPTXFile() throws {
        let context = try cudaDeviceOrSkip().makeContext()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "sebbu-cuda-\(UUID().uuidString)",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false
        )
        defer {
            try? FileManager.default.removeItem(at: directory)
        }

        let ptxFile = directory.appendingPathComponent("saxpy.ptx")
        try testSAXPYPTX.write(to: ptxFile, atomically: true, encoding: .utf8)

        let module = try context.loadModule(atPath: ptxFile.path)
        _ = try module.kernel(named: "saxpy")
    }

    func testPinnedCopyOwnerIsRetainedUntilSynchronization() throws {
        let context = try cudaDeviceOrSkip().makeContext()
        try context.withCurrent {
            let stream = try context.makeStream(flags: [.nonBlocking])
            let device: CUDA.DeviceBuffer<UInt32> = try context.allocate(count: 64)
            weak var weakPinnedStorage: CUDAPinnedAllocationStorage?

            do {
                var pinned = try context.allocatePinned(UInt32.self, count: 64)
                pinned.withUnsafeMutableBufferPointer { buffer in
                    buffer.initialize(repeating: 42)
                }
                weakPinnedStorage = pinned.storage
                try stream.copy(from: pinned, to: device)
            }

            XCTAssertNotNil(weakPinnedStorage)
            try stream.synchronize()
            XCTAssertNil(weakPinnedStorage)

            let destination = try context.allocatePinned(UInt32.self, count: 64)
            try stream.copy(from: device, to: destination)
            try stream.synchronize()
            destination.withUnsafeBufferPointer { buffer in
                XCTAssertEqual(Array(buffer), [UInt32](repeating: 42, count: 64))
            }
        }
    }

    func testSAXPYLaunchAndEventTiming() throws {
        let context = try cudaDeviceOrSkip().makeContext()
        try context.withCurrent {
            let stream = try context.makeStream(flags: [.nonBlocking])
            let start = try context.makeEvent()
            let end = try context.makeEvent()
            let module = try context.loadModule(ptx: testSAXPYPTX)
            let kernel = try module.kernel(named: "saxpy")

            let count = 1_003
            let input = (0..<count).map { Float($0) * 0.125 }
            let initial = (0..<count).map { Float($0) * -0.25 }
            var result = initial
            let scale: Float = 3.5

            let x: CUDA.DeviceBuffer<Float> = try context.allocate(count: count)
            let y: CUDA.DeviceBuffer<Float> = try context.allocate(count: count)
            try x.copy(from: input)
            try y.copy(from: initial)

            try start.record(on: stream)
            try kernel.launch(
                grid: .init(x: UInt32((count + 255) / 256)),
                block: .init(x: 256),
                on: stream
            ) { arguments in
                arguments.append(x.pointer)
                arguments.append(y.pointer)
                arguments.append(scale)
                arguments.append(Int32(count))
            }
            try end.record(on: stream)
            try stream.synchronize()
            try y.copy(to: &result)

            for index in result.indices {
                XCTAssertEqual(
                    result[index],
                    scale * input[index] + initial[index],
                    accuracy: 0.0001
                )
            }
            XCTAssertGreaterThanOrEqual(
                try end.elapsedTime(since: start),
                .zero
            )
        }
    }
}
