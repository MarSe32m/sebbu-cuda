// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import SebbuNVRTC

private let simpleKernel = #"""
extern "C" __global__ void scale(float *values, float factor) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    values[index] *= factor;
}
"""#

final class NVRTCIntegrationTests: XCTestCase {
    func testVersionAndSupportedArchitectures() throws {
        let version = try NVRTC.version
        XCTAssertGreaterThanOrEqual(version.major, 7)
        XCTAssertFalse(version.description.isEmpty)

        if NVRTC.features.contains(.supportedArchitectures) {
            let architectures = try NVRTC.supportedComputeArchitectures
            XCTAssertFalse(architectures.isEmpty)
            XCTAssertTrue(
                architectures.allSatisfy {
                    $0.rawValue.hasPrefix("compute_")
                }
            )
        }
    }

    func testOneShotPTXCompilation() throws {
        let ptx = try NVRTC.compile(
            simpleKernel,
            name: "scale.cu",
            architecture: .sm100,
            options: [.cxxStandard(.cxx17), .lineInfo]
        )
        XCTAssertTrue(ptx.contains(".entry scale"))
    }

    func testStructuredOptionsCompilation() throws {
        let options = NVRTCCompileOptions(
            cppStandard: .cxx17,
            defines: ["FACTOR": "4"]
        )
        let ptx = try NVRTC.compile(
            #"""
            #ifndef FACTOR
            #error FACTOR was not defined
            #endif
            extern "C" __global__ void configured(int *value) {
                *value = FACTOR;
            }
            """#,
            options: options
        )
        XCTAssertTrue(ptx.contains(".entry configured"))
    }

    func testInMemoryHeaderAndProgramLog() throws {
        let program = try NVRTC.Program(
            source: #"""
            #include "configuration.h"
            extern "C" __global__ void configured(int *value) {
                *value = CONFIGURED_VALUE;
            }
            """#,
            name: "configured.cu",
            headers: [
                .init(
                    source: "#define CONFIGURED_VALUE 17\n",
                    includeName: "configuration.h"
                )
            ]
        )
        try program.compile()
        XCTAssertTrue(try program.ptx().contains(".entry configured"))
        _ = try program.log()
    }

    func testLoweredName() throws {
        let program = try NVRTC.Program(
            source: #"""
            template<typename T>
            __global__ void templated(T *value) { *value += T(1); }
            """#,
            name: "templated.cu"
        )
        try program.addNameExpression("templated<int>")
        try program.compile(options: [.cxxStandard(.cxx17)])

        let name = try program.loweredName(for: "templated<int>")
        XCTAssertFalse(name.isEmpty)
        XCTAssertNotEqual(name, "templated<int>")
    }

    func testCompilationErrorIncludesLog() throws {
        let program = try NVRTC.Program(
            source: "extern \"C\" __global__ void broken( {",
            name: "broken.cu"
        )
        XCTAssertThrowsError(try program.compile()) { error in
            guard let error = error as? NVRTC.CompilationError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(error.error.code, .compilation)
            XCTAssertFalse(error.log.isEmpty)
        }
    }

    func testEmbeddedNullValidation() {
        XCTAssertThrowsError(
            try NVRTC.Program(source: "kernel\0truncated")
        ) { error in
            XCTAssertEqual(
                error as? NVRTC.ValidationError,
                .embeddedNull(field: "source")
            )
        }
    }

    func testCUBINOutputWhenSupported() throws {
        guard NVRTC.features.contains(.cubin),
              let compute = try NVRTC.supportedComputeArchitectures.last else {
            throw XCTSkip("This NVRTC version does not expose cubin output.")
        }
        let architectureNumber = compute.rawValue.dropFirst("compute_".count)
        let architecture = NVRTC.Architecture(
            rawValue: "sm_\(architectureNumber)"
        )
        let program = try NVRTC.Program(source: simpleKernel)
        try program.compile(options: [.architecture(architecture)])
        XCTAssertFalse(try program.cubin().isEmpty)
    }

    func testLTOIROutputWhenSupported() throws {
        guard NVRTC.features.contains(.ltoIR) else {
            throw XCTSkip("This NVRTC version does not expose LTO IR output.")
        }
        let program = try NVRTC.Program(source: simpleKernel)
        try program.compile(options: [.deviceLinkTimeOptimization])
        XCTAssertFalse(try program.ltoIR().isEmpty)
    }

    func testPCHAndFlowCallbackAPIsWhenSupported() throws {
        let program = try NVRTC.Program(source: simpleKernel)

        if NVRTC.features.contains(.flowCallbacks) {
            try program.setCancellationHandler { false }
        }

        try program.compile()
        if NVRTC.features.contains(.precompiledHeaders) {
            XCTAssertEqual(try program.pchCreationStatus(), .notAttempted)
            _ = try NVRTC.pchHeapSize
        }
    }

    func testBundledHeadersInformationWhenSupported() throws {
        guard NVRTC.features.contains(.bundledHeaders) else {
            throw XCTSkip("This NVRTC version does not bundle installable headers.")
        }
        let info = try NVRTC.bundledHeadersInfo
        XCTAssertTrue(info.areAvailable)
        XCTAssertGreaterThan(info.compressedByteCount, 0)
        XCTAssertGreaterThan(info.uncompressedByteCount, 0)
        XCTAssertGreaterThan(info.fileCount, 0)
    }
}
