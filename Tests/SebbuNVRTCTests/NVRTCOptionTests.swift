// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import SebbuNVRTC

final class NVRTCOptionTests: XCTestCase {
    func testCommonOptionsHaveDeterministicArguments() {
        let options = NVRTCCompileOptions(
            architecture: .sm100,
            cppStandard: .cxx20,
            fastMath: true,
            lineInfo: true,
            deviceDebug: true,
            includePaths: ["/first", "/second"],
            defines: ["ZETA": nil, "ALPHA": "3"],
            additionalOptions: ["--restrict"]
        )

        XCTAssertEqual(
            options.compileOptions.map(\.commandLineArgument),
            [
                "--gpu-architecture=sm_100",
                "--std=c++20",
                "--use_fast_math",
                "--generate-line-info",
                "--device-debug",
                "--include-path=/first",
                "--include-path=/second",
                "--define-macro=ALPHA=3",
                "--define-macro=ZETA",
                "--restrict",
            ]
        )
    }

    func testTypedOptionRendering() {
        let options: [NVRTC.CompileOption] = [
            .relocatableDeviceCode(true),
            .fastCompile(.middle),
            .maximumRegisterCount(64),
            .flushToZero(false),
            .preciseSquareRoot(true),
            .deviceLinkTimeOptimization,
            .includePath("include"),
            .warningsAsErrors([.reorder, .deprecatedDeclarations]),
            .diagnosticsAsErrors([123, 456]),
            .splitCompile(threadCount: 0),
            .timeTrace("trace.json"),
            .raw("--future-option"),
        ]

        XCTAssertEqual(
            options.map(\.commandLineArgument),
            [
                "--relocatable-device-code=true",
                "--Ofast-compile=mid",
                "--maxrregcount=64",
                "--ftz=false",
                "--prec-sqrt=true",
                "--dlink-time-opt",
                "--include-path=include",
                "--warning-as-error=reorder,deprecated-declarations",
                "--diag-error=123,456",
                "--split-compile=0",
                "--fdevice-time-trace=trace.json",
                "--future-option",
            ]
        )
    }

    func testArchitectureFactoriesAreForwardCompatible() {
        XCTAssertEqual(CUDAArchitecture.sm100.rawValue, "sm_100")
        XCTAssertEqual(
            CUDAArchitecture.compute(137, suffix: "a").rawValue,
            "compute_137a"
        )
        XCTAssertEqual(
            CUDAArchitecture(rawValue: "future_arch").rawValue,
            "future_arch"
        )
    }

    func testForwardCompatibleErrorCode() {
        XCTAssertEqual(NVRTC.Error.Code.compilation.rawValue, 6)
        XCTAssertEqual(NVRTC.Error.Code.busy.rawValue, 18)
        XCTAssertEqual(NVRTC.Error.Code(rawValue: 12_345).rawValue, 12_345)
    }
}
