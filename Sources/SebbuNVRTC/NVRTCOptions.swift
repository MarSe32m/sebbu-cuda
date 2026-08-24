// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

extension NVRTC {
    /// A forward-compatible CUDA virtual or real GPU architecture name.
    public struct Architecture: RawRepresentable, Sendable, Hashable,
        CustomStringConvertible {
        public let rawValue: String

        @inlinable
        public init(rawValue: String) {
            self.rawValue = rawValue
        }

        @inlinable
        public static func compute(
            _ value: Int,
            suffix: String = ""
        ) -> Self {
            Self(rawValue: "compute_\(value)\(suffix)")
        }

        @inlinable
        public static func sm(
            _ value: Int,
            suffix: String = ""
        ) -> Self {
            Self(rawValue: "sm_\(value)\(suffix)")
        }

        @inlinable
        public var description: String { rawValue }

        public static let compute75 = compute(75)
        public static let compute80 = compute(80)
        public static let compute86 = compute(86)
        public static let compute87 = compute(87)
        public static let compute89 = compute(89)
        public static let compute90 = compute(90)
        public static let compute90a = compute(90, suffix: "a")
        public static let compute100 = compute(100)
        public static let compute100f = compute(100, suffix: "f")
        public static let compute100a = compute(100, suffix: "a")
        public static let compute101 = compute(101)
        public static let compute101f = compute(101, suffix: "f")
        public static let compute101a = compute(101, suffix: "a")
        public static let compute103 = compute(103)
        public static let compute103f = compute(103, suffix: "f")
        public static let compute103a = compute(103, suffix: "a")
        public static let compute120 = compute(120)
        public static let compute120f = compute(120, suffix: "f")
        public static let compute120a = compute(120, suffix: "a")
        public static let compute121 = compute(121)
        public static let compute121f = compute(121, suffix: "f")
        public static let compute121a = compute(121, suffix: "a")

        public static let sm75 = sm(75)
        public static let sm80 = sm(80)
        public static let sm86 = sm(86)
        public static let sm87 = sm(87)
        public static let sm89 = sm(89)
        public static let sm90 = sm(90)
        public static let sm90a = sm(90, suffix: "a")
        public static let sm100 = sm(100)
        public static let sm100f = sm(100, suffix: "f")
        public static let sm100a = sm(100, suffix: "a")
        public static let sm101 = sm(101)
        public static let sm101f = sm(101, suffix: "f")
        public static let sm101a = sm(101, suffix: "a")
        public static let sm103 = sm(103)
        public static let sm103f = sm(103, suffix: "f")
        public static let sm103a = sm(103, suffix: "a")
        public static let sm120 = sm(120)
        public static let sm120f = sm(120, suffix: "f")
        public static let sm120a = sm(120, suffix: "a")
        public static let sm121 = sm(121)
        public static let sm121f = sm(121, suffix: "f")
        public static let sm121a = sm(121, suffix: "a")
    }

    public enum CXXStandard: String, Sendable, Hashable {
        case cxx03 = "c++03"
        case cxx11 = "c++11"
        case cxx14 = "c++14"
        case cxx17 = "c++17"
        case cxx20 = "c++20"
    }

    public enum FastCompileLevel: String, Sendable, Hashable {
        case disabled = "0"
        case minimum = "min"
        case middle = "mid"
        case maximum = "max"
    }

    public enum WarningKind: String, Sendable, Hashable {
        case all = "all-warnings"
        case reorder
        case deprecatedDeclarations = "deprecated-declarations"
    }

    /// A typed NVRTC command-line option.
    ///
    /// `raw` remains available for new toolkit options, so this API does not
    /// prevent clients from adopting NVRTC features before the wrapper updates.
    public enum CompileOption: Sendable, Hashable {
        case architecture(Architecture)
        case cxxStandard(CXXStandard)
        case fastMath
        case lineInfo
        case deviceDebug
        case relocatableDeviceCode(Bool)
        case extensibleWholeProgram
        case enableTile
        case tileOnly
        case simtOnly
        case defaultTile
        case diagnoseImplicitTileVariables
        case deviceOptimization
        case fastCompile(FastCompileLevel)
        case ptxAssemblerOptions(String)
        case maximumRegisterCount(Int)
        case flushToZero(Bool)
        case preciseSquareRoot(Bool)
        case preciseDivision(Bool)
        case fusedMultiplyAdd(Bool)
        case extraDeviceVectorization
        case modifyStackLimit(Bool)
        case deviceLinkTimeOptimization
        case optimizedLTOIR
        case optiXIR
        case jumpTableDensity(Int)
        case deviceStackProtector(Bool)
        case disableCache
        case randomSeed(String)
        case define(name: String, value: String?)
        case undefine(String)
        case includePath(String)
        case bundledHeaders(at: String)
        case preinclude(String)
        case noSourceInclude
        case builtinMoveForward(Bool)
        case builtinInitializerList(Bool)
        case automaticPCH
        case createPCH(String)
        case usePCH(String)
        case pchDirectory(String)
        case pchVerbose(Bool)
        case pchMessages(Bool)
        case instantiateTemplatesInPCH(Bool)
        case disableWarnings
        case warnOnReorder
        case warningsAsErrors([WarningKind])
        case restrictPointers
        case defaultDevice
        case deviceInt128
        case deviceFloat128
        case inlineOptimizationInfo
        case displayErrorNumbers(Bool)
        case diagnosticsAsErrors([Int])
        case suppressDiagnostics([Int])
        case diagnosticsAsWarnings([Int])
        case briefDiagnostics(Bool)
        case timingOutput(String)
        case splitCompile(threadCount: Int)
        case syntaxOnly
        case minimal
        case timeTrace(String)
        case raw(String)

        /// The argument passed as one string to `nvrtcCompileProgram`.
        public var commandLineArgument: String {
            switch self {
            case .architecture(let architecture):
                "--gpu-architecture=\(architecture.rawValue)"
            case .cxxStandard(let standard):
                "--std=\(standard.rawValue)"
            case .fastMath:
                "--use_fast_math"
            case .lineInfo:
                "--generate-line-info"
            case .deviceDebug:
                "--device-debug"
            case .relocatableDeviceCode(let enabled):
                "--relocatable-device-code=\(enabled)"
            case .extensibleWholeProgram:
                "--extensible-whole-program"
            case .enableTile:
                "--enable-tile"
            case .tileOnly:
                "--tile-only"
            case .simtOnly:
                "--simt-only"
            case .defaultTile:
                "--default-tile"
            case .diagnoseImplicitTileVariables:
                "--diagnose-implicit-tile-var"
            case .deviceOptimization:
                "--dopt=on"
            case .fastCompile(let level):
                "--Ofast-compile=\(level.rawValue)"
            case .ptxAssemblerOptions(let options):
                "--ptxas-options=\(options)"
            case .maximumRegisterCount(let count):
                "--maxrregcount=\(count)"
            case .flushToZero(let enabled):
                "--ftz=\(enabled)"
            case .preciseSquareRoot(let enabled):
                "--prec-sqrt=\(enabled)"
            case .preciseDivision(let enabled):
                "--prec-div=\(enabled)"
            case .fusedMultiplyAdd(let enabled):
                "--fmad=\(enabled)"
            case .extraDeviceVectorization:
                "--extra-device-vectorization"
            case .modifyStackLimit(let enabled):
                "--modify-stack-limit=\(enabled)"
            case .deviceLinkTimeOptimization:
                "--dlink-time-opt"
            case .optimizedLTOIR:
                "--gen-opt-lto"
            case .optiXIR:
                "--optix-ir"
            case .jumpTableDensity(let percentage):
                "--jump-table-density=\(percentage)"
            case .deviceStackProtector(let enabled):
                "--device-stack-protector=\(enabled)"
            case .disableCache:
                "--no-cache"
            case .randomSeed(let seed):
                "--frandom-seed=\(seed)"
            case .define(let name, let value):
                "--define-macro=\(name)\(value.map { "=\($0)" } ?? "")"
            case .undefine(let name):
                "--undefine-macro=\(name)"
            case .includePath(let path):
                "--include-path=\(path)"
            case .bundledHeaders(let path):
                "--use-bundled-headers=\(path)"
            case .preinclude(let header):
                "--pre-include=\(header)"
            case .noSourceInclude:
                "--no-source-include"
            case .builtinMoveForward(let enabled):
                "--builtin-move-forward=\(enabled)"
            case .builtinInitializerList(let enabled):
                "--builtin-initializer-list=\(enabled)"
            case .automaticPCH:
                "--pch"
            case .createPCH(let file):
                "--create-pch=\(file)"
            case .usePCH(let file):
                "--use-pch=\(file)"
            case .pchDirectory(let directory):
                "--pch-dir=\(directory)"
            case .pchVerbose(let enabled):
                "--pch-verbose=\(enabled)"
            case .pchMessages(let enabled):
                "--pch-messages=\(enabled)"
            case .instantiateTemplatesInPCH(let enabled):
                "--instantiate-templates-in-pch=\(enabled)"
            case .disableWarnings:
                "--disable-warnings"
            case .warnOnReorder:
                "--Wreorder"
            case .warningsAsErrors(let kinds):
                "--warning-as-error=\(kinds.map(\.rawValue).joined(separator: ","))"
            case .restrictPointers:
                "--restrict"
            case .defaultDevice:
                "--device-as-default-execution-space"
            case .deviceInt128:
                "--device-int128"
            case .deviceFloat128:
                "--device-float128"
            case .inlineOptimizationInfo:
                "--optimization-info=inline"
            case .displayErrorNumbers(let enabled):
                enabled ? "--display-error-number" : "--no-display-error-number"
            case .diagnosticsAsErrors(let numbers):
                "--diag-error=\(numbers.map(String.init).joined(separator: ","))"
            case .suppressDiagnostics(let numbers):
                "--diag-suppress=\(numbers.map(String.init).joined(separator: ","))"
            case .diagnosticsAsWarnings(let numbers):
                "--diag-warn=\(numbers.map(String.init).joined(separator: ","))"
            case .briefDiagnostics(let enabled):
                "--brief-diagnostics=\(enabled)"
            case .timingOutput(let file):
                "--time=\(file)"
            case .splitCompile(let threadCount):
                "--split-compile=\(threadCount)"
            case .syntaxOnly:
                "--fdevice-syntax-only"
            case .minimal:
                "--minimal"
            case .timeTrace(let file):
                "--fdevice-time-trace=\(file)"
            case .raw(let argument):
                argument
            }
        }
    }

    /// Convenient structured options for the common compilation path.
    public struct CompileOptions: Sendable, Hashable {
        public var architecture: Architecture?
        public var cppStandard: CXXStandard?
        public var fastMath: Bool
        public var lineInfo: Bool
        public var deviceDebug: Bool
        public var includePaths: [String]
        public var defines: [String: String?]
        public var additionalOptions: [String]

        @inlinable
        public init(
            architecture: Architecture? = nil,
            cppStandard: CXXStandard? = nil,
            fastMath: Bool = false,
            lineInfo: Bool = false,
            deviceDebug: Bool = false,
            includePaths: [String] = [],
            defines: [String: String?] = [:],
            additionalOptions: [String] = []
        ) {
            self.architecture = architecture
            self.cppStandard = cppStandard
            self.fastMath = fastMath
            self.lineInfo = lineInfo
            self.deviceDebug = deviceDebug
            self.includePaths = includePaths
            self.defines = defines
            self.additionalOptions = additionalOptions
        }

        /// Typed options in deterministic order. Macro names are sorted so
        /// identical dictionaries produce identical NVRTC invocations.
        public var compileOptions: [CompileOption] {
            var result: [CompileOption] = []
            if let architecture {
                result.append(.architecture(architecture))
            }
            if let cppStandard {
                result.append(.cxxStandard(cppStandard))
            }
            if fastMath { result.append(.fastMath) }
            if lineInfo { result.append(.lineInfo) }
            if deviceDebug { result.append(.deviceDebug) }
            result += includePaths.map(CompileOption.includePath)
            result += defines.sorted { $0.key < $1.key }.map {
                .define(name: $0.key, value: $0.value)
            }
            result += additionalOptions.map(CompileOption.raw)
            return result
        }
    }
}

public typealias CUDAArchitecture = NVRTC.Architecture
public typealias CXXStandard = NVRTC.CXXStandard
public typealias NVRTCCompileOption = NVRTC.CompileOption
public typealias NVRTCCompileOptions = NVRTC.CompileOptions
