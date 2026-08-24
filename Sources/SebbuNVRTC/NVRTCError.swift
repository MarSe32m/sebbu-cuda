// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CNVRTC

extension NVRTC {
    /// An error reported by the NVIDIA Runtime Compilation library.
    public struct Error: Swift.Error, Sendable, CustomStringConvertible {
        public let code: Code
        public let name: String
        public let message: String

        @inlinable
        public var description: String {
            "\(name) (\(code.rawValue)): \(message)"
        }

        internal init(
            _ result: Int32,
            detailedMessage: String? = nil
        ) {
            code = Code(rawValue: result)
            if let pointer = sebbuNvrtcGetErrorString(result) {
                name = String(cString: pointer)
            } else {
                name = "NVRTC_ERROR_\(result)"
            }
            message = detailedMessage ?? "NVRTC reported \(name)."
        }
    }

    /// A failure while compiling a program, including NVRTC's diagnostic log.
    public struct CompilationError: Swift.Error, Sendable,
        CustomStringConvertible {
        public let error: Error
        public let log: String

        @inlinable
        public var description: String {
            log.isEmpty ? error.description : "\(error.description)\n\(log)"
        }
    }

    /// An API that is absent from the CUDA Toolkit used to build the package.
    public struct UnsupportedFeatureError: Swift.Error, Sendable,
        Equatable, CustomStringConvertible {
        public let feature: String

        @inlinable
        public init(feature: String) {
            self.feature = feature
        }

        @inlinable
        public var description: String {
            "\(feature) is unavailable in the NVRTC version used to build this package."
        }
    }

    /// A validation failure detected before entering NVRTC.
    public enum ValidationError: Swift.Error, Sendable, Equatable,
        CustomStringConvertible {
        case embeddedNull(field: String)
        case integerConversionOverflow(value: Int, target: String)
        case negativeByteCount(Int)

        public var description: String {
            switch self {
            case .embeddedNull(let field):
                "\(field) must not contain an embedded null byte."
            case .integerConversionOverflow(let value, let target):
                "Value \(value) is not representable as \(target)."
            case .negativeByteCount(let count):
                "Byte count must be non-negative; received \(count)."
            }
        }
    }
}

extension NVRTC.Error {
    /// A forward-compatible NVRTC result code.
    public struct Code: RawRepresentable, Sendable, Hashable,
        CustomStringConvertible {
        public let rawValue: Int32

        @inlinable
        public init(rawValue: Int32) {
            self.rawValue = rawValue
        }

        public var description: String {
            guard let pointer = sebbuNvrtcGetErrorString(rawValue) else {
                return "NVRTC_ERROR_\(rawValue)"
            }
            return String(cString: pointer)
        }

        public static let success = Self(rawValue: 0)
        public static let outOfMemory = Self(rawValue: 1)
        public static let programCreationFailure = Self(rawValue: 2)
        public static let invalidInput = Self(rawValue: 3)
        public static let invalidProgram = Self(rawValue: 4)
        public static let invalidOption = Self(rawValue: 5)
        public static let compilation = Self(rawValue: 6)
        public static let builtinOperationFailure = Self(rawValue: 7)
        public static let noNameExpressionsAfterCompilation = Self(rawValue: 8)
        public static let noLoweredNamesBeforeCompilation = Self(rawValue: 9)
        public static let invalidNameExpression = Self(rawValue: 10)
        public static let internalError = Self(rawValue: 11)
        public static let timeFileWriteFailed = Self(rawValue: 12)
        public static let noPCHCreationAttempted = Self(rawValue: 13)
        public static let pchCreateHeapExhausted = Self(rawValue: 14)
        public static let pchCreate = Self(rawValue: 15)
        public static let cancelled = Self(rawValue: 16)
        public static let timeTraceFileWriteFailed = Self(rawValue: 17)
        public static let busy = Self(rawValue: 18)
    }
}
