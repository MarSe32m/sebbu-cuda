// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

extension CUDA {
    /// An error reported by the CUDA Driver API.
    public struct Error: Swift.Error, Sendable, CustomStringConvertible {
        public let code: Code
        public let name: String
        public let message: String

        @inlinable
        public var description: String {
            "\(name) (\(code.rawValue)): \(message)"
        }

        internal init(_ result: CUresult) {
            let rawCode = Int32(sebbuCuResultRawValue(result))
            code = Code(rawValue: rawCode)

            var namePointer: UnsafePointer<CChar>?
            if cuGetErrorName(result, &namePointer) == CUDA_SUCCESS,
               let namePointer {
                name = String(cString: namePointer)
            } else {
                name = "CUDA_ERROR_\(rawCode)"
            }

            var messagePointer: UnsafePointer<CChar>?
            if cuGetErrorString(result, &messagePointer) == CUDA_SUCCESS,
               let messagePointer {
                message = String(cString: messagePointer)
            } else {
                message = "No CUDA error description is available."
            }
        }
    }

    /// A validation failure detected before entering the CUDA Driver API.
    public enum ValidationError: Swift.Error, Sendable, Equatable,
        CustomStringConvertible {
        case negativeCount(Int)
        case sizeOverflow(count: Int, stride: Int)
        case bufferTooSmall(required: Int, available: Int)
        case contextMismatch
        case invalidDeviceOrdinal(Int)
        case integerConversionOverflow(value: Int, target: String)
        case missingHandle(resource: String)
        case emptyModuleImage
        case negativeSharedMemoryByteCount(Int)
        case zeroLaunchDimension

        public var description: String {
            switch self {
            case .negativeCount(let count):
                "Count must be non-negative; received \(count)."
            case .sizeOverflow(let count, let stride):
                "Byte count overflow for count \(count) and stride \(stride)."
            case .bufferTooSmall(let required, let available):
                "Buffer requires \(required) elements but has \(available)."
            case .contextMismatch:
                "CUDA resources belong to different contexts."
            case .invalidDeviceOrdinal(let ordinal):
                "CUDA device ordinal \(ordinal) is not representable."
            case .integerConversionOverflow(let value, let target):
                "Value \(value) is not representable as \(target)."
            case .missingHandle(let resource):
                "CUDA returned success without a \(resource) handle."
            case .emptyModuleImage:
                "A CUDA module image must not be empty."
            case .negativeSharedMemoryByteCount(let count):
                "Shared-memory byte count must be non-negative; received \(count)."
            case .zeroLaunchDimension:
                "CUDA grid and block dimensions must all be greater than zero."
            }
        }
    }
}

extension CUDA.Error {
    /// A forward-compatible CUDA error code.
    public struct Code: RawRepresentable, Sendable, Hashable {
        public let rawValue: Int32

        @inlinable
        public init(rawValue: Int32) {
            self.rawValue = rawValue
        }

        public static let invalidValue = Self(rawValue: 1)
        public static let outOfMemory = Self(rawValue: 2)
        public static let notInitialized = Self(rawValue: 3)
        public static let deinitialized = Self(rawValue: 4)
        public static let stubLibrary = Self(rawValue: 34)
        public static let noDevice = Self(rawValue: 100)
        public static let invalidDevice = Self(rawValue: 101)
        public static let invalidImage = Self(rawValue: 200)
        public static let invalidContext = Self(rawValue: 201)
        public static let invalidPTX = Self(rawValue: 218)
        public static let invalidSource = Self(rawValue: 300)
        public static let fileNotFound = Self(rawValue: 301)
        public static let invalidHandle = Self(rawValue: 400)
        public static let notFound = Self(rawValue: 500)
        public static let notReady = Self(rawValue: 600)
        public static let illegalAddress = Self(rawValue: 700)
        public static let launchOutOfResources = Self(rawValue: 701)
        public static let launchTimeout = Self(rawValue: 702)
        public static let launchFailed = Self(rawValue: 719)
        public static let notPermitted = Self(rawValue: 800)
        public static let notSupported = Self(rawValue: 801)
        public static let unknown = Self(rawValue: 999)
    }
}
