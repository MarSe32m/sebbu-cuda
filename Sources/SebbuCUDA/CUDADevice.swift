// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

extension CUDA {
    /// A small, copyable identifier for a CUDA device.
    public struct Device: Sendable, Hashable {
        // CUdevice is an integer typedef. Keeping the primitive representation
        // package-visible avoids leaking CCUDA through the Swift module API.
        package let rawValue: Int32

        /// Resolves a device by its zero-based ordinal.
        public init(ordinal: Int) throws {
            guard ordinal >= 0, let ordinal = Int32(exactly: ordinal) else {
                throw CUDA.ValidationError.invalidDeviceOrdinal(ordinal)
            }

            try cudaEnsureInitialized()
            var device: CUdevice = 0
            try cudaCheck(cuDeviceGet(&device, ordinal))
            rawValue = device
        }

        package init(rawValue: Int32) {
            self.rawValue = rawValue
        }

        /// The number of CUDA devices visible to the driver.
        public static var count: Int {
            get throws {
                try cudaEnsureInitialized()
                var count: Int32 = 0
                try cudaCheck(cuDeviceGetCount(&count))
                return Int(count)
            }
        }

        /// Every CUDA device currently visible to the driver.
        public static var all: [Device] {
            get throws {
                let deviceCount = try Self.count
                return try (0..<deviceCount).map { try Device(ordinal: $0) }
            }
        }

        /// The human-readable device name supplied by the driver.
        public var name: String {
            get throws {
                try cudaEnsureInitialized()
                var bytes = [CChar](repeating: 0, count: 256)
                try bytes.withUnsafeMutableBufferPointer { buffer in
                    try cudaCheck(
                        cuDeviceGetName(
                            buffer.baseAddress,
                            Int32(buffer.count),
                            rawValue
                        )
                    )
                }
                return bytes.withUnsafeBufferPointer {
                    String(cString: $0.baseAddress!)
                }
            }
        }

        /// Total global memory available on the device, in bytes.
        public var totalMemory: Int {
            get throws {
                try cudaEnsureInitialized()
                var byteCount = 0
                try cudaCheck(
                    sebbuCuDeviceTotalMem(&byteCount, rawValue)
                )
                return byteCount
            }
        }

        /// The device's CUDA compute capability.
        public var computeCapability: ComputeCapability {
            get throws {
                ComputeCapability(
                    major: try attribute(.computeCapabilityMajor),
                    minor: try attribute(.computeCapabilityMinor)
                )
            }
        }

        /// Queries a forward-compatible CUDA device attribute.
        public func attribute(_ attribute: Attribute) throws -> Int {
            try cudaEnsureInitialized()
            var value: Int32 = 0
            try cudaCheck(
                sebbuCuDeviceGetAttribute(
                    &value,
                    attribute.rawValue,
                    rawValue
                )
            )
            return Int(value)
        }
    }
}

extension CUDA.Device {
    public struct ComputeCapability: Sendable, Hashable,
        CustomStringConvertible {
        public let major: Int
        public let minor: Int

        @inlinable
        public init(major: Int, minor: Int) {
            self.major = major
            self.minor = minor
        }

        @inlinable
        public var description: String { "\(major).\(minor)" }
    }

    /// A forward-compatible CUDA device attribute identifier.
    public struct Attribute: RawRepresentable, Sendable, Hashable {
        public let rawValue: Int32

        @inlinable
        public init(rawValue: Int32) {
            self.rawValue = rawValue
        }

        public static let maximumThreadsPerBlock = Self(rawValue: 1)
        public static let maximumBlockDimensionX = Self(rawValue: 2)
        public static let maximumBlockDimensionY = Self(rawValue: 3)
        public static let maximumBlockDimensionZ = Self(rawValue: 4)
        public static let maximumGridDimensionX = Self(rawValue: 5)
        public static let maximumGridDimensionY = Self(rawValue: 6)
        public static let maximumGridDimensionZ = Self(rawValue: 7)
        public static let maximumSharedMemoryPerBlock = Self(rawValue: 8)
        public static let totalConstantMemory = Self(rawValue: 9)
        public static let warpSize = Self(rawValue: 10)
        public static let maximumRegistersPerBlock = Self(rawValue: 12)
        public static let clockRate = Self(rawValue: 13)
        public static let multiprocessorCount = Self(rawValue: 16)
        public static let concurrentKernels = Self(rawValue: 31)
        public static let memoryClockRate = Self(rawValue: 36)
        public static let globalMemoryBusWidth = Self(rawValue: 37)
        public static let l2CacheSize = Self(rawValue: 38)
        public static let maximumThreadsPerMultiprocessor = Self(rawValue: 39)
        public static let unifiedAddressing = Self(rawValue: 41)
        public static let computeCapabilityMajor = Self(rawValue: 75)
        public static let computeCapabilityMinor = Self(rawValue: 76)
        public static let managedMemory = Self(rawValue: 83)
    }
}
