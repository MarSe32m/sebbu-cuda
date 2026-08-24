// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CNVRTC

/// A namespace for the Swift NVIDIA Runtime Compilation wrapper.
public enum NVRTC {}

extension NVRTC {
    /// The version of the linked NVRTC implementation.
    public static var version: Version {
        get throws {
            var major: Int32 = 0
            var minor: Int32 = 0
            try nvrtcCheck(sebbuNvrtcVersion(&major, &minor))
            return Version(major: Int(major), minor: Int(minor))
        }
    }

    /// APIs present in the toolkit headers used to build this package.
    public static var features: Features {
        Features(rawValue: sebbuNvrtcFeatureMask())
    }

    /// Virtual compute architectures accepted by this NVRTC version.
    public static var supportedComputeArchitectures: [Architecture] {
        get throws {
            var count: Int32 = 0
            try nvrtcCheck(
                sebbuNvrtcGetNumSupportedArchs(&count),
                feature: "supported-architecture queries"
            )
            guard count > 0 else { return [] }

            var values = [Int32](repeating: 0, count: Int(count))
            try values.withUnsafeMutableBufferPointer {
                try nvrtcCheck(
                    sebbuNvrtcGetSupportedArchs($0.baseAddress),
                    feature: "supported-architecture queries"
                )
            }
            return values.map { .compute(Int($0)) }
        }
    }

    /// Compiles source to PTX using a short common-path API.
    public static func compile(
        _ source: String,
        name: String? = nil,
        architecture: Architecture? = nil,
        options: [CompileOption] = [],
        headers: [Header] = []
    ) throws -> String {
        let program = try Program(
            source: source,
            name: name,
            headers: headers
        )
        var compileOptions = options
        if let architecture {
            compileOptions.insert(.architecture(architecture), at: 0)
        }
        try program.compile(options: compileOptions)
        return try program.ptx()
    }

    /// Compiles source to PTX from structured options.
    public static func compile(
        _ source: String,
        name: String? = nil,
        options: CompileOptions,
        headers: [Header] = []
    ) throws -> String {
        let program = try Program(
            source: source,
            name: name,
            headers: headers
        )
        try program.compile(options: options)
        return try program.ptx()
    }

    public struct Version: Sendable, Hashable, CustomStringConvertible {
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

    public struct Features: OptionSet, Sendable, Hashable {
        public let rawValue: UInt64

        @inlinable
        public init(rawValue: UInt64) {
            self.rawValue = rawValue
        }

        public static let supportedArchitectures = Self(rawValue: 1 << 0)
        public static let cubin = Self(rawValue: 1 << 1)
        public static let ltoIR = Self(rawValue: 1 << 2)
        public static let optiXIR = Self(rawValue: 1 << 3)
        public static let precompiledHeaders = Self(rawValue: 1 << 4)
        public static let flowCallbacks = Self(rawValue: 1 << 5)
        public static let tileIR = Self(rawValue: 1 << 6)
        public static let bundledHeaders = Self(rawValue: 1 << 7)
    }

    /// An in-memory header supplied to `nvrtcCreateProgram`.
    public struct Header: Sendable, Hashable {
        public var source: String
        public var includeName: String

        @inlinable
        public init(source: String, includeName: String) {
            self.source = source
            self.includeName = includeName
        }
    }
}
