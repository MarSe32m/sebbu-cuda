// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CNVRTC

extension NVRTC {
    public struct BundledHeadersInfo: Sendable, Hashable {
        public let areAvailable: Bool
        public let compressedByteCount: Int
        public let uncompressedByteCount: Int
        public let cudaVersion: Version
        public let fileCount: Int

        @inlinable
        public init(
            areAvailable: Bool,
            compressedByteCount: Int,
            uncompressedByteCount: Int,
            cudaVersion: Version,
            fileCount: Int
        ) {
            self.areAvailable = areAvailable
            self.compressedByteCount = compressedByteCount
            self.uncompressedByteCount = uncompressedByteCount
            self.cudaVersion = cudaVersion
            self.fileCount = fileCount
        }
    }

    public struct BundledHeaderInstallationOptions: OptionSet, Sendable,
        Hashable {
        public let rawValue: UInt32

        @inlinable
        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        /// Skip installation when matching bundled headers already exist.
        public static let skipIfExists: Self = []

        /// Clear the destination before installing the headers.
        public static let forceOverwrite = Self(rawValue: 1 << 0)

        /// Return immediately when another process holds the install lock.
        public static let noWait = Self(rawValue: 1 << 1)
    }

    /// Information about the CUDA headers bundled into NVRTC 13.3 and later.
    public static var bundledHeadersInfo: BundledHeadersInfo {
        get throws {
            var rawInfo = SebbuNVRTCBundledHeadersInfo()
            var errorLog: UnsafePointer<CChar>?
            let result = sebbuNvrtcGetBundledHeadersInfo(
                &rawInfo,
                &errorLog
            )
            let details = errorLog.map { String(cString: $0) }
            try nvrtcCheck(
                result,
                feature: "bundled-header information",
                detailedMessage: details
            )
            return BundledHeadersInfo(
                areAvailable: rawInfo.available != 0,
                compressedByteCount: rawInfo.compressedSize,
                uncompressedByteCount: rawInfo.uncompressedSize,
                cudaVersion: Version(
                    major: Int(rawInfo.cudaVersionMajor),
                    minor: Int(rawInfo.cudaVersionMinor)
                ),
                fileCount: Int(rawInfo.numFiles)
            )
        }
    }

    /// Installs headers bundled with NVRTC into `path`.
    public static func installBundledHeaders(
        at path: String,
        options: BundledHeaderInstallationOptions = []
    ) throws {
        try nvrtcValidateCString(path, field: "bundled-header install path")
        var errorLog: UnsafePointer<CChar>?
        let result = path.withCString {
            sebbuNvrtcInstallBundledHeaders(
                $0,
                options.rawValue,
                &errorLog
            )
        }
        let details = errorLog.map { String(cString: $0) }
        try nvrtcCheck(
            result,
            feature: "bundled-header installation",
            detailedMessage: details
        )
    }

    /// Removes an installed bundled-header directory and all of its contents.
    /// The native NVRTC operation is recursive; only pass a dedicated path.
    public static func removeBundledHeaders(at path: String) throws {
        try nvrtcValidateCString(path, field: "bundled-header install path")
        var errorLog: UnsafePointer<CChar>?
        let result = path.withCString {
            sebbuNvrtcRemoveBundledHeaders($0, &errorLog)
        }
        let details = errorLog.map { String(cString: $0) }
        try nvrtcCheck(
            result,
            feature: "bundled-header removal",
            detailedMessage: details
        )
    }
}
