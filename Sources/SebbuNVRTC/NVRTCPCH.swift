// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CNVRTC

extension NVRTC {
    /// The current process-wide precompiled-header heap size in bytes.
    public static var pchHeapSize: Int {
        get throws {
            var size = 0
            try nvrtcCheck(
                sebbuNvrtcGetPCHHeapSize(&size),
                feature: "precompiled-header support"
            )
            return size
        }
    }

    /// Changes the process-wide precompiled-header heap size.
    public static func setPCHHeapSize(_ size: Int) throws {
        guard size >= 0 else {
            throw ValidationError.negativeByteCount(size)
        }
        try nvrtcCheck(
            sebbuNvrtcSetPCHHeapSize(size),
            feature: "precompiled-header support"
        )
    }

    public enum PCHCreationStatus: Sendable, Hashable {
        case created
        case notAttempted
        case heapExhausted
        case failed
    }
}
