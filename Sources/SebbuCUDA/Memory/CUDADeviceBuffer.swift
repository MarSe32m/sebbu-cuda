// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

internal final class CUDADeviceAllocationStorage {
    package let rawValue: UInt64
    internal let byteCount: Int
    internal let context: CUDA.Context

    internal init(
        rawValue: UInt64,
        byteCount: Int,
        context: CUDA.Context
    ) {
        self.rawValue = rawValue
        self.byteCount = byteCount
        self.context = context
    }

    deinit {
        guard rawValue != 0 else { return }
        context.withCurrentForCleanup(operation: "cuMemFree") {
            sebbuCuMemFree(rawValue)
        }
    }
}

extension CUDA {
    /// A uniquely owned, typed CUDA device allocation.
    ///
    /// A zero-count buffer is valid. It has byte count zero, address zero, and
    /// does not call `cuMemAlloc` or `cuMemFree`.
    public struct DeviceBuffer<Element: BitwiseCopyable>: ~Copyable {
        internal let storage: CUDADeviceAllocationStorage
        public let count: Int

        package var rawValue: UInt64 { storage.rawValue }

        internal init(
            storage: CUDADeviceAllocationStorage,
            count: Int
        ) {
            self.storage = storage
            self.count = count
        }

        public var byteCount: Int { storage.byteCount }

        public var pointer: CUDA.DevicePointer<Element> {
            CUDA.DevicePointer(rawValue: storage.rawValue)
        }
    }
}

extension CUDA.Context {
    /// Allocates typed device memory.
    public func allocate<Element: BitwiseCopyable>(
        _ type: Element.Type = Element.self,
        count: Int
    ) throws -> CUDA.DeviceBuffer<Element> {
        let byteCount = try cudaCheckedByteCount(
            count: count,
            stride: MemoryLayout<Element>.stride
        )

        if byteCount == 0 {
            return CUDA.DeviceBuffer(
                storage: CUDADeviceAllocationStorage(
                    rawValue: 0,
                    byteCount: 0,
                    context: self
                ),
                count: count
            )
        }

        var rawValue: CUdeviceptr = 0
        try withCurrent {
            try cudaCheck(sebbuCuMemAlloc(&rawValue, byteCount))
        }
        return CUDA.DeviceBuffer(
            storage: CUDADeviceAllocationStorage(
                rawValue: rawValue,
                byteCount: byteCount,
                context: self
            ),
            count: count
        )
    }
}
