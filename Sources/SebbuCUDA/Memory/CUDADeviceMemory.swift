// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

extension CUDA {
    /// A uniquely owned, untyped CUDA device allocation.
    ///
    /// Zero bytes is represented by address zero without calling CUDA's
    /// allocation or free routines.
    public struct DeviceMemory: ~Copyable {
        internal let storage: CUDADeviceAllocationStorage

        package var rawValue: UInt64 { storage.rawValue }
        public var byteCount: Int { storage.byteCount }

        internal init(storage: CUDADeviceAllocationStorage) {
            self.storage = storage
        }

        public var pointer: CUDA.DevicePointer<Void> {
            CUDA.DevicePointer(rawValue: storage.rawValue)
        }
    }
}

extension CUDA.Context {
    /// Allocates an untyped region of device memory.
    public func allocate(byteCount: Int) throws -> CUDA.DeviceMemory {
        guard byteCount >= 0 else {
            throw CUDA.ValidationError.negativeCount(byteCount)
        }

        if byteCount == 0 {
            return CUDA.DeviceMemory(
                storage: CUDADeviceAllocationStorage(
                    rawValue: 0,
                    byteCount: 0,
                    context: self
                )
            )
        }

        var rawValue: CUdeviceptr = 0
        try withCurrent {
            try cudaCheck(sebbuCuMemAlloc(&rawValue, byteCount))
        }
        return CUDA.DeviceMemory(
            storage: CUDADeviceAllocationStorage(
                rawValue: rawValue,
                byteCount: byteCount,
                context: self
            )
        )
    }
}
