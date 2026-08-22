// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

internal final class CUDAPinnedAllocationStorage {
    package let rawValue: UnsafeMutableRawPointer?
    internal let byteCount: Int
    internal let context: CUDA.Context

    internal init(
        rawValue: UnsafeMutableRawPointer?,
        byteCount: Int,
        context: CUDA.Context
    ) {
        self.rawValue = rawValue
        self.byteCount = byteCount
        self.context = context
    }

    deinit {
        guard let rawValue else { return }
        context.withCurrentForCleanup(operation: "cuMemFreeHost") {
            cuMemFreeHost(rawValue)
        }
    }
}

extension CUDA {
    /// A uniquely owned allocation of page-locked host memory.
    ///
    /// Newly allocated elements are uninitialized. Initialize them through the
    /// mutable buffer view before reading. The allocation is retained by a
    /// stream while it participates in an asynchronous transfer.
    public struct PinnedBuffer<Element: BitwiseCopyable>: ~Copyable {
        internal let storage: CUDAPinnedAllocationStorage
        public let count: Int

        internal init(
            storage: CUDAPinnedAllocationStorage,
            count: Int
        ) {
            self.storage = storage
            self.count = count
        }

        public var byteCount: Int { storage.byteCount }

        public borrowing func withUnsafeBufferPointer<Result>(
            _ body: (UnsafeBufferPointer<Element>) throws -> Result
        ) rethrows -> Result {
            let pointer = storage.rawValue?.assumingMemoryBound(to: Element.self)
            return try body(
                UnsafeBufferPointer(start: pointer, count: count)
            )
        }

        public mutating func withUnsafeMutableBufferPointer<Result>(
            _ body: (UnsafeMutableBufferPointer<Element>) throws -> Result
        ) rethrows -> Result {
            let pointer = storage.rawValue?.assumingMemoryBound(to: Element.self)
            return try body(
                UnsafeMutableBufferPointer(start: pointer, count: count)
            )
        }
    }
}

extension CUDA {
    /// Pinned-host allocation flags from the CUDA Driver API.
    ///
    /// This type deliberately lives outside generic `PinnedBuffer<Element>`.
    /// The flags configure the allocation itself and do not depend on the
    /// buffer's element type.
    public struct PinnedMemoryFlags: OptionSet, Sendable, Hashable {
        public let rawValue: UInt32

        @inlinable
        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        public static let portable = Self(rawValue: 0x01)
        public static let deviceMapped = Self(rawValue: 0x02)
        public static let writeCombined = Self(rawValue: 0x04)
    }
}

extension CUDA.Context {
    /// Allocates page-locked host memory suitable for asynchronous transfers.
    public func allocatePinned<Element: BitwiseCopyable>(
        _ type: Element.Type = Element.self,
        count: Int,
        flags: CUDA.PinnedMemoryFlags = []
    ) throws -> CUDA.PinnedBuffer<Element> {
        let byteCount = try cudaCheckedByteCount(
            count: count,
            stride: MemoryLayout<Element>.stride
        )

        if byteCount == 0 {
            return CUDA.PinnedBuffer(
                storage: CUDAPinnedAllocationStorage(
                    rawValue: nil,
                    byteCount: 0,
                    context: self
                ),
                count: count
            )
        }

        var rawValue: UnsafeMutableRawPointer?
        try withCurrent {
            try cudaCheck(
                cuMemHostAlloc(&rawValue, byteCount, flags.rawValue)
            )
        }
        guard let rawValue else {
            throw CUDA.ValidationError.missingHandle(
                resource: "pinned host allocation"
            )
        }

        return CUDA.PinnedBuffer(
            storage: CUDAPinnedAllocationStorage(
                rawValue: rawValue,
                byteCount: byteCount,
                context: self
            ),
            count: count
        )
    }
}

extension CUDA.Stream {
    /// Enqueues an asynchronous pinned-host-to-device copy.
    ///
    /// The stream retains both allocations until successful synchronization.
    public borrowing func copy<Element: BitwiseCopyable>(
        from source: borrowing CUDA.PinnedBuffer<Element>,
        to destination: borrowing CUDA.DeviceBuffer<Element>
    ) throws {
        try cudaRequireSameContext(storage.context, source.storage.context)
        try cudaRequireSameContext(storage.context, destination.storage.context)
        guard source.count <= destination.count else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: source.count,
                available: destination.count
            )
        }
        guard source.byteCount != 0 else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyHtoDAsync(
                    destination.rawValue,
                    source.storage.rawValue!,
                    source.byteCount,
                    storage.rawValue
                )
            )
        }
        storage.retainUntilSynchronization(
            source.storage,
            destination.storage
        )
    }

    /// Enqueues an asynchronous device-to-pinned-host copy.
    ///
    /// The stream retains both allocations until successful synchronization.
    public borrowing func copy<Element: BitwiseCopyable>(
        from source: borrowing CUDA.DeviceBuffer<Element>,
        to destination: borrowing CUDA.PinnedBuffer<Element>
    ) throws {
        try cudaRequireSameContext(storage.context, source.storage.context)
        try cudaRequireSameContext(storage.context, destination.storage.context)
        guard destination.count <= source.count else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: destination.count,
                available: source.count
            )
        }
        guard destination.byteCount != 0 else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyDtoHAsync(
                    destination.storage.rawValue!,
                    source.rawValue,
                    destination.byteCount,
                    storage.rawValue
                )
            )
        }
        storage.retainUntilSynchronization(
            source.storage,
            destination.storage
        )
    }
}
