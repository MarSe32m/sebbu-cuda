// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

extension CUDA.DeviceBuffer {
    /// Copies a contiguous host array into the start of this allocation.
    public borrowing func copy(from source: [Element]) throws {
        try source.withUnsafeBufferPointer { source in
            try copy(from: source)
        }
    }

    /// Copies a contiguous host buffer into the start of this allocation.
    public borrowing func copy(
        from source: UnsafeBufferPointer<Element>
    ) throws {
        guard source.count <= count else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: source.count,
                available: count
            )
        }
        let byteCount = try cudaCheckedByteCount(
            count: source.count,
            stride: MemoryLayout<Element>.stride
        )
        guard byteCount != 0 else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyHtoD(
                    storage.rawValue,
                    UnsafeRawPointer(source.baseAddress!),
                    byteCount
                )
            )
        }
    }

    /// Copies from the start of this allocation into a host array.
    /// The array's existing count determines how many elements are copied.
    public borrowing func copy(to destination: inout [Element]) throws {
        try destination.withUnsafeMutableBufferPointer { destination in
            try copy(to: destination)
        }
    }

    /// Copies from this allocation into a contiguous mutable host buffer.
    public borrowing func copy(
        to destination: UnsafeMutableBufferPointer<Element>
    ) throws {
        guard destination.count <= count else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: destination.count,
                available: count
            )
        }
        let byteCount = try cudaCheckedByteCount(
            count: destination.count,
            stride: MemoryLayout<Element>.stride
        )
        guard byteCount != 0 else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyDtoH(
                    UnsafeMutableRawPointer(destination.baseAddress!),
                    storage.rawValue,
                    byteCount
                )
            )
        }
    }
}

extension CUDA.DeviceMemory {
    /// Copies raw host bytes into the start of this allocation.
    public borrowing func copy(
        from source: UnsafeRawBufferPointer
    ) throws {
        guard source.count <= byteCount else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: source.count,
                available: byteCount
            )
        }
        guard !source.isEmpty else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyHtoD(
                    storage.rawValue,
                    source.baseAddress!,
                    source.count
                )
            )
        }
    }

    /// Copies raw bytes from this allocation into a host buffer.
    public borrowing func copy(
        to destination: UnsafeMutableRawBufferPointer
    ) throws {
        guard destination.count <= byteCount else {
            throw CUDA.ValidationError.bufferTooSmall(
                required: destination.count,
                available: byteCount
            )
        }
        guard !destination.isEmpty else { return }

        try storage.context.withCurrent {
            try cudaCheck(
                sebbuCuMemcpyDtoH(
                    destination.baseAddress!,
                    storage.rawValue,
                    destination.count
                )
            )
        }
    }
}
