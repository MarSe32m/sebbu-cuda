// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CNVRTC

@inline(__always)
internal func nvrtcCheck(
    _ result: Int32,
    feature: String? = nil,
    detailedMessage: String? = nil
) throws {
    guard result != 0 else { return }
    if result == -1 {
        throw NVRTC.UnsupportedFeatureError(
            feature: feature ?? "the requested API"
        )
    }
    throw NVRTC.Error(result, detailedMessage: detailedMessage)
}

internal func nvrtcValidateCString(
    _ value: String,
    field: String
) throws {
    guard !value.utf8.contains(0) else {
        throw NVRTC.ValidationError.embeddedNull(field: field)
    }
}

internal func nvrtcInt32Count(
    _ count: Int,
    field: String
) throws -> Int32 {
    guard let value = Int32(exactly: count) else {
        throw NVRTC.ValidationError.integerConversionOverflow(
            value: count,
            target: "Int32 for \(field)"
        )
    }
    return value
}

internal func nvrtcWithOptionalCString<Result>(
    _ value: String?,
    field: String,
    _ body: (UnsafePointer<CChar>?) throws -> Result
) throws -> Result {
    guard let value else { return try body(nil) }
    try nvrtcValidateCString(value, field: field)
    return try value.withCString(body)
}

internal func nvrtcWithCStringArray<Result>(
    _ values: [String],
    field: String,
    _ body: (
        UnsafePointer<UnsafePointer<CChar>?>?
    ) throws -> Result
) throws -> Result {
    guard !values.isEmpty else { return try body(nil) }

    var allocations: [UnsafeMutablePointer<CChar>] = []
    allocations.reserveCapacity(values.count)
    do {
        for value in values {
            try nvrtcValidateCString(value, field: field)
            let bytes = Array(value.utf8CString)
            let allocation = UnsafeMutablePointer<CChar>.allocate(
                capacity: bytes.count
            )
            bytes.withUnsafeBufferPointer {
                allocation.initialize(
                    from: $0.baseAddress!,
                    count: $0.count
                )
            }
            allocations.append(allocation)
        }
    } catch {
        for allocation in allocations {
            allocation.deallocate()
        }
        throw error
    }
    defer {
        for allocation in allocations {
            allocation.deallocate()
        }
    }

    let pointers: [UnsafePointer<CChar>?] = allocations.map {
        UnsafePointer($0)
    }
    return try pointers.withUnsafeBufferPointer {
        try body($0.baseAddress)
    }
}

internal func nvrtcReadString(
    feature: String? = nil,
    size: (UnsafeMutablePointer<Int>) -> Int32,
    read: (UnsafeMutablePointer<CChar>?) -> Int32
) throws -> String {
    var count = 0
    try nvrtcCheck(size(&count), feature: feature)
    guard count >= 0 else {
        throw NVRTC.ValidationError.negativeByteCount(count)
    }
    guard count > 0 else { return "" }

    var bytes = [CChar](repeating: 0, count: count)
    try bytes.withUnsafeMutableBufferPointer {
        try nvrtcCheck(read($0.baseAddress), feature: feature)
    }
    return bytes.withUnsafeBufferPointer {
        String(cString: $0.baseAddress!)
    }
}

internal func nvrtcReadBytes(
    feature: String,
    size: (UnsafeMutablePointer<Int>) -> Int32,
    read: (UnsafeMutablePointer<CChar>?) -> Int32
) throws -> [UInt8] {
    var count = 0
    try nvrtcCheck(size(&count), feature: feature)
    guard count >= 0 else {
        throw NVRTC.ValidationError.negativeByteCount(count)
    }
    guard count > 0 else { return [] }

    var bytes = [UInt8](repeating: 0, count: count)
    try bytes.withUnsafeMutableBytes { buffer in
        try nvrtcCheck(
            read(buffer.baseAddress?.assumingMemoryBound(to: CChar.self)),
            feature: feature
        )
    }
    return bytes
}
