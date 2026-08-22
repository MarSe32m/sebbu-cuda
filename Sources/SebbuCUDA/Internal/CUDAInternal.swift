internal import CCUDA

@inline(__always)
internal func cudaEnsureInitialized() throws {
    // cuInit(0) is explicitly safe to call more than once. Calling it at each
    // high-level entry point avoids global mutable initialization state.
    try cudaCheck(cuInit(0))
}

@inline(__always)
internal func cudaCheck(_ result: CUresult) throws {
    guard result == CUDA_SUCCESS else {
        throw CUDA.Error(result)
    }
}

@inline(__always)
internal func cudaCleanupCheck(
    _ result: CUresult,
    operation: StaticString = #function
) {
    if result != CUDA_SUCCESS {
        assertionFailure(
            "CUDA cleanup failed in \(operation): \(CUDA.Error(result))"
        )
    }
}

@inline(__always)
internal func cudaRequireSameContext(
    _ lhs: CUDA.Context,
    _ rhs: CUDA.Context
) throws {
    guard lhs.rawValue == rhs.rawValue else {
        throw CUDA.ValidationError.contextMismatch
    }
}

@inline(__always)
internal func cudaCheckedByteCount(
    count: Int,
    stride: Int
) throws -> Int {
    guard count >= 0 else {
        throw CUDA.ValidationError.negativeCount(count)
    }

    let (byteCount, overflow) = count.multipliedReportingOverflow(by: stride)
    guard !overflow else {
        throw CUDA.ValidationError.sizeOverflow(count: count, stride: stride)
    }
    return byteCount
}
