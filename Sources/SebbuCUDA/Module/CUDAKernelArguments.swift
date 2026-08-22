internal final class CUDAKernelArgumentStorage {
    private var allocations: [UnsafeMutableRawPointer] = []

    deinit {
        for allocation in allocations {
            allocation.deallocate()
        }
    }

    internal var count: Int { allocations.count }

    internal func append<T: BitwiseCopyable>(_ value: T) {
        let allocation = UnsafeMutableRawPointer.allocate(
            byteCount: max(1, MemoryLayout<T>.size),
            alignment: max(1, MemoryLayout<T>.alignment)
        )
        allocation.storeBytes(of: value, as: T.self)
        allocations.append(allocation)
    }

    internal func withUnsafeArgumentPointers<Result>(
        _ body: (UnsafeMutablePointer<UnsafeMutableRawPointer?>?) throws -> Result
    ) rethrows -> Result {
        var pointers: [UnsafeMutableRawPointer?] = allocations.map { $0 }
        return try pointers.withUnsafeMutableBufferPointer { pointers in
            try body(pointers.baseAddress)
        }
    }
}

extension CUDA {
    /// Scoped storage for CUDA kernel arguments.
    ///
    /// Each appended value receives a stable, correctly aligned address. The
    /// launch passes an array of those addresses to `cuLaunchKernel`, exactly as
    /// required by the Driver API's `void **kernelParams` contract.
    public struct KernelArguments {
        internal let storage: CUDAKernelArgumentStorage

        internal init() {
            storage = CUDAKernelArgumentStorage()
        }

        /// Appends a scalar or other bitwise-copyable value argument.
        public mutating func append<T: BitwiseCopyable>(_ value: T) {
            storage.append(value)
        }

        /// Appends the device address, not the Swift pointer wrapper bytes.
        public mutating func append<T>(
            _ pointer: CUDA.DevicePointer<T>
        ) {
            storage.append(pointer.rawValue)
        }
    }
}
