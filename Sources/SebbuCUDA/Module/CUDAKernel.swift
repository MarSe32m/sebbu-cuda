internal import CCUDA

extension CUDA {
    /// A borrowed CUDA function handle that strongly retains its module.
    public struct Kernel {
        package let rawValue: OpaquePointer
        internal let module: Module

        internal init(rawValue: OpaquePointer, module: Module) {
            self.rawValue = rawValue
            self.module = module
        }

        /// Enqueues this kernel on a stream.
        ///
        /// Argument storage remains valid through `cuLaunchKernel`; CUDA copies
        /// parameter values before the call returns. The stream retains the
        /// module until synchronization so it cannot unload during execution.
        /// Device pointers are intentionally non-owning: their allocations must
        /// also remain alive until the stream has completed the kernel.
        public borrowing func launch(
            grid: CUDA.Dim3,
            block: CUDA.Dim3,
            sharedMemoryBytes: Int = 0,
            on stream: borrowing CUDA.Stream,
            arguments buildArguments: (inout CUDA.KernelArguments) -> Void
        ) throws {
            try cudaRequireSameContext(module.context, stream.context)
            guard grid.x != 0, grid.y != 0, grid.z != 0,
                  block.x != 0, block.y != 0, block.z != 0 else {
                throw CUDA.ValidationError.zeroLaunchDimension
            }
            guard sharedMemoryBytes >= 0 else {
                throw CUDA.ValidationError.negativeSharedMemoryByteCount(
                    sharedMemoryBytes
                )
            }
            guard let sharedMemoryBytes = UInt32(exactly: sharedMemoryBytes) else {
                throw CUDA.ValidationError.integerConversionOverflow(
                    value: sharedMemoryBytes,
                    target: "UInt32"
                )
            }

            var arguments = CUDA.KernelArguments()
            buildArguments(&arguments)

            try module.context.withCurrent {
                try arguments.storage.withUnsafeArgumentPointers { pointers in
                    try cudaCheck(
                        cuLaunchKernel(
                            rawValue,
                            grid.x,
                            grid.y,
                            grid.z,
                            block.x,
                            block.y,
                            block.z,
                            sharedMemoryBytes,
                            stream.rawValue,
                            pointers,
                            nil
                        )
                    )
                }
            }

            stream.storage.retainUntilSynchronization(module)
        }
    }
}
