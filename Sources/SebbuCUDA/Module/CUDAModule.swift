// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

extension CUDA {
    /// A shared owner of a loaded CUDA module.
    ///
    /// Kernels retain their module, and streams retain a launched kernel's
    /// module until synchronization, preventing premature unload.
    public final class Module {
        package let rawValue: OpaquePointer
        internal let context: Context

        internal init(rawValue: OpaquePointer, context: Context) {
            self.rawValue = rawValue
            self.context = context
        }

        deinit {
            context.withCurrentForCleanup(operation: "cuModuleUnload") {
                cuModuleUnload(rawValue)
            }
        }

        /// Looks up a kernel entry point by its unmangled name.
        public func kernel(named name: String) throws -> CUDA.Kernel {
            var rawFunction: CUfunction?
            try context.withCurrent {
                try name.withCString { name in
                    try cudaCheck(
                        cuModuleGetFunction(
                            &rawFunction,
                            rawValue,
                            name
                        )
                    )
                }
            }
            guard let rawFunction else {
                throw CUDA.ValidationError.missingHandle(resource: "kernel")
            }
            return CUDA.Kernel(rawValue: rawFunction, module: self)
        }
    }
}

extension CUDA.Context {
    /// Loads a CUDA module from a file.
    ///
    /// The path may refer to a PTX, cubin, or fatbin file. Relative paths are
    /// resolved from the process's current working directory by the CUDA
    /// driver.
    public func loadModule(atPath path: String) throws -> CUDA.Module {
        var rawModule: CUmodule?
        try withCurrent {
            try path.withCString { path in
                try cudaCheck(cuModuleLoad(&rawModule, path))
            }
        }
        guard let rawModule else {
            throw CUDA.ValidationError.missingHandle(resource: "module")
        }
        return CUDA.Module(rawValue: rawModule, context: self)
    }

    /// Loads a PTX or binary module image from memory.
    ///
    /// CUDA consumes the image during this call; the input buffer does not need
    /// to remain alive after the method returns.
    public func loadModule(
        image: UnsafeRawBufferPointer
    ) throws -> CUDA.Module {
        guard !image.isEmpty, let baseAddress = image.baseAddress else {
            throw CUDA.ValidationError.emptyModuleImage
        }

        var rawModule: CUmodule?
        try withCurrent {
            try cudaCheck(cuModuleLoadData(&rawModule, baseAddress))
        }
        guard let rawModule else {
            throw CUDA.ValidationError.missingHandle(resource: "module")
        }
        return CUDA.Module(rawValue: rawModule, context: self)
    }

    /// Loads a null-terminated PTX string as a CUDA module.
    public func loadModule(ptx: String) throws -> CUDA.Module {
        var rawModule: CUmodule?
        try withCurrent {
            try ptx.withCString { ptx in
                try cudaCheck(cuModuleLoadData(&rawModule, ptx))
            }
        }
        guard let rawModule else {
            throw CUDA.ValidationError.missingHandle(resource: "module")
        }
        return CUDA.Module(rawValue: rawModule, context: self)
    }
}
