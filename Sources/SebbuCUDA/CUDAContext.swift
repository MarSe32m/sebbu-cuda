internal import CCUDA

extension CUDA {
    /// A reference-counted owner of a CUDA context.
    ///
    /// Child resources retain their context, so the native context cannot be
    /// destroyed while a stream, allocation, event, or module still uses it.
    public final class Context {
        package let rawValue: OpaquePointer

        fileprivate enum ReleaseKind {
            case destroy
            case releasePrimary(device: Int32)
        }

        private let releaseKind: ReleaseKind

        fileprivate init(
            rawValue: OpaquePointer,
            releaseKind: ReleaseKind
        ) {
            self.rawValue = rawValue
            self.releaseKind = releaseKind
        }

        deinit {
            switch releaseKind {
            case .destroy:
                cudaCleanupCheck(
                    sebbuCuCtxDestroy(rawValue),
                    operation: "cuCtxDestroy"
                )
            case .releasePrimary(let device):
                cudaCleanupCheck(
                    sebbuCuDevicePrimaryCtxRelease(device),
                    operation: "cuDevicePrimaryCtxRelease"
                )
            }
        }

        /// Runs a closure while this context is current on the calling thread.
        ///
        /// If another context is current, it is restored after `body` returns
        /// or throws. CUDA context currentness is thread-local; the closure must
        /// not escape work that assumes the context remains current.
        public func withCurrent<Result>(
            _ body: () throws -> Result
        ) throws -> Result {
            var current: CUcontext?
            try cudaCheck(cuCtxGetCurrent(&current))

            if current == rawValue {
                return try body()
            }

            try cudaCheck(sebbuCuCtxPushCurrent(rawValue))

            let result: Result
            do {
                result = try body()
            } catch {
                var popped: CUcontext?
                cudaCleanupCheck(
                    sebbuCuCtxPopCurrent(&popped),
                    operation: "cuCtxPopCurrent"
                )
                throw error
            }

            var popped: CUcontext?
            try cudaCheck(sebbuCuCtxPopCurrent(&popped))
            assert(
                popped == rawValue,
                "CUDA popped a context other than the one pushed."
            )
            return result
        }

        /// Runs a non-throwing cleanup operation with this context current.
        internal func withCurrentForCleanup(
            operation: StaticString,
            _ body: () -> CUresult
        ) {
            do {
                try withCurrent {
                    cudaCleanupCheck(body(), operation: operation)
                }
            } catch {
                assertionFailure(
                    "Could not make a CUDA context current for \(operation): " +
                    "\(error)"
                )
            }
        }
    }
}

extension CUDA.Context {
    /// Context creation flags from the CUDA Driver API.
    public struct Flags: OptionSet, Sendable, Hashable {
        public let rawValue: UInt32

        @inlinable
        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        /// Spin while waiting for CUDA work.
        public static let scheduleSpin = Self(rawValue: 0x01)
        /// Yield the host thread while waiting for CUDA work.
        public static let scheduleYield = Self(rawValue: 0x02)
        /// Block on a synchronization primitive while waiting.
        public static let scheduleBlocking = Self(rawValue: 0x04)
        /// Permit mapping pinned host memory into the device address space.
        public static let mapHostMemory = Self(rawValue: 0x08)
        /// Keep local-memory allocations after kernel launches.
        public static let localMemoryResizeToMaximum = Self(rawValue: 0x10)
    }
}

extension CUDA.Device {
    /// Creates a new CUDA context owned by the returned object.
    ///
    /// CUDA makes a newly created context current. This wrapper immediately
    /// pops it so context creation does not alter the caller's current context.
    public func makeContext(
        flags: CUDA.Context.Flags = []
    ) throws -> CUDA.Context {
        try cudaEnsureInitialized()

        var rawContext: CUcontext?
        try cudaCheck(
            sebbuCuCtxCreate(&rawContext, flags.rawValue, rawValue)
        )

        guard let rawContext else {
            throw CUDA.ValidationError.missingHandle(resource: "context")
        }

        var popped: CUcontext?
        do {
            try cudaCheck(sebbuCuCtxPopCurrent(&popped))
        } catch {
            cudaCleanupCheck(
                sebbuCuCtxDestroy(rawContext),
                operation: "cuCtxDestroy"
            )
            throw error
        }

        guard popped == rawContext else {
            cudaCleanupCheck(
                sebbuCuCtxDestroy(rawContext),
                operation: "cuCtxDestroy"
            )
            throw CUDA.ValidationError.missingHandle(
                resource: "newly-current context"
            )
        }

        return CUDA.Context(rawValue: rawContext, releaseKind: .destroy)
    }

    /// Retains this device's primary context.
    ///
    /// The returned object balances the retain with
    /// `cuDevicePrimaryCtxRelease`, rather than destroying the shared context.
    public func retainPrimaryContext() throws -> CUDA.Context {
        try cudaEnsureInitialized()

        var rawContext: CUcontext?
        try cudaCheck(cuDevicePrimaryCtxRetain(&rawContext, rawValue))
        guard let rawContext else {
            throw CUDA.ValidationError.missingHandle(
                resource: "primary context"
            )
        }

        return CUDA.Context(
            rawValue: rawContext,
            releaseKind: .releasePrimary(device: rawValue)
        )
    }
}
