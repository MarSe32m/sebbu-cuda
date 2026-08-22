internal import CCUDA

internal final class CUDAStreamStorage {
    package let rawValue: OpaquePointer
    internal let context: CUDA.Context

    // In-flight asynchronous copies and launches retain their backing owners
    // until successful synchronization. This is deliberate internal sharing;
    // the public Stream and allocation values remain noncopyable unique owners.
    private var inFlightOwners: [AnyObject] = []

    internal init(rawValue: OpaquePointer, context: CUDA.Context) {
        self.rawValue = rawValue
        self.context = context
    }

    deinit {
        if !inFlightOwners.isEmpty {
            var didSynchronize = false
            do {
                try context.withCurrent {
                    try cudaCheck(cuStreamSynchronize(rawValue))
                }
                didSynchronize = true
            } catch {
                assertionFailure(
                    "CUDA stream cleanup could not synchronize: \(error)"
                )
            }

            if !didSynchronize {
                // Releasing pinned or device memory while DMA may still be in
                // flight is unsafe. Leak only on an unrecoverable cleanup error.
                for owner in inFlightOwners {
                    _ = Unmanaged.passRetained(owner)
                }
            }
            inFlightOwners.removeAll(keepingCapacity: false)
        }

        context.withCurrentForCleanup(operation: "cuStreamDestroy") {
            sebbuCuStreamDestroy(rawValue)
        }
    }

    internal func retainUntilSynchronization(_ owner: AnyObject) {
        inFlightOwners.append(owner)
    }

    internal func retainUntilSynchronization(
        _ first: AnyObject,
        _ second: AnyObject
    ) {
        inFlightOwners.append(first)
        inFlightOwners.append(second)
    }

    internal func releaseSynchronizedOwners() {
        inFlightOwners.removeAll(keepingCapacity: true)
    }
}

extension CUDA {
    /// A uniquely owned CUDA stream.
    public struct Stream: ~Copyable {
        internal let storage: CUDAStreamStorage

        package var rawValue: OpaquePointer { storage.rawValue }
        internal var context: Context { storage.context }

        internal init(storage: CUDAStreamStorage) {
            self.storage = storage
        }

        /// Blocks until all work previously submitted to this stream finishes.
        public borrowing func synchronize() throws {
            try storage.context.withCurrent {
                try cudaCheck(cuStreamSynchronize(storage.rawValue))
            }
            storage.releaseSynchronizedOwners()
        }

        /// Temporarily exposes the native stream handle for interoperability.
        ///
        /// The handle is borrowed and must not be destroyed or retained beyond
        /// the closure. The callback does not require importing `CCUDA`. Work
        /// submitted through the raw handle is not lifetime-tracked by this
        /// wrapper; its resources must be kept alive by the caller.
        public borrowing func withUnsafeRawHandle<Result>(
            _ body: (OpaquePointer) throws -> Result
        ) rethrows -> Result {
            try body(storage.rawValue)
        }
    }
}

extension CUDA.Stream {
    /// Stream creation flags from the CUDA Driver API.
    public struct Flags: OptionSet, Sendable, Hashable {
        public let rawValue: UInt32

        @inlinable
        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        /// Allows the stream to execute independently of the legacy default
        /// stream's implicit synchronization behavior.
        public static let nonBlocking = Self(rawValue: 0x01)
    }
}

extension CUDA.Context {
    /// Creates a uniquely owned stream in this context.
    public func makeStream(
        flags: CUDA.Stream.Flags = []
    ) throws -> CUDA.Stream {
        var rawStream: CUstream?
        try withCurrent {
            try cudaCheck(cuStreamCreate(&rawStream, flags.rawValue))
        }
        guard let rawStream else {
            throw CUDA.ValidationError.missingHandle(resource: "stream")
        }
        return CUDA.Stream(
            storage: CUDAStreamStorage(rawValue: rawStream, context: self)
        )
    }
}
