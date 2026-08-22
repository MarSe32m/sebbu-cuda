// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

internal import CCUDA

internal final class CUDAEventStorage {
    package let rawValue: OpaquePointer
    internal let context: CUDA.Context

    internal init(rawValue: OpaquePointer, context: CUDA.Context) {
        self.rawValue = rawValue
        self.context = context
    }

    deinit {
        context.withCurrentForCleanup(operation: "cuEventDestroy") {
            sebbuCuEventDestroy(rawValue)
        }
    }
}

extension CUDA {
    /// A uniquely owned CUDA event.
    public struct Event: ~Copyable {
        internal let storage: CUDAEventStorage

        package var rawValue: OpaquePointer { storage.rawValue }
        internal var context: Context { storage.context }

        internal init(storage: CUDAEventStorage) {
            self.storage = storage
        }

        /// Records the event after work already queued on `stream`.
        public borrowing func record(
            on stream: borrowing CUDA.Stream
        ) throws {
            try cudaRequireSameContext(storage.context, stream.context)
            try storage.context.withCurrent {
                try cudaCheck(
                    cuEventRecord(storage.rawValue, stream.rawValue)
                )
            }
        }

        /// Blocks until this event has completed.
        public borrowing func synchronize() throws {
            try storage.context.withCurrent {
                try cudaCheck(cuEventSynchronize(storage.rawValue))
            }
        }

        /// Returns the elapsed duration between a start event and this event.
        /// Both events must have timing enabled and must have been recorded.
        public borrowing func elapsedTime(
            since start: borrowing Event
        ) throws -> Duration {
            try cudaRequireSameContext(storage.context, start.context)

            var milliseconds: Float = 0
            try storage.context.withCurrent {
                try cudaCheck(
                    sebbuCuEventElapsedTime(
                        &milliseconds,
                        start.rawValue,
                        storage.rawValue
                    )
                )
            }

            let nanoseconds = max(0, Double(milliseconds) * 1_000_000)
            return .nanoseconds(Int64(nanoseconds.rounded()))
        }
    }
}

extension CUDA.Event {
    /// Event creation flags from the CUDA Driver API.
    public struct Flags: OptionSet, Sendable, Hashable {
        public let rawValue: UInt32

        @inlinable
        public init(rawValue: UInt32) {
            self.rawValue = rawValue
        }

        public static let blockingSynchronization = Self(rawValue: 0x01)
        public static let disableTiming = Self(rawValue: 0x02)
        public static let interprocess = Self(rawValue: 0x04)
    }
}

extension CUDA.Context {
    /// Creates a uniquely owned event in this context.
    public func makeEvent(
        flags: CUDA.Event.Flags = []
    ) throws -> CUDA.Event {
        var rawEvent: CUevent?
        try withCurrent {
            try cudaCheck(cuEventCreate(&rawEvent, flags.rawValue))
        }
        guard let rawEvent else {
            throw CUDA.ValidationError.missingHandle(resource: "event")
        }
        return CUDA.Event(
            storage: CUDAEventStorage(rawValue: rawEvent, context: self)
        )
    }
}
