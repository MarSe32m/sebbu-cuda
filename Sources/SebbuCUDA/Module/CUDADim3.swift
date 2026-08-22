// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

extension CUDA {
    /// Three-dimensional CUDA grid or block dimensions.
    public struct Dim3: Sendable, Hashable {
        public var x: UInt32
        public var y: UInt32
        public var z: UInt32

        @inlinable
        public init(x: UInt32, y: UInt32 = 1, z: UInt32 = 1) {
            self.x = x
            self.y = y
            self.z = z
        }
    }
}
