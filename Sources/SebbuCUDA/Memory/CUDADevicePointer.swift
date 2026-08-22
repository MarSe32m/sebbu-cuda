extension CUDA {
    /// A non-owning, typed address in CUDA device memory.
    public struct DevicePointer<Pointee>: Sendable, Hashable {
        package let rawValue: UInt64

        package init(rawValue: UInt64) {
            self.rawValue = rawValue
        }

        /// Advances by `count * MemoryLayout<Pointee>.stride` bytes.
        ///
        /// Like Swift's unsafe pointer arithmetic, overflow and movement before
        /// address zero are programmer errors and trap a precondition.
        public func advanced(by count: Int) -> Self {
            let (byteCount, overflow) = count.multipliedReportingOverflow(
                by: MemoryLayout<Pointee>.stride
            )
            precondition(!overflow, "CUDA device pointer offset overflow.")
            return advanced(byByteCount: byteCount)
        }

        /// Advances by a raw byte count.
        public func advanced(byByteCount byteCount: Int) -> Self {
            if byteCount >= 0 {
                let (address, overflow) = rawValue.addingReportingOverflow(
                    UInt64(byteCount)
                )
                precondition(!overflow, "CUDA device pointer overflow.")
                return Self(rawValue: address)
            }

            let (address, underflow) = rawValue.subtractingReportingOverflow(
                UInt64(byteCount.magnitude)
            )
            precondition(!underflow, "CUDA device pointer underflow.")
            return Self(rawValue: address)
        }
    }
}
