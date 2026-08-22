internal import CCUDA

/// A namespace for the Swift CUDA Driver API wrapper.
public enum CUDA {}

extension CUDA {
    /// Initializes the CUDA Driver API.
    ///
    /// Calling this function repeatedly is harmless. High-level entry points
    /// also initialize the driver when necessary.
    public static func initialize() throws {
        try cudaEnsureInitialized()
    }

    /// The version supported by the installed CUDA driver.
    public static var driverVersion: Version {
        get throws {
            try cudaEnsureInitialized()
            var rawVersion: Int32 = 0
            try cudaCheck(cuDriverGetVersion(&rawVersion))
            return Version(
                major: Int(rawVersion) / 1_000,
                minor: (Int(rawVersion) % 1_000) / 10
            )
        }
    }

    /// A CUDA version represented as major and minor components.
    public struct Version: Sendable, Hashable, CustomStringConvertible {
        public let major: Int
        public let minor: Int

        @inlinable
        public init(major: Int, minor: Int) {
            self.major = major
            self.minor = minor
        }

        @inlinable
        public var description: String { "\(major).\(minor)" }
    }
}
