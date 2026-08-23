// swift-tools-version: 6.2

import Foundation
import PackageDescription

let environment = ProcessInfo.processInfo.environment

func appending(_ component: String, to root: String) -> String {
    if root.hasSuffix("/") || root.hasSuffix("\\") {
        return root + component
    }
    return root + "/" + component
}

#if os(Windows)
guard let cudaRoot = environment["CUDA_PATH"], !cudaRoot.isEmpty else {
    fatalError(
        "sebbu-cuda requires the CUDA Toolkit. Set CUDA_PATH to the toolkit " +
        "root (for example C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0)."
    )
}
#elseif os(Linux)
// Prefer CUDA_PATH. On some platforms, CUDA_HOME is defined instead. In most cases cuda is found in /usr/local/cuda
let cudaRoot = if let cuda_path = environment["CUDA_PATH"], !cuda_path.isEmpty {
    cuda_path
} else if let cuda_home = environment["CUDA_HOME"], !cuda_home.isEmpty {
    cuda_home
} else {
    "/usr/local/cuda"
}
#else
let cudaRoot = ""
#endif

let cudaIncludePath = appending("include", to: cudaRoot)
let cudaHeaderPath = appending("cuda.h", to: cudaIncludePath)

#if os(Windows) || os(Linux)
guard FileManager.default.fileExists(atPath: cudaHeaderPath) else {
    fatalError(
        "sebbu-cuda could not find cuda.h at \(cudaHeaderPath). " +
        "Install the CUDA Toolkit or set CUDA_PATH to its root."
    )
}
#endif

var cudaLinkerFlags: [String] = []

#if os(Windows)
let cudaLibraryPath = appending("lib/x64", to: cudaRoot)
let cudaImportLibrary = appending("cuda.lib", to: cudaLibraryPath)
guard FileManager.default.fileExists(atPath: cudaImportLibrary) else {
    fatalError(
        "sebbu-cuda could not find the CUDA Driver import library at " +
        "\(cudaImportLibrary). Check CUDA_PATH and the toolkit installation."
    )
}
cudaLinkerFlags = ["-L", cudaLibraryPath]
#elseif os(Linux)
// Normally the NVIDIA driver installation supplies libcuda through the
// system linker's search path. CUDA_LIBRARY_PATH is an explicit escape hatch
// for toolkit stubs, cross-compilation sysroots, and unusual installations.
if let cudaLibraryPath = environment["CUDA_LIBRARY_PATH"],
   !cudaLibraryPath.isEmpty {
    cudaLinkerFlags = ["-L", cudaLibraryPath]
}
#else
let cudaLibraryPath = ""
#endif

let cudaCSettings: [CSetting] = [
    .unsafeFlags(["-isystem", cudaIncludePath]),
]

let cudaSwiftSettings: [SwiftSetting] = [
    .unsafeFlags(["-Xcc", "-isystem", "-Xcc", cudaIncludePath]),
]

let cudaLinkerSettings: [LinkerSetting] = [
    .unsafeFlags(cudaLinkerFlags),
    .linkedLibrary("cuda"),
    .linkedLibrary("m", .when(platforms: [.linux]))
]

let package = Package(
    name: "sebbu-cuda",
    products: [
        .library(name: "SebbuCUDA", targets: ["SebbuCUDA"]),
        .executable(
            name: "sebbu-cuda-development",
            targets: ["Development"]
        ),
    ],
    targets: [
        .target(
            name: "CCUDA",
            path: "Sources/CCUDA",
            publicHeadersPath: "include",
            cSettings: cudaCSettings,
            linkerSettings: cudaLinkerSettings
        ),
        .target(
            name: "SebbuCUDA",
            dependencies: ["CCUDA"],
            path: "Sources/SebbuCUDA",
            swiftSettings: cudaSwiftSettings,
            linkerSettings: cudaLinkerSettings
        ),
        .executableTarget(
            name: "Development",
            dependencies: ["SebbuCUDA"],
            path: "Sources/Development",
            swiftSettings: cudaSwiftSettings,
            linkerSettings: cudaLinkerSettings
        ),
        .testTarget(
            name: "SebbuCUDATests",
            dependencies: ["SebbuCUDA"],
            path: "Tests/SebbuCUDATests",
            swiftSettings: cudaSwiftSettings,
            linkerSettings: cudaLinkerSettings
        ),
    ],
    swiftLanguageModes: [.v6]
)
