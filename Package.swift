// swift-tools-version: 6.2

import Foundation
import PackageDescription

let environment = ProcessInfo.processInfo.environment
let fileManager = FileManager.default

func environmentPath(_ name: String) -> String? {
    environment[name].flatMap { $0.isEmpty ? nil : $0 }
}

func appending(_ component: String, to root: String) -> String {
    if root.hasSuffix("/") || root.hasSuffix("\\") {
        return root + component
    }
    return root + "/" + component
}

#if os(Windows)
let cudaRoot = environmentPath("CUDA_PATH") ?? ""
#elseif os(Linux)
let cudaRoot = environmentPath("CUDA_PATH")
    ?? environmentPath("CUDA_HOME")
    ?? "/usr/local/cuda"
#else
let cudaRoot = ""
#endif

let cudaIncludePath = environmentPath("CUDA_INCLUDE_PATH")
    ?? (cudaRoot.isEmpty ? "" : appending("include", to: cudaRoot))
let cudaHeaderPath = cudaIncludePath.isEmpty
    ? ""
    : appending("cuda.h", to: cudaIncludePath)
let hasCUDAHeader = !cudaHeaderPath.isEmpty
    && fileManager.fileExists(atPath: cudaHeaderPath)

var cudaLinkerFlags: [String] = []
let canBuildCUDA: Bool

#if os(Windows) && arch(x86_64)
let cudaLibraryPath = environmentPath("CUDA_LIBRARY_PATH")
    ?? appending("lib/x64", to: cudaRoot)
let cudaImportLibrary = appending("cuda.lib", to: cudaLibraryPath)
let hasCUDAImportLibrary = fileManager.fileExists(atPath: cudaImportLibrary)
canBuildCUDA = hasCUDAHeader && hasCUDAImportLibrary
if canBuildCUDA {
    cudaLinkerFlags = ["-L", cudaLibraryPath]
}
#elseif os(Linux)
canBuildCUDA = hasCUDAHeader
if canBuildCUDA,
   let cudaLibraryPath = environment["CUDA_LIBRARY_PATH"],
   !cudaLibraryPath.isEmpty {
    cudaLinkerFlags = ["-L", cudaLibraryPath]
}
#else
canBuildCUDA = false
#endif

var products: [Product] = [
    .library(name: "SebbuCUDA", targets: ["SebbuCUDA"]),
]
var targets: [Target]

if canBuildCUDA {
    let cudaCSettings: [CSetting] = [
        .unsafeFlags(["-isystem", cudaIncludePath]),
    ]
    let cudaLinkerSettings: [LinkerSetting] = [
        .unsafeFlags(cudaLinkerFlags),
        .linkedLibrary("cuda"),
        .linkedLibrary("m", .when(platforms: [.linux])),
    ]

    products.append(
        .executable(
            name: "sebbu-cuda-development",
            targets: ["Development"]
        )
    )
    targets = [
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
            linkerSettings: cudaLinkerSettings
        ),
        .executableTarget(
            name: "Development",
            dependencies: ["SebbuCUDA"],
            path: "Sources/Development",
            linkerSettings: cudaLinkerSettings
        ),
        .testTarget(
            name: "SebbuCUDATests",
            dependencies: ["SebbuCUDA"],
            path: "Tests/SebbuCUDATests",
            linkerSettings: cudaLinkerSettings
        ),
    ]
} else {
    targets = [
        .target(
            name: "SebbuCUDA",
            path: "Sources/SebbuCUDAUnavailable"
        ),
        .testTarget(
            name: "SebbuCUDAUnavailableTests",
            dependencies: ["SebbuCUDA"],
            path: "Tests/SebbuCUDAUnavailableTests"
        ),
    ]
}

let package = Package(
    name: "sebbu-cuda",
    products: products,
    targets: targets,
    swiftLanguageModes: [.v6]
)
