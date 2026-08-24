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

func firstDirectory(
    containing files: [String],
    among candidates: [String]
) -> String? {
    let fileManager = FileManager.default
    var visited: Set<String> = []
    for candidate in candidates where !candidate.isEmpty {
        guard visited.insert(candidate).inserted else { continue }
        if files.allSatisfy({
            fileManager.fileExists(atPath: appending($0, to: candidate))
        }) {
            return candidate
        }
    }
    return nil
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
   let cudaLibraryPath = environmentPath("CUDA_LIBRARY_PATH") {
    cudaLinkerFlags = ["-L", cudaLibraryPath]
}
#else
canBuildCUDA = false
#endif

// NVRTC is independent of the Driver API. Select it separately so clients can
// compile CUDA C++ source even on a machine without a CUDA-capable GPU or
// installed NVIDIA driver. Use NVIDIA's shared NVRTC distribution on every
// platform. In particular, the Windows static archive is built with /MT while
// the Swift runtime is built with /MD, so they cannot safely coexist in one
// image.
let nvrtcIncludePath = environmentPath("NVRTC_INCLUDE_PATH")
    ?? cudaIncludePath
let nvrtcHeaderPath = nvrtcIncludePath.isEmpty
    ? ""
    : appending("nvrtc.h", to: nvrtcIncludePath)
let hasNVRTCHeader = !nvrtcHeaderPath.isEmpty
    && fileManager.fileExists(atPath: nvrtcHeaderPath)

#if os(Windows) && arch(x86_64)
let nvrtcPlatformSupported = true
let nvrtcLibraryFiles = ["nvrtc.lib"]
let nvrtcDefaultLibraryPaths = cudaRoot.isEmpty ? [] : [
    appending("lib/x64", to: cudaRoot),
]
#elseif os(Linux)
let nvrtcPlatformSupported = true
let nvrtcLibraryFiles = ["libnvrtc.so"]
let nvrtcDefaultLibraryPaths = cudaRoot.isEmpty ? [] : [
    appending("lib64", to: cudaRoot),
    appending("lib", to: cudaRoot),
    appending("targets/x86_64-linux/lib", to: cudaRoot),
    appending("targets/sbsa-linux/lib", to: cudaRoot),
    appending("targets/aarch64-linux/lib", to: cudaRoot),
]
#else
let nvrtcPlatformSupported = false
let nvrtcLibraryFiles: [String] = []
let nvrtcDefaultLibraryPaths: [String] = []
#endif

let nvrtcLibraryPath = firstDirectory(
    containing: nvrtcLibraryFiles,
    among: [
        environmentPath("NVRTC_LIBRARY_PATH") ?? "",
        environmentPath("CUDA_LIBRARY_PATH") ?? "",
    ] + nvrtcDefaultLibraryPaths
)
let canBuildNVRTC = nvrtcPlatformSupported
    && hasNVRTCHeader
    && nvrtcLibraryPath != nil

let nvrtcHeader = (try? String(
    contentsOfFile: nvrtcHeaderPath,
    encoding: .utf8
)) ?? ""

func nvrtcHasDeclaration(_ name: String) -> Bool {
    nvrtcHeader.contains(name)
}

var products: [Product] = [
    .library(name: "SebbuCUDA", targets: ["SebbuCUDA"]),
    .library(name: "SebbuNVRTC", targets: ["SebbuNVRTC"]),
]
var targets: [Target] = []

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
    targets += [
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
            dependencies: ["SebbuCUDA", "SebbuNVRTC"],
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
    targets += [
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

if canBuildNVRTC, let nvrtcLibraryPath {
    var nvrtcCSettings: [CSetting] = [
        .unsafeFlags(["-isystem", nvrtcIncludePath]),
    ]
    let optionalDeclarations: [(String, String)] = [
        ("nvrtcGetNumSupportedArchs", "SEBBU_CNVRTC_HAS_SUPPORTED_ARCHS"),
        ("nvrtcGetCUBINSize", "SEBBU_CNVRTC_HAS_CUBIN"),
        ("nvrtcGetLTOIRSize", "SEBBU_CNVRTC_HAS_LTO_IR"),
        ("nvrtcGetOptiXIRSize", "SEBBU_CNVRTC_HAS_OPTIX_IR"),
        ("nvrtcGetPCHHeapSize", "SEBBU_CNVRTC_HAS_PCH"),
        ("nvrtcSetFlowCallback", "SEBBU_CNVRTC_HAS_FLOW_CALLBACK"),
        ("nvrtcGetTileIRSize", "SEBBU_CNVRTC_HAS_TILE_IR"),
        ("nvrtcGetBundledHeadersInfo", "SEBBU_CNVRTC_HAS_BUNDLED_HEADERS"),
    ]
    for (declaration, define) in optionalDeclarations
        where nvrtcHasDeclaration(declaration) {
        nvrtcCSettings.append(.define(define))
    }

    let nvrtcLinkerSettings: [LinkerSetting] = [
        .unsafeFlags(["-L", nvrtcLibraryPath]),
        .linkedLibrary("nvrtc"),
    ]

    targets += [
        .target(
            name: "CNVRTC",
            path: "Sources/CNVRTC",
            publicHeadersPath: "include",
            cSettings: nvrtcCSettings,
            linkerSettings: nvrtcLinkerSettings
        ),
        .target(
            name: "SebbuNVRTC",
            dependencies: ["CNVRTC"],
            path: "Sources/SebbuNVRTC",
            linkerSettings: nvrtcLinkerSettings
        ),
        .testTarget(
            name: "SebbuNVRTCTests",
            dependencies: ["SebbuNVRTC"],
            path: "Tests/SebbuNVRTCTests",
            linkerSettings: nvrtcLinkerSettings
        ),
    ]
} else {
    targets += [
        .target(
            name: "SebbuNVRTC",
            path: "Sources/SebbuNVRTCUnavailable"
        ),
        .testTarget(
            name: "SebbuNVRTCUnavailableTests",
            dependencies: ["SebbuNVRTC"],
            path: "Tests/SebbuNVRTCUnavailableTests"
        ),
    ]
}

let package = Package(
    name: "sebbu-cuda",
    products: products,
    targets: targets,
    swiftLanguageModes: [.v6]
)
