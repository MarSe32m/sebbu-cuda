# sebbu-cuda

`sebbu-cuda` provides Swift-native, ownership-safe wrappers around NVIDIA CUDA
components. Its public products are independent:

- `SebbuCUDA` wraps a focused subset of the CUDA Driver API.
- `SebbuNVRTC` wraps the complete exported NVRTC C API through CUDA 13.3 and
  links NVIDIA's shared NVRTC library.

The raw `CCUDA` and `CNVRTC` Clang modules are implementation details. Neither
appears in a client's public Swift API, and `SebbuNVRTC` does not depend on
`CCUDA` or `SebbuCUDA`.

The Driver milestone covers initialization, devices, contexts, streams,
events, device and pinned memory, synchronous and asynchronous copies, module
loading, kernel lookup, argument packing, and kernel launch. The NVRTC product
covers version and architecture queries, program and in-memory-header
lifecycle, compilation and diagnostics, PTX, cubin, LTO IR, OptiX IR, CUDA Tile
IR, lowered names, PCH management, cancellation callbacks, and bundled-header
management.

Both Swift modules avoid Foundation.

## Requirements

- Swift 6.2 or newer
- Windows 10/11 x86-64 or a CUDA-supported Linux architecture
- A CUDA Toolkit development installation
- For `SebbuCUDA`, CUDA Driver headers and the Driver API import/stub library
- For `SebbuNVRTC`, `nvrtc.h`, the NVRTC link library (`nvrtc.lib` on Windows
  or `libnvrtc.so` on Linux), and the corresponding NVRTC runtime libraries
- An NVIDIA driver and supported GPU only when running CUDA Driver work

NVRTC itself can compile on a machine without a CUDA-capable GPU or installed
NVIDIA driver.

On Windows, set `CUDA_PATH` to the toolkit root. A standard NVIDIA installer
normally sets it automatically and adds its `bin` directory to `PATH`:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
swift build
```

On Linux and WSL, the manifest uses `CUDA_PATH`, then `CUDA_HOME`, and otherwise
checks `/usr/local/cuda`:

```bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
swift build
```

For a nonstandard layout, set these independently:

- `CUDA_INCLUDE_PATH`: directory containing `cuda.h`
- `CUDA_LIBRARY_PATH`: directory containing the Driver import/stub library
- `NVRTC_INCLUDE_PATH`: directory containing `nvrtc.h`
- `NVRTC_LIBRARY_PATH`: directory containing `nvrtc.lib` or `libnvrtc.so`

When a component is unavailable, its public product remains importable as an
unavailable placeholder. This keeps CUDA-free macOS, Windows ARM, and static
Linux package graphs buildable.

## Runtime compilation

Add the package and depend on the `SebbuNVRTC` product:

```swift
import SebbuNVRTC

let source = #"""
extern "C" __global__ void scale(float *values, float factor) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    values[index] *= factor;
}
"""#

let ptx = try NVRTC.compile(
    source,
    architecture: .sm100,
    options: [.cxxStandard(.cxx20), .fastMath, .lineInfo]
)
```

The structured common-option model is also available:

```swift
var options = NVRTCCompileOptions(
    architecture: .sm100,
    cppStandard: .cxx20,
    fastMath: true,
    lineInfo: true,
    includePaths: ["/path/to/include"],
    defines: ["BLOCK_SIZE": "256", "ENABLE_FEATURE": nil]
)
options.additionalOptions.append("--future-nvrtc-option")

let ptx = try NVRTC.compile(source, options: options)
```

`CUDAArchitecture`, `CXXStandard`, `NVRTCCompileOption`, and
`NVRTCCompileOptions` are aliases for their `NVRTC`-scoped counterparts. An
architecture can be constructed from a future raw name or with
`.compute(_:suffix:)` and `.sm(_:suffix:)`; clients are not limited to the
convenience constants known by this release.

### Advanced program API

`Program` exposes the complete multi-step lifecycle:

```swift
let program = try NVRTC.Program(
    source: #"""
    #include "configuration.h"
    template<typename T>
    __global__ void templatedKernel(T *value) { *value += T(VALUE); }
    """#,
    name: "kernel.cu",
    headers: [
        .init(
            source: "#define VALUE 42\n",
            includeName: "configuration.h"
        )
    ]
)

try program.addNameExpression("templatedKernel<int>")
try program.compile(options: [
    .architecture(.sm100),
    .cxxStandard(.cxx20),
    .fastMath,
])

let ptx = try program.ptx()
let cubin = try program.cubin()
let diagnostics = try program.log()
let loweredName = try program.loweredName(for: "templatedKernel<int>")
```

Compilation failures throw `NVRTC.CompilationError`, which includes both the
forward-compatible NVRTC error code and the program's diagnostic log. Raw
compiler arguments remain available with `.raw(...)`.

The following output and management APIs are also exposed:

- `cubin()`, `ltoIR()`, `optiXIR()`, and `tileIR()` return `[UInt8]`.
- `setCancellationHandler(_:)` wraps `nvrtcSetFlowCallback` and retains its
  Swift closure for the program lifetime.
- `NVRTC.pchHeapSize`, `setPCHHeapSize(_:)`, `pchCreationStatus()`, and
  `requiredPCHHeapSize()` cover the CUDA 12.8 PCH API.
- `NVRTC.bundledHeadersInfo`, `installBundledHeaders(at:options:)`, and
  `removeBundledHeaders(at:)` cover CUDA 13.3 bundled headers. Removal is the
  native recursive operation and deletes every item in the supplied directory.
- `NVRTC.features` reports which optional API groups were present in the
  toolkit headers used to build the package. Calls absent from an older toolkit
  throw `NVRTC.UnsupportedFeatureError`.

The only documented NVRTC helper without a Swift wrapper is
`nvrtcGetTypeName`. It is not an exported C ABI function: it is an opt-in,
inline C++ host helper over `std::type_info` and `std::string`, and therefore has
no generic Swift ABI counterpart. Every exported NVRTC C function is wrapped.

## CUDA Driver usage

Add the package and depend on the `SebbuCUDA` product:

```swift
import SebbuCUDA

try CUDA.initialize()
let device = try CUDA.Device(ordinal: 0)
let context = try device.makeContext()

try context.withCurrent {
    let stream = try context.makeStream(flags: [.nonBlocking])
    let values: [Float] = [1, 2, 3, 4]
    let deviceValues: CUDA.DeviceBuffer<Float> = try context.allocate(
        count: values.count
    )

    try deviceValues.copy(from: values)
    try stream.synchronize()
}
```

`loadModule(atPath:)` accepts PTX, cubin, and fatbin files. A PTX string from
`SebbuNVRTC` can be passed directly to `context.loadModule(ptx:)` when an
application depends on both products.

Relative module paths are resolved from the process's current working
directory. PTX is JIT-compiled by the installed NVIDIA driver, while a fatbin
can contain code for multiple GPU architectures.

CUDA Driver entry points that `cuda.h` exposes through ABI-selection macros,
such as `cuMemAlloc -> cuMemAlloc_v2`, are called through stable `sebbuCu...`
functions in `CCUDA`. New macro-aliased Driver APIs should follow the same
pattern rather than being referenced directly from Swift.

`Device`, `DevicePointer`, `Dim3`, versions, and compute capabilities are
copyable values. `Context` and `Module` are reference-counted owners. `Stream`,
`Event`, `DeviceBuffer`, `DeviceMemory`, and `PinnedBuffer` are noncopyable
resource values. Children retain their parents, and streams retain the owners
needed by in-flight asynchronous work.

Zero-sized allocations are valid and use address zero without calling CUDA.
Negative counts and byte-count overflow are rejected before entering the
driver.

## Development and tests

```bash
swift run sebbu-cuda-development
swift test
```

NVRTC integration tests compile CUDA C++ without a GPU. Driver integration
tests skip when no usable driver/device is present. CI builds debug and release
configurations with CUDA 13.3 on Linux and Windows and also verifies CUDA-free
native and static-Linux configurations.

## NVRTC linkage and NVIDIA software

The package links `nvrtc.lib` on Windows and `libnvrtc.so` on Linux. The NVRTC
and NVRTC-builtins DLLs/shared libraries must therefore be discoverable at
runtime (normally through the standard CUDA Toolkit installation). The shared
variant is required on Windows because the official Swift runtime uses the
dynamic MSVC runtime (`/MD`), while NVIDIA's static NVRTC archive uses the
incompatible static MSVC runtime (`/MT`). `SebbuCUDA` applications also use the
driver-provided `nvcuda.dll` on Windows or `libcuda.so.1` on Linux.

`sebbu-cuda` is licensed under the [Apache License 2.0](LICENSE). This license
applies only to this package's source code.

The NVIDIA CUDA Toolkit, headers, libraries, driver, and associated
software are not included and remain subject to NVIDIA's license terms,
including the
[NVIDIA Software Development Kit License Agreement](https://docs.nvidia.com/cuda/eula/index.html).

NVIDIA and CUDA are trademarks or registered trademarks of NVIDIA Corporation.
This project is independently developed and is not affiliated with or endorsed
by NVIDIA.
