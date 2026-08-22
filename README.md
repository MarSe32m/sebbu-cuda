# sebbu-cuda

`sebbu-cuda` is a Swift-native, ownership-safe wrapper around the NVIDIA CUDA
Driver API. The public product is `SebbuCUDA`; the raw `CCUDA` Clang module is
an implementation detail.

The first milestone covers driver initialization, devices, contexts, streams,
events, device and pinned memory, synchronous and asynchronous copies, module
loading, kernel lookup, argument packing, and kernel launch. The core module
does not import Foundation.

CUDA Driver entry points that `cuda.h` exposes through ABI-selection macros
such as `cuMemAlloc -> cuMemAlloc_v2` are called through stable `sebbuCu...`
functions in `CCUDA`. New macro-aliased Driver APIs should follow the same
pattern rather than being referenced directly from Swift.

## Requirements

- Swift 6.4 or newer
- Windows 10/11 x86-64 or Linux/WSL x86-64
- CUDA Toolkit headers and Driver API link library
- An NVIDIA driver and supported GPU to run CUDA work

On Windows, set `CUDA_PATH` to the toolkit root. A standard NVIDIA installer
normally sets it automatically:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
swift build
```

On Linux and WSL, the manifest uses `CUDA_PATH` when present and otherwise
checks `/usr/local/cuda`:

```bash
export CUDA_PATH=/usr/local/cuda
swift build
```

The system linker normally finds `libcuda` from the NVIDIA driver. For an
unusual sysroot or a toolkit stub directory, set `CUDA_LIBRARY_PATH` explicitly.
The package never installs or bundles a Linux kernel driver.

At runtime, Windows machines need the NVIDIA-provided `nvcuda.dll`; Linux/WSL
machines need the driver-provided `libcuda.so.1`. Applications do not ship
these driver files. PTX module loading does not require the CUDA runtime DLL or
`cudart` because this package uses the Driver API directly.

## Usage

Add the package and depend on the `SebbuCUDA` product. Users import only the
Swift module:

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

`Device`, `DevicePointer`, `Dim3`, versions and compute capabilities are
copyable values. `Context` and `Module` are reference-counted owners.
`Stream`, `Event`, `DeviceBuffer`, `DeviceMemory` and `PinnedBuffer` are
noncopyable resource values. Children retain their parents, and streams retain
the memory/module owners required by in-flight asynchronous work.

Zero-sized allocations are valid and use address zero without calling CUDA.
Negative counts and byte-count overflow are rejected before entering the
driver.

## Development executable and tests

The included executable JIT-loads a small PTX SAXPY kernel, launches it, checks
the result and records event timing:

```bash
swift run sebbu-cuda-development
swift test
```

Tests that require a CUDA device skip themselves when no usable driver/device
is present. Manifest evaluation still requires the toolkit's `cuda.h` and a
valid build-time Driver API link setup, and reports a focused diagnostic when
those prerequisites are missing.

## Scope

cuBLAS, cuFFT, cuSPARSE, NVRTC, graphs, virtual memory, peer access, and managed
memory are intentionally outside this first milestone. Future sibling targets
can use package-level opaque handles without exposing `CCUDA` to clients.

## License and NVIDIA software

`sebbu-cuda` is licensed under the [Apache License 2.0](LICENSE). This license
applies only to this package's own source code.

The NVIDIA CUDA Toolkit, CUDA headers, CUDA driver and associated libraries
are not included with `sebbu-cuda` and must be installed separately. They
remain subject to NVIDIA's own license terms, including the
[NVIDIA Software Development Kit License Agreement](https://docs.nvidia.com/cuda/eula/index.html).

NVIDIA and CUDA are trademarks or registered trademarks of NVIDIA Corporation.
This project is independently developed and is not affiliated with or endorsed
by NVIDIA.