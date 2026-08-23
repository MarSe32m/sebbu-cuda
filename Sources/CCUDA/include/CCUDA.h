// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#ifndef SEBBU_CCUDA_H
#define SEBBU_CCUDA_H

#include <stddef.h>

/*
 * Keep cuda.h private to CCUDA.c. SwiftPM can compile dependent and generated
 * Swift targets without inheriting CCUDA's target-local header search path.
 * The declarations below are the small, stable Driver API ABI surface used by
 * SebbuCUDA; their types and calling convention match NVIDIA's 64-bit ABI.
 */
#if defined(SEBBU_CCUDA_USE_TOOLKIT_HEADER)
#include <cuda.h>
#else
#include <stdint.h>

typedef int CUdevice;
typedef uint64_t CUdeviceptr;

typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;

typedef enum CUresult_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

#if defined(_WIN32)
#define SEBBU_CUDAAPI __stdcall
#else
#define SEBBU_CUDAAPI
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(SEBBU_CCUDA_USE_TOOLKIT_HEADER)
CUresult SEBBU_CUDAAPI cuInit(unsigned int flags);
CUresult SEBBU_CUDAAPI cuDriverGetVersion(int *driverVersion);
CUresult SEBBU_CUDAAPI cuGetErrorName(
    CUresult error,
    const char **name
);
CUresult SEBBU_CUDAAPI cuGetErrorString(
    CUresult error,
    const char **description
);

CUresult SEBBU_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult SEBBU_CUDAAPI cuDeviceGetCount(int *count);
CUresult SEBBU_CUDAAPI cuDeviceGetName(
    char *name,
    int length,
    CUdevice device
);
CUresult SEBBU_CUDAAPI cuDevicePrimaryCtxRetain(
    CUcontext *context,
    CUdevice device
);

CUresult SEBBU_CUDAAPI cuCtxGetCurrent(CUcontext *context);

CUresult SEBBU_CUDAAPI cuStreamCreate(
    CUstream *stream,
    unsigned int flags
);
CUresult SEBBU_CUDAAPI cuStreamSynchronize(CUstream stream);

CUresult SEBBU_CUDAAPI cuEventCreate(
    CUevent *event,
    unsigned int flags
);
CUresult SEBBU_CUDAAPI cuEventRecord(CUevent event, CUstream stream);
CUresult SEBBU_CUDAAPI cuEventSynchronize(CUevent event);
CUresult SEBBU_CUDAAPI cuEventElapsedTime(
    float *milliseconds,
    CUevent start,
    CUevent end
);

CUresult SEBBU_CUDAAPI cuMemHostAlloc(
    void **pointer,
    size_t byteCount,
    unsigned int flags
);
CUresult SEBBU_CUDAAPI cuMemFreeHost(void *pointer);

CUresult SEBBU_CUDAAPI cuModuleLoad(
    CUmodule *module,
    const char *path
);
CUresult SEBBU_CUDAAPI cuModuleLoadData(
    CUmodule *module,
    const void *image
);
CUresult SEBBU_CUDAAPI cuModuleUnload(CUmodule module);
CUresult SEBBU_CUDAAPI cuModuleGetFunction(
    CUfunction *function,
    CUmodule module,
    const char *name
);

CUresult SEBBU_CUDAAPI cuLaunchKernel(
    CUfunction function,
    unsigned int gridX,
    unsigned int gridY,
    unsigned int gridZ,
    unsigned int blockX,
    unsigned int blockY,
    unsigned int blockZ,
    unsigned int sharedMemoryByteCount,
    CUstream stream,
    void **kernelParameters,
    void **extra
);
#endif

/*
 * CUDA 13 added CUctxCreateParams to cuCtxCreate. Keep that source-level API
 * change inside the raw implementation target so SebbuCUDA can retain one
 * stable context factory across supported toolkit versions.
 */
CUresult sebbuCuCtxCreate(
    CUcontext *context,
    unsigned int flags,
    CUdevice device
);

/*
 * cuda.h exposes these source-level names as macros selecting versioned ABI
 * symbols (for example, cuMemAlloc -> cuMemAlloc_v2). Clang imports the target
 * declaration but does not reliably import the macro alias as a Swift
 * function. These stable C entry points keep that detail out of Swift.
 */
CUresult sebbuCuDeviceTotalMem(size_t *bytes, CUdevice device);

CUresult sebbuCuCtxDestroy(CUcontext context);
CUresult sebbuCuDevicePrimaryCtxRelease(CUdevice device);
CUresult sebbuCuCtxPushCurrent(CUcontext context);
CUresult sebbuCuCtxPopCurrent(CUcontext *context);

CUresult sebbuCuStreamDestroy(CUstream stream);
CUresult sebbuCuEventDestroy(CUevent event);

CUresult sebbuCuMemAlloc(CUdeviceptr *pointer, size_t byteCount);
CUresult sebbuCuMemFree(CUdeviceptr pointer);
CUresult sebbuCuMemcpyHtoD(
    CUdeviceptr destination,
    const void *source,
    size_t byteCount
);
CUresult sebbuCuMemcpyDtoH(
    void *destination,
    CUdeviceptr source,
    size_t byteCount
);
CUresult sebbuCuMemcpyHtoDAsync(
    CUdeviceptr destination,
    const void *source,
    size_t byteCount,
    CUstream stream
);
CUresult sebbuCuMemcpyDtoHAsync(
    void *destination,
    CUdeviceptr source,
    size_t byteCount,
    CUstream stream
);
CUresult sebbuCuEventElapsedTime(
    float *pMilliseconds,
    CUevent hStart,
    CUevent hEnd
);

/* Swift should not need to depend on the imported enum's raw-value type. */
CUresult sebbuCuDeviceGetAttribute(
    int *value,
    int attribute,
    CUdevice device
);

int sebbuCuResultRawValue(CUresult result);

#ifdef __cplusplus
}
#endif

#if !defined(SEBBU_CCUDA_USE_TOOLKIT_HEADER)
#undef SEBBU_CUDAAPI
#endif

#endif
