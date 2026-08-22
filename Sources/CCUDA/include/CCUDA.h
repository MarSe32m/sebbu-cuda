// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#ifndef SEBBU_CCUDA_H
#define SEBBU_CCUDA_H

#include <stddef.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
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

#endif
