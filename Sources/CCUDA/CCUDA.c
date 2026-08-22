#include "CCUDA.h"

CUresult sebbuCuCtxCreate(
    CUcontext *context,
    unsigned int flags,
    CUdevice device
) {
#if CUDA_VERSION >= 13000
    return cuCtxCreate(context, NULL, flags, device);
#else
    return cuCtxCreate(context, flags, device);
#endif
}

CUresult sebbuCuDeviceTotalMem(size_t *bytes, CUdevice device) {
    return cuDeviceTotalMem(bytes, device);
}

CUresult sebbuCuCtxDestroy(CUcontext context) {
    return cuCtxDestroy(context);
}

CUresult sebbuCuDevicePrimaryCtxRelease(CUdevice device) {
    return cuDevicePrimaryCtxRelease(device);
}

CUresult sebbuCuCtxPushCurrent(CUcontext context) {
    return cuCtxPushCurrent(context);
}

CUresult sebbuCuCtxPopCurrent(CUcontext *context) {
    return cuCtxPopCurrent(context);
}

CUresult sebbuCuStreamDestroy(CUstream stream) {
    return cuStreamDestroy(stream);
}

CUresult sebbuCuEventDestroy(CUevent event) {
    return cuEventDestroy(event);
}

CUresult sebbuCuMemAlloc(CUdeviceptr *pointer, size_t byteCount) {
    return cuMemAlloc(pointer, byteCount);
}

CUresult sebbuCuMemFree(CUdeviceptr pointer) {
    return cuMemFree(pointer);
}

CUresult sebbuCuMemcpyHtoD(
    CUdeviceptr destination,
    const void *source,
    size_t byteCount
) {
    return cuMemcpyHtoD(destination, source, byteCount);
}

CUresult sebbuCuMemcpyDtoH(
    void *destination,
    CUdeviceptr source,
    size_t byteCount
) {
    return cuMemcpyDtoH(destination, source, byteCount);
}

CUresult sebbuCuMemcpyHtoDAsync(
    CUdeviceptr destination,
    const void *source,
    size_t byteCount,
    CUstream stream
) {
    return cuMemcpyHtoDAsync(destination, source, byteCount, stream);
}

CUresult sebbuCuMemcpyDtoHAsync(
    void *destination,
    CUdeviceptr source,
    size_t byteCount,
    CUstream stream
) {
    return cuMemcpyDtoHAsync(destination, source, byteCount, stream);
}

CUresult sebbuCuEventElapsedTime(
    float *pMilliseconds, 
    CUevent hStart, 
    CUevent hEnd
) {
    return cuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

CUresult sebbuCuDeviceGetAttribute(
    int *value,
    int attribute,
    CUdevice device
) {
    return cuDeviceGetAttribute(value, (CUdevice_attribute)attribute, device);
}

int sebbuCuResultRawValue(CUresult result) {
    return (int)result;
}