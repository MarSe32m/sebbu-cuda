// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#ifndef SEBBU_CNVRTC_H
#define SEBBU_CNVRTC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Keep nvrtc.h private to CNVRTC.cpp. SwiftPM may compile a dependent Swift
 * target without inheriting this target's toolkit header search path. This
 * small C ABI also prevents NVIDIA enums and opaque implementation types from
 * becoming part of SebbuNVRTC's public Swift interface.
 */
typedef void *SebbuNVRTCProgram;
typedef int (*SebbuNVRTCFlowCallback)(void *payload, void *reserved);

typedef struct SebbuNVRTCBundledHeadersInfo {
    int available;
    size_t compressedSize;
    size_t uncompressedSize;
    int cudaVersionMajor;
    int cudaVersionMinor;
    unsigned int numFiles;
} SebbuNVRTCBundledHeadersInfo;

enum {
    SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT = -1
};

enum {
    SEBBU_NVRTC_FEATURE_SUPPORTED_ARCHS = UINT64_C(1) << 0,
    SEBBU_NVRTC_FEATURE_CUBIN = UINT64_C(1) << 1,
    SEBBU_NVRTC_FEATURE_LTO_IR = UINT64_C(1) << 2,
    SEBBU_NVRTC_FEATURE_OPTIX_IR = UINT64_C(1) << 3,
    SEBBU_NVRTC_FEATURE_PCH = UINT64_C(1) << 4,
    SEBBU_NVRTC_FEATURE_FLOW_CALLBACK = UINT64_C(1) << 5,
    SEBBU_NVRTC_FEATURE_TILE_IR = UINT64_C(1) << 6,
    SEBBU_NVRTC_FEATURE_BUNDLED_HEADERS = UINT64_C(1) << 7
};

const char *sebbuNvrtcGetErrorString(int result);
uint64_t sebbuNvrtcFeatureMask(void);

int sebbuNvrtcVersion(int *major, int *minor);
int sebbuNvrtcGetNumSupportedArchs(int *count);
int sebbuNvrtcGetSupportedArchs(int *architectures);

int sebbuNvrtcCreateProgram(
    SebbuNVRTCProgram *program,
    const char *source,
    const char *name,
    int headerCount,
    const char *const *headers,
    const char *const *includeNames
);
int sebbuNvrtcDestroyProgram(SebbuNVRTCProgram *program);
int sebbuNvrtcCompileProgram(
    SebbuNVRTCProgram program,
    int optionCount,
    const char *const *options
);

int sebbuNvrtcGetPTXSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetPTX(SebbuNVRTCProgram program, char *ptx);
int sebbuNvrtcGetCUBINSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetCUBIN(SebbuNVRTCProgram program, char *cubin);
int sebbuNvrtcGetLTOIRSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetLTOIR(SebbuNVRTCProgram program, char *ltoIR);
int sebbuNvrtcGetOptiXIRSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetOptiXIR(SebbuNVRTCProgram program, char *optiXIR);
int sebbuNvrtcGetTileIRSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetTileIR(SebbuNVRTCProgram program, char *tileIR);

int sebbuNvrtcGetProgramLogSize(SebbuNVRTCProgram program, size_t *size);
int sebbuNvrtcGetProgramLog(SebbuNVRTCProgram program, char *log);
int sebbuNvrtcAddNameExpression(
    SebbuNVRTCProgram program,
    const char *expression
);
int sebbuNvrtcGetLoweredName(
    SebbuNVRTCProgram program,
    const char *expression,
    const char **loweredName
);

int sebbuNvrtcGetPCHHeapSize(size_t *size);
int sebbuNvrtcSetPCHHeapSize(size_t size);
int sebbuNvrtcGetPCHCreateStatus(SebbuNVRTCProgram program);
int sebbuNvrtcGetPCHHeapSizeRequired(
    SebbuNVRTCProgram program,
    size_t *size
);
int sebbuNvrtcSetFlowCallback(
    SebbuNVRTCProgram program,
    SebbuNVRTCFlowCallback callback,
    void *payload
);

int sebbuNvrtcGetBundledHeadersInfo(
    SebbuNVRTCBundledHeadersInfo *info,
    const char **errorLog
);
int sebbuNvrtcInstallBundledHeaders(
    const char *installPath,
    unsigned int flags,
    const char **errorLog
);
int sebbuNvrtcRemoveBundledHeaders(
    const char *installPath,
    const char **errorLog
);

#ifdef __cplusplus
}
#endif

#endif
