// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#include "CNVRTC.h"

#include <nvrtc.h>

namespace {

nvrtcProgram unwrap(SebbuNVRTCProgram program) {
    return reinterpret_cast<nvrtcProgram>(program);
}

int raw(nvrtcResult result) {
    return static_cast<int>(result);
}

} // namespace

const char *sebbuNvrtcGetErrorString(int result) {
    if (result == SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT) {
        return "SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT";
    }
    return nvrtcGetErrorString(static_cast<nvrtcResult>(result));
}

uint64_t sebbuNvrtcFeatureMask(void) {
    uint64_t features = 0;
#if defined(SEBBU_CNVRTC_HAS_SUPPORTED_ARCHS)
    features |= SEBBU_NVRTC_FEATURE_SUPPORTED_ARCHS;
#endif
#if defined(SEBBU_CNVRTC_HAS_CUBIN)
    features |= SEBBU_NVRTC_FEATURE_CUBIN;
#endif
#if defined(SEBBU_CNVRTC_HAS_LTO_IR)
    features |= SEBBU_NVRTC_FEATURE_LTO_IR;
#endif
#if defined(SEBBU_CNVRTC_HAS_OPTIX_IR)
    features |= SEBBU_NVRTC_FEATURE_OPTIX_IR;
#endif
#if defined(SEBBU_CNVRTC_HAS_PCH)
    features |= SEBBU_NVRTC_FEATURE_PCH;
#endif
#if defined(SEBBU_CNVRTC_HAS_FLOW_CALLBACK)
    features |= SEBBU_NVRTC_FEATURE_FLOW_CALLBACK;
#endif
#if defined(SEBBU_CNVRTC_HAS_TILE_IR)
    features |= SEBBU_NVRTC_FEATURE_TILE_IR;
#endif
#if defined(SEBBU_CNVRTC_HAS_BUNDLED_HEADERS)
    features |= SEBBU_NVRTC_FEATURE_BUNDLED_HEADERS;
#endif
    return features;
}

int sebbuNvrtcVersion(int *major, int *minor) {
    return raw(nvrtcVersion(major, minor));
}

int sebbuNvrtcGetNumSupportedArchs(int *count) {
#if defined(SEBBU_CNVRTC_HAS_SUPPORTED_ARCHS)
    return raw(nvrtcGetNumSupportedArchs(count));
#else
    (void)count;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetSupportedArchs(int *architectures) {
#if defined(SEBBU_CNVRTC_HAS_SUPPORTED_ARCHS)
    return raw(nvrtcGetSupportedArchs(architectures));
#else
    (void)architectures;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcCreateProgram(
    SebbuNVRTCProgram *program,
    const char *source,
    const char *name,
    int headerCount,
    const char *const *headers,
    const char *const *includeNames
) {
    nvrtcProgram value = nullptr;
    const nvrtcResult result = nvrtcCreateProgram(
        &value,
        source,
        name,
        headerCount,
        const_cast<const char **>(headers),
        const_cast<const char **>(includeNames)
    );
    if (program != nullptr) {
        *program = reinterpret_cast<SebbuNVRTCProgram>(value);
    }
    return raw(result);
}

int sebbuNvrtcDestroyProgram(SebbuNVRTCProgram *program) {
    if (program == nullptr) {
        return raw(NVRTC_ERROR_INVALID_INPUT);
    }
    nvrtcProgram value = unwrap(*program);
    const nvrtcResult result = nvrtcDestroyProgram(&value);
    *program = reinterpret_cast<SebbuNVRTCProgram>(value);
    return raw(result);
}

int sebbuNvrtcCompileProgram(
    SebbuNVRTCProgram program,
    int optionCount,
    const char *const *options
) {
    return raw(nvrtcCompileProgram(
        unwrap(program),
        optionCount,
        const_cast<const char **>(options)
    ));
}

int sebbuNvrtcGetPTXSize(SebbuNVRTCProgram program, size_t *size) {
    return raw(nvrtcGetPTXSize(unwrap(program), size));
}

int sebbuNvrtcGetPTX(SebbuNVRTCProgram program, char *ptx) {
    return raw(nvrtcGetPTX(unwrap(program), ptx));
}

int sebbuNvrtcGetCUBINSize(SebbuNVRTCProgram program, size_t *size) {
#if defined(SEBBU_CNVRTC_HAS_CUBIN)
    return raw(nvrtcGetCUBINSize(unwrap(program), size));
#else
    (void)program;
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetCUBIN(SebbuNVRTCProgram program, char *cubin) {
#if defined(SEBBU_CNVRTC_HAS_CUBIN)
    return raw(nvrtcGetCUBIN(unwrap(program), cubin));
#else
    (void)program;
    (void)cubin;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetLTOIRSize(SebbuNVRTCProgram program, size_t *size) {
#if defined(SEBBU_CNVRTC_HAS_LTO_IR)
    return raw(nvrtcGetLTOIRSize(unwrap(program), size));
#else
    (void)program;
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetLTOIR(SebbuNVRTCProgram program, char *ltoIR) {
#if defined(SEBBU_CNVRTC_HAS_LTO_IR)
    return raw(nvrtcGetLTOIR(unwrap(program), ltoIR));
#else
    (void)program;
    (void)ltoIR;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetOptiXIRSize(SebbuNVRTCProgram program, size_t *size) {
#if defined(SEBBU_CNVRTC_HAS_OPTIX_IR)
    return raw(nvrtcGetOptiXIRSize(unwrap(program), size));
#else
    (void)program;
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetOptiXIR(SebbuNVRTCProgram program, char *optiXIR) {
#if defined(SEBBU_CNVRTC_HAS_OPTIX_IR)
    return raw(nvrtcGetOptiXIR(unwrap(program), optiXIR));
#else
    (void)program;
    (void)optiXIR;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetTileIRSize(SebbuNVRTCProgram program, size_t *size) {
#if defined(SEBBU_CNVRTC_HAS_TILE_IR)
    return raw(nvrtcGetTileIRSize(unwrap(program), size));
#else
    (void)program;
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetTileIR(SebbuNVRTCProgram program, char *tileIR) {
#if defined(SEBBU_CNVRTC_HAS_TILE_IR)
    return raw(nvrtcGetTileIR(unwrap(program), tileIR));
#else
    (void)program;
    (void)tileIR;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetProgramLogSize(SebbuNVRTCProgram program, size_t *size) {
    return raw(nvrtcGetProgramLogSize(unwrap(program), size));
}

int sebbuNvrtcGetProgramLog(SebbuNVRTCProgram program, char *log) {
    return raw(nvrtcGetProgramLog(unwrap(program), log));
}

int sebbuNvrtcAddNameExpression(
    SebbuNVRTCProgram program,
    const char *expression
) {
    return raw(nvrtcAddNameExpression(unwrap(program), expression));
}

int sebbuNvrtcGetLoweredName(
    SebbuNVRTCProgram program,
    const char *expression,
    const char **loweredName
) {
    return raw(nvrtcGetLoweredName(
        unwrap(program),
        expression,
        loweredName
    ));
}

int sebbuNvrtcGetPCHHeapSize(size_t *size) {
#if defined(SEBBU_CNVRTC_HAS_PCH)
    return raw(nvrtcGetPCHHeapSize(size));
#else
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcSetPCHHeapSize(size_t size) {
#if defined(SEBBU_CNVRTC_HAS_PCH)
    return raw(nvrtcSetPCHHeapSize(size));
#else
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetPCHCreateStatus(SebbuNVRTCProgram program) {
#if defined(SEBBU_CNVRTC_HAS_PCH)
    return raw(nvrtcGetPCHCreateStatus(unwrap(program)));
#else
    (void)program;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetPCHHeapSizeRequired(
    SebbuNVRTCProgram program,
    size_t *size
) {
#if defined(SEBBU_CNVRTC_HAS_PCH)
    return raw(nvrtcGetPCHHeapSizeRequired(unwrap(program), size));
#else
    (void)program;
    (void)size;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcSetFlowCallback(
    SebbuNVRTCProgram program,
    SebbuNVRTCFlowCallback callback,
    void *payload
) {
#if defined(SEBBU_CNVRTC_HAS_FLOW_CALLBACK)
    return raw(nvrtcSetFlowCallback(unwrap(program), callback, payload));
#else
    (void)program;
    (void)callback;
    (void)payload;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcGetBundledHeadersInfo(
    SebbuNVRTCBundledHeadersInfo *info,
    const char **errorLog
) {
#if defined(SEBBU_CNVRTC_HAS_BUNDLED_HEADERS)
    if (info == nullptr) {
        return raw(NVRTC_ERROR_INVALID_INPUT);
    }
    nvrtcBundledHeadersInfo value = {};
    const nvrtcResult result = nvrtcGetBundledHeadersInfo(&value, errorLog);
    info->available = value.available;
    info->compressedSize = value.compressedSize;
    info->uncompressedSize = value.uncompressedSize;
    info->cudaVersionMajor = value.cudaVersionMajor;
    info->cudaVersionMinor = value.cudaVersionMinor;
    info->numFiles = value.numFiles;
    return raw(result);
#else
    (void)info;
    (void)errorLog;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcInstallBundledHeaders(
    const char *installPath,
    unsigned int flags,
    const char **errorLog
) {
#if defined(SEBBU_CNVRTC_HAS_BUNDLED_HEADERS)
    return raw(nvrtcInstallBundledHeaders(installPath, flags, errorLog));
#else
    (void)installPath;
    (void)flags;
    (void)errorLog;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}

int sebbuNvrtcRemoveBundledHeaders(
    const char *installPath,
    const char **errorLog
) {
#if defined(SEBBU_CNVRTC_HAS_BUNDLED_HEADERS)
    return raw(nvrtcRemoveBundledHeaders(installPath, errorLog));
#else
    (void)installPath;
    (void)errorLog;
    return SEBBU_NVRTC_ERROR_UNSUPPORTED_BY_TOOLKIT;
#endif
}
