// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#if os(Windows) || os(Linux)
#warning("sebbu-cuda: CUDA was not found; building an unavailable placeholder module.")
#endif

/// A placeholder that keeps the `SebbuCUDA` product importable when CUDA is
/// unavailable. The real API is present only in CUDA-enabled builds.
@available(*, unavailable, message: "CUDA is unavailable in this package build.")
public enum CUDA {}
