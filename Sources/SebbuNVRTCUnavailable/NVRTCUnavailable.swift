// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

#if os(Windows) || os(Linux)
#warning("sebbu-cuda: the NVRTC development library was not found; building an unavailable placeholder module.")
#endif

/// A placeholder that keeps the `SebbuNVRTC` product importable when the
/// NVRTC development package and shared library are unavailable.
@available(*, unavailable, message: "NVRTC is unavailable in this package build.")
public enum NVRTC {}
