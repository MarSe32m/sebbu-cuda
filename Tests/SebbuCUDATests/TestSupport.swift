// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

import XCTest
import SebbuCUDA

let testSAXPYPTX = #"""
.version 6.4
.target sm_50
.address_size 64

.visible .entry saxpy(
    .param .u64 saxpy_x,
    .param .u64 saxpy_y,
    .param .f32 saxpy_a,
    .param .u32 saxpy_n
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [saxpy_x];
    ld.param.u64 %rd2, [saxpy_y];
    ld.param.f32 %f1, [saxpy_a];
    ld.param.u32 %r5, [saxpy_n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra DONE;
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f3, [%rd5];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd5], %f4;
DONE:
    ret;
}
"""#

func cudaDeviceOrSkip() throws -> CUDA.Device {
    do {
        try CUDA.initialize()
        guard try CUDA.Device.count > 0 else {
            throw XCTSkip("No CUDA device is visible to the driver.")
        }
        return try CUDA.Device(ordinal: 0)
    } catch {
        throw XCTSkip("CUDA integration test unavailable: \(error)")
    }
}
