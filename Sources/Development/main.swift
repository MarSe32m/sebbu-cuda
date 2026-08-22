import SebbuCUDA

private let saxpyPTX = #"""
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

@main
enum Development {
    static func testDaxpy() throws {
        try CUDA.initialize()

        let device = try CUDA.Device(ordinal: 0)
        print("Using \(try device.name), compute capability \(try device.computeCapability)")
        print("CUDA driver \(try CUDA.driverVersion)")
        print("Memory \(try device.totalMemory)")

        let context = try device.makeContext()
        try context.withCurrent {
            let stream = try context.makeStream(flags: [.nonBlocking])
            let start = try context.makeEvent()
            let end = try context.makeEvent()

            let n = 1024
            let hostX = (0..<n).map { Double($0) * 0.25 }
            let originalY = (0..<n).map { Double($0) * -0.5 }
            var hostY = originalY

            let x: CUDA.DeviceBuffer<Double> = try context.allocate(count: n)
            let y: CUDA.DeviceBuffer<Double> = try context.allocate(count: n)
            try x.copy(from: hostX)
            try y.copy(from: hostY)

            let module = try context.loadModule(atPath: "kernels/kernels.fatbin")
            let kernel = try module.kernel(named: "daxpy")
            let scale: Double = 2

            try start.record(on: stream)
            try kernel.launch(
                grid: .init(x: UInt32((n + 255) / 256)),
                block: .init(x: 256),
                on: stream
            ) { arguments in
                arguments.append(x.pointer)
                arguments.append(y.pointer)
                arguments.append(scale)
                arguments.append(Int32(n))
            }
            try end.record(on: stream)
            try stream.synchronize()
            try y.copy(to: &hostY)

            for index in hostY.indices {
                let expected = scale * hostX[index] + originalY[index]
                precondition(
                    abs(hostY[index] - expected) < 0.0001,
                    "SAXPY verification failed at index \(index)."
                )
            }

            print("DAXPY verified for \(n) elements")
            print("Kernel duration: \(try end.elapsedTime(since: start))")
        }
    }

    static func testSaxpy() throws {
        try CUDA.initialize()

        let device = try CUDA.Device(ordinal: 0)
        print("Using \(try device.name), compute capability \(try device.computeCapability)")
        print("CUDA driver \(try CUDA.driverVersion)")
        print("Memory \(try device.totalMemory)")

        let context = try device.makeContext()
        try context.withCurrent {
            let stream = try context.makeStream(flags: [.nonBlocking])
            let start = try context.makeEvent()
            let end = try context.makeEvent()

            let n = 1024
            let hostX = (0..<n).map { Float($0) * 0.25 }
            let originalY = (0..<n).map { Float($0) * -0.5 }
            var hostY = originalY

            let x: CUDA.DeviceBuffer<Float> = try context.allocate(count: n)
            let y: CUDA.DeviceBuffer<Float> = try context.allocate(count: n)
            try x.copy(from: hostX)
            try y.copy(from: hostY)

            let module = try context.loadModule(ptx: saxpyPTX)
            let kernel = try module.kernel(named: "saxpy")
            let scale: Float = 2

            try start.record(on: stream)
            try kernel.launch(
                grid: .init(x: UInt32((n + 255) / 256)),
                block: .init(x: 256),
                on: stream
            ) { arguments in
                arguments.append(x.pointer)
                arguments.append(y.pointer)
                arguments.append(scale)
                arguments.append(Int32(n))
            }
            try end.record(on: stream)
            try stream.synchronize()
            try y.copy(to: &hostY)

            for index in hostY.indices {
                let expected = scale * hostX[index] + originalY[index]
                precondition(
                    abs(hostY[index] - expected) < 0.0001,
                    "SAXPY verification failed at index \(index)."
                )
            }

            print("SAXPY verified for \(n) elements")
            print("Kernel duration: \(try end.elapsedTime(since: start))")
        }
    }

    static func main() throws {
        do {
            try testSaxpy()
            try testDaxpy()
        } catch {
            print("Test failed with error:", error)
        }
    }
}
