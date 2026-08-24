// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0
 
import SebbuCUDA

// This example loads the kernel from a precomplied fatbin located at kernels/kernels.fatbin
func daxpyExample() throws {
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