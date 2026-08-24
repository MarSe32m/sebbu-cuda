// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: Apache-2.0

@main
enum Development {
    static func main() throws {
        do { try saxpyExample() } 
        catch { print("Saxpy example failed with error:", error) }
        do { try daxpyExample() }
        catch { print("Daxpy example failed with error:", error) }
        do { try nvrtcExample() }
        catch { print("NVRTC exampled failed with error:", error) }
    }
}
