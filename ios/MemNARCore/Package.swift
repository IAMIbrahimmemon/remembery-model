// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MemNARCore",
    platforms: [.iOS("18.0"), .macOS("15.0")],
    products: [
        .library(name: "MemNARCore", targets: ["MemNARCore"]),
        .executable(name: "memnar-cli", targets: ["memnar-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/pytorch/executorch", branch: "swiftpm-1.5.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.5"),
    ],
    targets: [
        .target(
            name: "MemNARCore",
            dependencies: [
                .product(name: "executorch", package: "executorch"),
                .product(name: "backend_xnnpack", package: "executorch"),
                .product(name: "kernels_optimized", package: "executorch"),
                .product(name: "kernels_quantized", package: "executorch"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "memnar-cli",
            dependencies: ["MemNARCore"],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-all_load"])]
        ),
    ]
)
