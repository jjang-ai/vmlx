// swift-tools-version: 5.12
import PackageDescription

// vMLX — self-contained Apple Silicon LLM + VLM + image/video workstation.
//
// Single unified package. Everything lives in this one SwiftPM package tree:
//
//   • The MLX runtime (Cmlx, MLX, MLXNN, MLXFast, MLXFFT, MLXLinalg,
//     MLXOptimizers, MLXRandom) — vendored from `ml-explore/mlx-swift`
//     branch `osaurus-0.31.3`, with our `vmlx-patches-0.31.3` kernel
//     patches on top (GatherQMM output_shapes, scalar-zero DType fusion,
//     JANGTQ P1-P12 optimization stack). Formerly lived at
//     the pre-merge fork as a separate SwiftPM package; merged
//     into this manifest 2026-04-13 to give us one cohesive `vmlx` tree.
//
//   • The model library (vMLXLMCommon, vMLXLLM, vMLXVLM, vMLXEmbedders)
//     — vendored from vmlx-swift-lm.
//
//   • The image/video engine (vMLXFlux, vMLXFluxKit, vMLXFluxModels,
//     vMLXFluxVideo) — vendored from vmlx-flux.
//
//   • The engine adapter + app shell (vMLXEngine, vMLXServer, vMLXTheme,
//     vMLXCLI, vMLXApp) — vMLX-native, wraps the vendored factories with
//     JANG repack, prefix cache, SSM companion, reasoning suppression,
//     tool-call dispatch, ModelLibrary, SettingsStore, LogStore,
//     MetricsCollector, Hummingbird routes, SwiftUI chrome.
//
// JANGTQ kernel optimization work (compile-G fusion, decode fusion,
// subgroup matrix instructions, P1-P12 stack) lands DIRECTLY in the
// `Sources/Cmlx/mlx-generated/metal/` tree. No upstream round-trip.
//
// User-visible naming is `vmlx` / `vMLX` everywhere.

// Platform-specific build settings for the `Cmlx` target. Merged from
// `vmlx-swift/Package.swift` on 2026-04-13. Apple-Silicon path (#else)
// is the only branch that matters for shipping vMLX today; Linux path
// is kept for reference so future CI against a non-Apple runner stays
// a small delta.
#if os(Linux)
    let platformExcludes: [String] = [
        "framework",
        "include-framework",
        "metal-cpp",
        "mlx/mlx/backend/metal/allocator.cpp",
        "mlx/mlx/backend/metal/binary.cpp",
        "mlx/mlx/backend/metal/compiled.cpp",
        "mlx/mlx/backend/metal/conv.cpp",
        "mlx/mlx/backend/metal/copy.cpp",
        "mlx/mlx/backend/metal/custom_kernel.cpp",
        "mlx/mlx/backend/metal/device.cpp",
        "mlx/mlx/backend/metal/device_info.cpp",
        "mlx/mlx/backend/metal/distributed.cpp",
        "mlx/mlx/backend/metal/eval.cpp",
        "mlx/mlx/backend/metal/event.cpp",
        "mlx/mlx/backend/metal/fence.cpp",
        "mlx/mlx/backend/metal/fft.cpp",
        "mlx/mlx/backend/metal/hadamard.cpp",
        "mlx/mlx/backend/metal/indexing.cpp",
        "mlx/mlx/backend/metal/jit_kernels.cpp",
        "mlx/mlx/backend/metal/logsumexp.cpp",
        "mlx/mlx/backend/metal/matmul.cpp",
        "mlx/mlx/backend/metal/metal.cpp",
        "mlx/mlx/backend/metal/normalization.cpp",
        "mlx/mlx/backend/metal/primitives.cpp",
        "mlx/mlx/backend/metal/quantized.cpp",
        "mlx/mlx/backend/metal/reduce.cpp",
        "mlx/mlx/backend/metal/resident.cpp",
        "mlx/mlx/backend/metal/rope.cpp",
        "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/metal/scan.cpp",
        "mlx/mlx/backend/metal/slicing.cpp",
        "mlx/mlx/backend/metal/softmax.cpp",
        "mlx/mlx/backend/metal/sort.cpp",
        "mlx/mlx/backend/metal/ternary.cpp",
        "mlx/mlx/backend/metal/unary.cpp",
        "mlx/mlx/backend/metal/utils.cpp",
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/jit",
        "mlx/mlx/backend/gpu",
        "mlx/mlx/backend/no_cpu",
        "mlx/mlx/backend/cpu/gemms/bnns.cpp",
        "mlx-conditional",
        "mlx-c/mlx/c/metal.cpp",
        "mlx-c/mlx/c/fast.cpp",
    ]
    let cxxSettings: [CXXSetting] = []
    let linkerSettings: [LinkerSetting] = [
        .linkedLibrary("gfortran", .when(platforms: [.linux])),
        .linkedLibrary("blas", .when(platforms: [.linux])),
        .linkedLibrary("lapack", .when(platforms: [.linux])),
        .linkedLibrary("openblas", .when(platforms: [.linux])),
    ]
    let mlxSwiftExcludes: [String] = [
        "GPU+Metal.swift",
        "MLXArray+Metal.swift",
        "MLXFast.swift",
        "MLXFastKernel.swift",
    ]
#else
    let platformExcludes: [String] = [
        "mlx/mlx/backend/cpu/compiled.cpp",
        "mlx/mlx/backend/no_gpu",
        "mlx/mlx/backend/no_cpu",
        "mlx/mlx/backend/metal/no_metal.cpp",
        "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
        "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp",
    ]
    let cxxSettings: [CXXSetting] = [
        .headerSearchPath("metal-cpp"),
        .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"),
        .define("_METAL_"),
        // Bundle identity must match what mlx-swift's device.cpp
        // `load_swiftpm_library` looks up. It hard-codes the name
        // `mlx-swift_Cmlx` — if we renamed the bundle on the Swift side
        // we'd break Metal kernel loading at runtime. Leave as-is.
        .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
    ]
    let linkerSettings: [LinkerSetting] = [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate"),
    ]
    let mlxSwiftExcludes: [String] = []
#endif

let cmlx = Target.target(
    name: "Cmlx",
    path: "Sources/Cmlx",
    exclude: platformExcludes + [
        "vendor-README.md",
        "mlx-c/examples",
        "mlx-c/mlx/c/distributed.cpp",
        "mlx-c/mlx/c/distributed_group.cpp",
        "json",
        "fmt/test",
        "fmt/doc",
        "fmt/support",
        "fmt/src/os.cc",
        "fmt/src/fmt.cc",
        "mlx/mlx/backend/no_cpu/compiled.cpp",
        "mlx/ACKNOWLEDGMENTS.md",
        "mlx/CMakeLists.txt",
        "mlx/CODE_OF_CONDUCT.md",
        "mlx/CONTRIBUTING.md",
        "mlx/LICENSE",
        "mlx/MANIFEST.in",
        "mlx/README.md",
        "mlx/benchmarks",
        "mlx/cmake",
        "mlx/docs",
        "mlx/examples",
        "mlx/mlx.pc.in",
        "mlx/pyproject.toml",
        "mlx/python",
        "mlx/setup.py",
        "mlx/tests",
        "mlx/mlx/backend/cuda/allocator.cpp",
        "mlx/mlx/backend/cuda/compiled.cpp",
        "mlx/mlx/backend/cuda/conv.cpp",
        "mlx/mlx/backend/cuda/cublas_utils.cpp",
        "mlx/mlx/backend/cuda/cudnn_utils.cpp",
        "mlx/mlx/backend/cuda/custom_kernel.cpp",
        "mlx/mlx/backend/cuda/delayload.cpp",
        "mlx/mlx/backend/cuda/device_info.cpp",
        "mlx/mlx/backend/cuda/device.cpp",
        "mlx/mlx/backend/cuda/eval.cpp",
        "mlx/mlx/backend/cuda/fence.cpp",
        "mlx/mlx/backend/cuda/indexing.cpp",
        "mlx/mlx/backend/cuda/jit_module.cpp",
        "mlx/mlx/backend/cuda/load.cpp",
        "mlx/mlx/backend/cuda/matmul.cpp",
        "mlx/mlx/backend/cuda/primitives.cpp",
        "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/cuda/slicing.cpp",
        "mlx/mlx/backend/cuda/utils.cpp",
        "mlx/mlx/backend/cuda/worker.cpp",
        "mlx/mlx/backend/cuda/binary",
        "mlx/mlx/backend/cuda/conv",
        "mlx/mlx/backend/cuda/copy",
        "mlx/mlx/backend/cuda/device",
        "mlx/mlx/backend/cuda/gemms",
        "mlx/mlx/backend/cuda/quantized",
        "mlx/mlx/backend/cuda/reduce",
        "mlx/mlx/backend/cuda/steel",
        "mlx/mlx/backend/cuda/unary",
        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp",
        "mlx/mlx/distributed/jaccl/utils.cpp",
    ],
    cSettings: [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
    ],
    cxxSettings: cxxSettings + [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"),
        .define("MLX_VERSION", to: "\"0.31.1\""),
    ],
    linkerSettings: linkerSettings
)

let package = Package(
    name: "vmlx",
    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        // MLX runtime products (consumed by downstream vMLX* targets).
        .library(name: "MLX", targets: ["MLX"]),
        .library(name: "MLXRandom", targets: ["MLXRandom"]),
        .library(name: "MLXNN", targets: ["MLXNN"]),
        .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
        .library(name: "MLXFFT", targets: ["MLXFFT"]),
        .library(name: "MLXLinalg", targets: ["MLXLinalg"]),
        .library(name: "MLXFast", targets: ["MLXFast"]),

        // vMLX engine + app libraries.
        .library(name: "vMLXEngine", targets: ["vMLXEngine"]),
        .library(name: "vMLXServer", targets: ["vMLXServer"]),
        .library(name: "vMLXTheme",  targets: ["vMLXTheme"]),
        .library(name: "vMLXLMCommon",  targets: ["vMLXLMCommon"]),
        .library(name: "vMLXLLM",       targets: ["vMLXLLM"]),
        .library(name: "vMLXVLM",       targets: ["vMLXVLM"]),
        .library(name: "vMLXEmbedders", targets: ["vMLXEmbedders"]),
        .library(name: "vMLXWhisper",   targets: ["vMLXWhisper"]),
        .library(name: "vMLXTTS",       targets: ["vMLXTTS"]),
        .library(name: "vMLXFlux",       targets: ["vMLXFlux"]),
        .library(name: "vMLXFluxKit",    targets: ["vMLXFluxKit"]),
        .library(name: "vMLXFluxModels", targets: ["vMLXFluxModels"]),
        .library(name: "vMLXFluxVideo",  targets: ["vMLXFluxVideo"]),
        // APFS case-insensitive filesystems collide `vmlx` and `vMLX` on
        // the `.build/debug/` slot. Name the CLI `vmlxctl` to break the tie.
        .executable(name: "vmlxctl", targets: ["vMLXCLI"]),
        .executable(name: "vMLX",    targets: ["vMLXApp"]),
    ],
    dependencies: [
        // MLX runtime now lives in-tree (see Cmlx/MLX/etc. targets below).
        // Numerics is the only external dep MLX itself requires.
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        // Lightweight Swift HTTP server (sandbox-safe: no fork/exec).
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.5.0"),
        // NIOSSL backs HummingbirdTLS — used to terminate HTTPS at the
        // server when --ssl-keyfile/--ssl-certfile are provided. Pulls in
        // BoringSSL via SwiftNIO; ~5MB compiled, no other transitive deps.
        .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.27.0"),
        // CLI argument parsing.
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
        // Tokenizer + HuggingFace hub loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.21"),
        // Jinja template engine — pulled in transitively by swift-transformers
        // for chat-template rendering. Declared explicitly here so the vMLX
        // test target can import it directly for chat-template repro tests.
        .package(url: "https://github.com/johnmai-dev/Jinja", exact: "1.3.0"),
    ],
    targets: [

        // MARK: - MLX runtime (vendored from mlx-swift @ osaurus-0.31.3)

        cmlx,

        .target(
            name: "MLX",
            dependencies: [
                "Cmlx",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Sources/MLX",
            exclude: mlxSwiftExcludes,
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXRandom",
            dependencies: ["MLX"],
            path: "Sources/MLXRandom",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFast",
            dependencies: ["MLX", "Cmlx"],
            path: "Sources/MLXFast",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXNN",
            dependencies: ["MLX"],
            path: "Sources/MLXNN",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXOptimizers",
            dependencies: ["MLX", "MLXNN"],
            path: "Sources/MLXOptimizers",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFFT",
            dependencies: ["MLX"],
            path: "Sources/MLXFFT",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLinalg",
            dependencies: ["MLX"],
            path: "Sources/MLXLinalg",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // MARK: - Vendored LLM engine (from vmlx-swift-lm)
        //
        // All ongoing JANGTQ kernel optimization work (P1-P12 stack,
        // compile-G fusion, decode fusion, subgroup matrix instructions,
        // etc.) lands DIRECTLY in these target trees.

        .target(
            name: "vMLXLMCommon",
            dependencies: ["MLX", "MLXNN", "MLXOptimizers", "MLXRandom"],
            path: "Sources/vMLXLMCommon",
            exclude: [
                "README.md",
                "Documentation.docc",
                "BatchEngine/BATCH_ENGINE.md",
                "FlashMoE/README.md",
            ]
        ),
        .target(
            name: "vMLXLLM",
            dependencies: [
                "vMLXLMCommon", "MLX", "MLXNN", "MLXOptimizers",
            ],
            path: "Sources/vMLXLLM",
            exclude: [
                "README.md",
                "Documentation.docc",
            ]
        ),
        .target(
            name: "vMLXVLM",
            dependencies: [
                "vMLXLMCommon", "MLX", "MLXNN", "MLXOptimizers",
            ],
            path: "Sources/vMLXVLM",
            exclude: [
                "README.md",
            ]
        ),
        .target(
            name: "vMLXEmbedders",
            dependencies: ["vMLXLMCommon", "MLX", "MLXNN"],
            path: "Sources/vMLXEmbedders",
            exclude: [
                "README.md",
            ]
        ),

        // MARK: - Whisper (audio transcription)
        // Native MLX-Swift port of mlx-examples/whisper — encoder,
        // decoder, log-mel pipeline, tokenizer adapter, greedy decoding
        // loop. Consumed by `Engine.transcribe` to serve
        // `/v1/audio/transcriptions`.
        .target(
            name: "vMLXWhisper",
            dependencies: [
                "MLX", "MLXNN", "MLXFast",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/vMLXWhisper"
        ),

        // MARK: - TTS (text-to-speech)
        // Pure-Swift, no external audio deps. Placeholder tone backend
        // today, Kokoro neural backend scaffolded for follow-up. See
        // Sources/vMLXTTS/TTSEngine.swift top comment for handoff.
        .target(
            name: "vMLXTTS",
            dependencies: [],
            path: "Sources/vMLXTTS"
        ),

        // MARK: - Vendored image + video engine (from vmlx-flux)

        .target(
            name: "vMLXFluxKit",
            dependencies: [
                "MLX", "MLXNN", "MLXRandom", "vMLXLMCommon",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/vMLXFluxKit"
        ),
        .target(
            name: "vMLXFluxModels",
            dependencies: ["vMLXFluxKit", "MLX", "MLXNN"],
            path: "Sources/vMLXFluxModels"
        ),
        .target(
            name: "vMLXFluxVideo",
            dependencies: ["vMLXFluxKit", "MLX"],
            path: "Sources/vMLXFluxVideo"
        ),
        .target(
            name: "vMLXFlux",
            dependencies: [
                "vMLXFluxKit",
                "vMLXFluxModels",
                "vMLXFluxVideo",
            ],
            path: "Sources/vMLXFlux"
        ),

        // MARK: - Engine adapter
        // Wraps the vendored factories with vMLX glue: JANG repack, prefix
        // cache, SSM companion, reasoning suppression, tool-call dispatch,
        // ModelLibrary, SettingsStore, LogStore, MetricsCollector.
        .target(
            name: "vMLXEngine",
            dependencies: [
                "MLX",
                "vMLXLLM",
                "vMLXVLM",
                "vMLXLMCommon",
                "vMLXEmbedders",
                "vMLXWhisper",
                "vMLXFlux",
                "vMLXTTS",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/vMLXEngine"
        ),

        // MARK: - HTTP server
        .target(
            name: "vMLXServer",
            dependencies: [
                "vMLXEngine",
                "vMLXTTS",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HummingbirdTLS", package: "hummingbird"),
                .product(name: "NIOSSL", package: "swift-nio-ssl"),
            ],
            path: "Sources/vMLXServer"
        ),

        // MARK: - Theme tokens (Linear-inspired black/grey)
        .target(
            name: "vMLXTheme",
            path: "Sources/vMLXTheme"
        ),

        // MARK: - CLI executable → `vmlxctl`
        .executableTarget(
            name: "vMLXCLI",
            dependencies: [
                "vMLXEngine",
                "vMLXServer",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/vMLXCLI"
        ),

        // MARK: - SwiftUI app executable → `vMLX`
        .executableTarget(
            name: "vMLXApp",
            dependencies: [
                "vMLXEngine",
                "vMLXServer",
                "vMLXTheme",
            ],
            path: "Sources/vMLXApp"
        ),

        // MARK: - Tests
        .testTarget(
            name: "vMLXTests",
            dependencies: [
                "vMLXEngine",
                "vMLXServer",
                "vMLXTheme",
                "vMLXApp",
                "vMLXLMCommon",
                "vMLXWhisper",
                "vMLXTTS",
                .product(name: "Jinja", package: "Jinja"),
            ],
            path: "Tests/vMLXTests"
        ),
    ],
    cxxLanguageStandard: .gnucxx20
)
