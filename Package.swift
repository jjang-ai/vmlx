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
        // Bundle identity must match what SwiftPM actually produces for
        // this target. SwiftPM names bundles `${package}_${target}`, and
        // since our package is `vmlx` (not `mlx-swift`), the bundle is
        // `vmlx_Cmlx`. The older `mlx-swift_Cmlx` comment here was a
        // stale copy-paste from the upstream repo — keeping it meant
        // `load_swiftpm_library` searched for a bundle name that never
        // existed in our build, silent-failing every model load with
        // "Failed to load the default metallib." (2026-04-15 fix.)
        .define("SWIFTPM_BUNDLE", to: "\"vmlx_Cmlx\""),
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
        // Exclude the generated `.metal` sources so Xcode's MetalLink
        // phase does NOT produce a `default.metallib` in the bundle —
        // that collides with `resources: [.copy("default.metallib")]`
        // below and trips "Multiple commands produce …/default.metallib"
        // during `xcodebuild archive`. The pre-built `default.metallib`
        // under `Sources/Cmlx/` (or staged via `scripts/stage-metallib.sh`)
        // is the single source of truth.
        "mlx-generated/metal",
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp",
        "mlx/mlx/distributed/jaccl/utils.cpp",
    ],
    resources: [
        // Pre-built Metal library. `mlx-swift`'s `device.cpp`
        // `load_swiftpm_library` looks for `default.metallib` inside the
        // SwiftPM bundle `mlx-swift_Cmlx` at runtime. Without this the
        // SwiftPM debug build (vmlxctl + the app's .build variant) can't
        // load any model: "Failed to load the default metallib."
        //
        // Build with the helper at `scripts/build-metallib.sh` OR copy
        // a known-good prebuilt .metallib from a prior Xcode archive.
        .copy("default.metallib"),
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
        // vMLX §225 (2026-04-21): vendored at Vendor/Jinja so we can ship
        // the negative-step `slice()` Python-semantics fix without a
        // manual fork round-trip. Upstream reference:
        // https://github.com/johnmai-dev/Jinja @ 1.3.0. Patch details in
        // Vendor/Jinja/Sources/Utilities.swift.
        .package(path: "Vendor/Jinja"),
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
            dependencies: ["MLX", "MLXNN", "MLXOptimizers", "MLXRandom", "MLXFast"],
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
                "vMLXFluxKit",
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
                "vMLXLMCommon",
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

        // MARK: - Parser unit tests (runnable via `swift test`)
        // Minimal test target wiring — covers the parsers that live in
        // vMLXEngine (tool-call + reasoning). Other vMLXTests files depend
        // on vMLXApp (executable target, can't @testable import) so they
        // stay Xcode-project-only for now. Added 2026-04-18 to enable
        // CLI-level verification of ToolCallParser regression guards
        // (§25 Llama multi-tag fix).
        .testTarget(
            name: "vMLXParserTests",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "vMLXServer",
            ],
            path: "Tests/vMLXTests",
            // Curated minimal set that compiles clean against framework
            // targets (no vMLXApp @testable). 49 of 60 test files are
            // technically eligible but expanding triggers a Swift
            // compiler crash (reproducible on CacheBlockTests) — safer
            // to keep this subset runnable via `swift test` and leave
            // the rest Xcode-project-only until compiler settles.
            sources: [
                "ParserTests.swift",
                "AhoCorasickTests.swift",
                "NumPyPCG64Tests.swift",
                "CacheCoordinatorGenPromptLenTests.swift",
                "SettingsStoreClampTests.swift",
                "LogStoreTests.swift",
                "OllamaCapabilitiesTests.swift",
                "PerfFloorScriptTests.swift",
                "RegressionConstantsTests.swift",
                "CapabilityDetectionTests.swift",
                // §385 — DSML tool-call parser for DSV4 Flash/Pro.
                "DSMLToolCallParserTests.swift",
                // §385 — DSV4 capability detection (silver + jang stamp).
                "DeepseekV4CapabilityTests.swift",
                // §391 — DSV4 multi-turn matrix: reasoning leak, DSML
                // tool-call ordering, effort transitions, family
                // sampling defaults, parser registry routing.
                "DeepseekV4MultiTurnTests.swift",
                // §397 — SSM parity map (vmlx GH #103/#105/#107/#109/#110)
                // pinned across hybrid families.
                "SSMParityMapTests.swift",
                // §397 — DSV4 looping-output regression guards (4
                // candidates from the 2026-04-24 diagnostic note:
                // chat template, EOS, sampling defaults, tq_bits strip).
                "DeepseekV4LoopingGuardTests.swift",
                // §405 — FIM completion truncator (decode-loop tail
                // strip for code completions on /v1/completions when
                // `truncate_fim: true`).
                "FIMTruncatorTests.swift",
                // §407 — DSV4 Compressor + Indexer mask helpers ported
                // from mlx-lm PR #1195. Pure-tensor helpers tested
                // standalone before the attention forward integration.
                "DeepseekV4MaskHelpersTests.swift",
                // §410 — shape-authoritative bit-width inference. Pins
                // the preference order ((8,32) > (8,64) > … > (2,128))
                // and guards against the prior `knownGroupSize`-first
                // bug that picked (4,64) for ambiguous shapes — the
                // root cause of the DSV4 embed/wq_b miscloads.
                "JangShapeAuthoritativeTests.swift",
                // §421 — shape-authoritative routed-bits override
                // (peekRoutedBitsFromSafetensors + injectRoutedBits).
                // Synthetic safetensors files exercise the pre-decode
                // path that overrides mxtq_bits when bundle metadata
                // is missing or wrong. Pure file IO + JSON, no MLX dep.
                "JangtqRoutedBitsOverrideTests.swift",
                "ChatRequestValidationTests.swift",
                "ToolChoiceEnforcementTests.swift",
                "ToolCallReasoningMatrixTests.swift",
                "ReasoningInjectionTests.swift",
                "PromptLengthGuardTests.swift",
                "MCPConfigTests.swift",
                // Second batch 2026-04-18 — vMLXEngine-only tests
                "BenchmarkTests.swift",
                "BenchmarkSummaryTests.swift",
                "CacheStatsTests.swift",
                "DownloadManagerTests.swift",
                "DownloadSidecarTests.swift",
                "DownloadConcurrencyTests.swift",
                "EngineMidLoadCancelTests.swift",
                "EngineMemoryBudgetTests.swift",
                "SleepDuringDownloadTests.swift",
                "DeepSleepReclamationTests.swift",
                "WakeRetryLoopEscapeTests.swift",
                "IdleTimerDualCountdownTests.swift",
                "StandbyUnloadRaceTests.swift",
                "GenerationLockCancelTests.swift",
                "BackpressureTests.swift",
                "TurboQuantRegressionGuardTests.swift",
                "JangStampAutoActivationTests.swift",
                "TurboQuantRegressionTests.swift",
                "ModelLoadingRegressionTests.swift",
                "VLEndToEndRegressionTests.swift",
                "ChatReasoningRegressionTests.swift",
                "APISurfaceRegressionTests.swift",
                "ObservabilityRegressionTests.swift",
                "ChatSettingsPersistenceTests.swift",
                "SettingsIsolationTests.swift",
                "ModelLifecycleAPITests.swift",
                "EndToEndNuanceTests.swift",
                "HuggingFaceSearchTests.swift",
                "IdleTimerTests.swift",
                "IdleTimerResetOnStreamTests.swift",
                "MetricsAccuracyTests.swift",
                "MetricsCollectorTests.swift",
                "ModelLibraryTests.swift",
                "ModelLibraryWatcherTests.swift",
                "StreamCancellationTests.swift",
                "TurboQuantDefaultTests.swift",
                // Third batch 2026-04-18 — more vMLXEngine + vMLXLMCommon
                "ChatSettingsPopoverTests.swift",
                "EmbeddingsTests.swift",
                "MultiSessionEngineTests.swift",
                "MultiTurnVLImageTests.swift",
                "RealReasoningDispatchTests.swift",
                "SessionConfigFormTests.swift",
                "SettingsStoreTests.swift",
                "VLImagePipelineTests.swift",
                "TTSEngineTests.swift",
                "MemoryAwarePrefixCacheTests.swift",
                "DDTreeBuilderTests.swift",
                // Iter-40: SSM re-derive counter + stats wiring
                "SSMStateCacheReDeriveTests.swift",
                // Ralph-4 (GLM-5.1 template parity): repro tests for
                // swift-jinja against the GLM-5.1 glm_moe_dsa template
                // (enable_thinking on/off + tools-mode markers).
                "ChatTemplateReproTests.swift",
                // Ralph-12 (S02 smoke test): glm_moe_dsa factory alias
                // patches rope_parameters → rope_theta + routes to
                // DeepseekV3Model. Also pins registry advertised set.
                "ModelFactoryRegistrationTests.swift",
                // §331 CORS allowlist middleware — constructor-level
                // normalization guards (full HTTP roundtrip lives in
                // the live-verify harness).
                "CORSAllowlistMiddlewareTests.swift",
                // §338 (vmlx#47) tool-argument schema coercion — Int/
                // Bool/Number coercion against JSON Schema before MCP
                // dispatch, matching the Python fix.
                "SchemaArgumentCoercionTests.swift",
                // §340 — MCP clipboard import (Claude Desktop format).
                "MCPClipboardImportTests.swift",
                // §346 T6 — JANGTQ config mxtq_bits accepts both flat Int
                // and per-role dict. Qwen3.6-JANGTQ4 garbage root cause.
                "JANGTQConfigBitsDecodingTests.swift",
                // §347 lm-eval Phase 1 — tokenizer endpoints (/v1/tokenize,
                // /v1/detokenize, /v1/tokenizer_info + unprefixed aliases).
                "TokenizerRoutesRegistrationTests.swift",
                // §373 — generation_config.json fallback parity for
                // Qwen/Gemma/Nemotron recommended temp/top_p/top_k.
                "GenerationConfigDefaultsTests.swift",
                // §378 — sampling fallback priority (request > session
                // > generation_config > global) via resolutionTrace.
                "SamplingFallbackPriorityTests.swift",
                // Metal-dependent tests excluded — `swift test` can't load
                // the default.metallib from the SwiftPM bundle path;
                // JangDFlashDrafter, JangDFlashSpecDec, VisionEmbeddingCache,
                // PagedCacheBlockRelease, FlashMoE, TQDiskSerializer,
                // CacheCoordinatorRotatingGuard, CacheBlock stay Xcode-only.
            ]
        ),
    ],
    cxxLanguageStandard: .gnucxx20
)
