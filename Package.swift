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
        // Runtime examples — `swift run DSV4FlashRuntime <bundle>` etc.
        // Each example exercises the full vMLX runtime stack
        // (TurboQuant KV cache encode/decode, L2 BlockDiskCache,
        // SSM re-derive, JANGTQ load + chat encoder + reasoning +
        // tool dispatch). Sources live under `Examples/<Name>/`.
        .executable(name: "DSV4FlashRuntime", targets: ["DSV4FlashRuntime"]),
        .executable(name: "LagunaRuntime",    targets: ["LagunaRuntime"]),
        .executable(name: "Mistral3Runtime",  targets: ["Mistral3Runtime"]),
        .executable(name: "CacheMatrixProbe", targets: ["CacheMatrixProbe"]),
        // Real-bundle JANGPress smoke: opens an actual MoE bundle via
        // `JangPressMmapTier` + `JangPressEmbedTier`, drives a
        // synthetic routing pattern, reports per-acquire latency.
        // Doesn't load MLX — pure mmap+madvise + cache primitives.
        .executable(name: "JANGPressSmoke",   targets: ["JANGPressSmoke"]),
        // Real-bundle RSS measurement: opens a bundle via mmap,
        // force-faults all routed-expert pages, samples task RSS,
        // issues DONTNEED on the whole pool, samples RSS again,
        // reports the reclaim delta. The empirical answer to "does
        // JANGPress actually save RAM" on this host.
        .executable(name: "JANGPressRSSBench", targets: ["JANGPressRSSBench"]),
        // End-to-end Engine + JANGPress. Production code path: real
        // model load via Engine.load, real inference via engine.stream,
        // RSS sampled at every phase via mach_task_info. The honest
        // "what does JANGPress cost on tok/s + RSS" answer.
        .executable(name: "JANGPressE2E",      targets: ["JANGPressE2E"]),
        // Minimal debug harness — one request, prints every chunk
        // verbatim. For diagnosing the iter-13 0-chunks finding.
        .executable(name: "JANGPressDebug",    targets: ["JANGPressDebug"]),
        // Side-by-side baseline-vs-jangpress comparison bench. Same
        // prompt, recorded tok/s + content for both runs, diff for
        // coherency.
        .executable(name: "JANGPressCompare",  targets: ["JANGPressCompare"]),
        // Memory-pressure simulation: force-faults routed pages, then
        // inflates an anonymous balloon to make the kernel feel real
        // pressure, verifies reclaim happens AND data integrity post
        // re-fault. Production-readiness check for constrained-RAM hosts.
        .executable(name: "JANGPressBalloonBench", targets: ["JANGPressBalloonBench"]),
        // Multi-turn coherency: 3 conversational turns, validates each is
        // coherent + measures re-fault latency from iter-24 lazy reclaim.
        .executable(name: "JANGPressMultiTurn", targets: ["JANGPressMultiTurn"]),
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
        // vMLX 2026-04-30: vendored at Vendor/SwiftTransformers (pinned at
        // huggingface/swift-transformers @ f000aa7aec0e ≈ 0.1.21) so chat-
        // template + tokenizer behavior changes land directly here without
        // a remote fetch. Patches: see Vendor/SwiftTransformers/Sources/.
        .package(path: "Vendor/SwiftTransformers"),
        // Jinja template engine — used both by SwiftTransformers (chat
        // templates) and by vMLX engine code directly. Vendored at
        // Vendor/Jinja from johnmai-dev/Jinja @ 1.3.0. The vendored
        // SwiftTransformers manifest depends on `path: ../Jinja` so we
        // run a single Jinja implementation across the whole tree.
        // Patch details in Vendor/Jinja/Sources/Utilities.swift.
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
                "vMLXLMCommon", "vMLXLLM", "MLX", "MLXNN", "MLXOptimizers",
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
                .product(name: "Transformers", package: "SwiftTransformers"),
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
                .product(name: "Transformers", package: "SwiftTransformers"),
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
                .product(name: "Transformers", package: "SwiftTransformers"),
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

        // MARK: - Runtime examples
        //
        // `RuntimeShared` is a small helper library used by the three
        // per-model example executables. It captures the LoadOptions
        // that turn on TurboQuant KV cache, L2 BlockDiskCache,
        // SSMReDerive, and JANG repack — the same stack the production
        // `vmlxctl serve` uses, so the example output reflects real
        // runtime behavior end-to-end.
        .target(
            name: "RuntimeShared",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "vMLXLLM",
            ],
            path: "Examples/RuntimeShared"
        ),
        .executableTarget(
            name: "DSV4FlashRuntime",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/DSV4FlashRuntime"
        ),
        .executableTarget(
            name: "LagunaRuntime",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/LagunaRuntime"
        ),
        .executableTarget(
            name: "Mistral3Runtime",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/Mistral3Runtime"
        ),
        .executableTarget(
            name: "CacheMatrixProbe",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/CacheMatrixProbe"
        ),

        // Real-bundle JANGPress smoke runner — opens an actual
        // safetensors bundle via JangPressMmapTier +
        // JangPressEmbedTier, drives a synthetic routing pattern,
        // reports per-acquire latency. Independent of MLX — pure
        // mmap+madvise + cache primitives. Default bundle is
        // Laguna-XS.2-JANGTQ; override via argv[1].
        .executableTarget(
            name: "JANGPressSmoke",
            dependencies: [
                "vMLXLMCommon",
            ],
            path: "Examples/JANGPressSmoke"
        ),

        // RSS measurement bench — opens a real bundle, force-faults
        // routed-expert pages, samples task RSS via mach_task_info,
        // issues DONTNEED on the pool, samples again, reports the
        // delta. Independent of MLX.
        .executableTarget(
            name: "JANGPressRSSBench",
            dependencies: [
                "vMLXLMCommon",
            ],
            path: "Examples/JANGPressRSSBench"
        ),

        // Memory-pressure simulation bench. Inflates an anonymous
        // balloon to make the kernel feel real pressure, verifies
        // forceRelease reclaim happens AND data integrity post
        // re-fault. Production-readiness check for constrained-RAM
        // hosts. See Examples/JANGPressBalloonBench/main.swift.
        .executableTarget(
            name: "JANGPressBalloonBench",
            dependencies: [
                "vMLXLMCommon",
            ],
            path: "Examples/JANGPressBalloonBench"
        ),

        // Multi-turn coherency stress test for the iter-24 lazy reclaim.
        // Verifies turn 2+ refault correctly after turn 1 reclaim.
        .executableTarget(
            name: "JANGPressMultiTurn",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/JANGPressMultiTurn"
        ),

        // End-to-end Engine + JANGPress. Same dependency set as the
        // per-model runtime examples — pulls vMLXEngine, vMLXLMCommon,
        // RuntimeShared. Loads a real bundle via Engine.load with
        // enableJangPress=true, runs inference, measures
        // RSS + tok/s.
        .executableTarget(
            name: "JANGPressE2E",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/JANGPressE2E"
        ),

        // Debug harness — one request, prints every chunk verbatim.
        .executableTarget(
            name: "JANGPressDebug",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/JANGPressDebug"
        ),

        // Comparison bench — runs the same prompt with JangPress
        // off vs on, records tok/s + content for both, diffs for
        // coherency.
        .executableTarget(
            name: "JANGPressCompare",
            dependencies: [
                "vMLXEngine",
                "vMLXLMCommon",
                "RuntimeShared",
            ],
            path: "Examples/JANGPressCompare"
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
                "MLX",
                "vMLXEngine",
                "vMLXLMCommon",
                "vMLXLLM",
                "vMLXServer",
                "vMLXVLM",
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
                "BrowserToolSmokeTests.swift",
                "SSMCompanionDiskStoreTests.swift",
                "FamilySamplingDefaultsTests.swift",
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
                // 2026-05-01 — group-size-first parity fix tests.
                // Pin the new ordering (mirrors Python
                // `vmlx_engine/utils/jang_loader.py:_pre_fix_bits_from_shard`).
                // Catches a future refactor that swaps back to
                // bits-first and re-introduces the JANG_4M-CRACK
                // rms_norm crash.
                "JangLoaderInferBitsParityTests.swift",
                // 2026-05-01 — Mistral chat template fallback +
                // chat_template_kwargs propagation across all 4
                // protocols. Locks the renderer behavior so Mistral
                // 3 / 3.5 / 4 / Laguna / ministral3 stay coherent
                // when the johnmai-dev/Jinja parser can't handle
                // their native template.
                "MistralChatTemplateRendererTests.swift",
                // 2026-05-01 — DSV4 long-ctx default-on + SWA
                // per-layer routing across Mistral 3 (dense + JANGTQ
                // + VLM JANGTQ) and Gemma 4. Locked-in MMLU 200q:
                // 74.5 % (LC=0) → 81.5 % (LC=1, default ON, +7 pp
                // architecture-only).
                "SWAAndDSV4LongCtxRegressionTests.swift",
                // 2026-05-01 — heterogeneous-cache compile P0 fix per
                // ~/vmlx-swift-lm/SWIFT-PERF-FIXES.md (Qwen3.5 35B-A3B
                // 61 → 90-100 tok/s by relaxing the
                // homogeneous-KVCacheSimple bail to allow MambaCache
                // + CompilableKVCache passthrough). 2026-05-01 follow-up
                // adds the SWA decode-path port (CompilableRotatingKVCache)
                // landing Mistral 3 / Gemma 4 / DSV4 SWA on the compile
                // fast path.
                "HeterogeneousCompileDecodeTests.swift",
                // 2026-05-01 wake-from-standby pre-message surface —
                // Engine.wakeFromStandby(emitNotice:) closure overload
                // + WakeNoticeBox + SSEEncoder.leadingComments wiring.
                // OpenAIRoutes ChatCompletions is the canonical example;
                // Anthropic / Ollama mirror it as follow-ups.
                "WakeNoticeRegressionTests.swift",
                // 2026-05-01 inline tool-call card output capture —
                // ChatMessage.toolOutputs + ChatViewModel.applyChunk
                // capture of `.done`/`.error` ToolStatus messages.
                // Without this wiring InlineToolCallCard.output was
                // hardcoded nil and the expanded card body was dead
                // code — users running MCP through chat couldn't see
                // tool stdout/stderr.
                "InlineToolCallOutputTests.swift",
                // 2026-05-01 night Laguna LLM port from
                // ~/vmlx-swift-lm/Libraries/MLXLLM/Models/Laguna.swift.
                // Pins LagunaModel + LagunaConfiguration types,
                // factory dispatch, JANGTQ bits/seed probe, and
                // engine.unsupportedSwiftFamilies removal.
                "LagunaPortRegressionTests.swift",
                // 2026-05-01 night audit-loop iter 1 — i18n catalog
                // drift source-scan. The runtime contract test
                // (L10nEntryContractTests) is Xcode-only because
                // vMLXApp is an executable target; this source-scan
                // version reads Strings.swift directly so `swift test`
                // catches empty translations + format-specifier
                // placeholder drift across en/ja/ko/zh.
                "I18nDriftSourceScanTests.swift",
                // 2026-05-01 night audit-loop iter 1 — per-family
                // cache-type matrix pin. Locks the `cacheType` stamp
                // on every architecturally-significant silver-tier
                // row (MLA: mistral4 / deepseek V2-V4 / glm5 / kimi;
                // hybrid: qwen3_next / nemotron_h / granite_moe /
                // gemma3n; mamba: pure mamba / falcon_mamba /
                // qwen_mamba; kv: llama / qwen3 / mistral / gpt_oss).
                // A future refactor that flips one would crash on
                // first prefill — this catches it at test time.
                "CacheTypeMatrixTests.swift",
                // 2026-05-01 night audit-loop iter 2 — DSV4 tri-mode
                // attention per-layer routing pin. compress_ratio 0
                // (SWA boundary), 4 (CSA + Indexer top-k 512), 128
                // (HSA — compressor downsample only). Pins layer-0/
                // last-layer SWA, alternating CSA/HSA inner default,
                // Compressor + Indexer guards, RoPE dispatch
                // (compress_rope_theta + YaRN vs plain rope_theta),
                // newCache per-layer DSV4LayerCache vs RotatingKVCache,
                // VMLX_DSV4_LONG_CTX default-ON, and overlap-transform
                // only on CSA layers. 8/8 green.
                "DSV4TriModeAttentionTests.swift",
                // 2026-05-01 night audit-loop iter 2 — reasoning
                // ON/OFF state-machine pin. The most-regressed surface
                // in the engine (3+ regressions per
                // `feedback_reasoning_off_ui_stuck.md`). Pins
                // `inThinkBlock` initial seed, `alwaysParse` widening,
                // `harmonyActive` gating on `effectiveThinking`,
                // §15 vs BLOCKER #3 fork (both branches must remain),
                // budget-exhausted reroute, `seenReasoningMarker`
                // latch one-way, and parser reset propagating
                // `thinkInPrompt: modelStampsThink`. 8/8 green.
                "ReasoningStateMachineTests.swift",
                // 2026-05-01 night audit-loop iter 3 — chat input
                // button state matrix. canSend (text+image+video
                // accept, remote short-circuit, running+standby pass,
                // loading/stopped/error reject), buttonEnabled stays
                // tappable while generating (so Stop works), Send
                // icon flips to stop.fill, accessibility label flips,
                // placeholderText covers all 5 engine states,
                // helpText covers generating/canSend/per-state,
                // Cmd+Enter sends, Esc dismisses. 10/10 green.
                "InputBarButtonStateTests.swift",
                // 2026-05-01 night audit-loop iter 3 — SSM companion
                // cache gate. isHybrid=cacheType==hybrid (NOT name
                // regex), companionOn=isHybrid && enableSSMCompanion,
                // setHybrid only fires on AND-pass, AND-false-but-
                // hybrid logs the recompute, MLA+TQ exclusion logs,
                // CacheCoordinator init log includes hybrid flag,
                // hybrid silver rows present. 6/6 green.
                "SSMCompanionGatingTests.swift",
                // 2026-05-01 night audit-loop iter 4 — Tray
                // lifecycle button enablement matrix. Pins all 6
                // predicates' state-machine behavior + remote-session
                // short-circuit + canStart disjoint from canStop
                // (mutual-exclusion guard against running-while-
                // start-enabled regressions). 8/8 green.
                "LifecycleButtonMatrixTests.swift",
                // 2026-05-01 night audit-loop iter 4 — DownloadStatusBar
                // visibility per `feedback_download_window.md` (NO
                // silent downloads EVER). Pins both auto-open paths:
                // observeDownloads (sidebar trigger) +
                // forwardDownloadEvents (HTTP-initiated pulls), the
                // hasAutoOpenedDownloadsWindow latch, appOpenWindow
                // capture, and openDownloadsWindowIfReady public
                // hook. 5/5 green.
                "DownloadVisibilityTests.swift",
                // 2026-05-01 night audit-loop iter 4 — tool + reasoning
                // parser registry alias coverage. Pins ~50 aliases
                // (Qwen / Llama / Mistral / DeepSeek / Kimi / Granite
                // / Nemotron / StepFun / xLAM / Functionary / GLM /
                // MiniMax / Gemma / native / hermes) so a missing
                // route doesn't silently leak tool calls into content
                // OR reasoning into content. 4/4 green.
                "ToolParserAliasCoverageTests.swift",
                // 2026-05-01 night audit-loop iter 5 — VL image
                // marker matrix. Family-aware marker selection
                // (Gemma `<|image|>`, Mistral `[IMG]`, Qwen full
                // `<|vision_start|><|image_pad|><|vision_end|>` triple,
                // default `<image>`), auto-insert checks all 3 marker
                // variants before injecting (no double-insert),
                // family-resolved param threading, one-marker-per-image
                // count, video marker Qwen-only. 9/9 green.
                "VLImageMarkerTests.swift",
                // 2026-05-01 night audit-loop iter 5 — cache lifecycle
                // clearing. softSleep clears coord + DFlash adapters
                // (drafter+target), deepSleep full teardown (clear +
                // nil coord, drop loaded/path/jang/remote), wake
                // override-with-swap clears coord BEFORE load replay,
                // clearCaches keeps coord ref hot (panel intent), both
                // sleep paths cancel streams BEFORE clearing. 5/5 green.
                "CacheLifecycleClearingTests.swift",
                // 2026-05-01 night audit-loop iter 6 — Engine cancel
                // safety. cancelStream covers BOTH legacy
                // currentStreamTask + per-id streamTasksByID, snapshot
                // before walk (no mutation-during-iteration), remote
                // client cancel fires, per-id removes entry on success,
                // sleep paths cancel BEFORE clearing caches. 6/6 green.
                "CancellationSafetyTests.swift",
                // 2026-05-01 night audit-loop iter 6 — port allocator
                // collision recovery. firstAvailablePort throws on
                // exhaustion (NOT silent return-of-known-bad-port),
                // EngineError.portInUse case, lan flag threading,
                // GatewayServer + Server both surface portInUse,
                // L10n catalog has all 3 collision UX strings. 6/6 green.
                "PortAllocatorTests.swift",
                // 2026-05-01 night /jang sweep — API protocol parity
                // across OpenAI Chat (delta.reasoning_content +
                // delta.tool_calls), OpenAI Responses (response.*
                // event names), Anthropic /v1/messages (thinking +
                // tool_use blocks + input_json_delta), Ollama
                // /api/chat (message.tool_calls). All 4 routes
                // thread chat_template_kwargs + wake before streaming
                // + emit termination event. 12/12 green.
                "APIProtocolParityTests.swift",
                // 2026-05-01 night audit-loop iter 8 — settings
                // persistence + migration + live propagation.
                // schemaVersion=0 default + migration gate + family-
                // gated unsafe-pattern exclusion (nemotron-h /
                // qwen3-next / deepseek-v4 / kimi-k25 / kimi-k26 /
                // laguna). All 3 setters present. allSessionIDs
                // hydration + AppState.hydrateSessionsFromSettings.
                // applyKvCacheQuantization + applySettings live hooks
                // used by all 4 UI surfaces (SessionConfigForm /
                // AdvancedServerCard / ImageScreen / APIScreen). DB
                // seed on first-run. 9/9 green.
                "SettingsPersistenceLiveTests.swift",
                // 2026-05-01 night audit-loop iter 9 — Server tab
                // CachePanel architecture pills (hybrid / SWA / TQ).
                // Inverse-indicator gating on archBool flags, SSM
                // section re-emits hybridPill, total==0 placeholder,
                // 6-kind layer count breakdown plus TQ runtime-policy
                // fields. 8/8 green.
                "CachePanelPillsTests.swift",
                // 2026-05-04 production audit — cache architecture
                // stats must distinguish fresh-cache topology from
                // runtime TQ/compile policy. Guards CacheList recursion,
                // compiled cache kinds, TQ env overrides, MLA native
                // suppression, and compile-first suppression.
                "CacheArchitectureRuntimePolicyTests.swift",
                // 2026-05-01 night audit-loop iter 9 — MetricsCollector
                // public surface contract. actor isolation, 11-field
                // Snapshot, separate decode/prefill rolling buffers,
                // MLX.Memory.snapshot() for GPU, local resettable
                // peakGPUBytes, UUID-keyed subscribers, latency
                // buffer cap. 7/7 green.
                "MetricsCollectorContractTests.swift",
                // 2026-05-01 night audit-loop iter 9 — Tray dual-stage
                // idle countdown (A7 §259). Both soft+deep fields on
                // AppState, both rendered when running, deep visible
                // during soft sleep, deep sleep label has no
                // countdown, mmss formatter, legacy single-kind
                // fallback. 6/6 green.
                "IdleCountdownDualStageTests.swift",
                // 2026-05-01 night audit-loop iter 10 — Logs panel
                // toolbar surface. Pause/Resume + flushPending,
                // Auto-scroll + userScrolledUp guard, Compact toggle,
                // Newest/Oldest order, Export, Clear (drains engine
                // logs + local lines + pending), search filter
                // threading. 8/8 green.
                "LogsPanelToolbarTests.swift",
                // 2026-05-01 night audit-loop iter 10 — chat
                // persistence. Database upsertMessage / messages /
                // upsertSession / deleteMessage / deleteMessages
                // API. ON CONFLICT(id) DO UPDATE for streaming
                // idempotency. UPDATE clause includes content +
                // reasoning + tool_calls_json + is_streaming.
                // ChatViewModel restores via messages(for:) and
                // persists via upsertMessage. 5/5 green.
                "ChatPersistenceTests.swift",
                // 2026-05-01 night audit-loop iter 11 — input bar
                // media ingest. fileImporter accepts image+video
                // UTIs, drop handler same + .fileURL, security-scoped
                // resource wrapper on picker, video staging into
                // temp before defer{} closes scope (both picker and
                // drop paths), drop image uses Data load not URL,
                // MicRecorderButton + TTSPlaybackButton files exist.
                // 8/8 green.
                "InputBarMediaIngestTests.swift",
                // 2026-05-01 night audit-loop iter 12 — DSV4 HSA/CSA
                // parity with ~/jang/jang-tools/jang_tools/dsv4/
                // mlx_model.py. Compressor canonical weights (wkv,
                // wgate, ape, norm) + outDim doubles for overlap +
                // Indexer separate class with own Compressor + wq_b
                // + weights_proj. DSV4PoolState: separate
                // compressorState + indexerState (cadence isolation).
                // accumulateWindows + updatePool methods. Mask
                // helpers PR #1195 port + window mask formula docs.
                // 11/11 green.
                "DSV4HSACSAParityTests.swift",
                // 2026-05-03 Codex audit — DSV4 production diagnostics.
                // Shape traces remain available behind DSV4_DIAG /
                // DSV4_DUMP_LOGITS, but stderr writes and duplicate
                // q-projection diagnostics must not execute on the normal
                // HSA/CSA/SWA decode path. 3 contracts.
                "DSV4ProductionDiagnosticsContractTests.swift",
                // 2026-05-01 night audit-loop iter 12 — Gateway +
                // CORS + Auth + per-session HTTP routing production
                // lock. CORSAllowlist Set-based O(1), preflight 403
                // on disallowed, non-preflight Origin strip.
                // AuthTokenBox lock-protected + live update method.
                // Bearer accepts both Authorization+x-api-key.
                // Gateway: EngineResolver + EngineEnumerator
                // typealiases, portInUse throw on bind collision,
                // middleware stack order (CORS → Bearer → Admin →
                // RequestLogger). x-api-key + x-admin-token in
                // allowHeaders. Live credential rotation hook. AppState
                // per-session HTTP pool + state observer pool. 14/14 green.
                "GatewayCORSAuthProductionTests.swift",
                // 2026-05-01 night audit-loop iter 13 — /health
                // endpoint shape pin. Route registered, 6-state
                // mapping (stopped/loading/running/soft_sleep/
                // deep_sleep/error), engine="vmlx-swift" identity,
                // scheduling="serial-fifo" advertised, inflight +
                // waiting queue depth, idle_soft_seconds +
                // idle_deep_seconds when timer active, model field
                // resolved via modelLibrary displayName. 7/7 green.
                "HealthEndpointTests.swift",
                // 2026-05-01 night audit-loop iter 13 —
                // /v1/embeddings + Ollama /api/embed{,dings} parity.
                // Route registered, OpenAI-canonical caps
                // (maxInputs=2048 / maxStringBytes=32K /
                // maxTokens=8192), all 4 input shapes validated,
                // wakeFromStandby JIT-wake, body cap 16 MiB,
                // mapEngineError routes 4xx/5xx, Ollama adapters
                // share engine.embeddings pipeline. 8/8 green.
                "EmbeddingsEndpointTests.swift",
                // 2026-05-01 night audit-loop iter 13 — RequestLog
                // panel + middleware contract. Canonical log format
                // <METHOD> <PATH> -> <STATUS> (<MS>ms)[trace=<ID>],
                // thrown-error suffix, level escalation by status
                // (5xx→.error, 4xx→.warn), x-vmlx-trace-id capture,
                // RequestLogPanel.parse static helper, l10n strings,
                // panel wired into APIScreen. 7/7 green.
                "RequestLogPanelTests.swift",
                // 2026-05-01 night audit-loop iter 14 — MCP
                // production contract. MCPServerConfig canonical
                // fields + defaults + dual-key support (servers /
                // mcpServers from Claude Desktop) + collision
                // warning. Security: shell metachar block list (11
                // chars), NUL byte rejection in args/env, SSE
                // http/https scheme allowlist. Server manager:
                // namespaced-name split (`server__tool` with
                // multi-`__` tool name preserved), lazy server
                // start, processFailure → markServerDead.
                // Schema arg coercion wired BEFORE executeTool. All
                // 4 HTTP routes registered. Per-engine MCPServerManager
                // ownership for per-session isolation. Internal `bash`
                // routes through built-in tool, not MCP. Panel surfaces
                // enabled + skipSecurity toggles. 18/18 green.
                "MCPProductionContractTests.swift",
                // 2026-05-01 night audit-loop iter 15 — Anthropic
                // /v1/messages full shape. system field accepts
                // string + array forms, image content blocks accept
                // base64 + url sources, base64 repackaged as data:
                // URL with image/png default media_type, tool_use +
                // tool_result + server_tool_use + web_search_tool_result
                // blocks handled, SSE event names canonical
                // (message_start/content_block_start/content_block_stop/
                // message_delta/message_stop), stop_reason
                // tool_calls→tool_use mapping, usage on both
                // message_start + message_delta. 9/9 green.
                "AnthropicMessagesShapeTests.swift",
                // 2026-05-01 night audit-loop iter 15 — Image-gen
                // panel state. canSubmit gates: selected model,
                // non-empty prompt, edit-mode source-image required,
                // status.isActive blocks duplicate submit. submit()
                // double-guards canSubmit + selected. ImageGallery
                // takes isGenerating + currentStep + totalSteps +
                // elapsedSeconds. ImageMode threaded explicitly
                // (no regex inference). 6/6 green.
                "ImageGenPanelStateTests.swift",
                // 2026-05-01 night audit-loop iter 15 — MCP Claude
                // Desktop clipboard import behavior tests. Both
                // mcpServers + servers + bare-inner-block accepted.
                // Partial-success: invalid entries land in skipped
                // with reason, valid entries import. Hard rejects
                // (invalidJSON / noServerBlock / emptyServerBlock).
                // Deterministic alphabetical ordering. 7/7 green.
                "MCPClipboardImportTests2.swift",
                // 2026-05-01 night audit-loop iter 16 — Tray status
                // pill icon + color + label matrix. All 6 EngineState
                // cases map to UNIQUE SF symbols (iter-42: soft+deep
                // visually distinct), color palette canonical
                // (running=green/loading=orange/stopped=secondary/
                // standby=blue/error=red), remote sessions
                // short-circuit to .accentColor, 1-word state labels
                // for picker rows. icon(for:) static so MenuBarExtra
                // Scene closure can call without instance. 5/5 green.
                "TrayStatusPillMatrixTests.swift",
                // 2026-05-01 night audit-loop iter 16 — JANGTQ Metal
                // kernel library registration. 3 lazy-singleton
                // kernels (jangtq_hadamard_multiblock,
                // jangtq_fused_gate_up_swiglu, jangtq_gather_tq_matmul)
                // with canonical input/output name lists in correct
                // bind order. JANGTQKernelLibrary is namespace enum
                // (no instance state). JANGTQRuntimeCache is
                // final class @unchecked Sendable + .shared
                // singleton + loadSidecar / signs / codebook
                // accessors. 8/8 green.
                "JANGTQKernelLibraryTests.swift",
                // 2026-05-01 night audit-loop iter 17 — Paged
                // prefix-cache hash key contract. computeBlockHash
                // canonical 4-param signature (parentHash / tokenIds
                // / modelKey / mediaSalt). modelKey fed FIRST (chain
                // integrity). mediaSalt with `|media:` tag (no
                // collision with token bytes). tokenIds raw-buffer
                // hash (architecture-deterministic). PagedCacheManager
                // threads modelKey + mediaSalt. CacheCoordinator
                // strips genPromptLen with clamp. SSM-skip ANDs
                // isHybrid + genPromptLen>0. 7/7 green.
                "PrefixCacheHashContractTests.swift",
                // 2026-05-01 night audit-loop iter 17 — Terminal mode
                // sandbox. All 4 toggles @AppStorage-persisted
                // (readOnly / noNetwork / noDestructive / sandboxCwd),
                // each injects a SPECIFIC instruction line into
                // system prompt when ON (READ-ONLY MODE / NO-NETWORK
                // MODE / NO-DESTRUCTIVE / SANDBOX-CWD), enumerated
                // command lists for noNetwork (curl/wget/git/npm/pip)
                // + noDestructive (rm -rf, dd, mkfs, force-push,
                // hard-reset), sandboxCwd interpolates cwd.path,
                // user systemPromptOverride appends AFTER sandbox
                // lines (no shadowing). 7/7 green.
                "TerminalSandboxTests.swift",
                // 2026-05-01 night audit-loop iter 18 — Responses
                // SSE event-type matrix. All 9 required event names
                // present (response.created/in_progress/output_item.
                // added+done/output_text.delta+done/
                // reasoning_summary_text.delta/
                // function_call_arguments.delta/completed/failed),
                // response.created emitted FIRST, completed vs failed
                // mutually exclusive (single terminator), event type
                // field matches event name, [DONE] marker preserved. 6/6 green.
                "ResponsesAPIEventTypesTests.swift",
                // 2026-05-01 night audit-loop iter 18 — chunked
                // VLM prefill helper. Generic+@discardableResult
                // signature, single-call fast path (60% perf), inter-
                // chunk MLX-materialize cache + GPU.clearCache,
                // axis-1 (T) slice convention, final chunk return,
                // T = dim(1) extraction. 6/6 green.
                "ChunkedPrefillVLMContractTests.swift",
                // 2026-05-01 night audit-loop iter 18 — route
                // registration guard. Server.swift registers all 7
                // route groups in canonical order, exactly once each
                // (Hummingbird Router fatal-traps on dup path).
                // GatewayServer uses registerGateway variant for
                // /metrics so it aggregates across engines. Routes
                // register AFTER middleware stack (auth applies).
                // Library scan Task.detached so port-bind isn't
                // gated on disk I/O. 5/5 green.
                "RouteRegistrationGuardTests.swift",
                // 2026-05-01 night audit-loop iter 19 — GenerationLock
                // FIFO + cancel-aware + introspection. Actor isolation,
                // UUID-keyed waiters, withTaskCancellationHandler in
                // acquire (B2 §261 fix), release pops head-of-queue,
                // held stays true on baton-handoff, withLock releases
                // on throw, /health introspection accessors. Plus 3
                // BEHAVIOR tests (FIFO ordering, inflightOrQueued
                // count, withLock release-on-throw).
                "GenerationLockFairnessTests.swift",
                // 2026-05-01 night audit-loop iter 19 — bash tool
                // surface. Built-in bash precedence over MCP, cwd
                // fallback to terminalCwd, /bin/zsh -c shell, default
                // 120s timeout, JSON-Schema arg shape, stdout+stderr
                // captured separately, extractCwd helper. screenshot
                // + browser also precede MCP namespace fallback.
                "BashToolSurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 20 — EngineStateBanner
                // 5-state matrix + .easeOut(0.18) animation keyed on
                // distinct stateKey strings, legacy-panel CTA gated
                // on message text, deep-standby Wake-now CTA, remote
                // chat short-circuit returns EmptyView. 6/6 green.
                "EngineStateBannerTests.swift",
                // 2026-05-01 night audit-loop iter 20 — ModelLibrary
                // scan + load lifecycle. 300s freshness window,
                // scan(force:) bypass, markLoadStarted/Finished
                // coordination so deleteEntry refuses active loads.
                // PLUS Server.swift TLS configuration: requires both
                // key+cert, minTLSVersion=1.2, fall-back-to-plain on
                // misconfigured cert, NIOSSLCertificate.fromPEMFile
                // for LetsEncrypt fullchain compat. 8/8 green.
                "ModelLibraryScanTests.swift",
                // 2026-05-01 night audit-loop iter 20 — JANGTQ load
                // detection. jang_config.json filename constant +
                // search-path inclusion + weight_format field +
                // mxtq_seed field + target/ subdir fallback for
                // jangspec layouts. PLUS ChatScreen banner Retry
                // routes through loadChatModelInline (not just clear
                // bannerMessage). Auto-adopt running session on
                // ChatScreen mount. 7/7 green.
                "JANGTQLoadDetectionTests.swift",
                // 2026-05-01 night audit-loop iter 23 — Mic recorder
                // → Whisper pipeline contract. PCM16 mono 16kHz raw
                // matches Whisper's exact input format (drift =
                // software resample / mono-fail / wrong norm). Icon
                // + color flip on isRecording for canonical recording-
                // active signal. 6/6 green.
                "MicRecorderContractTests.swift",
                // 2026-05-01 night audit-loop iter 24 — SessionsSidebar
                // filteredSessions + selectSession cancel-before-swap
                // (otherwise streaming deltas mangle BOTH chats),
                // ReasoningBox default-expanded + userCollapsed
                // tracked separately from auto-expand. 4/4 green.
                "SessionsSidebarTests.swift",
                // 2026-05-01 night audit-loop iter 25 — Admin-auth
                // gate route coverage (iter-75 §103 fix). All 4 gate
                // categories: /admin/ /v1/cache/ adapter-mutation
                // (load/unload/fuse) + /api/delete. GET /v1/adapters
                // list NOT gated (SDK inspection). XFF leftmost peer
                // key for brute-force rate limit. Plus ChatExporter
                // markdown+json formats with reasoning section.
                // 6/6 green.
                "AdminAuthRouteCoverageTests.swift",
                // 2026-05-01 night audit-loop iter 26 — appearance +
                // locale persistence. All 4 appearance surfaces use
                // canonical `vmlx.appearance` key, locale userdefault
                // key `vmlx.uiLanguage`, current accessor fallback
                // chain (stored → fromSystem → .en), seedIfAbsent
                // guard, EnvironmentValues.appLocale injection +
                // default. UpdateBanner MITM defense comment. 6/6 green.
                "AppearanceLocaleSettingsTests.swift",
                // 2026-05-01 night audit-loop iter 27 — per-family
                // tool-call parser envelope tokens. Hermes/Qwen
                // regex with (?s) multiline flag, Llama python_tag,
                // Mistral [TOOL_CALLS] BOT, Kimi 5-token envelope
                // + section-alt alias, Granite single tool_call,
                // Nemotron nested function envelope. 7/7 green.
                "ToolCallParserFamilyEnvelopesTests.swift",
                // 2026-05-01 night audit-loop iter 28 — DSV4 DSML
                // tool-call envelope. Fullwidth U+FF5C `｜` sentinel,
                // `｜DSML｜` prefix, substring gate before regex,
                // string=true|false branches, JSON fragment parsing,
                // <think> strip first. 6/6 green.
                "DSMLToolCallParserContractTests.swift",
                // 2026-05-01 night audit-loop iter 29 — Whisper
                // /v1/audio/transcriptions + ImageGallery state. 7/7 green.
                "WhisperTranscribeContractTests.swift",
                // 2026-05-01 night audit-loop iter 30 — StreamingAccumulator
                // surface contract + 3 BEHAVIOR tests (passthrough,
                // tool-call buffer, reset). previous+current
                // private(set), buffered cache, hot-path scan capped
                // at maxMarkerLen-1 (iter-31 quadratic-scan fix),
                // feed assignment order, reset clears both. 8/8 green.
                "StreamingAccumulatorContractTests.swift",
                // 2026-05-01 night audit-loop iter 31 — RateLimit
                // sliding 60s window + OSAllocatedUnfairLock scoped
                // access (Swift 6 strict concurrency) + canonical
                // 429 OpenAI body + Retry-After header + XFF
                // leftmost peer key. DownloadManager Event Sendable
                // + AsyncStream subscribe. JangSpec canonical
                // filenames (jangspec.json / target/experts.jsidx
                // / target/hot_core.safetensors) + loadWeights
                // static method. 9/9 green.
                "RateLimitDownloadJangSpecTests.swift",
                // 2026-05-01 night audit-loop iter 32 — IdleTimer
                // actor + dual-stage countdown. actor isolation,
                // reset clears both fired flags (drift to one-only
                // = deep-sleep keeps firing post-resume),
                // sleepCountdowns returns (soft, deep) tuple A7 §259,
                // disabled returns nil-pair, setConfig deliberately
                // doesn't reset lastActivity. 5/5 green.
                "IdleTimerActorTests.swift",
                // 2026-05-01 night audit-loop iter 33-34 — VLM video
                // utils (CLIP_NORM_MEAN/STD, SIGLIP_NORM, frame
                // extractor zero-duration guard, samplesPerSecond
                // floor, 4 utility functions) + Theme tokens (13
                // canonical color tokens, dynamic(dark:light:)
                // adaptive theming, spacing scale xs/sm/md/lg). 11/11
                // green.
                "VLMVideoUtilsThemeTests.swift",
                // 2026-05-01 night audit-loop iter 35 — BrowserTool
                // headless WKWebView surface. 10-action enum
                // (open|click|type|scroll|screenshot|eval|back|
                // forward|reload|close), `navigate` alias,
                // selector-beats-(x,y) for click, close skips
                // snapshot, non-close finalizeWith snapshot,
                // .nonPersistent() data store, off-screen window at
                // (-50000,-50000), 20s nav timeout settles partial,
                // eval JSON serialization, OpenAI tool name
                // `browser`, only `action` required, type doc
                // `\\n` Enter, jsString escape coverage. 14/14
                // green.
                "BrowserToolSurfaceTests.swift",
                // 2026-05-02 audit-loop iter 95 — FluxKit ImageIO
                // PNG-write (final image gen step). Namespace enum,
                // writePNG @MainActor (NSBitmapImageRep main-thread
                // only), input contract doc (B,C=3,H,W float [0,1]
                // clamped), ndim 3/4 validation, batch dim squeeze,
                // channels 1/3 validation, normalize chain (clip
                // [0,1] → ×255 → round → uint8), CHW→HWC transpose,
                // color space switch (3→calibratedRGB / 1→
                // calibratedWhite + bitsPerPixel 24/8), createDirectory
                // intermediates (first-launch safety), filename
                // `prefix-uuid.png`, AppKit guard + notImplemented
                // fallback (NOT silent fail), PNG encode failure
                // throws invalidRequest, prefix default `vmlx`.
                // 14/14 green.
                "FluxImageIOContractTests.swift",
                // 2026-05-02 audit-loop iter 96 — DownloadManager actor
                // surface (HF model download manager). public actor,
                // Job 12-field struct (id/repo/displayName/totalBytes/
                // receivedBytes/bytesPerSecond/etaSeconds/status/error/
                // startedAt/localPath/requiresHFAuth) Sendable+Identifiable
                // +Codable, §327 forward-compat decoder defaults
                // requiresHFAuth=false when sidecar pre-§293 (avoids
                // history loss on upgrade), Status 6 cases (queued/
                // downloading/paused/completed/failed/cancelled), Event
                // 7 cases Sendable (started/progress/paused/resumed/
                // completed/failed/cancelled), §253b 3-job concurrent
                // cap with HF rate-limit ~30 req/min/IP citation,
                // maxConcurrentFiles=2 per-job shard cap, O7 §293
                // requiresHFAuth + "Paste HF token in Settings → API"
                // CTA doc, hfAuthToken private var (in-memory only,
                // UI persists in Keychain), 4 lifecycle ops public
                // (enqueue/pause/resume/cancel), liveDataTasks
                // [UUID: URLSessionDownloadTask] for native-task
                // cancel (NOT just Swift Task wrapper), 5-second
                // sliding window speed sampling. 16/16 green.
                "DownloadManagerSurfaceTests.swift",
                // 2026-05-02 audit-loop iter 97 — GlobalSettings tier-1
                // defaults contract (the values every freshly-created
                // session inherits). Pins kvCacheQuantization=turboquant
                // (user directive 2026-04-30 default-on), enableTurboQuant
                // = true (iter-64 reinstate), turboQuantBits=4 (sweet
                // spot 3.6×), enableDiskCache/MemoryCache/PrefixCache/
                // SSMCompanion/SSMReDerive/JANG/usePagedCache all true,
                // memoryCachePercent=0.20 (lowered from 0.30 on
                // 2026-04-26), pagedCacheBlockSize=64, maxCacheBlocks=
                // 1000, prefillStepSize=2048 (Python parity), maxNumSeqs
                // =5 (Mac override), maxPromptTokens=262144 (256k Metal
                // OOM guard), engineKind=batched, idleEnabled=true
                // (300/900s soft/deep), defaultHost=127.0.0.1 +
                // defaultLAN=false (security), gateway off + 8080,
                // sampler defaults (temp=0.7, topP=0.9, topK=0, minP=0,
                // repPen=1.0, maxTokens=32768, defaultEnableThinking=
                // nil tri-state), TLS off (empty key/cert), advanced
                // features off (smelt/flashMoe/distributed/jit/pld/
                // dflash all false; flashMoeSlotBank=64), JANGPress off
                // (mmap backend), block disk cache off, omniBackend=
                // stage1 (Stage2 opt-in), slidingWindowMode=auto +
                // size 16384, image defaults Schnell (4 steps, 3.5
                // guidance, 1024², 0.75 strength), SettingsTier 5
                // cases round-trip, SessionSettings.schemaVersion=0
                // (forces pre-v2 migration), isRemote false-on-nil/
                // empty + true-on-URL, ChatSettings all-nil overrides,
                // RequestOverride.from(req) pure-copy (no implicit
                // fills), JSON round-trip lossless.
                "GlobalSettingsDefaultsContractTests.swift",
                // 2026-05-02 audit-loop iter 98 — SettingsStore 4-tier
                // resolver BEHAVIOR contract (request > chat > session
                // > global > builtin). Pins tier precedence on numeric/
                // optional/tri-state fields, resolutionTrace identifies
                // contributing tier, audit-2026-04-16 UX #3 string-is-
                // canonical (kvCacheQuantization derives enableTurboQuant
                // — none/q4/q8 → false, turboquant → true even when
                // bool says false), §26 NO-REGRESSION clamp on init
                // (defaultMaxTokens<256 → reset to 32768 + persist
                // repair, healthy 8192 round-trips unchanged), BLOCKER
                // #2 schema-v0→v2 migration (qwen3 safe family bumps
                // none→turboquant; nemotron-h hybrid SSM left alone;
                // MLA families dsv4/kimi-k2/k25/k26 left alone; qwen3.5
                // -A3B / qwen3-next hybrid left alone; nil stays nil =
                // inherit-from-global; idempotent on schemaVersion>=2),
                // mcpEnabled chat-tier (iter-ralph §229 H4 propagates
                // false to resolved.mcpEnabledResolved + traces .chat;
                // unset traces .global), session engine-load fields
                // forward to resolved + trace .session, Engine.LoadOptions
                // (modelPath:from:) forwards kvCacheQuantization +
                // §403 sliding-window mode/size + enableSSMReDerive +
                // q8 still derives enableTurboQuant=false. 19 behavior
                // contracts.
                "SettingsStoreResolverContractTests.swift",
                // 2026-05-02 audit-loop iter 99 — ChatRequest JSON
                // decoding + alias-fold BEHAVIOR contract (entry point
                // for /v1/chat/completions, /api/chat, /v1/messages,
                // /v1/responses; drift breaks every multimodal client
                // silently). 30 contracts: ContentValue dual-shape
                // (.string / .parts), ContentPart image_url + video_url
                // snake-case + image_url.detail, sampler snake-case
                // (max_tokens/top_p/top_k/min_p/repetition_penalty),
                // reasoning + tool snake-case (enable_thinking/
                // reasoning_effort/tool_choice/session_id/chat_id),
                // applyMaxCompletionTokensAlias() folds when maxTokens
                // nil + max_tokens wins when both set + idempotent
                // + 0-guard, applyReasoningContainerAlias() (mlxstudio
                // #100) folds nested {reasoning:{effort}} when flat
                // nil + flat wins + empty-guard + idempotent, ToolChoice
                // 4 forms (auto/none/required/bare-name + nested
                // {type:function,function:{name}}), Tool function spec
                // decode, tool_calls assistant message, tool_call_id
                // snake-case on tool message, stream_options.include_usage
                // snake-case, thinking_budget + max_tool_iterations
                // (vMLX/Anthropic ext), frequency_penalty +
                // presence_penalty + logprobs + top_logprobs decode,
                // ContentValue.string + .parts encode→decode round-trip
                // lossless. 30/30 green.
                "ChatRequestDecodingContractTests.swift",
                // 2026-05-02 audit-loop iter 100 — OpenAIRoutes helper
                // BEHAVIOR contract (security-sensitive +
                // spec-sensitive helpers in
                // Sources/vMLXServer/Routes/OpenAIRoutes.swift).
                // Pins iter-117 §143 + iter-ralph §235 (L5)
                // redactHomeDir ($HOME→~, sibling /Users/<other>
                // → <redacted>, current user via $HOME branch, non-
                // /Users path unchanged), iter-104 §182 deriveOwnedBy
                // (org/repo→org for Qwen/mlx-community/HuggingFaceTB/
                // JANGQ-AI; bare basename→local; leading-slash safe;
                // empty→local) — drift was the old heuristic
                // mis-attributing JANG models to "dealignai" and
                // others to "mlx-community", formatTimestamp comma vs
                // dot separator (SRT vs VTT spec), hour rollover,
                // negative clamps to 0, zero baseline; singleCueSRT
                // canonical 1\n start-->end\ntext shape; singleCueVTT
                // WEBVTT\n\n header + dot separator; errorJSON
                // factory callable on .badRequest. 17 contracts.
                "OpenAIRoutesHelpersContractTests.swift",
                // 2026-05-02 audit-loop iter 101 — Engine.LoadOptions
                // bare-init alignment with GlobalSettings production
                // defaults (regression guard for the divergence found
                // during iter-100 audit). Pre-fix the bare-init path
                // (vMLXApp tray Start fallback, AdminRoutes admin/load
                // override, vMLXCLI bench harness) shipped silently
                // wrong defaults: prefillStepSize=1024 vs 2048,
                // maxCacheBlocks=500 vs 1000, kvCacheQuantization=
                // "none" while enableTurboQuant=true (internally
                // contradictory). Fix re-aligned all three; this test
                // pins so future drift fails CI loudly. Also pins
                // turboQuantBits, kvCacheGroupSize, all 8 cache-stack
                // toggles (enableJANG/PrefixCache/SSMCompanion/
                // SSMReDerive/MemoryCache/DiskCache/BlockDiskCache/
                // diskCacheMaxGB), §403 sliding-window mode+size, idle
                // lifecycle (Enabled/SoftSec/DeepSec), FlashMoE
                // slotBank=64 auto-size sentinel, advanced feature
                // toggles off (smelt/flashMoe/dflash/JangPress).
                // 12 contracts.
                "LoadOptionsGlobalSettingsAlignmentTests.swift",
                // 2026-05-02 audit-loop iter 102 — SSEEncoder JSON-helper
                // BEHAVIOR contract (wire format for every OpenAI-
                // compatible streaming endpoint; drift = silent break of
                // every streaming client). Pins chunkJSON top-level shape
                // (id/object=chat.completion.chunk/created/model/choices
                // [0]), finish_reason ALWAYS PRESENT (NSNull when
                // streaming non-terminal, string when terminal —
                // omit-when-nil drift breaks LangChain/OpenAI v2 SDK),
                // delta payload passthrough, sorted-keys deterministic
                // output (Python ensure_ascii+sort_keys parity),
                // asciiJSON ASCII-only escapes (CJK→\uXXXX, Latin-1
                // supplement→\uXXXX, emoji→surrogate pair 😀,
                // mixed-runs only-non-ASCII-escapes, ASCII unchanged,
                // empty/nested edge inputs); encodeLogprobs token/
                // logprob/bytes mandatory + top_logprobs omitted when
                // empty (OpenAI exclude_none) + nested top_logprobs
                // {token,logprob,bytes} preserved + empty input safe;
                // SSEMergedEvent exactly 2 cases (chunk+heartbeat);
                // sseHeartbeatInterval default 15s (env-overrideable
                // VMLX_SSE_HEARTBEAT_SEC, conservative for nginx 60s).
                // 18 contracts.
                "SSEEncoderHelpersContractTests.swift",
                // 2026-05-02 audit-loop iter 103 — AuthTokenBox +
                // constantTimeEquals BEHAVIOR contract (security-
                // critical). AuthTokenBox: default-init nil, ctor-set
                // readable, update() atomic + visible to subsequent
                // reads (iter-135 §161 — pre-fix the revoke-dialog
                // "any client using this key will immediately lose
                // access" promise was a lie until restart), update-
                // to-nil revokes, concurrent reads+updates safe.
                // constantTimeEquals (iter-87 §115 LAN timing-oracle
                // mitigation): equal-strings→true, different-same-
                // length→false, prefix-of-longer→false (both
                // directions; drift = attacker recovers key
                // byte-by-byte), empty==empty→true, empty-vs-nonempty
                // → false, Unicode/multi-byte UTF-8 equal→true,
                // unicode-differing→false (ï vs i), long API-key-
                // length pair correct, first-byte-wrong vs last-byte-
                // wrong both return false (early-exit `==` regression
                // catch). 13 contracts.
                "AuthTokenBoxContractTests.swift",
                // 2026-05-02 audit-loop iter 104 — Route body-size
                // limit DoS audit + regression guard. Every HTTP route
                // MUST cap req.collectBody(upTo:) — drift = unbounded
                // memory read → OOM. Pins per-route sensible limits:
                // OpenAI chat/responses/embeddings 32 MB (multi-image
                // VL headroom); OpenAI image gen+edit 16 MB; Ollama
                // /api/chat + /api/generate 32 MB (drift down breaks
                // VL base64-image clients silently); Anthropic
                // /v1/messages 32 MB; MCP routes 8 MB (chunky tool
                // calls bounded); Admin routes ≤64 KB (privileged
                // control endpoints — drift up = DoS surface widen);
                // Adapter routes 64 KB (LoRA spec JSON small);
                // comprehensive cross-file scan: every collectBody(
                // call MUST use upTo: cap, no uncapped reads. 9
                // contracts.
                "RouteBodyLimitContractTests.swift",
                // 2026-05-02 audit-loop iter 105 — JSONLEncoder helpers
                // BEHAVIOR contract (Ollama NDJSON wire format for
                // /api/chat + /api/generate; drift breaks LangChain /
                // Copilot / Open WebUI / OllamaJS latency UIs).
                // applyOllamaTimings: prompt_eval_count + eval_count
                // from Usage; nanosecond conversion (250ms = 250e6 ns —
                // ms drift breaks every spec client); eval_duration
                // derived as (total - prefill) clamped to 0 on
                // anomaly; load_duration always 0 (vMLX placeholder);
                // iter-120 §196 cache_detail passthrough; iter-125
                // §200 tokens_per_second + prompt_tokens_per_second +
                // ttft_ms vMLX-extensions in envelope; nil usage no-op;
                // nil totalMs skips total_duration + eval_duration;
                // nil prefillMs skips prompt_eval + eval. iso8601Now:
                // T/Z markers, fractional seconds, ISO8601DateFormatter
                // round-trip. parseJSONObject: valid round-trips,
                // invalid returns nil, non-object roots return nil.
                // 18 contracts.
                "JSONLEncoderHelpersContractTests.swift",
                // 2026-05-02 audit-loop iter 106 — RequestLogger
                // sensitive-data invariant + status-level + trace-id
                // contract (§107 privacy regression guard). Source-scan
                // pin: middleware MUST NOT reference .authorization /
                // x-api-key / x-admin-token headers (drift = bearer
                // tokens log in plaintext to LogStore + LogsPanel +
                // tail-to-disk hooks); MUST NOT read request.body /
                // response.body / collectBody (privacy: user prompts
                // + completions); MUST NOT read uri.query / uri.string
                // (admin tokens leak via ?admin_token=... query
                // params); MUST log request.uri.path only. Status→
                // level: 5xx→.error, 4xx→.warn, default .info.
                // Trace ID R1 §302: x-vmlx-trace-id header read,
                // `[tid=<id>]` suffix format, empty-tid suppression.
                // Elapsed-ms %.1f format + overflow-safe &- on
                // uptimeNanoseconds. minLevel param unused for
                // filtering (§305/§309/§319 LogStore global
                // live-swap). Category "server". Throw path .error +
                // re-throw. 16 contracts.
                "RequestLoggerInvariantContractTests.swift",
                // 2026-05-02 audit-loop iter 107 —
                // AdminAuthRateLimiter (iter-ralph §232 M3) BEHAVIOR
                // contract. Per-IP failed-auth counter for admin
                // token; 5/60s rolling window. 16 contracts: 1
                // failure not blocked; 5 failures block + retryAfter
                // 1-60s; 4 failures don't block (under threshold);
                // window scrolls off after 60s; strict >cutoff
                // boundary; recordSuccess clears all prior failures
                // (drift = legitimate users blocked after 4 misses);
                // peer isolation (drift = LAN-wide DoS via single
                // attacker); bounded growth on sustained 100-failure
                // attack still blocks (memory bound by recordFailure
                // cap maxFailures*4=20); retry-after ≥1s minimum;
                // fresh failure yields retryAfter≈60s. Source-scan:
                // maxFailures=5; windowSeconds=60; CRITICAL shouldBlock
                // BEFORE constantTimeEquals (blocked peers don't probe
                // timing oracle); iter-75 §103 destructive gate list
                // (/v1/adapters/load + unload + fuse + /api/delete);
                // recordFailure on miss; recordSuccess on pass. 16/16
                // green.
                "AdminAuthRateLimiterContractTests.swift",
                // 2026-05-02 audit-loop iter 108 — CORSAllowlistMiddleware
                // §331 deep BEHAVIOR contract (compensates for
                // Hummingbird .originBased echo-any-origin equivalent
                // to wildcard). 16 contracts. Constructor: empty input
                // empty allowlist (NO implicit wildcard), single origin
                // round-trips lowercased, mixed-case→lowercase
                // normalized, trailing slash + multiple trailing
                // slashes stripped, empty string filtered, duplicates
                // collapse via Set. Wildcard "*" stored LITERALLY (NOT
                // expanded — drift would defeat allowlist purpose).
                // Origin separation pins: http vs https distinct
                // (mixed-content guard), port distinct (x:8080 ≠ x),
                // subdomain distinct from parent (app.x ≠ x),
                // localhost ≠ 127.0.0.1 (common dev pattern). SOURCE-
                // SCAN: short-circuit when Origin absent/empty
                // (same-origin pass-through); OPTIONS preflight from
                // disallowed origin returns 403 .forbidden + cors_error
                // type body (drift = generic browser error); non-
                // OPTIONS strips Origin from head + preserves body
                // stream (drift = downstream CORSMiddleware echoes
                // back = allowlist defeated); Origin normalization
                // (lowercase + trim "/") matches allowlist
                // normalization. 16 contracts.
                "CORSAllowlistDeepContractTests.swift",
                // 2026-05-02 audit-loop iter 109 — ScreenshotTool §429
                // BEHAVIOR + source contract (Terminal-mode VL companion
                // to BashTool; previously zero test coverage).
                // Invocation defaults: nil region (full screen), 0s
                // delay, "screen" target. Custom round-trip preserved.
                // OpenAI schema: name=screenshot + type=function;
                // description mentions next-input attachment (drift =
                // model thinks tool does nothing visible); required
                // empty (model can call no-args); target.enum strictly
                // {screen, active_window} (drift = silent fallback to
                // screen on bad enum); region array of integer; delay
                // number type. Result struct 5 fields (path/savedBytes/
                // widthHint?/heightHint?/error?) for both success +
                // permission-denied shapes. SOURCE-SCAN: /usr/sbin/
                // screencapture path (system binary); -x -t png flags
                // always; -R x,y,w,h region argv; -W -o active_window
                // (drop shadow suppressed — VL models read shadows as
                // content); output named vmlx-sc-<unix>-<uuid>.png to
                // avoid collisions; PNG dimension extraction seek
                // offset 16 + read 8 bytes (canonical IHDR width/
                // height big-endian uint32); permission-denied hint
                // explicit "Screen & System Audio Recording" pointer
                // (not cryptic exit code); stderr captured; public
                // actor. 17 contracts.
                "ScreenshotToolContractTests.swift",
                // 2026-05-02 audit-loop iter 110 — ToolDispatcher
                // BEHAVIOR + source contract. ToolDispatchResult 4-
                // field shape + isError defaults false. SOURCE-SCAN:
                // dispatch order bash → screenshot → browser → MCP
                // namespaced (drift = MCP server with tool literally
                // named "bash" shadows in-process bash = security
                // regression because MCP version lacks sandboxing/
                // cwd-tracking); MCP namespace separator MUST be `__`
                // double-underscore (drift breaks every existing
                // server__tool config); bash cwd falls back to
                // terminalCwd (drift breaks `cd foo` threading across
                // Terminal turns); unknown tool returns JSON-shaped
                // error + isError=true + .warn log; browser tool
                // result hint "PNG attached to your next input" so VL
                // model reacts; recordScreenshot called for browser;
                // screenshot anti-loop hint "Don't try to call
                // screenshot again unless the user requests a new
                // capture"; recordScreenshot only on result.error=nil
                // (drift stages stale/missing path); §338 vmlx#47 MCP
                // argument coercion via coerceToolArguments before
                // tools/call (drift = LLMs emitting {"page":"3"} get
                // rejected by strict MCP servers); coercion gated on
                // schema lookup; MCP error JSON escapes embedded
                // quotes; dispatch start logs .info with name + id.
                // 16 contracts.
                "ToolDispatcherContractTests.swift",
                // 2026-05-02 audit-loop iter 111 — KeychainHelper
                // BEHAVIOR + invariants contract (security-critical
                // wrapper for HF token + remote API keys; previously
                // ZERO test coverage). Live macOS Keychain test with
                // unique-per-test account string + tearDown cleanup.
                // 16 contracts. Service catalog: hfToken raw value
                // "ai.jangq.vmlx.hf_token" + remoteAPIKey raw value
                // "ai.jangq.vmlx.remote_api_key" stable across releases
                // (drift = users' existing tokens orphaned, silent
                // re-auth required, perceived broken). Source pins:
                // kSecClassGenericPassword for all 3 SecItem queries
                // (load + delete + saveStatus); kSecAttrAccessible
                // WhenUnlocked (drift to AfterFirstUnlock or Always =
                // security regression). BEHAVIOR: save→load round-trip
                // bit-equivalent; has() true after save / false on
                // missing; save(nil) deletes (drift = stale token in
                // keychain that has() returns true on but token is
                // invalid); save("") deletes; save(nil) on missing =
                // idempotent true; delete missing = errSecItemNotFound
                // treated as success; double-delete idempotent; UPDATE-
                // first re-save replaces value (drift = saveStatus
                // tries SecItemAdd first → errSecDuplicateItem on
                // re-save); service isolation (hfToken delete doesn't
                // affect remoteAPIKey same account); account
                // isolation; Unicode/UTF-8 tokens round-trip; no
                // empty-string-stored state via save("") clears entry.
                // 16/16 green.
                "KeychainHelperContractTests.swift",
                // 2026-05-02 audit-loop iter 112 — HuggingFaceAuth
                // invariants + ValidationState contract (auth
                // coordinator for HF gated repos; previously zero
                // coverage). Source-scan + ValidationState BEHAVIOR.
                // 21 contracts. whoami-v2 URL exact (drift to /api/
                // whoami v1 = HF returns 404 since 2024); Bearer
                // prefix on Authorization header; ephemeral
                // URLSession (no shared/cached state); cachePolicy
                // .reloadIgnoringLocalCacheData; 401/403 → "Token
                // rejected by HuggingFace" (UI banner reads exact
                // string); generic 4xx/5xx surfaces status code
                // ("HF returned 500"); network errors surface
                // localizedDescription; username extracted from
                // `name` key (whoami-v2 spec); malformed-response
                // distinct error message; DownloadManager bound by
                // weak ref (drift = engine retained forever, memory
                // leak); pushToAllManagers prunes dead refs;
                // bind() pushes current token immediately to
                // freshly-bound manager (drift = gated downloads
                // fail silently between bind and next save);
                // clear() deletes Keychain + pushes nil to managers
                // + resets all @Published state (drift = stale token
                // survives "Forget" UI button); save() trims
                // whitespace + newlines (clipboard pastes); empty
                // trimmed → clear() short-circuit; valid path
                // persists Keychain BEFORE @Published state update;
                // validate=false still persists + pushes;
                // currentToken() returns nil for empty stored value
                // (drift = HF search panel attaches empty Bearer →
                // 401 silently). ValidationState equatable: .unknown
                // self-equal; .valid value-equal on username; .invalid
                // value-equal on reason; distinct cases not equal.
                // 21 contracts.
                "HuggingFaceAuthInvariantContractTests.swift",
                // 2026-05-02 audit-loop iter 115 — JangPress ×
                // {JANG/JANGTQ load, hybrid SSM, L2 disk cache, paged
                // cache, TurboQuant KV cache, expert dispatch, RoPE,
                // Hadamard, codebook} integration audit + regression
                // guard. 24 contracts. Axis E ⊥ axes A-D + SSM
                // orthogonality (controller MUST NOT reference any
                // KV/SSM cache type — same negative test on
                // JangPressMmapTier). 13 tile-pattern coverage; VL
                // prefix matcher 5 variants + backbone for Nemotron;
                // hybrid SSM regex MUST NOT match mixer.in_proj /
                // dt_proj / A_log (Mamba2 SSM state — would corrupt).
                // Engine init: cacheCoordinator BEFORE JangPress block;
                // toggles independently togglable from TurboQuant /
                // prefix / disk / block / SSM. Defaults pinned: .mmap
                // backend, .soft force mode, emberOn = opts || g (OR
                // fallback), pctRaw = opts!=0 ? opts : g, prefetchOn
                // = opts && g (single-side disable wins). Init logging
                // includes shards/experts/routedBytes/pct/keepHot;
                // missing loadedModelPath warns; AdminRoutes registers
                // /v1/cache/jangpress; production defaults pinned on
                // BOTH LoadOptions and GlobalSettings (enableJangPress=
                // true, pct=70, .mmap, .soft, prefetch=true);
                // controller is read-only; pct clamped [0,100];
                // alwaysHotFraction = 1.0 - Double(pct)/100.0;
                // quiesce timeout 30_000ms both paths; embed tier
                // (component F) co-init; .none backend clean no-op.
                "JangPressIntegrationAuditTests.swift",
                // Iter 143 — pin force-prefault opt-in
                // (`VMLX_JANGPRESS_FORCE_PREFAULT=1`). 8 contracts:
                // acquire reads forcePrefaultEnabled, dispatches between
                // touchRangeResident and advise(.willNeed), env var name
                // is canonical, accepted truthy literals are
                // 1/true/on/yes, JangPressShard exposes
                // touchRangeResident(range:), helper aligns to page
                // boundaries, walks via pageStride, returns page count,
                // doc cites the 0.15 tok/s Laguna regression that
                // motivated the opt-in.
                "JangPressForcePrefaultContractTests.swift",
                // Iter 144 — pin Stream.swift production-audit fixes:
                // task-registration race (capture-after-create TaskBox
                // + match-aware clear), MLXError path scrub, JangPress
                // TaskLocal rebind in detached SSM re-derive,
                // stopHit-tail-drop trace doc. 11 contracts.
                "StreamRaceFixContractTests.swift",
                // Iter 144 — pin lifecycle production-audit fixes:
                // GenerationLock cancel-during-wait (throwing
                // continuation, dropWaiter throws CancellationError,
                // withLock propagates throw); deepSleep JangPress
                // mmap/mach/embed release (mirrors stop() teardown);
                // transition() short-circuits on unchanged state. 9 contracts.
                "Iter144LifecycleFixContractTests.swift",
                // Iter 144 — load-task race + tool-call nil-args + Qwen
                // vision marker fixes. 11 contracts pinning:
                // _LoadTaskBox capture-after-create, takeCurrentLoadTask,
                // clearCurrentLoadTaskIfMatches, prior-task cancel +
                // await, self-register inside body, defer-based clear,
                // no-detached-registration regression, nil-arguments
                // filter at both parse sites, drop-log, Qwen vision
                // markers in hasAnyMarker.
                "Iter144LoadAndToolFixContractTests.swift",
                // Iter 144 (iter 5) — Auth/CORS audit fixes:
                // constantTimeEquals uses Int (no UInt8 truncation),
                // Bearer scheme case-insensitive, ?api_key= query
                // string fallback, admin 401 via JSONSerialization,
                // tool error result framing. 11 contracts including
                // 2 behavioural (constantTimeEquals null-pad, query
                // param parsing).
                "Iter144AuthAndToolFrameContractTests.swift",
                // Iter 144 (iter 6) — MCP stderr drain + DownloadManager
                // partial-cleanup, 206 stream-copy, KVO clamp. 9 contracts.
                "Iter144McpAndDownloadFixContractTests.swift",
                // Iter 144 (iter 7) — cache + library audit fixes:
                // ModelLibrary deleteEntry strict-subpath fence,
                // DiskCache atomic .tmp+rename, PagedCache known-race
                // doc. 7 contracts.
                "Iter144CacheAndLibraryFixContractTests.swift",
                // Iter 144 (iter 8) — UI/UX production-audit fixes:
                // newSession resets reasoningEnabled, selectSession
                // marks intentionalStop, standby(.soft) banner gates
                // spinner on isStreaming, InputBar onSubmit guards on
                // !isGenerating, TrayItem openAppWindow prefers main
                // over downloads, LogsPanel reverseOrder scroll
                // target, LogsPanel filter cache, RequestLogPanel
                // pause buffer + flush. 9 contracts.
                "Iter144UIFixContractTests.swift",
                // 2026-05-02 audit-loop iter 116 — JangPress lifecycle
                // brackets × reasoning ON/OFF/budget invariants.
                // CRITICAL guard against the reasoning-routing
                // regression class (3+ historical regressions in
                // feedback_reasoning_off_ui_stuck.md). 11 contracts:
                // willStartInference fires BEFORE reasoning parser
                // setup (so wake is unconditional); didFinishInference
                // in cleanup block AFTER stream consumer (runs on
                // cancel/throw/budget); UNGATED by effectiveThinking /
                // harmonyActive / suppressReasoning / hasTools (model-
                // state, not request-state); exactly ONE willStart +
                // ONE didFinish per Stream.swift; didFinish AFTER
                // clearCurrentStreamTask; thinking-budget-exhausted
                // path doesn't double-wake; brackets use optional
                // chain (no force-unwrap = no crash on disabled);
                // failsafe nil-check doc + .mmap-no-per-turn-wake doc
                // + quiesce-timer-cancel-on-next-willStart doc all
                // pinned. 11 contracts.
                "JangPressReasoningLifecycleContractTests.swift",
                // 2026-05-02 audit-loop iter 117 — MiniMax fp32 router
                // gate fix regression guard (upstream agent iter 23).
                // Bug: `MiniMaxSparseMoeBlock.callAsFunction` was matmul'ing
                // gate Linear at bf16 — Python ref does
                // `gates = self.gate(x.astype(mx.float32))`. With 154
                // experts (post-REAP-prune), bf16 precision causes
                // near-tied scores that argpartition top-k picks
                // differently each run = non-deterministic garbage at
                // T=0. Fixed in 3 sites this repo (MiniMax.swift:157
                // standard MoE forward, MiniMax.swift:567 flashMoE
                // shim router closure, MiniMaxJANGTQ.swift:178 quant
                // variant). 7 contracts: all 3 sites have
                // `gate(x.asType(.float32))`; rationale comment cites
                // Python parity (mlx_lm/models/minimax.py:178) +
                // canonical Python expression + bf16-causes-garbage
                // rationale + argpartition top-k mechanism; flashMoE
                // shim router closure casts; NO bare `gate(x)` call
                // without fp32 cast in either MiniMax file (drift =
                // silent re-intro of bug); doc cites vmlx-swift-lm
                // twin so paired fix isn't forgotten.
                "MiniMaxFP32RouterGateContractTests.swift",
                // 2026-05-02 audit-loop iter 118 — SSM (axis F) ×
                // JangPress (axis E) orthogonality + hybrid SSM
                // compatibility. 14 contracts. LoadOptions defaults
                // (enableSSMReDerive ON, both togglable independently);
                // GlobalSettings independent toggle; Engine.swift
                // gates SSM companion on `isHybrid && g.enableSSMCompanion`
                // — JangPress activation INDEPENDENT of isHybrid (pure-
                // attention models also benefit); SSMCompanionDiskStore
                // MUST NOT reference any JangPress type; SSMReDerive
                // MUST NOT reference JangPress; mmap tier regex MUST
                // NOT match Mamba2 SSM state weights (mixer.in_proj /
                // dt_proj / A_log / D / conv1d / x_proj — would
                // corrupt hybrid models); SSM companion disable log
                // independent of JangPress; all 4 combinations
                // (hybrid×{ssm on/off}×{jp on/off}) compose; AGENTS.md
                // cites hybrid SSM compatibility + Nemotron-H + 23
                // MoE layers; CACHE-ARCHITECTURE.md lists SSM as
                // orthogonal layer.
                "SSMJangPressOrthogonalityContractTests.swift",
                // 2026-05-02 audit-loop iter 119 — JangPress × VL
                // multi-turn (image_url + mediaSalt) + audio
                // (Parakeet/Whisper) orthogonality. 11 contracts.
                // computeMediaSalt called BEFORE coord.fetch (mediaSalt
                // is part of cache-key — VL prefix cache hit/miss is
                // independent of JangPress); coord.fetch receives
                // mediaSalt parameter (drift = text-only key, stale
                // hits on no-image entries); willStartInference NOT
                // conditioned on mediaSalt / preFetchSalt (model-state
                // brackets); coord.fetch signature includes tokens +
                // mediaSalt + genPromptLen together; JangPress
                // activation does NOT check vision_config / has_vision
                // / is_mllm / isMllm / VisualConfig (VL families also
                // benefit from MoE expert reclaim); mmap tier handles
                // no-MoE bundles gracefully (Whisper/Parakeet have
                // 0 routed-expert tiles); engine logs 0-expert bundle
                // as armed (not error); toggling JangPress doesn't
                // mutate any unrelated LoadOptions field; AGENTS.md
                // lists 5 VL prefixes + backbone for Nemotron; sniff
                // optimization documented.
                "JangPressVLMultiTurnContractTests.swift",
                // 2026-05-02 audit-loop iter 120 — JangPress API
                // surface + cacheStats contract. 13 contracts.
                // AdminRoutes registers GET /v1/cache/jangpress;
                // response top-level key {"jangPress":...}; none-
                // fallback `stats[jangPress] ?? {backend: none}`;
                // EngineError mapper used (not 501 fallback).
                // cacheStats() mmap variant 4 fields (backend +
                // shardCount + expertCount + totalRoutedBytes +
                // byLayer); mach variant 11 fields including 3
                // pressure counters; none variant minimal {backend:
                // none}; jangPressEmbed separate top-level key.
                // Field naming MUST be camelCase (no snake_case).
                // AGENTS.md documents GET endpoint + backend states +
                // 4 CLI flags. CLI declares all 4 flags.
                "JangPressAPISurfaceContractTests.swift",
                // 2026-05-02 audit-loop iter 121 — JangPress × RAM
                // accounting + M-chip support. 13 contracts.
                // Controller + MachCache both use
                // DispatchSourceMemoryPressure (Apple canonical, all
                // M-series + Intel); both dispatch on utility-qos
                // queue (NOT main, doesn't compete with UI/inference);
                // pressure source listens for .warning + .critical
                // events. AGENTS.md cites RSS double-count quirk
                // (resident_size counts each mapping; phys_footprint
                // is real) + mmap pages shared underlying physical
                // pages. NO Metal version gate (no MTLGPUFamily /
                // device.supportsFamily / MTLDevice.supportsFamily)
                // across all 5 JangPress source files; no unnecessary
                // OS version gate (#available macOS 26+/27+). Per-RAM
                // tier table covers 256+/128/64/32/16/8-16 GB rows;
                // doc cites Apple Silicon target. Controller holds
                // per-instance quiesceTimer (not global); queue label
                // ai.jangq.vmlx.jang-press.
                "JangPressMemoryAndChipContractTests.swift",
                // 2026-05-02 audit-loop iter 122 — JangPress × jang_config
                // + generation_config orthogonality. JangPress (axis E)
                // is a runtime mmap/Mach VM control — must NOT gate on
                // model bundle quant declarations (turboquant, mxtq_bits,
                // weight_format, hadamard, capabilities) or sampling
                // defaults (eos_token_id). Pinned: JangConfig + Gen-
                // ConfigFile carry ZERO JangPress fields; setupCache-
                // Coordinator JangPress activation block does NOT read
                // loadedJangConfig / weightFormat / hadamard / capabilities
                // / turboquant; activation gate is pure boolean
                // `emberOn = opts.enableJangPress || g.enableJangPress`;
                // Stream willStartInference / didFinishInference brackets
                // independent of jang/gen_config + loadedModelDefaults;
                // JangConfigRepair §411 + 5 JangPress tier sources free
                // of JangConfig / GenerationConfigFile imports; LoadOptions
                // ↔ GlobalSettings JangPress defaults symmetric (iter-101
                // alignment regression class extended to JangPress);
                // JANGTQ-native (weight_format=="mxtq") composes with
                // JangPress at page level (mmap tier doesn't parse
                // weight_format); doc cites JANGTQ compatibility.
                // 12 contracts.
                "JangPressJangGenConfigOrthogonalityContractTests.swift",
                // 2026-05-02 audit-loop iter 123 — JangPress is per-LOAD,
                // NOT per-chat / per-request. Mid-stream flip would
                // corrupt cold-tile state machine (mmap madvise / Mach
                // VM purgeable in-flight). Pinned: ChatSettings tier 3
                // + RequestOverride tier 4 + ChatRequest decoder + Set-
                // tingsStore 4-tier resolver carry ZERO JangPress fields
                // (camel + snake variants forbidden); Stream.swift per-
                // message dispatch MUST NOT read JangPress from chat /
                // request / resolved tier — only from emberController
                // ref captured at load; GlobalSettings + LoadOptions are
                // the canonical homes (5 fields: enable / pct / prefetch
                // / backend / forceMode); doc cites per-load applica-
                // tion (LoadOptions / "at load" / "per-load"). 9
                // contracts.
                "JangPressPerLoadTierContractTests.swift",
                // 2026-05-02 audit-loop iter 124 — JangPress CLI flag
                // wiring + GlobalSettings writeback. Pinned: --enable-
                // jangpress is canonical (matches doc + property name);
                // --enable-routed-expert-cache accepted as legacy alias
                // on the same property; all 4 props (enableJangPress /
                // jangPressCompressPct / jangPressBackend / jangPress-
                // ForceMode) declared; writeback site exists for each
                // (g.X = ..., dirty = true); compressPct clamped
                // [0,100]; backend + forceMode .lowercased() before
                // writeback (case-sensitive raw value dispatch); help-
                // text for compressPct cites iter-113 default (70),
                // not stale "Default 0". 8 contracts.
                "JangPressCLIWriteBackContractTests.swift",
                // 2026-05-02 audit-loop iter 125 — pct sweep × bundle
                // format / family contract. Engine MUST clamp pctRaw
                // to [0,100] + compute alwaysHotFraction = 1.0 -
                // pct/100.0 (linear inversion); JangPressMmapTier MUST
                // pass hotPercent = 100 - pct; all 13 routed-expert
                // tile-name regex patterns A-M present (covering
                // Qwen3.5-MoE / Mistral 4 / DSV3.x / Kimi K2 / Laguna /
                // Qwen3.6 / MiniMax M2.7 / Holo3 / DSV4 / Nemotron H
                // Cascade / Nemotron MXFP4 / Nemotron Cascade-2);
                // parseRoutedExpertName behavior verified on B / E /
                // F / H / J (per-expert, ids decoded) + A / C / L
                // (stacked, synthetic expert 0); negative cases (em-
                // bed / lm_head / self_attn) return nil; doc lists all
                // 13 Pattern A-M labels. 13 contracts.
                "JangPressPctSweepAndTilePatternContractTests.swift",
                // 2026-05-02 audit-loop iter 126 — Regression guard for
                // ml-explore/mlx#3461 + fix mlx#3462. Vendored mlx
                // osaurus-0.31.3 (MLX 0.31.1) ships the unfixed pattern:
                // commandBufferWithUnretainedReferences() + buffers
                // allocated MTLResourceHazardTrackingModeUntracked
                // (allocator.cpp:15). Swift structured concurrency
                // dropping MLXArray ref between encode + CB completion
                // hits "Invalid Resource" race (caught upstream under
                // TurboQuant KV cache load on M5 Max @ qwen35-35b-a3b
                // B=17/B=32, 0/10 → 10/10 after fix). vMLX patches
                // device.cpp:405 to use commandBuffer() retained
                // variant (~2.4% throughput cost). Pinned: device.cpp
                // does NOT contain commandBufferWithUnretainedReferences;
                // device.cpp uses commandBuffer() retained variant;
                // patch comment cites mlx#3461 / mlx#3462 + TurboQuant
                // trigger so a future mlx resync that reverts the patch
                // can find the upstream context. 4 contracts.
                "MLXUnretainedResourceRegressionTests.swift",
                // 2026-05-03 Codex — MLX StreamOrDevice wrapper
                // correctness. User hit `expected a non-empty mlx_stream`
                // during safetensors load; source audit found
                // StreamOrDevice.stream(custom) silently substituted
                // Device.defaultStream(). Pins exact-stream preservation.
                // 2 contracts.
                "StreamOrDeviceContractTests.swift",
                // 2026-05-02 audit-loop iter 127 — mlxstudio#138 parity
                // fix: kvCacheQuantization ↔ enableTurboQuant CLI write-
                // back symmetry. SettingsStore resolver derives Bool
                // from canonical string (`enableTurboQuant ⇔ kvCache-
                // Quantization == "turboquant"`) at SettingsStore.swift
                // :485 (iter-101 pin), so CLI flags that touched only
                // ONE of the pair were silently re-derived. Pinned:
                // `--disable-turbo-quant` writes BOTH flags + demotes
                // string off "turboquant"; `--enable-turbo-quant`
                // writes BOTH flags + promotes string to "turboquant";
                // `--kv-cache-quantization <q>` writes string AND de-
                // rives Bool; CLI patch cites mlxstudio#138 / Iter 127;
                // resolver derive line preserved; LoadOptions + Glob-
                // alSettings defaults satisfy the (Bool ⇔ String ==
                // "turboquant") invariant. 7 contracts.
                "KVCacheQuantizationCLISymmetryTests.swift",
                // 2026-05-02 audit-loop iter 128 — vmlx#121 / vmlx#133
                // fix: NSOpenPanel macOS 26 XPC connection invalidated
                // (Code=4099) on ad-hoc-signed builds. NSOpenPanelSafe
                // (Sources/vMLXApp/Common/) detects XPC failure
                // (panel returned no URL in <50ms = no UI rendered)
                // and falls back to NSAlert + NSTextField manual-path
                // entry. ModelDirectoriesPanel.pickAndAddDir uses
                // the helper. Pinned: helper is public enum with
                // pick(configure:fallbackTitle:fallbackMessage:
                // canChooseFiles:); PickResult shape (url / used-
                // Fallback / failureReason); XPC heuristic is
                // elapsed<0.05 + cancelled response; manual fallback
                // shape (NSAlert + NSTextField + Use Path/Cancel);
                // tilde expansion + path existence + directory-vs-
                // file validation; ModelDirectoriesPanel uses helper
                // (no raw panel.runModal in that function body);
                // patch sites cite vmlx#121 / #133 / Iter 128. 9
                // contracts.
                "NSOpenPanelSafeContractTests.swift",
                // 2026-05-02 audit-loop iter 130 — vmlx#131 fix:
                // MCPServerConfig now accepts a dedicated `headers:
                // [String: String]?` field for SSE / Streamable-HTTP
                // transport authentication (Exa / GitHub / Anthropic-
                // hosted MCP). Pinned: public optional headers dict
                // on MCPServerConfig; CodingKeys for both MCPServer-
                // Config + RawServerEntry include `case headers`;
                // RawServerEntry.toConfig forwards headers; MCPJSON-
                // RPCClient SSE startup AND POST paths both iterate
                // server.headers + set on URLRequest; default init
                // leaves headers nil (back-compat); JSON round-trip
                // preserves headers; Claude-Desktop `mcpServers.<name>
                // .headers` decoder works; patches cite vmlx#131 /
                // Iter 130. 9 contracts.
                "MCPHeadersFieldContractTests.swift",
                // 2026-05-02 audit-loop iter 132 — JangPress recordRoute
                // API surface (DORMANT). Pinned: recordRoute(layer:
                // experts:) signature exists; routingFreq storage +
                // totalRoutes counter present; lock-protected `&+=`
                // overflow-wrapping bumps; wire-up TODO cited in
                // JANGPRESS-PRODUCTION.md + JANGPRESS-STATUS.md;
                // dormancy guard — zero `.recordRoute(` callers in
                // Sources/ outside the controller, OR a JangPress-
                // RouteSink/Recorder protocol wires the call (so a
                // partial wire-up can't silently land). 5 contracts.
                "JangPressRecordRouteSurfaceContractTests.swift",
                // 2026-05-03 Codex — canonical JangPress mmap + load
                // warmup policy. Pins production-on canonical
                // safetensors mmap for JANG/JANGTQ bundles, disable
                // escape hatch, DSV4/giant-model blocking warmup
                // deferral, and non-reflection memory-pressure logging.
                // 3 contracts.
                "JangPressCanonicalMmapAndWarmupPolicyTests.swift",
                // 2026-05-03 Claude port — JangPressCanonicalExpert-
                // Advisor (router-aware per-layer hot-set tracker)
                // adapted from upstream osaurus reference. Pins:
                // singleton + Cmlx C-ABI direct call (no dlsym),
                // configure() takes explicit booleans (no LoadOptions
                // struct), per-load counter reset, env knobs
                // (JANGPRESS_ROUTER_*), Engine wires configure after
                // canonical mmap activation + disables on unload/load
                // failure, cacheStats surfaces routerAdvisor sub-key,
                // telemetry forwards realized [Int32] indices, hot-set
                // overflow LRU eviction + size-cap + force-disable env.
                // 13 contracts (8 source-shape + 5 behavior).
                "JangPressCanonicalExpertAdvisorTests.swift",
                // 2026-05-02 audit-loop iter 133 — chat persistence +
                // model-loaded-on-chat invariants. Pinned: ChatSettings
                // .modelAlias field exists + Codable round-trip pre-
                // serves it; ChatViewModel reads `chatOverrides?.model-
                // Alias ?? fallbackModelPath`; ChatScreen.loadChatMod-
                // elInline consults `engine.settings.chat(chatId)?.
                // modelAlias` first; ChatScreen rebinds on chat switch
                // via `.task(id: vm.activeSessionId)`; fast-path order
                // (running/loading → rebind only, .standby → wakeFrom-
                // Standby, .stopped/.error → startSession) prevents
                // redundant weight reload Metal crashes; SettingsStore
                // exposes chat setter so modelAlias survives app re-
                // start. 7 contracts.
                "ChatPersistModelLoadInvariantsTests.swift",
                // 2026-05-02 audit-loop iter 134 — i18n no-regression
                // for picker-fallback strings introduced in iter 128/
                // 129. Pinned: L10n.PickerFallback namespace exposes
                // 10 canonical entries (modelDir / cwd / mcpJson /
                // tlsFile / chatWorkingDir titles+messages, manual-
                // PathBanner + manualPathFailure formats); every
                // entry has all 4 locales (en/ja/ko/zh) non-empty;
                // format strings carry `%@` placeholder; locales
                // distinct (no copy-paste bug); every NSOpenPanel-
                // Safe.pick call site routes fallbackTitle + fallback-
                // Message through L10n.PickerFallback.* (no raw
                // English literals). 5 contracts.
                "I18nNoRegressionPickerFallbackTests.swift",
                // 2026-05-03 audit-loop iter 135 — JangPress UI
                // controls (master toggle + compress % stepper +
                // backend picker + force mode picker + prefetch
                // toggle) added to SessionConfigForm.advancedSection.
                // Audit found task #196 was marked complete but
                // delivered CLI parity only — vMLXApp UI had ZERO
                // JangPress controls. This iter ships the missing
                // controls writing to GlobalSettings (per task #195
                // per-load not per-chat invariant). 8 contracts.
                "JangPressUIControlsContractTests.swift",
                // 2026-05-04 iter 143 — JangPressPrestacker port from
                // vmlx-swift-lm. Pins prestacker file + env knobs +
                // Engine.load wiring + LoadOption + GlobalSettings +
                // SettingsStore forwarding + CLI flag presence.
                "JangPressPrestackerContractTests.swift",
                // 2026-05-04 iter 144 — pin phys_footprint sampling +
                // advisor snapshot dump in JANGPressMultiTurn bench.
                // Reference doc explicitly warns RSS alone is wrong
                // for mmap-backed weights; this contract prevents
                // silent regression to RSS-only output.
                "JangPressMultiTurnBenchInstrumentationTests.swift",
                // 2026-05-03 audit-loop iter 136 — vmlx-swift-lm
                // parity audit (cache reuse / SSM re-derive / L2
                // disk / TurboQuant KV / SLIDING-1 / framework
                // optimizations). Cross-checked against
                // /Users/eric/vmlx-swift-lm/docs/{CACHE-NUANCES,
                // CACHE-STATUS,SLIDING-WINDOW}.md +
                // /Users/eric/vmlx-swift-lm/SWIFT-PERF-FIXES.md.
                // Pins SLIDING-1 LayerKind + rotating round-trip,
                // Gemma4 fp16→fp32 SDPA upcast, Memory.clearCache
                // every 256 tokens, processor nil fast-path,
                // needsCacheQuantization guard, _compiledComputeG,
                // TurboQuantKVCache extract via tq.state, existing
                // 9-VLM ChunkedPrefillVLM coverage. Documents the
                // 4-Qwen-family-VLM gap as iter-137+ followup. 12
                // contracts.
                "VmlxSwiftLmParityAuditTests.swift",
                // 2026-05-03 audit-loop iter 137 — chat → model
                // compatibility warning. ChatScreen.select(_ alias)
                // now runs CapabilityDetector.detect on BOTH the
                // chat's prior modelAlias AND the new alias being
                // selected, and surfaces a non-blocking
                // vm.bannerMessage when family/cacheType/reasoning
                // parser/toolParser differ and the chat already
                // has turns. Recommends starting a new chat. 3
                // contracts.
                "ChatModelCompatWarningContractTests.swift",
                // 2026-05-03 audit-loop iter 138 + 139 — gateway
                // protocol fan-out. Pre-iter-138 gateway exposed only
                // OpenAI /v1/chat/completions + /v1/completions +
                // embeddings + images. Per user directive added
                // /v1/messages (Anthropic), /api/chat + /api/generate
                // (Ollama), /v1/responses (OpenAI Responses).
                // AnthropicRoutes.handleMessages, OllamaRoutes.handle{
                // Chat,Generate}, OpenAIRoutes.handleResponses all
                // extracted as public static helpers; per-session
                // routes are 3-line wrappers; gateway delegates with
                // model-keyed engine resolution. 8 contracts (iter 139
                // expanded the iter-138 count).
                "GatewayProtocolFanoutContractTests.swift",
                // 2026-05-03 audit-loop iter 140 — /health cache
                // visibility + continuous-batching honesty. Adds
                // explicit `continuous_batching: false` boolean +
                // `cache_summary` block (paged hit_rate/hits/misses,
                // disk hit_rate/bytes, ssm entry count, jangpress
                // backend, architecture hybrid_ssm/sliding_window/
                // turbo_quant flags) to /health payload. 6 contracts.
                "HealthCacheVisibilityContractTests.swift",
                // 2026-05-03 audit-loop iter 141 — pytorch-Swift port +
                // vLLM-style paged-cache + text-only turn cache reuse.
                // Pins MLXNN imports across 10 load-bearing core files,
                // computeMediaSalt nil-on-text-only invariant,
                // PagedCacheManager + BlockHashMap + CacheBlock
                // ref-counted pool surface (vLLM PagedAttention
                // analog), Stream.swift text-only short-circuit
                // documentation, CacheCoordinator threads mediaSalt
                // and exposes matchedTokens. 10 contracts.
                "PytorchPortAndVLLMCachingContractTests.swift",
                // 2026-05-03 audit-loop iter 142 — token-by-token
                // streaming + heartbeat. Pins TokenIterator's
                // asyncEval per-token pipeline, SSE/NDJSON
                // sseMergeWithHeartbeat coverage on all 5 encoders
                // (chat/completions/responses/ollama-chat/ollama-gen),
                // `: keep-alive\n\n` SSE comment on heartbeat,
                // VMLX_SSE_HEARTBEAT_SEC env override (default 15s),
                // HeartbeatHandle cancel-on-terminate (iter-18 fix),
                // streaming response headers (text/event-stream +
                // no-cache), `[DONE]\n\n` terminator, per-token
                // data-frame guarantee (no batching). 8 contracts.
                "StreamingHeartbeatContractTests.swift",
                // 2026-05-02 audit-loop iter 94 — FluxKit JangBridge
                // (vMLXLMCommon.JangLoader thin wrapper for Flux
                // models). Namespace enum, isJangModel + loadConfig
                // (nil on non-JANG NOT throw) + detect (tuple) all
                // forward to vMLXLMCommon.JangLoader, doc cites
                // bridge-vs-reexport rationale (533 lines avoid),
                // doc lists 4 supported families (Mistral 4 / Gemma
                // 4 / Nemotron H / Qwen 3.5), @preconcurrency import
                // (Swift 6 strict-concurrency compat), 3 entrypoints
                // public static. 9/9 green.
                "FluxJangBridgeContractTests.swift",
                // 2026-05-02 audit-loop iter 93 — FlowMatchEulerScheduler
                // (FLUX/ZImage/Qwen-Image/FIBO rectified flow).
                // Sendable, sigmas length steps+1 (Euler N+1 for
                // N transitions), 0...steps inclusive linspace,
                // flow-match shift formula `shift*σ/(1+(shift-1)*σ)`,
                // timesteps = sigmas[..steps] * 1000, computeShift
                // resolution lerp clamp [0,1] mflux constants
                // (256/4096), Euler step `latent + velocity *
                // (σ_next - σ_current)`, mflux defaults
                // (imageSeqLen=4096, baseShift=0.5, maxShift=1.15),
                // doc cites mflux Python ref + Euler formula + 5
                // user-facing models, 3 fields immutable let,
                // stepCount convenience accessor. 13/13 green.
                "FlowMatchEulerSchedulerContractTests.swift",
                // 2026-05-02 audit-loop iter 92 — FluxKit Requests +
                // Events surface. ImageGenRequest Sendable +
                // canonical defaults (1024×1024/20 steps/3.5
                // guidance/1 image/.png), ImageEditRequest mask
                // white-edit/black-keep convention doc + 0..1
                // strength range + width/height Optional `nil →
                // match source`, VideoGenRequest WAN 2.1 defaults
                // (1280×720/121 frames @ 24fps/50 steps/5.0
                // guidance), ImageGenEvent 5 cases + VideoGenEvent
                // 5 cases with frame field, failed.hfAuth Bool
                // for "Add HF token" CTA, preview optional doc,
                // step 1-indexed doc, ImageFormat 3 cases, all 4
                // request structs Sendable. 13/13 green.
                "FluxRequestEventsContractTests.swift",
                // 2026-05-02 audit-loop iter 91 — TransformersTokenizer
                // Loader bridge (swift-transformers ↔ vMLXLMCommon).
                // Conforms to vMLXLMCommon.TokenizerLoader, catches
                // unsupportedTokenizer + retries with shadow dir,
                // doc lists known unsupported names, iter-98 §125
                // shadow cleanup via defer (avoid /var/folders
                // accumulation), fallbackTokenizerClass family-aware
                // (qwen → Qwen2Tokenizer, llama → LlamaTokenizer,
                // gemma → GemmaTokenizer, default PreTrainedTokenizer)
                // + checks model_type top-level OR text_config
                // (VLM-nested), shadow path tmp+UUID for concurrency,
                // shadow symlinks (NOT copies — multi-GB safetensors
                // guard), missing tokenizer_config.json minimal
                // fallback, Bridge @unchecked Sendable, decode
                // tokenIds → decode tokens name remap doc, macro
                // replacement historical context. 12/12 green.
                "TransformersTokenizerLoaderContractTests.swift",
                // 2026-05-02 audit-loop iter 90 — WiredMemoryUtils +
                // WiredMemoryMeasurement (real-prefill memory budget
                // derivation). Measurement Sendable + 6 immutable
                // fields, totalBytes max(0, …) per-field clamp,
                // namespace enum, pad fallback chain (eos →
                // unknown → 0), makeTokenIds count>0 guard +
                // clip/pad logic + empty-encode pad-fallback (no
                // infinite loop), prefillOnly materializes logits
                // both branches + maybeQuantizeKVCache both branches
                // + nil-on-empty-cache, helpers private static, seed
                // text default ` hello` (space-prefix encodes
                // cleanly across tokenizers). 13/13 green.
                "WiredMemoryUtilsContractTests.swift",
                // 2026-05-02 audit-loop iter 89 — RemoteEngineClient
                // (HTTP-backed engine for remote sessions). public
                // actor, Kind 3 cases (openai/ollama/anthropic) +
                // lowercase fallback to .openai (wrong-case misroute
                // guard), Connection 3 cases (connecting/connected/
                // unreachable), 5 RemoteError cases + LocalizedError
                // + 200-char body truncation, 4 immutable config
                // fields, iter-127 §153 liveStreamTask Task<Void,
                // Error> (NOT URLSessionDataTask — cancel no-op
                // bug guard) + doc cites cancel route silent-fail
                // rationale, Q4 §300 connection private(set) default
                // connecting + idempotent health probe, apiKey
                // empty-string normalized to nil, doc cites same-
                // signature design + 3 auth modes per protocol.
                // 13/13 green.
                "RemoteEngineClientContractTests.swift",
                // 2026-05-02 audit-loop iter 88 — Engine.streamReal
                // cancellation + reasoning routing contract.
                // 6-section doc structure (reasoning split / off
                // suppression / tool-call streaming / TTFT / metrics
                // / idle), §15 NO-REGRESSION reasoning-OFF route to
                // content (NOT strand UI), Python tool-call markers
                // cited (12 from server.py._TOOL_CALL_MARKERS),
                // 3-prong cancellation fix (currentStreamTask expose
                // + multi-point isCancelled checks +
                // withTaskCancellationHandler around perform), Python
                // SimpleEngine prefill-not-interruptible bug
                // inheritance + fix doc, AsyncThrowingStream<StreamChunk,
                // Error> signature, Evaluate.swift:1611 cross-ref
                // for prepare(input:) usage, recordTokenBatch per
                // burst + recordRequest on completion. 11/11 green.
                "StreamRealCancellationContractTests.swift",
                // 2026-05-02 audit-loop iter 87 — MediaProcessing
                // CIImage utilities (shared image preprocessing
                // surface for every VLM). sRGB↔linear tone curve
                // filters (correct directions), bestFitScale
                // min(other/size) per axis + bestFit pixel-aligned
                // round, aspectRatioForResample 1/input*desired
                // formula, resampleLanczos lanczosScaleTransform +
                // scale=yScale + aspectRatio=xScale/yScale,
                // resampleBicubic bicubicScaleTransform, both crop-
                // to-exact-rect, normalize PyTorch torchvision URL
                // doc + per-channel rVector/gVector/bVector inverse-
                // std + bias=-mean/std + alpha passthrough,
                // asMLXArray planar [1, C, H, W] shape doc. 12/12
                // green.
                "MediaProcessingContractTests.swift",
                // 2026-05-02 audit-loop iter 86 — VLMModelFactory
                // coverage scan + tri-mode mistral3 dispatch. 17
                // canonical VLM families, lfm2_vl/lfm2-vl dual
                // spelling (HF inconsistency), llava_qwen2 →
                // FastVLM alias, qwen3_5_moe via FormatSniff.isMXTQ
                // → Qwen35MoEJANGTQ, doc cites 4-site lock-step
                // with LLM factory, mistral3 tri-mode (mistral4 →
                // Mistral4VLM, ministral3 JANGTQ → Mistral3VLMJANGTQ
                // bits/seed defaults 2/42, ministral3 dense → throw
                // clear `use Python panel` error, else → Mistral3VLM),
                // nemotron_h_omni full multimodal wrapper, doc 2-way
                // omni registration intent, _creators nonisolated
                // (unsafe) Swift 6 strict, supportedModelTypes =
                // Set(creators.keys), shared singleton,
                // VLMProcessorTypeRegistry separate enum.
                // 15/15 green after iter 143 Nemotron-H Omni alias +
                // processor fallback fixes.
                "VLMTypeRegistryCoverageTests.swift",
                // 2026-05-02 audit-loop iter 85 — LLMTypeRegistry
                // coverage scan (every model_type → Model class
                // dispatch). 14 dense families + 12 MoE + 4 hybrid
                // SSM + 25 long-context/recent. JANGTQ routing via
                // FormatSniff.isMXTQ (≥5 call sites — handles VLM
                // wrapper nested config + uppercase). qwen3_5_moe /
                // minimax_m2 / glm4_moe → JANGTQ types. deepseek_v4
                // resolveQuantOverrides §389 nested fold. kimi_k25
                // shared dispatch. nemotron_h JANGTQ context model.
                // glm_moe_dsa rope_parameters patch. voxtral / dots_ocr explicit-
                // throw closures. mistral→Llama alias, acereason→
                // Qwen2 alias, ernie4_5_moe→dense alias (audit
                // 2026-04-16), exaone_moe→Exaone4 alias,
                // gemma3_text/gemma4_text VLM-nested aliases.
                // 16/16 green.
                "LLMTypeRegistryCoverageTests.swift",
                // 2026-05-04 Codex production audit — Nemotron-H
                // JANGTQ context + per-expert tq_packed/tq_norms
                // stacking for Omni/Nano hybrid SSM bundles.
                "NemotronHLatentMoETests.swift",
                // 2026-05-02 audit-loop iter 84 — DeepseekV4MaskHelpers
                // §407 (extends existing behavioral basics with
                // deeper formula pins). Doc cites §407 + PR #1195
                // head SHA + line refs + C5 bug rationale +
                // visibility composition formula. Namespace enum,
                // window mask `(offset+S) - windowLen + cacheK`
                // raw-pos formula + left .<= right .> sliding-window
                // edges + logicalAnd combine, compressed visibility
                // `(k+1)*ratio <= q_pos+1` block-causal staircase,
                // indexerSelected eq + .any(axis: -2) + 4D-broadcast
                // kRange, all 3 helpers insert head axis at 1
                // (broadcasts onto SDPA), qPos broadcast to (B, S),
                // qPos = Int32(offset) + range(0..<S), 3 helpers
                // public static. 13/13 green.
                "DSV4MaskHelpersDeepContractTests.swift",
                // 2026-05-02 audit-loop iter 83 — MCPClipboardImport
                // §340 (Claude Desktop / Cursor / Windsurf / Zed
                // paste-import). Doc cites §340 + 4 client targets,
                // Result struct (servers + skipped + 2 conveniences),
                // 3 ImportError cases, acceptance order mcpServers
                // > servers > inner-block heuristic, inner-block
                // requires ALL values are dicts (random JSON guard),
                // empty block throws, per-entry skip+continue
                // (partial imports preserved), entries sorted for
                // deterministic UI, transport auto-detect priority,
                // timeout accepts BOTH `timeout` AND `timeout_seconds`,
                // skipSec accepts BOTH snake AND camel, enabled
                // defaults true, invalidJSON throws (NOT empty),
                // parse signature. 14/14 green.
                "MCPClipboardImportContractTests.swift",
                // 2026-05-02 audit-loop iter 82 — MCPTypes (config +
                // transport + tool surface). MCPTransport 2 cases
                // (stdio/sse — http deferred), MCPServerState 4
                // cases (disconnected/connecting/connected/error)
                // Equatable for state-change diffs, MCPServerConfig
                // Sendable+Codable+Equatable, defaults (.stdio/
                // enabled/30s/skipSec=false), skip_security_validation
                // snake_case CodingKey (Python parity), validateShape
                // per-transport (stdio→cmd / sse→url), MCPConfig
                // accepts BOTH `servers` AND Claude `mcpServers`,
                // collision: servers WINS + warning recorded,
                // decodeWarnings nonisolated(unsafe) + NSLock,
                // drainDecodeWarnings clears (drain semantics),
                // defaults maxToolCalls=10/timeout=30, RawServerEntry
                // transport auto-detect (explicit > url-present sse
                // > stdio), defaults on convert, MCPTool.fullName
                // `server__tool` DOUBLE underscore Python parity,
                // inputSchemaJSON raw Data (round-trip preservation).
                // 15/15 green.
                "MCPTypesContractTests.swift",
                // 2026-05-02 audit-loop iter 81 — MCPConfigLoader
                // search path + load + save (Claude/Cursor MCP
                // config import). VMLX_MCP_CONFIG env var, 3-element
                // search path canonical order (cwd → ~/.config/vmlx
                // → ~/.config/vmlx-engine legacy), find priority
                // explicit > env > defaults, env path tilde-expand,
                // load returns empty MCPConfig (NOT throw) on missing,
                // JSON decode error wraps with filename in
                // configInvalid, 2-pass collect-all-errors validation
                // (NOT first-fail), shape BEFORE security per server,
                // skipSecurityValidation honored, errors join `; `,
                // save @discardableResult URL + atomic write +
                // pretty+sorted JSON + default path defaultSearchPaths
                // [1] + createDirectory intermediates. 14/14 green.
                "MCPConfigLoaderContractTests.swift",
                // 2026-05-02 audit-loop iter 80 — MCPSecurity validator
                // (defensive checks before MCP subprocess launch).
                // Namespace enum, validate dispatches by transport,
                // stdio non-empty command + shell-metachar reject in
                // command name, canonical 11-char danger blacklist
                // (; | & ` $ ( ) < > \\n \\r), args explicitly allow
                // metachar (paths) but reject NUL, env key reject =
                // AND NUL, env value reject NUL, SSE non-empty URL
                // + URL-parse-then-scheme + http/https allowlist
                // (NOT file:// / javascript:), pure (no FS/process/
                // log), 4 future-validation surfaces documented,
                // returns Optional<String> (nil=OK, NOT throws).
                // 14/14 green.
                "MCPSecurityValidatorTests.swift",
                // 2026-05-02 audit-loop iter 79 — Engine + LoadOptions
                // surface (per-load configuration). Engine is public
                // actor, EngineKind 2 cases (simple/batched) +
                // §163 NO-OP doc, LoadOptions Sendable, kind=
                // .batched default, maxNumSeqs=5 (Mac single-user),
                // prefillStepSize=1024, maxCacheBlocks=500,
                // enableTurboQuant=true (iter-64 vMLX v2 native +
                // MLA cacheTypeIsMLA + hybrid KVCacheSimple-only
                // auto-skip guards), 4 cache-stack flags default
                // true (prefix/SSM/memory/disk), enableBlockDisk
                // Cache=false (gated), enableSSMReDerive=true
                // (Python #103/#109 parity), idleSoftSec=300/
                // idleDeepSec=900/idleEnabled=true, KV defaults
                // (none/64/4), §403 SW (auto/16384), §444 JangPress
                // off + 0 + true defaults, defaultEnableThinking
                // Bool?=nil tri-state, enableJANG=true, diskCacheMax
                // GB=10.0 matching CacheCoordinatorConfig. 19/19
                // green.
                "EngineLoadOptionsContractTests.swift",
                // 2026-05-02 audit-loop iter 78 — LogprobsCollector
                // + TokenLogprob OpenAI-spec contract. TokenLogprob
                // 4 fields (token/logprob/bytes/topLogprobs) +
                // §163.B1 OpenAI bytes-array doc, TopTokenLogprob
                // 3 fields (NO recursion), both Sendable;
                // LogprobsCollector LogitProcessor conform,
                // collectedLogprobs private(set), prompt resets,
                // process noop (capture separate), batch=1 §163
                // precondition, bf16→fp32 upcast before logSoftmax,
                // sampledLogprob index formula, bytes populated at
                // capture, §163.B2 N-slot insertion (NOT argSort
                // O(VlogV)), gate on v > worst, min(topLogprobs,
                // vocab) clamp, init topLogprobs=0 default, doc
                // post-penalty timing. 17/17 green.
                "LogprobsCollectorContractTests.swift",
                // 2026-05-02 audit-loop iter 77 — Penalty processors
                // (Repetition / Presence / Frequency) + composing
                // PenaltyProcessor (Python mlx-lm parity). All 3
                // conform to LogitProcessor, RepetitionContext
                // multiplicative penalty (negative * pen, positive
                // / pen — Python parity), all 3 guard validTokens
                // (NaN-on-empty guard), Presence scatter-write,
                // Frequency GPU histogram via .at[indices].add(ones)
                // (no CPU sync), Frequency subtracts scaled
                // histogram, all 3 didSample append + prompt
                // loadPrompt; PenaltyProcessor 3 Optionals + chain
                // order rep→presence→frequency + prompt forwards
                // all 3, shared TokenRing storage, capacity init
                // (not fixed), repetition logits[0..., indices]
                // axis. 13/13 green.
                "PenaltyProcessorContractTests.swift",
                // 2026-05-02 audit-loop iter 76 — GenerateParameters
                // + TopPSampler/ArgMaxSampler contract. KV mode 3
                // cases (none/affine/turboQuant) Sendable+Equatable
                // + turboQuant 3-bit defaults, GenerateParameters
                // Sendable, canonical defaults (temp=0.6/topP=1.0/
                // topK=0/minP=0.0/groupSize=64/contextSize=20/
                // prefillStepSize=512), enableCompiledDecode default
                // false (gated activation), kvMode default .none
                // (no 2026-04-16 perf regression), samplerSeed
                // Optional nil + outside init (relink-safe), logprobs
                // false + topLogprobs 0 defaults, ArgMaxSampler
                // argMax(axis:-1), TopPSampler iter-64 caller-seed
                // honor, bf16→fp32 upcast, filter order top_p→min_p
                // →top_k Python parity, applyTopP strict `>`,
                // applyMinP log-space threshold, applyTopK short-
                // circuit + argPartition O(V), both wrap in
                // withRandomState. 17/17 green.
                "GenerateParametersSamplerContractTests.swift",
                // 2026-05-02 audit-loop iter 75 — TurboQuantSwitch
                // Linear + TurboQuantSwitchGLU MoE expert variant.
                // Inherits Module + canonical tq_packed/tq_norms
                // ParameterInfo, 5 immutable config fields, defaults
                // bits=2/seed=42, 3D packed [n_experts, out, packed]
                // (distinguishes from Dense 2D), 2D norms [n_experts,
                // out], rotate-then-gather forward, output reshape
                // appends [1, outFeatures] broadcast K dim;
                // SwitchGLU 3 @ModuleInfo (gate_proj/up_proj/down_
                // proj), down-proj inverts dim direction (hidden→
                // input), §426 swigluLimit defaults 0 + DSV4 doc
                // cite + meta[5] kernel wiring, compiled cache key
                // `bt{batch}.K{K}.b{bits}`, 4 sidecar lookups
                // (signsIn/signsDn/cbGate/cbDown), 12-arg compile
                // body order locked, shapeless: true compile, output
                // reshape appends inputDims (NOT hiddenDims), BUG3
                // doc + DSV4_DUMP_LOGITS env diag preserved (one-
                // shot _GTQ.fired latch). 19/19 green.
                "TurboQuantSwitchLinearContractTests.swift",
                // 2026-05-02 audit-loop iter 74 — ToolParameter +
                // ToolParameterType JSON-Schema contract. indirect
                // enum (nested types), 7 cases (string/bool/int/
                // double/array/object/data), schema mappings:
                // .string→string, .bool→boolean, .int→integer,
                // .double→number (JSON-Schema spec NOT Swift names),
                // .data→string+contentEncoding=base64 (image upload),
                // .array recurses items, .object recurses props +
                // required-array propagation, .schema merges
                // description + extraProperties, .required/.optional
                // factories set isRequired correctly, extraProperties
                // defaults [:], 5 fields immutable let. 14/14 green.
                "ToolParameterContractTests.swift",
                // 2026-05-02 audit-loop iter 73 — JSONValue type-
                // safe JSON enum (used for ToolCall args, MCP
                // responses, schema synthesis). Hashable+Codable+
                // Sendable, 7 canonical cases (null/bool/int/double/
                // string/array/object), decoder canonical try-order
                // (nil→bool→int→double→string→array→object — drift
                // = `1` decoded as `.double(1.0)`), throws
                // dataCorruptedError on no-match (NOT silent
                // default), .from(Any) dispatches all 7 + default
                // `String(describing:)` fallback, .anyValue NSNull
                // round-trip + nested .map/.mapValues, .asSchema
                // maps all 7 to JSON Schema type field + array
                // recurses via first element + object recurses
                // per-key, encoder dispatches all cases.
                // 12/12 green.
                "JSONValueContractTests.swift",
                // 2026-05-02 audit-loop iter 72 — ToolCallFormat +
                // ToolCallProcessor surface (streaming tool-call
                // detection across 9 model families). Format 9 cases
                // + Sendable+Codable+CaseIterable, ToolCallParser
                // Sendable + 4-member surface (startTag/endTag/parse/
                // parseEOS), default parseEOS splits on startTag +
                // filters empty + compactMaps, inline single parse,
                // createParser routes all 9 formats correctly,
                // infer detects 8 family prefixes (lfm2/glm4/gemma4/
                // gemma3-or-gemma/minimax/nemotron/qwen3_5/mistral3),
                // lowercased() before match, nil for unrecognized,
                // gemma4 BEFORE gemma3 (false-match guard);
                // ToolCallProcessor State 3 cases, default format
                // .json, processChunk dispatches inline vs tagged,
                // isInlineFormat = startTag nil, processEOS state +
                // buffer guards + parser.parseEOS + reset to normal.
                // 19/19 green.
                "ToolCallFormatProcessorTests.swift",
                // 2026-05-02 audit-loop iter 71 — Tool + ToolCall
                // surface (typed tool wrapper + model-emitted call
                // payload). ToolSpec typealias [String: any
                // Sendable], ToolProtocol Sendable, Tool generics
                // Codable Input/Output, handler @Sendable async
                // throws, name extraction from schema.function.name,
                // OpenAI-shaped schema construction, required-params
                // via isRequired predicate, schema-only init
                // overload; ToolCall Hashable+Codable+Sendable +
                // nested Function same, arguments [String:
                // JSONValue], init converts via JSONValue.from,
                // execute() guards name match + throws nameMismatch,
                // round-trips args via JSON serialize+decode,
                // ToolError LocalizedError actionable description,
                // Function.name immutable. 15/15 green.
                "ToolToolCallSurfaceTests.swift",
                // 2026-05-02 audit-loop iter 70 — Chat.Message +
                // MessageGenerator surface. Role 4 cases (user/
                // assistant/system/tool), Message init 4 params +
                // images/videos fields, 3 role factories accept full
                // VL args, tool() factory content-only (NO images/
                // videos), MessageGenerator Sendable + 3 generate
                // overloads, default generate(message:) emits role+
                // content, default generate(messages:) for-loop +
                // append (NOT map for @Sendable safety), Chat top-
                // level namespace enum. 10/10 green.
                "ChatMessageGeneratorContractTests.swift",
                // 2026-05-01 night audit-loop iter 69 — UserInput
                // multi-modal prompt + media surface. Message
                // typealias [String: any Sendable], Prompt 3 cases
                // (text/messages/chat) + description handling, Image
                // 3 cases (ciImage/url/array) + asCIImage full
                // pipeline (3-dim precondition + 0..1→0..255 norm +
                // planar→pixel transpose + RGBA pad), CIImage RGBA8
                // sRGB format, Video 3 cases (avAsset/url/frames),
                // asAVAsset @available deprecated + fatalError on
                // .frames, prompt didSet rebuilds chat media,
                // string-init wraps as .user chat msg, Processing
                // resize CGSize? Sendable, VideoFrame CIImage+
                // CMTime, tools + additionalContext Optional. 14/14
                // green.
                "UserInputContractTests.swift",
                // 2026-05-01 night audit-loop iter 68 — CacheCoordinator
                // unified-tier surface (L1 paged → L1.5 memory → L2
                // disk → SSM companion sidecar). 4 CacheDetail enum
                // cases (paged/memory/disk/miss), CacheFetchResult.
                // hit 6-param shape, Sendable, final class @unchecked
                // Sendable, 4 sub-caches public let (paged/disk/
                // memory Optional + ssm always-present), config-
                // gated init, SSM cache always initializes (NOT
                // gated — nil-deref guard), §441c SSM disk under
                // baseDir/ssm_companion subdir, hybrid lock+withLock
                // (OSAllocatedUnfairLock), shouldSkipSSMStorage
                // internal static for testability + AND formula
                // (NOT OR), PagedCacheManager full args, DiskCache
                // default dir tmp/vmlx_disk_cache, MemoryAware
                // estimate via nbytes, config immutable let.
                // 16/16 green.
                "CacheCoordinatorSurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 67 — JangConfigRepair
                // §411 (one-time on-disk JANG/JANGTQ bundle config
                // repair when declared quantization disagrees with
                // safetensors shapes). Doc cites §411 + §410 cross-
                // reference, opt-in env gate VMLX_REPAIR_BAD_JANG_
                // CONFIG=1 (default OFF — uninvited modification
                // guard), idempotent marker `.jang-config-repaired-
                // v1`, marker check before config read, backup at
                // `config.json.bak` skip-if-exists (preserve original),
                // atomic write via tmp + replaceItemAt, JSON output
                // pretty + sorted-keys (deterministic diff), patched
                // entries stamp `mode: \"affine\"`, marker body ISO
                // 8601 timestamp + count, collectMismatches nil when
                // no quantization block, .skip continues (consistent
                // with absence), per-key wins over top-level,
                // `model.`-stripped path variant tried (converter
                // convention diff), OR mismatch detection, no-
                // declaration continues. 16/16 green.
                "JangConfigRepairContractTests.swift",
                // 2026-05-01 night audit-loop iter 66 — CacheHelpers
                // extract/restore (5 public funcs handle paged
                // block + SSM companion + disk cache restore dance).
                // extractLayerData returns per-layer optional KV
                // tuples + handles all 4 KV kinds (KVCacheSimple /
                // QuantizedKVCache / TurboQuantKVCache / CacheList),
                // toUnquantized for QuantizedKVCache, nil-return for
                // SSM/Mamba/Arrays/Rotating, restoreLayerData
                // validates via CacheValidator + logs reject
                // (tier:paged), extractSSMStates returns flat array
                // + handles 3 kinds (Mamba/Arrays/CacheList), uses
                // canonical [conv_state, hidden_state] from
                // mamba.state, restoreSSMStates handles fresh
                // cache + uses [.ellipsis] snapshot, restoreFromDisk
                // Arrays @discardableResult Int + doc v1/v2 format
                // distinction, CacheList index iteration. 15/15
                // green.
                "CacheHelpersExtractRestoreTests.swift",
                // 2026-05-01 night audit-loop iter 65 — SSMReDerive
                // (#103/#105/#107/#109/#110 Python parity). Two
                // paths: maybeReDeriveSSMState (turn-end fresh
                // prefill, O(prompt_len) cost) + captureCleanSSMState
                // Inline (§440 native #109 — capture mid-prefill at
                // boundary, ZERO extra cost). Both share 4 gate
                // predicates (enabled/isHybrid/genGP>0/prompt-longer-
                // than-gp), stripped prefix formula `prefix(count -
                // genPromptLen)`, empty-stripped guard, best-effort
                // error-swallowing (catch + log + return — NEVER
                // break stream), markReDeriveFired() on success,
                // boundary stripped.count, stderr breadcrumb prefix
                // `[vmlx][cache/ssm-rederive]`, inline-capture/
                // status prefix for path distinguishability, doc
                // lists hybrid families + cites §439 fp32 cast
                // prerequisite. 14/14 green.
                "SSMReDeriveContractTests.swift",
                // 2026-05-01 night audit-loop iter 64 — SSMStateCache
                // LRU + deep-copy + L2 disk fallthrough surface
                // (Nemotron-H / Qwen3.5-A3B / Jamba / FalconH1 /
                // GraniteMoE / Gemma3n hybrid SSM models).
                // final @unchecked Sendable, OSAllocatedUnfairLock,
                // maxEntries=50 default, FetchResult isComplete
                // flag, store deep-copy via `arr * 1` (NOT view —
                // historical 2026-03-28b bug), historical-bug doc
                // citation, removeAll same-key BEFORE append (no
                // duplicates), LRU removeFirst (oldest eviction),
                // disk write-through `try?` (swallow errors,
                // generation-resilient), empty-states treated as
                // miss (osa-jang ba07392 bug fix), LRU touch
                // remove+append, fetch deep-copy `$0 * 1`, §441
                // disk fallthrough full semantics (subtract phantom
                // miss + bump hit + hydrate + LRU cap + deep-copy
                // return), clear resets all stats, reDerives `&+=`
                // overflow-trunc, P0-2 canonical hash formula
                // (modelKey-first + mediaSalt + |tokens: tag + JSON
                // ints, Python byte-parity), jsonEncodeIntList
                // no-whitespace, counters private(set), diskStore
                // nil default. 19/19 green.
                "SSMStateCacheContractTests.swift",
                // 2026-05-01 night audit-loop iter 63 — ModelAdapter +
                // ModelAdapterFactory + Registry surface (LoRA / DoRA
                // fine-tune adapter loader). 2 error cases (unsupported
                // AdapterType + incompatibleModelType), Sendable
                // protocol, 3 ops (load/fuse/unload), perform(with:)
                // defer-cleanup pattern (defer BEFORE load — leak
                // guard on throw), `fine_tune_type` JSON key,
                // `adapter_config.json` filename, download pattern
                // `*.safetensors` + `*.json` only, lora + dora both
                // registered to LoRAContainer (DoRA reuses), factory
                // final class Sendable, registry @unchecked Sendable
                // + lock.withLock get/set, createAdapter throws
                // unsupportedAdapterType on miss, empty init creates
                // empty dict. 14/14 green.
                "ModelAdapterFactoryContractTests.swift",
                // 2026-05-01 night audit-loop iter 62 — CacheCoordinator
                // Config Sendable + production-tuned defaults.
                // usePagedCache=true, enableDiskCache=true (L2 + TQ
                // 26× compression default), enableMemoryCache=false
                // opt-in (RAM cost real), memoryCachePercent=0.30
                // capped 32 GB, memoryCacheTTL=0 (no force-evict),
                // pagedBlockSize=64 (matches kvCacheGroupSize),
                // maxCacheBlocks=1000 (~64K tokens), diskCacheMaxGB=
                // 10.0, diskCacheDir nil (lazy temp), ssmMaxEntries=
                // 50 (LRU companion cap), modelKey String? optional
                // (cross-model poisoning prevention) + doc rationale.
                // 13/13 green.
                "CacheCoordinatorConfigContractTests.swift",
                // 2026-05-01 night audit-loop iter 61 — CompilableRotating
                // KVCache (compile-safe SWA drop-in for Mistral 3 /
                // Gemma 4 / DSV4 sliding layers). Inherits BaseKVCache,
                // offsetArray 1D [1] int32 (DynamicSlice compat),
                // init precondition `maxSize > keep`, init(from:)
                // defaults keep=0 + min(seqLen, maxSize) clamp +
                // seeds both offsetArray AND super.offset, update
                // tensor modulo `((prev - keep) % writeWindow) + keep`,
                // _updateInternal preserves compile object identity
                // (keys/values/offsetArray), update returns FULL
                // static buffer, makeMask `min(offsetArr, maxSize)` +
                // `rinds .< writtenLen`, broadcast on n>1, state
                // trim valid portion + full-buffer when validLen==
                // maxCacheSize, trim min(current,n) cap, isTrimmable
                // true, maxSize accessor, lazy maskRinds, copy
                // preserves state, doc calls out decode-only +
                // prefill-not-handled. 17/17 green.
                "CompilableRotatingKVCacheContractTests.swift",
                // 2026-05-03 Codex — compiled TurboQuant KV cache
                // runtime test. Ports the vmlx-swift-lm Stage 2
                // CompilableTurboQuantKVCache and verifies a tiny
                // quantized Llama's compiled TQ decode matches the
                // uncompiled TurboQuant path. Also pins Evaluate's
                // TurboQuantKVCache -> CompilableTurboQuantKVCache
                // promotion branch so Laguna/JANGTQ cannot silently
                // bail out of the compiled fast path.
                "CompilableTurboQuantKVCacheTests.swift",
                // 2026-05-01 CRITICAL — JANGTQ TQ KV cache auto-
                // activation pin (iter 60). Status-of-field audit
                // confirmed every JANGTQ bundle in the wild OMITS
                // the `turboquant` block (verified across 10+
                // bundles spanning Kimi K2.6, MiniMax M2.7, Qwen
                // 3.5/3.6, DeepSeek V4, Nemotron-Omni, Laguna,
                // Mistral 4). Auto-activation path at
                // Stream.swift:2755 is DORMANT. TQ KV stays OFF for
                // JANGTQ bundles unless user toggles
                // `enableTurboQuant` (consistent with 2026-04-16
                // perf audit ruling — default-on TQ cost 25-40 %
                // on MoE/hybrid). This test pins:
                // JangTurboQuant() default enabled=false +
                // bits=4/4, JangLoader fall-through to canonical
                // default on missing block, status-of-field audit
                // comment + verified-families list, Stream.swift
                // 5-step precedence cascade, JANG path marked
                // DORMANT, user-toggle marked LIVE OPT-IN, MLA
                // killswitch short-circuits BEFORE
                // jangTQ||enableTurboQuant, user-toggle uses global
                // turboQuantBits (NOT JANG bits), activation sets
                // kvMode .turboQuant + clears kvBits (no double-
                // quant), VMLX_DISABLE_TURBO_QUANT killswitch +
                // cacheType==\"mla\" predicate. 13/13 green.
                "JangTQAutoActivationContractTests.swift",
                // 2026-05-01 night audit-loop iter 59 — IntOrIntArray
                // + StringOrNumber + GenerationConfigFile JSON-
                // decoding union types. Storage `[Int]`, single-init
                // wraps as [v], decode tries [Int] BEFORE single Int,
                // throws typeMismatch on neither, encodes single
                // when count==1 (round-trip canonical); StringOrNumber
                // 6 cases (string/int/float/ints/floats/bool),
                // canonical decode order Int→Float→[Int]→[Float]→
                // Bool→String, asInt no String/Float coercion +
                // Bool→0/1 + ints[1] extract, asFloat no String
                // coercion + Int/Bool/[Int]/[Float] coerce, asInts
                // no coercion at all, asFloats coerce Int+Bool;
                // GenerationConfigFile `eos_token_id` key + IntOr
                // IntArray? type. 17/17 green.
                "JSONDecodingTypesContractTests.swift",
                // 2026-05-01 night audit-loop iter 58 — TQHadamard
                // randomized rotation contract (TurboQuant arXiv:
                // 2504.19874). Decompose descending-pow2 blocks
                // (96→[64,32]) via MSB extraction (bitWidth-lzc-1),
                // legacy drand48 generateRandomSigns rngLock-protected
                // (default seed=0), sign mapping `drand48()<0.5 ?
                // -1.0 : 1.0`, butterfly h=1,2,4,...,d/2 with
                // h*=2 doubling, butterfly pair `[a+b, a-b]` (NOT
                // swapped), reshape `[n, d/(2h), 2, h]`, final
                // scale `1/sqrt(d)` orthogonality, forward `H(D*x)`
                // (signs first), inverse `D*H(y)` (transform first),
                // multi-block per-block slice+sign+transform+
                // accumulate, concat axis -1, shape restored at
                // end, single-block short-circuit no-concat. 18/18
                // green.
                "TQHadamardContractTests.swift",
                // 2026-05-01 night audit-loop iter 57 — NumPyPCG64
                // canonical-magic-number source scan (extends the
                // existing NumPyPCG64Tests behavioral regression).
                // PCG64 multiplier 0x2360ED051FC65DA4/0x4385DF649FCCF645,
                // SeedSequence magic numbers (initA/multA/initB/
                // multB/mixMultL/mixMultR/xshift), 3-step PCG init
                // dance markers, inc shift `(seedInc << 1) | 1`
                // 128-bit with carry + odd-force, init calls
                // lcgStep twice, XSL-RR formula (xor + rot=>>58 +
                // (64 - rot) & 63), nextUInt64 step BEFORE xsl-rr,
                // generate splits uint64 LOW-half first, sign
                // extraction top bit `u32 >> 31`, sign mapping
                // `bit==0 ? -1 : 1`, seed=0 returns [0] (not empty),
                // pool size 4, cross-mix pool loop, generateState
                // pool stride `i % pool.count`, mix formula. 16/16
                // green.
                "NumPyPCG64SourceScanTests.swift",
                // 2026-05-01 night audit-loop iter 56 — TQCodebook
                // (Lloyd-Max optimal scalar quantizer for TurboQuant
                // arXiv:2504.19874). CodebookKey Hashable+Sendable,
                // CodebookCache final @unchecked Sendable, lock+
                // defer unlock around get/set, sharedCodebookCache
                // singleton, betaPDF uses lgamma + 1e-30 boundary
                // guard, computeCodebook iterations=200 default,
                // cache-first early-return, 10000-grid integration,
                // PDF normalized to unit integral, centroid init
                // 3/sqrt(dim) effective support (with max-1
                // guard), midpoint decision boundaries, outer
                // boundaries [-1.0, 1.0] sphere support, centroid
                // update moment/max(mass, 1e-10) NaN guard, empty-
                // bin continue, output sorted ascending, result
                // cached BEFORE return, quantizeScalar boundary-
                // crossing sum (Metal-friendly), dequantize int32
                // cast for take, trapezoid array-length guard.
                // 19/19 green.
                "TQCodebookContractTests.swift",
                // 2026-05-01 night audit-loop iter 55 — JANGTQRuntimeCache
                // shared sidecar (Hadamard signs + Lloyd-Max
                // codebooks). Singleton via .shared, private init,
                // @unchecked Sendable, sidecar load scans BOTH
                // signs. AND codebook. prefixes, signs key
                // `signs.{in}.{seed}`, codebook key `codebook.{in}.
                // {bits}`, MISS path regen via NumPyPCG64 +
                // TQCodebook (sidecar OPTIONAL for MiniMax-M2.7-
                // JANGTQ-CRACK + Qwen3.6-JANGTQ_2L), MISS path
                // caches generated, all reads lock+unlock + load
                // lock+defer unlock, storage private. 14/14 green.
                "JANGTQRuntimeCacheContractTests.swift",
                // 2026-05-01 night audit-loop iter 54 — SuScaledRoPE
                // (Phi-3.5 / Mistral 3.5 / Mistral 4 long-context)
                // + applyRotaryPosition cache-offset routing.
                // SuScaledRoPE defaults: maxPos=131072, origMax=
                // 4096, base=10000, factors=[1.0]; even-dim
                // precondition; freqs = longFactor * pow(base,
                // arange/dim); defaultScale = sqrt(1+log(factor)/
                // log(orig)); _scale = longMScale ?? (factor<1 ? 1
                // : defaultScale); both call overloads multiply
                // first-`dimensions` cols by _scale BEFORE
                // MLXFast.RoPE; both pass _freqs; traditional=false
                // hardcoded; deprecated alias preserved.
                // applyRotaryPosition: CompilableKVCache check
                // FIRST (compile path), passes offsetArray (no GPU
                // readback); BatchKVCache passes per-sequence B-
                // shaped offsetArray; default falls back
                // `cache?.offset ?? 0`; generic R: RoPELayer
                // constraint. 17/17 green.
                "SuScaledRoPEApplicationTests.swift",
                // 2026-05-01 night audit-loop iter 53 — JANGTQDenseLinear
                // (drop-in MLXNN.Linear replacement for TurboQuant
                // dense layers — Mistral 3 / 3.5 / Laguna / Mistral
                // 4 JANGTQ). Inherits from Module, @ParameterInfo
                // canonical keys (tq_packed/tq_norms/biases),
                // defaults (bits=2 production, seed=42 Python parity,
                // bias=false), valsPerU32=32/bits + packedCols
                // ceiling division, initial uint32 zeros packed +
                // float16 zeros norms + fp32 biases, forward
                // fatalErrors on missing signs/codebook sidecar
                // (NOT silent-zero noise dispatch), Hadamard rotate
                // BEFORE gather, singleton-expert reshape `[1, out,
                // packed]` + zero-index rhsIndices for n_experts=1
                // degeneracy, output dtype restored to input,
                // bias-add AFTER dtype restore + cast to out.dtype,
                // leading-shape preservation via dropLast/reshape,
                // immutable `let` for inFeatures/outFeatures/bits/
                // seed/hasBias. 16/16 green.
                "JANGTQDenseLinearContractTests.swift",
                // Real-bundle numeric parity for dense JANGTQ: loads
                // Mistral-Medium-3.5 layer-0 q_proj and compares Swift
                // JANGTQDenseLinear against Python jang_tools reference
                // L2 + first-token values. Skips when the local bundle
                // is absent.
                "JANGTQDenseLinearNumericParityTests.swift",
                // 2026-05-01 night audit-loop iter 52 — FlashMoEConfig
                // Python-parity contract pin. Prefetch 2 cases
                // (none/temporal), Sendable+Equatable struct,
                // defaults: enabled=false (opt-in), slotBankSize=64
                // (Python parity), prefetch=.none, cacheIOSplit=4
                // (production-tuned), expertIndexPath=nil (lazy);
                // init preconditions slot+io >= 1, from(dictionary:)
                // accepts BOTH `slot_bank_size` AND legacy `slot_bank`
                // alias, clamps slot+io to max(1, …), unknown
                // prefetch raw falls back .none, enabled defaults
                // false on missing key; toDictionary() emits
                // canonical slot_bank_size (not legacy), prefetch.
                // rawValue (not enum description), omits nil
                // expert_index_path, 4 core fields always present.
                // 17/17 green.
                "FlashMoEConfigContractTests.swift",
                // 2026-05-01 night audit-loop iter 51 — Compilable
                // KVCache + DynamicSlice contract pin. dynamicSlice
                // → mlx_slice_dynamic 8-arg + dynamicSliceUpdate →
                // mlx_slice_update_dynamic 7-arg, both fatalError
                // on rc != 0. CompilableKVCache OVERFLOW BIN
                // pattern: offsetArray 1D [1] int32 (NOT scalar
                // for DynamicSlice compat), maxLength=4096 + step=
                // 256 defaults, _updateInternal preserves object
                // identity (compile stateInputs match), update()
                // returns FULL buffer not slice, innerState 3-array
                // [keys, values, offsetArray], makeMask n=1 direct
                // reshape (decode hot path), n>1 range+offset
                // (prefill), causal `linds >= rinds`, sliding window
                // `windowSize - 1` offset, state trim to valid
                // portion (no padded disk store), trim min(current,n)
                // cap, isTrimmable=true, lazy maskRinds (one-shot
                // alloc). 16/16 green.
                "CompilableKVCacheDynSliceTests.swift",
                // 2026-05-01 night audit-loop iter 50 — MediaSalt
                // VLM cache fingerprint + Tokenizer protocol
                // surface. computeMediaSalt returns nil for text-
                // only, image:/video: tags BEFORE pixel bytes,
                // image branch BEFORE video for determinism, hash
                // order dimCount→dims→dtype→bytes (NaN-on-rank
                // collision guard), .noCopyIfContiguous (no per-
                // turn allocation), lowercase hex digest format,
                // SHA256 from CryptoKit, dims hashed as Int64
                // (drift to Int32 truncates >2GB tensors).
                // Tokenizer protocol Sendable, special-token
                // accessors (bos/eos/unknown + eosTokenId helper),
                // 3-overload applyChatTemplate convenience chain,
                // missingChatTemplate error, NaiveStreamingDetokenizer
                // halts on REPLACEMENT CHARACTER, newline triggers
                // startNewSegment, startNewSegment carries last
                // token (unicode-boundary mojibake guard). 14/14
                // green.
                "MediaSaltTokenizerContractTests.swift",
                // 2026-05-01 night audit-loop iter 49 — HardwareInfo +
                // WiredMemoryPolicies. isCompiledDecodeSupported
                // default true (re-enabled 2026-04-13 for SwiGLU/
                // GeGLU/softcap micro-fusions, +1.9 tok/s on
                // Qwen3.5-35B), VMLX_DISABLE_COMPILE_DECODE=1
                // operator escape hatch, machineIdentifier helper +
                // sysctl `hw.machine`, isAppleSilicon = macOS &&
                // arm64, all 4 policies Hashable+Sendable, SumPolicy
                // limit clamp(baseline+sum) + canAdmit projected==
                // clamp, clamp falls back GPU.maxRecommended when
                // no cap, recommendedWorkingSetBytes Metal-guarded,
                // MaxPolicy max(baseline, max ?? 0), FixedPolicy
                // ignores activeSizes, BudgetPolicy max(0,baseBytes)
                // floor + UUID-identity-based ==/hash + clamp
                // includes baseBytes + canAdmit includes baseBytes
                // + id defaults UUID(), both SumPolicy+BudgetPolicy
                // clamp uses `max(0, cap)` to guard negative cap.
                // 18/18 green.
                "HardwareWiredMemoryContractTests.swift",
                // 2026-05-01 night audit-loop iter 48 — LogStore +
                // ThermalMonitor surface (extends existing
                // LogStoreTests basics). LogStore actor isolation,
                // subscribe nonisolated, Level 5 cases canonical
                // order, rank canonical, capacity 5000 default,
                // ring `max(1, capacity)` floor, append drops oldest
                // at >=capacity, R4 §305 _globalMinLevel default
                // .info, append early-returns below min level,
                // replayCount 50, iter-29 tombstone fix
                // (markAndUnregister inserts BEFORE unregister,
                // register checks + finishes late, set caps at 64),
                // register replays recent matching subscriber level,
                // export NDJSON (JSON + 0x0A), ISO 8601 dates,
                // LogFilter matches 4 axes (level/cats/contains/
                // since), case-insensitive contains. ThermalMonitor
                // 4 cases, @unknown default→.nominal, canonical
                // didChange notification. 19/19 green.
                "LogStoreThermalContractTests.swift",
                // 2026-05-01 night audit-loop iter 47 — JangSpec
                // bundle format constants + loader contract.
                // bundleVersion=1, jangspec.json filename, target/
                // subdir paths (experts.jsidx + hot_core.safetensors),
                // expert blob `target/experts-%05d.bin` zero-pad,
                // 4096 page-aligned blobs, JSPE/SJIX magic numbers,
                // canonical header sizes (32/36/28/24), TensorKind
                // 3 cases (gate=0/up=1/down=2), TensorDType 3 cases
                // (qweight=0/scales=1/biases=2), Manifest
                // CodingKeys mirror Python builder, isBundle()
                // checks manifest, hot-core BEFORE fat-expert
                // check, fat path short-circuits return BEFORE
                // legacy walk (3× RAM avoidance), legacy path
                // autoreleasepool per-layer (~3× peak guard),
                // version-mismatch throws unsupportedVersion,
                // hot-core missing throws fileMissing, local
                // shard cache (no leak across loads),
                // .mappedIfSafe shard load (no 30 GB RSS spike).
                // 17/17 green.
                "JangSpecBundleFormatTests.swift",
                // 2026-05-01 night audit-loop iter 46 — RoPEUtils
                // canonical-formula pin (Llama3RoPE + YarnRoPE).
                // Llama3 init signature, fatalError without scaling,
                // factor defaults (1.0/1.0/4.0/8192), 2π wavelens,
                // smoothFactors + smoothFreqs canonical formulas,
                // YaRN even-dim precondition, correctionDim formula,
                // correctionRange clamp [0, dim-1], mscale scale<=1
                // short-circuit, mscale formula 0.1*log+1, linearRamp
                // min==max guard, _mscale = ratio, final _freqs
                // blend, freqs+mscale public accessors for DSV4
                // inverse-rope, mscale gated on `_mscale != 1.0`,
                // both YaRN+Llama3 call overloads pass precomputed
                // freqs to MLXFast.RoPE. 18/18 green.
                "RoPEUtilsContractTests.swift",
                // 2026-05-01 night audit-loop iter 45 — MetricsCollector
                // deeper behavioral pin (extends existing
                // MetricsCollectorContractTests basics). Peak GPU
                // monotone-increasing via max(), resetPeakMemory
                // resets to lastGPUActiveBytes (not 0),
                // disableGPUProbe XCTest auto-detect via
                // XCTestConfigurationFilePath, gpuMemoryBytes
                // short-circuits when disabled, zero-GPU warning
                // one-shot via warnedZeroGPU flag, warning to
                // stderr (not stdout), subscribe yields immediate
                // snapshot, last-unsubscribe cancels pollTask,
                // pollInterval default 1.0s, queueDepth +
                // activeRequests clamp >= 0, trimWindow `<` cutoff,
                // CPU ticks `&-` overflow subtract, nice in busy
                // (not idle), zero-total returns 0, recordTokenBatch
                // splits prefill vs decode buffers. 18/18 green.
                "MetricsCollectorDeepContractTests.swift",
                // 2026-05-01 night audit-loop iter 44 — Sparkline +
                // DownloadFormat helpers. Sparkline default
                // smoothingAlpha 0.3 (first-token zap dampener),
                // canonical EMA formula, alpha>=1.0 short-circuit
                // raw input, points() range clamp >= 0.0001 (NaN
                // guard), y-axis flipped for SwiftUI top-left
                // origin, <2 sample dashed mid-line empty state
                // (no UI flash to 0-height), default lineWidth 1.5,
                // round line cap + join, .easeOut(0.25) animation,
                // gradient fill top→bottom 0.35 opacity. Download
                // helpers: speed(0)→"—", eta(nil)→"—", ETA tier
                // format (Ns / Nm / %.1fh), bytes formatter
                // [useKB,MB,GB] + .file count style, Job.fraction
                // clamp [0, 1.0] guard NaN+overshoot, active-job
                // filter includes .paused. 16/16 green.
                "SparklineDownloadFormatTests.swift",
                // 2026-05-01 night audit-loop iter 43 — MarkdownView
                // fenced-block parser + ValidatedField surface.
                // Segment 2 cases (.prose/.code), parse static for
                // tests + previews, scans triple-backtick, finds
                // closing fence, appends trailing prose,
                // .inlineOnlyPreservingWhitespace markdown options,
                // try?-guarded fallback to plain Text(s) on parse
                // error, copy timer 1.5s, AppKit-guarded pasteboard,
                // horizontal ScrollView for long code lines,
                // ValidatedField clamp + validate static helpers,
                // commit clamps BEFORE invariant check, "Not a
                // number" + text reset, "out of range" message on
                // clamp, slider clears error only on drag-end. 16/16
                // green.
                "MarkdownValidatedFieldSurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 42 — RouteCatalog
                // single-source-of-truth contract. 8 family chips,
                // 6 modality cases, 3 auth tiers, 5 HTTP methods, all
                // canonical OpenAI routes (chat / completions /
                // embeddings / rerank / models / images*2 / audio*4
                // / responses + cancels), Anthropic /v1/messages +
                // count_tokens, all 13 Ollama paths, admin lifecycle
                // routes admin-gated, /v1/cache/clear admin-gated,
                // MCP routes bearer-auth, /health + /metrics +
                // /api/version auth=.none for probes, streaming flag
                // set on 5 SSE routes, curl POSIX `'\"'\"'` single-
                // quote escape, URL emitted last, all 4 placeholder
                // variants replaced (<model>/<emb>/<img>/<rr>),
                // admin-auth branch before bearer (same precedence
                // as ToolDispatcher), byFamily iterates allCases for
                // deterministic order. 17/17 green.
                "RouteCatalogContractTests.swift",
                // 2026-05-01 night audit-loop iter 41 — vMLXWhisper
                // pipeline surface. WhisperConfig 10 CodingKeys
                // (n_mels / n_audio_* / n_vocab / n_text_*),
                // isMultilingual `>= 51865` threshold, audio
                // constants paper-canonical (16k sr / 400 nFFT /
                // 160 hop / 30s chunk / 480000 nSamples / 3000
                // nFrames), §117 defer-before-write tmp cleanup,
                // 16kHz mono Float32 non-interleaved AVAudioFormat,
                // truncatedToThirtySeconds flag on result, > 30.5
                // truncation threshold (float wiggle), eot break,
                // nTextCtx-1 break, maxNewTokens=224 default,
                // 6 canonical special tokens resolved, noSpeech
                // fallback chain (`<|nospeech|>` → `<|nocaptions|>`),
                // decode() filters BOTH `< timestampBegin` AND
                // `!= eot`, multilingual lang map gated on
                // isMultilingual. 13/13 green.
                "WhisperPipelineSurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 40 — vMLXTTS
                // engine surface + WAV encoder + PlaceholderSynth.
                // 5-format enum (wav/mp3/flac/opus/pcm) + canonical
                // Content-Type mappings, speed clamp [0.25, 4.0] in
                // BOTH TTSRequest init AND PlaceholderSynth render
                // (defense in depth), missingInput on empty, unknown
                // format throws unsupportedFormat, speed accepts Int
                // OR Double, mp3/flac/opus gracefully fall back to
                // WAV bytes (NOT throw — clients still get playable
                // audio), backend label stamped on result for caller
                // observability, 24 kHz sample rate, FNV-1a stable
                // hash with canonical 1469598103934665603 seed +
                // 1099511628211 prime (NOT Swift salted hashValue),
                // tone amp 0.18 × Int16.max no clipping, WAV mono
                // PCM16 RIFF/WAVE format=1 fmt-chunk-size=16, all
                // little-endian writes via v.littleEndian, rawPCMLE
                // (low,high) byte order. 14/14 green.
                "TTSEngineSurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 39 — SettingsStore
                // 4-tier resolver + I8 §274 silent-drop fix +
                // BLOCKER #2 migration + clamp. SettingsTier 5 cases
                // (global / session / chat / request / builtin),
                // SettingsStore is `public actor`, resolver request
                // > chat > session > global cascade, enableThinking
                // + systemPrompt explicit if/else cascade (NOT ??),
                // §26 defaultMaxTokens<256 clamp + persist-fix,
                // BLOCKER #2 session migration one-shot via
                // migrateSessionIfNeeded, default kvCacheQuantization
                // = "turboquant", default enableDiskCache = true
                // + diskCacheMaxGB = 10.0, default
                // enableBlockDiskCache = false (PCM integration
                // pending), maxNumSeqs=5 Mac override,
                // prefillStepSize=2048 server-default, maxPromptTokens
                // =262_144 256K guard, kvCacheGroupSize=64 +
                // pagedCacheBlockSize=64 alignment, defaultHost
                // 127.0.0.1 + defaultPort 8000 + defaultLAN=false
                // (loopback only), gateway opt-in (enabled=false,
                // port=8080, LAN=false), allowedOrigins=["*"]
                // local-dev friendly, resolver starts from global
                // snapshot, trace fallthrough `.global` not
                // `.builtin`, debounce default 500 ms. 17/17 green.
                "SettingsResolverContractTests.swift",
                // 2026-05-01 night audit-loop iter 38 — FluxEngine
                // (image gen) + ModelRegistry surface. ModelKind 4
                // cases (imageGen / imageEdit / imageUpscale /
                // videoGen), FluxError 6 user-visible cases (unknown
                // / notLoaded / wrongModelKind / weightsNotFound /
                // notImplemented / invalidRequest), 4 capability
                // protocols (ImageGenerator / Editor / Upscaler /
                // VideoGenerator), `actor FluxEngine` isolation,
                // load throws unknownModel on bad name, load doc
                // declares NO silent HF download (DownloadManager
                // stages first), all 3 perform* helpers throw
                // notLoaded then wrongModelKind, video stub throws
                // notImplemented, loadedIsPlaceholder() helper for
                // /v1/images/generations warning header,
                // lookupFuzzy full normalization (lower → HF prefix
                // strip → -Nbit strip → `.`/`_` collapse), direct
                // hit BEFORE collapsed fallback, register lock-
                // protected, all() sorts by name (deterministic UI
                // order). 13/13 green.
                "FluxEngineRegistrySurfaceTests.swift",
                // 2026-05-01 night audit-loop iter 37 — Embedders
                // model-type registry + pooling priority. 9 model-type
                // entries (bert+roberta+xlm-roberta+distilbert all
                // share BertModel, nomic_bert with pooler=false,
                // qwen3, gemma3+gemma3_text+gemma3n share
                // EmbeddingGemma), pooling CLS>Mean>Max>Last>first
                // priority chain, ST CodingKeys, `1_Pooling/config.
                // json` path, layernorm-before-truncate-before-l2
                // ordering, eps 1e-5, last-token uses per-row mask
                // sum (NOT fixed `-1`), CLS falls back to first
                // token, 14 canonical model defaults in bootstrap,
                // bootstrap state machine idle→bootstrapping→
                // bootstrapped, qwen3 4-bit DWQ memory-safe
                // default. 13/13 green.
                "EmbedderRegistryPoolingTests.swift",
                // 2026-05-01 night audit-loop iter 36 — NemotronH-Omni
                // preprocessors. CLIP norm canonical OpenAI values,
                // NVLM dynamic-tile defaults (imageSize=512,
                // minNum=1, maxNum=12, useThumbnail=true), thumbnail
                // skipped for 1×1 grid, Slaney mel constants
                // (fSp=200/3, minLogHz=1000, logstep=log(6.4)/27.0),
                // 2/widthHz Slaney norm, mel filterbank cache keyed
                // on full param tuple, Parakeet STFT defaults
                // (sr=16000, nFFT=512, hop=160, win=400, nMels=128,
                // preemphasis=0.97), Hann periodic=false, STFT
                // center-pad with nFFT/2, log zero-guard 2^-24,
                // Bessel-corrected variance, audio load target
                // format Float32 mono non-interleaved at 16 kHz,
                // video zero-duration guard, samplesPerSecond floor,
                // video defaults (T=2 = RADIO video_embedder shape),
                // EVS pruningRate=0.7 default, EVS keeps first group,
                // EVS cosine-sim 1e-8 epsilon. 18/18 green.
                "NemotronOmniPreprocessorTests.swift",
                // §421 — shape-authoritative routed-bits override
                // (peekRoutedBitsFromSafetensors + injectRoutedBits).
                // Synthetic safetensors files exercise the pre-decode
                // path that overrides mxtq_bits when bundle metadata
                // is missing or wrong. Pure file IO + JSON, no MLX dep.
                "JangtqRoutedBitsOverrideTests.swift",
                // 2026-05-01 — JangPressMachCache (Mach
                // purgeable-memory MoE expert tier). Synthesizes
                // a few-MB tile per "expert", exercises register +
                // acquire + release + pinHot + disk-refault round-trip.
                // No real bundle / model needed — Mach + memcpy only.
                "JangPressMachCacheTests.swift",
                // 2026-05-01 — JangPressController, the failsafe
                // idle-time driver that wakes all volatile tiles
                // before each inference and compresses cold tiles
                // only when idle for `quiesceTimeoutMs` OR memory
                // pressure event arrives. State machine + frequency
                // tracking + disarm-wake all under test.
                "JangPressControllerTests.swift",
                // 2026-05-01 — JangPressPressureBench. Synthetic
                // memory-pressure stress test that registers ~80 GB
                // of synthetic expert tiles, simulates Zipfian top-6
                // of 256 routing across 43 layers, inflates a balloon
                // to force kernel pressure, then samples acquire
                // latency to verify the kernel actually compresses.
                // @Suite is `.disabled` by default — opt in with
                // `swift test --filter JangPressPressureBench`.
                "JangPressPressureBench.swift",
                // 2026-05-01 — JangPressSmokeBench. Smaller variant
                // (~2 GB) of the pressure bench that runs in normal
                // CI / dev workflow. Registers 512 tiles, drives 200
                // decode steps, allocates a same-size balloon, then
                // samples acquire latency to verify p95 < 100 ms
                // under light pressure. Enabled by default.
                "JangPressSmokeBench.swift",
                // 2026-05-01 — JangPressShard. Page-cache-backed
                // safetensors shard with header parsing, tensor
                // index, and madvise wrappers. Doesn't require a
                // real bundle — synthesizes a minimal safetensors
                // file in tmp, verifies header parse + byte-range
                // lookup + advise calls succeed.
                "JangPressShardTests.swift",
                // 2026-05-01 — JangPressMmapTier. Bundle-aware
                // wrapper that opens every safetensors shard, parses
                // the tensor index, and identifies routed-expert
                // tiles via name regex (per-expert + stacked-switch_mlp
                // layouts). acquire/release issue madvise(WILLNEED)
                // / madvise(DONTNEED) on the relevant byte ranges.
                // Synthesizes a fake bundle with two shards in tmp.
                "JangPressMmapTierTests.swift",
                // 2026-05-01 — JangPressEmbedTier (component F).
                // Page-level Zipfian compression for embed_tokens
                // + lm_head. Tier records token activity, applies
                // MADV_WILLNEED/DONTNEED per row based on frequency.
                // Synthesizes a 100-vocab × 16-hidden bundle in tmp.
                "JangPressEmbedTierTests.swift",
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
                "ServerScreenSelectionContractTests.swift",
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
                // 2026-05-03 Codex audit — storage-only block-disk cache
                // tests are SwiftPM-safe and must not remain hidden behind
                // the Metal-dependent excluded-test bucket.
                "BlockDiskCacheTests.swift",
                // 2026-05-04 Codex audit — paged-cache refcount and
                // short-tail-block behavior. Keeps sub-block prompt
                // cache reuse runnable under SwiftPM instead of hiding
                // behind Xcode-only test discovery.
                "PagedCacheBlockReleaseTests.swift",
                // 2026-05-04 Codex audit — opt-in block-level disk L2
                // integration through CacheCoordinator. Stores paged KV
                // blocks as safetensors payloads and materializes them
                // back into pinned CacheBlock instances on a fresh
                // coordinator.
                "CacheCoordinatorBlockDiskIntegrationTests.swift",
                // 2026-05-03 Codex audit — reasoning fallthrough mirror is
                // source-only and validates ChatViewModel's reasoning-off
                // reroute contract without importing vMLXApp.
                "ReasoningFallthroughContractTests.swift",
                // 2026-05-03 Codex audit — real C++ safetensors mmap
                // loader parity. Env-gated VMLX_MMAP_SAFETENSORS path
                // must produce byte-identical tensors to the default
                // reader before it can be used as the canonical
                // file-backed weight layer for JangPress.
                "MmapSafetensorsLoadTests.swift",
                // 2026-05-04 Codex audit — L2 disk cache source
                // contracts, including prompt-boundary cache snapshot
                // capture before decode mutates live cache references.
                "L2DiskHybridSSMContractTests.swift",
                // Iter 143 round 1-4 — newly added contract tests.
                // All pure-source-string-checking; no MLX/Metal init.
                "TurboQuantDefaultOnContractTests.swift",
                "JangPressMTPExclusionContractTests.swift",
                "JangPressEmbedTierWiringContractTests.swift",
                "CacheValidatorGqaContractTests.swift",
                "ContinuousBatchingContractTests.swift",
                // Metal-dependent tests excluded — `swift test` can't load
                // the default.metallib from the SwiftPM bundle path;
                // JangDFlashDrafter, JangDFlashSpecDec, VisionEmbeddingCache,
                // PagedCacheBlockRelease, FlashMoE, TQDiskSerializer,
                // CacheCoordinatorRotatingGuard, CacheBlock stay Xcode-only.
            ]
        ),
        // MARK: - Track-1 image-gen smoke tests
        // Cover FLUX.1 Schnell/Dev, FLUX.2 Klein, Qwen-Image, Z-Image,
        // FIBO, Bria. Each variant SKIPS when `VMLX_SWIFT_TEST_WEIGHTS`
        // env var is unset (CI without weights stays green). With
        // weights present: load → generate → assert RMS variance >
        // threshold (proves non-noise) and write a scratch PNG.
        .testTarget(
            name: "vMLXFluxTrack1Tests",
            dependencies: [
                "vMLXFluxKit",
                "vMLXFluxModels",
                "MLX",
            ],
            path: "Tests/vMLXFluxTests",
            exclude: [
                "Track2SmokeTests.swift",
                "Track3SmokeTests.swift",
            ],
            sources: [
                "Track1SmokeTests.swift",
                "Track1SmokeTests+ZImageReal.swift",
            ]
        ),
        // MARK: - Track-2 image-edit smoke tests
        // Pure-MLX unit tests on EditOps + skip-on-missing-env weights
        // smoke tests for FLUX.1 Kontext, FLUX.1 Fill, Qwen-Image-Edit.
        // Set VMLX_SWIFT_TEST_WEIGHTS + VMLX_SWIFT_TEST_FIXTURES to run
        // the full end-to-end smoke; otherwise the weights-dependent
        // tests skip cleanly.
        .testTarget(
            name: "vMLXFluxTrack2Tests",
            dependencies: [
                "vMLXFluxKit",
                "vMLXFluxModels",
                "MLX",
            ],
            path: "Tests/vMLXFluxTests",
            exclude: [
                "Track1SmokeTests.swift",
                "Track1SmokeTests+ZImageReal.swift",
                "Track3SmokeTests.swift",
            ],
            sources: [
                "Track2SmokeTests.swift",
            ]
        ),
        // Track 3 — Wan video smoke. Same skip-when-no-weights pattern as
        // Track 2: each variant test reads `VMLX_SWIFT_TEST_WEIGHTS` and
        // skips if unset. With weights present: generate a 16-frame MP4
        // at 320×576, assert valid container + per-channel pixel range +
        // consecutive-frame correlation > 0.5.
        .testTarget(
            name: "vMLXFluxTrack3Tests",
            dependencies: [
                "vMLXFluxKit",
                "vMLXFluxVideo",
                "MLX",
            ],
            path: "Tests/vMLXFluxTests",
            exclude: [
                "Track1SmokeTests.swift",
                "Track1SmokeTests+ZImageReal.swift",
                "Track2SmokeTests.swift",
            ],
            sources: [
                "Track3SmokeTests.swift",
            ]
        ),
    ],
    cxxLanguageStandard: .gnucxx20
)
