// swift-tools-version: 5.9
import PackageDescription

// Vendored copy of huggingface/swift-transformers @ f000aa7aec0e (0.1.21).
//
// Why vendored: vMLX needs to ship without runtime upstream drift. The chat-
// template path (Tokenizers → Jinja) is on the hot path of every chat
// request, and we need to be able to land model-family-specific fixes
// (Mistral 4 reasoning_effort, GLM-4.7 think_in_template, DSV4 generation
// prompt nuances) without coordinating with huggingface. Patches land
// directly in this tree.
//
// The dependency on Jinja is rewritten from the upstream
// `https://github.com/johnmai-dev/Jinja` URL to the local `Vendor/Jinja`
// path so we run a single Jinja implementation across the whole tree
// (engine + tokenizer + chat templates).

let swiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency"),
]

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
    ],
    dependencies: [
        .package(path: "../Jinja"),
    ],
    targets: [
        .target(name: "Hub", resources: [.process("FallbackConfigs")], swiftSettings: swiftSettings),
        .target(name: "Tokenizers", dependencies: ["Hub", .product(name: "Jinja", package: "Jinja")]),
        .target(name: "TensorUtils"),
        .target(name: "Generation", dependencies: ["Tokenizers", "TensorUtils"]),
        .target(name: "Models", dependencies: ["Tokenizers", "Generation", "TensorUtils"]),
    ]
)
