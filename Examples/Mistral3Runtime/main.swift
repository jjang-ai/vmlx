// vMLX Examples — Mistral3Runtime
//
// Drives a Mistral-Medium-3.5-128B (mistral3 wrapper + ministral3 text +
// pixtral vision) JANGTQ bundle through the vMLX runtime stack.
//
// Why this example exists:
//   1. `mistral3` is a wrapper class — its outer model_type routes
//      through `dispatchMistral3LLM` / `dispatchMistral3VLM` depending
//      on whether `vision_config` is present. JANGTQ converts only
//      the text decoder; vision_tower + multi_modal_projector + lm_head
//      stay fp16 passthrough (per `modules_to_not_convert`).
//   2. Mistral 3.5 does NOT have trained `<think>` reasoning. The
//      example exercises chat + tools without reasoning.
//   3. Tool format is Mistral's instruct-style JSON
//      (`[TOOL_CALLS][{name,arguments}]`). Routed via
//      `ToolCallFormat.mistral` for `mistral3` model_type.
//   4. Vision input flows through `PixtralImageProcessor` — currently
//      gated on the vision tower fold-in landing. Until then the
//      runtime is text-only and image inputs are dropped at the
//      ingest seam (no crash).
//
// Build & run:
//   swift run --package-path /Users/eric/vmlx/swift Mistral3Runtime \
//       /Users/eric/.mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ
//
// Cache cascade specifics for Mistral 3.5:
//   • The text decoder is plain MLA + dense MLP (no MoE). KV cache
//     is straightforward — TurboQuant compresses the prefix at
//     prefill end, window stays float for decode.
//   • BlockDiskCache storage exists but is not in the live fetch/store path.
//     Prompt-level DiskCache is the persistent tier today.
//   • SSM re-derive is a no-op (no Mamba layers).
//   • FP8 source: bundle ships per-tensor `weight_scale_inv` (NOT
//     `weight_scale`); the JANGTQ converter handles both. The runtime
//     just sees post-MXTQ weights.

import Foundation
import MLX
import MLXNN
import vMLXEngine
import vMLXLMCommon
import RuntimeShared

let bundlePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : (FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ").path)

let bundle = URL(fileURLWithPath: bundlePath)
guard FileManager.default.fileExists(atPath: bundle.path) else {
    print("bundle not found: \(bundle.path)")
    exit(2)
}

let cacheDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("vmlx-examples-mistral3")
let slidingWindowMode = ProcessInfo.processInfo.environment["VMLX_MISTRAL3_SLIDING_MODE"]
    ?? "auto"

print("=== Mistral 3.5 runtime example ===")
print("bundle: \(bundle.lastPathComponent)")
print("cache:  \(cacheDir.path)")

if ProcessInfo.processInfo.environment["VMLX_TQ_DENSE_PARITY"] == "1" {
    let sidecar = bundle.appendingPathComponent("jangtq_runtime.safetensors")
    let shard = bundle.appendingPathComponent("model-00003-of-00037.safetensors")
    try JANGTQRuntimeCache.shared.loadSidecar(from: sidecar)
    let arrays = try MLX.loadArrays(url: shard)
    let specs: [(String, Int, Int, Float)] = [
        ("model.language_model.layers.0.self_attn.q_proj", 12_288, 12_288, 11.3163357),
        ("model.language_model.layers.0.mlp.gate_proj", 12_288, 28_672, 15.5277491),
        ("model.language_model.layers.0.mlp.up_proj", 12_288, 28_672, 15.4472542),
        ("model.language_model.layers.0.mlp.down_proj", 28_672, 12_288, 15.8572350),
    ]
    for (base, inFeatures, outFeatures, refL2) in specs {
        guard
            let packed = arrays["\(base).tq_packed"],
            let norms = arrays["\(base).tq_norms"]
        else {
            fatalError("\(base) TQ tensors missing from \(shard.path)")
        }
        let linear = JANGTQDenseLinear(
            inFeatures: inFeatures, outFeatures: outFeatures, bits: 2, seed: 42)
        try linear.update(parameters: ModuleParameters.unflattened([
            "tq_packed": packed,
            "tq_norms": norms,
        ]), verify: [.all])
        let x = MLXArray(0 ..< inFeatures).asType(.float32).reshaped([1, inFeatures])
            / MLXArray(Float(inFeatures))
        let y = linear(x).asType(.float32)
        MLX.eval(y)
        let l2 = sqrt((y * y).sum()).item(Float.self)
        let first16 = y[0..., 0 ..< 16].flattened().asArray(Float.self)
        print("[tq-parity] \(base) l2=\(l2) python-ref=\(refL2) delta=\(abs(l2 - refL2))")
        print("[tq-parity] \(base) first16=\(first16)")
    }
    exit(0)
}

if ProcessInfo.processInfo.environment["VMLX_TOKENIZER_PROBE"] == "1" {
    let tokenizer = try await TransformersTokenizerLoader().load(from: bundle)
    let text = "<s>[MODEL_SETTINGS]{\"reasoning_effort\": \"none\"}[/MODEL_SETTINGS][INST]Say hi in one short sentence.[/INST]"
    let ids = tokenizer.encode(text: text, addSpecialTokens: false)
    print("[tokenizer-probe] len=\(ids.count) ids=\(Array(ids.prefix(80)))")
    print("[tokenizer-probe] decoded=\(tokenizer.decode(tokenIds: Array(ids.prefix(20)), skipSpecialTokens: false))")
    exit(0)
}

if ProcessInfo.processInfo.environment["VMLX_EMBED_PARITY"] == "1" {
    let shard2 = bundle.appendingPathComponent("model-00002-of-00037.safetensors")
    let shard3 = bundle.appendingPathComponent("model-00003-of-00037.safetensors")
    var arrays = try MLX.loadArrays(url: shard2)
    for (k, v) in try MLX.loadArrays(url: shard3) {
        arrays[k] = v
    }
    guard
        let weight = arrays["model.language_model.embed_tokens.weight"],
        let scales = arrays["model.language_model.embed_tokens.scales"],
        let biases = arrays["model.language_model.embed_tokens.biases"]
    else {
        fatalError("embed_tokens affine tensors missing")
    }
    let emb = QuantizedEmbedding(
        embeddingCount: 131_072, dimensions: 12_288,
        groupSize: 64, bits: 8, mode: .affine)
    try emb.update(parameters: ModuleParameters.unflattened([
        "weight": weight,
        "scales": scales,
        "biases": biases,
    ]), verify: [.all])
    let ids = MLXArray([
        1, 36, 19227, 80277, 1302, 13196, 1609, 1449, 2811, 1429,
        17670, 46005, 37, 3, 67935, 14994, 1294, 1925, 4958, 19286, 1046, 4,
    ]).reshaped([1, 22])
    let y = emb(ids).asType(.float32)
    MLX.eval(y)
    let l2 = sqrt((y * y).sum()).item(Float.self)
    let first = y[0..., 0 ..< 1, 0 ..< 16].flattened().asArray(Float.self)
    let last = y[0..., 21 ..< 22, 0 ..< 16].flattened().asArray(Float.self)
    print("[embed-parity] l2=\(l2) python-ref=0.86298245 delta=\(abs(l2 - 0.86298245))")
    print("[embed-parity] first-token-first16=\(first)")
    print("[embed-parity] last-token-first16=\(last)")
    exit(0)
}

let opts = RuntimeShared.makeLoadOptions(
    bundle: bundle,
    cacheDir: cacheDir,
    kvCacheBits: 4,
    kvCacheGroupSize: 64,
    slidingWindowMode: slidingWindowMode
)
RuntimeShared.reportLoadOptions(opts)

let engine = Engine()
let t0 = Date()
try await RuntimeShared.awaitLoad(engine, options: opts)
print("[runtime] loaded in \(Int(Date().timeIntervalSince(t0))) s")

if ProcessInfo.processInfo.environment["VMLX_MISTRAL3_QUICK"] == "1" {
    print("\n--- quick chat ---")
    let quickTemperature = Double(
        ProcessInfo.processInfo.environment["VMLX_MISTRAL3_QUICK_TEMP"] ?? "0.0") ?? 0.0
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg("Say hi in one short sentence.")],
        maxTokens: 64,
        temperature: quickTemperature
    )
    let (_, content, _) = try await RuntimeShared.drainStream(engine, req)
    print("[quick.content] \(content)")
    RuntimeShared.assertNoLeak(content)
    await RuntimeShared.reportCacheStats(engine)
    print("\n=== quick done ===")
    exit(0)
}

// MARK: 1 — Single-turn chat
do {
    print("\n--- single-turn chat ---")
    let prompts = [
        "Briefly explain the difference between TCP and UDP.",
        "Translate 'Good morning, how are you?' to French.",
        "Write a one-paragraph summary of the Pythagorean theorem.",
    ]
    for p in prompts {
        let req = RuntimeShared.makeRequest(
            [RuntimeShared.userMsg(p)],
            maxTokens: 192, temperature: 0.0
        )
        let (_, content, _) = try await RuntimeShared.drainStream(engine, req, printContent: false)
        print("USER: \(p)")
        print("ASSISTANT: \(content.prefix(220))\n")
        RuntimeShared.assertNoLeak(content)
    }
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: 2 — Tool calling (Mistral instruct JSON)
do {
    print("\n--- tool calling ---")
    let sys = """
    You have access to: get_weather(location: string, units: string) -> string.
    Use Mistral instruct format: when calling tools, emit a [TOOL_CALLS] block
    followed by a JSON array of {name, arguments}.
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.systemMsg(sys),
         RuntimeShared.userMsg("What's the weather in Tokyo? Use celsius.")],
        maxTokens: 192, temperature: 0.0
    )
    let (_, content, tools) = try await RuntimeShared.drainStream(engine, req, printContent: false)
    print("[TOOLS PARSED] \(tools)")
    print("content:    \(content.prefix(120))")
}

// MARK: 3 — Long-context recall (exercises TurboQuant compression)
do {
    print("\n--- long context recall ---")
    let filler = String(repeating: """
    Mountains rolled across the horizon, draped in mist that thinned each morning.
    The road wound past three small villages whose names had faded from local memory.

    """, count: 60)
    let needle = "The hidden phrase is MISTRAL-CARP-991."
    let prompt = """
    Read carefully:

    \(filler)

    \(needle)

    \(filler)

    What is the hidden phrase? Answer with just the phrase itself.
    """
    let req = RuntimeShared.makeRequest(
        [RuntimeShared.userMsg(prompt)],
        maxTokens: 32, temperature: 0.0
    )
    let (_, content, _) = try await RuntimeShared.drainStream(engine, req)
    if content.contains("MISTRAL-CARP-991") {
        print("PASS — long-context recall works (TurboQuant prefix compression OK)")
    } else {
        print("FAIL — needle not retrieved. Likely TurboQuant decode bug or window too small.")
    }
    await RuntimeShared.reportCacheStats(engine)
}

// MARK: 4 — Multi-turn cache reuse
do {
    print("\n--- multi-turn cache reuse ---")
    var msgs: [ChatRequest.Message] = []
    let asks = [
        "What is the largest moon of Saturn?",
        "How does it compare in size to Earth's moon?",
        "Could it host life?",
    ]
    for ask in asks {
        msgs.append(RuntimeShared.userMsg(ask))
        let req = RuntimeShared.makeRequest(msgs, maxTokens: 192, temperature: 0.0)
        let (_, ans, _) = try await RuntimeShared.drainStream(engine, req, printContent: false)
        print(">>> \(ask)\n<<< \(ans.prefix(160))…\n")
        msgs.append(RuntimeShared.assistantMsg(ans))
    }
    await RuntimeShared.reportCacheStats(engine)
    print("[note] blockDisk.wired=false today; prompt-level DiskCache is the persistent tier.")
}

print("\n=== done ===")
