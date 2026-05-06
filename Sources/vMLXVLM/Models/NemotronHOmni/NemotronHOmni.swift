// NemotronHOmni.swift
// Native Swift multimodal wrapper for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.
//
// Combines:
//   • LLM (NemotronHModel from MLXLLM)
//   • RADIO ViT vision tower
//   • Parakeet Conformer audio encoder
//   • mlp1 vision projector + sound_projection audio projector
//
// Mirrors jang_tools/nemotron_omni/model.py NemotronHOmni.
//
// §435 — ported from `vmlx-swift-lm` 2026-04-29.
// §436 — multimodal forward path UNGATED 2026-04-29 after adding
// `embedTokens(_:)` and `callAsFunction(inputsEmbeds:cache:)` to
// NemotronHModel + `forwardFromEmbeddings(_:cache:)` to the backbone.

import Foundation
import AVFoundation
import MLX
import vMLXLLM
import vMLXLMCommon
import MLXNN
import CoreImage

// MARK: - Configuration

/// Top-level config for NemotronHOmni.
/// Decoded from the omni bundle's `config.json` (which is the LLM config —
/// the wrapper hardcodes the multimodal dims since they are fixed in V3).
public struct NemotronHOmniConfiguration: Codable, Sendable {
    public let llmConfig: NemotronHConfiguration

    // Multimodal dims — fixed for Nemotron-3-Nano-Omni V3 (matches config_omni.json).
    public let imageSize: Int
    public let downsampleRatio: Float
    public let vitHiddenSize: Int
    public let visionPatchSize: Int
    public let visionNumBlocks: Int
    public let visionNumHeads: Int
    public let visionNumClsTokens: Int
    public let visionMaxGrid: Int
    public let projectorHiddenSize: Int

    public let soundHiddenSize: Int
    public let soundNumLayers: Int
    public let soundNumHeads: Int
    public let soundFFHidden: Int
    public let soundConvKernel: Int
    public let soundProjectionHidden: Int
    public let soundNumMelBins: Int
    public let soundSampleRate: Int

    public let imageContextTokenId: Int
    public let videoContextTokenId: Int
    public let soundContextTokenId: Int
    public let jangtqContext: NemotronHJANGTQContext?

    enum JANGTQCodingKeys: String, CodingKey {
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqSeed = "mxtq_seed"
    }

    public init(from decoder: Decoder) throws {
        // The bundle's config.json is the LLM config directly. Decode it as
        // NemotronHConfiguration; multimodal dims are fixed defaults.
        self.llmConfig = try NemotronHConfiguration(from: decoder)

        // Hardcoded V3 multimodal dims (match config_omni.json from
        // OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-{MXFP4,JANGTQ4,JANGTQ2}).
        self.imageSize = 512
        self.downsampleRatio = 0.5
        self.vitHiddenSize = 1280
        self.visionPatchSize = 16
        self.visionNumBlocks = 32
        self.visionNumHeads = 16
        self.visionNumClsTokens = 10
        self.visionMaxGrid = 128
        self.projectorHiddenSize = 20480

        self.soundHiddenSize = 1024
        self.soundNumLayers = 24
        self.soundNumHeads = 8
        self.soundFFHidden = 4096
        self.soundConvKernel = 9
        self.soundProjectionHidden = 4096
        self.soundNumMelBins = 128
        self.soundSampleRate = 16000

        self.imageContextTokenId = 18
        self.videoContextTokenId = 131_081
        self.soundContextTokenId = 27

        let c = try? decoder.container(keyedBy: JANGTQCodingKeys.self)
        let weightFormat = (try? c?.decodeIfPresent(String.self, forKey: .weightFormat)) ?? nil
        if weightFormat?.lowercased() == "mxtq" {
            let bits = (try? c?.decodeIfPresent(Int.self, forKey: .mxtqBits)) ?? nil
            let seed = (try? c?.decodeIfPresent(Int.self, forKey: .mxtqSeed)) ?? nil
            self.jangtqContext = NemotronHJANGTQContext(
                bits: bits ?? 2,
                mxtqSeed: seed ?? 42)
        } else {
            self.jangtqContext = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        try llmConfig.encode(to: encoder)
    }
}

// MARK: - Multimodal model

public class NemotronHOmni: Module, VLMModel, KVCacheDimensionProvider, LoRAModel {

    @ModuleInfo(key: "language_model") private var languageModel: NemotronHModel

    // Tower modules. The on-disk weights for these are fp16/bf16 (NOT
    // quantized); sanitize() routes them through the remap helpers.
    //
    // NOTE: @ModuleInfo keys must be single-segment (no dots). Multi-level
    // namespaces from the bundle's safetensors keys are flattened by
    // sanitize() into one-segment paths that match these keys directly.
    @ModuleInfo(key: "vision_model") private var radioModel: NemotronHRADIOVisionModel
    @ModuleInfo(key: "mlp1") private var visionMLP: NemotronHVisionMLPProjector
    @ModuleInfo(key: "sound_encoder") private var soundEncoder: NemotronHParakeetEncoder
    @ModuleInfo(key: "sound_projection") private var soundProjection: NemotronHSoundProjector

    public let config: NemotronHOmniConfiguration

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: NemotronHOmniConfiguration) {
        self.config = config

        if let jangtq = config.jangtqContext {
            self._languageModel.wrappedValue = NemotronHModel(
                jangtqContext: jangtq, configuration: config.llmConfig)
        } else {
            self._languageModel.wrappedValue = NemotronHModel(config.llmConfig)
        }
        self._radioModel.wrappedValue = NemotronHRADIOVisionModel(
            embedDim: config.vitHiddenSize,
            numBlocks: config.visionNumBlocks,
            numHeads: config.visionNumHeads,
            patchSize: config.visionPatchSize,
            numClsTokens: config.visionNumClsTokens,
            maxGrid: config.visionMaxGrid)
        // Post-pixel-shuffle dim = vit_hidden * (1/downsample_ratio)^2 = 1280 * 4 = 5120
        let postShuffleDim = config.vitHiddenSize
            * Int(round(1.0 / config.downsampleRatio))
            * Int(round(1.0 / config.downsampleRatio))
        self._visionMLP.wrappedValue = NemotronHVisionMLPProjector(
            inDim: postShuffleDim,
            projectorDim: config.projectorHiddenSize,
            llmDim: config.llmConfig.hiddenSize)

        self._soundEncoder.wrappedValue = NemotronHParakeetEncoder(
            hiddenSize: config.soundHiddenSize,
            numLayers: config.soundNumLayers,
            numHeads: config.soundNumHeads,
            ffHidden: config.soundFFHidden,
            convKernel: config.soundConvKernel)
        self._soundProjection.wrappedValue = NemotronHSoundProjector(
            soundHidden: config.soundHiddenSize,
            projectionHidden: config.soundProjectionHidden,
            llmHidden: config.llmConfig.hiddenSize)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// LM hot path — takes raw token IDs and produces logits (text-only).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.callAsFunction(inputs, cache: cache)
    }

    /// VLM prepare — accepts LMInput with text + optional image/video/audio.
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let convertedCache = cache.compactMap { $0 as KVCache }

        if input.image == nil && input.video == nil && input.audio == nil {
            // Text-only fast path.
            return .tokens(input.text)
        }

        // Build embeddings for tokens + splice multimodal at placeholder tokens.
        let textEmbeds = languageModel.embedTokens(input.text.tokens)
        // Vision-only path (image OR video tiles in `input.image.pixels`).
        var spliced = textEmbeds
        if let pixelValues = input.image?.pixels {
            let imageEmbeds = extractImageEmbeds(pixelValues: pixelValues)
            spliced = spliceAtToken(
                tokens: input.text.tokens,
                inputsEmbeds: spliced,
                replacement: imageEmbeds,
                tokenId: config.imageContextTokenId)
        }
        if let videoPixels = input.video?.pixels {
            let videoEmbeds = extractImageEmbeds(pixelValues: videoPixels, video: true)
            spliced = spliceAtToken(
                tokens: input.text.tokens,
                inputsEmbeds: spliced,
                replacement: videoEmbeds,
                tokenId: config.videoContextTokenId)
        }
        if let audio = input.audio {
            let audioEmbeds = audio.waveforms.map { extractAudioEmbeds(waveform: $0) }
            if !audioEmbeds.isEmpty {
                let replacement = audioEmbeds.count == 1
                    ? audioEmbeds[0]
                    : MLX.concatenated(audioEmbeds, axis: 0)
                spliced = spliceAtToken(
                    tokens: input.text.tokens,
                    inputsEmbeds: spliced,
                    replacement: replacement,
                    tokenId: config.soundContextTokenId)
            }
        }

        let logits = languageModel.callAsFunction(
            inputsEmbeds: spliced, cache: convertedCache)
        return .logits(LMOutput(logits: logits))
    }

    // MARK: - Multimodal embedding extraction

    /// Run RADIO + mlp1 on a (B, 3, H, W) pixel tensor (already CLIP-normalized).
    /// Returns flat (totalTokens, llmHidden) embeddings in tile-row-major order.
    public func extractImageEmbeds(pixelValues: MLXArray, video: Bool = false) -> MLXArray {
        var feats = radioModel(pixelValues, video: video)
        // Strip cls/register tokens (first numClsTokens)
        feats = feats[0..., config.visionNumClsTokens..., 0...]
        // Reshape (N, P, D) → (N, h, w, D) where h=w=sqrt(P)
        let N = feats.dim(0)
        let P = feats.dim(1)
        let D = feats.dim(2)
        let side = Int(Double(P).squareRoot())
        precondition(side * side == P,
                     "RADIO patch count must be a perfect square; got P=\(P)")
        feats = feats.reshaped([N, side, side, D])
        // Pixel shuffle (scale = 0.5)
        feats = nemotronOmniPixelShuffle(feats, scaleFactor: config.downsampleRatio)
        // Flatten spatial dims → (N, tokens, post_shuffle_dim)
        let tokens = feats.dim(1) * feats.dim(2)
        let cIn = feats.dim(3)
        feats = feats.reshaped([N, tokens, cIn])
        // mlp1 projector → (N, tokens, llm_hidden)
        feats = visionMLP(feats)
        // Flatten to (N*tokens, llm_hidden)
        return feats.reshaped([N * tokens, feats.dim(-1)])
    }

    /// Run STFT + Parakeet + sound_projection on a 16 kHz mono waveform.
    /// Returns flat (frames, llmHidden) embeddings.
    public func extractAudioEmbeds(waveform: [Float]) -> MLXArray {
        let mel = nemotronOmniExtractMelFeatures(
            waveform,
            sampleRate: config.soundSampleRate,
            nMels: config.soundNumMelBins)
        var feats = soundEncoder(mel) // (1, F_sub, 1024)
        feats = soundProjection(feats) // (1, F_sub, llm_hidden)
        let f = feats.dim(1)
        let h = feats.dim(2)
        return feats.reshaped([f, h])
    }

    /// Splice `replacement` embeddings at every position where `tokens == tokenId`.
    /// Lengths must match. Returns embedding tensor of same shape as inputsEmbeds.
    private func spliceAtToken(
        tokens: MLXArray,
        inputsEmbeds: MLXArray,
        replacement: MLXArray,
        tokenId: Int
    ) -> MLXArray {
        // tokens: (B, T) or (T,); inputsEmbeds: (B, T, D); replacement: (N, D)
        let mask = MLX.equal(tokens, MLXArray(tokenId))
        // Squeeze batch dim to (T,), find positions
        let flatMask = mask.reshaped([-1])
        let positions = flatMask.asArray(Int.self)
        // Build a boolean mask broadcastable over D
        let D = inputsEmbeds.dim(-1)
        var maskExpanded = mask.expandedDimensions(axis: -1)
        maskExpanded = MLX.broadcast(maskExpanded, to: inputsEmbeds.shape)

        // Count placeholder positions; assemble a scattered tensor by iterating.
        let nReplace = positions.reduce(0, +)
        if nReplace == 0 { return inputsEmbeds }
        precondition(nReplace == replacement.dim(0),
                     "Multimodal placeholder count (\(nReplace)) does not match replacement embeds (\(replacement.dim(0)))")

        // Build replacement-broadcast tensor: same shape as inputsEmbeds with
        // replacement[i] at the i-th placeholder slot, zeros elsewhere.
        let replaceBuffer = MLXArray.zeros(inputsEmbeds.shape, dtype: inputsEmbeds.dtype)
        var replIdx = 0
        let totalSlots = positions.count
        let B = inputsEmbeds.dim(0)
        precondition(B == 1, "spliceAtToken currently supports batch=1 only")
        for slot in 0 ..< totalSlots {
            if positions[slot] != 0 {
                let row = replacement[replIdx ..< (replIdx + 1)] // (1, D)
                replaceBuffer[0, slot, 0..<D] = row.reshaped([D])
                replIdx += 1
            }
        }
        return MLX.where(maskExpanded, replaceBuffer.asType(inputsEmbeds.dtype), inputsEmbeds)
    }

    // MARK: - Sanitize

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // 1. Route all keys: LLM keys go through NemotronHModel.sanitize via
        //    "language_model." prefix; vision/audio/projector go through their
        //    own remap helpers.
        var llmKeys = [String: MLXArray]()
        var visionKeys = [String: MLXArray]()
        var soundKeys = [String: MLXArray]()
        var mlp1Keys = [String: MLXArray]()
        var soundProjKeys = [String: MLXArray]()

        for (k, v) in weights {
            if k.hasPrefix("vision_model.radio_model.") {
                visionKeys[k] = v
            } else if k.hasPrefix("sound_encoder.") {
                soundKeys[k] = v
            } else if k.hasPrefix("mlp1.") {
                mlp1Keys[k] = v
            } else if k.hasPrefix("sound_projection.") {
                soundProjKeys[k] = v
            } else if k.hasPrefix("vision_model.input_conditioner.") {
                // Skip — preprocess applies CLIP norm.
                continue
            } else {
                // Treat as LLM weight — strip any leading "language_model." (rare)
                // and forward to NemotronHModel.sanitize via a fresh dict.
                llmKeys[k] = v
            }
        }

        // LLM sanitize (handles conv1d transpose, JANG expert remap, expert stacking).
        let llmSanitized = languageModel.sanitize(weights: llmKeys)
        // Multimodal remap.
        let visionRemapped = remapRadioWeights(visionKeys)
        let soundRemapped = remapParakeetWeights(soundKeys)
        let mlp1Remapped = remapMlp1Weights(mlp1Keys)
        let soundProjRemapped = remapSoundProjectionWeights(soundProjKeys)

        // Combine under @ModuleInfo single-segment prefixes:
        //   "language_model.*"   → NemotronHModel root
        //   "vision_model.*"     → NemotronHRADIOVisionModel root (RADIO ViT body)
        //   "mlp1.*"             → NemotronHVisionMLPProjector root
        //   "sound_encoder.*"    → NemotronHParakeetEncoder root
        //   "sound_projection.*" → NemotronHSoundProjector root
        // The remap helpers return unprefixed paths; we add the single
        // top-level segment here.
        var out = [String: MLXArray]()
        for (k, v) in llmSanitized { out["language_model.\(k)"] = v }
        for (k, v) in visionRemapped { out["vision_model.\(k)"] = v }
        for (k, v) in soundRemapped { out["sound_encoder.\(k)"] = v }
        for (k, v) in mlp1Remapped { out["mlp1.\(k)"] = v }
        for (k, v) in soundProjRemapped { out["sound_projection.\(k)"] = v }

        return out
    }
}

// MARK: - User input processor (UserInputProcessor)

public struct NemotronHOmniProcessorConfiguration: Codable, Sendable {
    public let processorClass: String?
    public let imageSize: Int
    public let minNumTiles: Int
    public let maxNumTiles: Int
    public let useThumbnail: Bool

    public init() {
        self.processorClass = nil
        self.imageSize = 512
        self.minNumTiles = 1
        self.maxNumTiles = 12
        self.useThumbnail = true
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        self.imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 512
        self.minNumTiles = try c.decodeIfPresent(Int.self, forKey: .minNumTiles) ?? 1
        self.maxNumTiles = try c.decodeIfPresent(Int.self, forKey: .maxNumTiles) ?? 12
        self.useThumbnail = try c.decodeIfPresent(Bool.self, forKey: .useThumbnail) ?? true
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageSize = "image_size"
        case minNumTiles = "min_num_tiles"
        case maxNumTiles = "max_num_tiles"
        case useThumbnail = "use_thumbnail"
    }
}

public struct NemotronHOmniProcessor: UserInputProcessor {
    private let config: NemotronHOmniProcessorConfiguration
    private let tokenizer: any Tokenizer
    private static let soundSampleRate = 16000
    private static let soundNumMelBins = 128

    public init(_ config: NemotronHOmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Tile-preprocess images into (totalTiles, 3, H, W) MLX pixel values.
    public func preprocess(images: [CIImage]) throws -> (MLXArray, [Int]) {
        let (pixels, counts) = nemotronOmniPreprocessImages(
            images,
            imageSize: config.imageSize,
            minNum: config.minNumTiles,
            maxNum: config.maxNumTiles,
            useThumbnail: config.useThumbnail)
        return (pixels, counts)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Build prompt with NVLM 1-D placeholders. After tile selection we
        // know N total tiles → expand 256 image tokens per tile (post pixel
        // shuffle 32×32 → 16×16).
        var processedImage: LMInput.ProcessedImage?
        var processedVideo: LMInput.ProcessedVideo?
        var processedAudio: LMInput.ProcessedAudio?
        var totalImageTokens = 0
        var totalVideoTokens = 0
        var totalAudioTokens = 0
        let tokensPerTile = 256

        if !input.images.isEmpty {
            let ciImages = try input.images.map { try $0.asCIImage() }
            let (pixels, counts) = try preprocess(images: ciImages)
            processedImage = LMInput.ProcessedImage(
                pixels: pixels,
                frames: counts.map { THW($0, config.imageSize, config.imageSize) })
            let totalTiles = counts.reduce(0, +)
            totalImageTokens = totalTiles * tokensPerTile
        }
        if !input.videos.isEmpty {
            let videoPixels = try await preprocess(videos: input.videos)
            processedVideo = LMInput.ProcessedVideo(
                pixels: videoPixels,
                frames: [THW(videoPixels.dim(0), config.imageSize, config.imageSize)])
            totalVideoTokens = videoPixels.dim(0) * tokensPerTile
        }
        if !input.audios.isEmpty {
            var waveforms: [[Float]] = []
            for audio in input.audios {
                let waveform = try loadAudioWaveform(audio)
                guard !waveform.isEmpty else { continue }
                waveforms.append(waveform)
                let mel = nemotronOmniExtractMelFeatures(
                    waveform, sampleRate: Self.soundSampleRate,
                    nMels: Self.soundNumMelBins)
                totalAudioTokens += Self.parakeetSubsampledFrameCount(mel.dim(1))
            }
            if !waveforms.isEmpty {
                processedAudio = LMInput.ProcessedAudio(
                    waveforms: waveforms,
                    sampleRate: Self.soundSampleRate)
            }
        }

        // Insert media placeholders into the user message before tokenization.
        // Source convention (Python `model.py`): "<img>" + N×"<image>" + "</img>\n"
        var media = ""
        if totalImageTokens > 0 {
            media += "<img>"
            media += String(repeating: "<image>", count: totalImageTokens)
            media += "</img>\n"
        }
        if totalVideoTokens > 0 {
            media += String(repeating: "<video>", count: totalVideoTokens)
            media += "\n"
        }
        if totalAudioTokens > 0 {
            media += String(repeating: "<so_embedding>", count: totalAudioTokens)
            media += "\n"
        }

        // Build messages with media prepended to the LAST user message.
        var messages = Qwen2VLMessageGenerator().generate(from: input)
        if !media.isEmpty {
            for i in (0 ..< messages.count).reversed() {
                if (messages[i]["role"] as? String) == "user" {
                    if let oldContent = messages[i]["content"] as? String {
                        // Engine.buildChatMessages may have inserted one
                        // generic `<image>` / `<video>` marker per attachment.
                        // Nemotron owns exact placeholder counts (256 per
                        // visual tile/group, one per audio encoder frame), so
                        // strip generic markers before prepending the precise
                        // media block.
                        let cleaned = oldContent
                            .replacingOccurrences(of: "<image>", with: "")
                            .replacingOccurrences(of: "<video>", with: "")
                            .replacingOccurrences(of: "<so_embedding>", with: "")
                        messages[i]["content"] = media + cleaned
                    } else {
                        messages[i]["content"] = media
                    }
                    break
                }
            }
        }

        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo,
            audio: processedAudio)
    }

    private static func parakeetSubsampledFrameCount(_ frames: Int) -> Int {
        // Three stride-2 convs with kernel=3 padding=1 -> ceil(T/2)^3.
        let once = (frames + 1) / 2
        let twice = (once + 1) / 2
        return (twice + 1) / 2
    }

    private func preprocess(videos: [UserInput.Video]) async throws -> MLXArray {
        var batches: [MLXArray] = []
        for video in videos {
            let url: URL
            switch video {
            case .url(let u):
                url = u
            case .avAsset(let asset):
                guard let avURL = (asset as? AVURLAsset)?.url else {
                    throw VLMError.processing("Nemotron-H Omni requires URL-backed videos")
                }
                url = avURL
            case .frames:
                throw VLMError.processing("Nemotron-H Omni frame-backed videos are not wired yet")
            }
            batches.append(try await nemotronOmniPreprocessVideo(
                url: url,
                imageSize: config.imageSize,
                videoTemporalPatchDim: 2))
        }
        return batches.count == 1 ? batches[0] : MLX.concatenated(batches, axis: 0)
    }

    private func loadAudioWaveform(_ audio: UserInput.Audio) throws -> [Float] {
        switch audio {
        case .url(let url):
            return try nemotronOmniLoadAudioFile(
                url, targetSampleRate: Double(Self.soundSampleRate))
        case .samples(let samples, let sampleRate):
            guard sampleRate == Self.soundSampleRate else {
                throw VLMError.processing(
                    "Nemotron-H Omni raw audio samples must be \(Self.soundSampleRate) Hz; got \(sampleRate)")
            }
            return samples
        }
    }
}
