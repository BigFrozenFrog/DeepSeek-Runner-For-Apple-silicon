import Foundation
import DS4Core
import DS4Metal

// DS4Engine: the GUI's inference service, now backed by the PURE-SWIFT engine
// (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder) instead of the C ds4.
// Same API surface ChatStore already consumes (InferenceService actor, send ->
// event stream, modelInfo, resetConversation) so the SwiftUI app is unchanged.
//
// Generation uses StreamingDecoder (per-layer load/compute/evict) so the 164GB
// model fits in 16GB. NOTE: the naive streaming path reloads every layer per
// token (slow); the expert-cache fast path is a follow-up. Multi-turn context
// reuse is also a follow-up — each send renders system + current user turn.

public enum ChatRole: Sendable { case system, user, assistant }

public enum DS4ThinkMode: Sendable {
    case none, high
    var core: ThinkMode { self == .high ? .high : .none }
}

public struct SamplingParams: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var minP: Float
    public var seed: UInt64
    public init(temperature: Float = 0.6, topK: Int = 0, topP: Float = 0.95, minP: Float = 0.05, seed: UInt64 = 0xD54) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP; self.seed = seed
    }
}

public struct StreamingOptions: Sendable {
    public var enabled: Bool
    public var cacheSpec: String?
    public init(enabled: Bool = true, cacheSpec: String? = nil) { self.enabled = enabled; self.cacheSpec = cacheSpec }
}

public struct ModelInfo: Sendable {
    public let name: String
    public let layers: Int
    public let nEmbd: Int
    public let nVocab: Int
    public let contextSize: Int
    public let routedQuantBits: Int
    public let kvCacheBytes: UInt64
}

public enum GenEvent: Sendable {
    case reasoning(String)
    case text(String)
    case progress(String)   // prefill/decode status (e.g. "prefill 3/11" or "12 tok · 1.4 tok/s")
}

public actor InferenceService {
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let tok: Tokenizer
    private let decoder: StreamingDecoder
    private let dims: DSV4Dims
    private let contextSize: Int
    private let modelName: String
    private var systemPrompt: String?
    /// The full token sequence the decoder's KV cache currently corresponds to
    /// (prompt + generated, append-only). A follow-up only prefills the new tokens
    /// past this — the shared prefix is reused, not recomputed.
    private var convoTokens: [Int32] = []
    /// Cross-conversation answer cache: an identical FIRST-turn question (same model +
    /// text + system + think mode) is returned instantly from here, no generation.
    /// Survives resetConversation (shared across chats) and — persisted to JSON — app
    /// restarts. Does NOT update the KV state.
    private struct CachedAnswer: Codable { var reasoning: String; var text: String }
    private var qaCache: [String: CachedAnswer] = [:]
    /// Per-model key so different models don't collide on the same question.
    private func qaKey(system: String?, think: DS4ThinkMode, text: String) -> String {
        "\(modelName)\u{1}\(system ?? "")\u{1}\(think)\u{1}\(text)"
    }
    private let qaCacheURL: URL = {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("DwarfStar", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("qa_cache.json")
    }()
    private func saveQACache() {
        if let data = try? JSONEncoder().encode(qaCache) { try? data.write(to: qaCacheURL, options: .atomic) }
    }

    public init(modelPath: String, metalSourceDir: String? = nil, contextSize: Int,
                systemPrompt: String?, streaming: StreamingOptions,
                minimumRAMMode: Bool, perLayerStreaming: Bool) throws {
        // Kernels are embedded in the binary — no metal/ folder needed.
        self.rt = try MetalRuntime()
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        self.tok = try Tokenizer(model: model)
        // Configure the MoE/router quant scheme from the GGUF (Q4_K+Q8 vs IQ2_XXS/Q2_K+F16).
        var configuredDims = DSV4Shape.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        configuredDims.gateQuant = mq.gate; configuredDims.upQuant = mq.up
        configuredDims.downQuant = mq.down; configuredDims.routerF16 = mq.routerF16
        self.dims = configuredDims
        self.contextSize = contextSize
        self.systemPrompt = systemPrompt
        self.modelName = (modelPath as NSString).lastPathComponent
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                              attnFactor: 1, betaFast: 32, betaSlow: 1)
        // Mapped-experts streaming: experts are no-copy mmap views over the full
        // expert tensors (all 256/layer); the GPU reads only the selected rows and
        // the OS page cache caches them across tokens — no per-token re-gather.
        // Fast 16GB path (C --ssd-streaming model): non-routed weights are no-copy mmap
        // (resident via page cache, evictable), only the 6 selected experts gathered/token.
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                                       nLayers: DSV4Shape.nLayer, maxKeys: contextSize)
        // Load persisted answers from previous sessions (inline: an actor init can't
        // call isolated methods under Swift 6 concurrency, but can set stored props).
        if let data = try? Data(contentsOf: qaCacheURL),
           let decoded = try? JSONDecoder().decode([String: CachedAnswer].self, from: data) {
            qaCache = decoded
        }
    }

    public func modelInfo() -> ModelInfo {
        // Raw KV cache footprint: nLayer x contextSize x headDim x F32.
        let kv = UInt64(DSV4Shape.nLayer) * UInt64(contextSize) * UInt64(dims.headDim) * 4
        return ModelInfo(name: modelName, layers: DSV4Shape.nLayer, nEmbd: dims.nEmbd,
                         nVocab: dims.vocab, contextSize: contextSize, routedQuantBits: 4, kvCacheBytes: kv)
    }

    public func resetConversation(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
        convoTokens = []   // drop the cached KV prefix; next turn prefills from scratch
    }

    public func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGeneration(userText: userText, think: thinkMode,
                                                 sampling: sampling, maxTokens: maxTokens,
                                                 continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGeneration(userText: String, think: DS4ThinkMode, sampling: SamplingParams,
                               maxTokens: Int, continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        // Cross-conversation answer cache: an identical standalone question (fresh chat)
        // is returned instantly, no prefill/generation.
        let firstTurn = convoTokens.isEmpty
        let cacheKey = firstTurn ? qaKey(system: systemPrompt, think: think, text: userText) : nil
        if let key = cacheKey, let hit = qaCache[key] {
            continuation.yield(.progress("risposta dalla cache ⚡"))
            if !hit.reasoning.isEmpty { continuation.yield(.reasoning(hit.reasoning)) }
            continuation.yield(.text(hit.text))
            continuation.yield(.progress(""))
            return
        }

        // This turn's NEW tokens (append-only multi-turn). Turn 1 renders the full
        // chat prompt (BOS+system+user+...); a follow-up only appends user+text+assistant
        // so the shared prefix already in the KV cache (convoTokens) is reused.
        let delta: [Int32]
        if convoTokens.isEmpty {
            delta = tok.encodeChatPrompt(system: systemPrompt, prompt: userText, think: think.core)
        } else {
            var d: [Int32] = [tok.userId]
            d.append(contentsOf: tok.tokenize(userText))
            d.append(tok.assistantId)
            d.append(think.core.enabled ? tok.thinkStartId : tok.thinkEndId)
            delta = d
        }

        // Context guard. If appending overflows, drop the cached history and retry as a
        // fresh single turn (a recurrent KV cache can't drop a prefix, only reset).
        guard convoTokens.count + delta.count < contextSize else {
            if convoTokens.isEmpty {
                continuation.yield(.text("[Errore: il messaggio (\(delta.count) token, file inclusi) supera il contesto di \(contextSize). Aumenta il contesto o accorcia il file.]"))
                return
            }
            convoTokens = []
            continuation.yield(.progress("contesto pieno — riparto la conversazione"))
            return try runGeneration(userText: userText, think: think, sampling: sampling,
                                     maxTokens: maxTokens, continuation: continuation)
        }

        // Prefill ONLY the new tokens, continuing from the cached position.
        var pos = convoTokens.count
        let reused = pos
        var lastLogits: [Float] = []
        if reused > 0 {
            continuation.yield(.progress("riuso \(reused) token in cache · prefill \(delta.count) nuovi…"))
        }
        for (i, t) in delta.enumerated() {
            try Task.checkCancellation()
            lastLogits = try decoder.forward(token: Int(t), pos: pos, nKeys: pos + 1)
            pos += 1
            convoTokens.append(t)
            continuation.yield(.progress("prefill \(i + 1)/\(delta.count)" + (reused > 0 ? "  (+\(reused) in cache)" : "")))
        }

        var rng = sampling.seed
        var inReasoning = think == .high       // prompt ends with <think> when enabled
        var pending: [UInt8] = []
        var outText = "", outReasoning = ""    // accumulate the full answer for the cache

        func flush(_ asReasoning: Bool) {
            guard !pending.isEmpty, let s = String(bytes: pending, encoding: .utf8) else { return }
            pending.removeAll(keepingCapacity: true)
            if asReasoning { outReasoning += s } else { outText += s }
            continuation.yield(asReasoning ? .reasoning(s) : .text(s))
        }

        var produced = 0
        let genStart = Date()
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if Int32(next) == tok.eosId { break }
            if inReasoning && Int32(next) == tok.thinkEndId {
                flush(true)
                inReasoning = false
            } else {
                pending.append(contentsOf: tok.tokenText(Int32(next)))
                flush(inReasoning)
            }
            produced += 1
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
            convoTokens.append(Int32(next))   // generated tokens stay in the cached KV prefix
            let elapsed = Date().timeIntervalSince(genStart)
            continuation.yield(.progress(String(format: "%d token · %.2f tok/s", produced,
                                                 elapsed > 0 ? Double(produced) / elapsed : 0)))
        }
        flush(inReasoning)
        // Cache the completed answer for an identical future first-turn question, and
        // persist it so it survives app restarts.
        if let key = cacheKey, !outText.isEmpty {
            qaCache[key] = CachedAnswer(reasoning: outReasoning, text: outText)
            saveQACache()
        }
        continuation.yield(.progress(""))   // clear status when done
    }
}
