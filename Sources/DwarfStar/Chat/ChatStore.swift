import SwiftUI
import DS4Engine

/// A message as shown in the UI: reasoning and visible answer are kept apart so
/// the chain-of-thought can be collapsed.
struct UIMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    var reasoning: String = ""
    var text: String
}

/// A text file attached to the next message; its content is prepended to the
/// prompt so the model can read it.
struct Attachment: Identifiable {
    let id = UUID()
    let name: String
    let content: String
    let byteCount: Int
    let truncated: Bool
}

/// Main-thread view model. Owns the `InferenceService` actor and mirrors its
/// streamed output into observable UI state.
@MainActor
@Observable
final class ChatStore {
    enum Phase: Equatable {
        case needsModel
        case loading
        case ready
        case failed(String)
    }

    // Configuration (editable before loading). Defaults adapt to dev vs bundle.
    var modelPath = AppEnvironment.defaultModelPath
    var scriptDir = AppEnvironment.resourceDir   // download_model.sh / gguf
    var contextSize = 8192
    var systemPrompt = ""
    var streamingEnabled = false
    var streamingCacheSpec = ""     // e.g. "32GB"; empty = auto
    /// Tier-A per-layer streaming: tells the engine not to pin or warm the
    /// model so macOS pages out cold layers. Required on tight-RAM machines.
    var minimumRAMMode = false
    /// Tier-B per-layer streaming: drive decode one layer at a time and call
    /// MADV_DONTNEED on each finished layer's weights. Costs latency per token
    /// but caps the resident working-set to roughly one layer.
    var perLayerStreaming = false

    // Discovered GGUF files on disk.
    var discoveredModels: [DiscoveredModel] = []

    // Last applied preset explanation, shown in the load screen.
    var presetNote: String?

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
    var messages: [UIMessage] = []
    var input = ""
    var think = false
    var isGenerating = false
    var status = ""          // live prefill/decode progress (e.g. "prefill 3/11", "12 token · 1.4 tok/s")
    var attachments: [Attachment] = []   // text files to read into the next message
    var attachError: String?

    private var service: InferenceService?
    private var generation: Task<Void, Never>?

    var isReady: Bool { if case .ready = phase { return true } else { return false } }

    /// Scan the configured directories for GGUF files.
    func scanModels() {
        let gguf = (scriptDir as NSString).appendingPathComponent("gguf")
        discoveredModels = ModelCatalog.scan(directories: [scriptDir, gguf])
    }

    /// Apply the preset recommended for the detected RAM: sets streaming, cache,
    /// and context, prefers a 2-bit model if one is on disk, and notes how to get
    /// one otherwise.
    func applyRecommendedPreset() {
        scanModels()
        let preset = HardwarePresets.forRAM(MemoryInfo.physicalBytes)
        streamingEnabled = preset.streaming
        streamingCacheSpec = preset.cacheSpec
        contextSize = preset.contextSize
        minimumRAMMode = preset.minimumRAMMode
        perLayerStreaming = preset.perLayerStreaming

        var note = preset.summary
        if preset.prefersTwoBit {
            if let twoBit = discoveredModels.first(where: { HardwarePresets.isTwoBit($0.name) }) {
                modelPath = twoBit.path
                note += " Selezionato il modello 2-bit: \(twoBit.name)."
            } else {
                note += " Nessun modello 2-bit trovato: scaricalo con il pulsante “Scarica…” (target q2-imatrix) o `./download_model.sh q2-imatrix`."
            }
        }
        presetNote = note
    }

    /// Open the model off the main thread, then flip to `.ready`.
    /// The security-scoped model URL we keep open for the whole session (the GGUF is
    /// mmap'd, so access must persist). nil until the user picks a model or a saved
    /// bookmark is restored.
    private var modelAccessURL: URL?

    /// User picked a model file: grant + persist sandbox access via a security-scoped
    /// bookmark, so it reopens on the next launch without re-asking.
    func chooseModel(_ url: URL) {
        modelAccessURL?.stopAccessingSecurityScopedResource()
        guard url.startAccessingSecurityScopedResource() else {
            phase = .failed("Permesso negato per \(url.lastPathComponent)")
            return
        }
        modelAccessURL = url
        modelPath = url.path
        if let bm = try? url.bookmarkData(options: .withSecurityScope,
                                          includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(bm, forKey: "modelBookmark")
        }
    }

    /// Re-open the previously chosen model (sandbox: resolve the saved bookmark).
    func restoreModelBookmark() {
        guard modelAccessURL == nil,
              let bm = UserDefaults.standard.data(forKey: "modelBookmark") else { return }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: bm, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale),
              url.startAccessingSecurityScopedResource() else { return }
        modelAccessURL = url
        modelPath = url.path
    }

    func load() {
        guard phase != .loading else { return }
        phase = .loading
        let path = modelPath, ctx = contextSize
        let sys = systemPrompt
        let streaming = StreamingOptions(enabled: streamingEnabled,
                                         cacheSpec: streamingCacheSpec.isEmpty ? nil : streamingCacheSpec)
        let minRAM = minimumRAMMode
        let perLayer = perLayerStreaming
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: sys.isEmpty ? nil : sys,
                                               streaming: streaming,
                                               minimumRAMMode: minRAM,
                                               perLayerStreaming: perLayer)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
                }
            } catch {
                await MainActor.run { self.phase = .failed("\(error)") }
            }
        }
    }

    /// Read a (text) file and queue it as an attachment for the next message.
    func attach(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        // Don't even load files that are obviously too big into memory.
        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
           let size = attrs[.size] as? Int, size > 20 * 1024 * 1024 {
            attachError = "\(url.lastPathComponent) troppo grande (\(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)))"
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            attachError = "Impossibile leggere \(url.lastPathComponent)"; return
        }
        let maxBytes = 256 * 1024   // cap so a huge file can't blow the context window
        let truncated = data.count > maxBytes
        let slice = truncated ? data.prefix(maxBytes) : data
        let content = String(decoding: slice, as: UTF8.self)
        attachments.append(Attachment(name: url.lastPathComponent, content: content,
                                      byteCount: data.count, truncated: truncated))
        attachError = nil
    }

    func removeAttachment(_ id: UUID) { attachments.removeAll { $0.id == id } }

    /// Send the current input and stream the reply into the last message.
    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let service, (!text.isEmpty || !attachments.isEmpty), !isGenerating else { return }
        input = ""
        let atts = attachments
        attachments = []
        // The model sees the file contents prepended; the bubble shows a compact
        // 📎 list + the typed text.
        var parts: [String] = []
        for a in atts {
            let note = a.truncated ? " (troncato a 256 KB)" : ""
            parts.append("=== File: \(a.name)\(note) ===\n\(a.content)")
        }
        if !text.isEmpty { parts.append(text) }
        let fullPrompt = parts.joined(separator: "\n\n")
        var shown = text
        if !atts.isEmpty {
            let names = atts.map { "📎 \($0.name)" }.joined(separator: "  ")
            shown = text.isEmpty ? names : "\(names)\n\(text)"
        }
        messages.append(UIMessage(role: .user, text: shown))
        messages.append(UIMessage(role: .assistant, text: ""))
        let index = messages.count - 1
        isGenerating = true
        let thinkMode: DS4ThinkMode = think ? .high : .none

        generation = Task { [weak self] in
            let stream = await service.send(userText: fullPrompt,
                                            thinkMode: thinkMode,
                                            sampling: SamplingParams(),
                                            maxTokens: 4096)
            do {
                for try await event in stream {
                    guard let self, index < self.messages.count else { break }
                    switch event {
                    case .reasoning(let r): self.messages[index].reasoning += r
                    case .text(let t): self.messages[index].text += t
                    case .progress(let p): self.status = p
                    }
                }
            } catch {
                let tail = EngineLog.shared.tail()
                self?.messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty {
                    self?.messages[index].text += "\n\n--- log motore ---\n\(tail)"
                }
            }
            self?.isGenerating = false
            self?.status = ""
        }
    }

    func stop() {
        generation?.cancel()
    }

    func newChat() {
        guard let service else { return }
        let sys = systemPrompt.isEmpty ? nil : systemPrompt
        messages.removeAll()
        Task { await service.resetConversation(systemPrompt: sys) }
    }
}
