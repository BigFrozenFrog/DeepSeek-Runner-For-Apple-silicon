import SwiftUI
import DS4Engine
import UniformTypeIdentifiers
import AppKit   // NSPasteboard for 1-click copy

struct ChatView: View {
    @Bindable var store: ChatStore
    @State private var showImporter = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            composer
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(store.info?.name ?? "DeepSeek V4")
                    .font(.headline)
                if let info = store.info {
                    Text("\(info.layers) layer · \(info.routedQuantBits)-bit · ctx \(info.contextSize) · KV ~\(kvSize(info.kvCacheBytes))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Toggle("Thinking", isOn: $store.think)
                .toggleStyle(.switch)
            Button {
                store.newChat()
            } label: {
                Label("Nuova chat", systemImage: "square.and.pencil")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private func kvSize(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(store.messages) { message in
                        MessageRow(message: message)
                            .id(message.id)
                    }
                }
                .padding()
            }
            .onChange(of: store.messages.last?.text) {
                if let last = store.messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 4) {
        if store.isGenerating && !store.status.isEmpty {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(store.status)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        if let err = store.attachError {
            Text(err).font(.caption).foregroundStyle(.red)
        }
        if !store.attachments.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(store.attachments) { a in
                        HStack(spacing: 4) {
                            Image(systemName: "doc.text").font(.caption2)
                            Text(a.name).font(.caption).lineLimit(1)
                            Text(ByteCountFormatter.string(fromByteCount: Int64(a.byteCount), countStyle: .file))
                                .font(.caption2).foregroundStyle(.secondary)
                            Button { store.removeAttachment(a.id) } label: {
                                Image(systemName: "xmark.circle.fill").font(.caption2)
                            }.buttonStyle(.plain)
                        }
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                    }
                }
            }
        }
        HStack(alignment: .bottom, spacing: 8) {
            Button { showImporter = true } label: {
                Image(systemName: "paperclip")
            }
            .help("Allega un file di testo da leggere")
            .disabled(store.isGenerating)
            TextField("Scrivi un messaggio…", text: $store.input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
                .onSubmit { store.send() }
            if store.isGenerating {
                Button(role: .destructive) { store.stop() } label: {
                    Image(systemName: "stop.fill")
                }
            } else {
                Button { store.send() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                }
                .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && store.attachments.isEmpty)
            }
        }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.plainText, .text, .sourceCode, .json, .xml,
                                            .commaSeparatedText, .propertyList, .data],
                      allowsMultipleSelection: true) { result in
            switch result {
            case .success(let urls): for u in urls { store.attach(u) }
            case .failure(let e): store.attachError = e.localizedDescription
            }
        }
    }
}

struct MessageRow: View {
    let message: UIMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                if !message.reasoning.isEmpty {
                    ReasoningView(text: message.reasoning)
                }
                if !message.text.isEmpty {
                    if message.role == .assistant {
                        // Split prose from fenced code blocks; code gets a 1-click copy.
                        ForEach(Array(parseMessage(message.text).enumerated()), id: \.offset) { _, seg in
                            switch seg {
                            case .prose(let p):
                                Text(p)
                                    .textSelection(.enabled)
                                    .padding(10)
                                    .background(bubbleColor)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            case .code(let lang, let body):
                                CodeBlockView(lang: lang, code: body)
                            }
                        }
                    } else {
                        Text(message.text)
                            .textSelection(.enabled)
                            .padding(10)
                            .background(bubbleColor)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                } else if message.role == .assistant && message.reasoning.isEmpty {
                    ProgressView().controlSize(.small)
                }
            }
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }

    private var bubbleColor: Color {
        message.role == .user ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)
    }
}

/// One parsed chunk of an assistant message.
enum MessageSegment {
    case prose(String)
    case code(lang: String?, body: String)
}

/// Split a message into prose and ``` fenced code blocks. Handles streaming: an
/// unterminated code fence (still being generated) is shown as a code block.
func parseMessage(_ text: String) -> [MessageSegment] {
    var segs: [MessageSegment] = []
    var prose = "", code = "", lang: String? = nil
    var inCode = false
    func flushProse() {
        if !prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { segs.append(.prose(prose)) }
        prose = ""
    }
    for line in text.components(separatedBy: "\n") {
        if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
            if inCode {
                segs.append(.code(lang: lang, body: code)); code = ""; lang = nil; inCode = false
            } else {
                flushProse()
                let l = line.trimmingCharacters(in: .whitespaces).dropFirst(3).trimmingCharacters(in: .whitespaces)
                lang = l.isEmpty ? nil : String(l)
                inCode = true
            }
        } else if inCode {
            code += (code.isEmpty ? "" : "\n") + line
        } else {
            prose += (prose.isEmpty ? "" : "\n") + line
        }
    }
    if inCode { segs.append(.code(lang: lang, body: code)) } else { flushProse() }
    return segs
}

/// A code block rendered in monospace with a header (language + 1-click copy).
struct CodeBlockView: View {
    let lang: String?
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(lang ?? "codice")
                    .font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { copied = false }
                } label: {
                    Label(copied ? "Copiato" : "Copia", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.caption2)
                }
                .buttonStyle(.borderless)
                .help("Copia il codice")
            }
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(Color.black.opacity(0.22))

            // Show ALL lines (long ones wrap). fixedSize(vertical:) stops SwiftUI from
            // clipping the Text to one line — the bug where only the top/comments showed.
            Text(code.isEmpty ? " " : code)
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
        }
        .background(Color.black.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.secondary.opacity(0.2)))
    }
}

/// Collapsible chain-of-thought block.
struct ReasoningView: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Ragionamento", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
