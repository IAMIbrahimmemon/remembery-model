import Foundation

/// OpenAI-compatible chat client for Muse Spark Open on Baseten (System 2 + memory reader).
public final class LLMClient: @unchecked Sendable {
    public var baseURL = URL(string: "https://inference.baseten.co/v1/chat/completions")!
    public var model = "meta-models/muse-spark-open-1026"
    let apiKey: String
    /// USD per 1M tokens (input, output), as datagen/llm_clients.py PRICES.
    public var price = (1.25, 4.25)
    private let lock = NSLock()
    private var spent = 0.0
    public var usd: Double { lock.withLock { spent } }

    public init(apiKey: String) { self.apiKey = apiKey }

    public func chatJSON(_ prompt: String, maxTokens: Int = 16000) async throws -> [String: Any] {
        var req = URLRequest(url: baseURL, timeoutInterval: 180)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["model": model, "temperature": 0.0, "max_tokens": maxTokens, "reasoning_effort": "low",
                                   "response_format": ["type": "json_object"],
                                   "messages": [["role": "user", "content": prompt]]]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw LLMError.http((resp as? HTTPURLResponse)?.statusCode ?? -1, String(decoding: data.prefix(300), as: UTF8.self))
        }
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let u = obj?["usage"] as? [String: Any] {
            let pin = (u["prompt_tokens"] as? Double) ?? 0, pout = (u["completion_tokens"] as? Double) ?? 0
            lock.withLock { spent += (pin * price.0 + pout * price.1) / 1e6 }
        }
        let content = ((obj?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String ?? ""
        return try Self.parseJSON(content)
    }

    static func parseJSON(_ text: String) throws -> [String: Any] {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let r = t.range(of: "```(?:json)?\\s*([\\s\\S]*?)```", options: .regularExpression) {
            t = String(t[r]).replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        }
        if let s = t.firstIndex(of: "{"), let e = t.lastIndex(of: "}") { t = String(t[s...e]) }
        guard let obj = try JSONSerialization.jsonObject(with: Data(t.utf8)) as? [String: Any] else { throw LLMError.parse }
        return obj
    }
}

public enum LLMError: Error, LocalizedError {
    case http(Int, String), parse
    public var errorDescription: String? {
        switch self {
        case .http(let c, let m): return "HTTP \(c): \(m)"
        case .parse: return "could not parse model JSON"
        }
    }
}

/// eval/s2_v3.py: Muse Spark Open checks the uncertain memories of one window against its transcript.
public enum System2 {
    static let prompt = """
    A small on-device model extracted possible memories from a conversation recorded by the user's AI glasses
    (the user is "ME"). It was unsure about the ones below. Check each against the transcript.

    TRANSCRIPT:
    {transcript}

    UNSURE MEMORIES (frame [trigger]: role=filler; "S2 said:" means it comes from S2's line):
    {items}

    For each: "confirm" if the transcript supports it and it is worth remembering; "fix" if it is about something real
    but a filler, person or detail is wrong (give the corrected memory as one plain sentence); "reject" if unsupported,
    wrong, or not worth remembering.

    Return JSON: {"verdicts": [{"i": int, "verdict": "confirm"|"fix"|"reject", "fixed": str|null}]}
    """

    public struct Result { public var items = 0, confirm = 0, fix = 0, reject = 0, error = 0; public var ms = 0.0 }

    public struct Verdict {
        public let id: String
        public let status: MemStatus
        public let fixed: String?
        public let note: String?
    }

    /// Ask System 2 about the pending memories of one window. Touches nothing shared; apply() writes the verdicts.
    public static func check(window w: Window, items: [Memory], client: LLMClient) async -> ([Verdict], Result) {
        var res = Result(items: items.count)
        guard !items.isEmpty else { return ([], res) }
        let t0 = Date()
        let p = prompt.replacingOccurrences(of: "{transcript}", with: Privacy.redacted(w).joined(separator: "\n"))
            .replacingOccurrences(of: "{items}", with: items.enumerated().map { "\($0.offset). \($0.element.text)" }.joined(separator: "\n"))
        let r = try? await client.chatJSON(p, maxTokens: 6000)
        res.ms = Date().timeIntervalSince(t0) * 1000
        var out: [Int: Verdict] = [:]
        for v in (r?["verdicts"] as? [[String: Any]]) ?? [] {
            guard let i = v["i"] as? Int, items.indices.contains(i), out[i] == nil, let kind = v["verdict"] as? String else { continue }
            let id = items[i].id
            switch kind {
            case "confirm":
                res.confirm += 1
                out[i] = Verdict(id: id, status: .confirmed, fixed: nil, note: nil)
            case "fix":
                if let fixed = v["fixed"] as? String, !fixed.isEmpty {
                    res.fix += 1
                    out[i] = Verdict(id: id, status: .fixed, fixed: fixed, note: nil)
                } else {
                    res.error += 1
                    out[i] = Verdict(id: id, status: .unchecked, fixed: nil, note: "fix without a corrected memory")
                }
            case "reject":
                res.reject += 1
                out[i] = Verdict(id: id, status: .rejected, fixed: nil, note: nil)
            default:
                res.error += 1
                out[i] = Verdict(id: id, status: .unchecked, fixed: nil, note: "unknown verdict")
            }
        }
        for i in items.indices where out[i] == nil {
            res.error += 1
            out[i] = Verdict(id: items[i].id, status: .unchecked, fixed: nil, note: r == nil ? "System 2 unavailable" : "no verdict")
        }
        return (items.indices.map { out[$0]! }, res)
    }

    public static func apply(_ verdicts: [Verdict], to store: MemoryStore) {
        for v in verdicts { store.setVerdict(v.id, v.status, fixed: v.fixed, note: v.note) }
    }

    /// check + apply in one call (command line).
    public static func verify(window w: Window, ids: [String], store: MemoryStore, client: LLMClient) async -> Result {
        let (vs, r) = await check(window: w, items: ids.compactMap { store.memory($0) }, client: client)
        apply(vs, to: store)
        return r
    }
}

/// Ask: BM25 over memory lines and the utterance each came from; Muse Spark Open writes the answer from them. On-device memories
/// (passwords, PINs, codes) are matched on the phone and shown to the user, never sent.
public enum Reader {
    static let prompt = """
    You are the memory of the user's AI glasses. Answer the question using ONLY the memories below and the lines they
    were extracted from. Each memory is "frame [trigger]: role=filler"; ME is the user, and "S2 said:" marks what
    someone else said. If several memories
    about the same thing conflict, the most recent one (by its date) is the current truth. If memory does not support
    an answer, say you don't have that in memory.
    Today is {now}.

    MEMORIES:
    {facts}

    QUESTION: {q}

    Return JSON: {"answer": str, "cited": [memory ids], "confidence": "high"|"medium"|"low"|"none"}
    """
    static let stop: Set<String> = Set(("the a an of to in on at for is are was were do did does what when where who which how my me i you with and or it "
                                        + "that this have has had be been will would can could should about from said").split(separator: " ").map(String.init))

    static func tok(_ s: String) -> [String] {
        let lower = s.lowercased()
        let re = try! NSRegularExpression(pattern: "[a-z0-9']+")
        return re.matches(in: lower, range: NSRange(lower.startIndex..., in: lower)).compactMap { Range($0.range, in: lower).map { String(lower[$0]) } }
            .filter { !stop.contains($0) && $0.count > 1 }
    }

    public static func retrieve(_ store: MemoryStore, _ q: String, k: Int = 40) -> [Memory] {
        let qt = tok(q)
        let rows = store.memories.filter(\.status.isMemory)
        let docs = rows.map { tok($0.text.replacingOccurrences(of: "_", with: " ") + " " + $0.evidence) }
        let N = Double(max(1, docs.count))
        let avg = docs.isEmpty ? 1 : Double(docs.map(\.count).reduce(0, +)) / N
        var df: [String: Int] = [:]
        for d in docs { for t in Set(d) { df[t, default: 0] += 1 } }
        var scored: [(Double, Memory)] = []
        for (m, d) in zip(rows, docs) {
            var tf: [String: Int] = [:]
            for t in d { tf[t, default: 0] += 1 }
            var s = 0.0
            for t in qt { if let f = tf[t] {
                let dft = Double(df[t] ?? 0), ff = Double(f)
                s += log(1 + (N - dft + 0.5) / (dft + 0.5)) * ff * 2.2 / (ff + 1.2 * (0.25 + 0.75 * Double(d.count) / avg))
            } }
            s += ["high": 0.3, "medium": 0.1][m.importance] ?? 0
            scored.append((s, m))
        }
        scored.sort { $0.0 != $1.0 ? $0.0 > $1.0 : $0.1.tObserved > $1.1.tObserved }
        let top = scored.prefix(k).filter { $0.0 > 0.3 }.map(\.1)
        store.touch(Set(top.map(\.id)))
        return top
    }

    public struct Answer {
        public let answer: String
        public let cited: [Memory]
        public let onDevice: [Memory]   // matching sensitive memories, shown on the phone only
        public let confidence: String
        public let retrieved: Int
        public let ms: Double
    }

    /// `all` = retrieve(store, q), done by the caller on the store's actor.
    public static func answer(_ all: [Memory], _ q: String, now: String, client: LLMClient) async throws -> Answer {
        let t0 = Date()
        let local = all.filter { $0.status == .onDevice }
        let items = all.filter { $0.status != .onDevice }
        let facts = items.map { "[\($0.id)] (\($0.tObserved)) \($0.text) | said: \"\($0.evidence.prefix(300))\"" }.joined(separator: "\n")
        let p = prompt.replacingOccurrences(of: "{now}", with: now).replacingOccurrences(of: "{facts}", with: facts.isEmpty ? "(none)" : facts)
            .replacingOccurrences(of: "{q}", with: q)
        let r = try await client.chatJSON(p)
        let cited = (r["cited"] as? [Any] ?? []).compactMap { c in items.first { $0.id == "\(c)" } }
        return Answer(answer: r["answer"] as? String ?? "", cited: cited, onDevice: local, confidence: r["confidence"] as? String ?? "",
                      retrieved: items.count, ms: Date().timeIntervalSince(t0) * 1000)
    }
}
