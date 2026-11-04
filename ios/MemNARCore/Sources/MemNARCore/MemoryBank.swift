import Foundation

/// One recording session: the whole transcript and the windows it was cut into.
public struct Session: Codable, Identifiable, Hashable {
    public let id: String
    public let started: Date
    public var ended: Date?
    public var title: String
    public var utterances: [Utterance]
    public var windowIds: [String]
    public var scene: String

    public init(id: String = String(UUID().uuidString.prefix(8)).lowercased(), started: Date = Date(), title: String,
                scene: String = "live conversation") {
        self.id = id; self.started = started; self.title = title; self.utterances = []; self.windowIds = []; self.scene = scene
    }
}

/// Everything the glasses remember, saved on the phone as one JSON file.
public final class MemoryBank {
    public private(set) var sessions: [Session] = []
    public private(set) var records: [MemoryRecord] = []
    public let url: URL?

    public init(url: URL?) {
        self.url = url
        guard let url, let data = try? Data(contentsOf: url) else { return }
        struct Saved: Decodable { let sessions: [Session]; let records: [MemoryRecord] }
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        if let s = try? d.decode(Saved.self, from: data) { sessions = s.sessions; records = s.records }
    }

    public func save() {
        guard let url else { return }
        struct Saved: Encodable { let sessions: [Session]; let records: [MemoryRecord] }
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        if let data = try? e.encode(Saved(sessions: sessions, records: records)) { try? data.write(to: url, options: .atomic) }
    }

    public func upsert(_ s: Session) {
        if let i = sessions.firstIndex(where: { $0.id == s.id }) { sessions[i] = s } else { sessions.append(s) }
    }

    public func add(_ rs: [MemoryRecord]) {
        let have = Set(records.map(\.id))
        records += rs.filter { !have.contains($0.id) }
    }

    public func delete(session id: String) {
        sessions.removeAll { $0.id == id }
        records.removeAll { $0.sessionId == id }
    }

    public func delete(record id: String) { records.removeAll { $0.id == id } }

    public func wipe() { sessions = []; records = [] }

    public func session(_ id: String) -> Session? { sessions.first { $0.id == id } }

    /// Known entities of recent memories, as the model's linking state for the next window (at most `limit`).
    public func recentState(limit: Int = 32) -> [StateEntity] {
        var out: [StateEntity] = [], seen = Set<String>()
        for r in records.reversed() {
            for e in r.entities where ["PERSON", "PET", "ORG", "PLACE", "VENUE", "OBJECT", "DEVICE", "VEHICLE", "EVENT", "APPOINTMENT"].contains(e.coarse) {
                let k = e.text.lowercased()
                if seen.insert(k).inserted { out.append(StateEntity(ref: "m\(out.count + 1)", type: e.type, canonical: e.text)) }
                if out.count >= limit { return out }
            }
        }
        return out
    }

    /// People and things mentioned most, for the memory browser.
    public func topEntities(_ coarse: Set<String>, limit: Int = 20) -> [(String, Int)] {
        var c: [String: Int] = [:]
        for r in records { for e in Set(r.entities.filter { coarse.contains($0.coarse) }.map { $0.text }) { c[e, default: 0] += 1 } }
        return c.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.prefix(limit).map { ($0.key, $0.value) }
    }
}

/// Ask without a network: BM25 over each memory's line, frames, roles, entities and context.
public enum LocalSearch {
    static let stop: Set<String> = Set(("the a an of to in on at for is are was were do did does what when where who which how my me i you with and or it "
                                        + "that this have has had be been will would can could should about from said tell remember").split(separator: " ").map(String.init))

    public static func tokens(_ s: String) -> [String] {
        let lower = s.lowercased()
        let re = try! NSRegularExpression(pattern: "[a-z0-9']+")
        return re.matches(in: lower, range: NSRange(lower.startIndex..., in: lower))
            .compactMap { Range($0.range, in: lower).map { String(lower[$0]) } }
            .filter { !stop.contains($0) && $0.count > 1 }
    }

    static func doc(_ r: MemoryRecord) -> [String] {
        var parts = [r.text, r.speaker]
        for f in r.frames { parts.append(f.frame.replacingOccurrences(of: "_", with: " ")); parts.append(f.trigger); parts += f.roles.flatMap(\.fillers) }
        parts += r.entities.flatMap { [$0.text, $0.type] }
        if let c = r.context { parts.append(c.text) }
        return tokens(parts.joined(separator: " "))
    }

    public struct Hit: Identifiable {
        public let record: MemoryRecord
        public let score: Double
        public let matched: Set<String>
        public var id: String { record.id }
    }

    public static func search(_ records: [MemoryRecord], _ q: String, k: Int = 12) -> [Hit] {
        let qt = Set(tokens(q))
        guard !qt.isEmpty else { return [] }
        let docs = records.map(doc)
        let n = Double(max(1, docs.count))
        let avg = docs.isEmpty ? 1 : Double(docs.map(\.count).reduce(0, +)) / n
        var df: [String: Int] = [:]
        for d in docs { for t in Set(d) { df[t, default: 0] += 1 } }
        var hits: [Hit] = []
        for (r, d) in zip(records, docs) {
            var tf: [String: Int] = [:]
            for t in d { tf[t, default: 0] += 1 }
            var s = 0.0
            var m = Set<String>()
            for t in qt { if let f = tf[t] {
                let dft = Double(df[t] ?? 0), ff = Double(f)
                s += log(1 + (n - dft + 0.5) / (dft + 0.5)) * ff * 2.2 / (ff + 1.2 * (0.25 + 0.75 * Double(d.count) / avg))
                m.insert(t)
            } }
            if s > 0 { hits.append(Hit(record: r, score: s + 0.2 * r.score, matched: m)) }
        }
        return Array(hits.sorted { $0.score != $1.score ? $0.score > $1.score : ($0.record.date, $0.record.time) > ($1.record.date, $1.record.time) }.prefix(k))
    }
}

/// Optional: Muse writes a sentence answer from the matching memories (needs a key; sensitive memories never sent).
public enum RecordReader {
    static let prompt = """
    You are the memory of the user's AI glasses. Answer the question using ONLY the memories below. Each memory is a
    line the glasses heard, "[kinds of fact] SPEAKER: "what was said"", with its date; ME is the user. If memories
    conflict, the most recent one is the current truth. If memory does not support an answer, say you don't have that
    in memory. Today is {now}.

    MEMORIES:
    {facts}

    QUESTION: {q}

    Return JSON: {"answer": str, "cited": [memory ids], "confidence": "high"|"medium"|"low"|"none"}
    """

    public static func answer(_ hits: [MemoryRecord], _ q: String, now: String, client: LLMClient) async throws -> (String, [String], String) {
        let send = hits.filter { !$0.sensitive }
        let facts = send.map { "[\($0.id)] (\($0.date) \($0.time)) \($0.line)" }.joined(separator: "\n")
        let p = prompt.replacingOccurrences(of: "{now}", with: now).replacingOccurrences(of: "{facts}", with: facts.isEmpty ? "(none)" : facts)
            .replacingOccurrences(of: "{q}", with: q)
        let r = try await client.chatJSON(p)
        return (r["answer"] as? String ?? "", (r["cited"] as? [Any] ?? []).map { "\($0)" }, r["confidence"] as? String ?? "")
    }
}
