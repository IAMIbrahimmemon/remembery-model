import Foundation

/// Decision 37 on device: every utterance System 1 finds something in becomes a candidate line, a small validator
/// keeps the ones worth remembering, and each kept line is stored as a structured record.
/// Ports eval/validator.py (line_features with feat 2, g_line, needs_context, record, score) and the speaker and
/// pronoun rules of model/formats.py _names. `memnar-cli lines` checks it against Python.

public struct ValidatorSpec: Codable {
    public var minP: Double
    public var entLines: Bool
    public var clip: Int
    public var ctx: Bool
    public var feat: Int
}

/// model/validator_g2.json: logistic regression over the line features, standardized with mu / sd.
public struct LineValidator: Decodable {
    public let w: [Double]
    public let mu: [Double]
    public let sd: [Double]
    public let threshold: Double
    public let spec: ValidatorSpec
    public let names: [String]

    public static func load(_ url: URL) throws -> LineValidator {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return try d.decode(LineValidator.self, from: Data(contentsOf: url))
    }

    public func score(_ x: [Double]) -> Double {
        var z = 0.0
        for j in 0..<min(x.count, w.count) { z += w[j] * (x[j] - mu[j]) / sd[j] }
        return 1 / (1 + exp(-max(-30, min(30, z))))
    }
}

public struct RecordRole: Codable, Hashable {
    public let role: String
    public let fillers: [String]
}

public struct RecordFrame: Codable, Hashable {
    public let frame: String
    public let trigger: String
    public var p: Double
    public let modality: String
    public let tense: String
    public let roles: [RecordRole]
}

public struct RecordEntity: Codable, Hashable {
    public let text: String
    public let coarse: String
    public let type: String
}

public struct RecordContext: Codable, Hashable {
    public let speaker: String
    public let text: String
}

/// One stored memory: the line as said, plus what System 1 understood about it.
public struct MemoryRecord: Codable, Identifiable, Hashable {
    public let id: String
    public var sessionId: String
    public let windowId: String
    public let date: String
    public let time: String
    public let utt: Int
    public let speaker: String          // "ME" or a speaker label
    public let text: String             // the utterance
    public let line: String             // format G: "[Frames] SPEAKER: "text"" (+ context)
    public let frames: [RecordFrame]
    public let entities: [RecordEntity]
    public let context: RecordContext?
    public let score: Double            // validator probability
    public var sensitive: Bool          // password / PIN / code: kept on the phone, never sent anywhere
}

enum PyText {
    /// str.split() then " ".join(...)
    static func collapse(_ s: String) -> String { s.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ") }
    static func words(_ s: String) -> Int { s.split(whereSeparator: { $0.isWhitespace }).count }
    /// model/decode3.py _clip(text, "", width): the first `width` code points, "…" if cut.
    static func clipStart(_ s: String, _ width: Int) -> String {
        let t = collapse(s)
        let sc = Array(t.unicodeScalars)
        if sc.count <= width { return t }
        var v = String.UnicodeScalarView()
        v.append(contentsOf: sc[0..<width])
        return String(v) + "…"
    }
    static func hasDigit(_ s: String) -> Bool { s.unicodeScalars.contains { $0.properties.numericType == .decimal || $0.properties.numericType == .digit } }
}

public enum MemoryLines {
    static let rank: [String: Double] = ["low": 0, "medium": 1, "high": 2]
    static let groups: [String: String] = [
        "PERSON": "who", "GROUP": "who", "ORG": "who", "ROLE": "who", "PET": "who", "DATETIME": "when", "DURATION": "when",
        "RECURRENCE": "when", "PLACE": "where", "VENUE": "where", "ADDRESS": "where", "SPOT": "where", "MONEY": "num",
        "QUANTITY": "num", "CONTACT": "num", "IDENTIFIER": "num", "HEALTH": "health", "MEDICATION": "health", "DIET": "health",
        "EVENT": "event", "APPOINTMENT": "event", "TASK": "event", "TRIP": "event", "MILESTONE": "event"]
    static let groupOrder = ["who", "when", "where", "num", "health", "event"]
    static let key: Set<String> = ["when", "where", "num", "health", "event"]
    static let pronStart = try! NSRegularExpression(
        pattern: "^\\W*(it|that|this|they|them|he|she|those|these|yes|yeah|no|so|and|but|oh)\\b", options: [.caseInsensitive])

    static func pronounStart(_ s: String) -> Bool { pronStart.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }

    public static func needsContext(_ u: Int, _ tx: [String]) -> Bool {
        u > 0 && (PyText.words(tx[u]) < 8 || pronounStart(tx[u]))
    }

    /// eval/validator.py line_features: (features, frame names in order) or nil when u is not a candidate.
    public static func features(_ u: Int, _ graphs: [Graph3], _ sp: [String], _ tx: [String], _ spec: ValidatorSpec)
        -> ([Double], [String])? {
        var fs: [(Int, Frame3)] = []
        for (gi, g) in graphs.enumerated() {
            for f in g.frames where f.utt == u && f.p >= spec.minP && !f.roles.isEmpty { fs.append((gi, f)) }
        }
        let ents = graphs.flatMap { $0.entities.filter { $0.utt == u } }
        var grp = Dictionary(uniqueKeysWithValues: groupOrder.map { ($0, 0.0) })
        for e in ents { if let k = groups[e.coarse] { grp[k] = 1 } }
        if fs.isEmpty && !(spec.entLines && key.contains { grp[$0] == 1 }) { return nil }
        let n = Double(graphs.count)
        let names = fs.map(\.1.frame)
        let ps = fs.isEmpty ? [0.0] : fs.map(\.1.p)
        var x: [Double] = [
            1, Double(Set(fs.map(\.0)).count), Double(fs.count) / n, ps.max()!, ps.reduce(0, +) / Double(ps.count),
            fs.map(\.1.frameP).max() ?? 0, (fs.map { rank[$0.1.importance] ?? 1 }.max() ?? 0) / 2, Double(ents.count) / n]
        x += groupOrder.map { grp[$0]! }
        x += [log1p(Double(PyText.words(tx[u]))), sp[u] == "ME" ? 1 : 0, PyText.hasDigit(tx[u]) ? 1 : 0,
              tx[u].contains("?") ? 1 : 0, !names.isEmpty && Set(names) == ["Opinion"] ? 1 : 0,
              graphs.map(\.salience).reduce(0, +) / n, fs.contains { $0.1.flags["about_user"] == true } ? 1 : 0]
        if spec.feat >= 2 {
            let prev = graphs.contains { $0.entities.contains { $0.utt == u - 1 } }
            x += [ps.max()! >= 0.5 ? 1 : 0, fs.isEmpty ? 1 : 0, PyText.words(tx[u]) < 8 ? 1 : 0, pronounStart(tx[u]) ? 1 : 0,
                  prev ? 1 : 0]
        }
        var seen = Set<String>(), shown: [String] = []
        for nm in names.map({ $0.replacingOccurrences(of: "_", with: " ") }) where seen.insert(nm).inserted { shown.append(nm) }
        return (x, shown)
    }

    /// eval/validator.py g_line.
    public static func line(_ u: Int, _ frames: [String], _ sp: [String], _ tx: [String], _ spec: ValidatorSpec) -> String {
        let text = spec.clip > 0 ? PyText.clipStart(tx[u], spec.clip) : PyText.collapse(tx[u])
        var s = frames.isEmpty ? "\(sp[u]): \"\(text)\"" : "[\(frames.joined(separator: ", "))] \(sp[u]): \"\(text)\""
        if spec.ctx && needsContext(u, tx) { s += " (after \(sp[u - 1]): \"\(PyText.clipStart(tx[u - 1], 160))\")" }
        return s
    }

    // model/formats.py _names (resolve=True): display names as seen from frame f's line.
    static let pron: [String: String] = ["it": "thing", "that": "thing", "this": "thing", "those": "any", "these": "any",
                                         "them": "any", "they": "any", "he": "PERSON", "him": "PERSON", "she": "PERSON",
                                         "her": "PERSON", "one": "thing"]
    static let first: Set<String> = ["i", "me", "myself", "we", "us"]
    static let you = try! NSRegularExpression(pattern: "\\b(you|your|yours|yourself)\\b", options: [.caseInsensitive])

    static func names(_ g: Graph3, _ sp: [String], _ tx: [String], _ f: Frame3) -> [String: String] {
        var name: [String: String] = ["USER": "ME"]
        for e in g.entities { name[e.id] = e.text }
        let order = g.entities.indices.sorted { (g.entities[$0].utt, $0) < (g.entities[$1].utt, $1) }
        for (k, i) in order.enumerated() {
            guard let want = pron[g.entities[i].text.lowercased()] else { continue }
            for j in order[..<k].reversed() {
                let c = g.entities[j]
                let low = c.text.lowercased()
                if pron[low] != nil || first.contains(low) { continue }
                if want == "any" || (want == "PERSON") == (c.coarse == "PERSON") { name[g.entities[i].id] = c.text; break }
            }
        }
        let u = f.utt
        let other = u >= 0 && u < sp.count && sp[u] != "ME" ? sp[u] : nil
        if let o = other, you.firstMatch(in: tx[u], range: NSRange(tx[u].startIndex..., in: tx[u])) == nil { name["USER"] = o }
        let me = other ?? "ME"
        for (k, v) in name where k != "USER" && first.contains(v.lowercased()) { name[k] = me }
        return name
    }

    /// eval/validator.py record (without ids, session and score; the caller adds them).
    static func recordParts(_ u: Int, _ graphs: [Graph3], _ sp: [String], _ tx: [String])
        -> ([RecordFrame], [RecordEntity], RecordContext?) {
        var fr: [RecordFrame] = []
        for g in graphs {
            let byId = Dictionary(g.frames.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
            for f in g.frames where f.utt == u && !f.roles.isEmpty {
                let nm = names(g, sp, tx, f)
                let roles = f.roles.map { r, xs in
                    RecordRole(role: r, fillers: xs.map { x in byId[x].map { "[\($0.frame.replacingOccurrences(of: "_", with: " ")): \($0.trigger)]" } ?? nm[x] ?? x })
                }
                let item = RecordFrame(frame: f.frame, trigger: f.trigger, p: f.p, modality: f.modality, tense: f.tense, roles: roles)
                if let i = fr.firstIndex(where: { $0.frame == item.frame && $0.trigger == item.trigger && Set($0.roles) == Set(item.roles) }) {
                    fr[i].p = max(fr[i].p, item.p)  // both models found it
                } else {
                    fr.append(item)
                }
            }
        }
        var ents: [RecordEntity] = [], at: [String: Int] = [:]
        for g in graphs { for e in g.entities where e.utt == u {
            let k = "\(e.text)|\(e.coarse)"
            if let i = at[k] { ents[i] = RecordEntity(text: e.text, coarse: e.coarse, type: e.type) }  // later model wins, as the Python dict
            else { at[k] = ents.count; ents.append(RecordEntity(text: e.text, coarse: e.coarse, type: e.type)) }
        } }
        return (fr, ents, u > 0 ? RecordContext(speaker: sp[u - 1], text: tx[u - 1]) : nil)
    }

    /// All kept memories of a window. graphs: one per System 1 model (two in the v4 system, one with v3b).
    public static func memories(_ w: Window, _ graphs: [Graph3], _ v: LineValidator, sessionId: String = "",
                                threshold: Double? = nil) -> [MemoryRecord] {
        let sp = w.speakerLabels
        let tx = w.utterances.map(\.text)
        let t = threshold ?? v.threshold
        var out: [MemoryRecord] = []
        for u in tx.indices {
            guard let (x, frames) = features(u, graphs, sp, tx, v.spec) else { continue }
            let s = v.score(x)
            guard s >= t else { continue }
            if Privacy.isNeverStore(tx[u]) { continue }  // card and government ID numbers are never stored
            let (fr, ents, ctx) = recordParts(u, graphs, sp, tx)
            out.append(MemoryRecord(id: "\(w.windowId)#\(u)", sessionId: sessionId, windowId: w.windowId, date: w.date,
                                    time: w.utterances[u].t, utt: u, speaker: sp[u], text: tx[u], line: line(u, frames, sp, tx, v.spec),
                                    frames: fr, entities: ents, context: ctx, score: (s * 1000).rounded() / 1000,
                                    sensitive: Privacy.isCredential(tx[u]) || fr.contains { $0.frame == "Credential" }))
        }
        return out
    }
}

extension Graph3 {
    /// A graph as eval/predict3.py writes it ({"entities": [...], "frames": [...], "salience": x}), for parity checks.
    public static func fromJSON(_ g: [String: Any]) -> Graph3 {
        let ents = (g["entities"] as? [[String: Any]] ?? []).map { e in
            Entity3(id: e["id"] as! String, text: e["text"] as! String, utt: e["utt"] as! Int, type: e["type"] as? String ?? "",
                    typeP: e["type_p"] as? Double ?? 1, coarse: e["coarse"] as? String ?? "TOPIC", p: e["p"] as? Double ?? 1,
                    link: e["link"] as? String)
        }
        let attrKeys = ["modality", "tense", "source", "sensitivity", "importance"]
        let frames = (g["frames"] as? [[String: Any]] ?? []).map { f in
            let roles = (f["roles"] as? [String: [String]] ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
            var attrs: [String: String] = [:]
            for k in attrKeys { if let v = f[k] as? String { attrs[k] = v } }
            return Frame3(id: f["id"] as! String, frame: f["frame"] as! String, frameP: f["frame_p"] as? Double ?? 1,
                          trigger: f["trigger"] as! String, utt: f["utt"] as! Int, p: f["p"] as? Double ?? 1, roles: roles, attrs: attrs,
                          flags: ["update_of_prior": f["update_of_prior"] as? Bool ?? false, "about_user": f["about_user"] as? Bool ?? false])
        }
        return Graph3(entities: ents, frames: frames, salience: g["salience"] as? Double ?? 0.5)
    }
}
