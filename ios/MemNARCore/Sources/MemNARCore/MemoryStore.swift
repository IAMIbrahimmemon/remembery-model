import Foundation

/// v3 memory store: compiled memory lines plus the frame graph of every window they came from.
/// Routing follows eval/s2_v3.py: frames with p >= hi are stored, frames with lo <= p < hi wait for System 2.
public enum MemStatus: String, CaseIterable {
    case stored        // System 1 alone, p >= hi
    case pending       // waiting for System 2
    case confirmed     // System 2 confirmed it
    case fixed         // System 2 rewrote it
    case rejected      // System 2 rejected it
    case unchecked     // System 2 failed or was unavailable; not used for answers (as in s2_v3)
    case onDevice      // sensitive (decision 25): stored, never sent to System 2 or the reader

    public var isMemory: Bool { [.stored, .confirmed, .fixed, .onDevice].contains(self) }
}

public struct Memory: Identifiable, Hashable {
    public let id: String
    public var text: String
    public let original: String       // the System 1 line (differs from text after a System 2 fix)
    public let windowId: String
    public let tObserved: String
    public let frame: String
    public let p: Double
    public let importance: String
    public let speaker: String?       // nil when the user said it
    public let evidence: String       // the line the frame was evoked in (retrieval and the reader)
    public var status: MemStatus
    public var note: String?
    public var access = 0
}

/// Decision 25: passwords, PINs and door codes stay on the phone (stored, never sent to System 2 or the reader);
/// full card numbers and government ID numbers are never stored.
public enum Privacy {
    static let credential = try! NSRegularExpression(pattern:
        "\\b(pass ?words?|passcodes?|pins?|pin (numbers?|codes?)|(door|gate|garage|alarm|lock|access|entry|wi-?fi|security|safe) codes?|log ?ins?|credentials?)\\b",
        options: [.caseInsensitive])
    static let cardNumber = try! NSRegularExpression(pattern: "\\b(?:\\d[ -]?){12,18}\\d\\b")
    static let govId = try! NSRegularExpression(pattern: "\\b\\d{3}-\\d{2}-\\d{4}\\b")
    static let idWords = try! NSRegularExpression(pattern:
        "\\b(social security|passport|driver'?s licen[cs]e|national id|tax id|card number|credit card|debit card|ssn)\\b",
        options: [.caseInsensitive])
    static let digits = try! NSRegularExpression(pattern: "\\d{4,}")

    static func has(_ re: NSRegularExpression, _ s: String) -> Bool { re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }

    public static func isCredential(_ s: String) -> Bool { has(credential, s) }
    public static func isNeverStore(_ s: String) -> Bool { has(cardNumber, s) || has(govId, s) || (has(idWords, s) && has(digits, s)) }

    /// Utterances a frame touches: its trigger's and every entity filler's.
    static func utterances(_ f: Frame3, _ g: Graph3) -> Set<Int> {
        var u: Set<Int> = [f.utt]
        let byId = Dictionary(uniqueKeysWithValues: g.entities.map { ($0.id, $0) })
        for (_, xs) in f.roles { for x in xs { if let e = byId[x] { u.insert(e.utt) } } }
        return u
    }

    public enum Class { case normal, onDevice, drop }

    public static func classify(_ f: Frame3, _ g: Graph3, _ w: Window) -> Class {
        let texts = utterances(f, g).filter { $0 < w.utterances.count }.map { w.utterances[$0].text }
        if texts.contains(where: isNeverStore) { return .drop }
        if f.frame == "Credential" || texts.contains(where: isCredential) { return .onDevice }
        return .normal
    }

    /// The transcript System 2 sees: lines with a credential or an ID number are withheld.
    public static func redacted(_ w: Window) -> [String] {
        let spk = w.speakerLabels
        return w.utterances.enumerated().map { i, u in
            let text = isNeverStore(u.text) || isCredential(u.text) ? "(withheld: kept on the phone)" : u.text
            return "[\(i)] \(spk[i]): \(text)"
        }
    }
}

public final class MemoryStore {
    public private(set) var memories: [Memory] = []
    public private(set) var graphs: [String: Graph3] = [:]
    public private(set) var windows: [String: Window] = [:]
    public let lo: Double
    public let hi: Double

    public init(lo: Double, hi: Double) { self.lo = lo; self.hi = hi }

    public struct IngestCounts { public var stored = 0, pending = 0, onDevice = 0, dropped = 0 }

    /// Store a window's sure lines; uncertain lines are returned as pending (their ids) for System 2.
    @discardableResult
    public func ingest(_ w: Window, _ g: Graph3) -> (IngestCounts, [String]) {
        windows[w.windowId] = w
        graphs[w.windowId] = g
        let when = "\(w.date)T\(w.utterances.first?.t ?? "00:00:00")"
        let spk = w.speakerLabels
        let (sure, unsure) = Compile.route(g, w, lo: lo, hi: hi)
        var c = IngestCounts()
        var pending: [String] = []
        for (isSure, items) in [(true, sure), (false, unsure)] {
            for (line, f) in items {
                let cls = Privacy.classify(f, g, w)
                if cls == .drop { c.dropped += 1; continue }
                // a sensitive uncertain line is kept on the phone too: it cannot be checked without sending it
                let status: MemStatus = cls == .onDevice ? .onDevice : (isSure ? .stored : .pending)
                let m = Memory(id: String(UUID().uuidString.prefix(8)).lowercased(), text: line, original: line,
                               windowId: w.windowId, tObserved: when, frame: f.frame, p: f.p, importance: f.importance,
                               speaker: f.utt < spk.count && spk[f.utt] != "ME" ? spk[f.utt] : nil,
                               evidence: f.utt < w.utterances.count ? "\(spk[f.utt]): \(w.utterances[f.utt].text)" : "", status: status)
                memories.append(m)
                switch status {
                case .onDevice: c.onDevice += 1
                case .pending: c.pending += 1; pending.append(m.id)
                default: c.stored += 1
                }
            }
        }
        return (c, pending)
    }

    public func memory(_ id: String) -> Memory? { memories.first { $0.id == id } }

    public func setVerdict(_ id: String, _ status: MemStatus, fixed: String? = nil, note: String? = nil) {
        guard let i = memories.firstIndex(where: { $0.id == id }) else { return }
        memories[i].status = status
        if let t = fixed { memories[i].text = t }
        memories[i].note = note
    }

    func touch(_ ids: Set<String>) {
        for i in memories.indices where ids.contains(memories[i].id) { memories[i].access += 1 }
    }

    public var counts: [MemStatus: Int] { Dictionary(grouping: memories, by: \.status).mapValues(\.count) }
}
