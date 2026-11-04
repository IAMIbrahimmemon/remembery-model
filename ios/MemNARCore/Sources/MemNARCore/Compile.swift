import Foundation

/// Port of model/decode3.py compile_memories: one readable line per frame with at least one filled role, e.g.
///     Borrowing [has]: Borrower=Mike; Theme=my ladder; Lender=ME; Time=from july
public enum Compile {
    static let first: Set<String> = ["i", "me", "myself", "we", "us"]
    static let you = try! NSRegularExpression(pattern: "\\b(you|your|yours|yourself)\\b", options: [.caseInsensitive])
    static let rank = ["low": 0, "medium": 1, "high": 2]

    static func saysYou(_ s: String) -> Bool { you.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil }

    /// Lines in order, each with the frame it came from (the first frame, when identical lines are merged).
    /// speakers[i] = "ME" or the speaker label of utterance i: a frame evoked in someone else's line is recorded as
    /// theirs ("S2 said: ..."). With uttText, a role the model filled with ME inside someone else's line that never says
    /// "you"/"your" is given to that speaker. core: frame -> core roles (needed only for requireCore). context: all
    /// frames of the window, for showing a role filled by a frame that g leaves out.
    public static func lines(_ g: Graph3, minP: Double = 0, requireCore: Bool = false, minImportance: String = "low",
                             speakers: [String]? = nil, uttText: [String]? = nil, minRoles: Int = 1,
                             core: [String: Set<String>] = [:], context: [Frame3]? = nil) -> [(line: String, frame: Frame3)] {
        var name: [String: String] = [USER: "ME"]
        for e in g.entities { name[e.id] = e.text }
        var fr: [String: Frame3] = [:]
        for f in context ?? g.frames { fr[f.id] = f }
        var out: [(line: String, frame: Frame3)] = []
        for f in g.frames {
            if f.p < minP || f.roles.isEmpty { continue }
            if f.roles.reduce(0, { $0 + $1.1.count }) < minRoles { continue }
            if requireCore && Set(f.roles.map(\.0)).isDisjoint(with: (core[f.frame] ?? []).union(["Condition", "New_value"])) { continue }
            if (rank[f.importance] ?? 1) < rank[minImportance]! { continue }
            let inRange = speakers.map { 0 <= f.utt && f.utt < $0.count } ?? false
            let other: String? = inRange && speakers![f.utt] != "ME" ? speakers![f.utt] : nil
            var here = name
            if let o = other, let txt = uttText, !saysYou(txt[f.utt]) { here[USER] = o }
            let me = other ?? "ME"  // first-person pronouns kept as entities mean the speaker of that line
            for (k, v) in here where k != USER && first.contains(v.lowercased()) { here[k] = me }

            func show(_ x: String) -> String {
                if let t = fr[x] { return "[\(t.frame.replacingOccurrences(of: "_", with: " ")): \(t.trigger)]" }
                return here[x] ?? x
            }
            let roles = f.roles.map { r, xs in "\(r)=\(xs.map(show).joined(separator: " & "))" }.joined(separator: "; ")
            var tags = f.modality != "asserted" ? [f.modality] : []
            if f.tense == "past" || f.tense == "future" { tags.append(f.tense) }
            if f.flags["update_of_prior"] == true { tags.append("changed") }
            var line = "\(f.frame.replacingOccurrences(of: "_", with: " ")) [\(f.trigger)]: \(roles)"
                + (tags.isEmpty ? "" : " (\(tags.joined(separator: ", ")))")
            if let o = other { line = "\(o) said: \(line)" }
            if !out.contains(where: { $0.line == line }) { out.append((line, f)) }
        }
        return out
    }

    public static func memories(_ g: Graph3, minP: Double = 0, speakers: [String]? = nil, uttText: [String]? = nil) -> [String] {
        lines(g, minP: minP, speakers: speakers, uttText: uttText).map(\.line)
    }

    /// System 2 routing (eval/s2_v3.py): frames with p >= hi are stored; lines of frames with lo <= p < hi that are not
    /// already stored go to System 2.
    public static func route(_ g: Graph3, _ w: Window, lo: Double, hi: Double)
        -> (sure: [(line: String, frame: Frame3)], unsure: [(line: String, frame: Frame3)]) {
        let spk = w.speakerLabels, txt = w.utterances.map(\.text)
        let sure = lines(g, minP: hi, speakers: spk, uttText: txt)
        let band = Graph3(entities: g.entities, frames: g.frames.filter { lo <= $0.p && $0.p < hi }, salience: g.salience)
        let unsure = lines(band, speakers: spk, uttText: txt, context: g.frames).filter { u in !sure.contains { $0.line == u.line } }
        return (sure, unsure)
    }
}
