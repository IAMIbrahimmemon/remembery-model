import Foundation
import Tokenizers

/// Port of model/featurize.py (render) and model/nar3.py (featurize3) for inference. Offsets are UTF-8 byte offsets; the Python side
/// uses character offsets, which coincide for ASCII and keep the same token boundaries otherwise.
public struct Rendered {
    public let text: String
    public let bytes: [UInt8]
    public let uttStart: [Int]
    public let slotPos: [Int]
    public let userPos: Int
    public let stateRefs: [String]
}

public func render(_ w: Window, maxState: Int = 32) -> Rendered {
    var out = "[STATE]\n"
    var pos = out.utf8.count
    var slotPos: [Int] = []
    var refs: [String] = []
    // Python's w.state[-max_state:] keeps everything when max_state == 0; mirrored for parity
    for (k, s) in w.state.suffix(maxState == 0 ? w.state.count : maxState).enumerated() {
        let line = "#\(k + 1) \(s.type) \(s.canonical)\n"
        slotPos.append(pos)
        refs.append(s.ref)
        out += line
        pos += line.utf8.count
    }
    let userPos = pos
    out += "[USER]\n"
    pos += "[USER]\n".utf8.count
    var uttStart: [Int] = []
    for u in w.utterances {
        let spk = w.speakers[u.spk] == USER ? "USER" : u.spk
        let prefix = "[\(u.t)][\(spk)][asr=\(String(format: "%.2f", u.asrConf))] "
        uttStart.append(pos + prefix.utf8.count)
        let line = prefix + u.text + "\n"
        out += line
        pos += line.utf8.count
    }
    return Rendered(text: out, bytes: Array(out.utf8), uttStart: uttStart, slotPos: slotPos, userPos: userPos, stateRefs: refs)
}

public struct Features {
    public let inputIds: [Int]          // original vocabulary ids (before pruning remap)
    public let offsets: [(Int, Int)]
    public let uttMask: [Bool]
    public let wordStart: [Bool]        // token starts a word (featurize3's word_start)
    public let wordEnd: [Bool]
    public let tokUtt: [Int]            // utterance index per token, -1 outside
    public let spkNames: [String]       // other speakers, sorted labels (featurize3's spk_names)
    public let spkTok: [[Int]]          // per other speaker: the token ending their label in each line they speak
    public let uttSpk: [Int]            // per utterance: 0 USER, k + 1 for spkNames[k]
    public let slotIdx: [Int]
    public let userIdx: Int
    public let rendered: Rendered
}

/// Python str.isalnum() / str.isspace() for one Unicode scalar.
func pyIsAlnum(_ u: Unicode.Scalar) -> Bool {
    switch u.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
         .decimalNumber, .letterNumber, .otherNumber: return true
    default: return u.properties.numericType != nil
    }
}

func pyIsSpace(_ u: Unicode.Scalar) -> Bool { u.properties.isWhitespace || (0x1C...0x1F).contains(u.value) }

/// Per UTF-8 byte: is the character it belongs to a word character (alnum or apostrophe) / a space.
func byteClasses(_ text: String) -> (word: [Bool], space: [Bool]) {
    var word: [Bool] = [], space: [Bool] = []
    for u in text.unicodeScalars {
        let n = String(u).utf8.count
        let w = pyIsAlnum(u) || u == "'", sp = pyIsSpace(u)
        word += [Bool](repeating: w, count: n)
        space += [Bool](repeating: sp, count: n)
    }
    return (word, space)
}

/// Python str.strip().
func pyStrip(_ s: String) -> String {
    let u = Array(s.unicodeScalars)
    var a = 0, b = u.count
    while a < b && pyIsSpace(u[a]) { a += 1 }
    while b > a && pyIsSpace(u[b - 1]) { b -= 1 }
    var out = String.UnicodeScalarView()
    out.append(contentsOf: u[a..<b])
    return String(out)
}

/// GPT-2 byte-level BPE: each byte maps to one printable unicode scalar. Inverting it gives each token's byte length.
private let byteDecoder: [Character: UInt8] = {
    var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
    var cs = bs
    var n = 0
    for b in 0..<256 where !bs.contains(b) {
        bs.append(b)
        cs.append(256 + n)
        n += 1
    }
    var m: [Character: UInt8] = [:]
    for (b, c) in zip(bs, cs) { m[Character(UnicodeScalar(c)!)] = UInt8(b) }
    return m
}()

public final class Featurizer {
    let tokenizer: Tokenizer
    let maxLen: Int
    let specialIds: Set<Int>
    private var tokenBytes: [Int: Int] = [:]

    public init(tokenizer: Tokenizer, maxLen: Int, bosId: Int) {
        self.tokenizer = tokenizer
        self.maxLen = maxLen
        self.specialIds = [bosId]
    }

    private func byteLen(_ id: Int) -> Int {
        if let n = tokenBytes[id] { return n }
        var n = 0
        if !specialIds.contains(id), let tok = tokenizer.convertIdToToken(id) {
            for ch in tok { n += byteDecoder[ch] != nil ? 1 : ch.utf8.count }
        }
        tokenBytes[id] = n
        return n
    }

    public func featurize(_ w: Window) -> Features {
        var maxState = 32
        var r = render(w, maxState: maxState)
        var ids = tokenizer.encode(text: r.text)
        while ids.count > maxLen && maxState > 0 {
            maxState = max(0, maxState - 8)
            r = render(w, maxState: maxState)
            ids = tokenizer.encode(text: r.text)
        }
        ids = Array(ids.prefix(maxLen))
        var offsets: [(Int, Int)] = []
        var pos = 0
        for id in ids {
            let n = byteLen(id)
            offsets.append(n == 0 ? (0, 0) : (pos, pos + n))
            pos += n
        }
        // binary search over non-empty tokens, as in _char_to_token
        let idx = offsets.indices.filter { offsets[$0].1 > offsets[$0].0 }
        let starts = idx.map { offsets[$0].0 }
        let ends = idx.map { offsets[$0].1 }
        func bisectRight(_ a: [Int], _ x: Int) -> Int {
            var lo = 0, hi = a.count
            while lo < hi { let mid = (lo + hi) / 2; if x < a[mid] { hi = mid } else { lo = mid + 1 } }
            return lo
        }
        func bisectLeft(_ a: [Int], _ x: Int) -> Int {
            var lo = 0, hi = a.count
            while lo < hi { let mid = (lo + hi) / 2; if a[mid] < x { lo = mid + 1 } else { hi = mid } }
            return lo
        }
        func find(_ cs: Int, _ ce: Int) -> (Int, Int)? {
            let k = bisectRight(ends, cs)
            let j = bisectLeft(starts, ce) - 1
            if k >= idx.count || j < 0 || j < k { return nil }
            return (idx[k], idx[j])
        }
        func tokAt(_ c: Int) -> Int { find(c, c + 1)?.0 ?? 0 }
        var mask = [Bool](repeating: false, count: ids.count)
        var tokUtt = [Int](repeating: -1, count: ids.count)
        for (ui, u) in w.utterances.enumerated() {
            let cs = r.uttStart[ui]
            if let sp = find(cs, cs + u.text.utf8.count) {
                for i in sp.0...sp.1 { mask[i] = true; tokUtt[i] = ui }
            }
        }
        // word boundaries, as featurize3: a token starts a word if the char before its first non-space char is not a
        // word char, and ends one if the char right after it is not
        let (isWordByte, isSpaceByte) = byteClasses(r.text)
        func isWord(_ c: Int) -> Bool { c >= 0 && c < isWordByte.count && isWordByte[c] }
        let ws = offsets.map { (a, b) -> Bool in
            var f = a
            while f < b && isSpaceByte[f] { f += 1 }
            return b > a && !isWord(f - 1)
        }
        let we = offsets.map { (a, b) in b > a && !isWord(b) }
        // speakers, as featurize3: the label sits just before "][asr=..] " (all ASCII, so bytes = chars)
        let userLbl = w.speakers.sorted { $0.key < $1.key }.first { $0.value == USER }?.key
        let spkNames = Array(Set(w.utterances.map(\.spk)).subtracting(userLbl.map { [$0] } ?? [])).sorted()
        var spkTok = [[Int]](repeating: [], count: spkNames.count)
        var uttSpk: [Int] = []
        for (ui, u) in w.utterances.enumerated() {
            if u.spk == userLbl { uttSpk.append(0); continue }
            let k = spkNames.firstIndex(of: u.spk)!
            uttSpk.append(k + 1)
            let ce = r.uttStart[ui] - "][asr=\(String(format: "%.2f", u.asrConf))] ".utf8.count
            if let sp = find(ce - u.spk.utf8.count, ce), sp.1 < ids.count { spkTok[k].append(sp.1) }
        }
        return Features(inputIds: ids, offsets: offsets, uttMask: mask, wordStart: ws, wordEnd: we, tokUtt: tokUtt,
                        spkNames: spkNames, spkTok: spkTok, uttSpk: uttSpk,
                        slotIdx: r.slotPos.map(tokAt), userIdx: tokAt(r.userPos), rendered: r)
    }
}
