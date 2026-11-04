import ExecuTorch
import Foundation
import Tokenizers

/// System 1 v3 on device: memnar.pte (encode / score / link / roles / attrs) + the host logic of
/// export/ondevice3.py host_extract3, which reproduces model/decode3.py extract3.
public final class Extractor {
    public let meta: ModelMeta
    public let featurizer: Featurizer
    let module: Module
    let remap: [Int32]?

    public struct Timing {
        public var featurizeMs = 0.0, encodeMs = 0.0, scoreMs = 0.0, linkMs = 0.0, rolesMs = 0.0, attrsMs = 0.0, totalMs = 0.0
        public var tokens = 0, candidates = 0
    }
    public private(set) var lastTiming = Timing()

    /// dir holds memnar.pte, memnar_meta.json, vocab_remap.bin and tokenizer/.
    public static func load(from dir: URL) async throws -> Extractor {
        let meta = try makeDecoder().decode(ModelMeta.self, from: Data(contentsOf: dir.appendingPathComponent("memnar_meta.json")))
        let tok = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer"))
        var remap: [Int32]? = nil
        if meta.vocabPruned {
            let data = try Data(contentsOf: dir.appendingPathComponent("vocab_remap.bin"))
            remap = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        // mmap: the program's weights stay file-backed pages, so two models fit next to each other on the phone
        let module = Module(filePath: dir.appendingPathComponent("memnar.pte").path, loadMode: .mmap)
        for m in ["encode", "score", "link", "roles", "attrs"] { try module.load(m) }
        return Extractor(meta: meta, featurizer: Featurizer(tokenizer: tok, maxLen: meta.maxLen, bosId: meta.bosId),
                         module: module, remap: remap)
    }

    init(meta: ModelMeta, featurizer: Featurizer, module: Module, remap: [Int32]?) {
        self.meta = meta; self.featurizer = featurizer; self.module = module; self.remap = remap
    }

    private func run(_ method: String, _ inputs: [any ValueConvertible]) throws -> [Value] {
        try module.execute(method, inputs)
    }

    private static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

    private func ints(_ v: [Int]) -> Tensor<Int64> { Tensor<Int64>(v.map(Int64.init), shape: [v.count]) }

    /// Pad a list of token positions to the program's minimum dynamic size by repeating the last one.
    private func pad(_ xs: [Int]) -> [Int] { xs + [Int](repeating: xs.last ?? 0, count: max(0, meta.padMin - xs.count)) }

    private func floats(_ v: Value, _ name: String) throws -> [Float] {
        guard let t: Tensor<Float> = v.tensor() else { throw ExtractError.badOutput(name) }
        return t.scalars()
    }

    private func longs(_ v: Value, _ name: String) throws -> [Int] {
        guard let t: Tensor<Int64> = v.tensor() else { throw ExtractError.badOutput(name) }
        return t.scalars().map { Int($0) }
    }

    /// Thresholds default to the model's (System 1 alone); the app passes trgThresh = s2Lo so System 2 sees the band.
    public func extract(_ w: Window, entThresh: Double? = nil, trgThresh: Double? = nil, roleThresh: Double? = nil) throws -> (Graph3, Features) {
        let entT = entThresh ?? meta.entThresh, trgT = trgThresh ?? meta.trgThresh, roleT = roleThresh ?? meta.roleThresh
        var tm = Timing()
        let t0 = Self.now()
        let f = featurizer.featurize(w)
        tm.featurizeMs = Self.now() - t0
        let L = f.inputIds.count
        let d = meta.d
        tm.tokens = L

        // encode
        var t = Self.now()
        let ids: [Int64] = f.inputIds.map { id in Int64(remap.map { id < $0.count ? Int($0[id]) : 0 } ?? id) }
        let h = try floats(run("encode", [Tensor<Int64>(ids, shape: [1, L])])[0], "encode")
        tm.encodeMs = Self.now() - t
        let hTensor = { Tensor<Float>(h, shape: [L, d]) }

        // candidate spans: <= MAX_SPAN tokens inside one utterance run, word-aligned start and end (nar3.candidate_spans)
        var cands: [(Int, Int)] = []
        for s in 0..<L where f.uttMask[s] {
            for wd in 0..<meta.maxSpan {
                let e = s + wd
                if e >= L || !f.uttMask[e] { break }
                if f.wordStart[s] && f.wordEnd[e] { cands.append((s, e)) }
            }
        }
        tm.candidates = cands.count

        // entity and trigger heads over all candidates, in chunks
        t = Self.now()
        var pEnt: [Float] = [], typ: [Int] = [], typeP: [Float] = [], coarse: [Int] = []
        var pTrg: [Float] = [], frame: [Int] = [], frameP: [Float] = []
        for c in stride(from: 0, to: cands.count, by: meta.chunk) {
            let ch = cands[c..<min(c + meta.chunk, cands.count)]
            let o = try run("score", [hTensor(), ints(pad(ch.map(\.0))), ints(pad(ch.map(\.1)))])
            let n = ch.count
            pEnt += try floats(o[0], "score").prefix(n); typ += try longs(o[1], "score").prefix(n)
            typeP += try floats(o[2], "score").prefix(n); coarse += try longs(o[3], "score").prefix(n)
            pTrg += try floats(o[4], "score").prefix(n); frame += try longs(o[5], "score").prefix(n)
            frameP += try floats(o[6], "score").prefix(n)
        }
        tm.scoreMs = Self.now() - t

        // greedy: highest p first (stable on ties), no overlap, at most max_items; p < thr compares in float32
        func pick(_ p: [Float], _ thr: Double) -> [Int] {
            let order = p.indices.sorted { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 }
            var used = [Bool](repeating: false, count: L)
            var out: [Int] = []
            for i in order {
                if p[i] < Float(thr) || out.count >= meta.maxItems { break }
                let (s, e) = cands[i]
                if (s...e).contains(where: { used[$0] }) { continue }
                for k in s...e { used[k] = true }
                out.append(i)
            }
            return out.sorted { cands[$0] < cands[$1] }
        }

        let bytes = f.rendered.bytes
        func spanText(_ s: Int, _ e: Int) -> String {
            let a = f.offsets[s].0, b = f.offsets[e].1
            return pyStrip(String(decoding: bytes[a..<max(a, b)], as: UTF8.self))
        }
        // utterance of a token: by the last byte of the token (its start may be the space before the utterance)
        func whereUtt(_ s: Int) -> Int {
            let c = f.offsets[s].1 - 1
            return f.rendered.uttStart.indices.filter { f.rendered.uttStart[$0] <= c }.max() ?? 0
        }

        var ents: [Entity3] = []
        let ei = pick(pEnt, entT)
        t = Self.now()
        if !ei.isEmpty {
            var si = [Int](repeating: 0, count: meta.nSlots)
            var sv = [Float](repeating: 0, count: meta.nSlots)
            for (k, idx) in f.slotIdx.prefix(meta.nSlots).enumerated() { si[k] = idx; sv[k] = 1 }
            let link = try longs(run("link", [hTensor(), ints(pad(ei.map { cands[$0].0 })), ints(pad(ei.map { cands[$0].1 })),
                                              ints(si), Tensor<Float>(sv, shape: [meta.nSlots])])[0], "link")
            for (k, i) in ei.enumerated() {
                let (s, e) = cands[i]
                let refs = f.rendered.stateRefs
                ents.append(Entity3(id: "e\(k + 1)", text: spanText(s, e), utt: whereUtt(s), type: meta.types[typ[i]],
                                    typeP: pyRound3(Double(typeP[i])), coarse: meta.coarse[coarse[i]], p: pyRound3(Double(pEnt[i])),
                                    link: link[k] > 0 && link[k] - 1 < refs.count ? refs[link[k] - 1] : nil))
            }
        }
        tm.linkMs = Self.now() - t

        var frames: [Frame3] = []
        let ti = pick(pTrg, trgT)
        let tSt = pad(ti.map { cands[$0].0 }), tEn = pad(ti.map { cands[$0].1 })
        t = Self.now()
        let ao = try run("attrs", [hTensor(), Tensor<Float>(f.uttMask.map { $0 ? 1 : 0 }, shape: [L]), ints(tSt), ints(tEn)])
        let at = try longs(ao[0], "attrs"), fl = try floats(ao[1], "attrs"), sal = try floats(ao[2], "attrs")[0]
        tm.attrsMs = Self.now() - t
        if !ti.isEmpty {
            t = Self.now()
            // arguments: entities, USER, triggers, then (v4) other speakers by label; one pair per (trigger, argument
            // other than itself), in extract3's order. spk: 0 USER, k + 1 speaker k, -1 otherwise
            var args: [(name: String, s: Int, e: Int, kind: Int, spk: Int)] = zip(ents, ei).map { ($0.id, cands[$1].0, cands[$1].1, 0, -1) }
                + [(USER, f.userIdx, f.userIdx, 1, 0)]
                + ti.enumerated().map { ("f\($0.offset + 1)", cands[$0.element].0, cands[$0.element].1, 2, -1) }
            let speakers = meta.speakers ?? false, nsMax = meta.nsMax ?? 32
            if speakers { for (k, nm) in f.spkNames.prefix(nsMax).enumerated() { args.append((nm, k, k, 4, k + 1)) } }
            var pairs: [(k: Int, q: Int)] = []
            for k in ti.indices { for (q, a) in args.enumerated() where a.name != "f\(k + 1)" { pairs.append((k, q)) } }
            var xs: [any ValueConvertible] = [hTensor(), ints(pad(pairs.map { cands[ti[$0.k]].0 })), ints(pad(pairs.map { cands[ti[$0.k]].1 })),
                      ints(pad(pairs.map { args[$0.q].s })), ints(pad(pairs.map { args[$0.q].e })),
                      ints(pad(pairs.map { args[$0.q].kind })), ints(pad(pairs.map { frame[ti[$0.k]] })), ints(f.tokUtt)]
            if speakers {
                // the speaker of each trigger's line (-9 outside a line); speaker label tokens as a flat list, padded
                // with the dummy speaker nsMax, which the program ignores
                let tSpk = pairs.map { pq -> Int in let u = f.tokUtt[cands[ti[pq.k]].0]; return u >= 0 ? f.uttSpk[u] : -9 }
                var sIdx: [Int] = [], sSeg: [Int] = []
                for (k, toks) in f.spkTok.prefix(nsMax).enumerated() { sIdx += toks; sSeg += [Int](repeating: k, count: toks.count) }
                while sIdx.count < meta.padMin { sIdx.append(0); sSeg.append(nsMax) }
                xs += [ints(pad(pairs.map { args[$0.q].spk })), ints(pad(tSpk)), ints(sIdx), ints(sSeg)]
            }
            let ro = try run("roles", xs)
            let rj = try longs(ro[0], "roles"), rp = try floats(ro[1], "roles")
            tm.rolesMs = Self.now() - t
            let nK = meta.attrKeys.count, nF = meta.flags.count
            for (k, i) in ti.enumerated() {
                let (s, e) = cands[i]
                let fid = frame[i]
                let roleNames = meta.frameRoles[fid]
                var order: [String] = []
                var best: [String: [(Double, String)]] = [:]
                for (n, pq) in pairs.enumerated() where pq.k == k {
                    let j = rj[n], pr = Double(rp[n])
                    if j < roleNames.count && pr >= roleT {
                        if best[roleNames[j]] == nil { order.append(roleNames[j]) }
                        best[roleNames[j], default: []].append((pr, args[pq.q].name))
                    }
                }
                // lst.sort(reverse=True): by p, then by name, both descending
                let roles: [(String, [String])] = order.map { r in
                    let lst = best[r]!.sorted { $0.0 != $1.0 ? $0.0 > $1.0 : $0.1 > $1.1 }
                    return (r, lst.prefix(r.hasPrefix("Co") ? 2 : 1).map(\.1))
                }
                var attrs: [String: String] = [:]
                for (q, a) in meta.attrKeys.enumerated() { attrs[a] = meta.attrValues[a]![at[k * nK + q]] }
                var flags: [String: Bool] = [:]
                for (q, fName) in meta.flags.enumerated() { flags[fName] = Double(fl[k * nF + q]) > 0.5 }
                frames.append(Frame3(id: "f\(k + 1)", frame: meta.frames[fid], frameP: pyRound3(Double(frameP[i])),
                                     trigger: spanText(s, e), utt: whereUtt(s), p: pyRound3(Double(pTrg[i])),
                                     roles: roles, attrs: attrs, flags: flags))
            }
        }
        tm.totalMs = Self.now() - t0
        lastTiming = tm
        return (Graph3(entities: ents, frames: frames, salience: pyRound3(Double(sal))), f)
    }
}

public enum ExtractError: Error { case badOutput(String) }
