import Foundation
import MemNARCore

/// Parity check: Swift featurizer + extractor + compile_memories vs Python over the same .pte (ios/Model/parity3.json,
/// written by export/ondevice3.py).
///   swift run -c release memnar-cli <model dir>                 parity on every fixture, memories printed for the probes
///   BASETEN_KEY=... swift run -c release memnar-cli <model dir> --s2 probe_kitchen probe_work
///                                                               + System 2 on those windows' uncertain memories
struct Fixture: Decodable {
    let window: Window
    let inputIds: [Int]
    let uttMask: [Int]
    let wordStart: [Int]
    let wordEnd: [Int]
    let tokUtt: [Int]
    let slotIdx: [Int]
    let userIdx: Int
    let graph: G
    let memoriesAll: [String]
    let memoriesSure: [String]
    let memoriesUnsure: [String]
    let graphS1: G
    let memoriesS1: [String]

    struct G: Decodable {
        let entities: [E]
        let frames: [F]
        let salience: Double
    }
    struct E: Decodable { let id: String; let text: String; let utt: Int; let type: String; let typeP: Double; let coarse: String; let p: Double; let link: String? }
    struct F: Decodable {
        let id: String; let frame: String; let frameP: Double; let trigger: String; let utt: Int; let p: Double
        let modality: String; let tense: String; let source: String; let sensitivity: String; let importance: String
        let updateOfPrior: Bool; let aboutUser: Bool
    }
}

func entKey(_ e: Fixture.E) -> String { "\(e.id)|\(e.text)|\(e.utt)|\(e.type)|\(e.coarse)|\(e.link ?? "-")" }
func entKey(_ e: Entity3) -> String { "\(e.id)|\(e.text)|\(e.utt)|\(e.type)|\(e.coarse)|\(e.link ?? "-")" }
/// Role names come from the raw JSON: convertFromSnakeCase would also rewrite dictionary keys ("Place_of_employment").
func frameKey(_ f: Fixture.F, _ r: [String: [String]]) -> String {
    let roles = r.keys.sorted().map { "\($0)=\(r[$0]!.joined(separator: ","))" }.joined(separator: ";")
    return "\(f.id)|\(f.frame)|\(f.trigger)|\(f.utt)|\(roles)|\(f.modality)|\(f.tense)|\(f.source)|\(f.sensitivity)|\(f.importance)|\(f.updateOfPrior)|\(f.aboutUser)"
}
func frameKey(_ f: Frame3) -> String {
    let r = Dictionary(uniqueKeysWithValues: f.roles.map { ($0.0, $0.1) })
    let roles = r.keys.sorted().map { "\($0)=\(r[$0]!.joined(separator: ","))" }.joined(separator: ";")
    let a = f.attrs
    return "\(f.id)|\(f.frame)|\(f.trigger)|\(f.utt)|\(roles)|\(a["modality"]!)|\(a["tense"]!)|\(a["source"]!)|\(a["sensitivity"]!)|\(a["importance"]!)|\(f.flags["update_of_prior"]!)|\(f.flags["about_user"]!)"
}

/// Graphs equal: same entities and frames in the same order, and every probability within 0.001 (rounded to 3 places
/// on both sides, so this only allows a last-digit rounding difference).
func sameGraph(_ g: Graph3, _ p: Fixture.G, _ rawRoles: [[String: [String]]]) -> (Bool, String) {
    let se = g.entities.map(entKey), pe = p.entities.map(entKey)
    let sf = g.frames.map(frameKey), pf = zip(p.frames, rawRoles).map { frameKey($0, $1) }
    var why: [String] = []
    if se != pe { why.append("entities swift-only \(Set(se).subtracting(pe).sorted()) python-only \(Set(pe).subtracting(se).sorted())") }
    if sf != pf { why.append("frames swift-only \(Set(sf).subtracting(pf).sorted()) python-only \(Set(pf).subtracting(sf).sorted())") }
    if se == pe {
        for (a, b) in zip(g.entities, p.entities) where abs(a.p - b.p) > 0.0011 || abs(a.typeP - b.typeP) > 0.0011 { why.append("p \(a.text) \(a.p)/\(b.p)") }
    }
    if sf == pf {
        for (a, b) in zip(g.frames, p.frames) where abs(a.p - b.p) > 0.0011 || abs(a.frameP - b.frameP) > 0.0011 { why.append("p \(a.trigger) \(a.p)/\(b.p)") }
    }
    if abs(g.salience - p.salience) > 0.0011 { why.append("salience \(g.salience)/\(p.salience)") }
    return (why.isEmpty, why.joined(separator: "; "))
}

func diff(_ name: String, _ s: [String], _ p: [String]) -> String? {
    s == p ? nil : "\(name): swift-only \(s.filter { !p.contains($0) }) python-only \(p.filter { !s.contains($0) })\(Set(s) == Set(p) ? " (order)" : "")"
}

let args = CommandLine.arguments

/// `memnar-cli phonon <Phonon dir> <parity dir>`: Swift features, tokens and text vs Python (parity.json, written by
/// export/phonon_coreml.py into ios/Phonon_build/parity); `memnar-cli phonon <Phonon dir> file.f32 ...` transcribes
/// files (16 kHz mono float32).
if args.count > 2 && args[1] == "phonon" {
    struct PFix: Decodable {
        let audio: String; let frames: Int; let featuresHead: [[Float]]; let featuresSum: Double; let featuresAbsSum: Double
        let tokens: [Int]; let probs: [Double]; let text: String
    }
    let dir = URL(fileURLWithPath: args[2])
    let t0 = Date()
    let asr = try! PhononASR(dir: dir)
    print(String(format: "Phonon loaded in %.1f s", Date().timeIntervalSince(t0)))
    func samples(_ u: URL) -> [Float] { let d = try! Data(contentsOf: u); return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    let pdir = URL(fileURLWithPath: args.count > 3 ? args[3] : "\(args[2])/../Phonon_build/parity")
    if args.count > 3 && args[3].hasSuffix(".f32") {
        for p in args[3...] { let r = try! asr.transcribe(samples(URL(fileURLWithPath: p))); print(String(format: "%.0f ms %.2f  %@", r.ms, r.confidence, r.text)) }
        exit(0)
    }
    let fx = try! makeDecoder().decode([PFix].self, from: Data(contentsOf: pdir.appendingPathComponent("parity.json")))
    var featOK = 0, tokOK = 0, textOK = 0, ms = 0.0, audio = 0.0, maxHead: Float = 0, maxSumRel = 0.0
    for f in fx {
        let x = samples(pdir.appendingPathComponent(f.audio))
        let (feats, F) = asr.frontend.features(x)
        let M = asr.meta.nMels
        var head: Float = 0
        for t in 0..<f.featuresHead.count { for m in 0..<f.featuresHead[t].count { head = max(head, abs(feats[t * M + m] - f.featuresHead[t][m])) } }
        let sum = feats.reduce(0.0) { $0 + Double($1) }, abs_ = feats.reduce(0.0) { $0 + Double(abs($1)) }
        let rel = abs(abs_ - f.featuresAbsSum) / f.featuresAbsSum
        maxHead = max(maxHead, head); maxSumRel = max(maxSumRel, rel)
        if F == f.frames && head < 1e-3 && rel < 1e-4 { featOK += 1 } else { print("features \(f.audio): frames \(F)/\(f.frames) head \(head) sum \(sum)/\(f.featuresSum) rel \(rel)") }
        let r = try! asr.transcribe(x)
        ms += r.ms; audio += Double(x.count) / 16000
        if r.tokens == f.tokens { tokOK += 1 } else { print("tokens \(f.audio):\n  python \(f.text)\n  swift  \(r.text)") }
        if r.text == f.text { textOK += 1 }
    }
    print(String(format: "phonon parity: features %d/%d (max head diff %.1e, abs-sum rel %.1e), tokens %d/%d, text %d/%d; %.0f ms for %.0f s of audio (%.0fx real time)",
                 featOK, fx.count, maxHead, maxSumRel, tokOK, fx.count, textOK, fx.count, ms, audio, audio * 1000 / ms))
    exit(0)
}

/// `memnar-cli lines <validator.json> <memeval set.jsonl> <expected.jsonl> <pred.jsonl>...`: the Swift memory lines
/// (MemoryLines.memories) over Python's own graphs must equal eval/validator.py apply's output.
if args.count > 1 && args[1] == "lines" {
    let v = try! LineValidator.load(URL(fileURLWithPath: args[2]))
    func jsonl(_ p: String) -> [[String: Any]] {
        try! String(contentsOfFile: p, encoding: .utf8).split(separator: "\n").map { try! JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
    }
    let windows = jsonl(args[2 + 1]).map { r -> Window in
        let w = (r["window"] as? [String: Any]) ?? r
        return try! makeDecoder().decode(Window.self, from: JSONSerialization.data(withJSONObject: w))
    }
    let expected = Dictionary(uniqueKeysWithValues: jsonl(args[4]).map { ($0["window_id"] as! String, $0["memories"] as! [String]) })
    let preds = args[5...].map { p in Dictionary(uniqueKeysWithValues: jsonl(p).map { ($0["window_id"] as! String, Graph3.fromJSON($0["graph"] as! [String: Any])) }) }
    var ok = 0, lines = 0
    for w in windows {
        let got = MemoryLines.memories(w, preds.compactMap { $0[w.windowId] }, v).map(\.line)
        let want = expected[w.windowId] ?? []
        lines += got.count
        if got == want { ok += 1 } else {
            print("DIFF \(w.windowId): swift-only \(got.filter { !want.contains($0) }) python-only \(want.filter { !got.contains($0) })")
        }
    }
    print("memory-line parity \(ok)/\(windows.count) windows, \(lines) lines")
    exit(0)
}

/// v4 fixtures (parity4.json, export/ondevice3.py on a speakers run): tokens, speaker features and graphs at the app's
/// cutoffs (0.5 / 0.5 / 0.35).
struct Fixture4: Decodable {
    let window: Window
    let inputIds: [Int]
    let uttMask: [Int]
    let wordStart: [Int]
    let wordEnd: [Int]
    let tokUtt: [Int]
    let slotIdx: [Int]
    let userIdx: Int
    let spkNames: [String]
    let spkTok: [[Int]]
    let uttSpk: [Int]
    let graph: Fixture.G
}

/// `memnar-cli <Model> [<Model_b>] [--validator path]`, v4: parity of each model folder that has parity4.json, then the
/// pooled memory lines (MemoryLines over both graphs) of the probes, for reading by hand.
func parity4(_ dirs: [URL], validator: URL) async throws {
    var exs: [Extractor] = []
    for dir in dirs {
        let t0 = Date()
        let ex = try await Extractor.load(from: dir)
        exs.append(ex)
        print(String(format: "%@: model loaded in %.1f s (%@)", dir.lastPathComponent, Date().timeIntervalSince(t0), ex.meta.run))
        let data = try Data(contentsOf: dir.appendingPathComponent("parity4.json"))
        let fixtures = try makeDecoder().decode([Fixture4].self, from: data)
        let raw = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        var okTok = 0, okGraph = 0, times: [Double] = []
        for (fi, fx) in fixtures.enumerated() {
            let w = fx.window
            let (g, f) = try ex.extract(w, entThresh: 0.5, trgThresh: 0.5, roleThresh: 0.35)
            let tm = ex.lastTiming
            times.append(tm.totalMs)
            let b = { (x: [Bool]) in x.map { $0 ? 1 : 0 } }
            var tokWhy: [String] = []
            if f.inputIds != fx.inputIds { tokWhy.append("ids swift \(f.inputIds.count) python \(fx.inputIds.count)") }
            if b(f.uttMask) != fx.uttMask { tokWhy.append("utt_mask") }
            if b(f.wordStart) != fx.wordStart || b(f.wordEnd) != fx.wordEnd { tokWhy.append("word bounds") }
            if f.tokUtt != fx.tokUtt { tokWhy.append("tok_utt") }
            if f.slotIdx != fx.slotIdx || f.userIdx != fx.userIdx { tokWhy.append("slots/user") }
            if f.spkNames != fx.spkNames || f.spkTok != fx.spkTok || f.uttSpk != fx.uttSpk {
                tokWhy.append("speakers swift \(f.spkNames) \(f.spkTok) \(f.uttSpk) python \(fx.spkNames) \(fx.spkTok) \(fx.uttSpk)")
            }
            let rr = ((raw[fi]["graph"] as! [String: Any])["frames"] as! [[String: Any]]).map { $0["roles"] as! [String: [String]] }
            let (gSame, gWhy) = sameGraph(g, fx.graph, rr)
            okTok += tokWhy.isEmpty ? 1 : 0
            okGraph += gSame ? 1 : 0
            let spkFillers = g.frames.flatMap { $0.roles.flatMap(\.1) }.filter { f.spkNames.contains($0) }.count
            print(String(format: "  %@ tokens %@ graph %@ | %d speakers, %d speaker fillers, %d tok, %d ents, %d frames, %.0f ms (encode %.0f, roles %.0f)",
                         w.windowId, tokWhy.isEmpty ? "OK" : "DIFF", gSame ? "OK" : "DIFF", f.spkNames.count + 1, spkFillers,
                         tm.tokens, g.entities.count, g.frames.count, tm.totalMs, tm.encodeMs, tm.rolesMs))
            for why in tokWhy + (gSame ? [] : [gWhy]) { print("     ", why) }
        }
        let n = fixtures.count
        print(String(format: "%@: token parity %d/%d, graph parity %d/%d, mean %.0f ms per window on this Mac",
                     dir.lastPathComponent, okTok, n, okGraph, n, times.reduce(0, +) / Double(max(1, n))))
    }
    // pooled memories of the probes, as the app stores them
    let v = try LineValidator.load(validator)
    let fx0 = try makeDecoder().decode([Fixture4].self, from: Data(contentsOf: dirs[0].appendingPathComponent("parity4.json")))
    for fx in fx0 where fx.window.worldId == "probe" {
        let w = fx.window
        let graphs = try exs.map { try $0.extract(w, entThresh: 0.5, trgThresh: 0.5, roleThresh: 0.35).0 }
        print("\(w.windowId):")
        for (i, u) in w.utterances.enumerated() { print("    [\(i)] \(w.speakerLabels[i]): \(u.text)") }
        for r in MemoryLines.memories(w, graphs, v) { print(String(format: "    kept %.2f  %@", r.score, r.line)) }
    }
}

let modelDir = URL(fileURLWithPath: args.count > 1 && !args[1].hasPrefix("--") ? args[1] : "../Model")
if FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("parity4.json").path) {
    var dirs = [modelDir]
    if args.count > 2 && !args[2].hasPrefix("--") { dirs.append(URL(fileURLWithPath: args[2])) }
    let vpath = args.firstIndex(of: "--validator").map { args[$0 + 1] } ?? "../../model/validator_g2.json"
    let sema4 = DispatchSemaphore(value: 0)
    Task {
        do { try await parity4(dirs, validator: URL(fileURLWithPath: vpath)) } catch { print("error:", error) }
        sema4.signal()
    }
    sema4.wait()
    exit(0)
}
let s2Windows: [String] = args.firstIndex(of: "--s2").map { Array(args[($0 + 1)...]) } ?? []

let sema = DispatchSemaphore(value: 0)
Task {
    do {
        let t0 = Date()
        let ex = try await Extractor.load(from: modelDir)
        let meta = ex.meta
        print(String(format: "model loaded in %.1f s (%@, %@)", Date().timeIntervalSince(t0), meta.run, meta.quant))
        let data = try Data(contentsOf: modelDir.appendingPathComponent("parity3.json"))
        let fixtures = try makeDecoder().decode([Fixture].self, from: data)
        let raw = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        func rawRoles(_ i: Int, _ key: String) -> [[String: [String]]] {
            ((raw[i][key] as! [String: Any])["frames"] as! [[String: Any]]).map { $0["roles"] as! [String: [String]] }
        }
        var okTok = 0, okGraph = 0, okMem = 0, okS1 = 0
        var times: [Double] = []
        for (fi, fx) in fixtures.enumerated() {
            let w = fx.window
            let (g, f) = try ex.extract(w, trgThresh: meta.s2Lo)
            let tm = ex.lastTiming
            times.append(tm.totalMs)
            let b = { (x: [Bool]) in x.map { $0 ? 1 : 0 } }
            var tokWhy: [String] = []
            if f.inputIds != fx.inputIds {
                let first = zip(f.inputIds, fx.inputIds).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                tokWhy.append("ids swift \(f.inputIds.count) python \(fx.inputIds.count), first diff \(first.map(String.init) ?? "-")")
            }
            if b(f.uttMask) != fx.uttMask { tokWhy.append("utt_mask") }
            if b(f.wordStart) != fx.wordStart { tokWhy.append("word_start") }
            if b(f.wordEnd) != fx.wordEnd { tokWhy.append("word_end") }
            if f.tokUtt != fx.tokUtt { tokWhy.append("tok_utt") }
            if f.slotIdx != fx.slotIdx || f.userIdx != fx.userIdx { tokWhy.append("slots/user") }
            let (gSame, gWhy) = sameGraph(g, fx.graph, rawRoles(fi, "graph"))
            let (sure, unsure) = Compile.route(g, w, lo: meta.s2Lo, hi: meta.s2Hi)
            let memWhy = [diff("all", Compile.memories(g, speakers: w.speakerLabels, uttText: w.utterances.map(\.text)), fx.memoriesAll),
                          diff("sure", sure.map(\.line), fx.memoriesSure), diff("unsure", unsure.map(\.line), fx.memoriesUnsure)].compactMap { $0 }
            let (g1, _) = try ex.extract(w)
            let (g1Same, g1Why) = sameGraph(g1, fx.graphS1, rawRoles(fi, "graph_s1"))
            let m1 = Compile.memories(g1, minP: meta.trgThresh, speakers: w.speakerLabels, uttText: w.utterances.map(\.text))
            let s1Why = [g1Same ? nil : "graph_s1: \(g1Why)", diff("s1", m1, fx.memoriesS1)].compactMap { $0 }
            okTok += tokWhy.isEmpty ? 1 : 0
            okGraph += gSame ? 1 : 0
            okMem += memWhy.isEmpty ? 1 : 0
            okS1 += s1Why.isEmpty ? 1 : 0
            print(String(format: "%@ tokens %@ graph %@ memories %@ s1 %@ | %d tok, %d cands, %d ents, %d frames, %.0f ms (encode %.0f, score %.0f, link %.0f, roles %.0f, attrs %.0f)",
                         w.windowId, tokWhy.isEmpty ? "OK" : "DIFF", gSame ? "OK" : "DIFF", memWhy.isEmpty ? "OK" : "DIFF", s1Why.isEmpty ? "OK" : "DIFF",
                         tm.tokens, tm.candidates, g.entities.count, g.frames.count, tm.totalMs, tm.encodeMs, tm.scoreMs, tm.linkMs, tm.rolesMs, tm.attrsMs))
            for why in tokWhy + (gSame ? [] : [gWhy]) + memWhy + s1Why { print("   ", why) }
            if w.worldId == "probe" {
                for (i, u) in w.utterances.enumerated() { print("    [\(i)] \(w.speakerLabels[i]): \(u.text)") }
                for m in m1 { print("    S1:", m) }
                for m in unsure.map(\.line) { print("    unsure:", m) }
            }
        }
        let n = fixtures.count
        print("token parity \(okTok)/\(n), graph parity \(okGraph)/\(n), memory parity \(okMem)/\(n), System 1 parity \(okS1)/\(n)")
        print(String(format: "mean %.0f ms per window on this Mac", times.reduce(0, +) / Double(max(1, n))))

        // System 2 on the uncertain memories of the named windows (key from the environment, never printed)
        if !s2Windows.isEmpty, let key = ProcessInfo.processInfo.environment["BASETEN_KEY"], !key.isEmpty {
            let client = LLMClient(apiKey: key)
            let store = MemoryStore(lo: meta.s2Lo, hi: meta.s2Hi)
            for fx in fixtures where s2Windows.contains(fx.window.windowId) {
                let (g, _) = try ex.extract(fx.window, trgThresh: meta.s2Lo)
                let (c, pending) = store.ingest(fx.window, g)
                print("\(fx.window.windowId): stored \(c.stored), to System 2 \(c.pending), on device \(c.onDevice), dropped \(c.dropped)")
                let r = await System2.verify(window: fx.window, ids: pending, store: store, client: client)
                print(String(format: "  System 2: %d items in %.1f s: confirm %d fix %d reject %d no-verdict %d", r.items, r.ms / 1000, r.confirm, r.fix, r.reject, r.error))
                for m in store.memories where m.windowId == fx.window.windowId {
                    print("  [\(m.status.rawValue)] \(m.text)\(m.text != m.original ? "   (was: \(m.original))" : "")")
                }
            }
            print(String(format: "Baseten spend this run: $%.4f", client.usd))
        }
    } catch {
        print("error:", error)
    }
    sema.signal()
}
sema.wait()
