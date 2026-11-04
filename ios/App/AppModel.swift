import Foundation
import MemNARCore
import SwiftUI

struct World: Decodable, Identifiable {
    let worldId: String
    let windows: [Window]
    var id: String { worldId }
}

/// One System 1 model on its own actor, so extraction never blocks recording or the UI.
actor S1Worker {
    let ex: Extractor
    init(_ ex: Extractor) { self.ex = ex }
    /// The cutoffs the line validator was trained with (decision 37): entities 0.5, triggers 0.5, roles 0.35.
    func graph(_ w: Window) throws -> (Graph3, Extractor.Timing) {
        let (g, _) = try ex.extract(w, entThresh: 0.5, trgThresh: 0.5, roleThresh: 0.35)
        return (g, ex.lastTiming)
    }
    var run: String { ex.meta.run }
}

/// One processed window, for the device page.
struct WindowRun: Identifiable {
    let id = UUID()
    let windowId: String
    let lines: Int
    let kept: Int
    let ms: Double          // all models, sequentially
    let tokens: Int
}

enum Voice: String, CaseIterable, Identifiable {
    case me, other
    var id: String { rawValue }
    var label: String { self == .me ? "Me (the wearer)" : "Someone else" }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var status = "Loading System 1…"
    @Published var ready = false
    @Published var loadMs = 0.0
    @Published var lastError: String?
    @Published var version = 0                // bumps whenever the bank changes
    @Published var live: Session?             // the session being recorded
    @Published var queued = 0                 // windows waiting for System 1
    @Published var processing = false
    @Published var runs: [WindowRun] = []
    @Published var worlds: [World] = []
    @Published var models: [String] = []
    /// Phonon-2.5 English (speech to text on the Neural Engine); nil until loaded, or when the app ships without it.
    @Published var asr: PhononASR?
    @Published var asrLoadMs = 0.0
    @Published var asrPending = Bundle.main.url(forResource: "Phonon", withExtension: nil) != nil

    // settings
    @AppStorage("voice") var voice: Voice = .me
    @AppStorage("strictness") var strictness: Double = -1   // validator cutoff; -1 = the shipped one
    @AppStorage("windowChars") var windowChars: Int = 2000  // transcript characters per System 1 window

    let bank: MemoryBank
    private(set) var validator: LineValidator?
    private var workers: [S1Worker] = []
    private var queue: [Window] = []
    private var cutFrom = 0                   // first utterance of the live session not yet sent to System 1
    let client: LLMClient?

    var cutoff: Double { strictness >= 0 ? strictness : (validator?.threshold ?? 0.6) }

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        bank = MemoryBank(url: docs.appendingPathComponent("memnar_bank.json"))
        if let url = Bundle.main.url(forResource: "Secrets", withExtension: "plist"),
           let d = NSDictionary(contentsOf: url), let k = d["BasetenKey"] as? String, !k.isEmpty {
            client = LLMClient(apiKey: k)
        } else {
            client = nil
        }
        // `-tested`: put Memory strictness back on the validator's tested cutoff (the scored system)
        if ProcessInfo.processInfo.arguments.contains("-tested") { UserDefaults.standard.removeObject(forKey: "strictness") }
        Task { await load() }
    }

    /// Model/ holds System 1 (the v4 system ships a second model in Model_b/); validator_g2.json is the line validator.
    private func load() async {
        if let pdir = Bundle.main.url(forResource: "Phonon", withExtension: nil) {
            // loads next to System 1; the first launch compiles it for the Neural Engine (cached afterwards)
            Task.detached(priority: .userInitiated) {
                let t = Date()
                do {
                    let a = try PhononASR(dir: pdir)
                    let ms = Date().timeIntervalSince(t) * 1000
                    await MainActor.run {
                        self.asr = a; self.asrLoadMs = ms; self.asrPending = false
                        print(String(format: "[load] Phonon-2.5 English after %.1f s, footprint %.0f MB", ms / 1000, self.footprintMB))
                    }
                } catch {
                    await MainActor.run { self.asrPending = false; print("[load] Phonon failed: \(error)") }
                }
            }
        }
        do {
            let t0 = Date()
            guard let vurl = Bundle.main.url(forResource: "validator_g2", withExtension: "json") else { throw AppError("validator_g2.json missing from the app") }
            validator = try LineValidator.load(vurl)
            for name in ["Model", "Model_b"] {
                guard let dir = Bundle.main.url(forResource: name, withExtension: nil),
                      FileManager.default.fileExists(atPath: dir.appendingPathComponent("memnar.pte").path) else { continue }
                let ex = try await Extractor.load(from: dir)
                workers.append(S1Worker(ex))
                models.append(ex.meta.run)
                print(String(format: "[load] %@ %@ after %.1f s, footprint %.0f MB", ex.meta.run, name, Date().timeIntervalSince(t0), footprintMB))
                if worlds.isEmpty, let data = try? Data(contentsOf: dir.appendingPathComponent("worlds.json")) {
                    worlds = (try? makeDecoder().decode([World].self, from: data)) ?? []
                }
            }
            guard !workers.isEmpty else { throw AppError("no System 1 model in the app (Model/memnar.pte)") }
            loadMs = Date().timeIntervalSince(t0) * 1000
            status = "System 1 ready"
            ready = true
            print("[bank] \(bank.sessions.count) sessions, \(bank.records.count) memories on launch")
            if let i = ProcessInfo.processInfo.arguments.firstIndex(of: "-ask"), let q = ProcessInfo.processInfo.arguments[safe: i + 1] {
                for (k, h) in search(q).prefix(3).enumerated() { print(String(format: "[ask] %d. %.2f %@", k + 1, h.score, h.record.line)) }
            }
            if let i = ProcessInfo.processInfo.arguments.firstIndex(of: "-autotest") {
                await autotest(Int(ProcessInfo.processInfo.arguments[safe: i + 1] ?? "") ?? 3)
            }
        } catch {
            status = "Failed to load: \(error)"
        }
    }

    // MARK: recording

    func begin() -> Session {
        let df = DateFormatter()
        df.dateFormat = "EEE d MMM, HH:mm"
        let s = Session(title: "Recording · \(df.string(from: Date()))")
        live = s
        cutFrom = 0
        bank.upsert(s)
        bank.save()
        version += 1
        return s
    }

    /// A finished line from the recorder.
    func add(line text: String, at: Date, conf: Double) {
        guard var s = live else { return }
        let tf = DateFormatter()
        tf.dateFormat = "HH:mm:ss"
        s.utterances.append(Utterance(spk: voice == .me ? "S1" : "S2", t: tf.string(from: at), text: text,
                                      asrConf: (conf * 100).rounded() / 100))
        live = s
        bank.upsert(s)
        let pending = s.utterances[cutFrom...]
        if pending.map(\.text.count).reduce(0, +) >= windowChars || pending.count >= 40 { cut() }
        version += 1
    }

    /// Recording stopped: the rest of the transcript becomes the last window.
    func end() {
        guard var s = live else { return }
        cut()
        s = live ?? s
        s.ended = Date()
        bank.upsert(s)
        bank.save()
        live = nil
        version += 1
    }

    private func cut() {
        guard var s = live, cutFrom < s.utterances.count else { return }
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let w = Window(windowId: "\(s.id)_w\(s.windowIds.count)", worldId: s.id, day: 1, date: df.string(from: s.started),
                       speakers: ["S1": USER, "S2": "UNKNOWN"], utterances: Array(s.utterances[cutFrom...]),
                       state: bank.recentState(), scene: s.scene)
        cutFrom = s.utterances.count
        s.windowIds.append(w.windowId)
        live = s
        bank.upsert(s)
        enqueue(w, session: s.id)
    }

    // MARK: System 1

    private var sessionOf: [String: String] = [:]

    func enqueue(_ w: Window, session: String) {
        sessionOf[w.windowId] = session
        queue.append(w)
        queued = queue.count
        // set before the task starts, so windows queued in the same turn share one drain loop
        if !processing { processing = true; Task { await drain() } }
    }

    private func drain() async {
        while !queue.isEmpty {
            let w = queue.removeFirst()
            queued = queue.count
            do {
                var graphs: [Graph3] = [], ms = 0.0, tokens = 0
                for wk in workers {
                    let (g, t) = try await wk.graph(w)
                    graphs.append(g)
                    ms += t.totalMs
                    tokens = t.tokens
                }
                guard let v = validator else { continue }
                let recs = MemoryLines.memories(w, graphs, v, sessionId: sessionOf[w.windowId] ?? "", threshold: cutoff)
                bank.add(recs)
                bank.save()
                runs.insert(WindowRun(windowId: w.windowId, lines: w.utterances.count, kept: recs.count, ms: ms, tokens: tokens), at: 0)
                version += 1
            } catch {
                lastError = "System 1 failed on \(w.windowId): \(error)"
            }
        }
        processing = false
    }

    /// The bundled demo conversations, as one session each.
    func loadDemo() {
        for world in worlds {
            var s = Session(title: "Demo · \(world.worldId)", scene: "demo")
            for w in world.windows {
                s.utterances += w.utterances
                s.windowIds.append(w.windowId)
            }
            s.ended = Date()
            bank.upsert(s)
            for w in world.windows { enqueue(w, session: s.id) }
        }
        bank.save()
        version += 1
    }

    // MARK: memory

    func records(session id: String) -> [MemoryRecord] { bank.records.filter { $0.sessionId == id } }

    func delete(session id: String) { bank.delete(session: id); bank.save(); version += 1 }
    func delete(record id: String) { bank.delete(record: id); bank.save(); version += 1 }
    func wipe() { bank.wipe(); bank.save(); runs = []; version += 1 }

    func rename(session id: String, to title: String) {
        guard var s = bank.session(id) else { return }
        s.title = title
        bank.upsert(s)
        bank.save()
        version += 1
    }

    func search(_ q: String) -> [LocalSearch.Hit] { LocalSearch.search(bank.records, q) }

    func write(_ q: String, _ hits: [MemoryRecord]) async throws -> (String, [String], String) {
        guard let client else { throw AppError("No model key bundled for written answers") }
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        return try await RecordReader.answer(hits, q, now: df.string(from: Date()), client: client)
    }

    var exportURL: URL? { bank.url }

    // MARK: device

    /// `-autotest N`: process the first N demo windows; timings and kept memories go to the console.
    private func autotest(_ n: Int) async {
        print(String(format: "[autotest] %d model(s) (%@) loaded in %.1f s, footprint %.0f MB, validator cutoff %.2f%@", workers.count,
                     models.joined(separator: " + "), loadMs / 1000, footprintMB, cutoff, strictness >= 0 ? " (set by hand)" : ""))
        bank.delete(session: "autotest")   // leftovers of an interrupted run would hide this run's records (same ids)
        let ws = worlds.flatMap(\.windows).prefix(n)
        for w in ws { enqueue(w, session: "autotest") }
        while processing || !queue.isEmpty { try? await Task.sleep(nanoseconds: 200_000_000) }
        for r in runs.reversed() {
            print(String(format: "[autotest] %@ %d lines, %d tok: %.0f ms, kept %d", r.windowId, r.lines, r.tokens, r.ms, r.kept))
        }
        for m in bank.records where m.sessionId == "autotest" { print("[autotest]   \(m.line)") }
        bank.delete(session: "autotest")
        bank.save()
        print(String(format: "[autotest] done, footprint %.0f MB", footprintMB))
    }

    var footprintMB: Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }
}

struct AppError: LocalizedError {
    let msg: String
    init(_ m: String) { msg = m }
    var errorDescription: String? { msg }
}

enum TypeColor {
    static func color(_ t: String) -> Color {
        switch t {
        case "PERSON", "PET", "GROUP", "ORG", "ROLE": return .blue
        case "PLACE", "VENUE", "ADDRESS", "SPOT": return .green
        case "DATETIME", "DURATION", "RECURRENCE": return .orange
        case "EVENT", "APPOINTMENT", "TASK", "TRIP", "ACTIVITY", "PROJECT", "COURSE", "GOAL", "MILESTONE": return .purple
        case "HEALTH", "MEDICATION", "DIET": return .red
        case "MONEY", "QUANTITY": return .teal
        default: return .gray
        }
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
