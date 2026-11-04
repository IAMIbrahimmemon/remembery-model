import MemNARCore
import SwiftUI

@main
struct MemNARApp: App {
    @StateObject var model = AppModel()
    var body: some Scene {
        WindowGroup { RootView().environmentObject(model).tint(Theme.ink) }
    }
}

struct RootView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        if m.ready {
            TabView {
                RecordView().tabItem { Label("Home", systemImage: "house") }
                MemoriesView().tabItem { Label("Memories", systemImage: "brain.head.profile") }
                AskView().tabItem { Label("Ask", systemImage: "bubble.left") }
                SessionsView().tabItem { Label("Sessions", systemImage: "doc.text") }
                DeviceView().tabItem { Label("Settings", systemImage: "gearshape") }
            }
            .alert("Error", isPresented: Binding(get: { m.lastError != nil }, set: { if !$0 { m.lastError = nil } })) {
                Button("OK", role: .cancel) {}
            } message: { Text(m.lastError ?? "") }
        } else {
            VStack(spacing: 14) {
                Text("MEM-NAR").font(Theme.serif(34)).foregroundStyle(Theme.ink)
                ProgressView()
                Text(m.status).font(.callout).foregroundStyle(Theme.muted).multilineTextAlignment(.center).padding(.horizontal)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.canvas.ignoresSafeArea())
        }
    }
}

// MARK: - Home (record)

struct RecordView: View {
    @EnvironmentObject var m: AppModel
    @StateObject var rec = LiveRecorder()
    @State var pane = 0

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                Group {
                    if rec.recording || m.live != nil { live } else { home }
                }
                recordBar.padding(.bottom, 10)
            }
            .background(Theme.canvas.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .onAppear {
                rec.authorize()
                rec.onLine = { text, at, conf in m.add(line: text, at: at, conf: conf) }
                rec.onStop = { m.end() }
                if let a = m.asr { rec.use(a) }
            }
            .onChange(of: m.asr != nil) { _, loaded in if loaded, let a = m.asr { rec.use(a) } }
            .task { await scriptedRecording() }
        }
    }

    // idle: recent sessions grouped by day
    var home: some View {
        let _ = m.version
        let sessions = Array(m.bank.sessions.reversed())
        let cal = Calendar.current
        let days = Dictionary(grouping: sessions) { cal.startOfDay(for: $0.started) }.sorted { $0.key > $1.key }
        return ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageTitle(title: "Today", caption: Date().formatted(.dateTime.weekday(.wide).month(.wide).day()))
                if let n = rec.note { Text(n).font(.footnote).foregroundStyle(Theme.stop).card() }
                HStack(spacing: 10) {
                    stat("\(m.bank.records.count)", "memories")
                    stat("\(m.bank.sessions.count)", "sessions")
                    stat(m.models.count == 2 ? "2" : "\(m.models.count)", m.models.count == 1 ? "model" : "models")
                }
                if sessions.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Nothing recorded yet").font(Theme.serif(20)).foregroundStyle(Theme.ink)
                        Text("Tap Record and talk. Everything is transcribed and remembered on this phone; memories appear while you talk.")
                            .font(.subheadline).foregroundStyle(Theme.muted)
                    }.card(padding: 18)
                }
                ForEach(days, id: \.key) { day, ss in
                    VStack(alignment: .leading, spacing: 8) {
                        GroupLabel(text: cal.isDateInToday(day) ? "Today" : cal.isDateInYesterday(day) ? "Yesterday"
                                   : day.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day()))
                        ForEach(ss) { s in NavigationLink { SessionDetail(id: s.id) } label: { SessionRow(s: s) }.buttonStyle(.plain) }
                    }
                }
            }
            .padding(.horizontal, 18).padding(.top, 12).padding(.bottom, 110)
        }
    }

    // recording (or a session waiting for its last window)
    var live: some View {
        let _ = m.version
        return VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    PageTitle(title: m.live?.title.replacingOccurrences(of: "Recording · ", with: "") ?? "Recording",
                              caption: rec.recording ? "Listening" : "Finishing")
                    LevelBars(level: rec.level, active: rec.recording)
                }
                HStack(spacing: 10) {
                    stat("\(m.live?.utterances.count ?? 0)", "lines")
                    stat("\(m.live.map { m.records(session: $0.id).count } ?? 0)", "memories")
                    stat(m.processing ? "…" : "\(m.queued)", m.processing ? "thinking" : "queued")
                }
                HStack(spacing: 8) {
                    Pill(text: "Transcript", selected: pane == 0).onTapGesture { pane = 0 }
                    Pill(text: "Memories", selected: pane == 1).onTapGesture { pane = 1 }
                    Spacer()
                    Image(systemName: rec.onDevice ? "lock.shield" : "network").foregroundStyle(Theme.faint)
                        .accessibilityLabel(rec.onDevice ? "Speech stays on this phone" : "Speech recognition may use Apple's servers")
                }
                if let n = rec.note { Text(n).font(.footnote).foregroundStyle(Theme.stop) }
            }
            .padding(.horizontal, 18).padding(.top, 12)
            if pane == 0 { LiveTranscript(rec: rec) } else { LiveMemories() }
        }
    }

    var recordBar: some View {
        Button { toggle() } label: {
            HStack(spacing: 10) {
                if rec.recording, let t0 = rec.started {
                    RoundedRectangle(cornerRadius: 3).fill(Theme.stop).frame(width: 14, height: 14)
                    TimelineView(.periodic(from: t0, by: 1)) { ctx in
                        Text(elapsed(ctx.date.timeIntervalSince(t0))).font(.headline.monospacedDigit())
                    }
                    Text("Stop").font(.headline)
                } else {
                    Image(systemName: "mic.fill").font(.headline)
                    Text(m.asrPending ? "Preparing speech model…" : rec.authorized ? "Record" : "Allow the microphone in Settings").font(.headline)
                }
            }
            .foregroundStyle(rec.recording ? Theme.ink : Theme.canvas)
            .padding(.horizontal, 26).frame(height: 54)
            .background(rec.recording ? Theme.surface : Theme.ink, in: Capsule())
            .overlay(Capsule().strokeBorder(rec.recording ? Theme.hairline : .clear))
            .shadow(color: .black.opacity(0.12), radius: 14, y: 5)
        }
        .buttonStyle(.plain)
        .disabled(!rec.authorized || m.asrPending)   // the first launch compiles Phonon for the Neural Engine
        .accessibilityLabel(rec.recording ? "Stop recording" : "Start recording")
    }

    func toggle() {
        Task {
            if rec.recording {
                await rec.stop()
                m.end()
            } else {
                _ = m.begin()
                do { try rec.start() } catch { m.end(); m.lastError = "Microphone unavailable: \(error.localizedDescription)" }
            }
        }
    }

    /// `-record N` launch argument (testing from the Mac): start recording once the model and permissions are
    /// ready, log progress every 30 s, stop after N seconds and log the session's lines and memories.
    func scriptedRecording() async {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-record"), let secs = Double(args[safe: i + 1] ?? "") else { return }
        while !(m.ready && rec.authorized && !m.asrPending) { try? await Task.sleep(nanoseconds: 300_000_000) }
        if let a = m.asr { rec.use(a) }
        _ = m.begin()
        do { try rec.start() } catch { print("[record] could not start: \(error)"); m.end(); return }
        print("[record] started, speech to text: \(rec.asr != nil ? "Phonon-2.5 English" : "Apple"), on device: \(rec.onDevice)")
        let t0 = Date()
        var next = 30.0
        while rec.recording && Date().timeIntervalSince(t0) < secs {
            try? await Task.sleep(nanoseconds: 500_000_000)
            let t = Date().timeIntervalSince(t0)
            if t >= next {
                next += 30
                let s = m.live
                let lv = rec.levelStats().map { String(format: ", input floor %.4f median %.4f p90 %.4f", $0.floor, $0.median, $0.p90) } ?? ""
                print(String(format: "[record] %3.0f s: %d lines, %d windows cut, %d processed, %d memories, %@, app state %d%@",
                             t, s?.utterances.count ?? 0, s?.windowIds.count ?? 0, m.runs.count,
                             s.map { m.records(session: $0.id).count } ?? 0, m.processing ? "thinking" : "idle",
                             UIApplication.shared.applicationState.rawValue, lv))
            }
        }
        let sid = m.live?.id
        if rec.recording { await rec.stop(); m.end() } else { print("[record] stopped early: \(rec.note ?? "")") }
        while m.processing { try? await Task.sleep(nanoseconds: 200_000_000) }
        guard let sid, let s = m.bank.session(sid) else { return }
        print(String(format: "[record] done after %.0f s: %d lines, %d windows, %d memories", Date().timeIntervalSince(t0),
                     s.utterances.count, s.windowIds.count, m.records(session: sid).count))
        for u in s.utterances { print("[record]   \(u.t) \(u.text)") }
        for r in m.records(session: sid) { print("[record]   memory: \(r.line)") }
    }

    func stat(_ v: String, _ k: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(v).font(Theme.serif(24)).foregroundStyle(Theme.ink).monospacedDigit()
            Text(k).font(.caption).foregroundStyle(Theme.muted)
        }
        .card(padding: 12)
    }

    func elapsed(_ s: TimeInterval) -> String {
        let t = Int(s)
        return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60) : String(format: "%02d:%02d", t / 60, t % 60)
    }
}

/// A session in a list: document tile, title, counts, start time.
struct SessionRow: View {
    @EnvironmentObject var m: AppModel
    let s: Session
    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.sunken)
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.hairline))
                .frame(width: 42, height: 42)
                .overlay(Image(systemName: m.live?.id == s.id ? "waveform" : "doc.text").foregroundStyle(m.live?.id == s.id ? Theme.live : Theme.muted))
            VStack(alignment: .leading, spacing: 3) {
                Text(s.title).font(.body).foregroundStyle(Theme.ink).lineLimit(1)
                Text("\(m.records(session: s.id).count) memories · \(s.utterances.count) lines").font(.subheadline).foregroundStyle(Theme.muted)
            }
            Spacer(minLength: 4)
            Text(s.started.formatted(date: .omitted, time: .shortened)).font(.subheadline.monospacedDigit()).foregroundStyle(Theme.muted)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct LiveTranscript: View {
    @EnvironmentObject var m: AppModel
    @ObservedObject var rec: LiveRecorder

    var body: some View {
        let _ = m.version
        let lines = m.live?.utterances ?? []
        let kept = Set(m.live.map { m.records(session: $0.id).map(\.text) } ?? [])
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    if lines.isEmpty && rec.partial.isEmpty {
                        Text(rec.recording ? "Listening…" : "The transcript appears here as you talk.").foregroundStyle(Theme.muted)
                    }
                    ForEach(Array(lines.enumerated()), id: \.offset) { i, u in
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(String(u.t.prefix(5))).font(.caption.monospacedDigit()).foregroundStyle(Theme.faint).frame(width: 40, alignment: .leading)
                            Text(u.text).font(.body).foregroundStyle(kept.contains(u.text) ? Theme.ink : Theme.muted)
                            if kept.contains(u.text) { Spacer(minLength: 0); Image(systemName: "brain").font(.caption).foregroundStyle(Theme.live) }
                        }.id(i)
                    }
                    if !rec.partial.isEmpty {
                        Text(rec.partial).font(.body).foregroundStyle(Theme.faint).padding(.leading, 50).id("partial")
                    }
                }
                .padding(.horizontal, 18).padding(.bottom, 110)
            }
            .onChange(of: lines.count) { _, _ in withAnimation { proxy.scrollTo(lines.count - 1, anchor: .bottom) } }
            .onChange(of: rec.partial) { _, _ in proxy.scrollTo("partial", anchor: .bottom) }
        }
    }
}

struct LiveMemories: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        let _ = m.version
        let id = m.live?.id ?? m.bank.sessions.last?.id
        let recs = id.map { m.records(session: $0) } ?? []
        ScrollView {
            LazyVStack(spacing: 10) {
                if recs.isEmpty {
                    Text(m.processing ? "Reading the conversation…" : "Memories appear here every couple of minutes while you record, and when you stop.")
                        .font(.subheadline).foregroundStyle(Theme.muted).card()
                }
                ForEach(recs.reversed()) { r in MemoryCard(r: r) }
            }
            .padding(.horizontal, 18).padding(.bottom, 110)
        }
    }
}

// MARK: - Memory card and detail

struct MemoryCard: View {
    @EnvironmentObject var m: AppModel
    let r: MemoryRecord
    var matched: Set<String> = []
    @State var open = false

    var body: some View {
        Button { open = true } label: {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 6) {
                    Text(r.speaker == "ME" ? "You" : r.speaker).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.ink)
                    Text("· \(r.date) \(String(r.time.prefix(5)))").font(.subheadline).foregroundStyle(Theme.muted)
                    Spacer()
                    if r.sensitive { Image(systemName: "lock.fill").font(.caption).foregroundStyle(Theme.muted) }
                }
                Text(highlight(r.text)).font(.body).foregroundStyle(Theme.ink).lineLimit(6).multilineTextAlignment(.leading)
                if let c = r.context, MemoryCard.shortOrPronoun(r.text) {
                    Text("after \(c.speaker == "ME" ? "you" : c.speaker): “\(c.text)”").font(.footnote).foregroundStyle(Theme.muted).lineLimit(2)
                }
                Chips(frames: Array(Set(r.frames.map(\.frame))).sorted(), entities: r.entities)
            }
            .card()
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $open) { MemoryDetail(r: r).environmentObject(m) }
    }

    static func shortOrPronoun(_ s: String) -> Bool {
        s.split(separator: " ").count < 8 || ["it", "that", "this", "they", "them", "he", "she", "yes", "yeah", "no", "so", "and", "but", "oh"]
            .contains(s.lowercased().split(whereSeparator: { !$0.isLetter }).first.map(String.init) ?? "")
    }

    func highlight(_ s: String) -> AttributedString {
        var a = AttributedString(s)
        for e in r.entities {
            if let rg = a.range(of: e.text) { a[rg].underlineStyle = .single; a[rg].underlineColor = UIColor(TypeColor.color(e.coarse).opacity(0.6)) }
        }
        for t in matched {
            var from = a.startIndex
            while let rg = a[from...].range(of: t, options: .caseInsensitive) { a[rg].backgroundColor = Theme.mark; from = rg.upperBound }
        }
        return a
    }
}

struct Chips: View {
    let frames: [String]
    let entities: [RecordEntity]
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 5) {
                ForEach(frames, id: \.self) { f in
                    Text(f.replacingOccurrences(of: "_", with: " ")).font(.caption)
                        .foregroundStyle(Theme.muted).padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Theme.sunken, in: Capsule())
                }
                ForEach(entities.prefix(8), id: \.self) { e in
                    HStack(spacing: 4) {
                        Circle().fill(TypeColor.color(e.coarse)).frame(width: 5, height: 5)
                        Text(e.type.lowercased()).font(.caption)
                    }
                    .foregroundStyle(Theme.muted).padding(.horizontal, 8).padding(.vertical, 3)
                    .overlay(Capsule().strokeBorder(Theme.hairline))
                }
            }
        }
    }
}

struct MemoryDetail: View {
    @EnvironmentObject var m: AppModel
    @Environment(\.dismiss) var dismiss
    let r: MemoryRecord

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(r.speaker == "ME" ? "You" : r.speaker) · \(r.date) \(String(r.time.prefix(5)))").font(.subheadline).foregroundStyle(Theme.muted)
                        Text("“\(r.text)”").font(Theme.serif(24)).foregroundStyle(Theme.ink)
                        if let c = r.context {
                            Text("Before it, \(c.speaker == "ME" ? "you" : c.speaker): “\(c.text)”").font(.callout).foregroundStyle(Theme.muted)
                        }
                        if let s = m.bank.session(r.sessionId) { Text(s.title).font(.footnote).foregroundStyle(Theme.faint) }
                        if r.sensitive { Label("Sensitive: kept on this phone, never sent anywhere", systemImage: "lock.fill").font(.footnote).foregroundStyle(Theme.muted) }
                    }
                    GroupLabel(text: "What System 1 understood")
                    if r.frames.isEmpty { Text("No frame; kept for its time, place or number.").foregroundStyle(Theme.muted).card() }
                    ForEach(Array(r.frames.enumerated()), id: \.offset) { _, f in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(f.frame.replacingOccurrences(of: "_", with: " ")).font(.headline).foregroundStyle(Theme.ink)
                                Text("“\(f.trigger)”").font(.subheadline).foregroundStyle(Theme.muted)
                                Spacer()
                                Text(String(format: "%.2f", f.p)).font(.caption.monospacedDigit()).foregroundStyle(Theme.faint)
                            }
                            ForEach(f.roles, id: \.self) { rr in
                                HStack(alignment: .top) {
                                    Text(rr.role.replacingOccurrences(of: "_", with: " ")).font(.subheadline).foregroundStyle(Theme.muted).frame(width: 120, alignment: .leading)
                                    Text(rr.fillers.map { $0 == "ME" ? "you" : $0 }.joined(separator: ", ")).font(.subheadline).foregroundStyle(Theme.ink)
                                }
                            }
                            if f.modality != "asserted" || f.tense != "present" {
                                Text([f.modality == "asserted" ? nil : f.modality, f.tense == "present" ? nil : f.tense].compactMap { $0 }.joined(separator: " · "))
                                    .font(.caption).foregroundStyle(Theme.muted)
                            }
                        }
                        .card()
                    }
                    if !r.entities.isEmpty {
                        GroupLabel(text: "Mentioned")
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(r.entities, id: \.self) { e in
                                HStack {
                                    Circle().fill(TypeColor.color(e.coarse)).frame(width: 7, height: 7)
                                    Text(e.text).foregroundStyle(Theme.ink)
                                    Spacer()
                                    Text(e.type.lowercased()).font(.subheadline).foregroundStyle(Theme.muted)
                                }
                            }
                        }.card()
                    }
                    HStack {
                        Text("Validator score \(String(format: "%.2f", r.score))").font(.footnote).foregroundStyle(Theme.faint)
                        Spacer()
                        Button("Forget", role: .destructive) { m.delete(record: r.id); dismiss() }.font(.subheadline)
                    }
                }
                .padding(18)
            }
            .background(Theme.canvas.ignoresSafeArea())
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}

// MARK: - Memories

struct MemoriesView: View {
    @EnvironmentObject var m: AppModel
    @State var q = ""
    @State var person: String?
    @State var kind: String?

    var body: some View {
        NavigationStack {
            let _ = m.version
            let all = m.bank.records
            let people = m.bank.topEntities(["PERSON", "PET", "ORG"], limit: 15)
            let kinds = Dictionary(grouping: all.flatMap { Set($0.frames.map(\.frame)) }, by: { $0 }).mapValues(\.count)
                .sorted { $0.value > $1.value }.prefix(12).map(\.key)
            let shown = all.filter { r in
                (person == nil || r.entities.contains { $0.text == person }) && (kind == nil || r.frames.contains { $0.frame == kind })
                    && (q.isEmpty || r.text.localizedCaseInsensitiveContains(q) || r.line.localizedCaseInsensitiveContains(q))
            }
            List {
                PageTitle(title: "Memories", caption: "\(shown.count) of \(all.count)").cardRow()
                SearchField(text: $q, prompt: "Filter memories").cardRow()
                if !people.isEmpty { filterRow(people.map(\.0), $person).cardRow() }
                if !kinds.isEmpty { filterRow(kinds, $kind).cardRow() }
                if all.isEmpty { Text("Nothing remembered yet. Record a conversation.").foregroundStyle(Theme.muted).card().cardRow() }
                ForEach(shown.reversed().prefix(400)) { r in MemoryCard(r: r).cardRow() }
                    .onDelete { idx in for i in idx { m.delete(record: Array(shown.reversed().prefix(400))[i].id) } }
            }
            .listStyle(.plain)
            .paper()
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    func filterRow(_ items: [String], _ sel: Binding<String?>) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(items, id: \.self) { p in
                    Pill(text: p.replacingOccurrences(of: "_", with: " "), selected: sel.wrappedValue == p)
                        .onTapGesture { sel.wrappedValue = sel.wrappedValue == p ? nil : p }
                }
            }
        }
    }
}

struct SearchField: View {
    @Binding var text: String
    let prompt: String
    var onSubmit: () -> Void = {}
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(Theme.faint)
            TextField(prompt, text: $text).submitLabel(.search).onSubmit(onSubmit)
            if !text.isEmpty { Button { text = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.faint) } }
        }
        .padding(.horizontal, 14).frame(height: 44)
        .background(Theme.surface, in: Capsule())
        .overlay(Capsule().strokeBorder(Theme.hairline))
    }
}

// MARK: - Ask

struct AskView: View {
    @EnvironmentObject var m: AppModel
    @State var q = ""
    @State var asked = ""
    @State var hits: [LocalSearch.Hit] = []
    @State var written: (String, String)?
    @State var writing = false

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        PageTitle(title: asked.isEmpty ? "Ask anything" : asked, caption: asked.isEmpty ? "Searched on this phone. Nothing leaves it." : "You asked")
                        if asked.isEmpty {
                            ForEach(["Where did I put my keys?", "When is the dentist?", "What did Ana say about the move?"], id: \.self) { s in
                                Button { q = s; go() } label: {
                                    HStack { Text(s).foregroundStyle(Theme.ink); Spacer(); Image(systemName: "arrow.up.right").foregroundStyle(Theme.faint) }
                                        .card()
                                }.buttonStyle(.plain)
                            }
                        } else if hits.isEmpty {
                            Text("Nothing in memory matches that.").foregroundStyle(Theme.muted).card()
                        } else {
                            GroupLabel(text: "Best match")
                            MemoryCard(r: hits[0].record, matched: hits[0].matched)
                            if hits.count > 1 {
                                GroupLabel(text: "Also related").padding(.top, 6)
                                ForEach(hits.dropFirst()) { h in MemoryCard(r: h.record, matched: h.matched) }
                            }
                            if m.client != nil {
                                if let w = written {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(w.0).foregroundStyle(Theme.ink)
                                        Text("\(w.1) confidence · Muse read only the matching memories above (sensitive ones withheld)").font(.caption).foregroundStyle(Theme.muted)
                                    }.card()
                                } else {
                                    Button { write() } label: { Pill(text: writing ? "Writing…" : "Write an answer with Muse (online)", icon: "text.bubble") }
                                        .buttonStyle(.plain).disabled(writing)
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 18).padding(.top, 12).padding(.bottom, 100)
                }
                HStack(spacing: 8) {
                    TextField("Ask anything", text: $q).submitLabel(.search).onSubmit { go() }
                    Button { go() } label: {
                        Image(systemName: "arrow.up").font(.subheadline.weight(.bold)).foregroundStyle(Theme.canvas)
                            .frame(width: 34, height: 34).background(q.isEmpty ? Theme.faint : Theme.ink, in: Circle())
                    }.disabled(q.isEmpty)
                }
                .padding(.leading, 18).padding(.trailing, 8).frame(height: 52)
                .background(Theme.surface, in: Capsule())
                .overlay(Capsule().strokeBorder(Theme.hairline))
                .shadow(color: .black.opacity(0.08), radius: 12, y: 4)
                .padding(.horizontal, 14).padding(.bottom, 10)
            }
            .background(Theme.canvas.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    func go() {
        guard !q.isEmpty else { return }
        asked = q
        hits = m.search(q)
        written = nil
        q = ""
    }

    func write() {
        writing = true
        Task {
            do {
                let (a, _, c) = try await m.write(asked, hits.map(\.record))
                written = (a, c)
            } catch { m.lastError = error.localizedDescription }
            writing = false
        }
    }
}

// MARK: - Sessions

struct SessionsView: View {
    @EnvironmentObject var m: AppModel
    var body: some View {
        NavigationStack {
            let _ = m.version
            List {
                PageTitle(title: "Sessions", caption: "\(m.bank.sessions.count) recordings").cardRow()
                if m.bank.sessions.isEmpty { Text("No recordings yet.").foregroundStyle(Theme.muted).cardRow() }
                ForEach(m.bank.sessions.reversed()) { s in
                    NavigationLink { SessionDetail(id: s.id) } label: { SessionRow(s: s) }.cardRow()
                }
                .onDelete { idx in for i in idx { m.delete(session: Array(m.bank.sessions.reversed())[i].id) } }
                if !m.worlds.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Button { m.loadDemo() } label: { Pill(text: "Add the demo conversations", icon: "play.fill") }.buttonStyle(.plain)
                        Text("Scripted conversations bundled with the model, for trying the memory without recording.")
                            .font(.footnote).foregroundStyle(Theme.muted)
                    }.padding(.top, 8).cardRow()
                }
            }
            .listStyle(.plain)
            .paper()
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

struct SessionDetail: View {
    @EnvironmentObject var m: AppModel
    let id: String
    @State var title = ""
    @State var pane = 0

    var body: some View {
        let _ = m.version
        let s = m.bank.session(id)
        let recs = m.records(session: id)
        let kept = Set(recs.map(\.text))
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                TextField("Title", text: $title).font(Theme.serif(30)).foregroundStyle(Theme.ink).onSubmit { m.rename(session: id, to: title) }
                if let s {
                    Text(s.started.formatted(date: .abbreviated, time: .shortened)
                         + (s.ended.map { " · \(max(1, Int($0.timeIntervalSince(s.started) / 60))) min" } ?? ""))
                        .font(.subheadline).foregroundStyle(Theme.muted)
                }
                HStack(spacing: 8) {
                    Pill(text: "Memories \(recs.count)", selected: pane == 0).onTapGesture { pane = 0 }
                    Pill(text: "Transcript \(s?.utterances.count ?? 0)", selected: pane == 1).onTapGesture { pane = 1 }
                }
                if pane == 0 {
                    if recs.isEmpty { Text("No memories from this session yet.").foregroundStyle(Theme.muted) }
                    ForEach(recs) { r in MemoryCard(r: r) }
                } else {
                    ForEach(Array((s?.utterances ?? []).enumerated()), id: \.offset) { _, u in
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(String(u.t.prefix(5))).font(.caption.monospacedDigit()).foregroundStyle(Theme.faint).frame(width: 40, alignment: .leading)
                            Text(u.text).foregroundStyle(kept.contains(u.text) ? Theme.ink : Theme.muted)
                        }
                    }
                }
            }
            .padding(18)
        }
        .background(Theme.canvas.ignoresSafeArea())
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { title = s?.title ?? "" }
    }
}

// MARK: - Settings and device

struct DeviceView: View {
    @EnvironmentObject var m: AppModel
    @AppStorage("voice") var voice: Voice = .me
    @AppStorage("strictness") var strictness: Double = -1
    @State var confirmWipe = false

    var body: some View {
        NavigationStack {
            let _ = m.version
            List {
                PageTitle(title: "Settings").listRowBackground(Color.clear).listRowInsets(EdgeInsets(top: 0, leading: 4, bottom: 4, trailing: 0))
                Section {
                    Picker("Who is talking", selection: $voice) { ForEach(Voice.allCases) { Text($0.label).tag($0) } }
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Memory strictness")
                            Spacer()
                            Text(String(format: "%.2f", strictness >= 0 ? strictness : (m.validator?.threshold ?? 0.6))).monospacedDigit().foregroundStyle(Theme.muted)
                        }
                        Slider(value: Binding(get: { strictness >= 0 ? strictness : (m.validator?.threshold ?? 0.6) }, set: { strictness = $0 }), in: 0.3...0.85, step: 0.05)
                            .tint(Theme.live)
                        Text("Higher keeps fewer, more useful lines. The tested setting is \(String(format: "%.2f", m.validator?.threshold ?? 0.6)).")
                            .font(.caption).foregroundStyle(Theme.muted)
                        if strictness >= 0 { Button("Use the tested setting") { strictness = -1 }.font(.caption) }
                    }
                } header: { Text("Memory") } footer: {
                    Text("Telling speakers apart is not in this version: every line is credited to the voice chosen here.")
                }
                Section("Speech to text") {
                    row("Engine", m.asr != nil ? "Phonon-2.5 English" : m.asrPending ? "Preparing Phonon (first launch only)…" : "Apple (on device)")
                    if m.asr != nil { row("Load time", String(format: "%.1f s", m.asrLoadMs / 1000)) }
                }
                Section("System 1 on this phone") {
                    row("Models", m.models.isEmpty ? "-" : m.models.joined(separator: " + "))
                    row("Load time", String(format: "%.1f s", m.loadMs / 1000))
                    row("Memory footprint", String(format: "%.0f MB", m.footprintMB))
                    row("Thermal state", ["nominal", "fair", "serious", "critical"][ProcessInfo.processInfo.thermalState.rawValue])
                    if !m.runs.isEmpty {
                        let n = Double(m.runs.count)
                        row("Windows read", "\(m.runs.count)")
                        row("Mean per window", String(format: "%.0f ms", m.runs.map(\.ms).reduce(0, +) / n))
                        row("Lines kept", "\(m.runs.map(\.kept).reduce(0, +)) of \(m.runs.map(\.lines).reduce(0, +))")
                    }
                }
                Section("Storage") {
                    row("Sessions", "\(m.bank.sessions.count)")
                    row("Memories", "\(m.bank.records.count)")
                    if let url = m.exportURL, FileManager.default.fileExists(atPath: url.path) {
                        ShareLink(item: url) { Label("Export all memories (JSON)", systemImage: "square.and.arrow.up") }
                    }
                    Button("Erase everything", role: .destructive) { confirmWipe = true }
                }
            }
            .listStyle(.insetGrouped)
            .paper()
            .toolbar(.hidden, for: .navigationBar)
            .confirmationDialog("Erase all sessions and memories on this phone?", isPresented: $confirmWipe, titleVisibility: .visible) {
                Button("Erase", role: .destructive) { m.wipe() }
            }
        }
    }

    func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).foregroundStyle(Theme.muted).font(.callout.monospacedDigit()).multilineTextAlignment(.trailing) }
    }
}
