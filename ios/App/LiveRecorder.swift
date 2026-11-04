import AVFoundation
import Foundation
import MemNARCore
import Speech

/// Continuous transcription on the phone: one tap starts it and it keeps listening until stopped (also with the
/// screen locked). With Phonon (`asr`, the default) the audio is cut into lines at pauses and each line is transcribed
/// on the Neural Engine (`PhononLines`). Without it, Apple's recognizer is used: a pause of `pause` seconds, or
/// `maxSegment` seconds of speech, closes the current line and the next request starts at once on the same stream.
@MainActor
final class LiveRecorder: ObservableObject {
    @Published var recording = false
    @Published var authorized = false
    @Published var onDevice = false
    @Published var partial = ""
    @Published var level: Float = 0
    @Published var started: Date?
    @Published var note: String?

    /// Called on the main actor for every finished line: (text, when it started, mean recognizer confidence).
    var onLine: ((String, Date, Double) -> Void)?
    /// Called when recording stops on its own (an interruption), so the session can be closed and its tail read.
    var onStop: (() -> Void)?
    var pause = 1.4
    var maxSegment = 50.0
    /// Phonon-2.5 English; when set, it replaces Apple's recognizer.
    var asr: PhononASR?
    private var lines: PhononLines?
    /// Input levels of the Phonon line cutter since the last call (nil with Apple's recognizer).
    func levelStats() -> (floor: Float, median: Float, p90: Float)? { lines?.stats() }

    private let engine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let box = RequestBox()
    private var seg: Segment?
    private var timer: Timer?
    private var observers: [NSObjectProtocol] = []

    final class Segment {
        let request: SFSpeechAudioBufferRecognitionRequest
        var task: SFSpeechRecognitionTask?
        let start = Date()
        var lastChange = Date()
        var text = ""
        var conf = 0.9
        var done = false
        init(_ r: SFSpeechAudioBufferRecognitionRequest) { request = r }
    }

    /// The audio tap runs on the audio thread and appends to whichever request is current.
    final class RequestBox: @unchecked Sendable {
        private let lock = NSLock()
        private var req: SFSpeechAudioBufferRecognitionRequest?
        func set(_ r: SFSpeechAudioBufferRecognitionRequest?) { lock.lock(); req = r; lock.unlock() }
        func append(_ b: AVAudioPCMBuffer) { lock.lock(); let r = req; lock.unlock(); r?.append(b) }
    }

    init() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] n in
            let began = (n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init) == .began
            Task { @MainActor in
                guard let self, began, self.recording else { return }
                await self.stop()
                self.note = "Recording stopped by an interruption (call, Siri or another app)."
                self.onStop?()
            }
        })
    }

    /// Phonon needs only the microphone; Apple's recognizer also needs speech-recognition permission.
    func authorize() {
        SFSpeechRecognizer.requestAuthorization { st in
            AVAudioApplication.requestRecordPermission { ok in
                Task { @MainActor in
                    self.speechAllowed = st == .authorized
                    self.authorized = ok && (st == .authorized || self.asr != nil)
                    self.onDevice = self.asr != nil || (self.recognizer?.supportsOnDeviceRecognition ?? false)
                }
            }
        }
    }

    private var speechAllowed = false

    /// Called once Phonon has loaded (after `authorize`, which may have run without it).
    func use(_ asr: PhononASR) {
        self.asr = asr
        onDevice = true
        if AVAudioApplication.shared.recordPermission == .granted { authorized = true }
    }

    func start() throws {
        note = nil
        if let asr { try startPhonon(asr); return }
        guard let recognizer, recognizer.isAvailable, speechAllowed else { throw NSError(domain: "speech", code: 1, userInfo: [NSLocalizedDescriptionKey: "Speech recognition is not available"]) }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default, options: [.allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        let input = engine.inputNode
        let fmt = input.outputFormat(forBus: 0)
        let box = self.box
        input.installTap(onBus: 0, bufferSize: 2048, format: fmt) { [weak self] buf, _ in
            box.append(buf)
            guard let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength)
            var s: Float = 0
            for i in 0..<n { s += ch[i] * ch[i] }
            let rms = sqrt(s / Float(max(1, n)))
            let lvl = min(1, max(0, (20 * log10(max(rms, 1e-6)) + 55) / 50))  // about -55 dB .. -5 dB -> 0 .. 1
            Task { @MainActor in self?.level = 0.7 * (self?.level ?? 0) + 0.3 * lvl }
        }
        startSegment()
        engine.prepare()
        try engine.start()
        recording = true
        started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in Task { @MainActor in self?.tick() } }
    }

    private func startSegment() {
        guard let recognizer else { return }
        let r = SFSpeechAudioBufferRecognitionRequest()
        r.shouldReportPartialResults = true
        r.addsPunctuation = true
        if recognizer.supportsOnDeviceRecognition { r.requiresOnDeviceRecognition = true }
        let s = Segment(r)
        seg = s
        box.set(r)
        s.task = recognizer.recognitionTask(with: r) { [weak self, weak s] result, error in
            Task { @MainActor in
                guard let self, let s else { return }
                if let result {
                    let t = result.bestTranscription.formattedString
                    if t != s.text { s.text = t; s.lastChange = Date() }
                    let segs = result.bestTranscription.segments.map(\.confidence).filter { $0 > 0 }
                    if !segs.isEmpty { s.conf = Double(segs.reduce(0, +)) / Double(segs.count) }
                    if s === self.seg { self.partial = t }
                    if result.isFinal { self.finish(s) }
                }
                if error != nil { self.finish(s) }
            }
        }
    }

    private func tick() {
        guard recording, let s = seg else { return }
        let now = Date()
        let quiet = now.timeIntervalSince(s.lastChange), long = now.timeIntervalSince(s.start)
        if !s.text.isEmpty && (quiet >= pause || long >= maxSegment) {
            rotate()
        } else if s.text.isEmpty && long >= 55 {
            rotate()  // a long silence: renew the request before the recognizer gives up on it
        }
    }

    /// Close the current line and keep listening with a fresh request.
    private func rotate() {
        guard let old = seg else { return }
        startSegment()
        partial = ""
        old.request.endAudio()
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            self.finish(old)  // no final result in time: keep what was recognized
        }
    }

    private func finish(_ s: Segment) {
        guard !s.done else { return }
        s.done = true
        s.task?.cancel()
        let text = s.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { onLine?(text, s.start, s.conf) }
        if s === seg && recording {  // the current request ended on its own (error or final): start another
            partial = ""
            box.set(nil)
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 300_000_000)  // no tight loop if the recognizer keeps failing
                if self.recording && self.seg === s { self.startSegment() }
            }
        }
    }

    // MARK: Phonon

    private func startPhonon(_ asr: PhononASR) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default, options: [.allowBluetooth])   // with the system's gain control
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        let input = engine.inputNode
        let fmt = input.outputFormat(forBus: 0)
        let t0 = Date()
        let pl = PhononLines(asr: asr, inputFormat: fmt, pause: 1.0, maxLine: 20)
        pl.onLine = { [weak self] text, offset, conf in
            Task { @MainActor in self?.onLine?(text, t0.addingTimeInterval(offset), conf) }
        }
        pl.onPartial = { [weak self] text in Task { @MainActor in if self?.recording == true { self?.partial = text } } }
        lines = pl
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { [weak self] buf, _ in
            let rms = pl.append(buf)
            let lvl = min(1, max(0, (20 * log10(max(rms, 1e-6)) + 55) / 50))
            Task { @MainActor in self?.level = 0.7 * (self?.level ?? 0) + 0.3 * lvl }
        }
        engine.prepare()
        try engine.start()
        recording = true
        started = t0
    }

    func stop() async {
        guard recording else { return }
        recording = false
        timer?.invalidate()
        timer = nil
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        if let pl = lines {
            lines = nil
            await pl.finish()   // the line in progress is transcribed before the session closes
            partial = ""
            level = 0
            started = nil
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            return
        }
        let last = seg
        seg = nil
        box.set(nil)
        last?.request.endAudio()
        if let last {
            for _ in 0..<20 where !last.done { try? await Task.sleep(nanoseconds: 100_000_000) }
            finish(last)
        }
        partial = ""
        level = 0
        started = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}

/// Phonon's line cutter: 16 kHz mono audio in 20 ms frames. No audio is dropped while someone speaks: a line holds
/// everything since the previous cut and closes after `pause` seconds of quiet that follow speech (or at `maxLine`).
/// Speech and quiet are judged against a noise floor that follows the quietest frames (falls fast, rises slowly), so
/// soft voices and sentence endings stay in. Lines are transcribed one at a time on a serial queue; while someone
/// speaks, the line so far is transcribed about every 1.5 s for the live preview (skipped when busy).
final class PhononLines: @unchecked Sendable {
    var onLine: ((String, TimeInterval, Double) -> Void)?
    var onPartial: ((String) -> Void)?

    private let asr: PhononASR
    private let converter: AVAudioConverter?
    private let out = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private let pauseFrames: Int, maxSamples: Int
    private let queue = DispatchQueue(label: "phonon.lines", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = 0

    // audio-thread state (guarded by `lock`)
    private var cur: [Float] = []                 // audio since the last cut
    private var curStart = 0                      // sample offset of `cur` since start
    private var total = 0
    private var speech = false                    // `cur` has speech in it
    private var quietFrames = 0
    private var floor: Float = 0.002              // noise floor (RMS)
    private var lastPartial = 0
    private var carry: [Float] = []
    private var levels: [Float] = []              // frame RMS since the last `stats()`

    init(asr: PhononASR, inputFormat: AVAudioFormat, pause: Double, maxLine: Double) {
        self.asr = asr
        converter = AVAudioConverter(from: inputFormat, to: out)
        pauseFrames = Int(pause / 0.02)
        maxSamples = min(Int(maxLine * 16000), asr.maxSamples)
    }

    /// Audio thread: convert, look at 20 ms frames, close lines. Returns the buffer's RMS for the level meter.
    func append(_ buf: AVAudioPCMBuffer) -> Float {
        guard let converter, let o = AVAudioPCMBuffer(pcmFormat: out, frameCapacity: AVAudioFrameCount(Double(buf.frameLength) * 16000 / buf.format.sampleRate) + 64) else { return 0 }
        var fed = false
        var err: NSError?
        converter.convert(to: o, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buf
        }
        guard err == nil, let ch = o.floatChannelData?[0] else { return 0 }
        let x = Array(UnsafeBufferPointer(start: ch, count: Int(o.frameLength)))
        var s: Float = 0
        for v in x { s += v * v }
        lock.lock(); defer { lock.unlock() }
        carry += x
        var i = 0
        while carry.count - i >= 320 { frame(carry[i..<(i + 320)]); i += 320 }
        carry.removeFirst(i)
        return sqrt(s / Float(max(1, x.count)))
    }

    private func frame(_ f: ArraySlice<Float>) {
        var e: Float = 0
        for v in f { e += v * v }
        let rms = sqrt(e / 320)
        levels.append(rms)
        floor = rms < floor ? 0.8 * floor + 0.2 * rms : min(floor * 1.0005, 0.02)   // follows the quiet frames
        let loud = rms > max(0.0015, floor * 2.5)
        let quiet = rms < max(0.001, floor * 1.8)
        cur += f
        total += f.count
        if loud { speech = true; quietFrames = 0 } else if quiet { quietFrames += 1 }
        if !speech {
            if cur.count > 16000 { let drop = cur.count - 8000; cur.removeFirst(drop); curStart += drop }   // keep 0.5 s
            return
        }
        if quietFrames >= pauseFrames || cur.count >= maxSamples {
            close(trim: quietFrames >= pauseFrames)
        } else if cur.count - lastPartial >= 24000 {
            lastPartial = cur.count
            partial(cur)
        }
    }

    /// Hand the current line to the transcription queue (lock held); the quiet tail starts the next line.
    private func close(trim: Bool) {
        var line = cur
        let tail = trim ? max(0, min(line.count, (quietFrames - 15) * 320)) : 0   // keep 0.3 s after the last word
        line.removeLast(tail)
        let offset = Double(curStart) / 16000
        cur = Array(cur.suffix(min(tail, 4800))); curStart = total - cur.count
        speech = false; quietFrames = 0; lastPartial = 0
        busy += 1
        queue.async { [self] in
            if let r = try? asr.transcribe(line), !r.text.isEmpty { onLine?(r.text, offset, r.confidence) }
            onPartial?("")
            lock.lock(); busy -= 1; lock.unlock()
        }
    }

    private func partial(_ line: [Float]) {
        guard busy == 0 else { return }
        busy += 1
        queue.async { [self] in
            if let r = try? asr.transcribe(line) { onPartial?(r.text) }
            lock.lock(); busy -= 1; lock.unlock()
        }
    }

    /// Noise floor and median / 90th-percentile frame level since the last call (for the `-record` log).
    func stats() -> (floor: Float, median: Float, p90: Float) {
        lock.lock(); defer { lock.unlock() }
        let l = levels.sorted(); levels = []
        guard !l.isEmpty else { return (floor, 0, 0) }
        return (floor, l[l.count / 2], l[l.count * 9 / 10])
    }

    /// Recording stopped: transcribe what is left and wait for every queued line.
    func finish() async {
        lock.lock()
        if speech && cur.count > 3200 { close(trim: false) }
        lock.unlock()
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in queue.async { c.resume() } }
    }
}
