import Accelerate
import CoreML
import Foundation

/// Phonon-2.5 English (FermionResearch; weights CC-BY-4.0, from nvidia/parakeet-tdt-0.6b-v3) on the phone: log-mel
/// front end, FastConformer encoder (Core ML, Neural Engine), TDT greedy decoding with the LSTM decoder and joint
/// network (Core ML, CPU). `export/phonon_coreml.py` writes the folder; this mirrors its `features` and `greedy`.
/// The first load compiles the encoder for the Neural Engine (minutes on first launch); Core ML caches it after that.
public struct PhononMeta: Codable {
    public let blank: Int
    public let vocabSize: Int
    public let durations: [Int]
    public let maxSymbolsPerStep: Int
    public let frames: [Int]
    public let d: Int
    public let nMels: Int
    public let nFft: Int
    public let winLength: Int
    public let hopLength: Int
    public let preemphasis: Float
    public let logGuard: Float
    public let normEps: Float
    public let pieces: [String]
}

/// Normalized log-mel features exactly as transformers' ParakeetFeatureExtractor computes them for one utterance.
public struct MelFrontend {
    let meta: PhononMeta
    let filters: [Float]          // [nMels, nFft / 2 + 1]
    let window: [Float]           // Hann (symmetric) of winLength, centred in nFft
    let fft: vDSP.FFT<DSPSplitComplex>

    public init(meta: PhononMeta, filters: [Float]) {
        self.meta = meta
        self.filters = filters
        var w = [Float](repeating: 0, count: meta.nFft)
        let off = (meta.nFft - meta.winLength) / 2
        for n in 0..<meta.winLength {
            w[off + n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(meta.winLength - 1))
        }
        window = w
        fft = vDSP.FFT(log2n: vDSP_Length(log2(Double(meta.nFft))), radix: .radix2, ofType: DSPSplitComplex.self)!
    }

    /// 16 kHz mono samples -> features [frames, nMels] (row major) and the frame count.
    public func features(_ x: [Float]) -> (feats: [Float], frames: Int) {
        let L = x.count, nFft = meta.nFft, hop = meta.hopLength, half = nFft / 2, bins = half + 1, M = meta.nMels
        // pre-emphasis, then centre padding with zeros (torch.stft center=True, pad_mode="constant")
        var padded = [Float](repeating: 0, count: L + nFft)
        if L > 0 {
            padded[half] = x[0]
            for i in 1..<L { padded[half + i] = x[i] - meta.preemphasis * x[i - 1] }
        }
        let frames = 1 + L / hop
        var power = [Float](repeating: 0, count: frames * bins)
        var re = [Float](repeating: 0, count: half), im = [Float](repeating: 0, count: half)
        var frame = [Float](repeating: 0, count: nFft)
        for t in 0..<frames {
            for k in 0..<nFft { frame[k] = padded[t * hop + k] * window[k] }
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBufferPointer { fp in
                        fp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                        }
                    }
                    fft.forward(input: split, output: &split)
                }
            }
            // vDSP's packed real FFT is 2x the DFT; bin 0 holds DC (real) and Nyquist (imag)
            let row = t * bins
            power[row] = (re[0] * re[0]) / 4
            power[row + half] = (im[0] * im[0]) / 4
            for k in 1..<half { power[row + k] = (re[k] * re[k] + im[k] * im[k]) / 4 }
        }
        // mel [frames, M] = power [frames, bins] x filters^T
        var mel = [Float](repeating: 0, count: frames * M)
        vDSP_mmul(power, 1, transposed(filters, M, bins), 1, &mel, 1, vDSP_Length(frames), vDSP_Length(M), vDSP_Length(bins))
        for i in 0..<mel.count { mel[i] = log(mel[i] + meta.logGuard) }
        // per-band normalization over the valid frames (unbiased std); frames past them are zero
        let valid = max(1, (L + half * 2 - nFft) / hop)
        for m in 0..<M {
            var mean: Float = 0
            for t in 0..<valid { mean += mel[t * M + m] }
            mean /= Float(valid)
            var v: Float = 0
            for t in 0..<valid { let d = mel[t * M + m] - mean; v += d * d }
            let std = sqrt(v / Float(max(1, valid - 1)))
            for t in 0..<frames { mel[t * M + m] = t < valid ? (mel[t * M + m] - mean) / (std + meta.normEps) : 0 }
        }
        return (mel, frames)
    }

    private func transposed(_ a: [Float], _ rows: Int, _ cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        vDSP_mtrans(a, 1, &out, 1, vDSP_Length(cols), vDSP_Length(rows))
        return out
    }
}

public final class PhononASR: @unchecked Sendable {
    public struct Result {
        public let text: String
        public let tokens: [Int]
        public let confidence: Double        // mean probability of the emitted tokens
        public let ms: Double
    }

    public let meta: PhononMeta
    public let frontend: MelFrontend
    let encoders: [Int: MLModel]                 // one fixed-shape function per frame count (`f512`, `f2048`)
    let decoder: MLModel, joint: MLModel
    private let lock = NSLock()

    /// `dir` holds phonon.json, mel_filters.bin and the compiled encoder / decoder / joint `.mlmodelc`.
    public init(dir: URL, encoderUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        meta = try makeDecoder().decode(PhononMeta.self, from: Data(contentsOf: dir.appendingPathComponent("phonon.json")))
        let fb = try Data(contentsOf: dir.appendingPathComponent("mel_filters.bin"))
        frontend = MelFrontend(meta: meta, filters: fb.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) })
        var encs: [Int: MLModel] = [:]
        for f in meta.frames {
            let ce = MLModelConfiguration(); ce.computeUnits = encoderUnits; ce.functionName = "f\(f)"
            encs[f] = try MLModel(contentsOf: dir.appendingPathComponent("encoder.mlmodelc"), configuration: ce)
        }
        encoders = encs
        let cc = MLModelConfiguration(); cc.computeUnits = .cpuOnly
        decoder = try MLModel(contentsOf: dir.appendingPathComponent("decoder.mlmodelc"), configuration: cc)
        joint = try MLModel(contentsOf: dir.appendingPathComponent("joint.mlmodelc"), configuration: cc)
    }

    /// Longest input of one encoder call, in samples (the largest frame bucket).
    public var maxSamples: Int { (meta.frames.last! - 1) * meta.hopLength }

    /// Transcribe one utterance (16 kHz mono). Longer audio is cut into encoder-sized pieces.
    public func transcribe(_ samples: [Float]) throws -> Result {
        lock.lock(); defer { lock.unlock() }
        let t0 = Date()
        var toks: [Int] = [], probs: [Double] = []
        var i = 0
        repeat {
            let piece = Array(samples[i..<min(samples.count, i + maxSamples)])
            let (t, p) = try decode(piece)
            toks += t; probs += p
            i += maxSamples
        } while i < samples.count
        let conf = probs.isEmpty ? 0 : probs.reduce(0, +) / Double(probs.count)
        return Result(text: detok(toks), tokens: toks, confidence: conf, ms: Date().timeIntervalSince(t0) * 1000)
    }

    public func detok(_ ids: [Int]) -> String {
        ids.map { meta.pieces[$0] }.filter { !($0.hasPrefix("<") && $0.hasSuffix(">")) }.joined()
            .replacingOccurrences(of: "\u{2581}", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// Encoder states [T, d] for one piece.
    func encode(_ samples: [Float]) throws -> (states: [Float], T: Int) {
        let (feats, F) = frontend.features(samples)
        let P = meta.frames.first { $0 >= F } ?? meta.frames.last!
        let n = min(F, P), M = meta.nMels
        let x = try MLMultiArray(shape: [1, NSNumber(value: P), NSNumber(value: M)], dataType: .float32)
        let xp = x.dataPointer.bindMemory(to: Float.self, capacity: P * M)
        xp.initialize(repeating: 0, count: P * M)
        feats.withUnsafeBufferPointer { xp.update(from: $0.baseAddress!, count: n * M) }
        let nv = try MLMultiArray(shape: [1], dataType: .int32)
        nv[0] = NSNumber(value: n)
        let out = try encoders[P]!.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": x, "n_valid": nv]))
        let e = out.featureValue(for: "enc")!.multiArrayValue!
        let T = (n + 7) / 8, d = meta.d
        return (floats(e, count: T * d), T)
    }

    /// TDT greedy decoding, as `greedy` in export/phonon_coreml.py.
    func decode(_ samples: [Float]) throws -> ([Int], [Double]) {
        let (enc, T) = try encode(samples)
        let d = meta.d, V = meta.vocabSize, blank = meta.blank
        var h = try zeros([2, 1, d]), c = try zeros([2, 1, d])
        var dec = try step(blank, &h, &c)
        let encT = try MLMultiArray(shape: [1, NSNumber(value: d)], dataType: .float32)
        let ep = encT.dataPointer.bindMemory(to: Float.self, capacity: d)
        var t = 0, toks: [Int] = [], probs: [Double] = []
        for _ in 0..<(meta.maxSymbolsPerStep * T) {
            if t >= T { break }
            enc.withUnsafeBufferPointer { ep.update(from: $0.baseAddress! + t * d, count: d) }
            let lg = floats(try joint.prediction(from: MLDictionaryFeatureProvider(dictionary: ["enc": encT, "d": dec]))
                .featureValue(for: "logits")!.multiArrayValue!, count: V + meta.durations.count)
            var tok = 0, best = -Float.infinity
            for k in 0..<V where lg[k] > best { best = lg[k]; tok = k }
            var di = 0, bd = -Float.infinity
            for k in 0..<meta.durations.count where lg[V + k] > bd { bd = lg[V + k]; di = k }
            let du = meta.durations[di]
            if tok == blank { t += max(du, 1); continue }
            var z: Double = 0
            for k in 0..<V { z += exp(Double(lg[k] - best)) }
            toks.append(tok); probs.append(1 / z)
            t += du
            dec = try step(tok, &h, &c)
        }
        return (toks, probs)
    }

    /// One decoder step; the state arrays are copied out (Core ML may reuse its output buffers).
    private func step(_ token: Int, _ h: inout MLMultiArray, _ c: inout MLMultiArray) throws -> MLMultiArray {
        let tk = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tk[0] = NSNumber(value: token)
        let out = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["token": tk, "h": h, "c": c]))
        h = try copy(out.featureValue(for: "h_out")!.multiArrayValue!)
        c = try copy(out.featureValue(for: "c_out")!.multiArrayValue!)
        return try copy(out.featureValue(for: "d")!.multiArrayValue!)
    }

    private func zeros(_ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        a.dataPointer.bindMemory(to: Float.self, capacity: a.count).initialize(repeating: 0, count: a.count)
        return a
    }

    private func copy(_ a: MLMultiArray) throws -> MLMultiArray {
        let b = try MLMultiArray(shape: a.shape, dataType: .float32)
        let v = floats(a, count: a.count)
        v.withUnsafeBufferPointer { b.dataPointer.bindMemory(to: Float.self, capacity: a.count).update(from: $0.baseAddress!, count: a.count) }
        return b
    }

    /// The first `count` values of a (contiguous, row-major) array as Float, whatever its storage type.
    private func floats(_ a: MLMultiArray, count: Int) -> [Float] {
        switch a.dataType {
        case .float32:
            return Array(UnsafeBufferPointer(start: a.dataPointer.bindMemory(to: Float.self, capacity: count), count: count))
        case .float16:
            let p = a.dataPointer.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(p[$0]) }
        default:
            return (0..<count).map { a[$0].floatValue }
        }
    }
}
