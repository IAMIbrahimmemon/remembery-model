import Foundation

/// Exported alongside memnar.pte by export/ondevice3.py.
public struct ModelMeta: Decodable {
    public let version: Int
    public let run: String
    public let d: Int
    public let maxSpan: Int
    public let maxLen: Int
    public let nSlots: Int
    public let chunk: Int
    public let maxItems: Int
    public let padMin: Int
    public let entThresh: Double
    public let trgThresh: Double
    public let roleThresh: Double
    public let s2Lo: Double
    public let s2Hi: Double
    public let frames: [String]
    public let frameRoles: [[String]]
    public let frameCore: [[String]]?
    public let types: [String]
    public let coarse: [String]
    public let attrKeys: [String]
    public let attrValues: [String: [String]]
    public let flags: [String]
    public let bosId: Int
    public let vocabPruned: Bool
    public let quant: String
    public let speakers: Bool?      // v4: other speakers are role fillers (roles takes speaker arguments)
    public let nsMax: Int?          // v4: other speakers per window the roles program takes
}

public struct Utterance: Codable, Hashable {
    public var spk: String
    public var t: String
    public var text: String
    public var asrConf: Double

    public init(spk: String, t: String, text: String, asrConf: Double = 1.0) {
        self.spk = spk; self.t = t; self.text = text; self.asrConf = asrConf
    }

    enum CodingKeys: String, CodingKey { case spk, t, text, asrConf }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        spk = try c.decode(String.self, forKey: .spk)
        t = try c.decode(String.self, forKey: .t)
        text = try c.decode(String.self, forKey: .text)
        asrConf = try c.decodeIfPresent(Double.self, forKey: .asrConf) ?? 1.0
    }
}

public struct StateEntity: Codable, Hashable {
    public var ref: String
    public var type: String
    public var canonical: String

    public init(ref: String, type: String, canonical: String) {
        self.ref = ref; self.type = type; self.canonical = canonical
    }
}

/// One ~2-3 minute conversation window (schema/memory_schema.py; gold labels are not needed on device).
public struct Window: Codable, Identifiable, Hashable {
    public var windowId: String
    public var worldId: String
    public var day: Int
    public var date: String
    public var speakers: [String: String]
    public var utterances: [Utterance]
    public var state: [StateEntity]
    public var scene: String

    public var id: String { windowId }
    public var isSpoken: Bool { windowId.hasSuffix("_asr") }

    public init(windowId: String, worldId: String, day: Int, date: String, speakers: [String: String],
                utterances: [Utterance], state: [StateEntity] = [], scene: String = "") {
        self.windowId = windowId; self.worldId = worldId; self.day = day; self.date = date
        self.speakers = speakers; self.utterances = utterances; self.state = state; self.scene = scene
    }

    enum CodingKeys: String, CodingKey { case windowId, worldId, day, date, speakers, utterances, state, scene }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        windowId = try c.decode(String.self, forKey: .windowId)
        worldId = try c.decodeIfPresent(String.self, forKey: .worldId) ?? ""
        day = try c.decodeIfPresent(Int.self, forKey: .day) ?? 1
        date = try c.decodeIfPresent(String.self, forKey: .date) ?? ""
        speakers = try c.decode([String: String].self, forKey: .speakers)
        utterances = try c.decode([Utterance].self, forKey: .utterances)
        state = try c.decodeIfPresent([StateEntity].self, forKey: .state) ?? []
        scene = try c.decodeIfPresent(String.self, forKey: .scene) ?? ""
    }

    /// The user's speaker label ("S1" when none is marked USER), as eval/predict3.py.
    public var userSpeaker: String { speakers.filter { $0.value == USER }.map(\.key).sorted().first ?? "S1" }
    /// "ME" or the speaker label of each utterance (compile_memories' `speakers`).
    public var speakerLabels: [String] { let u = userSpeaker; return utterances.map { $0.spk == u ? "ME" : $0.spk } }
}

/// One entity of the v3 graph (model/decode3.py extract3).
public struct Entity3: Hashable, Identifiable {
    public let id: String
    public let text: String
    public let utt: Int
    public let type: String
    public let typeP: Double
    public let coarse: String
    public let p: Double
    public let link: String?
}

/// One frame instance: trigger + frame + roles (role -> filler ids, in extract3's order) + attributes.
public struct Frame3: Hashable, Identifiable {
    public let id: String
    public let frame: String
    public let frameP: Double
    public let trigger: String
    public let utt: Int
    public let p: Double
    public let roles: [(String, [String])]
    public let attrs: [String: String]
    public let flags: [String: Bool]

    public var modality: String { attrs["modality"] ?? "asserted" }
    public var tense: String { attrs["tense"] ?? "present" }
    public var importance: String { attrs["importance"] ?? "medium" }

    public static func == (a: Frame3, b: Frame3) -> Bool {
        a.id == b.id && a.frame == b.frame && a.trigger == b.trigger && a.utt == b.utt && a.p == b.p
            && a.roles.map(\.0) == b.roles.map(\.0) && a.roles.map(\.1) == b.roles.map(\.1) && a.attrs == b.attrs && a.flags == b.flags
    }
    public func hash(into h: inout Hasher) { h.combine(id); h.combine(frame); h.combine(trigger) }
}

public struct Graph3 {
    public let entities: [Entity3]
    public let frames: [Frame3]
    public let salience: Double

    public init(entities: [Entity3], frames: [Frame3], salience: Double) {
        self.entities = entities; self.frames = frames; self.salience = salience
    }
}

public let USER = "USER"

public func makeDecoder() -> JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
}

/// Python's round(x, 3): printf rounds the exact binary value, as CPython does.
public func pyRound3(_ x: Double) -> Double { Double(String(format: "%.3f", x))! }
