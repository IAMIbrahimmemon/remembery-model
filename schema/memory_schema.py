"""Typed records shared by data generation, training, the memory store and System 2."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ontology.ontology import ATTRIBUTES

Modality = Literal[ATTRIBUTES["modality"]]  # type: ignore[valid-type]
Tense = Literal[ATTRIBUTES["tense"]]  # type: ignore[valid-type]
Source = Literal[ATTRIBUTES["source"]]  # type: ignore[valid-type]
Sensitivity = Literal[ATTRIBUTES["sensitivity"]]  # type: ignore[valid-type]


class Utterance(BaseModel):
    spk: str                       # diarization label as the model sees it: S1, S2...
    t: str                         # "HH:MM:SS" wall-clock start
    text: str
    asr_conf: float = 1.0          # mean ASR word confidence (1.0 for clean text)


class Mention(BaseModel):
    utt: int                       # utterance index in the window
    start: int                     # char offset inside utterance.text
    end: int
    text: str


class Entity(BaseModel):
    ref: str                       # stable id across the storyline (coreference cluster)
    type: str
    canonical: str                 # canonical name ("Priya Shah")
    mentions: list[Mention]
    is_new: bool = True            # False if ref was already in [STATE] for this window


class Relation(BaseModel):
    head: str                      # entity ref or "USER"
    tail: str
    label: str
    modality: Modality = "asserted"
    tense: Tense = "present"
    source: Source = "self_stated"
    sensitivity: Sensitivity = "normal"
    update_of_prior: bool = False
    about_user: bool = False
    role: Optional[str] = None     # verbatim slot fillers
    amount: Optional[str] = None
    to: Optional[str] = None
    evidence: list[int] = Field(default_factory=list)  # utterance indices
    importance: Literal["low", "medium", "high"] = "medium"


class StateEntity(BaseModel):
    ref: str
    type: str
    canonical: str


class Window(BaseModel):
    window_id: str
    world_id: str
    day: int
    date: str                      # ISO date of the window
    speakers: dict[str, str]       # S1 -> "USER" / entity ref / "UNKNOWN"
    utterances: list[Utterance]
    state: list[StateEntity] = Field(default_factory=list)  # carried context entering this window
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    salience: float = 0.5          # 0 = nothing worth remembering, 1 = critical
    scene: str = ""
    noise: Literal["clean", "text_noise", "audio_roundtrip"] = "clean"


class MemoryItem(BaseModel):
    """A relation as stored in the memory graph, with calibrated confidence."""
    id: str
    head: str
    tail: str
    label: str
    p: float
    attrs: dict = Field(default_factory=dict)
    t_observed: str                # when it was said (ISO datetime)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None  # set when superseded
    status: Literal["stored", "queued", "verified", "rejected", "superseded"] = "stored"
    evidence_text: str = ""
