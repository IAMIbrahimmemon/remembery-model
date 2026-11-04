"""Turn a Window into model inputs + training targets.

Input text layout (what System 1 sees at inference too):

    [STATE]
    #1 PERSON Priya Shah
    #2 EVENT product launch
    [USER]
    [10:02:11][USER][asr=0.93] so are we still on for thursday
    [10:02:15][S2][asr=0.61] no we pushed it, um, launch is next monday now

The wearer's voice is enrolled, so their lines are labelled USER; everyone else is S2, S3...
Targets are built from gold char offsets mapped through the tokenizer's offset_mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ontology.ontology import ATTRIBUTES, ENTITY_LIST, FLAGS, RELATION_LIST, USER
from schema.memory_schema import Window

MAX_SPAN = 12          # max span width in tokens
MAX_STATE = 32
ENT2ID = {t: i + 1 for i, t in enumerate(ENTITY_LIST)}   # 0 = NONE
REL2ID = {r: i for i, r in enumerate(RELATION_LIST)}     # multi-label, no NONE column
ATTR2ID = {k: {v: i for i, v in enumerate(vals)} for k, vals in ATTRIBUTES.items()}
IMPORTANCE = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Rendered:
    text: str
    utt_char_start: list[int]                 # char offset of each utterance's text in `text`
    slot_char: list[int]                      # char offset of each state slot marker ("#k")
    user_char: int                            # char offset of the [USER] marker
    state_refs: list[str]


def render(w: Window, max_state: int = MAX_STATE) -> Rendered:
    parts: list[str] = ["[STATE]\n"]
    pos = len(parts[0])
    slot_char, state_refs = [], []
    for k, s in enumerate(w.state[-max_state:]):
        line = f"#{k + 1} {s.type} {s.canonical}\n"
        slot_char.append(pos)
        state_refs.append(s.ref)
        parts.append(line)
        pos += len(line)
    user_char = pos
    parts.append("[USER]\n")
    pos += len("[USER]\n")
    utt_char_start = []
    for u in w.utterances:
        spk = "USER" if w.speakers.get(u.spk) == USER else u.spk
        prefix = f"[{u.t}][{spk}][asr={u.asr_conf:.2f}] "
        utt_char_start.append(pos + len(prefix))
        line = prefix + u.text + "\n"
        parts.append(line)
        pos += len(line)
    return Rendered("".join(parts), utt_char_start, slot_char, user_char, state_refs)


def _char_to_token(offsets: list[tuple[int, int]]):
    """Return f(char_start, char_end) -> (tok_start, tok_end_inclusive) or None (binary search)."""
    import bisect
    idx = [i for i, (a, b) in enumerate(offsets) if b > a]
    starts = [offsets[i][0] for i in idx]
    ends = [offsets[i][1] for i in idx]

    def find(cs: int, ce: int):
        k = bisect.bisect_right(ends, cs)            # first token whose end > cs
        j = bisect.bisect_left(starts, ce) - 1       # last token whose start < ce
        if k >= len(idx) or j < 0 or j < k:
            return None
        return idx[k], idx[j]
    return find


@dataclass
class Example:
    input_ids: torch.Tensor
    slot_idx: torch.Tensor                     # token idx of each state slot
    user_idx: int
    utt_mask: torch.Tensor                     # 1 for tokens inside utterance text (span candidates)
    # gold
    mention_spans: list[tuple[int, int, int]] = field(default_factory=list)  # (ts, te, type_id)
    mention_entity: list[int] = field(default_factory=list)                  # entity index per mention
    mention_coref: list[int] = field(default_factory=list)                   # 0 = NEW, k = slot k (1-based)
    n_entities: int = 0
    rel_pairs: list[tuple[int, int]] = field(default_factory=list)           # entity idx, -1 = USER
    rel_labels: list[list[int]] = field(default_factory=list)                # multi-hot over relations
    rel_attrs: list[dict] = field(default_factory=list)
    salience: float = 0.5
    meta: dict = field(default_factory=dict)


def featurize(w: Window, tok, max_len: int = 1024) -> Example | None:
    # Drop state entries until the window fits.
    max_state = MAX_STATE
    while True:
        r = render(w, max_state)
        enc = tok(r.text, return_offsets_mapping=True, add_special_tokens=True)
        if len(enc["input_ids"]) <= max_len or max_state == 0:
            break
        max_state = max(0, max_state - 8)
    ids = enc["input_ids"][:max_len]
    offsets = [tuple(o) for o in enc["offset_mapping"][:max_len]]
    find = _char_to_token(offsets)

    def tok_at(c):
        sp = find(c, c + 1)
        return sp[0] if sp else 0

    utt_mask = torch.zeros(len(ids), dtype=torch.bool)
    for ui, u in enumerate(w.utterances):
        cs = r.utt_char_start[ui]
        sp = find(cs, cs + len(u.text))
        if sp:
            utt_mask[sp[0]:sp[1] + 1] = True

    ex = Example(input_ids=torch.tensor(ids), slot_idx=torch.tensor([tok_at(c) for c in r.slot_char], dtype=torch.long),
                 user_idx=tok_at(r.user_char), utt_mask=utt_mask, salience=w.salience,
                 meta={"window_id": w.window_id, "text": r.text, "offsets": offsets, "state_refs": r.state_refs,
                       "utt_char_start": r.utt_char_start})

    slot_of = {ref: k + 1 for k, ref in enumerate(r.state_refs)}
    ent_index: dict[str, int] = {}
    for e in w.entities:
        spans = []
        for m in e.mentions:
            cs = r.utt_char_start[m.utt] + m.start
            sp = find(cs, r.utt_char_start[m.utt] + m.end)
            if sp and sp[1] - sp[0] < MAX_SPAN:
                spans.append(sp)
        if not spans:
            continue
        idx = ent_index[e.ref] = len(ent_index)
        for ts, te in spans:
            ex.mention_spans.append((ts, te, ENT2ID[e.type]))
            ex.mention_entity.append(idx)
            ex.mention_coref.append(slot_of.get(e.ref, 0))
    ex.n_entities = len(ent_index)

    pair_rows: dict[tuple[int, int], int] = {}
    for rel in w.relations:
        h = -1 if rel.head == USER else ent_index.get(rel.head)
        t = -1 if rel.tail == USER else ent_index.get(rel.tail)
        if h is None or t is None:
            continue
        if (h, t) not in pair_rows:
            pair_rows[(h, t)] = len(ex.rel_pairs)
            ex.rel_pairs.append((h, t))
            ex.rel_labels.append([0] * len(RELATION_LIST))
            ex.rel_attrs.append({
                **{k: ATTR2ID[k][getattr(rel, k)] for k in ATTRIBUTES},
                **{f: int(getattr(rel, f)) for f in FLAGS},
                "importance": IMPORTANCE[rel.importance],
            })
        ex.rel_labels[pair_rows[(h, t)]][REL2ID[rel.label]] = 1
    return ex
