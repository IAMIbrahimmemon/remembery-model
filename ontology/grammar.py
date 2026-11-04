"""Memory grammar v3: FrameNet frames + life-memory frames, typed roles, open entity types.

A memory is a FRAME instance: a trigger span in the transcript, a frame label, and role fillers that point at
entities (or USER, or another frame). Every frame and role has a text description; the model scores spans and pairs
against embeddings of those descriptions, so the label set is open (new frames need only a description).

    frame    Borrowing            "The Borrower takes possession of a Theme ... will give it back to the Lender"
    roles    Borrower, Lender, Theme, Time, Duration, ...
    words    borrow.v, loan.n ...

Sources: FrameNet 1.7 (1,073 frames with lexical units, CC BY 3.0, https://framenet.icsi.berkeley.edu) and
CUSTOM_FRAMES below for memory needs FrameNet does not have. Build once:

    uv run --with nltk python -m ontology.grammar        # -> data/grammar/grammar.json (committed: ontology/grammar.json)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "grammar.json"
KEEP_PERIPHERAL = ("Addressee", "Recipient", "Cause", "Effect", "Destination_time", "Time", "Place", "Duration", "Frequency", "Purpose", "Explanation", "Means", "Manner",
                   "Beneficiary", "Co-participant", "Result", "Circumstances", "Instrument", "Role", "Goal", "Source")
MAX_ROLES = 14
# every frame can carry these (the compiler reads them as conditions / changed values)
UNIVERSAL_ROLES = {
    "Condition": "A condition that must hold for the frame to happen (\"when the ladder comes back\").",
    "New_value": "The new value when the frame states a change of an earlier fact (\"moved to the 14th\").",
}

# memory-specific frames (roles: name -> description)
CUSTOM_FRAMES = {
    "Appointment": ("A booked slot with a provider or service: doctor, repair visit, haircut, interview, reservation.",
                    {"Client": "Who the appointment is for.", "Provider": "The person or business providing it.",
                     "Service": "What it is for.", "Time": "When.", "Place": "Where.",
                     "Reference": "Confirmation number or booking reference."},
                    ["appointment.n", "booking.n", "reservation.n", "book.v", "reserve.v", "come out.v"]),
    "To_do": ("A task someone needs or intends to do, or asked someone else to do; reminders.",
              {"Agent": "Who should do it.", "Task": "What must be done.", "Requester": "Who asked for it.",
               "Deadline": "By when.", "Recipient": "Who the task is aimed at (\"tell Priya\")."},
              ["need to.v", "have to.v", "gotta.v", "remind.v", "remember to.v", "todo.n"]),
    "Allergy": ("Someone is allergic or intolerant to a substance, food or medication.",
                {"Patient": "Who has the allergy.", "Allergen": "What they react to.", "Reaction": "The reaction."},
                ["allergic.a", "allergy.n", "intolerant.a", "intolerance.n"]),
    "Contact_information": ("A way to reach a person or business: phone number, email, handle, address.",
                            {"Person": "Whose contact it is.", "Contact": "The number, email, handle or address.",
                             "Medium": "Phone, email, app."},
                            ["number.n", "email.n", "handle.n", "reach.v", "text.v"]),
    "Credential": ("A password, PIN, code or login for a device, network, account or door. Recognized so that "
                   "secrets can be withheld; only the existence of the credential is remembered.",
                   {"Owner": "Whose it is.", "Service": "What it unlocks (wifi, door, account).",
                    "Secret": "The secret itself (never stored)."},
                   ["password.n", "pin.n", "passcode.n", "code.n", "login.n"]),
    "Price_of": ("What something costs or cost someone.",
                 {"Goods": "The item or service.", "Money": "The amount.", "Payer": "Who pays.",
                  "Seller": "Who sells.", "Time": "When."},
                 ["cost.v", "price.n", "charge.v", "pay.v", "grand.n", "bucks.n"]),
    "Location_of_object": ("Where a specific object is kept or was left.",
                           {"Object": "The object.", "Spot": "Where it is.", "Owner": "Whose object.",
                            "Time": "Since or when."},
                           ["leave.v", "keep.v", "put.v", "drawer.n", "where.adv"]),
    "Absence": ("Someone is away, out, off work, unavailable or travelling for a period.",
                {"Person": "Who is away.", "Reason": "Why.", "Until": "Until when.", "Place": "Where they are."},
                ["out.a", "away.a", "off.a", "leave.n", "vacation.n", "unavailable.a"]),
    "Opinion": ("Someone states an opinion, like, dislike or judgement about something.",
                {"Holder": "Who holds the opinion.", "Topic": "What it is about.", "Stance": "The judgement.",
                 "Reason": "Why."},
                ["like.v", "love.v", "hate.v", "think.v", "prefer.v", "dislike.v", "favorite.a"]),
    "Budget_status": ("Spending compared with a budget or limit: over, under, on track.",
                      {"Account": "Budget, bill or account.", "Amount": "Amount spent or over.",
                       "Owner": "Whose budget.", "Period": "Which period.", "Cause": "What drove it."},
                      ["budget.n", "over budget.a", "bill.n", "spend.v"]),
}


def _clean(t: str, n: int = 240) -> str:
    t = re.sub(r"<[^>]+>", "", t or "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    first = m.group(1) if m else t
    return (first if len(first) > 40 else t)[:n].strip()


def build():
    import nltk
    nltk.data.path.insert(0, "data/real/nltk")
    from nltk.corpus import framenet as fn

    frames = {}
    for f in fn.frames():
        if not f.lexUnit:
            continue
        core = [k for k, v in f.FE.items() if v.coreType in ("Core", "Core-Unexpressed")]
        peri = [k for k in KEEP_PERIPHERAL if k in f.FE and k not in core]
        roles = {k: _clean(f.FE[k].definition, 160) for k in (core + peri)[:MAX_ROLES]}
        frames[f.name] = {"def": _clean(f.definition), "roles": roles, "words": sorted(f.lexUnit)[:40],
                          "core": [k for k in core if k in roles], "source": "framenet"}
    for name, (d, roles, words) in CUSTOM_FRAMES.items():
        frames[name] = {"def": d, "roles": roles, "words": words, "core": [r for r in roles if r not in
                        ("Time", "Place", "Reference", "Medium", "Period", "Reason")], "source": "custom"}
    for f in frames.values():
        f["roles"].update(UNIVERSAL_ROLES)
    lemma_index: dict[str, list[str]] = {}
    for name, f in frames.items():
        for w in f["words"]:
            lemma = w.rsplit(".", 1)[0].lower()
            lemma_index.setdefault(lemma, []).append(name)
    g = {"version": 3, "frames": frames, "lemma_index": lemma_index}
    OUT.write_text(json.dumps(g, indent=0, sort_keys=True))
    print(len(frames), "frames,", sum(len(f["roles"]) for f in frames.values()), "roles,", len(lemma_index), "lemmas")


def load() -> dict:
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    build()
