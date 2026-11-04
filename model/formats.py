"""Memory line formats compiled from a System 1 graph (decision 36). All are rules, so the phone can run them.

    base  compile_memories(evidence=True): "Possession [have]: Owner=ME; Possession=car | source: "..."" (decision 35)
    A     a plain-English sentence per frame from a template ("I have a car"), plus the source line
    B     A, and a frame that fills another frame's role is written inside it ("I need to tell the dentist ...")
          instead of as its own line
    C     A + B, plus times, places, amounts and counts said in the same line that are not role fillers
    E     A + B + C, with "it", "that", "he", ... replaced by the entity they most recently refer to
    D     A + B + C + E, grouped into one memory per participant ("Mike: has my ladder; works at Acme")
    G     the source line itself, tagged with its speaker and the frames found in it

    render(g, fmt, speakers, utt_text) -> list[str]
"""

from __future__ import annotations

import re

from model.decode3 import _FIRST, _YOU, _clip, compile_memories

# {Role} is a filler, {t} the trigger, {v:has/have} a verb that agrees with the template's first filler, [...] is
# dropped unless every filler inside it is present. Filled roles the template does not use are appended.
T = {
    "Opinion": "{Holder}'s view[ of {Topic}]: {t}",
    "Desiring": "{Experiencer} {v:wants/want} [{Event}][{Focal_participant}]",
    "Possession": "{Owner} {v:has/have} [{Possession}]",
    "Commerce_buy": "{Buyer} {t}[ {Goods}][ at {Place}][ for {Purpose}]",
    "Statement": "{Speaker} {t}[ to {Addressee}][: {Message}][ about {Topic}]",
    "Price_of": "[{Goods} ]costs[ {Payer}][ {Money}]",
    "Kinship": "{Ego}'s {t}[: {Alter}]",
    "Scheduling": "{Agent} {t}[ {Event}][ for {Scheduled_time}][ for {Beneficiary}][ at {Place}]",
    "Education_teaching": "{Student} {t}[ {Course}][ {Subject}][ at {Institution}]",
    "To_do": "{Agent} {v:needs/need} to[ {Task}][ by {Deadline}][ for {Requester}]",
    "Being_named": "{Entity} is called {Name}",
    "Being_employed": "{Employee} {t}[ as {Position}][ at {Employer}][ doing {Task}]",
    "Residence": "{Resident} {t}[ in {Location}]",
    "Appointment": "{Client} {v:has/have} a {t}[ for {Service}][ with {Provider}][ at {Place}]",
    "Deciding": "{Cognizer} decided[ {Decision}]",
    "Event": "{Event} happens[ at {Place}]",
    "Giving": "{Donor} {t}[ {Recipient}][ {Theme}]",
    "Locative_relation": "{Figure} is {t} {Ground}",
    "Cardinal_numbers": "[{Entity}: ]{Number}",
    "Travel": "{Traveler} {t}[ to {Goal}][ from {Source}][ by {Means}]",
    "Measure_duration": "[{Process} ]for {Count} {Unit}",
    "Awareness": "{Cognizer} {v:knows/know}[ {Content}]",
    "Activity_start": "{Agent} {t}[ {Activity}]",
    "Location_of_object": "[{Owner}'s ]{Object} is[ in {Spot}]",
    "Membership": "{Member} is a member of {Group}",
    "Needing": "{Cognizer} {v:needs/need}[ {Requirement}]",
    "Performers_and_roles": "{Performer} {t}[ in {Performance}][ as {Role}]",
    "Have_associated": "{Topical_entity} has {Entity}",
    "Ingestion": "{Ingestor} {t}[ {Ingestibles}][ at {Place}]",
    "Social_event": "{Attendee} {v:goes/go} to[ {Social_event}][ at {Place}]",
    "Causation": "[{Cause} ]{t}[ {Effect}][ for {Affected}]",
    "Choosing": "{Cognizer} {t}[ {Chosen}][ over {Possibilities}]",
    "Personal_relationship": "{Partner_1} and {Partner_2} are {t}",
    "Contact_information": "{Person}'s {t}[ is {Contact}][ ({Medium})]",
    "Frequency": "[{Salient_entity} ][{Event} ]{t}",
    "Reading_activity": "{Reader} {t}[ {Text}]",
    "Medical_conditions": "{Patient} {v:has/have}[ {Ailment}]",
    "Activity_finish": "{Agent} {t}[ {Activity}]",
    "Performing_arts": "{Performer} {t}[ {Performance}]",
    "Receiving": "{Recipient} {t}[ {Theme}][ by {Means}]",
    "Age": "{Entity} is {Age}",
    "Request": "{Speaker} asked {Addressee}[ {Message}]",
    "Size": "{Entity} is {t}[ ({Dimension})]",
    "Perception_active": "{Perceiver_agentive} {t}[ {Phenomenon}]",
    "Exercising": "{Agent} {t}",
    "Purpose": "{Agent} {v:aims/aim}[ to {Goal}]",
    "Intentionally_act": "{Agent} {t}[ {Act}][ to {Purpose}]",
    "Origin": "{Entity} is from {Origin}",
    "Commerce_pay": "{Buyer} {t}[ {Money}][ for {Goods}]",
    "Cogitation": "{Cognizer} {t}[ about {Topic}]",
    "Possibility": "{Possible_event} is possible",
    "Arriving": "{Theme} {t}[ at {Goal}]",
    "Questioning": "{Speaker} asked[ {Addressee}][ about {Topic}]",
    "Budget_status": "{Owner}'s {Account} is {t}[ ({Amount})][ because of {Cause}]",
    "Similarity": "{Entity_1} is {t}[ {Entity_2}]",
    "Allergy": "{Experiencer} is allergic to {Allergen}",
    "Lending": "{Lender} {t}[ {Borrower}][ {Theme}]",
    "Borrowing": "{Borrower} {t}[ {Theme}][ from {Lender}]",
    "Hiring": "{Employer} {t}[ {Employee}][ as {Position}]",
    "Quitting": "{Employee} {t}[ {Position}][ at {Employer}]",
    "Departing": "{Theme} {t}[ from {Source}]",
}
_TIMEY = ("Time", "Scheduled_time", "Deadline", "Date")
_EXTRA = {"DATETIME": "when", "DURATION": "for how long", "RECURRENCE": "how often", "PLACE": "where", "VENUE": "where",
          "ADDRESS": "where", "MONEY": "amount", "QUANTITY": "number"}
_PRON = {"it": "thing", "that": "thing", "this": "thing", "those": "any", "these": "any", "them": "any",
         "they": "any", "he": "PERSON", "him": "PERSON", "she": "PERSON", "her": "PERSON", "one": "thing"}
_MOD = {"planned": "plans: ", "hedged": "maybe: ", "hypothetical": "hypothetically: ", "question": "asked whether: ",
        "negated": "NOT: "}


def _names(g: dict, speakers, utt_text, f: dict, resolve: bool) -> dict:
    """Display name of every id as seen from frame f's line (same speaker rules as compile_memories)."""
    ents = g["entities"]
    name = {"USER": "ME", **{e["id"]: e["text"] for e in ents}}
    if resolve:  # a pronoun points at the latest earlier non-pronoun entity of a fitting kind
        order = sorted(range(len(ents)), key=lambda i: (ents[i]["utt"], i))
        for k, i in enumerate(order):
            want = _PRON.get(ents[i]["text"].lower())
            if not want:
                continue
            for j in reversed(order[:k]):
                c = ents[j]
                if c["text"].lower() in _PRON or c["text"].lower() in _FIRST:
                    continue
                if want == "any" or (want == "PERSON") == (c["coarse"] == "PERSON"):
                    name[ents[i]["id"]] = c["text"]
                    break
    u = f.get("utt", -1)
    other = speakers[u] if speakers and 0 <= u < len(speakers) and speakers[u] != "ME" else None
    if other and utt_text and not _YOU.search(utt_text[u]):
        name["USER"] = other
    me = other or "ME"
    return {k: (me if k != "USER" and v.lower() in _FIRST else v) for k, v in name.items()}


def _fill(tpl: str, vals: dict, t: str) -> tuple[str, set]:
    used = set()

    def sub(s: str, need_all: bool):
        ok = True

        def rep(m):
            nonlocal ok
            k = m.group(1)
            if k == "t":
                return t
            if k.startswith("v:"):
                return k[2:]  # resolved below
            if k in vals:
                used.add(k)
                return vals[k]
            ok = False
            return ""
        out = re.sub(r"\{([^}]+)\}", rep, s)
        return out if ok or not need_all else ""
    body = re.sub(r"\[([^\]]*)\]", lambda m: sub(m.group(1), True), tpl)
    first = re.search(r"\{([A-Z][^}]*)\}", tpl)
    i_subject = first and vals.get(first.group(1)) == "I"
    body = re.sub(r"\{v:([^/}]+)/([^}]+)\}", lambda m: m.group(2) if i_subject else m.group(1), body)
    body = sub(body, False)
    body = re.sub(r"(^|\W)I's\b", r"\1my", body)
    return " ".join(body.split()).strip(" :"), used


def _sentence(g, f, fr, speakers, utt_text, nest: bool, resolve: bool, depth=0) -> str:
    nm = _names(g, speakers, utt_text, f, resolve)

    def show(x):
        if x in fr:
            if nest and depth < 2:
                return _sentence(g, fr[x], fr, speakers, utt_text, nest, resolve, depth + 1)
            return f"[{fr[x]['frame'].replace('_', ' ')}: {fr[x]['trigger']}]"
        v = nm.get(x, x)
        return "I" if v == "ME" else v
    vals = {r: " & ".join(show(x) for x in xs) for r, xs in f["roles"].items()}
    t = f["trigger"].lower()
    vals = {r: v for r, v in vals.items() if not (r == "Alter" and v.lower().split()[-1:] == t.split()[-1:])}
    tpl = T.get(f["frame"])
    if tpl:
        s, used = _fill(tpl, vals, f["trigger"])
    else:
        s, used = f"{f['frame'].replace('_', ' ').lower()} ({f['trigger']})", set()
    rest = [f"{r.replace('_', ' ').lower()}: {v}" for r, v in vals.items() if r not in used]
    if rest:
        s += " (" + "; ".join(rest) + ")"
    if depth == 0:
        s = _MOD.get(f.get("modality"), "") + s
    if f.get("tense") in ("past", "future") and depth == 0:
        s += f" [{f['tense']}]"
    if f.get("update_of_prior") and depth == 0:
        s += " [changed]"
    return s


def _extras(g, f, text: str, shown: str) -> str:
    """Time/place/amount entities near the trigger (within 80 characters) that the frame does not already use."""
    used = {x for xs in f["roles"].values() for x in xs}
    at = text.lower().find(f["trigger"].lower())
    have = {r for r in f["roles"] if r in _TIMEY}
    out = []
    for e in g["entities"]:
        k = _EXTRA.get(e["coarse"])
        pos = text.lower().find(e["text"].lower())
        if (e["utt"] == f.get("utt") and e["id"] not in used and k and not (k == "when" and have)
                and pos >= 0 and abs(pos - at) <= 80 and e["text"] not in shown):
            out.append(f"{k}: {e['text']}")
    return f" ({'; '.join(dict.fromkeys(out))})" if out else ""


def _subject(f, nm) -> str:
    for xs in f["roles"].values():
        for x in xs:
            if x in nm:
                return nm[x]
    return "other"


def render(g: dict, fmt: str, speakers: list[str], utt_text: list[str], min_p: float = 0.5) -> list[str]:
    if fmt == "base":
        return compile_memories(g, min_p=min_p, speakers=speakers, utt_text=utt_text, evidence=True)
    frames = [f for f in g["frames"] if f["p"] >= min_p and f["roles"]]
    fr = {f["id"]: f for f in g["frames"]}
    if fmt == "G":
        by = {}
        for f in frames:
            by.setdefault(f["utt"], []).append(f["frame"].replace("_", " "))
        return [f"[{', '.join(dict.fromkeys(v))}] {speakers[u]}: \"{_clip(utt_text[u], '')}\"" for u, v in sorted(by.items())]
    nest = fmt in ("B", "C", "E", "D")
    resolve = fmt in ("E", "D")
    if nest:  # frames shown inside a kept parent are not repeated on their own
        inner = {x for f in frames for xs in f["roles"].values() for x in xs if x in fr}
        frames = [f for f in frames if f["id"] not in inner]
    lines, groups = [], {}
    for f in frames:
        s = _sentence(g, f, fr, speakers, utt_text, nest, resolve)
        if fmt in ("C", "E", "D"):
            u0 = f.get("utt", -1)
            s += _extras(g, f, utt_text[u0] if 0 <= u0 < len(utt_text) else "", s)
        u = f.get("utt", -1)
        who = speakers[u] if 0 <= u < len(speakers) else "ME"
        if who != "ME" and not re.search(rf"\b{who}\b", s):
            s = f"{who} said: {s}"
        src = f' | source: "{_clip(utt_text[u], f["trigger"])}"' if 0 <= u < len(utt_text) else ""
        if fmt == "D":
            nm = _names(g, speakers, utt_text, f, resolve)
            groups.setdefault(_subject(f, nm), []).append(s + src)
        elif s + src not in lines:
            lines.append(s + src)
    if fmt == "D":
        return [f"{('I' if k == 'ME' else k)}: " + " || ".join(dict.fromkeys(v)) for k, v in groups.items()]
    return lines
