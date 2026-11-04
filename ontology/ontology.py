"""Daily-life memory ontology: the single source of truth for labels.

Entity types are span labels. Relations are typed edges between spans (or between a
span and the wearer, USER). Item flags and attributes are predicted per entity/relation.

Coverage was designed against the question categories of personal-memory benchmarks
(LongMemEval, LoCoMo, PerLTQA, LifeDialBench, MSC personas) and refined by
ontology/mine_benchmarks.py, which maps benchmark questions onto these labels and
reports gaps.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- entities
# name -> (group, description used in generation/annotation prompts)
ENTITY_TYPES: dict[str, tuple[str, str]] = {
    # people & social
    "PERSON": ("social", "A specific person, named or clearly identifiable (\"my sister\", \"Dr. Patel\")."),
    "PET": ("social", "A specific animal companion (\"Biscuit\", \"our cat\")."),
    "GROUP": ("social", "An informal group of people: family unit, friend group, team, club, class."),
    "ORG": ("social", "Company, school, hospital, government body, store chain, brand as an organization."),
    "ROLE": ("social", "Job title, position or social role (\"project manager\", \"landlord\", \"coach\")."),
    # places
    "PLACE": ("place", "Named or specific location: city, country, neighborhood, park, landmark."),
    "VENUE": ("place", "A business or public venue: restaurant, cafe, gym, store, clinic, bar, office."),
    "ADDRESS": ("place", "A street address or precise location description usable to find a place."),
    "SPOT": ("place", "A spot inside a home/car/office where things are kept (\"top drawer\", \"kitchen counter\")."),
    # time
    "DATETIME": ("time", "A date, time, or relative time expression (\"next Tuesday\", \"at 3\", \"tonight\")."),
    "DURATION": ("time", "A length of time (\"two weeks\", \"45 minutes\")."),
    "RECURRENCE": ("time", "A repeating schedule (\"every Monday\", \"twice a month\")."),
    # things
    "OBJECT": ("thing", "A personal physical item (keys, umbrella, charger, jacket)."),
    "DEVICE": ("thing", "Electronic device (phone, laptop, headphones, TV)."),
    "VEHICLE": ("thing", "Car, bike, scooter, specific vehicle."),
    "DOCUMENT": ("thing", "Passport, contract, form, receipt, ticket, prescription, report."),
    "PRODUCT": ("thing", "A product or brand-name item someone uses, wants or bought."),
    "FOOD": ("thing", "Food, dish, drink, ingredient, cuisine."),
    "CLOTHING": ("thing", "Clothing or accessory."),
    "GIFT": ("thing", "An item explicitly given or intended as a gift."),
    # activities & events
    "EVENT": ("event", "A social or one-off occurrence: party, wedding, concert, meeting, game, trip day."),
    "APPOINTMENT": ("event", "A booked slot with a provider: doctor, dentist, haircut, interview, repair."),
    "TASK": ("event", "A to-do or chore someone needs to do (\"renew the registration\")."),
    "TRIP": ("event", "Travel or a journey (\"the Lisbon trip\", \"flight home\")."),
    "ACTIVITY": ("event", "Hobby, sport, exercise or recurring pastime (\"climbing\", \"book club\")."),
    "PROJECT": ("event", "A work, school or personal project with an ongoing goal."),
    "COURSE": ("event", "A class, course, program, exam or certification."),
    "GOAL": ("event", "A stated personal goal or aspiration (\"run a half marathon\")."),
    "MILESTONE": ("event", "Life milestone: birthday, anniversary, graduation, new job, move, birth."),
    # info & values
    "MONEY": ("info", "An amount of money (\"$40\", \"half the rent\")."),
    "QUANTITY": ("info", "A non-money number with meaning (age, count, size, score, distance)."),
    "CONTACT": ("info", "Phone number, email, handle, or way to contact someone."),
    "IDENTIFIER": ("info", "Code or reference number: flight, order, confirmation, gate code, room number, plate."),
    "MEDIA": ("info", "Book, movie, show, song, album, podcast, game, article, video."),
    "TOPIC": ("info", "A subject being discussed or studied (\"tax rules\", \"machine learning\")."),
    "SKILL": ("info", "A skill or language someone has or is learning."),
    "ACCOUNT": ("info", "Subscription, membership, bill, or service account (\"Netflix\", \"gym membership\")."),
    # health & body (recorded as memory, never used to give advice)
    "HEALTH": ("health", "A condition, injury, symptom or allergy someone mentions having."),
    "MEDICATION": ("health", "A medication or supplement someone takes."),
    "DIET": ("health", "A dietary pattern or restriction (\"vegetarian\", \"no gluten\")."),
    # added from benchmark mining (ontology/coverage_report.json)
    "FEELING": ("state", "A feeling someone STATES in words (\"stressed\", \"so proud\"); never inferred from tone or faces."),
    "TRAIT": ("state", "A descriptive attribute stated about someone/something: trait, color, breed, style, appearance."),
    "ISSUE": ("state", "A problem, malfunction or challenge (\"the GPS stopped working\", \"a leak\", \"visa delay\")."),
}

# --------------------------------------------------------------------------- relations
# name -> (group, head types, tail types, description). "USER" is the wearer and is
# always an allowed head/tail where a PERSON is allowed.
P = ("PERSON",)
PP = ("PERSON", "PET", "GROUP")
WHO = ("PERSON", "GROUP", "ORG")
THING = ("OBJECT", "DEVICE", "VEHICLE", "DOCUMENT", "PRODUCT", "CLOTHING", "GIFT", "FOOD", "MEDIA")
LOC = ("PLACE", "VENUE", "ADDRESS", "SPOT")
WHEN = ("DATETIME", "RECURRENCE")
EVT = ("EVENT", "APPOINTMENT", "TASK", "TRIP", "ACTIVITY", "PROJECT", "COURSE", "MILESTONE")
ANY = tuple(ENTITY_TYPES)

RELATIONS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {
    # social graph
    "family_of": ("social", P, PP, "Family tie; the family role is the 'role' attribute (mother, cousin...)."),
    "partner_of": ("social", P, P, "Spouse, partner, dating."),
    "ex_of": ("social", P, P, "Former partner."),
    "friend_of": ("social", P, P, "Friendship."),
    "coworker_of": ("social", P, P, "Works with."),
    "manager_of": ("social", P, P, "Manages / is boss of."),
    "roommate_of": ("social", P, P, "Lives with (non-family)."),
    "neighbor_of": ("social", P, P, "Lives near."),
    "knows": ("social", P, P, "Acquainted, unspecified tie."),
    "met_at": ("social", P, LOC + EVT, "Where/at what two people met (head = person met)."),
    "member_of": ("social", P, ("GROUP", "ORG"), "Belongs to a group or org."),
    "has_pet": ("social", P, ("PET",), "Owns or cares for a pet."),
    # profile
    "alias_of": ("profile", ANY, ANY, "Nickname or alternate name for the same entity."),
    "has_role": ("profile", P, ("ROLE",), "Job title or role."),
    "works_at": ("profile", P, ("ORG", "VENUE"), "Employer or workplace."),
    "studies_at": ("profile", P, ("ORG",), "School or university."),
    "lives_in": ("profile", PP, LOC, "Current residence."),
    "from_place": ("profile", P, LOC, "Hometown or origin."),
    "age_is": ("profile", PP, ("QUANTITY",), "Age."),
    "birthday_on": ("profile", PP, ("DATETIME",), "Birthday or birth date."),
    "has_skill": ("profile", P, ("SKILL",), "Can do / speaks."),
    "learning": ("profile", P, ("SKILL", "COURSE", "TOPIC", "ACTIVITY"), "Currently learning."),
    "contact_of": ("profile", WHO, ("CONTACT",), "Contact information."),
    "has_account": ("profile", WHO, ("ACCOUNT",), "Has a subscription/membership/bill."),
    # preferences & opinions (stated in words; not inferred emotion)
    "likes": ("preference", WHO, ANY, "Stated liking or enjoyment."),
    "dislikes": ("preference", WHO, ANY, "Stated dislike."),
    "favorite_is": ("preference", WHO, ANY, "Stated favorite."),
    "wants": ("preference", WHO, ANY, "Wants, wishes for, would like to have/do."),
    "prefers_over": ("preference", ANY, ANY, "A is preferred over B (head preferred)."),
    "recommends": ("preference", P, ANY, "Head person recommends tail to someone."),
    "has_goal": ("preference", P, ("GOAL",), "Personal goal."),
    # plans, commitments, schedule
    "plans": ("plan", WHO, EVT, "Intends to do or attend."),
    "commits_to": ("plan", P, EVT, "Explicit promise (\"I'll send it tonight\")."),
    "asks_to_do": ("plan", P, EVT, "Head requests the listener/user to do tail task."),
    "assigned_to": ("plan", ("TASK", "PROJECT"), WHO, "Task is the responsibility of tail."),
    "scheduled_for": ("plan", EVT, WHEN, "When an event/task happens."),
    "deadline_is": ("plan", ("TASK", "PROJECT", "COURSE", "DOCUMENT", "ACCOUNT"), ("DATETIME",), "Due date."),
    "located_at": ("plan", EVT + WHO, LOC, "Where an event happens / where someone is."),
    "attends": ("plan", WHO, EVT, "Takes part / will take part."),
    "hosts": ("plan", WHO, EVT, "Organizes or hosts."),
    "invited_to": ("plan", WHO, EVT, "Is invited."),
    "with_person": ("plan", EVT, PP, "Event/appointment is with tail (e.g. dentist appointment with Dr. Lee)."),
    "lasts": ("plan", EVT, ("DURATION",), "Duration of event."),
    "cancels": ("plan", WHO, EVT, "Cancels an event/plan."),
    "reschedules_to": ("plan", EVT, WHEN, "Event moved to a new time (also set update_of)."),
    "reminds": ("plan", P, EVT, "Someone reminds about a task/event."),
    "needs": ("plan", WHO + EVT, ANY, "Needs an item/thing (\"bring a passport\")."),
    # objects & possessions
    "owns": ("object", WHO, THING + ("PET", "ACCOUNT"), "Owns."),
    "placed_at": ("object", THING, LOC, "Where an item was put/left."),
    "lent_to": ("object", THING, PP, "Item lent to tail."),
    "borrowed_from": ("object", THING, WHO, "Item borrowed from tail."),
    "gave": ("object", P, THING, "Head gave/gifts tail item (recipient in with_person or 'to' attribute)."),
    "received": ("object", P, THING, "Head received item."),
    "lost": ("object", P, THING, "Lost item."),
    "found_at": ("object", THING, LOC, "Item found at location."),
    # money & transactions
    "bought": ("money", WHO, THING + ("ACCOUNT", "EVENT", "TRIP"), "Purchased."),
    "costs": ("money", THING + EVT + ("ACCOUNT",), ("MONEY",), "Price/cost."),
    "owes": ("money", P, P, "Head owes tail money (amount in 'amount' attribute or a costs edge)."),
    "paid": ("money", P, ANY, "Paid for something / someone."),
    "ordered": ("money", WHO, THING, "Ordered (delivery, online, restaurant)."),
    # health (record only)
    "has_condition": ("health", PP, ("HEALTH",), "Has condition/allergy/injury."),
    "takes": ("health", PP, ("MEDICATION",), "Takes medication/supplement."),
    "follows_diet": ("health", P, ("DIET",), "Follows a diet."),
    # experiences & history
    "visited": ("experience", WHO, LOC + ("EVENT",), "Went to / visited."),
    "traveled_to": ("experience", WHO, LOC, "Traveled to."),
    "consumed": ("experience", P, ("MEDIA", "FOOD"), "Watched/read/listened/ate."),
    "does_activity": ("experience", P, ("ACTIVITY",), "Does a hobby/sport."),
    "started": ("experience", WHO, EVT + ("ACTIVITY", "SKILL", "ACCOUNT"), "Began something."),
    "stopped": ("experience", WHO, EVT + ("ACTIVITY", "ACCOUNT", "MEDICATION"), "Quit/ended something."),
    "achieved": ("experience", WHO, ("GOAL", "MILESTONE", "EVENT", "COURSE", "TASK", "PROJECT"), "Accomplished / completed."),
    "happened_on": ("experience", ANY, ("DATETIME",), "When something happened (event, purchase, first use...)."),
    "created": ("experience", WHO, THING + EVT + ("MEDIA", "FOOD"), "Made, cooked, wrote, painted, built, designed."),
    "sold": ("money", WHO, THING + ("VEHICLE",), "Sold an item."),
    "feels": ("state", P + ("GROUP",), ("FEELING",), "Stated feeling; what it is about goes in an 'about' edge or the 'to' slot."),
    "motivated_by": ("state", ANY, ANY, "Head was motivated / inspired / caused by tail (reason why)."),
    "described_as": ("state", ANY, ("TRAIT",), "Stated description of head."),
    "has_issue": ("state", ANY, ("ISSUE",), "Head has a problem/malfunction/challenge."),
    "helps": ("social", WHO, ANY, "Helps, supports, teaches or serves tail."),
    "uses": ("experience", WHO, THING + ("ACCOUNT", "SKILL", "VEHICLE", "DEVICE"), "Uses regularly."),
    "reminds_of": ("state", ANY, ANY, "Head reminds someone of / symbolizes tail."),
    "quantity_of": ("info", ANY, ("QUANTITY",), "Count, size or amount of head."),
    "believes": ("state", P, ("TOPIC", "TRAIT"), "Stated belief or value."),
    # info
    "identifier_of": ("info", ("IDENTIFIER",), ANY, "Code/number belongs to tail (flight number of trip)."),
    "about": ("info", EVT + ("MEDIA", "DOCUMENT"), ("TOPIC",) + ANY, "Topic of a meeting/media/document."),
    "part_of": ("info", ANY, ANY, "Head is part of / belongs to tail."),
}

# --------------------------------------------------------------------------- attributes
# Per-relation (and per-entity where noted) categorical attributes predicted by heads.
ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "modality": ("asserted", "planned", "hedged", "hypothetical", "question", "negated"),
    "tense": ("past", "present", "future"),
    "source": ("self_stated", "told_by_other", "overheard"),  # relative to the fact's subject
    "sensitivity": ("normal", "personal", "sensitive"),  # sensitive => health/finance/credentials
}
# Binary flags per relation
FLAGS: tuple[str, ...] = ("update_of_prior", "about_user")
# Free attribute slots that are filled by a span (not a class)
SLOT_ATTRS: tuple[str, ...] = ("role", "amount", "to")

# Never stored, regardless of model output. Passwords and PINs are stored (owner decision 2026-10-29, docs/decisions.md
# #25) as sensitive memories: on the device only, never sent to System 2 or the reader.
DROP_ALWAYS = ("full card numbers", "government ID numbers")
SENSITIVE_ON_DEVICE = ("passwords", "PINs", "door and gate codes")

USER = "USER"  # reserved entity id for the wearer

ENTITY_LIST = list(ENTITY_TYPES)
RELATION_LIST = list(RELATIONS)


def describe_for_prompt() -> str:
    """Compact human-readable ontology for LLM prompts."""
    lines = ["ENTITY TYPES:"]
    for name, (group, desc) in ENTITY_TYPES.items():
        lines.append(f"- {name} [{group}]: {desc}")
    lines.append("\nRELATIONS (head -> tail):")
    for name, (group, heads, tails, desc) in RELATIONS.items():
        h = "ANY" if heads == ANY else "|".join(heads)
        t = "ANY" if tails == ANY else "|".join(tails)
        lines.append(f"- {name} [{group}] ({h} -> {t}): {desc}")
    lines.append("\nRELATION ATTRIBUTES:")
    for k, v in ATTRIBUTES.items():
        lines.append(f"- {k}: one of {', '.join(v)}")
    lines.append(f"- flags: {', '.join(FLAGS)} (booleans)")
    lines.append(f"- slot attrs (optional, a verbatim span): {', '.join(SLOT_ATTRS)}")
    lines.append(f"\nThe wearer is the special entity {USER}; head/tail may be {USER} wherever PERSON is allowed.")
    lines.append(f"Never record: {', '.join(DROP_ALWAYS)}.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"{len(ENTITY_TYPES)} entity types, {len(RELATIONS)} relations")
    print(describe_for_prompt())
