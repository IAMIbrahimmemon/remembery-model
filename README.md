# Remembery: on-device memory for AI glasses

[![Watch the Remembery demo (2:58)](media/remembery_demo.jpg)](media/remembery_demo.mp4)

**[Watch the demo video (2:58)](media/remembery_demo.mp4)** · [direct link](https://d2ol7oe51mr4n9.cloudfront.net/user_3LWi4JbaoVmXG6lQZ36EVWNWRH0/8e946f4c-80ac-4a07-9588-4ea7457c3986.mp4)

AI glasses hear many hours of conversation a day. Remembery turns that conversation into memories **on the phone**, with
no cloud and no LLM running all day. Remembery's System 1, a small Meta **MobileLLM-R1 (360M)** with non-autoregressive heads, reads the live
transcript in one forward pass per window. It picks out what happened (plans, appointments, people, places, times,
health, money) and keeps the lines worth remembering as structured records you can search offline.

> Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Indie track.

## Results

On 85 held-out real conversations (Switchboard phone calls, Taskmaster task dialogues, AMI-style meetings) that no
model trained on. Each conversation has a checklist of facts worth remembering. For every system, a fact is covered,
partial or missed, and each stored memory is correct, partial, wrong or trivial.

| System | Recall | Precision | F1 |
|---|---|---|---|
| **Remembery System 1, on the phone (sealed set, scored once)** | **0.799** | **0.783** | **0.791** |
| Remembery System 1 (dev set, 89 conversations) | 0.774 | 0.788 | 0.781 |
| Large cloud LLM labeling the same conversations (dev, 83 conversations) | 0.814 | 0.892 | 0.851 |
| Muse Spark labeling the same conversations (dev, 83 conversations) | 0.806 | 0.912 | 0.856 |
| Store every line of 4+ words, no model (dev) | 0.910 | 0.499 | 0.644 |

The phone model catches 95–98% of what the large LLMs catch (recall). Its F1 is about 92% of theirs, and wrong memories
are 0.2%. Most of its precision loss is true but trivial lines (21%), not errors.

On the iPhone 17 (Release build):

| What | Measured |
|---|---|
| System 1, both models, one full window (~1,000 tokens, a couple of minutes of talk) | 4.5–5.4 s |
| Load (both models) | 4.3–5.4 s |
| Memory footprint | ~4.5 GB: two fp32 models (360M × 4 bytes ≈ 1.4 GB of weights each) plus runtime. Next: int8 weights (trained with quantization in the loop, since plain int4/fp16 changed memories) and one model instead of two would cut this several-fold. |
| Speech to text (Phonon-2.5 English, Neural Engine), natural two-voice test | 831 / 832 words, word error 5.0% (Apple's on-device recognizer: 43%) |

## How it works

```
mic ─► Phonon-2.5 English (Core ML, Neural Engine) ─► lines cut at pauses
     ─► window of ~2,000 characters (or 40 lines) ─► System 1, two models, one pass each (ExecuTorch, CPU)
            entities · triggers · frames with roles, each with a probability
     ─► candidate lines: a frame with a role, or a time / place / number / health / event entity
     ─► line validator (26 features from the graphs, logistic, cutoff 0.60)
     ─► memory record: the line tagged with its frames, speaker, date, roles, entities, previous line
     ─► memory bank on the phone ─► Ask: offline search (BM25); optional written answer from Muse Spark
```

- **System 1** is MobileLLM-R1 360M with a bidirectional adapter and heads for spans, entity types, frame triggers,
  roles (speakers can fill roles), attributes and salience. It is non-autoregressive: one forward pass scores every
  span and relation at once, so cost is fixed per window and there is no generation to drift.
- **Two models pooled.** Two training runs (v4e `best.pt` and v4f `last.pt`) read every window and their graphs are
  pooled. Pooling is part of the score.
- **Memory format G.** A memory is the speaker's own words, tagged with the kinds of fact the models found:
  `[Appointment, Being employed] ME: "yeah he's coming thursday morning, between eight and noon, so I gotta work from home"`.
  Short or pronoun-led lines carry the previous line as context. Quoting the speaker keeps wrong memories near zero.
- **Line validator.** A logistic model over features of the pooled graphs keeps only the useful lines
  (`model/validator_g2.json`).
- **Privacy.** Everything runs on the phone. Lines with passwords, PINs or codes are kept but flagged sensitive and
  never sent anywhere; card and government ID numbers are never stored. Nothing leaves the phone unless the user asks
  Muse for a written answer.

## Meta tools used

| Role | Meta tool |
|---|---|
| System 1 backbone, on the phone | MobileLLM-R1-360M, fine-tuned |
| On-device runtime | ExecuTorch (XNNPACK), five-method `.pte` per model, fp32 |
| Early training labels | Muse Spark (Meta Model API) |
| Optional written answers in the app | Muse Spark |

The final training labels for real conversations (about 21,600 windows) were written by large LLMs, in
the same frame grammar Muse used first.

## Repository

| Path | What |
|---|---|
| `memories.py` | Run the final system on transcript windows (Python): both models, pooling, format G, validator, records |
| `model/` | System 1 architecture (`nar3.py`), featurizer, decoder, line formats, validator weights |
| `ontology/` | entity types and the frame grammar (`grammar.json`) |
| `eval/` | `predict3.py` (one model on windows), `validator.py apply` (lines and records from graphs) |
| `export/` | `ondevice3.py`: System 1 to ExecuTorch `.pte`; `phonon_coreml.py`: speech to text to Core ML |
| `ios/` | SwiftUI app and the `MemNARCore` Swift package (featurizer, extractor, memory lines, bank, search, Phonon) |
| `data/probe/` | two short test conversations |
| `data/dev/` | the dev set: 89 real conversations (`dev_windows.jsonl`) and their fact checklists (`dev_checklists.jsonl`, 376 facts) |
| `media/` | the demo video |

## Run it

The checkpoints, phone model folders and speech model are on Hugging Face:
[`IAMIbrahim/memnar-system1`](https://huggingface.co/IAMIbrahim/memnar-system1). The backbone config and
tokenizer come from [`facebook/MobileLLM-R1-360M-base`](https://huggingface.co/facebook/MobileLLM-R1-360M-base) (gated: request
access and `hf auth login` first).

**Python (Mac or Linux, CPU is fine):**

```
uv sync
hf download IAMIbrahim/memnar-system1 --include "checkpoints/*" --local-dir hf
PYTHONPATH=. uv run python memories.py --ckpts hf/checkpoints --windows data/probe/probe_v1.jsonl \
    --out memories.jsonl --records records.jsonl
```

The kitchen probe keeps 7 of 8 lines and the work probe 6 of 6. A window is one JSON line:
`{"window_id", "date", "speakers": {"S1": "USER", "S2": "UNKNOWN"}, "utterances": [{"spk", "t", "text"}]}`.

**Dev set.** Run the same command on the 89 held-out dev conversations (Switchboard, Taskmaster-1, QMSum meetings;
no training window comes from them):

```
PYTHONPATH=. uv run python memories.py --ckpts hf/checkpoints --windows data/dev/dev_windows.jsonl \
    --out dev_memories.jsonl --records dev_records.jsonl
```

Each line of `data/dev/dev_checklists.jsonl` lists the facts worth remembering in that conversation, marked `must` or
`nice`. Recall is covered facts (+½ partial) over all facts; precision is correct memories (+½ partial) over all stored
memories. The sealed test set is kept private so it stays a clean held-out score.

**iPhone (Xcode 16+, iOS 18+; tested on an iPhone 17):**

```
hf download IAMIbrahim/memnar-system1 --include "ios/*" --local-dir hf
unzip -q hf/ios/memnar-ios-model-v4.zip -d ios       # ios/Model   (v4e)
unzip -q hf/ios/memnar-ios-model-v4-b.zip -d ios     # ios/Model_b (v4f)
unzip -q hf/ios/memnar-ios-phonon.zip -d ios         # ios/Phonon  (speech to text; without it, Apple's recognizer)
cd ios && xcodegen generate && open MemNAR.xcodeproj  # set your signing team, build Release to the phone
```

The Device tab should list two models and Memory strictness at the tested 0.60 (launch argument `-tested` resets it).
`-autotest 6` prints per-window time and kept lines. For a written answer in Ask, add `ios/App/Secrets.plist` with a
`BasetenKey` string (optional; search works offline without it).

**Checks that the phone runs the scored system** (Mac, no phone needed):

```
cd ios/MemNARCore
swift run -c release memnar-cli ../Model ../Model_b     # Swift = Python over each .pte: tokens and graphs 22/22
```

**Export your own `.pte`** (from a checkpoint): `uv sync --extra export`, then
`PYTHONPATH=. uv run python -m export.ondevice3 --run hf/checkpoints/v4e --ckpt best.pt --vocab hf/vocab3.json --out ios/Model`
(and `v4f`, `last.pt`, `ios/Model_b`). Weights stay fp32: int4 and fp16 changed the memories of most windows.

## Licenses

- **Code:** Apache-2.0 (`LICENSE`). Commercial use is allowed; keep the `NOTICE` file and credit
  "Remembery by Ibrahim Memon".
- **System 1 weights:** fine-tuned from MobileLLM-R1-360M-base, so they carry Meta's FAIR Noncommercial Research
  License (research use only). A commercial version needs the same training on a commercially licensed backbone.
- **Speech model** used by the app: Phonon-2.5 English (FermionResearch, CC-BY-4.0); not part of this project's work.

Details in `THIRD_PARTY.md`.
