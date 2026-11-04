# Third-party models and data

The code in this repository is licensed under Apache-2.0 (`LICENSE`, `NOTICE`): commercial use is allowed with credit. The model weights it runs are not covered by that license:

| What | Source | License |
|---|---|---|
| System 1 backbone | [facebook/MobileLLM-R1-360M-base](https://huggingface.co/facebook/MobileLLM-R1-360M-base) (Meta) | FAIR Noncommercial Research License. The fine-tuned System 1 checkpoints and `.pte` files are derivatives and carry the same license: noncommercial research use only. |
| Speech to text | Phonon-2.5 English (FermionResearch), a compressed [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0. The Core ML conversion (`export/phonon_coreml.py`) keeps the weights unchanged. |
| Frame grammar | Frame names in `ontology/grammar.json` are derived from [FrameNet](https://framenet.icsi.berkeley.edu) | CC BY 3.0 |
| Test meeting audio (used in the demo) | AMI Meeting Corpus, EN2002a | CC BY 4.0 |
| Dev set conversations (`data/dev/`) | [Switchboard Dialog Act Corpus](https://github.com/cgpotts/swda) | CC BY-NC-SA 3.0 |
| | [Taskmaster-1](https://github.com/google-research-datasets/Taskmaster) (Google) | CC BY 4.0 |
| | [QMSum](https://github.com/Yale-LILY/QMSum) meeting transcripts (AMI, ICSI) | MIT; AMI and ICSI transcripts CC BY 4.0 |
| Dev set fact checklists | written for this project | Apache-2.0, with the conversations' licenses above |
