"""System 1 non-autoregressive memory extractor.

backbone (causal decoder, e.g. MobileLLM) -> proj -> bidirectional adapter -> heads:
  span   : every span <= MAX_SPAN tokens inside utterances -> entity type (or NONE)
  coref  : each mention -> NEW or one of the [STATE] slots
  rel    : every ordered entity pair (+ USER) -> multi-label relations
  attrs  : per pair -> modality / tense / source / sensitivity / importance + binary flags
  sal    : window salience
All outputs come from one forward pass. Losses = CE/BCE + lambda * Brier for calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from model.featurize import ATTR2ID, MAX_SPAN
from ontology.ontology import ENTITY_LIST, FLAGS, RELATION_LIST

N_TYPES = len(ENTITY_LIST) + 1
N_REL = len(RELATION_LIST)


def _patched_config(name: str):
    """MobileLLM-Pro keeps rope_theta at top level; newer transformers reads rope_parameters."""
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is None:
            continue
        theta = getattr(c, "rope_theta", None) or (c.to_dict().get("rope_theta"))
        rp = getattr(c, "rope_parameters", None)
        if theta and isinstance(rp, dict) and rp.get("rope_theta") is None:
            scaling = c.to_dict().get("rope_scaling") or {}
            c.rope_parameters = {**rp, **scaling, "rope_theta": theta}
    return cfg


def load_backbone(name: str, dtype=torch.float32):
    """Return the base transformer (no LM head) and its hidden size."""
    cfg = _patched_config(name)
    try:
        m = AutoModel.from_pretrained(name, config=cfg, dtype=dtype, trust_remote_code=True)
    except (ValueError, KeyError):
        m = AutoModelForCausalLM.from_pretrained(name, config=cfg, dtype=dtype, trust_remote_code=True)
    base = getattr(m, "model", m)
    if hasattr(base, "language_model"):
        base = base.language_model
    cfg = getattr(base, "config", m.config)
    hidden = getattr(cfg, "hidden_size", None) or cfg.get_text_config().hidden_size
    return base, hidden


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class NARMemory(nn.Module):
    def __init__(self, backbone_name: str, d: int = 512, adapter_layers: int = 2, dtype=torch.float32,
                 brier_weight: float = 1.0, none_weight: float = 1.0):
        super().__init__()
        self.backbone, hidden = load_backbone(backbone_name, dtype)
        self.proj = nn.Linear(hidden, d)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.adapter = nn.TransformerEncoder(layer, adapter_layers) if adapter_layers > 0 else nn.Identity()
        self.width = nn.Embedding(MAX_SPAN, 64)
        span_dim = 2 * d + 64
        self.span_head = mlp(span_dim, d, N_TYPES)
        self.span_proj = nn.Linear(span_dim, d)
        self.coref_q = nn.Linear(d, d)
        self.coref_k = nn.Linear(d, d)
        self.new_slot = nn.Parameter(torch.randn(d) * 0.02)
        pair_dim = 3 * d
        self.rel_head = mlp(pair_dim, d, N_REL)
        self.attr_heads = nn.ModuleDict({k: mlp(pair_dim, d // 2, len(v)) for k, v in ATTR2ID.items()})
        self.attr_heads["importance"] = mlp(pair_dim, d // 2, 3)
        self.flag_head = mlp(pair_dim, d // 2, len(FLAGS))
        self.sal_head = mlp(d, d // 2, 1)
        self.brier_weight = brier_weight
        self.none_weight = none_weight
        # Per-head temperatures, fit after training (model/calibrate.py).
        self.register_buffer("temps", torch.ones(4))  # span, coref, rel, attrs
        # decode-time Platt scaling fit on (p, correct) of decoded output: span a,b ; rel a,b
        self.register_buffer("platt", torch.tensor([1.0, 0.0, 1.0, 0.0]))

    # ------------------------------------------------------------------ encoder
    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = self.proj(h.float())
        if isinstance(self.adapter, nn.TransformerEncoder):
            h = self.adapter(h, src_key_padding_mask=~attention_mask.bool())
        return h

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def candidate_spans(utt_mask: torch.Tensor):
        """All (start, end) with end-start < MAX_SPAN fully inside utterance tokens."""
        idx = utt_mask.nonzero().flatten().tolist()
        on = set(idx)
        spans = []
        for s in idx:
            for w in range(MAX_SPAN):
                e = s + w
                if e not in on:
                    break
                spans.append((s, e))
        return spans

    def span_reps(self, h, spans):
        st = torch.tensor([s for s, _ in spans], device=h.device)
        en = torch.tensor([e for _, e in spans], device=h.device)
        return torch.cat([h[st], h[en], self.width(en - st)], -1)

    def pair_reps(self, ent):
        n = ent.shape[0]
        a = ent.unsqueeze(1).expand(n, n, -1)
        b = ent.unsqueeze(0).expand(n, n, -1)
        return torch.cat([a, b, a * b], -1)

    # ------------------------------------------------------------------ training
    def loss_one(self, h, ex, n_neg_spans: int = 1024, n_hard: int = 512, n_distract: int = 8):
        dev = h.device
        losses = {}
        gold = {(s, e): t for s, e, t in ex.mention_spans}
        cands = self.candidate_spans(ex.utt_mask)
        # hard negatives: wrong boundaries around a real mention; random negatives: everything else
        hard, rand = [], []
        for c in cands:
            if c in gold:
                continue
            (hard if any(c[0] <= ge and c[1] >= gs for gs, ge in gold) else rand).append(c)
        if len(hard) > n_hard:
            hard = [hard[i] for i in torch.randperm(len(hard))[:n_hard].tolist()]
        if len(rand) > n_neg_spans:
            rand = [rand[i] for i in torch.randperm(len(rand))[:n_neg_spans].tolist()]
        neg = hard + rand
        spans = list(gold) + neg
        if not spans:
            return {}
        reps = self.span_reps(h, spans)
        logits = self.span_head(reps)
        tgt = torch.tensor([gold.get(s, 0) for s in spans], device=dev)
        w = torch.ones(N_TYPES, device=dev)
        w[0] = self.none_weight
        losses["span"] = F.cross_entropy(logits, tgt, weight=w)
        losses["span_brier"] = brier(logits, tgt)
        # distractor "entities" for the relation head: the model's own most entity-like negatives + a few random
        distract = []
        if neg:
            with torch.no_grad():
                p_ent = 1 - logits[len(gold):].float().softmax(-1)[:, 0]
            k = min(n_distract - 2, len(neg))
            top = p_ent.topk(k).indices.tolist() if k > 0 else []
            extra = torch.randperm(len(neg))[:2].tolist()
            distract = [neg[i] for i in dict.fromkeys(top + extra)]

        if ex.mention_spans:
            m_spans = [(s, e) for s, e, _ in ex.mention_spans]
            m = self.span_proj(self.span_reps(h, m_spans))
            # coref against [NEW] + state slots
            slots = torch.cat([self.new_slot[None], h[ex.slot_idx.to(dev)]], 0) if len(ex.slot_idx) else self.new_slot[None]
            cl = self.coref_q(m) @ self.coref_k(slots).T / m.shape[-1] ** 0.5
            ct = torch.tensor(ex.mention_coref, device=dev)
            losses["coref"] = F.cross_entropy(cl, ct)
            # entity reps = mean of mentions; index n = USER
            ent_of = torch.tensor(ex.mention_entity, device=dev)
            ent = torch.zeros(ex.n_entities, m.shape[-1], device=dev, dtype=m.dtype).index_add_(0, ent_of, m)
            ent = ent / torch.bincount(ent_of, minlength=ex.n_entities).clamp(min=1)[:, None]
            parts = [ent]
            if distract:  # zero-label rows/cols: relations must not fire on non-entities
                parts.append(self.span_proj(self.span_reps(h, distract)).to(ent.dtype))
            ent = torch.cat(parts + [h[ex.user_idx][None].to(ent.dtype)], 0)  # USER stays last (index -1)
            n = ent.shape[0]
            pr = self.pair_reps(ent)                                    # n x n x 3d
            rl = self.rel_head(pr)                                      # n x n x R
            rt = torch.zeros(n, n, N_REL, device=dev)
            for (hh, tt), lab in zip(ex.rel_pairs, ex.rel_labels):
                rt[hh, tt] = torch.tensor(lab, device=dev, dtype=torch.float)
            offdiag = ~torch.eye(n, dtype=torch.bool, device=dev)
            losses["rel"] = F.binary_cross_entropy_with_logits(rl[offdiag], rt[offdiag], pos_weight=torch.tensor(2.0, device=dev))
            losses["rel_brier"] = ((torch.sigmoid(rl[offdiag]) - rt[offdiag]) ** 2).mean()
            if ex.rel_pairs:
                hi = torch.tensor([p[0] for p in ex.rel_pairs], device=dev)
                ti = torch.tensor([p[1] for p in ex.rel_pairs], device=dev)
                pp = pr[hi, ti]
                al = 0
                for k, head in self.attr_heads.items():
                    t = torch.tensor([a[k] for a in ex.rel_attrs], device=dev)
                    al = al + F.cross_entropy(head(pp), t)
                ft = torch.tensor([[a[f] for f in FLAGS] for a in ex.rel_attrs], device=dev, dtype=torch.float)
                al = al + F.binary_cross_entropy_with_logits(self.flag_head(pp), ft)
                losses["attrs"] = al / (len(self.attr_heads) + 1)
        pooled = h[ex.utt_mask.to(dev)].mean(0)
        sal = torch.sigmoid(self.sal_head(pooled)).squeeze()
        losses["sal"] = (sal - ex.salience) ** 2
        return losses

    def forward(self, batch):
        """batch: dict(input_ids, attention_mask, examples). Returns (total_loss, parts)."""
        h = self.encode(batch["input_ids"], batch["attention_mask"])
        parts: dict[str, torch.Tensor] = {}
        for i, ex in enumerate(batch["examples"]):
            for k, v in self.loss_one(h[i, : len(ex.input_ids)], ex).items():
                parts[k] = parts.get(k, 0) + v / len(batch["examples"])
        total = sum(v * (self.brier_weight if k.endswith("brier") else 1.0) for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


def brier(logits, target):
    p = logits.softmax(-1)
    oh = F.one_hot(target, p.shape[-1]).float()
    return ((p - oh) ** 2).sum(-1).mean()


def collate(examples, pad_id: int):
    L = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e.input_ids)] = e.input_ids
        mask[i, : len(e.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask, "examples": examples}
