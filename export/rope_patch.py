"""Real-valued rotary embeddings for export (ExecuTorch can't lower complex tensors).

Llama4/MobileLLM compute RoPE as x * e^{i*theta} in complex numbers. Here the same rotation is
done with cos/sin on (even, odd) pairs, which is numerically equivalent.
"""

from __future__ import annotations

import sys

import torch
from transformers.models.llama4 import modeling_llama4 as L4


def _rotary_forward(self, x, position_ids):
    inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    freqs = (inv.to(x.device) @ position_ids[:, None, :].float()).transpose(1, 2)
    return torch.stack([freqs.cos(), freqs.sin()], -1) * self.attention_scaling  # (B, T, D/2, 2)


def _apply_rotary_emb(xq, xk, cs):
    c, s = cs[..., 0][:, :, None, :], cs[..., 1][:, :, None, :]

    def rot(x):
        x2 = x.float().reshape(*x.shape[:-1], -1, 2)
        a, b = x2[..., 0], x2[..., 1]
        return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(3).type_as(x)
    return rot(xq), rot(xk)


def apply():
    L4.Llama4TextRotaryEmbedding.forward = _rotary_forward
    L4.apply_rotary_emb = _apply_rotary_emb
    for name, mod in list(sys.modules.items()):  # custom remote code (MobileLLM-Pro) imports these names
        if "mobilellm" in name.lower():
            if hasattr(mod, "apply_rotary_emb"):
                mod.apply_rotary_emb = _apply_rotary_emb
