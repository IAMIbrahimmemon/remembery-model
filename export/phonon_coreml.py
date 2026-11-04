"""Phonon-2.5 English (FermionResearch, CC-BY-4.0; a compressed parakeet-tdt-0.6b-v3) -> Core ML for the iPhone app.

The container is read with Fermion's reference reader (`fermion_container.py`, Apache-2.0, from the model repo) into the
stock transformers ParakeetForTDT. Three Core ML programs are written to `ios/Phonon_build/` and compiled into `ios/Phonon/` (what the app bundles):

- `encoder.mlpackage`: normalized log-mel features [1, F, 128] + valid frame count -> projected encoder states
  [1, T, 640] (T = ceil(F / 8)); one function `f<F>` per size in `FRAMES`, sharing the weights (padding is masked
  inside, so results match an unpadded run).
- `decoder.mlpackage`: one LSTM prediction-network step (token, h, c) -> (d, h, c).
- `joint.mlpackage`: (encoder frame, d) -> 8,193 token + 5 duration logits.

Weights stay exactly what Phonon ships: five-value encoder rows ({0, +-lo, +-hi}) and int6 rows (q * scale) are
stored as per-row lookup tables (3-bit and 6-bit palettes), which reproduce them bit for bit in fp16 (no other
quantization). Mel filters and the tokenizer's pieces go next to the models for the Swift front end; parity
fixtures (audio, features, tokens, text) go to `ios/Phonon_build/parity/` for `memnar-cli phonon`.

    python -m export.phonon_coreml --container model.fermion --base base/ --ref-dir <phonon repo> --wav test.wav
    python -m export.phonon_coreml ... --check-only     # Python: padded encoder + manual greedy loop = generate()
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# feature frames per encoder call (10 ms each): one fixed-shape function per size, since the Neural Engine does not
# compile the variable-length (enumerated-shape) program; the recorder cuts lines at 20 s
FRAMES = [512, 2048]
D = 640


MASK_FILL = -1e4   # finite in fp16: exp(-1e4) is still exactly 0 next to any real score


def _finite_mask_attention():
    """transformers fills masked attention scores with finfo(float32).min, which is -inf in fp16. Padded frames mask
    every key, so their softmax row becomes NaN on fp16 hardware and the next layer spreads it into real frames. Same
    attention with a finite fill: valid frames are unchanged, padded frames stay finite (and are zeroed by the conv)."""
    from transformers.models.parakeet import modeling_parakeet as mp

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        b, n = hidden_states.shape[:-1]
        shape = (b, n, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        heads = self.config.num_attention_heads
        q_u = q + self.bias_u.view(1, heads, 1, self.head_dim)
        q_v = q + self.bias_v.view(1, heads, 1, self.head_dim)
        rk = self.relative_k_proj(position_embeddings).view(b, -1, heads, self.head_dim)
        bd = self._rel_shift(q_v @ rk.permute(0, 2, 3, 1))[..., :n] * self.scaling
        if attention_mask is not None:
            bd = bd.masked_fill(attention_mask.logical_not(), MASK_FILL)
        out, w = mp.eager_attention_forward(self, q_u, k, v, bd, scaling=self.scaling)
        return self.o_proj(out.reshape(b, n, -1).contiguous()), w

    mp.ParakeetEncoderAttention.forward = forward


def load(container: str, base: str, ref_dir: str):
    sys.path.insert(0, ref_dir)
    from reference_transformers import load_model  # Fermion's reference loader (model repo)
    _finite_mask_attention()
    model, processor, info = load_model(container, base)
    return model.float().eval(), processor, info


class Encoder(nn.Module):
    """features [1, F, 128], n_valid [1] (int32) -> encoder_projector(states) [1, T, 640]."""

    def __init__(self, m):
        super().__init__()
        self.enc, self.proj = m.encoder, m.encoder_projector

    def forward(self, features, n_valid):
        mask = (torch.arange(features.shape[1], dtype=torch.int32)[None, :] < n_valid[:, None]).long()
        out = self.enc(input_features=features, attention_mask=mask)
        return self.proj(out.last_hidden_state)


class Decoder(nn.Module):
    """token [1, 1] (int32), h, c [2, 1, 640] -> d [1, 640], h, c."""

    def __init__(self, m):
        super().__init__()
        self.emb, self.lstm, self.proj = m.decoder.embedding, m.decoder.lstm, m.decoder.decoder_projector

    def forward(self, token, h, c):
        x = self.emb(token.long())
        y, (h2, c2) = self.lstm(x, (h, c))
        return self.proj(y[:, 0]), h2, c2


class Joint(nn.Module):
    """enc [1, 640], d [1, 640] -> logits [1, 8198] (8,193 tokens incl. blank, then 5 durations)."""

    def __init__(self, m):
        super().__init__()
        self.act, self.head = m.joint.activation, m.joint.head

    def forward(self, enc, d):
        return self.head(self.act(enc + d))


def greedy(enc: torch.Tensor, dec: Decoder, joint: Joint, cfg) -> tuple[list[int], list[float]]:
    """The loop the app runs (= ParakeetTDTGenerationMixin greedy): returns token ids and each token's probability."""
    blank, V = cfg.blank_token_id, cfg.vocab_size
    durs = list(cfg.durations)
    h = torch.zeros(2, 1, D); c = torch.zeros(2, 1, D)
    d, h, c = dec(torch.tensor([[blank]], dtype=torch.int32), h, c)
    t, T, toks, probs = 0, enc.shape[1], [], []
    for _ in range(cfg.max_symbols_per_step * T):
        if t >= T:
            break
        lg = joint(enc[:, t], d)[0]
        tok = int(lg[:V].argmax()); du = durs[int(lg[V:].argmax())]
        if tok == blank:
            t += max(du, 1)
            continue
        toks.append(tok); probs.append(float(torch.softmax(lg[:V], -1)[tok]))
        t += du
        d, h, c = dec(torch.tensor([[tok]], dtype=torch.int32), h, c)
    return toks, probs


def features(processor, wav: np.ndarray) -> torch.Tensor:
    return processor.feature_extractor([wav], sampling_rate=16000, return_tensors="pt")["input_features"]


def check(model, processor, wavs: list[np.ndarray]):
    enc, dec, joint = Encoder(model), Decoder(model), Joint(model)
    cfg = model.config
    for i, w in enumerate(wavs):
        f = features(processor, w)
        F = f.shape[1]
        P = next(x for x in FRAMES if x >= F)
        with torch.no_grad():
            e0 = enc(f, torch.tensor([F], dtype=torch.int32))
            fp = torch.cat([f, torch.zeros(1, P - F, 128)], 1)
            e1 = enc(fp, torch.tensor([F], dtype=torch.int32))[:, : e0.shape[1]]
            toks, _ = greedy(e1, dec, joint, cfg)
            ref = model.generate(input_features=f).sequences[0].tolist()
        ref = [x for x in ref if x != cfg.blank_token_id]
        text = processor.batch_decode([toks], skip_special_tokens=True)[0]
        print(f"utt {i}: F={F} pad={P} max|enc pad - unpadded|={float((e1 - e0).abs().max()):.2e} "
              f"tokens equal generate(): {toks == ref}  {text[:80]!r}")


def wav_lines(path: str, n: int) -> list[np.ndarray]:
    """Split a 16 kHz wav at silences (the say-rendered test has 0.7 s gaps) into its first n lines."""
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    quiet = np.convolve(np.abs(w) > 1e-3, np.ones(4000), "same") == 0   # 0.25 s of digital silence
    out, start, inside = [], None, False
    for k in range(0, len(w), 160):
        if not quiet[k] and start is None:
            start = k
        if quiet[k] and start is not None:
            out.append(w[start:k]); start = None
            if len(out) == n:
                break
    return out


def convert(model, work: Path):
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    work.mkdir(parents=True, exist_ok=True)
    pal = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        mode="unique", granularity="per_grouped_channel", group_size=1, weight_threshold=2048))
    target = ct.target.iOS18
    t = time.time()
    enc = Encoder(model).eval()
    desc = ct.utils.MultiFunctionDescriptor()
    for f in FRAMES:
        with torch.no_grad():
            ts = torch.jit.trace(enc, (torch.zeros(1, f, 128), torch.tensor([f], dtype=torch.int32)), check_trace=False)
        m = ct.convert(ts, inputs=[ct.TensorType("features", shape=(1, f, 128), dtype=np.float32),
                                   ct.TensorType("n_valid", shape=(1,), dtype=np.int32)],
                       outputs=[ct.TensorType("enc", dtype=np.float32)], minimum_deployment_target=target,
                       compute_precision=ct.precision.FLOAT16, convert_to="mlprogram")
        cto.palettize_weights(m, pal).save(str(work / f"encoder_{f}.mlpackage"))   # identical weights: stored once
        desc.add_function(str(work / f"encoder_{f}.mlpackage"), src_function_name="main", target_function_name=f"f{f}")
    desc.default_function_name = f"f{FRAMES[0]}"
    ct.utils.save_multifunction(desc, str(work / "encoder.mlpackage"))
    import shutil
    for f in FRAMES:
        shutil.rmtree(work / f"encoder_{f}.mlpackage")
    print(f"encoder converted and palettized in {time.time() - t:.0f} s"); t = time.time()
    dec = Decoder(model).eval()
    ex = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(2, 1, D), torch.zeros(2, 1, D))
    with torch.no_grad():
        ts = torch.jit.trace(dec, ex)
    m = ct.convert(ts, inputs=[ct.TensorType("token", shape=(1, 1), dtype=np.int32),
                               ct.TensorType("h", shape=(2, 1, D), dtype=np.float32),
                               ct.TensorType("c", shape=(2, 1, D), dtype=np.float32)],
                   outputs=[ct.TensorType("d", dtype=np.float32), ct.TensorType("h_out", dtype=np.float32),
                            ct.TensorType("c_out", dtype=np.float32)],
                   minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32, convert_to="mlprogram")
    cto.palettize_weights(m, pal).save(str(work / "decoder.mlpackage"))
    jn = Joint(model).eval()
    with torch.no_grad():
        ts = torch.jit.trace(jn, (torch.zeros(1, D), torch.zeros(1, D)))
    m = ct.convert(ts, inputs=[ct.TensorType("enc", shape=(1, D), dtype=np.float32),
                               ct.TensorType("d", shape=(1, D), dtype=np.float32)],
                   outputs=[ct.TensorType("logits", dtype=np.float32)],
                   minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32, convert_to="mlprogram")
    cto.palettize_weights(m, pal).save(str(work / "joint.mlpackage"))
    print(f"decoder + joint done in {time.time() - t:.0f} s")


def verify(model, processor, work: Path, wavs: list[np.ndarray], units: str):
    """Core ML (the three programs, on the given compute units) vs PyTorch fp32, line by line."""
    import coremltools as ct
    cu = getattr(ct.ComputeUnit, units)
    me = {f: ct.models.MLModel(str(work / "encoder.mlpackage"), compute_units=cu, function_name=f"f{f}") for f in FRAMES}
    md = ct.models.MLModel(str(work / "decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    mj = ct.models.MLModel(str(work / "joint.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)

    def dec(token, h, c):
        r = md.predict({"token": token.numpy().astype(np.int32), "h": h.numpy(), "c": c.numpy()})
        # copies: Core ML may hand back its output buffers, which must not be fed back in as inputs
        return (torch.from_numpy(r["d"].copy()), torch.from_numpy(r["h_out"].copy()),
                torch.from_numpy(r["c_out"].copy()))

    def joint(e, d):
        return torch.from_numpy(mj.predict({"enc": e.numpy(), "d": d.numpy()})["logits"].copy())

    same, n, ms, dmax = 0, 0, 0.0, 0.0
    tdec, tjn = Decoder(model), Joint(model)
    for w in wavs:
        f = features(processor, w)
        F = f.shape[1]
        P = next(x for x in FRAMES if x >= F)
        fp = np.concatenate([f.numpy(), np.zeros((1, P - F, 128), np.float32)], 1)
        t = time.time()
        e = torch.from_numpy(me[P].predict({"features": fp, "n_valid": np.array([F], np.int32)})["enc"].astype(np.float32))
        ms += (time.time() - t) * 1000
        with torch.no_grad():
            e0 = Encoder(model)(f, torch.tensor([F], dtype=torch.int32))
            e = e[:, : e0.shape[1]]
            assert not torch.isnan(e).any(), "NaN in the Core ML encoder output"
            dmax = max(dmax, float((e - e0).abs().max()))
            ref, _ = greedy(e0, tdec, tjn, model.config)
        got, _ = greedy(e, dec, joint, model.config)
        n += 1; same += got == ref
        if got != ref:
            print("  torch:  ", processor.batch_decode([ref], skip_special_tokens=True)[0])
            print("  coreml: ", processor.batch_decode([got], skip_special_tokens=True)[0])
    print(f"Core ML ({units}) vs torch fp32: identical tokens {same}/{n} lines, max |enc diff| {dmax:.3f}, "
          f"encoder {ms / n:.0f} ms per line on this Mac")


def detok(pieces: list[str], ids: list[int]) -> str:
    """The app's decoding: pieces joined, the SentencePiece word marker as a space, <...> control tokens dropped."""
    return "".join(pieces[i] for i in ids if not (pieces[i].startswith("<") and pieces[i].endswith(">"))
                   ).replace("\u2581", " ").strip()


def assets(model, processor, base: str, out: Path, work: Path, wavs: list[np.ndarray]):
    """phonon.json (decoding constants, tokenizer pieces), mel_filters.bin, compiled .mlmodelc, parity fixtures."""
    import subprocess
    cfg = model.config
    vocab = json.load(open(Path(base) / "tokenizer.json"))["model"]["vocab"]
    pieces = [""] * len(vocab)
    for p, i in vocab.items():
        pieces[i] = p
    out.mkdir(parents=True, exist_ok=True)
    fe = processor.feature_extractor
    fe.mel_filters.numpy().astype("<f4").tofile(out / "mel_filters.bin")    # [128, 257]
    meta = {"model": "FermionResearch/Phonon-2.5-English", "license": "CC-BY-4.0 (weights), Apache-2.0 (Fermion code)",
            "base_model": "nvidia/parakeet-tdt-0.6b-v3", "blank": cfg.blank_token_id, "vocab_size": cfg.vocab_size,
            "durations": list(cfg.durations), "max_symbols_per_step": cfg.max_symbols_per_step, "frames": FRAMES,
            "subsampling": 8, "d": D, "n_mels": 128, "n_fft": fe.n_fft, "win_length": fe.win_length,
            "hop_length": fe.hop_length, "preemphasis": fe.preemphasis, "log_guard": 2.0 ** -24, "norm_eps": 1e-5,
            "pieces": pieces}
    json.dump(meta, open(out / "phonon.json", "w"))
    for name in ("encoder", "decoder", "joint"):
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(work / f"{name}.mlpackage"), str(out)], check=True,
                       capture_output=True)
    fx, ok = [], 0
    (work / "parity").mkdir(parents=True, exist_ok=True)
    enc, dec, jn = Encoder(model), Decoder(model), Joint(model)
    for i, w in enumerate(wavs):
        f = features(processor, w)
        with torch.no_grad():
            toks, probs = greedy(enc(f, torch.tensor([f.shape[1]], dtype=torch.int32)), dec, jn, cfg)
        text = processor.batch_decode([toks], skip_special_tokens=True)[0].strip()
        ok += detok(pieces, toks) == text
        w.astype("<f4").tofile(work / "parity" / f"{i:02d}.f32")
        fx.append({"audio": f"{i:02d}.f32", "frames": int(f.shape[1]), "features_head": f[0, :4, :8].tolist(),
                   "features_sum": float(f.sum()), "features_abs_sum": float(f.abs().sum()),
                   "tokens": toks, "probs": probs, "text": text})
    json.dump(fx, open(work / "parity" / "parity.json", "w"))
    print(f"assets written; Swift-style detokenization = processor.batch_decode on {ok}/{len(wavs)} lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--base", required=True, help="nvidia/parakeet-tdt-0.6b-v3 config/processor/tokenizer files")
    ap.add_argument("--ref-dir", required=True, help="folder with Fermion's fermion_container.py, reference_transformers.py")
    ap.add_argument("--out", default="ios/Phonon", help="what the app bundles: compiled models, phonon.json, mel filters")
    ap.add_argument("--work", default="ios/Phonon_build", help=".mlpackage sources and parity fixtures (not bundled)")
    ap.add_argument("--wav", help="16 kHz test audio with silent gaps between lines")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--verify", type=int, default=0, help="after converting (or with --skip-convert), compare N lines")
    ap.add_argument("--skip-convert", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    model, processor, info = load(a.container, a.base, a.ref_dir)
    print(f"loaded {info['params']:,} params in {time.time() - t0:.0f} s")
    if a.wav and not a.verify:
        check(model, processor, wav_lines(a.wav, 6))
    if a.check_only:
        return
    if not a.skip_convert:
        convert(model, Path(a.work))
    if a.wav:
        assets(model, processor, a.base, Path(a.out), Path(a.work), wav_lines(a.wav, 71))
    if a.verify:
        wavs = wav_lines(a.wav, a.verify)
        for units in ("CPU_AND_NE",):
            verify(model, processor, Path(a.work), wavs, units)


if __name__ == "__main__":
    main()
