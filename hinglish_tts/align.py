"""Word-level forced alignment for romanized Hinglish with torchaudio's MMS_FA, plus transcript QA.

The pocket-tts loader cuts every utterance at a word boundary (prompt before,
target after), so it needs word timestamps. MMS_FA (Pratap et al., 2023) is a
CTC aligner trained on 1,130 languages over the uroman alphabet (a-z and '):
romanized Hindi and English words are both plain letter sequences to it, which
is exactly our transcript format.

Details that matter:
  * Batched emissions. The bundle's model wrapper layer-normalizes the *whole*
    padded batch at once, which mixes utterances; we call the inner wav2vec2
    with per-utterance normalization instead (identical to batch size 1).
  * fp16 on GPUs without bf16 (T4).
  * `torchaudio.functional.forced_align` + `merge_tokens` per utterance give
    token spans, grouped into words.
  * A per-character CTC loss for every clip: clips whose transcript does not
    match the audio (the most damaging data error for TTS) score high and are
    dropped by `hinglish_tts.data.prepare`.

If torchaudio lacks its alignment extension, the same weights from the Hub
(MahmoudAshraf/mms-300m-1130-forced-aligner) and Kyutai's Viterbi are used.

    python -m hinglish_tts.align --in clips.jsonl --out aligned.jsonl [--shard 0 --num-shards 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
from pathlib import Path

import numpy as np
import soundfile
import torch

logger = logging.getLogger("hinglish.align")

HF_FALLBACK = "MahmoudAshraf/mms-300m-1130-forced-aligner"


def word_tokens(words: list[str], vocab: dict[str, int]) -> tuple[list[int], list[int], list[str]]:
    """Character ids (no word delimiter) and the word index of each id.

    Returns (tokens, word_of, normalized words); words with no alignable letter
    (pure punctuation) normalize to "" and get no timestamps.
    """
    tokens: list[int] = []
    word_of: list[int] = []
    norm: list[str] = []
    k = 0
    for w in words:
        chars = [c for c in w.lower().replace("’", "'") if c in vocab and len(c) == 1 and c.isascii() and (c.isalpha() or c == "'")]
        norm.append("".join(chars))
        if not chars:
            continue
        for c in chars:
            tokens.append(vocab[c])
            word_of.append(k)
        k += 1
    return tokens, word_of, norm


class MMSAligner:
    """MMS forced aligner: torchaudio MMS_FA, or its Hub port as a fallback."""

    sample_rate = 16000

    def __init__(self, device: torch.device, backend: str = "auto"):
        self.device = device
        self.backend = ""
        if backend in ("auto", "torchaudio"):
            try:
                self._init_torchaudio()
            except Exception as exc:  # noqa: BLE001
                if backend == "torchaudio":
                    raise
                logger.warning(f"torchaudio MMS_FA unavailable ({exc!r}); using the Hub port {HF_FALLBACK}")
        if not self.backend:
            self._init_hf()
        if device.type == "cuda":
            self.net = self.net.half()
        self.dtype = next(self.net.parameters()).dtype
        logger.info(f"aligner: {self.backend} on {device} ({self.dtype})")

    def _init_torchaudio(self) -> None:
        import torchaudio.functional as taF
        from torchaudio.pipelines import MMS_FA

        # Probe the compiled alignment op before downloading 1.2 GB of weights.
        taF.forced_align(torch.zeros(1, 4, 3).log_softmax(-1), torch.tensor([[1]], dtype=torch.int32), blank=0)
        bundle_model = MMS_FA.get_model(with_star=False)
        self.net = bundle_model.model.to(self.device).eval()  # inner Wav2Vec2Model: no batch-wide layer_norm
        self.vocab = MMS_FA.get_dict(star=None)
        self.blank = self.vocab["-"]
        self.sample_rate = MMS_FA.sample_rate
        self.backend = "torchaudio-MMS_FA"

    def _init_hf(self) -> None:
        import transformers
        from transformers import AutoTokenizer, Wav2Vec2ForCTC

        transformers.logging.set_verbosity_error()
        self.net = Wav2Vec2ForCTC.from_pretrained(HF_FALLBACK).to(self.device).eval()  # ty: ignore[invalid-argument-type]
        self.vocab = AutoTokenizer.from_pretrained(HF_FALLBACK).get_vocab()
        self.blank = self.vocab["<blank>"]
        self.backend = "hf-port"

    @torch.no_grad()
    def emissions(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        """Log-probs [B, T, V] (fp32) and valid frame counts for 16 kHz mono waves."""
        lens = [len(w) for w in wavs]
        x = torch.zeros(len(wavs), max(lens))
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w)
            x[i, : lens[i]] = (t - t.mean()) / torch.sqrt(t.var(unbiased=False) + 1e-5)  # = layer_norm per utterance
        x = x.to(self.device, self.dtype)
        lengths = torch.tensor(lens, device=self.device)
        if self.backend.startswith("torchaudio"):
            out, out_lens = self.net(x, lengths)
            frames = out_lens.tolist() if out_lens is not None else [out.shape[1]] * len(wavs)
        else:
            mask = (torch.arange(x.shape[1], device=self.device)[None] < lengths[:, None]).long()
            out = self.net(x, attention_mask=mask).logits
            frames = [int(self.net._get_feat_extract_output_lengths(n)) for n in lens]
        return out.float().log_softmax(-1), [int(f) for f in frames]

    def word_spans(self, log_probs: torch.Tensor, tokens: list[int], word_of: list[int]) -> list[tuple[int, int]] | None:
        """[start, end) frame span of each alignable word, or None if unalignable."""
        if self.backend.startswith("torchaudio"):
            import torchaudio.functional as taF

            try:
                path, scores = taF.forced_align(
                    log_probs[None], torch.tensor([tokens], dtype=torch.int32, device=log_probs.device), blank=self.blank
                )
            except Exception:  # noqa: BLE001 -- e.g. transcript longer than the audio allows
                return None
            spans = taF.merge_tokens(path[0], scores[0].exp(), blank=self.blank)
            if len(spans) != len(tokens):
                return None
            words: dict[int, tuple[int, int]] = {}
            for span, w in zip(spans, word_of, strict=True):
                s, e = words.get(w, (span.start, span.end))
                words[w] = (min(s, span.start), max(e, span.end))
            return [words[i] for i in range(max(word_of) + 1)]
        from training.scripts.align_data import batched_word_spans  # pocket-tts checkout

        T = torch.tensor([log_probs.shape[0]], device=log_probs.device)
        res = batched_word_spans(log_probs[None], T, [tokens], [word_of], self.blank)[0]
        return None if res is None or any(r is None for r in res) else [(s, e + 1) for s, e in res]  # type: ignore[misc]


def _load_16k(path: str, sr_out: int) -> np.ndarray:
    from .data.audio import resample

    wav, sr = soundfile.read(path, dtype="float32", always_2d=True)
    return resample(wav.mean(axis=1), int(sr), sr_out)


@torch.no_grad()
def align_manifest(
    in_path: Path, out_path: Path, device: str | None = None, batch_size: int = 16, shard: int = 0,
    num_shards: int = 1, backend: str = "auto",
) -> tuple[int, int]:  # fmt: skip
    """Align the shard-th slice of `in_path` into `out_path`.

    Every input row is written back with "words" ([{"word","start","end"}],
    None times for punctuation-only tokens) and "align_nll" (per-character CTC
    loss), or with "align_error" when it could not be aligned.
    Returns (aligned, failed).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    aligner = MMSAligner(dev, backend)
    lines = [l for i, l in enumerate(in_path.read_text(encoding="utf-8").splitlines()) if l.strip() and i % num_shards == shard]
    entries = [json.loads(l) for l in lines]
    order = sorted(range(len(entries)), key=lambda i: entries[i]["duration"])  # little padding per batch
    q: queue.Queue[tuple[int, np.ndarray | Exception] | None] = queue.Queue(maxsize=4 * batch_size)

    def reader() -> None:
        for i in order:
            try:
                q.put((i, _load_16k(entries[i]["path"], aligner.sample_rate)))
            except Exception as exc:  # noqa: BLE001
                q.put((i, exc))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    ok = failed = 0
    done = False
    while not done:
        batch = []
        while len(batch) < batch_size:
            item = q.get()
            if item is None:
                done = True
                break
            i, wav = item
            words = entries[i]["transcript"].split()
            tokens, word_of, norm = word_tokens(words, aligner.vocab)
            if isinstance(wav, Exception) or not tokens:
                entries[i]["align_error"] = "unreadable_audio" if isinstance(wav, Exception) else "no_alignable_words"
                failed += 1
                continue
            batch.append((i, wav, tokens, word_of, words, norm))
        if not batch:
            continue
        log_probs, frames = aligner.emissions([b[1] for b in batch])
        targets = torch.cat([torch.tensor(b[2]) for b in batch]).to(dev)
        tlens = torch.tensor([len(b[2]) for b in batch], device=dev)
        nll = (torch.nn.functional.ctc_loss(log_probs.transpose(0, 1), targets, torch.tensor(frames, device=dev), tlens,
                                            blank=aligner.blank, reduction="none", zero_infinity=True) / tlens).tolist()  # fmt: skip
        for k, (i, wav, tokens, word_of, words, norm) in enumerate(batch):
            spans = aligner.word_spans(log_probs[k, : frames[k]], tokens, word_of)
            if spans is None or not np.isfinite(nll[k]) or nll[k] == 0.0:
                entries[i]["align_error"] = "no_ctc_path"
                failed += 1
                continue
            sec_per_frame = len(wav) / aligner.sample_rate / frames[k]
            timed, w_idx = [], 0
            for w, nw in zip(words, norm, strict=True):
                if not nw:
                    timed.append({"word": w, "start": None, "end": None})
                    continue
                s, e = spans[w_idx]
                w_idx += 1
                timed.append({"word": w, "start": round(s * sec_per_frame, 3), "end": round(e * sec_per_frame, 3)})
            entries[i]["words"] = timed
            entries[i]["align_nll"] = round(float(nll[k]), 4)
            ok += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:  # input order
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return ok, failed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backend", choices=["auto", "torchaudio", "hf"], default="auto")
    args = ap.parse_args()
    ok, bad = align_manifest(args.inp, args.out, None, args.batch_size, args.shard, args.num_shards, args.backend)
    logger.info(f"{args.out}: {ok} aligned, {bad} failed")


if __name__ == "__main__":
    main()
