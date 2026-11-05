"""Compare the released English Pocket TTS with the Hinglish models.

Every system reads the same sentences (eval/hinglish_test_sentences.tsv: Hindi,
code-mixed, Indian English, questions, numbers) in the same voices (prompts from
speakers held out of training), at the stock inference settings.

Metrics
  ctc_nll   per-character CTC loss of the romanized input text under the MMS
            aligner, given the generated audio. Lower = the audio says the text.
            Works for romanized Hindi and English alike, which no off-the-shelf
            ASR does.
  cer       greedy MMS transcription vs. the input, both reduced to a
            spelling-insensitive skeleton (aa->a, w->v, ph->f, ...), since
            Hinglish has no standard spelling.
  sim       WavLM speaker-verification cosine to the voice prompt.
  utmos     predicted MOS (if utmos-pytorch is installed).
  fail      share of generations that were empty or never emitted EOS.

    python -m hinglish_tts.evaluate --systems baseline=released:english_2026-09 \
        student=run:/kaggle/working/runs/student --test /data/manifests/test_aligned.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
import torch

logger = logging.getLogger("hinglish.eval")
SR_ASR = 16000


@dataclass
class System:
    name: str
    kind: str  # "released" (a pocket-tts config) or "run" (a training run dir)
    path: str
    cfg: float


def parse_system(spec: str, pocket_tts_dir: Path) -> System:
    """name=released:<language or yaml>[@cfg] | name=run:<run_dir>[@cfg]"""
    name, rest = spec.split("=", 1)
    kind, path = rest.split(":", 1)
    cfg = 1.0
    if "@" in path:
        path, cfg_s = path.rsplit("@", 1)
        cfg = float(cfg_s)
    if kind == "released" and not path.endswith(".yaml"):
        path = str(pocket_tts_dir / "pocket_tts" / "config" / f"{path}.yaml")
    return System(name, kind, path, cfg)


def load_system(system: System, device: torch.device) -> tuple[Any, Any, Any]:
    """(TrainableTTS, Mimi, pocket-tts Config) on `device`, eval mode."""
    from training.args import TrainArgs, load_args
    from training.modules.builders import build_models, load_model_config

    if system.kind == "released":
        args = TrainArgs(model_config=system.path, start_from_pretrained=True)
        model, mimi, config = build_models(args)
    else:
        from training.eval.librispeech import load_run

        model, mimi, _step = load_run(system.path, device, use_ema=True)
        run_args = load_args(Path(system.path) / "args.yaml")
        config = load_model_config(run_args.model_config, run_args.model_overrides)
    return model.to(device).eval(), mimi.to(device).eval(), config


def load_wav(path: str, sr: int, max_sec: float | None = None) -> torch.Tensor:
    from .data.audio import resample

    wav, in_sr = soundfile.read(path, dtype="float32", always_2d=True)
    wav = resample(wav.mean(axis=1), int(in_sr), sr)
    if max_sec:
        wav = wav[: int(max_sec * sr)]
    return torch.from_numpy(np.ascontiguousarray(wav))


def pick_voices(test_manifest: Path, n: int, max_prompt_sec: float = 6.0) -> list[dict[str, Any]]:
    """Up to n held-out speakers (alternating sources/genders), one clean prompt each.

    The prompt ends at a word boundary at most `max_prompt_sec` in, so no voice
    starts the generation mid-word.
    """
    rows = [json.loads(l) for l in test_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Manifests carry only the README keys; source/speaker/gender live in clips_metadata.jsonl.
    meta_path = test_manifest.parent / "clips_metadata.jsonl"
    if meta_path.exists():
        meta = {m["path"]: m for m in map(json.loads, meta_path.read_text(encoding="utf-8").splitlines()) if m}
        rows = [{**meta.get(r["path"], {}), **r} for r in rows]
    for r in rows:
        r.setdefault("source", "test")
        r.setdefault("speaker", Path(r["path"]).stem)
    by_speaker: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: (r.get("align_nll", 9), -r["duration"])):
        if r["duration"] >= 4.0 and r["speaker"] not in by_speaker:
            by_speaker[r["speaker"]] = r
    pools: dict[tuple[str, str | None], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in by_speaker.values():
        pools[(r["source"], r.get("gender"))].append(r)
    voices: list[dict[str, Any]] = []
    while len(voices) < n and any(pools.values()):
        for key in sorted(pools, key=str):
            if pools[key] and len(voices) < n:
                voices.append(pools[key].pop(0))
    for v in voices:
        ends = [w["end"] for w in v.get("words", []) if w.get("end") and w["end"] <= max_prompt_sec]
        v["prompt_sec"] = max(ends) + 0.1 if ends else min(v["duration"], max_prompt_sec)
        v["voice_id"] = re.sub(r"[^A-Za-z0-9]+", "_", v["speaker"])[-24:]
    return voices


def skeleton(text: str) -> str:
    """Spelling-insensitive form of romanized text for CER."""
    s = re.sub(r"[^a-z]", "", text.lower())
    for a, b in (("ph", "f"), ("w", "v"), ("z", "j"), ("q", "k"), ("ck", "k"), ("x", "ks"), ("ee", "i"), ("oo", "u")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Scorer:
    def __init__(self, device: torch.device, with_sim: bool = True, with_utmos: bool = True):
        from .align import MMSAligner, word_tokens

        self.device = device
        self.word_tokens = word_tokens
        self.aligner = MMSAligner(device)
        self.vocab, self.blank = self.aligner.vocab, self.aligner.blank
        self.id2char = {i: c for c, i in self.vocab.items() if len(c) == 1 and (c.isalpha() or c == "'")}
        self.sim_fe = self.sim_model = self.utmos = None
        if with_sim:
            try:
                from transformers import AutoFeatureExtractor, WavLMForXVector

                self.sim_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
                self.sim_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()  # ty: ignore[invalid-argument-type]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"speaker similarity disabled: {exc}")
        if with_utmos:
            try:
                from utmos_pytorch import UTMOSScoreTorch

                self.utmos = UTMOSScoreTorch(device=str(device))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"UTMOS disabled: {exc}")

    @torch.no_grad()
    def intelligibility(self, wav16: torch.Tensor, text: str) -> tuple[float, float, str]:
        all_log_probs, frames = self.aligner.emissions([wav16.numpy()])
        log_probs = all_log_probs[0, : frames[0]]
        tokens, _, _ = self.word_tokens(text.split(), self.vocab)
        T = log_probs.shape[0]
        nll = torch.nn.functional.ctc_loss(
            log_probs[:, None], torch.tensor([tokens], device=self.device), torch.tensor([T]),
            torch.tensor([len(tokens)]), blank=self.blank, reduction="sum", zero_infinity=True,
        ).item() / max(1, len(tokens))  # fmt: skip
        ids = log_probs.argmax(-1).tolist()
        hyp = "".join(self.id2char.get(i, "") for k, i in enumerate(ids) if i != self.blank and (k == 0 or i != ids[k - 1]))
        ref_s, hyp_s = skeleton(text), skeleton(hyp)
        return nll, edit_distance(hyp_s, ref_s) / max(1, len(ref_s)), hyp

    @torch.no_grad()
    def similarity(self, a16: torch.Tensor, b16: torch.Tensor) -> float | None:
        if self.sim_model is None:
            return None
        inputs = self.sim_fe([a16.numpy(), b16.numpy()], sampling_rate=SR_ASR, return_tensors="pt", padding=True).to(self.device)
        emb = self.sim_model(**inputs).embeddings
        return float(torch.nn.functional.cosine_similarity(emb[0:1], emb[1:2]).item())

    @torch.no_grad()
    def mos(self, wav16: torch.Tensor) -> float | None:
        if self.utmos is None:
            return None
        return float(self.utmos.score(wav16[None, None].to(self.device)))


@torch.no_grad()
def synthesize(
    model: Any, mimi: Any, config: Any, texts: list[str], voices: list[torch.Tensor],
    cfg: float, temp: float, eos_threshold: float, max_sec: float, device: torch.device,
) -> list[tuple[np.ndarray | None, bool]]:  # fmt: skip
    """Batched generation with pocket-tts' text preparation. Returns (wav, hit_max_len)."""
    from pocket_tts.models.text_chunking import prepare_text_prompt
    from training.eval.librispeech import latents_to_wav

    tok = model.flow_lm.conditioner.tokenizer
    tokens = []
    for t in texts:
        prepared, _ = prepare_text_prompt(
            t, config.pad_with_spaces_for_short_inputs, config.remove_semicolons,
            config.append_terminal_punctuation, config.capitalize_first_letter, config.replace_characters,
        )  # fmt: skip
        tokens.append(torch.tensor(tok.encode(prepared), dtype=torch.long))
    voice_latents = [mimi.encode_to_latent(v[None, None].to(device))[0] for v in voices]
    max_frames = int(max_sec * mimi.frame_rate)
    outs = model.generate(tokens, voice_latents, max_frames=max_frames, temp=temp, n_steps=1, cfg_coef=cfg, eos_threshold=eos_threshold)
    results = []
    for lat in outs:
        wav = latents_to_wav(mimi, lat, device)
        results.append((None if wav is None else wav.float().cpu().numpy(), lat.shape[0] >= max_frames))
    return results


def evaluate(
    systems: list[System], test_manifest: Path, sentences_tsv: Path, out_dir: Path, n_voices: int = 4,
    temp: float = 0.3, eos_threshold: float = -4.0, max_sec: float = 20.0, batch_size: int = 16,
    seed: int = 0, with_sim: bool = True, with_utmos: bool = True,
) -> dict[str, Any]:  # fmt: skip
    from .data.audio import resample
    from .text import HinglishFrontend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sentences_tsv, encoding="utf-8") as f:
        sentences = [(r["category"], r["text"]) for r in csv.DictReader(f, delimiter="\t")]
    frontend = HinglishFrontend()
    voices = pick_voices(test_manifest, n_voices)
    if not voices:
        raise ValueError(f"no usable voice prompts in {test_manifest}")
    prompts24 = [load_wav(v["path"], 24000, v["prompt_sec"]) for v in voices]
    prompts16 = [torch.from_numpy(resample(p.numpy(), 24000, SR_ASR)) for p in prompts24]
    (out_dir / "voices").mkdir(exist_ok=True)
    for v, p in zip(voices, prompts24, strict=True):
        soundfile.write(out_dir / "voices" / f"{v['voice_id']}.wav", p.numpy(), 24000)
    items = [(vi, si) for vi in range(len(voices)) for si in range(len(sentences))]
    logger.info(f"{len(voices)} voices x {len(sentences)} sentences = {len(items)} items per system")

    generated: dict[str, list[tuple[np.ndarray | None, bool]]] = {}
    timing: dict[str, float] = {}
    for system in systems:
        torch.manual_seed(seed)
        model, mimi, config = load_system(system, device)
        t0, audio_sec = time.time(), 0.0
        gens: list[tuple[np.ndarray | None, bool]] = []
        for b in range(0, len(items), batch_size):
            chunk = items[b : b + batch_size]
            texts = [frontend(sentences[si][1]) for _, si in chunk]
            gens += synthesize(model, mimi, config, texts, [prompts24[vi] for vi, _ in chunk],
                               system.cfg, temp, eos_threshold, max_sec, device)  # fmt: skip
        for (vi, si), (wav, _) in zip(items, gens, strict=True):
            if wav is not None:
                audio_sec += len(wav) / 24000
                d = out_dir / system.name
                d.mkdir(exist_ok=True)
                soundfile.write(d / f"{voices[vi]['voice_id']}_{si:02d}.wav", wav, 24000)
        timing[system.name] = (time.time() - t0) / max(audio_sec, 1e-6)
        generated[system.name] = gens
        logger.info(f"{system.name}: {len(gens)} generations, GPU real-time factor {timing[system.name]:.3f}")
        del model, mimi
        torch.cuda.empty_cache() if device.type == "cuda" else None

    scorer = Scorer(device, with_sim, with_utmos)
    records = []
    for system in systems:
        for (vi, si), (wav, capped) in zip(items, generated[system.name], strict=True):
            cat, raw = sentences[si]
            text = frontend(raw)
            rec: dict[str, Any] = {"system": system.name, "voice": voices[vi]["voice_id"], "idx": si, "category": cat, "text": text}
            if wav is None or len(wav) < 2400:
                rec.update(fail=1)
                records.append(rec)
                continue
            w16 = torch.from_numpy(resample(wav, 24000, SR_ASR))
            nll, cer, hyp = scorer.intelligibility(w16, text)
            rec.update(fail=int(capped), ctc_nll=nll, cer=cer, hyp=hyp, sec=len(wav) / 24000,
                       sim=scorer.similarity(w16, prompts16[vi]), utmos=scorer.mos(w16))  # fmt: skip
            records.append(rec)

    def agg(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        out: dict[str, float | None] = {"n": len(rows), "fail": float(np.mean([r.get("fail", 0) for r in rows])) if rows else None}
        for k in ("ctc_nll", "cer", "sim", "utmos"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        return out

    summary: dict[str, Any] = {"voices": [v["voice_id"] for v in voices], "systems": {}}
    for system in systems:
        rows = [r for r in records if r["system"] == system.name]
        summary["systems"][system.name] = {
            "overall": agg(rows),
            "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in sorted({s[0] for s in sentences})},
            "gpu_rtf": timing[system.name],
            "cfg": system.cfg,
            "source": f"{system.kind}:{system.path}",
        }
    (out_dir / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "results.md").write_text(results_markdown(summary))
    write_listening_page(out_dir, systems, voices, sentences, frontend)
    return summary


def results_markdown(summary: dict[str, Any]) -> str:
    def f(v: float | None, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    lines = [
        "| system | CTC NLL (lower better) | CER (lower better) | speaker sim (higher better) | UTMOS (higher better) | failed |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in summary["systems"].items():
        o = s["overall"]
        lines.append(f"| {name} | {f(o['ctc_nll'], '.3f')} | {f(o['cer'], '.3f')} | {f(o['sim'], '.3f')} | {f(o['utmos'], '.2f')} | {f(o['fail'], '.1%')} |")
    lines += ["", "CER by category:", "", "| system | " + " | ".join(next(iter(summary["systems"].values()))["by_category"]) + " |"]
    lines.append("|---" * (1 + len(next(iter(summary["systems"].values()))["by_category"])) + "|")
    for name, s in summary["systems"].items():
        lines.append(f"| {name} | " + " | ".join(f(c["cer"], ".3f") for c in s["by_category"].values()) + " |")
    return "\n".join(lines) + "\n"


def write_listening_page(out_dir: Path, systems: list[System], voices: list[dict[str, Any]], sentences: list[tuple[str, str]], frontend: Any) -> None:
    head = "".join(f"<th>{html.escape(s.name)}</th>" for s in systems)
    rows = []
    for vi, v in enumerate(voices[:2]):
        for si, (cat, raw) in enumerate(sentences):
            cells = "".join(
                f'<td><audio controls preload="none" src="{s.name}/{v["voice_id"]}_{si:02d}.wav"></audio></td>' for s in systems
            )
            rows.append(f"<tr><td><b>{html.escape(cat)}</b><br>{html.escape(frontend(raw))}<br>"
                        f'<small>voice: <audio controls preload="none" src="voices/{v["voice_id"]}.wav"></audio></small></td>{cells}</tr>')  # fmt: skip
    page = (
        "<!doctype html><meta charset='utf-8'><title>Hinglish Pocket TTS - listening test</title>"
        "<style>body{font-family:system-ui;margin:2em}td,th{border-bottom:1px solid #ddd;padding:6px;vertical-align:top}"
        "audio{width:210px}</style><h1>Hinglish Pocket TTS - listening test</h1>"
        f"<table><tr><th>sentence</th>{head}</tr>{''.join(rows)}</table>"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", required=True, help="name=released:<lang|yaml>[@cfg] or name=run:<dir>[@cfg]")
    ap.add_argument("--pocket-tts-dir", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True, help="test_aligned.jsonl (voice prompts)")
    ap.add_argument("--sentences", type=Path, default=Path(__file__).resolve().parent.parent / "eval" / "hinglish_test_sentences.tsv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voices", type=int, default=4)
    ap.add_argument("--max-sentences", type=int, default=0, help="debug: use only the first N sentences")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-utmos", action="store_true")
    args = ap.parse_args()
    systems = [parse_system(s, args.pocket_tts_dir) for s in args.systems]
    sentences = args.sentences
    if args.max_sentences:
        lines = sentences.read_text(encoding="utf-8").splitlines()[: args.max_sentences + 1]
        sentences = args.out / "_sentences.tsv"
        args.out.mkdir(parents=True, exist_ok=True)
        sentences.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = evaluate(systems, args.test, sentences, args.out, args.voices, with_sim=not args.no_sim, with_utmos=not args.no_utmos)
    print(results_markdown(summary))


if __name__ == "__main__":
    main()
