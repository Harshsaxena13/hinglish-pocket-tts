"""Prepare the Hinglish TTS corpus: clips -> splits -> alignment -> QA -> manifests + STATS.md.

    python -m hinglish_tts.data.prepare --out /kaggle/temp/hinglish/data            # everything
    python -m hinglish_tts.data.prepare --out ... --stage clips                     # download + process + split
    python -m hinglish_tts.data.prepare --out ... --stage align --gpus 2            # align + QA + report

Stage "clips" (see build.py for the per-clip checks):
    Devanagari -> natural Roman Hinglish, English loanwords back to English,
    digits spelled out, bad transcripts dropped; 24 kHz mono (Mimi's rate),
    silence trimmed, loudness normalized, 1-15 s. Then a 2% test split
    (speaker-disjoint when a source has many speakers) and a small valid split.
Stage "align":
    word timestamps with torchaudio MMS_FA, then QA: unaligned clips and train
    clips whose transcript does not match the audio (CTC loss outliers) are
    dropped; train clips too short to cut into prompt + target (< 1 s either
    side of every word boundary) are kept out of training only.

Outputs in <out>/manifests/, in the format the pocket-tts training README requires
(one JSON object per line, exactly these keys):
    {"path": ".../clip.flac", "duration": 6.42, "transcript": "Aap kaise ho?",
     "words": [{"word": "Aap", "start": 0.31, "end": 0.52}, ...]}
    train_aligned.jsonl  valid_aligned.jsonl  test_aligned.jsonl
plus clips_metadata.jsonl (source, speaker, raw text, alignment score, drop reason
for every clip) and <out>/STATS.md + stats.json (hours per source, kept/dropped and why).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .build import plan_jobs, run_shards, stable_unit

logger = logging.getLogger("hinglish.prepare")

PROJECT_DIR = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
README_KEYS = ("path", "duration", "transcript", "words")
MIN_CUT_SEC = 1.0  # pocket-tts DataLoader.MIN_CUT_SEC: audio needed on both sides of a cut
REASONS = {
    "bad_transcript": "other script, markup, or fewer than 3 words after cleanup",
    "too_short": "shorter than min_sec after silence trimming",
    "too_long": "longer than max_sec after silence trimming",
    "too_quiet": "active speech below -45 dBFS",
    "implausible_speaking_rate": "letters per second outside 4-28: transcript cannot match the audio",
    "undecodable_audio": "missing or corrupt audio",
    "unreadable_audio": "clip written but could not be read back",
    "no_alignable_words": "no letters the aligner knows",
    "no_ctc_path": "forced alignment found no path (transcript longer than the audio allows)",
    "low_alignment_coverage": "fewer than 80% of the words got timestamps",
    "ctc_mismatch": "per-character CTC loss in the worst tail: transcript likely does not match the audio",
    "no_prompt_cut": "(train only) no word boundary with >= 1 s of audio on both sides, so no prompt/target cut",
    "shard_failed": "a whole parquet shard failed to download or parse (counted once per shard)",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- splits


def split_clips(rows: list[dict[str, Any]], recipe: dict[str, Any]) -> dict[str, str]:
    """path -> split. Test: `test_frac` of every source (default 2%), whole speakers
    when the source has >= 20 of them so the test voices are unseen; valid: a small
    random slice of the rest for the trainer's validation loss."""
    seed = recipe.get("seed", 0)
    test_frac = float(recipe.get("test_frac", 0.02))
    split: dict[str, str] = {}
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for src, items in sorted(by_source.items()):
        n_test = max(1, round(test_frac * len(items))) if len(items) >= 10 else 0
        speakers: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in items:
            speakers[r["speaker"]].append(r)
        chosen: list[dict[str, Any]] = []
        if len(speakers) >= 20:
            for spk in sorted(speakers, key=lambda s: stable_unit(seed, "test-speaker", s)):
                if len(chosen) >= n_test:
                    break
                if len(chosen) + len(speakers[spk]) <= 1.5 * n_test:  # skip speakers that would overshoot
                    chosen += speakers[spk]
        if len(chosen) < n_test:  # few speakers (or only big ones): random clips
            taken = {r["path"] for r in chosen}
            rest = sorted((r for r in items if r["path"] not in taken), key=lambda r: stable_unit(seed, "test", r["path"]))
            chosen += rest[: n_test - len(chosen)]
        for r in chosen:
            split[r["path"]] = "test"
    remaining = sorted((r for r in rows if r["path"] not in split), key=lambda r: stable_unit(seed, "valid", r["path"]))
    n_valid = int(np.clip(round(float(recipe.get("valid_frac", 0.01)) * len(remaining)),
                          int(recipe.get("valid_min", 64)), int(recipe.get("valid_max", 300))))  # fmt: skip
    for i, r in enumerate(remaining):
        split[r["path"]] = "valid" if i < n_valid else "train"
    return split


# ----------------------------------------------------------------------------- stage: clips


def prepare_clips(recipe_path: Path, out: Path, lexicon: str | None = None, workers: int | None = None,
                  hours_scale: float = 1.0, only: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
    recipe = yaml.safe_load(recipe_path.read_text())
    if hours_scale != 1.0:
        for src in recipe["sources"]:
            src["max_hours"] = float(src["max_hours"]) * hours_scale
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(recipe, out, lexicon, only)
    logger.info(f"{len(jobs)} shards from {len({j.source['name'] for j in jobs})} sources")
    results = run_shards(jobs, workers or int(recipe.get("workers", 3)))
    rows = [r for shard in sorted((out / "shards").glob("*.jsonl")) for r in _read_jsonl(shard)]
    split = split_clips(rows, recipe)
    for r in rows:
        r["split"] = split[r["path"]]
    _write_jsonl(out / "manifests" / "clips.jsonl", rows)
    clip_stats: dict[str, Any] = {"sources": {}, "recipe": str(recipe_path), "lexicon": lexicon}
    for res in results:
        s = clip_stats["sources"].setdefault(res.source, {"seen": 0, "kept": 0, "rejected": collections.Counter()})
        s["seen"] += res.seen
        s["kept"] += res.kept
        s["rejected"].update(res.rejected)
    (out / "clip_stats.json").write_text(json.dumps(clip_stats, indent=2, default=dict))
    write_stats(out)
    counts = collections.Counter(r["split"] for r in rows)
    hours = round(sum(r["duration"] for r in rows) / 3600, 2)
    logger.info(f"clips: {dict(counts)}, {hours} h")
    return {"clips": dict(counts), "hours": hours}


# ----------------------------------------------------------------------------- stage: align


def _has_prompt_cut(words: list[dict[str, Any]], duration: float) -> bool:
    """Same rule as the pocket-tts DataLoader: some boundary with >= 1 s on both sides."""
    for prev, cur in zip(words, words[1:]):
        if prev["end"] is None or cur["start"] is None:
            continue
        cut = 0.5 * (prev["end"] + cur["start"])
        if cut >= MIN_CUT_SEC and duration - cut >= MIN_CUT_SEC:
            return True
    return False


def align_and_finalize(out: Path, gpus: int, env: dict[str, str] | None = None,
                       drop_frac: float = 0.03, max_nll: float = 2.5) -> dict[str, Any]:  # fmt: skip
    from ..align import align_manifest

    man = out / "manifests"
    clips = man / "clips.jsonl"
    n = max(1, gpus)
    parts = [man / f"_aligned_part{i}.jsonl" for i in range(n)]
    if n == 1:
        align_manifest(clips, parts[0])
    else:
        procs = []
        for i in range(n):
            cmd = [sys.executable, "-m", "hinglish_tts.align", "--in", str(clips), "--out", str(parts[i]),
                   "--shard", str(i), "--num-shards", str(n)]  # fmt: skip
            procs.append(subprocess.Popen(cmd, env={**(env or os.environ), "CUDA_VISIBLE_DEVICES": str(i)}, cwd=PROJECT_DIR))
        if any(p.wait() != 0 for p in procs):
            raise RuntimeError("an alignment worker failed")
    rows = [r for p in parts for r in _read_jsonl(p)]

    train_nll = np.array([r["align_nll"] for r in rows if r["split"] == "train" and "align_nll" in r])
    threshold = min(max_nll, float(np.quantile(train_nll, 1 - drop_frac))) if len(train_nll) else max_nll
    manifests: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in rows:
        reason = r.pop("align_error", None)
        if reason is None:
            coverage = sum(w["start"] is not None for w in r["words"]) / max(1, len(r["words"]))
            if coverage < 0.8:
                reason = "low_alignment_coverage"
            elif r["split"] == "train" and r["align_nll"] > threshold:
                reason = "ctc_mismatch"
            elif r["split"] == "train" and not _has_prompt_cut(r["words"], r["duration"]):
                reason = "no_prompt_cut"
        r["status"] = "kept" if reason is None else "dropped"
        r["reason"] = reason
        if reason is None:
            manifests[r["split"]].append({k: r[k] for k in README_KEYS})
    for split, items in manifests.items():
        _write_jsonl(man / f"{split}_aligned.jsonl", items)
    _write_jsonl(man / "clips_metadata.jsonl", [{k: v for k, v in r.items() if k != "words"} for r in rows])
    for p in parts:
        p.unlink(missing_ok=True)
    info = {"ctc_nll_threshold": round(threshold, 4), **{s: len(v) for s, v in manifests.items()}}
    (out / "align_stats.json").write_text(json.dumps(info, indent=2))
    write_stats(out)
    return info


# ----------------------------------------------------------------------------- STATS.md


def write_stats(out: Path) -> Path:
    """STATS.md + stats.json: hours per source and split, clips kept / dropped and why."""
    clip_stats = json.loads((out / "clip_stats.json").read_text()) if (out / "clip_stats.json").exists() else {"sources": {}}
    meta = _read_jsonl(out / "manifests" / "clips_metadata.jsonl")
    aligned = bool(meta)
    rows = meta or _read_jsonl(out / "manifests" / "clips.jsonl")
    align_info = json.loads((out / "align_stats.json").read_text()) if (out / "align_stats.json").exists() else {}
    sources = sorted(set(clip_stats["sources"]) | {r["source"] for r in rows})

    from ..text import Romanizer

    english = Romanizer()
    per: dict[str, dict[str, Any]] = {}
    for src in sources:
        cs = clip_stats["sources"].get(src, {"seen": 0, "kept": 0, "rejected": {}})
        mine = [r for r in rows if r["source"] == src]
        kept = [r for r in mine if r.get("status", "kept") == "kept"]
        dropped = collections.Counter(cs.get("rejected", {}))
        dropped.update(r["reason"] for r in mine if r.get("status") == "dropped")
        hours = {s: round(sum(r["duration"] for r in kept if r["split"] == s) / 3600, 2) for s in SPLITS}
        words = [w for r in kept for w in r["transcript"].split()]
        en_share = (
            sum(t.origin == "en" for r in kept[:400] for t in english.tokens(r["raw_text"]) if t.origin in ("en", "hi"))
            / max(1, sum(t.origin in ("en", "hi") for r in kept[:400] for t in english.tokens(r["raw_text"])))
        )
        per[src] = {
            "sampled": cs.get("seen", len(mine)), "kept": len(kept), "dropped": dict(dropped), "hours": hours,
            "total_hours": round(sum(hours.values()), 2), "speakers": len({r["speaker"] for r in kept}),
            "mean_sec": round(float(np.mean([r["duration"] for r in kept])), 2) if kept else 0.0,
            "words": len(words), "english_word_share": round(en_share, 3),
        }  # fmt: skip
    totals = {s: round(sum(p["hours"][s] for p in per.values()), 2) for s in SPLITS}
    stats = {"generated": time.strftime("%Y-%m-%d %H:%M"), "aligned": aligned, "sources": per, "hours": totals, "align": align_info}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    reasons = sorted({k for p in per.values() for k in p["dropped"]})
    lines = [
        "# Hinglish TTS corpus statistics", "",
        f"Generated {stats['generated']} by `hinglish_tts.data.prepare`"
        + ("" if aligned else " (before alignment: alignment-stage drops not counted yet)") + ".", "",
        f"**Total: {sum(totals.values()):.2f} h** (train {totals['train']} h, valid {totals['valid']} h, test {totals['test']} h).", "",
        "## Hours per source", "",
        "| source | train h | valid h | test h | total h | clips | speakers | mean clip s | English-word share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for src, p in per.items():
        h = p["hours"]
        lines.append(f"| {src} | {h['train']} | {h['valid']} | {h['test']} | {p['total_hours']} | {p['kept']} | "
                     f"{p['speakers']} | {p['mean_sec']} | {p['english_word_share']:.0%} |")  # fmt: skip
    lines += ["", "## Clips kept and dropped", "",
              "`sampled` = rows drawn by the hour-budget sampler; rows outside the budget are not counted as dropped.", "",
              "| source | sampled | kept | " + " | ".join(reasons) + " |",
              "|---|---|---|" + "---|" * len(reasons)]  # fmt: skip
    for src, p in per.items():
        lines.append(f"| {src} | {p['sampled']} | {p['kept']} | " + " | ".join(str(p["dropped"].get(k, 0)) for k in reasons) + " |")
    lines += ["", "### Drop reasons", ""] + [f"- `{k}`: {REASONS.get(k, k)}" for k in reasons]
    if align_info:
        lines += ["", f"Alignment: torchaudio MMS_FA; train clips with per-character CTC loss above "
                      f"{align_info.get('ctc_nll_threshold')} were dropped as transcript mismatches."]  # fmt: skip
    examples = [r for r in rows if r.get("status", "kept") == "kept"][:: max(1, len(rows) // 12)][:12]
    lines += ["", "## Transcript examples (raw -> training text)", "", "| source | raw | training text |", "|---|---|---|"]
    for r in examples:
        raw = r["raw_text"].replace("|", "/")[:120]
        lines.append(f"| {r['source']} | {raw} | {r['transcript'][:120]} |")
    path = out / "STATS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", type=Path, default=PROJECT_DIR / "configs" / "data_recipe.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["clips", "align", "all"], default="all")
    ap.add_argument("--lexicon", default=None, help="native<TAB>roman lexicon (text/lexicon.py builds one from Aksharantar)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--hours-scale", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--gpus", type=int, default=None)
    args = ap.parse_args()
    if args.stage in ("clips", "all"):
        print(json.dumps(prepare_clips(args.recipe, args.out, args.lexicon, args.workers, args.hours_scale, args.only)))
    if args.stage in ("align", "all"):
        import torch

        gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
        print(json.dumps(align_and_finalize(args.out, gpus)))
    print((args.out / "STATS.md").read_text())


if __name__ == "__main__":
    main()
