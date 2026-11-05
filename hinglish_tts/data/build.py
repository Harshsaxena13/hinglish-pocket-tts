"""Shard-parallel clip extraction from Hugging Face parquet datasets.

Each shard is downloaded, processed and deleted by one worker process, so the
stage is resumable at shard granularity and its disk footprint stays small. A
source's `repo` may also be a local folder of parquet files.

Per clip: metadata filters -> transcript to romanized Hinglish (digits spelled
out, untrustworthy text dropped) -> decode -> mono, 24 kHz -> silence trim ->
duration window -> loudness check and normalization -> speaking-rate sanity
check -> 16-bit FLAC. Every rejection is counted by reason for STATS.md.

Output: <out>/audio/<source>/<shard>/<clip>.flac and <out>/shards/<source>_<shard>.jsonl
(used by `hinglish_tts.data.prepare`, which splits, aligns and reports).
"""

from __future__ import annotations

import collections
import fnmatch
import hashlib
import json
import logging
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import audio

logger = logging.getLogger("hinglish.data")

OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


@dataclass
class ShardJob:
    source: dict[str, Any]
    filename: str
    index: int
    out_dir: str
    hours_cap: float
    sample_rate: int
    min_sec: float
    max_sec: float
    variant_prob: float
    loudness_db: float
    seed: int
    lexicon: str | None = None


@dataclass
class ShardResult:
    source: str
    index: int
    kept: int = 0
    hours: float = 0.0
    seen: int = 0  # rows drawn by the budget sampler; the rest of the shard is simply unused
    rejected: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def stable_unit(*parts: object) -> float:
    """Deterministic pseudo-random number in [0, 1) from the given parts."""
    h = hashlib.sha1("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _class_names(parquet_file: Any) -> dict[str, list[str]]:
    """ClassLabel names of each column, from the `huggingface` parquet metadata."""
    meta = parquet_file.schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return {}
    try:
        features = json.loads(raw).get("info", {}).get("features", {})
    except json.JSONDecodeError:
        return {}
    return {
        k: v["names"]
        for k, v in features.items()
        if isinstance(v, dict) and v.get("_type") == "ClassLabel" and "names" in v
    }


def _label(value: Any, names: list[str] | None) -> str | None:
    if value is None:
        return None
    if names is not None and isinstance(value, int) and 0 <= value < len(names):
        return names[value]
    return str(value)


def _passes(row: dict[str, Any], filters: dict[str, list[Any]]) -> str | None:
    """Name of the first failed filter, or None."""
    for col, (op, ref) in filters.items():
        val = row.get(col)
        if val is None:
            continue
        try:
            if not OPS[op](float(val), float(ref)):
                return f"filter_{col}"
        except (TypeError, ValueError):
            continue
    return None


def _speaker(row: dict[str, Any], src: dict[str, Any], names: dict[str, list[str]], shard: int) -> str:
    name = src["name"]
    if src.get("speaker") and row.get(src["speaker"]) is not None:
        return f"{name}:{row[src['speaker']]}"
    if src.get("speaker_from_gender") and src.get("gender"):
        return f"{name}:{_label(row.get(src['gender']), names.get(src['gender']))}"
    return f"{name}:shard{shard}"


def process_shard(job: ShardJob) -> ShardResult:
    """Download one parquet shard, keep the clips that pass every check, delete it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ..text import HinglishFrontend

    t0 = time.time()
    src = job.source
    name = src["name"]
    out_dir = Path(job.out_dir)
    done_path = out_dir / "shards" / f"{name}_{job.index:03d}.jsonl"
    stats_path = done_path.with_suffix(".stats.json")
    if done_path.exists() and stats_path.exists():
        return ShardResult(**json.loads(stats_path.read_text()))

    dl_dir = out_dir / "_downloads" / f"{name}_{job.index:03d}"
    if Path(src["repo"]).is_dir():
        local = str(Path(src["repo"]) / job.filename)
    else:
        local = hf_hub_download(src["repo"], job.filename, repo_type="dataset", local_dir=dl_dir)
    pf = pq.ParquetFile(local)
    names = _class_names(pf)
    columns = set(pf.schema_arrow.names)
    audio_col = src.get("audio", "audio")
    text_cols = [c for c in src.get("text", ["text"]) if c in columns]
    if audio_col not in columns or not text_cols:
        raise ValueError(f"{src['repo']}/{job.filename}: need {audio_col} and one of {src['text']}, has {sorted(columns)}")
    filters = {k: v for k, v in (src.get("filters") or {}).items() if k in columns}
    meta_cols = sorted(
        {c for c in (src.get("speaker"), src.get("gender"), "duration") if c and c in columns} | set(filters)
    )

    # Plan the sampling rate from metadata when durations are known; otherwise use
    # the recipe's sample_frac and let the hour cap stop the shard.
    keep_prob = float(src.get("sample_frac", 1.0))
    if "duration" in columns:
        meta = pf.read(columns=meta_cols).to_pylist()
        eligible = sum(
            float(r["duration"] or 0)
            for r in meta
            if _passes(r, filters) is None and job.min_sec <= float(r["duration"] or 0) <= job.max_sec + 2.0
        )
        if eligible > 0:
            keep_prob = min(1.0, 1.15 * job.hours_cap * 3600 / eligible)

    frontend = HinglishFrontend(job.lexicon)
    clip_dir = out_dir / "audio" / name / f"{job.index:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    res = ShardResult(name, job.index)
    rejected: collections.Counter[str] = collections.Counter()
    rows_out: list[str] = []
    row_idx = -1
    for batch in pf.iter_batches(batch_size=32, columns=[audio_col, *text_cols, *meta_cols]):
        for row in batch.to_pylist():
            row_idx += 1
            if res.hours >= job.hours_cap:
                break
            if stable_unit(job.seed, name, job.index, row_idx) >= keep_prob:
                continue
            res.seen += 1
            failed = _passes(row, filters)
            if failed:
                rejected[failed] += 1
                continue
            raw_text = next((str(row[c]) for c in text_cols if row.get(c)), "")
            clip_id = f"{name}_{job.index:03d}_{row_idx:06d}"
            text = frontend.training_text(raw_text, random.Random(clip_id), job.variant_prob)
            if text is None:
                rejected["bad_transcript"] += 1
                continue
            blob = row[audio_col]
            data = blob.get("bytes") if isinstance(blob, dict) else blob
            try:
                wav, sr = audio.decode(data)
            except Exception:  # noqa: BLE001 -- missing or corrupt audio
                rejected["undecodable_audio"] += 1
                continue
            wav = audio.trim_silence(audio.resample(wav, sr, job.sample_rate), job.sample_rate)
            duration = len(wav) / job.sample_rate
            if duration < job.min_sec:
                rejected["too_short"] += 1
                continue
            if duration > job.max_sec:
                rejected["too_long"] += 1
                continue
            if not audio.loudness_ok(wav):
                rejected["too_quiet"] += 1
                continue
            letters = sum(c.isalpha() for c in text)
            if not 4.0 <= letters / duration <= 28.0:  # transcript cannot match the audio
                rejected["implausible_speaking_rate"] += 1
                continue
            path = clip_dir / f"{clip_id}.flac"
            audio.write_flac(path, audio.normalize_loudness(wav, job.sample_rate, job.loudness_db), job.sample_rate)
            gender = _label(row.get(src["gender"]), names.get(src["gender"])) if src.get("gender") else None
            rows_out.append(
                json.dumps(
                    {
                        "path": str(path.resolve()),
                        "duration": round(duration, 3),
                        "transcript": text,
                        "source": name,
                        "speaker": _speaker(row, src, names, job.index),
                        "gender": gender,
                        "raw_text": raw_text,
                        "sample_rate_in": sr,
                    },
                    ensure_ascii=False,
                )
            )
            res.kept += 1
            res.hours += duration / 3600
        if res.hours >= job.hours_cap:
            break

    res.rejected = dict(rejected)
    res.seconds = time.time() - t0
    done_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = done_path.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in rows_out), encoding="utf-8")
    os.replace(tmp, done_path)
    stats_path.write_text(json.dumps(asdict(res)))
    shutil.rmtree(dl_dir, ignore_errors=True)
    return res


def _list_files(repo: str, pattern: str) -> list[str]:
    if Path(repo).is_dir():
        return sorted(str(p.relative_to(repo)) for p in Path(repo).rglob("*") if fnmatch.fnmatch(str(p.relative_to(repo)), pattern))
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if fnmatch.fnmatch(f, pattern))


def plan_jobs(recipe: dict[str, Any], out_dir: Path, lexicon: str | None, only: list[str] | None = None) -> list[ShardJob]:
    jobs: list[ShardJob] = []
    for src in recipe["sources"]:
        if only and src["name"] not in only:
            continue
        try:
            files = _list_files(src["repo"], src["files"])
        except Exception as exc:  # noqa: BLE001 -- gated or unreachable repo
            if src.get("optional"):
                logger.warning(f"skipping optional source {src['name']}: {type(exc).__name__}: {exc}")
                continue
            raise
        if not files:
            raise FileNotFoundError(f"{src['repo']}: no files match {src['files']}")
        rng = random.Random(f"{recipe.get('seed', 0)}-{src['name']}")
        if src.get("max_shards") and src["max_shards"] < len(files):
            files = sorted(rng.sample(files, src["max_shards"]))
        cap = float(src["max_hours"]) / len(files)
        for i, f in enumerate(files):
            jobs.append(
                ShardJob(
                    source=src, filename=f, index=i, out_dir=str(out_dir), hours_cap=cap,
                    sample_rate=int(recipe.get("sample_rate", 24000)),
                    min_sec=float(src.get("min_sec", recipe.get("min_sec", 1.0))),
                    max_sec=float(src.get("max_sec", recipe.get("max_sec", 15.0))),
                    variant_prob=float(recipe.get("variant_prob", 0.0)),
                    loudness_db=float(recipe.get("loudness_db", -20.0)),
                    seed=int(recipe.get("seed", 0)), lexicon=lexicon,
                )  # fmt: skip
            )
    # Interleave sources so concurrent downloads hit different repos.
    by_src: dict[str, list[ShardJob]] = collections.defaultdict(list)
    for j in jobs:
        by_src[j.source["name"]].append(j)
    interleaved: list[ShardJob] = []
    while any(by_src.values()):
        for k in list(by_src):
            if by_src[k]:
                interleaved.append(by_src[k].pop(0))
    return interleaved


def run_shards(jobs: list[ShardJob], workers: int) -> list[ShardResult]:
    results: list[ShardResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_shard, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad shard must not sink the corpus
                logger.error(f"shard {j.source['name']}#{j.index} ({j.filename}) failed: {exc!r}")
                results.append(ShardResult(j.source["name"], j.index, rejected={"shard_failed": 1}))
                continue
            results.append(r)
            logger.info(
                f"{r.source}#{r.index}: kept {r.kept} clips / {r.hours:.2f} h of {r.seen} sampled "
                f"in {r.seconds:.0f}s, rejected {r.rejected}"
            )
    return results
