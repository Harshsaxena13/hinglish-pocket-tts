"""Upload the prepared corpus to a private Hugging Face dataset, and pull it back.

Preparing the data takes about an hour of a 12 h Kaggle session; uploading it
once lets later sessions (RUN_MODE="train") skip straight to training.

Repo layout (paths relative to the data root, so they work on any machine):
    audio/<source>-<k>.tar        ~1 GB tar shards of the 24 kHz FLAC clips
    manifests/{train,valid,test}_aligned.jsonl, clips_metadata.jsonl
    STATS.md, stats.json, lexicon_hi.tsv (if built), README.md (dataset card)

A Hub repo handles ~10 large files far better than ~50k small ones.

    python -m hinglish_tts.data.hub push --data /kaggle/temp/hinglish/data --repo harshsxna/hinglish-tts-data
    python -m hinglish_tts.data.hub pull --data /kaggle/temp/hinglish/data --repo harshsxna/hinglish-tts-data
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

DEFAULT_REPO = "harshsxna/hinglish-tts-data"
MANIFESTS = ("train_aligned.jsonl", "valid_aligned.jsonl", "test_aligned.jsonl", "clips_metadata.jsonl")
EXTRAS = ("STATS.md", "stats.json")


def _rewrite(src: Path, dst: Path, fn: Any) -> int:
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                row = json.loads(line)
                row["path"] = fn(row["path"])
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n


def pack(data_dir: Path, staging: Path, shard_bytes: int = 1 << 30, lexicon: Path | None = None) -> dict[str, Any]:
    """Write the repo layout into `staging`: tar shards of every clip the manifests use."""
    data_dir, staging = data_dir.resolve(), staging.resolve()
    man = data_dir / "manifests"
    (staging / "manifests").mkdir(parents=True, exist_ok=True)
    (staging / "audio").mkdir(exist_ok=True)

    def rel(p: str) -> str:
        return Path(p).resolve().relative_to(data_dir).as_posix()

    used: dict[str, list[str]] = collections.defaultdict(list)
    for name in MANIFESTS[:3]:
        for line in (man / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = rel(json.loads(line)["path"])
                used[r.split("/")[1]].append(r)  # audio/<source>/...
    shards = 0
    for source, files in sorted(used.items()):
        k, size, tar = 0, 0, None
        for r in sorted(set(files)):
            if tar is None or size >= shard_bytes:
                if tar is not None:
                    tar.close()
                    k += 1
                tar = tarfile.open(staging / "audio" / f"{source}-{k:03d}.tar", "w")
                size, shards = 0, shards + 1
            tar.add(data_dir / r, arcname=r)
            size += (data_dir / r).stat().st_size
        if tar is not None:
            tar.close()
    counts = {name: _rewrite(man / name, staging / "manifests" / name, rel) for name in MANIFESTS if (man / name).exists()}
    for extra in EXTRAS:
        if (data_dir / extra).exists():
            shutil.copy2(data_dir / extra, staging / extra)
    if lexicon is not None and lexicon.exists():
        shutil.copy2(lexicon, staging / "lexicon_hi.tsv")
    stats = (data_dir / "STATS.md").read_text(encoding="utf-8") if (data_dir / "STATS.md").exists() else ""
    (staging / "README.md").write_text(
        "---\nlicense: other\nlanguage: [hi, en]\ntags: [tts, hinglish, code-switching]\n---\n\n"
        "# Hinglish TTS training data (prepared)\n\nRomanized Hinglish transcripts, 24 kHz mono FLAC, word "
        "alignments from torchaudio MMS_FA, built by `hinglish_tts.data.prepare` from MUCS 2021, IndicVoices-R, "
        "IIT Madras Indic TTS and Svarah. **Private**: each source keeps its own license; do not redistribute "
        "without checking them.\n\nRestore with `python -m hinglish_tts.data.hub pull`.\n\n" + stats,
        encoding="utf-8",
    )
    return {"shards": shards, "clips": sum(len(set(v)) for v in used.values()), "manifests": counts}


def unpack(staging: Path, data_dir: Path) -> dict[str, Any]:
    """Extract the shards into `data_dir` and write manifests with absolute paths."""
    data_dir = data_dir.resolve()
    (data_dir / "manifests").mkdir(parents=True, exist_ok=True)
    for tar_path in sorted((staging / "audio").glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts]
            tar.extractall(data_dir, members=members)
    counts = {}
    for name in MANIFESTS:
        if (staging / "manifests" / name).exists():
            counts[name] = _rewrite(staging / "manifests" / name, data_dir / "manifests" / name, lambda p: str(data_dir / p))
    for extra in EXTRAS:
        if (staging / extra).exists():
            shutil.copy2(staging / extra, data_dir / extra)
    missing = [
        p for p in (json.loads(l)["path"] for l in (data_dir / "manifests" / "train_aligned.jsonl").read_text().splitlines()[:200] if l)
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest paths missing after unpack, e.g. {missing[0]}")
    return counts


def push(data_dir: Path, repo_id: str = DEFAULT_REPO, lexicon: Path | None = None, path_in_repo: str = "") -> str:
    from huggingface_hub import HfApi

    staging = data_dir / "_hub_upload"
    shutil.rmtree(staging, ignore_errors=True)
    info = pack(data_dir, staging, lexicon=lexicon)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(staging), path_in_repo=path_in_repo or None,
                      commit_message=f"prepared Hinglish TTS data: {info['clips']} clips")  # fmt: skip
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[data] pushed {info['clips']} clips in {info['shards']} shards to https://huggingface.co/datasets/{repo_id}")
    return repo_id


def pull(data_dir: Path, repo_id: str = DEFAULT_REPO, lexicon_out: Path | None = None, path_in_repo: str = "") -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    staging = data_dir / "_hub_download"
    pattern = f"{path_in_repo.rstrip('/')}/*" if path_in_repo else None
    snapshot_download(repo_id, repo_type="dataset", local_dir=staging, allow_patterns=pattern)
    root = staging / path_in_repo if path_in_repo else staging
    counts = unpack(root, data_dir)
    if lexicon_out is not None and (root / "lexicon_hi.tsv").exists():
        lexicon_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "lexicon_hi.tsv", lexicon_out)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[data] pulled https://huggingface.co/datasets/{repo_id}: {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--data", type=Path, required=True, help="data root (the one with manifests/ and audio/)")
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO") or DEFAULT_REPO)
    ap.add_argument("--lexicon", type=Path, default=None)
    args = ap.parse_args()
    if args.action == "push":
        push(args.data, args.repo, args.lexicon)
    else:
        pull(args.data, args.repo, args.lexicon)


if __name__ == "__main__":
    main()
