"""Rebuild a pocket-tts export from the checkpoint-sync repo alone (no Kaggle output needed).

    python scripts/export_from_hub.py                    # list, download weights, build exports/<name>/
    python scripts/export_from_hub.py --push             # ... and upload it to <repo>/exports/<name>/

Downloads only <run>/model.safetensors (the EMA export) and small text files
(*.json, *.jsonl, *.yaml, *.md under 2 MB): never optimizer state (*.pt) or data.
The weights land in the Hugging Face cache and are hard-linked into the export
folder, so they take disk space once.

The export folder holds:
    model.safetensors     the fine-tuned weights
    config.yaml           hf:// config: the weights in <repo>/exports/<name>/, the base
                          model's English tokenizer unchanged (Kyutai's pinned file)
    config_local.yaml     the same with a local weights path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SMALL = (".json", ".jsonl", ".yaml", ".yml", ".md", ".txt")
MAX_SMALL = 2 * 1024 * 1024


def human(n: int | None) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("HF_SYNC_REPO", "harshsxna/pocket-tts-hinglish"))
    ap.add_argument("--run", default="teacher", help="run folder under runs/")
    ap.add_argument("--base", default="english_2026-09_24l", help="pocket-tts base config the run was trained from")
    ap.add_argument("--name", default="hinglish-pocket-tts-24l", help="export folder name")
    ap.add_argument("--out", type=Path, default=ROOT / "exports")
    ap.add_argument("--push", action="store_true", help="upload model.safetensors, config.yaml, README.md")
    args = ap.parse_args()

    import yaml
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    from hinglish_tts.export import _config_dict, _write_yaml

    api = HfApi()
    files = [f for f in api.list_repo_tree(args.repo, recursive=True) if hasattr(f, "size")]
    run_prefix = f"runs/{args.run}/"
    print(f"== {args.repo}: {run_prefix}")
    for f in sorted(files, key=lambda f: f.path):
        if f.path.startswith(run_prefix):
            print(f"  {human(f.size):>10}  {f.path}")
    print("== other folders: " + ", ".join(f"{d} ({human(sum(f.size for f in files if f.path.startswith(d + '/')))})"
                                           for d in sorted({f.path.split('/')[0] for f in files if '/' in f.path})))  # fmt: skip

    folder = args.out / args.name
    meta = args.out / "_hub_meta"
    folder.mkdir(parents=True, exist_ok=True)
    weights = f"{run_prefix}model.safetensors"
    if weights not in {f.path for f in files}:
        raise SystemExit(f"{weights} is not in {args.repo}")
    small = [f.path for f in files if f.path.endswith(SMALL) and (f.size or 0) <= MAX_SMALL
             and (f.path.startswith(run_prefix) or f.path.startswith("eval/") or "/" not in f.path)]  # fmt: skip
    for path in small:
        hf_hub_download(args.repo, path, local_dir=meta)
    print(f"== downloaded {len(small)} small files -> {meta}")

    print(f"== downloading {weights} ...", flush=True)
    cached = Path(hf_hub_download(args.repo, weights)).resolve()
    target = folder / "model.safetensors"
    if target.exists() and not target.samefile(cached):
        target.unlink()
    if not target.exists():
        os.link(cached, target)  # same blob as the HF cache: no second copy on disk
    print(f"   {human(target.stat().st_size)} -> {target}")

    import pocket_tts

    base_cfg = Path(pocket_tts.__file__).parent / "config" / f"{args.base}.yaml"
    base_tok = yaml.safe_load(base_cfg.read_text())["flow_lm"]["lookup_table"]["tokenizer_path"]
    header = f"# Hinglish Pocket TTS ({args.run} run), derived from {args.base}; tokenizer unchanged.\n"
    _write_yaml(folder / "config_local.yaml", _config_dict(base_cfg, str(target.resolve()), base_tok), header)
    hub_weights = f"hf://{args.repo}/exports/{args.name}/model.safetensors"
    _write_yaml(folder / "config.yaml", _config_dict(base_cfg, hub_weights, base_tok), header)
    print(f"== configs written; tokenizer: {base_tok}")

    if args.push:
        ops = [CommitOperationAdd(f"exports/{args.name}/{p.name}", str(p))
               for p in (target, folder / "config.yaml", folder / "README.md") if p.exists()]  # fmt: skip
        commit = api.create_commit(args.repo, ops, commit_message=f"Export {args.name} from runs/{args.run}")
        print(json.dumps({"uploaded": [op.path_in_repo for op in ops], "commit": commit.oid}, indent=1))
        print(f"config: hf://{args.repo}/exports/{args.name}/config.yaml")


if __name__ == "__main__":
    main()
