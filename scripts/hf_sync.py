"""CLI for syncing training runs with the private Hugging Face repo.

The pipeline does all of this automatically (watch during training, restore
before training / eval / export). Use this script to do it by hand:

    python scripts/hf_sync.py push    --run-dir /kaggle/working/runs/teacher --root /kaggle/working
    python scripts/hf_sync.py restore --run-dir /kaggle/working/runs/teacher --root /kaggle/working [--what all|checkpoint|export]
    python scripts/hf_sync.py watch   --run-dir /kaggle/working/runs/teacher --root /kaggle/working   # until Ctrl-C

Repo: --repo, else $HF_SYNC_REPO, else harshsxna/pocket-tts-hinglish. Needs HF_TOKEN (write).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hinglish_tts.hf_sync import DEFAULT_REPO, RunSync, ensure_repo, push_run, restore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "restore", "watch"])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="paths in the repo are relative to this dir")
    ap.add_argument("--repo", default=os.environ.get("HF_SYNC_REPO") or DEFAULT_REPO)
    ap.add_argument("--what", choices=["all", "checkpoint", "export"], default="all")
    args = ap.parse_args()
    if args.action == "push":
        ensure_repo(args.repo)
        print("pushed step", push_run(args.repo, args.run_dir, args.root))
    elif args.action == "restore":
        print("restored step", restore(args.repo, args.run_dir, args.root, args.what))
    else:
        sync = RunSync(args.repo, args.run_dir, args.root).start()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            sync.stop()


if __name__ == "__main__":
    main()
