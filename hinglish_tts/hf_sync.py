"""Mirror training runs to a private Hugging Face repo, and restore them in a new session.

Kaggle sessions end after 12 h and take /kaggle/working with them unless the
notebook is committed. While a run trains, `RunSync` watches its run dir and,
after every checkpoint, uploads in a single commit:

    <prefix>/checkpoint_<step>.pt   model + EMA (what the trainer resumes from)
    <prefix>/optim_<step>.pt        optimizer state (exact resume; HF_SYNC_OPTIM=0 skips it)
    <prefix>/model.safetensors      pocket-tts export of the EMA weights
    <prefix>/progress.jsonl, args.yaml, runner_summary.json, samples/*.wav (new ones only)

and deletes the previous checkpoint files, then squashes the repo history so
only the latest checkpoint is stored. `restore` downloads a run back into its
run dir; the upstream trainer then resumes from it automatically.

Repo: $HF_SYNC_REPO (default harshsxna/pocket-tts-hinglish, private); set it to
"" to disable. Needs an HF_TOKEN with write access.

    python scripts/hf_sync.py restore --run-dir /kaggle/working/runs/teacher --root /kaggle/working
    python scripts/hf_sync.py push --run-dir /kaggle/working/runs/teacher --root /kaggle/working
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hinglish.hf_sync")

DEFAULT_REPO = "harshsxna/pocket-tts-hinglish"
_CKPT = re.compile(r"checkpoint_(\d+)\.pt$")
SMALL_FILES = ("progress.jsonl", "args.yaml", "runner_summary.json")


def sync_repo() -> str | None:
    """The sync repo id, or None when syncing is disabled or impossible."""
    repo = os.environ.get("HF_SYNC_REPO", DEFAULT_REPO).strip()
    if not repo or not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return None
    return repo


def _api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def latest_step(run_dir: Path) -> int | None:
    steps = [int(m.group(1)) for p in run_dir.glob("checkpoint_*.pt") if (m := _CKPT.match(p.name))]
    return max(steps) if steps else None


def _complete(run_dir: Path, step: int, with_optim: bool) -> bool:
    """True once the trainer has finished writing checkpoint `step` and its export."""
    ckpt = run_dir / f"checkpoint_{step:08d}.pt"
    export = run_dir / "model.safetensors"
    if any(run_dir.glob("*.tmp")) or not ckpt.exists() or not export.exists():
        return False
    if with_optim and not (run_dir / f"optim_{step:08d}.pt").exists():
        return False
    return export.stat().st_mtime >= ckpt.stat().st_mtime


def push_run(repo_id: str, run_dir: Path, root: Path, step: int | None = None, api: Any = None,
             with_optim: bool | None = None, squash: bool = True) -> int | None:  # fmt: skip
    """Upload the latest checkpoint of `run_dir` (path in the repo = path relative to `root`)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    api = api or _api()
    with_optim = os.environ.get("HF_SYNC_OPTIM", "1") == "1" if with_optim is None else with_optim
    step = step if step is not None else latest_step(run_dir)
    if step is None:
        return None
    prefix = run_dir.resolve().relative_to(root.resolve()).as_posix()
    stage = run_dir / ".hf_staging" / str(step)
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "samples").mkdir(parents=True)
    # Hard links: instant, no extra disk, and the upload survives the trainer deleting
    # or replacing these files when it writes the next checkpoint.
    big = [f"checkpoint_{step:08d}.pt", "model.safetensors"] + ([f"optim_{step:08d}.pt"] if with_optim else [])
    for name in big:
        os.link(run_dir / name, stage / name)
    for name in SMALL_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, stage / name)
    existing = set(api.list_repo_files(repo_id))
    for wav in sorted((run_dir / "samples").glob("*.wav")):
        if f"{prefix}/samples/{wav.name}" not in existing:
            os.link(wav, stage / "samples" / wav.name)
    ops: list[Any] = [
        CommitOperationAdd(path_in_repo=f"{prefix}/{p.relative_to(stage).as_posix()}", path_or_fileobj=str(p))
        for p in sorted(stage.rglob("*")) if p.is_file()
    ]  # fmt: skip
    keep = {f"{prefix}/{n}" for n in big}
    ops += [
        CommitOperationDelete(path_in_repo=f)
        for f in sorted(existing)
        if f.startswith(f"{prefix}/") and re.search(r"/(checkpoint|optim)_\d+\.pt$", f) and f not in keep
    ]
    t0 = time.time()
    api.create_commit(repo_id=repo_id, operations=ops, commit_message=f"{prefix}: step {step}")
    if squash:
        try:
            api.super_squash_history(repo_id=repo_id)  # drop superseded checkpoints from storage
        except Exception as exc:  # noqa: BLE001 -- storage cleanup is best effort
            logger.warning(f"could not squash {repo_id} history: {exc!r}")
    shutil.rmtree(run_dir / ".hf_staging", ignore_errors=True)
    logger.info(f"synced {prefix} step {step} to {repo_id} in {time.time() - t0:.0f}s")
    print(f"[hf_sync] {prefix} step {step} -> https://huggingface.co/{repo_id} ({time.time() - t0:.0f}s)", flush=True)
    return step


def push_files(repo_id: str, files: dict[Path, str], api: Any = None) -> None:
    """Upload small loose files (pipeline state, STATS.md, eval results) in one commit."""
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files.items() if Path(src).exists()]
    if ops:
        (api or _api()).create_commit(repo_id=repo_id, operations=ops, commit_message="sync pipeline files")


def restore(repo_id: str, run_dir: Path, root: Path, what: str = "all", api: Any = None) -> int | None:
    """Download a run from the repo into `run_dir`, unless a newer checkpoint is already local.

    what: "all" (resume training), "checkpoint" (evaluation: no optimizer state),
    or "export" (model.safetensors + metadata only). Returns the restored step.
    """
    from huggingface_hub import hf_hub_download

    api = api or _api()
    prefix = run_dir.resolve().relative_to(root.resolve()).as_posix()
    try:
        files = [f for f in api.list_repo_files(repo_id) if f.startswith(f"{prefix}/")]
    except Exception as exc:  # noqa: BLE001 -- repo not created yet, offline, no access
        logger.info(f"nothing to restore from {repo_id}: {type(exc).__name__}")
        return None
    remote = [int(m.group(1)) for f in files if (m := _CKPT.search(f))]
    remote_step = max(remote) if remote else None
    local = latest_step(run_dir)
    if what != "export" and (remote_step is None or (local is not None and local >= remote_step)):
        return local
    if what == "export" and (run_dir / "model.safetensors").exists():
        return local

    def wanted(f: str) -> bool:
        if what == "export":
            return f.endswith(("model.safetensors", *SMALL_FILES))
        if "/optim_" in f:
            return what == "all"
        return True

    for f in files:
        if wanted(f):
            hf_hub_download(repo_id, f, local_dir=root)
    print(f"[hf_sync] restored {prefix} ({what}, step {remote_step}) from https://huggingface.co/{repo_id}", flush=True)
    return remote_step


def ensure_repo(repo_id: str, api: Any = None, repo_type: str = "model") -> None:
    (api or _api()).create_repo(repo_id, private=True, exist_ok=True, repo_type=repo_type)


def check_access(repo_id: str, api: Any = None, repo_type: str = "model") -> str | None:
    """None if the token can write to `repo_id` (creating it, private, if needed), else why not."""
    api = api or _api()
    try:
        me = api.whoami()
    except Exception as exc:  # noqa: BLE001
        return f"the HF token is invalid ({type(exc).__name__})"
    namespaces = {me.get("name")} | {o.get("name") for o in me.get("orgs", [])}
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    owner = repo_id.split("/")[0]
    if owner not in namespaces:
        return f"the HF token belongs to {me.get('name')!r} (orgs: {sorted(n for n in namespaces if n != me.get('name'))}), which cannot write to {owner!r}"
    if role == "read":
        return "the HF token is read-only; create one with write access"
    try:
        ensure_repo(repo_id, api, repo_type)
    except Exception as exc:  # noqa: BLE001
        return f"cannot create {repo_id}: {exc!r}"
    return None


class RunSync:
    """Background watcher: push every new complete checkpoint of `run_dir` while it trains."""

    def __init__(self, repo_id: str, run_dir: Path, root: Path, poll_sec: float = 30.0, api: Any = None):
        self.repo_id, self.run_dir, self.root, self.poll_sec = repo_id, run_dir, root, poll_sec
        self.api = api
        self.pushed = latest_step(run_dir)  # what a restore brought back is already on the Hub
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "RunSync":
        ensure_repo(self.repo_id, self.api)
        self._thread.start()
        return self

    def _maybe_push(self) -> None:
        step = latest_step(self.run_dir)
        with_optim = os.environ.get("HF_SYNC_OPTIM", "1") == "1"
        if step is None or step == self.pushed or not _complete(self.run_dir, step, with_optim):
            return
        try:
            self.pushed = push_run(self.repo_id, self.run_dir, self.root, step, self.api, with_optim)
        except Exception as exc:  # noqa: BLE001 -- a failed upload must never stop training
            print(f"[hf_sync] upload of step {step} failed, will retry: {exc!r}", flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_sec):
            self._maybe_push()

    def stop(self, final_wait_sec: float = 300.0) -> None:
        """Stop watching, then push the final checkpoint (written at the budget stop)."""
        self._stop.set()
        self._thread.join()
        deadline = time.time() + final_wait_sec
        step = latest_step(self.run_dir)
        while step is not None and step != self.pushed and time.time() < deadline:
            if _complete(self.run_dir, step, os.environ.get("HF_SYNC_OPTIM", "1") == "1"):
                self._maybe_push()
                break
            time.sleep(5)
