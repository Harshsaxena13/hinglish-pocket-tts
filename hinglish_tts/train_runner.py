"""Launch a pocket-tts training run with a wall-clock budget.

Kaggle sessions die hard at 12 h, so runs are bounded by time, not steps: when
the budget is spent the runner touches <run_dir>/STOP, and the (upstream)
trainer checkpoints after its current step, exports model.safetensors (EMA
weights) and exits. Re-running the same stage resumes from that checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import signal
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

# Lines worth showing in a notebook; everything still goes to the log file.
_INTERESTING = re.compile(
    r"valid @|saved |exported|wrote \d+ samples|mixed precision|trainable params|precomput|"
    r"resumed|first step|termination|done|peak GPU|Error|error|Traceback|overflow|WARNING|stitch_frames"
)
_STEP = re.compile(r"\] step (\d+) \| lr")


def render_config(template: str | Path, out: str | Path, **variables: str) -> Path:
    """Substitute ${VAR} placeholders and check the result parses."""
    text = string.Template(Path(template).read_text()).substitute(variables)
    yaml.safe_load(text)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def training_env(pocket_tts_dir: Path, project_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pocket_tts_dir), str(project_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # NCCL peer-to-peer over the T4s' PCIe link has been flaky on Kaggle; host
    # staging is a little slower and always works.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("POCKET_TTS_PRECOMPUTE_BATCH", "16")
    env.setdefault("POCKET_TTS_DECODE_WORKERS", "3")
    return env


def last_progress(run_dir: Path) -> dict[str, Any]:
    """Latest train / valid / checkpoint events from the run's progress.jsonl."""
    out: dict[str, Any] = {}
    path = run_dir / "progress.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[ev.get("type", "?")] = ev
    return out


def run_training(
    config: Path,
    run_dir: Path,
    budget_sec: float,
    pocket_tts_dir: Path,
    project_dir: Path,
    gpus: int,
    log_every_steps: int = 250,
    stop_grace_sec: float = 900,
) -> dict[str, Any]:
    """Run training until it finishes or `budget_sec` elapses; return a summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runner.log"
    if gpus > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={gpus}", "--module", "training.train", str(config)]  # fmt: skip
    else:
        cmd = [sys.executable, "-m", "training.train", str(config)]
    env = training_env(pocket_tts_dir, project_dir)
    print(f"[runner] {' '.join(cmd)}\n[runner] budget {budget_sec / 3600:.2f} h, log {log_path}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=pocket_tts_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )  # fmt: skip
    stop_file = run_dir / "STOP"
    state = {"stop_sent": False}

    def watchdog() -> None:
        while proc.poll() is None:
            elapsed = time.time() - t0
            if not state["stop_sent"] and elapsed >= budget_sec:
                print(f"[runner] budget reached after {elapsed / 3600:.2f} h: requesting a checkpoint + stop", flush=True)
                stop_file.touch()
                state["stop_sent"] = True
            if state["stop_sent"] and elapsed >= budget_sec + stop_grace_sec:
                print("[runner] trainer did not stop in time: sending SIGTERM", flush=True)
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(120)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                return
            time.sleep(15)

    threading.Thread(target=watchdog, daemon=True).start()
    last_shown = -1
    with open(log_path, "a", buffering=1) as log:  # line-buffered: tail -f works during the run
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            m = _STEP.search(line)
            if m:
                step = int(m.group(1))
                if step <= 10 or step // log_every_steps > last_shown:
                    last_shown = step // log_every_steps
                    print(line.rstrip(), flush=True)
            elif _INTERESTING.search(line):
                print(line.rstrip(), flush=True)
    code = proc.wait()
    elapsed = time.time() - t0
    progress = last_progress(run_dir)
    summary = {
        "exit_code": code,
        "hours": round(elapsed / 3600, 3),
        "stopped_by_budget": state["stop_sent"],
        "last_checkpoint_step": progress.get("checkpoint", {}).get("step"),
        "last_train": progress.get("train", {}).get("metrics"),
        "last_valid": progress.get("valid", {}).get("metrics"),
        "export": str(run_dir / "model.safetensors") if (run_dir / "model.safetensors").exists() else None,
    }
    (run_dir / "runner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[runner] {json.dumps(summary)}", flush=True)
    if code != 0 and summary["export"] is None:
        raise RuntimeError(f"training failed (exit {code}) with nothing exported; see {log_path}")
    return summary
