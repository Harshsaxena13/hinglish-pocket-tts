"""Generate kaggle/notebook.ipynb (cells defined here, so the notebook stays reviewable in diffs).

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}  # fmt: skip


CELLS = [
    md("""
# Hinglish Pocket TTS on Kaggle (2x T4)

Fine-tunes [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for Indian Hinglish (romanized
Hindi-English code-switching, Indian accent), so it reads text like
*"Mera laptop hang ho gaya, isliye meeting thodi der se join karunga."*: a 24-layer teacher, distilled into
the 6-layer CPU model.

**Before running:** *Settings* -> Accelerator **GPU T4 x2**, Internet **on**. *Add-ons -> Secrets*:
`HF_TOKEN` (write access; its account must have accepted the terms of
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), and optionally of
[ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) for more Indian-English voices);
`GITHUB_TOKEN` only if the project repo is private.
Then *Save Version -> Save & Run All (Commit)*: the run keeps going without the browser, and everything worth
keeping (runs, eval, exports) lands in `/kaggle/working`, the version's output.

| `RUN_MODE` | what runs |
|---|---|
| `"setup"` | download -> prepare (romanize, align, QA) -> upload the corpus to the private HF dataset -> 200-step smoke test -> full training -> eval -> export |
| `"train"` | pull the prepared corpus from the HF dataset (minutes instead of ~1 h) -> smoke test -> training -> eval -> export |

Every checkpoint (+ `progress.jsonl`, audio samples) is mirrored to `HF_SYNC_REPO` (latest only); a later
session resumes from it automatically. Training stops `HARD_STOP_HOURS` after the first cell ran.
Re-running a cell skips the stages that already finished.
"""),
    code("""
import os, subprocess, time
os.environ.setdefault("HINGLISH_SESSION_START", str(time.time()))  # the clock for the hard stop

# ---- settings ------------------------------------------------------------------------------
RUN_MODE = "setup"          # "setup" = prepare + upload data, then train | "train" = pull data from HF, then train
REPO_URL = "https://github.com/Harshsaxena13/hinglish-pocket-tts.git"  # this project
REPO_REF = "main"
HF_SYNC_REPO = "harshsxna/pocket-tts-hinglish"   # private model repo: latest checkpoint, progress, samples, export
HF_DATASET_REPO = "harshsxna/hinglish-tts-data"  # private dataset repo: the prepared corpus
HARD_STOP_HOURS = 11.0      # no training step after this many hours into the session
SMOKE_STEPS = 200
DATA_HOURS_SCALE = 1.0      # 0.1 for a quick dry run of the whole notebook
SKIP_STUDENT = False        # True: all training time to the 24-layer teacher
TEACHER_HOURS = None        # None: ~65% of the training time left; a number fixes it (the hard stop still applies)
STUDENT_HOURS = None        # None: whatever is left before the hard stop
# ---------------------------------------------------------------------------------------------

assert RUN_MODE in ("setup", "train"), RUN_MODE
WORK, OUT = "/kaggle/temp", "/kaggle/working"   # temp: wiped after the session; working: kept, 20 GB
SRC, VENV = f"{WORK}/hinglish-pocket-tts", f"{WORK}/venv"
PY = f"{VENV}/bin/python"
os.environ.update({
    "RUN_MODE": RUN_MODE, "HINGLISH_WORK": f"{WORK}/hinglish", "HINGLISH_OUT": OUT,
    "HF_HOME": f"{WORK}/hf_cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "UV_CACHE_DIR": f"{WORK}/uv_cache", "UV_PROJECT_ENVIRONMENT": VENV, "UV_LINK_MODE": "copy",
    "HF_SYNC_REPO": HF_SYNC_REPO, "HF_DATASET_REPO": HF_DATASET_REPO,
    "HARD_STOP_HOURS": str(HARD_STOP_HOURS), "SMOKE_STEPS": str(SMOKE_STEPS),
})
ARGS = f"--hours-scale {DATA_HOURS_SCALE}" + (" --skip-student" if SKIP_STUDENT else "")
ARGS += (f" --teacher-hours {TEACHER_HOURS}" if TEACHER_HOURS else "") + (f" --student-hours {STUDENT_HOURS}" if STUDENT_HOURS else "")


def sh(cmd: str, check: bool = True) -> int:
    \"\"\"Run a shell command, stream its output into the notebook, raise on failure.\"\"\"
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0 and check:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd}")
    return proc.returncode
"""),
    md("## 1. GPU check"),
    code("""
sh("nvidia-smi")
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()
print(f"{len(smi)} GPU(s):", smi)
if len(smi) < 2 or not all("T4" in g for g in smi):
    print("WARNING: the configs are sized for 2x T4 (16 GB each); pick 'GPU T4 x2' in Settings -> Accelerator")
sh(f"mkdir -p {WORK} && df -h {WORK} {OUT}")
"""),
    md("## 2. Hugging Face login (Kaggle secret `HF_TOKEN`)"),
    code("""
from kaggle_secrets import UserSecretsClient

_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = _secrets.get_secret("HF_TOKEN")   # never printed, never written to the repo
try:
    GITHUB_TOKEN = _secrets.get_secret("GITHUB_TOKEN")
except Exception:
    GITHUB_TOKEN = ""
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
me = HfApi().whoami()
print("logged in as", me["name"], "| token role:", me.get("auth", {}).get("accessToken", {}).get("role"))
for repo in (HF_SYNC_REPO, HF_DATASET_REPO):
    if repo.split("/")[0] not in {me["name"], *[o["name"] for o in me.get("orgs", [])]}:
        print(f"WARNING: {me['name']} cannot write to {repo}; change the repo names in the settings cell")
"""),
    md("## 3. Clone the project"),
    code("""
url = REPO_URL.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@") if GITHUB_TOKEN else REPO_URL
print(f"$ git clone {REPO_URL} ({REPO_REF}) -> {SRC}")
subprocess.run(f"rm -rf {SRC} && git clone -q --depth 1 -b {REPO_REF} {url} {SRC} && git -C {SRC} remote set-url origin {REPO_URL}",
               shell=True, check=True, capture_output=True)  # captured: the URL may carry the GitHub token
sh(f"git -C {SRC} log -1 --oneline")
os.chdir(SRC)
"""),
    md("""
## 4. Environment: `uv sync` with the PyTorch build that matches the CUDA driver

`hinglish_tts.kaggle_env` reads the driver's CUDA version, then tries the matching builds (`cu130`, `cu128`,
`cu126`) with `uv sync --extra <build>`. Each candidate must pass a check inside the uv env:
`torch.cuda.is_available()`, an fp16 matmul on both GPUs, and torchaudio's alignment op.
"""),
    code("""
sh("pip install -q uv")
sh("python -m hinglish_tts.kaggle_env")
sh(f'{PY} -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"')
"""),
    md("## 5. Pocket TTS training code (pinned commit + T4/fp16 patch), gated-model access check"),
    code('sh(f"{PY} -m hinglish_tts.pipeline setup {ARGS}")'),
    md("## 6. Download (`RUN_MODE='setup'`)\nRomanization lexicon (Aksharantar), then every corpus shard: filter, romanize (digits spelled out), 24 kHz, trim, loudness-normalize, 1-15 s; 2% test split."),
    code("""
if RUN_MODE == "setup":
    sh(f"{PY} -m hinglish_tts.pipeline data {ARGS}")
"""),
    md("## 7. Prepare: torchaudio MMS_FA alignment + transcript QA -> manifests + `STATS.md` (`RUN_MODE='setup'`)\nWord timestamps for the loader's prompt/target cuts; clips whose transcript doesn't match the audio are dropped."),
    code("""
if RUN_MODE == "setup":
    sh(f"{PY} -m hinglish_tts.pipeline align {ARGS}")
"""),
    md("## 8. Upload the prepared corpus to the private HF dataset (`setup`) / pull it (`train`)"),
    code("""
if RUN_MODE == "setup":
    sh(f"{PY} -m hinglish_tts.pipeline push_data {ARGS}", check=False)  # training does not depend on the upload
else:
    sh(f"{PY} -m hinglish_tts.pipeline pull_data {ARGS}")
"""),
    md("## 9. Smoke test: 200 training steps\nMust reach its checkpoint with a finite loss; reports speed, peak GPU memory and fp16 overflows. Stops the notebook if it fails."),
    code('sh(f"{PY} -m hinglish_tts.pipeline smoke {ARGS}")'),
    md("""
## 10. Full training, hard stop at `HARD_STOP_HOURS`
24-layer teacher (~65% of the time left), then the 6-layer distilled student. The student learns the teacher's
guided (CFG 2) output, so it runs in one pass with the stock pocket-tts runtime. Checkpoints and audio samples
every ~1000 steps in `/kaggle/working/runs/{teacher,student}`, mirrored to `HF_SYNC_REPO`.
"""),
    code('sh(f"{PY} -m hinglish_tts.pipeline teacher student {ARGS}", check=False)'),
    md("""
## 11. Evaluate and export
English baseline vs Hinglish teacher vs Hinglish student, with voice prompts from the held-out test split. Then
pocket-tts-ready folders in `/kaggle/working/exports` (weights, tokenizer, config, voices, model card); the
6-layer one is also uploaded to `HF_SYNC_REPO`.
"""),
    code('sh(f"{PY} -m hinglish_tts.pipeline eval export {ARGS}", check=False)'),
    md("## 12. Results"),
    code("""
import glob, json
from pathlib import Path
from IPython.display import Audio, Markdown, display

out = Path(OUT)
if (out / "pipeline_state.json").exists():
    state = json.loads((out / "pipeline_state.json").read_text())
    display(Markdown("**Stages:** " + ", ".join(f"{k}: {'ok' if v.get('done') else 'FAILED'} ({v.get('hours', 0):.2f} h)" for k, v in state.items())))
    if "smoke" in state:
        print("smoke:", state["smoke"].get("result"))
for f in ("data/STATS.md", "eval/results.md"):
    if (out / f).exists():
        display(Markdown((out / f).read_text()))
if (out / "eval/index.html").exists():
    print(f"Listening page: {OUT}/eval/index.html (download the eval folder)")
for system in ["english_baseline", "hinglish_teacher_24l", "hinglish_student_6l"]:
    for f in sorted(glob.glob(f"{OUT}/eval/{system}/*_0[67].wav"))[:2]:
        display(Markdown(f"`{system}` - {Path(f).name}")); display(Audio(f))
"""),
    md("## 13. Try it: `infer.py` with the exported 6-layer model (CPU)"),
    code("""
export = f"{OUT}/exports/hinglish-pocket-tts"
if Path(export).exists():
    sh(f'{PY} infer.py --model {export} --device cpu --out {OUT}/demo.wav '
       f'--text "Aapka order ₹1,499 ka hai aur वो कल शाम तक deliver हो जाएगा."')
    display(Audio(f"{OUT}/demo.wav"))
"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = ROOT / "kaggle" / "notebook.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
