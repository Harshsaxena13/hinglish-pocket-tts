"""Create the uv environment with the PyTorch build that matches this machine's CUDA driver.

Standard library only: it runs before the project's environment exists.

    python -m hinglish_tts.kaggle_env            # uv sync + verify on the GPU; prints the env's python

The driver's CUDA version (from nvidia-smi) bounds which builds can run: CUDA 13
wheels need a 13.x driver, while CUDA 12.x wheels run on any 12.x driver (minor
version compatibility). Candidates are tried newest first; each one is installed
with `uv sync --extra <build>` and must pass a check *inside the uv env*:
torch.cuda.is_available(), an fp16 matmul on every GPU (catches "no kernel image
for this architecture") and torchaudio's forced-alignment op. The first build
that passes wins. Without a GPU it installs the CPU build (laptops).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

# (uv extra, minimum driver CUDA version)
CUDA_BUILDS = [("cu130", (13, 0)), ("cu128", (12, 0)), ("cu126", (12, 0))]

VERIFY = r"""
import json, torch, torchaudio, torchaudio.functional as F
info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available(), "gpus": []}
if info["available"]:
    for i in range(torch.cuda.device_count()):
        x = torch.randn(256, 256, device=f"cuda:{i}", dtype=torch.float16)
        (x @ x).sum().item()  # raises if the build has no kernels for this GPU
        info["gpus"].append(f"{torch.cuda.get_device_name(i)} (sm_{''.join(map(str, torch.cuda.get_device_capability(i)))})")
    F.forced_align(torch.zeros(1, 4, 3).log_softmax(-1).cuda(), torch.tensor([[1]], dtype=torch.int32).cuda(), blank=0)
info["torchaudio"] = torchaudio.__version__
print("VERIFY " + json.dumps(info))
"""


def driver_cuda_version(nvidia_smi_output: str | None = None) -> tuple[int, int] | None:
    """Highest CUDA version the installed driver supports, or None without a GPU."""
    if nvidia_smi_output is None:
        try:
            nvidia_smi_output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_output or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def candidate_builds(cuda: tuple[int, int] | None) -> list[str]:
    if cuda is None:
        return ["cpu"]
    return [extra for extra, need in CUDA_BUILDS if cuda >= need] or ["cpu"]


def env_python() -> str:
    venv = os.environ.get("UV_PROJECT_ENVIRONMENT", ".venv")
    return os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")


def try_build(extra: str, project_dir: str) -> dict | None:
    print(f"\n[env] uv sync --extra {extra}", flush=True)
    if subprocess.run(["uv", "sync", "--extra", extra], cwd=project_dir).returncode != 0:
        return None
    out = subprocess.run([env_python(), "-c", VERIFY], cwd=project_dir, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("VERIFY ")), None)
    if line is None:
        print(f"[env] {extra}: verification failed\n{out.stderr[-2000:]}", flush=True)
        return None
    info = json.loads(line[len("VERIFY "):])
    if extra != "cpu" and not info["available"]:
        print(f"[env] {extra}: torch.cuda.is_available() is False ({info})", flush=True)
        return None
    return {"extra": extra, **info}


def main() -> None:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cuda = driver_cuda_version()
    builds = candidate_builds(cuda)
    print(f"[env] driver CUDA {cuda}, candidate builds {builds}", flush=True)
    for extra in builds:
        info = try_build(extra, project_dir)
        if info:
            print(f"[env] OK: {json.dumps(info)}\n[env] python: {env_python()}", flush=True)
            return
    sys.exit(f"[env] no PyTorch build in {builds} works on this machine")


if __name__ == "__main__":
    main()
