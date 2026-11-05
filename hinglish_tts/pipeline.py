"""End-to-end Hinglish Pocket TTS pipeline, sized for one 12 h Kaggle session on 2x T4.

    python -m hinglish_tts.pipeline all --mode setup     # prepare data, upload it, train, evaluate, export
    python -m hinglish_tts.pipeline all --mode train     # pull the prepared data from the Hub, then train...
    python -m hinglish_tts.pipeline smoke teacher        # selected stages

Stages (each is skipped once done; --force reruns):
    setup      clone pocket-tts at a pinned commit, patch it for T4, check gated model access
    data       romanization lexicon + clips (download, filter, romanize, 24 kHz, trim, loudness, 2% test)
    align      torchaudio MMS_FA word alignment, transcript QA, manifests + STATS.md
    push_data  upload the prepared corpus to a private HF dataset          (mode setup)
    pull_data  download it instead of preparing it                          (mode train)
    smoke      200-step training run: must reach its checkpoint with a finite loss
    teacher    fine-tune the 24-layer teacher                               (time-boxed)
    student    distill into the 6-layer CPU model                           (time-boxed)
    eval       baseline vs teacher vs student on held-out voices
    export     pocket-tts-ready folders, voices, model cards, Hub upload

Training stops --hard-stop-hours after HINGLISH_SESSION_START (set by the notebook's
first cell; default 11 h, leaving the last hour of a 12 h session for eval and
export). Checkpoints are mirrored to $HF_SYNC_REPO (see hf_sync.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hf_sync import RunSync, check_access, push_files, restore, sync_repo

PROJECT_DIR = Path(__file__).resolve().parent.parent
STAGES = ["setup", "data", "align", "push_data", "pull_data", "smoke", "teacher", "student", "eval", "export"]
MODES = {
    "setup": ["setup", "data", "align", "push_data", "smoke", "teacher", "student", "eval", "export"],
    "train": ["setup", "pull_data", "smoke", "teacher", "student", "eval", "export"],
}
PREREQUISITES = {"setup", "data", "align", "pull_data", "smoke"}  # a failure here stops the run
TEACHER_CONFIG = "english_2026-09_24l"
STUDENT_CONFIG = "english_2026-09"
BENCH_TEXTS = [
    "Namaste! Aaj ka mausam bahut accha hai, chalo bahar chalte hain.",
    "Mera laptop hang ho gaya, isliye main meeting thodi der se join karunga.",
    "Please make sure you submit the report before Friday evening.",
]
REQUIRED_MODULES = ["torch", "torchaudio", "transformers", "sphn", "soxr", "pyarrow", "jiwer", "einops", "sentencepiece", "huggingface_hub"]


@dataclass
class Settings:
    work: Path
    out: Path
    session_start: float
    hard_stop_hours: float
    teacher_share: float
    teacher_hours: float | None
    student_hours: float | None
    smoke_steps: int
    gpus: int
    hours_scale: float
    hf_repo: str | None
    dataset_repo: str
    skip_student: bool
    eval_voices: int
    force: bool
    recipe: Path = PROJECT_DIR / "configs" / "data_recipe.yaml"

    @property
    def pocket(self) -> Path:
        return self.work / "pocket-tts"

    @property
    def data(self) -> Path:
        return self.work / "data"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def lexicon(self) -> Path:
        return self.work / "lexicon" / "aksharantar_hi.tsv"

    @property
    def runs(self) -> Path:
        return self.out / "runs"

    @property
    def state_path(self) -> Path:
        return self.out / "pipeline_state.json"

    def elapsed_hours(self) -> float:
        return (time.time() - self.session_start) / 3600

    def train_hours_left(self) -> float:
        """Hours until the hard stop: no training step runs past it."""
        return self.hard_stop_hours - self.elapsed_hours()


def banner(s: Settings, msg: str) -> None:
    print(f"\n{'=' * 78}\n[{time.strftime('%H:%M:%S')}] {msg}  (session {s.elapsed_hours():.2f} h in, "
          f"training hard stop in {s.train_hours_left():.2f} h)\n{'=' * 78}", flush=True)  # fmt: skip


def load_state(s: Settings) -> dict[str, Any]:
    return json.loads(s.state_path.read_text()) if s.state_path.exists() else {}


def save_state(s: Settings, state: dict[str, Any]) -> None:
    s.state_path.parent.mkdir(parents=True, exist_ok=True)
    s.state_path.write_text(json.dumps(state, indent=2))


def run(cmd: list[str], **kw: Any) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def env_with_paths(s: Settings) -> dict[str, str]:
    from .train_runner import training_env

    return training_env(s.pocket, PROJECT_DIR)


# ----------------------------------------------------------------------------- setup + data


def stage_setup(s: Settings) -> dict[str, Any]:
    from .pocket_patch import POCKET_TTS_COMMIT, POCKET_TTS_REPO, apply

    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"missing packages {missing}: create the environment first (python -m hinglish_tts.kaggle_env)")
    if not (s.pocket / ".git").exists():
        s.pocket.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", POCKET_TTS_REPO, str(s.pocket)])
    run(["git", "-C", str(s.pocket), "checkout", "--quiet", POCKET_TTS_COMMIT])
    for line in apply(s.pocket):
        print("  ", line)
    import torch

    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    info = {"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "amp": "float16 + GradScaler" if caps and caps[0][0] < 8 else "bfloat16", "pocket_tts": POCKET_TTS_COMMIT}  # fmt: skip
    _check_base_model_access()
    print(json.dumps(info, indent=2))
    return info


def _check_base_model_access() -> None:
    """Fail in minutes, not after an hour of data prep, if the gated base weights are out of reach."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download("kyutai/pocket-tts", f"languages/{TEACHER_CONFIG}/tokenizer.json")
    except Exception as exc:  # noqa: BLE001 -- GatedRepoError, 401, 403
        who = "this HF_TOKEN's account" if os.environ.get("HF_TOKEN") else "anonymous access (HF_TOKEN is not set)"
        raise RuntimeError(
            f"cannot download kyutai/pocket-tts with {who}: {type(exc).__name__}. The model is gated: log in to "
            "https://huggingface.co/kyutai/pocket-tts with the token's account and accept the terms (auto-approved)."
        ) from exc
    print("kyutai/pocket-tts: access ok")


def _publish_stats(s: Settings) -> None:
    """Copy STATS.md into the persisted output (/kaggle/working/data/STATS.md) and show it."""
    stats = s.data / "STATS.md"
    if stats.exists():
        (s.out / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(stats, s.out / "data" / "STATS.md")
        print(stats.read_text())


def stage_data(s: Settings) -> dict[str, Any]:
    from .data.prepare import prepare_clips
    from .text.lexicon import build_aksharantar_lexicon

    lexicon: str | None = None
    try:
        lexicon = str(build_aksharantar_lexicon(s.lexicon))
    except Exception as exc:  # noqa: BLE001 -- the rules + curated lexicon still work
        print(f"WARNING: Aksharantar lexicon unavailable ({exc!r}); using rules + curated lexicon only")
    report = prepare_clips(s.recipe, s.data, lexicon=lexicon, hours_scale=s.hours_scale)
    _publish_stats(s)
    return {**report, "lexicon": lexicon}


def stage_align(s: Settings) -> dict[str, Any]:
    from .data.prepare import align_and_finalize

    report = align_and_finalize(s.data, max(1, s.gpus), env=env_with_paths(s))
    _publish_stats(s)
    return report


def stage_push_data(s: Settings) -> dict[str, Any]:
    from .data.hub import push

    problem = check_access(s.dataset_repo, repo_type="dataset")
    if problem:
        raise RuntimeError(f"not uploading the dataset: {problem}")
    push(s.data, s.dataset_repo, lexicon=s.lexicon)
    return {"repo": f"https://huggingface.co/datasets/{s.dataset_repo}"}


def stage_pull_data(s: Settings) -> dict[str, Any]:
    from .data.hub import pull

    counts = pull(s.data, s.dataset_repo, lexicon_out=s.lexicon)
    _publish_stats(s)
    return counts


# ----------------------------------------------------------------------------- training


def _out_of_memory(run_dir: Path) -> bool:
    log = run_dir / "runner.log"
    return log.exists() and "out of memory" in log.read_text(errors="ignore")[-200_000:].lower()


def _train(s: Settings, name: str, template: str, hours: float, overrides: dict[str, Any] | None = None,
           sync: bool = True) -> dict[str, Any]:  # fmt: skip
    """Train with a wall-clock budget; on CUDA OOM, halve the micro-batch (same
    effective batch via accumulation) and resume from the last checkpoint."""
    import yaml

    from .train_runner import render_config, run_training

    hours = min(hours, s.train_hours_left())
    if hours < 0.1:
        raise RuntimeError(f"only {hours:.2f} h left before the hard stop: not starting {name}")
    run_dir = s.runs / name
    config = render_config(
        PROJECT_DIR / "configs" / template, s.runs / f"{name}.yaml",
        POCKET_TTS=str(s.pocket), DATA=str(s.data), RUNS=str(s.runs),
    )  # fmt: skip
    cfg = yaml.safe_load(config.read_text())
    cfg["run_dir"] = str(run_dir)
    cfg.update(overrides or {})
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    deadline = time.time() + hours * 3600
    repo = sync_repo() if sync else None
    watcher = None
    if repo:  # resume from the Hub if a previous session got further than this one
        try:
            restore(repo, run_dir, s.out, "all")
            watcher = RunSync(repo, run_dir, s.out).start()
        except Exception as exc:  # noqa: BLE001 -- syncing is a convenience, never a reason not to train
            print(f"WARNING [hf_sync] disabled for {name}: {exc!r}", flush=True)
            watcher = None
    summary = None
    try:
        for attempt in range(3):
            if attempt:
                (run_dir / "runner.log").rename(run_dir / f"runner_oom{attempt}.log")
            try:
                summary = run_training(config, run_dir, deadline - time.time(), s.pocket, PROJECT_DIR, max(1, s.gpus))
            except RuntimeError:
                summary = None
            failed = summary is None or summary["exit_code"] != 0
            cfg = yaml.safe_load(config.read_text())
            if not (failed and _out_of_memory(run_dir) and cfg["batch_size"] > 1 and deadline - time.time() > 600):
                break
            cfg["batch_size"] //= 2
            cfg["grad_accum_steps"] *= 2
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"[{name}] CUDA out of memory: retrying with batch_size {cfg['batch_size']} x accum {cfg['grad_accum_steps']}")
    finally:
        if watcher is not None:
            watcher.stop()  # uploads the final (hard-stop) checkpoint
    if summary is None:
        raise RuntimeError(f"{name} training failed with nothing exported; see {run_dir / 'runner.log'}")
    cfg = yaml.safe_load(config.read_text())
    return {**summary, "batch_size": cfg["batch_size"], "grad_accum_steps": cfg["grad_accum_steps"]}


def stage_smoke(s: Settings) -> dict[str, Any]:
    """A short real run on the real data: catches fp16 overflow, OOM, DDP and data problems in
    ~15 minutes instead of hours in, and measures the speed the full run will get."""
    import math

    n = s.smoke_steps
    overrides = {"max_steps": n, "ckpt_freq": n, "valid_freq": n, "sample_freq": n, "log_freq": 10, "num_valid_batches": 4}
    summary = _train(s, "smoke", "hinglish_ft_t4.yaml", 1.0, overrides, sync=False)
    run_dir = s.runs / "smoke"
    events = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().splitlines() if l.strip()]
    speeds = sorted(e["it_s"] for e in events if e.get("type") == "train" and e.get("step", 0) > 20 and e.get("it_s"))
    loss = (summary.get("last_train") or {}).get("loss")
    log = (run_dir / "runner.log").read_text(errors="ignore")
    peak = next((l.split("peak GPU memory")[1].strip() for l in log.splitlines() if "peak GPU memory" in l), None)
    result = {
        "steps": summary.get("last_checkpoint_step"), "loss": loss,
        "it_s": round(speeds[len(speeds) // 2], 3) if speeds else None, "peak_gpu_memory": peak,
        "overflow_warnings": log.count("fp16 overflow"), "batch_size": summary["batch_size"],
        "grad_accum_steps": summary["grad_accum_steps"],
    }  # fmt: skip
    if result["it_s"]:
        result["steps_per_hour"] = round(result["it_s"] * 3600)
    print(f"[smoke] {json.dumps(result)}")
    for f in [*run_dir.glob("*.pt"), run_dir / "model.safetensors"]:  # keep logs and samples, free ~6 GB
        f.unlink(missing_ok=True)
    if summary["exit_code"] != 0 or result["steps"] != n or loss is None or not math.isfinite(loss):
        raise RuntimeError(f"smoke run failed: {result}; see {run_dir / 'runner.log'}")
    return result


def _batch_from_smoke(s: Settings) -> dict[str, Any]:
    smoke = load_state(s).get("smoke", {}).get("result") or {}
    return {k: smoke[k] for k in ("batch_size", "grad_accum_steps") if k in smoke}


def stage_teacher(s: Settings) -> dict[str, Any]:
    left = s.train_hours_left()
    hours = s.teacher_hours or (left * s.teacher_share if not s.skip_student else left)
    print(f"teacher budget: {hours:.2f} h (hard stop in {left:.2f} h)")
    return _train(s, "teacher", "hinglish_ft_t4.yaml", hours, _batch_from_smoke(s))


def stage_student(s: Settings) -> dict[str, Any]:
    _restore(s, "teacher", "export")
    if not (s.runs / "teacher" / "model.safetensors").exists():
        raise RuntimeError("no teacher export at runs/teacher/model.safetensors: run the teacher stage first")
    hours = s.student_hours or s.train_hours_left()
    print(f"student budget: {hours:.2f} h (hard stop in {s.train_hours_left():.2f} h)")
    return _train(s, "student", "hinglish_distill_t4.yaml", hours)


def _trained(s: Settings, name: str) -> bool:
    return any((s.runs / name).glob("checkpoint_*.pt"))


def _restore(s: Settings, name: str, what: str) -> None:
    """Fetch a run from the sync repo when this session does not have it locally."""
    repo = sync_repo()
    if repo:
        try:
            restore(repo, s.runs / name, s.out, what)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING [hf_sync] could not restore {name}: {exc!r}")


# ----------------------------------------------------------------------------- eval + export


def stage_eval(s: Settings) -> dict[str, Any]:
    for name in ("teacher", "student"):
        _restore(s, name, "checkpoint")
    systems = [f"english_baseline=released:{STUDENT_CONFIG}@1.0"]
    if _trained(s, "teacher"):
        systems.append(f"hinglish_teacher_24l=run:{s.runs / 'teacher'}@2.0")
    if _trained(s, "student"):
        systems.append(f"hinglish_student_6l=run:{s.runs / 'student'}@1.0")
    cmd = [sys.executable, "-m", "hinglish_tts.evaluate", "--systems", *systems, "--pocket-tts-dir", str(s.pocket),
           "--test", str(s.manifests / "test_aligned.jsonl"), "--out", str(s.out / "eval"), "--voices", str(s.eval_voices)]  # fmt: skip
    run(cmd, env=env_with_paths(s), cwd=PROJECT_DIR)
    return json.loads((s.out / "eval" / "results.json").read_text())


def stage_export(s: Settings) -> dict[str, Any]:
    from .evaluate import pick_voices
    from .export import export_model, model_card, push_to_hub

    sys.path.insert(0, str(s.pocket))
    for name in ("teacher", "student"):
        _restore(s, name, "export")
    results_md = (s.out / "eval" / "results.md").read_text() if (s.out / "eval" / "results.md").exists() else None
    data_report = json.loads((s.data / "stats.json").read_text()) if (s.data / "stats.json").exists() else None
    voices = []
    voice_dir = s.out / "eval" / "voices"
    if voice_dir.exists():
        voices = [{"id": p.stem, "wav": p} for p in sorted(voice_dir.glob("*.wav"))]
        # The model copies the prompt's acoustics: ship full-band voices, not 16 kHz MUCS ones.
        voices = [v for v in voices if not v["id"].startswith("mucs")] or voices
    elif (s.manifests / "test_aligned.jsonl").exists():
        import soundfile

        tmp = s.out / "exports" / "_voices"
        tmp.mkdir(parents=True, exist_ok=True)
        for v in pick_voices(s.manifests / "test_aligned.jsonl", s.eval_voices):
            wav, sr = soundfile.read(v["path"])
            soundfile.write(tmp / f"{v['voice_id']}.wav", wav[: int(v["prompt_sec"] * sr)], sr)
            voices.append({"id": v["voice_id"], "wav": tmp / f"{v['voice_id']}.wav"})
    out: dict[str, Any] = {}
    sync = sync_repo()
    targets = [("student", STUDENT_CONFIG, "hinglish-pocket-tts"), ("teacher", TEACHER_CONFIG, "hinglish-pocket-tts-24l")]
    for run_name, base, folder_name in targets:
        run_dir = s.runs / run_name
        if not (run_dir / "model.safetensors").exists():
            continue
        base_cfg = s.pocket / "pocket_tts" / "config" / f"{base}.yaml"
        folder = s.out / "exports" / folder_name
        info = export_model(run_dir, base_cfg, folder, voices, BENCH_TEXTS if run_name == "student" else None)
        if s.lexicon.exists():
            shutil.copy2(s.lexicon, folder / "lexicon_hi.tsv")
        public = (s.hf_repo if run_name == "student" else f"{s.hf_repo}-24l") if s.hf_repo else None
        (folder / "README.md").write_text(model_card(folder_name, public, info, results_md, data_report), encoding="utf-8")
        uploads = [(public, "")] if public else []
        if sync and run_name == "student":  # the CPU model, ready for infer.py, next to the checkpoints
            uploads.append((sync, f"exports/{folder_name}"))
        for repo, prefix in uploads:
            try:
                info.setdefault("hub", []).append(push_to_hub(folder, repo, base_cfg, path_in_repo=prefix, pin=repo != sync))
            except Exception as exc:  # noqa: BLE001 -- the local export is still complete
                print(f"WARNING: upload to {repo}/{prefix} failed: {exc!r}")
        out[folder_name] = info
        print(f"exported {folder}: voices {info.get('voices')}, cpu_rtf={info.get('cpu_rtf_2_threads')}, hub={info.get('hub')}")
    shutil.rmtree(s.out / "exports" / "_voices", ignore_errors=True)
    for opt in s.runs.glob("*/optim_*.pt"):  # resumable state is on the Hub (or not needed any more)
        opt.unlink()
    return out


STAGE_FUNCS = {
    "setup": stage_setup, "data": stage_data, "align": stage_align, "push_data": stage_push_data,
    "pull_data": stage_pull_data, "smoke": stage_smoke, "teacher": stage_teacher, "student": stage_student,
    "eval": stage_eval, "export": stage_export,
}  # fmt: skip


def _outputs_present(s: Settings, stage: str) -> bool:
    """Setup/data/align write to /kaggle/temp, which a new session starts without."""
    if stage == "setup":
        return (s.pocket / "training" / "train.py").exists()
    if stage == "data":
        return (s.manifests / "clips.jsonl").exists()
    if stage in ("align", "pull_data"):
        return (s.manifests / "train_aligned.jsonl").exists()
    return True


def _sync_state(s: Settings) -> None:
    repo = sync_repo()
    if not repo:
        return
    try:
        push_files(repo, {s.state_path: "pipeline_state.json", s.data / "STATS.md": "data/STATS.md",
                          s.out / "eval" / "results.md": "eval/results.md"})  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- bookkeeping only
        print(f"[hf_sync] could not upload pipeline state: {exc!r}")


def run_stages(s: Settings, stages: list[str]) -> None:
    state = load_state(s)
    for stage in stages:
        if stage == "student" and s.skip_student:
            continue
        if state.get(stage, {}).get("done") and not s.force and _outputs_present(s, stage):
            print(f"[{stage}] already done, skipping (use --force to rerun)")
            continue
        banner(s, f"stage: {stage}")
        t0 = time.time()
        try:
            result = STAGE_FUNCS[stage](s)
        except Exception as exc:
            state[stage] = {"done": False, "error": repr(exc), "hours": round((time.time() - t0) / 3600, 3)}
            save_state(s, state)
            _sync_state(s)
            if stage in PREREQUISITES:
                raise
            # Training stages leave usable checkpoints; later stages evaluate/export whatever exists.
            print(f"ERROR in stage {stage}: {exc!r} -- continuing with the next stage", flush=True)
            continue
        state[stage] = {"done": True, "hours": round((time.time() - t0) / 3600, 3), "result": result}
        save_state(s, state)
        _sync_state(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    ap.add_argument("--mode", choices=list(MODES), default=os.environ.get("RUN_MODE", "setup"), help="stage list for 'all'")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("HINGLISH_WORK", "/kaggle/temp/hinglish")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("HINGLISH_OUT", "/kaggle/working")))
    ap.add_argument("--hard-stop-hours", type=float, default=float(os.environ.get("HARD_STOP_HOURS", 11.0)),
                    help="no training after this many hours since HINGLISH_SESSION_START")  # fmt: skip
    ap.add_argument("--teacher-share", type=float, default=0.65, help="share of the training time for the teacher")
    ap.add_argument("--teacher-hours", type=float, default=None)
    ap.add_argument("--student-hours", type=float, default=None)
    ap.add_argument("--smoke-steps", type=int, default=int(os.environ.get("SMOKE_STEPS", 200)))
    ap.add_argument("--hours-scale", type=float, default=1.0, help="scale every data source's hour budget")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO_ID") or None, help="optional public model repo")
    ap.add_argument("--dataset-repo", default=os.environ.get("HF_DATASET_REPO") or "harshsxna/hinglish-tts-data")
    ap.add_argument("--skip-student", action="store_true")
    ap.add_argument("--eval-voices", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.gpus is None:
        import torch

        args.gpus = torch.cuda.device_count()
    s = Settings(
        work=args.work, out=args.out, session_start=float(os.environ.get("HINGLISH_SESSION_START", time.time())),
        hard_stop_hours=args.hard_stop_hours, teacher_share=args.teacher_share, teacher_hours=args.teacher_hours,
        student_hours=args.student_hours, smoke_steps=args.smoke_steps, gpus=args.gpus, hours_scale=args.hours_scale,
        hf_repo=args.hf_repo, dataset_repo=args.dataset_repo, skip_student=args.skip_student,
        eval_voices=args.eval_voices, force=args.force,
    )  # fmt: skip
    sys.path.insert(0, str(s.pocket))
    repo = sync_repo()
    if repo:
        problem = check_access(repo)
        if problem:
            print(f"\n{'!' * 78}\nWARNING: checkpoint sync to {repo} is OFF for this run: {problem}.\n"
                  f"Set HF_SYNC_REPO in the first cell to a repo this token can write.\n{'!' * 78}\n")  # fmt: skip
            os.environ["HF_SYNC_REPO"] = ""
            repo = None
    if repo and not s.state_path.exists():  # a new session: continue where the last one stopped
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo, "pipeline_state.json", local_dir=s.out)
            print(f"[hf_sync] restored pipeline_state.json from https://huggingface.co/{repo}")
        except Exception:  # noqa: BLE001 -- first session, or no access
            pass
    status = f"checkpoints sync to https://huggingface.co/{repo}" if repo else "checkpoint sync disabled"
    print(f"[hf_sync] {status}")
    run_stages(s, MODES[args.mode] if "all" in args.stages else args.stages)


if __name__ == "__main__":
    main()
