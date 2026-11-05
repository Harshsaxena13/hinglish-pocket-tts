"""End-to-end smoke test on a laptop CPU with at most 50 real clips.

Runs every stage of the Kaggle pipeline on a tiny corpus: fetch clips through the
datasets-server API (no shard downloads) -> build -> align -> train a few steps
-> distill a few steps -> evaluate -> export -> synthesize. It checks plumbing,
not quality: the models train for seconds.

    python scripts/local_smoke_test.py --pocket-tts /path/to/pocket-tts --work /tmp/smoke

Without an HF token that has accepted kyutai/pocket-tts' terms, the ungated
"without voice cloning" weights are used (zeroed Mimi encoder): every code path
still runs, but the audio is meaningless. Downloads ~1.5 GB of model weights
(MMS aligner + the 6-layer model) into the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MAX_CLIPS = 50
# (name, repo, split, clips, columns to keep besides audio)
SOURCES = [
    ("mucs", "dianavdavidson/MUCS-Hinglish", "train", 18, ["transcript", "speaker_id"]),
    ("ivr_hi", "SPRINGLab/IndicVoices-R_Hindi", "train", 12, ["text", "verbatim", "speaker_id", "gender", "snr", "cer", "duration"]),
    ("indictts_hi", "SPRINGLab/IndicTTS-Hindi", "train", 10, ["text", "gender"]),
    ("indictts_en", "SPRINGLab/IndicTTS-English", "train", 6, ["text"]),
]  # fmt: skip
GATED_6L = "hf://kyutai/pocket-tts/languages/english_2026-09/model.safetensors@983151f13aaeab1b13c1e5e3c2c383d49a9edf3f"
UNGATED_6L = "hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-09/model.safetensors@e7205b6ee50e654a5ea19f0e9df2b0813b05e921"


def fetch_url(url: str, attempts: int = 5) -> bytes:
    """GET with retries: the datasets-server API returns transient 5xx under load."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError as exc:  # URLError / HTTPError / timeouts
            if i == attempts - 1:
                raise
            print(f"  retrying after {exc}")
            time.sleep(5 * 2**i)
    raise AssertionError("unreachable")


def get_json(url: str) -> dict:
    return json.loads(fetch_url(url))


def fetch(parquet_root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert sum(s[3] for s in SOURCES) <= MAX_CLIPS
    for name, repo, split, n, cols in SOURCES:
        out = parquet_root / name / "part-0.parquet"
        if out.exists():
            continue
        q = urllib.parse.urlencode({"dataset": repo, "config": "default", "split": split, "offset": 0, "length": n})
        rows = get_json(f"https://datasets-server.huggingface.co/rows?{q}")["rows"]
        table: dict[str, list] = {"audio": [], **{c: [] for c in cols}}
        for r in rows[:n]:
            row = r["row"]
            audio = row.get("audio") or row.get("audio_filepath")
            src = audio[0]["src"] if isinstance(audio, list) else audio["src"]
            table["audio"].append({"bytes": fetch_url(src), "path": src.rsplit("/", 1)[-1].split("?")[0]})
            for c in cols:
                table[c].append(row.get(c))
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), out)
        print(f"fetched {len(table['audio'])} clips of {repo} [{split}]")


def patched_model_config(pocket: Path, name: str, out_dir: Path, ungated: bool) -> Path:
    raw = yaml.safe_load((pocket / "pocket_tts" / "config" / f"{name}.yaml").read_text())
    if ungated:
        raw["weights_path"] = raw["weights_path_without_voice_cloning"]
    out = out_dir / f"{name}.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


def render(template: str, out: Path, overrides: dict, **variables: str) -> Path:
    from hinglish_tts.train_runner import render_config

    path = render_config(ROOT / "configs" / template, out, **variables)
    cfg = yaml.safe_load(path.read_text())
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pocket-tts", type=Path, required=True, help="pocket-tts checkout (patched)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    work, pocket = args.work.resolve(), args.pocket_tts.resolve()
    from hinglish_tts.pocket_patch import apply

    apply(pocket)
    sys.path.insert(0, str(pocket))
    from hinglish_tts.train_runner import run_training, training_env

    ungated = not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache" / "huggingface" / "token").exists()
    if ungated:
        print("NOTE: no HF token -> ungated weights with a zeroed Mimi encoder; audio will be meaningless")
    env = training_env(pocket, ROOT)
    env["POCKET_TTS_ALLOW_CPU"] = "1"
    os.environ.update({k: env[k] for k in ("POCKET_TTS_ALLOW_CPU",)})

    print("\n## 1. fetch <= 50 clips")
    fetch(work / "parquet")

    print("\n## 2. prepare clips (romanize, 24 kHz, trim, loudness, 1-15 s, 2% test split)")
    from hinglish_tts.data.prepare import align_and_finalize, prepare_clips

    recipe = yaml.safe_load((ROOT / "configs" / "data_recipe.yaml").read_text())
    recipe.update(valid_min=4, valid_max=4, workers=2)
    sources = []
    for s in recipe["sources"]:
        if (work / "parquet" / s["name"]).exists():
            s = {**s, "repo": str(work / "parquet" / s["name"]), "files": "*.parquet", "max_hours": 1.0, "sample_frac": 1.0}
            s.pop("max_shards", None)
            sources.append(s)
    recipe["sources"] = sources
    (work / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    print(json.dumps(prepare_clips(work / "recipe.yaml", work / "data")))

    print("\n## 3. align (torchaudio MMS_FA, CPU) + QA + STATS.md")
    print(json.dumps(align_and_finalize(work / "data", 1, env=env)))
    first = json.loads((work / "data" / "manifests" / "train_aligned.jsonl").read_text().splitlines()[0])
    assert list(first) == ["path", "duration", "transcript", "words"], list(first)
    print("aligned example:", first["transcript"], "\n ", [(w["word"], w["start"], w["end"]) for w in first["words"]][:8])
    print((work / "data" / "STATS.md").read_text())

    print("\n## 3b. pack for the HF dataset, unpack on a 'new machine' (RUN_MODE=train path)")
    from hinglish_tts.data.hub import pack, unpack

    print(json.dumps(pack(work / "data", work / "hub_dataset")))
    print(json.dumps(unpack(work / "hub_dataset", work / "data_pulled")))
    data = work / "data_pulled"  # everything below trains from the relocated copy

    cfg_dir = work / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    six = patched_model_config(pocket, "english_2026-09", cfg_dir, ungated)
    small = {"batch_size": 2, "grad_accum_steps": 1, "max_steps": args.steps, "data.loader_procs": 1,
             "data.num_bucket_batches": 1, "valid_freq": args.steps, "num_valid_batches": 1, "ckpt_freq": args.steps,
             "sample_freq": args.steps, "log_freq": 1, "optim.warmup_steps": 1}  # fmt: skip
    variables = {"POCKET_TTS": str(pocket), "DATA": str(data), "RUNS": str(work / "runs")}

    print("\n## 4. teacher fine-tune (6-layer stand-in for the 24-layer teacher, CPU)")
    teacher_cfg = render("hinglish_ft_t4.yaml", work / "runs" / "teacher.yaml", {**small, "model_config": str(six)}, **variables)
    print(json.dumps(run_training(teacher_cfg, work / "runs" / "teacher", 1800, pocket, ROOT, 1)))

    print("\n## 5. student distillation (CPU)")
    student_cfg = render("hinglish_distill_t4.yaml", work / "runs" / "student.yaml",
                         {**small, "model_config": str(six), "distill_teacher_config": str(six),
                          "student_init_from": UNGATED_6L if ungated else GATED_6L}, **variables)  # fmt: skip
    print(json.dumps(run_training(student_cfg, work / "runs" / "student", 1800, pocket, ROOT, 1)))

    print("\n## 6. evaluate (CPU, 2 sentences x 1 voice)")
    subprocess.run([sys.executable, "-m", "hinglish_tts.evaluate", "--systems", f"baseline=released:{six}@1.0",
                    f"student=run:{work / 'runs' / 'student'}@1.0", "--pocket-tts-dir", str(pocket),
                    "--test", str(data / "manifests" / "test_aligned.jsonl"), "--out", str(work / "eval"),
                    "--voices", "1", "--max-sentences", "2", "--no-sim", "--no-utmos"], check=True, env=env, cwd=ROOT)  # fmt: skip

    print("\n## 7. export + CPU benchmark")
    from hinglish_tts.export import export_model, model_card

    voices = [{"id": p.stem, "wav": p} for p in sorted((work / "eval" / "voices").glob("*.wav"))]
    info = export_model(work / "runs" / "student", six, work / "exports" / "hinglish-pocket-tts", voices,
                        ["Namaste, aaj ka mausam bahut accha hai."])  # fmt: skip
    (work / "exports" / "hinglish-pocket-tts" / "README.md").write_text(model_card("hinglish-pocket-tts", None, info, None,
                                                                                          json.loads((work / "data" / "stats.json").read_text())))
    print(json.dumps(info, indent=1))

    print("\n## 8. infer.py: export folder (default voice), then a raw training checkpoint")
    export = work / "exports" / "hinglish-pocket-tts"
    subprocess.run([sys.executable, str(ROOT / "infer.py"), "--model", str(export), "--threads", "2",
                    "--text", "Kal ki meeting 10:30 बजे है, please time pe aana.", "--out", str(work / "hello.wav")],
                   check=True, env=env, cwd=work)  # fmt: skip
    ckpt = sorted((work / "runs" / "student").glob("checkpoint_*.pt"))[-1]
    subprocess.run([sys.executable, str(ROOT / "infer.py"), "--model", str(export), "--checkpoint", str(ckpt),
                    "--voice", str(voices[0]["wav"]), "--text", "Aap kaise ho?", "--out", str(work / "ckpt.wav")],
                   check=True, env=env, cwd=work)  # fmt: skip
    import soundfile

    for wav in ("hello.wav", "ckpt.wav"):
        info_wav = soundfile.info(work / wav)
        assert info_wav.samplerate == 24000 and info_wav.frames > 0, (wav, info_wav)
    print(f"\nSMOKE TEST PASSED in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
