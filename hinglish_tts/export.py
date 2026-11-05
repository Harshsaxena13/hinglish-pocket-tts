"""Package a trained model for the stock pocket-tts runtime (and optionally the Hub).

    <out>/<name>/
        model.safetensors     flow_lm (EMA) + mimi weights, pocket-tts format
        tokenizer.json        the English tokenizer the model reads romanized text with
        config_local.yaml     absolute local paths: TTSModel.load_model(config=...)
        config.yaml           hf:// paths pinned to the upload commit (after --push)
        voices/<id>.wav|.safetensors   ready-made Hinglish voices (prompt + KV state)
        README.md             model card with usage, data, recipe and eval results
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("hinglish.export")

HINDI_PUNCT = {"।": ".", "॥": "."}


def _config_dict(base_config: Path, weights: str, tokenizer: str) -> dict[str, Any]:
    raw = yaml.safe_load(base_config.read_text())
    raw["weights_path"] = weights
    raw.pop("weights_path_without_voice_cloning", None)
    raw["flow_lm"]["lookup_table"]["tokenizer_path"] = tokenizer
    raw.setdefault("replace_characters", {}).update(HINDI_PUNCT)
    raw["default_temperature"] = 0.3
    return raw


def _write_yaml(path: Path, data: dict[str, Any], header: str) -> None:
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def export_model(
    run_dir: Path, base_config: Path, out_dir: Path, voices: list[dict[str, Any]] | None = None,
    bench_texts: list[str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Copy weights + tokenizer, write configs, export voices, benchmark CPU speed."""
    from pocket_tts.utils.utils import download_if_necessary

    out_dir.mkdir(parents=True, exist_ok=True)
    src = run_dir / "model.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: did the run save a checkpoint?")
    shutil.copy2(src, out_dir / "model.safetensors")
    raw = yaml.safe_load(base_config.read_text())
    tok_src = download_if_necessary(raw["flow_lm"]["lookup_table"]["tokenizer_path"])
    shutil.copy2(tok_src, out_dir / "tokenizer.json")
    local = _config_dict(base_config, str((out_dir / "model.safetensors").resolve()), str((out_dir / "tokenizer.json").resolve()))
    header = f"# Hinglish Pocket TTS ({run_dir.name}), derived from {base_config.name}.\n"
    _write_yaml(out_dir / "config_local.yaml", local, header)

    info: dict[str, Any] = {"run": str(run_dir), "base_config": base_config.name}
    progress = run_dir / "runner_summary.json"
    if progress.exists():
        info["training"] = json.loads(progress.read_text())

    from pocket_tts import TTSModel, export_model_state

    model = TTSModel.load_model(config=str(out_dir / "config_local.yaml"))
    (out_dir / "voices").mkdir(exist_ok=True)
    exported = []
    for v in voices or []:
        wav = out_dir / "voices" / f"{v['id']}.wav"
        shutil.copy2(v["wav"], wav)
        state = model.get_state_for_audio_prompt(str(wav))
        export_model_state(state, out_dir / "voices" / f"{v['id']}.safetensors")
        exported.append(v["id"])
    info["voices"] = exported

    if bench_texts and exported:
        import torch

        torch.set_num_threads(2)
        state = model.get_state_for_audio_prompt(str(out_dir / "voices" / f"{exported[0]}.safetensors"))
        t0, audio_sec = time.time(), 0.0
        for text in bench_texts:
            audio = model.generate_audio(state, text)
            audio_sec += audio.shape[-1] / model.sample_rate
        info["cpu_rtf_2_threads"] = round((time.time() - t0) / max(audio_sec, 1e-6), 3)
    (out_dir / "export_info.json").write_text(json.dumps(info, indent=2))
    return info


def model_card(name: str, repo_id: str | None, info: dict[str, Any], results_md: str | None, data_report: dict[str, Any] | None) -> str:
    cfg = f"hf://{repo_id}/config.yaml" if repo_id else "./config_local.yaml"
    data_lines = ""
    if data_report:
        rows = [f"| {k} | {v['hours']['train']} | {v['kept']} | {v['speakers']} |" for k, v in data_report.get("sources", {}).items()]
        data_lines = "| source | train hours | clips | speakers |\n|---|---|---|---|\n" + "\n".join(rows)
    rtf = info.get("cpu_rtf_2_threads")
    return f"""---
license: cc-by-4.0
language: [hi, en]
tags: [text-to-speech, tts, hinglish, code-switching, pocket-tts, voice-cloning]
base_model: kyutai/pocket-tts
---

# {name}

Kyutai [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) fine-tuned for **Hinglish**:
Indian-accented speech with Hindi-English code-switching, read from **romanized** text
(`"Kal ki meeting cancel ho gayi hai, toh hum Monday ko milte hain."`). It keeps the stock
architecture, so it runs on CPU with the regular `pocket-tts` package and clones voices
from a short prompt.

## Usage

```bash
uvx pocket-tts generate --config {cfg} \\
    --voice voices/{(info.get('voices') or ['<voice>'])[0]}.wav \\
    --text "Namaste! Aaj ka mausam bahut accha hai, chalo bahar chalte hain."
```

The model was trained on romanized Hinglish with numbers spelled out. Raw text with
Devanagari, digits, currency or emoji should go through the bundled frontend first:

```python
from hinglish_tts.synthesize import HinglishTTS
tts = HinglishTTS("{cfg}", voice="voices/<voice>.wav")
audio = tts("Mera order ₹1,499 ka tha aur वो कल deliver होगा.")  # 24 kHz numpy array
```

{f"CPU speed: real-time factor {rtf} with 2 threads (lower is faster)." if rtf else ""}

## Training

1. **Teacher**: Kyutai's 24-layer `english_2026-09_24l` fine-tuned on the data below (English
   tokenizer kept, sampler head frozen, fp16 on 2x T4).
2. **Student**: the 6-layer `english_2026-09` distilled from that teacher with
   classifier-free guidance (coefficient 2) baked in, so it runs in a single pass.

Transcripts were romanized with a Hinglish-convention transliterator (schwa deletion,
casual spellings, English loanwords mapped back to English, acronyms restored), spelling
variants were sampled for robustness, and every clip was force-aligned with MMS; clips
whose transcript did not match the audio were dropped.

{data_lines}

## Evaluation

Held-out speakers, curated Hindi / code-mixed / English sentences. Intelligibility is measured
with a romanized-text CTC model (MMS), see the training repo for details.

{results_md or "(not run)"}

## Limitations

- Expects romanized Hinglish; use the frontend for Devanagari and numbers.
- Part of the code-switched data is 16 kHz; use a clean, full-band voice prompt for best quality.
- The model imitates the prompt's voice: obtain consent before cloning anyone's voice.

## Licenses

Base model: Kyutai Pocket TTS (CC-BY-4.0). Training data: MUCS 2021 Hinglish (CC-BY-4.0),
IndicVoices-R (CC-BY-4.0), IIT Madras Indic TTS (see its license), Svarah (CC-BY-4.0, if used).
Check every dataset's terms before commercial use.
"""


def push_to_hub(folder: Path, repo_id: str, base_config: Path, private: bool = True, path_in_repo: str = "",
                pin: bool = True) -> str:  # fmt: skip
    """Upload `folder` (optionally into a sub-folder), then write config.yaml with hf:// paths,
    so `pocket-tts --config hf://<repo>/<path>/config.yaml` works.

    pin=True pins the paths to the upload commit (reproducible). Use pin=False for the
    checkpoint-sync repo: hf_sync squashes its history, which deletes that commit.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""
    commit = api.upload_folder(
        repo_id=repo_id, folder_path=str(folder), path_in_repo=path_in_repo or None,
        ignore_patterns=["config_local.yaml", "config.yaml", "*.tmp"],
        commit_message=f"Upload Hinglish Pocket TTS {prefix}".strip(),
    )  # fmt: skip
    at = f"@{commit.oid}" if pin else ""
    hub_cfg = _config_dict(base_config, f"hf://{repo_id}/{prefix}model.safetensors{at}", f"hf://{repo_id}/{prefix}tokenizer.json{at}")
    header = f"# Hinglish Pocket TTS, weights pinned to commit {commit.oid}.\n" if pin else "# Hinglish Pocket TTS (latest weights in this repo).\n"
    cfg_path = folder / ("config.yaml" if pin else "config.sync.yaml")
    _write_yaml(cfg_path, hub_cfg, header)
    api.upload_file(path_or_fileobj=str(cfg_path), path_in_repo=f"{prefix}config.yaml", repo_id=repo_id,
                    commit_message="Add hub config")  # fmt: skip
    if not pin:
        cfg_path.unlink()
    return f"https://huggingface.co/{repo_id}/tree/main/{prefix}"
