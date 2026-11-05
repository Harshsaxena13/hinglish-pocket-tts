"""Hinglish text -> speech with an exported model or a training checkpoint (CPU by default).

Text goes through the Hinglish frontend (Devanagari -> Roman, numbers -> words,
cleanup), then the stock pocket-tts generator. See infer.py for the CLI.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .text import HinglishFrontend

BASE_LANGUAGES = {6: "english_2026-09", 24: "english_2026-09_24l"}
# Fallback voice prompt when none is given and the export ships none: VCTK p248, an Indian-English speaker.
FALLBACK_VOICE = "hf://kyutai/tts-voices/vctk/p248_023_enhanced.wav"


def _hf_split(uri: str) -> tuple[str, str, str | None]:
    """hf://owner/repo/path/to/file@rev -> (repo_id, path, revision)."""
    rest = uri.removeprefix("hf://")
    rev = None
    if "@" in rest:
        rest, rev = rest.rsplit("@", 1)
    parts = rest.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:]), rev


def fetch(path: str) -> str:
    """Local path for a local file or an hf:// URI."""
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    repo, file, rev = _hf_split(path)
    return hf_hub_download(repo, file, revision=rev)


def sibling(config: str, name: str) -> str | None:
    """A file next to the config (local or on the Hub), or None if there is none."""
    if config.startswith("hf://"):
        repo, file, rev = _hf_split(config)
        folder = file.rsplit("/", 1)[0] + "/" if "/" in file else ""
        try:
            return fetch(f"hf://{repo}/{folder}{name}" + (f"@{rev}" if rev else ""))
        except Exception:  # noqa: BLE001
            return None
    p = Path(config).parent / name
    return str(p) if p.exists() else None


def default_voice(config: str) -> str | None:
    """The first voice shipped with an export (voices/*.wav)."""
    if config.startswith("hf://"):
        from huggingface_hub import HfApi

        repo, file, rev = _hf_split(config)
        folder = file.rsplit("/", 1)[0] + "/" if "/" in file else ""
        try:
            wavs = sorted(f for f in HfApi().list_repo_files(repo, revision=rev) if f.startswith(f"{folder}voices/") and f.endswith(".wav"))
        except Exception:  # noqa: BLE001
            return None
        return fetch(f"hf://{repo}/{wavs[0]}") if wavs else None
    wavs = sorted((Path(config).parent / "voices").glob("*.wav"))
    return str(wavs[0]) if wavs else None


def checkpoint_layers(checkpoint: str) -> int:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    layers = {int(m.group(1)) for k in payload["model"] if (m := re.match(r"flow_lm\.transformer\.layers\.(\d+)\.", k))}
    return max(layers) + 1


class HinglishTTS:
    def __init__(self, config: str | None, voice: str | None, numbers: str = "en", temp: float | None = None,
                 device: str = "cpu", checkpoint: str | None = None):  # fmt: skip
        from pocket_tts import TTSModel

        if checkpoint is not None:
            checkpoint = fetch(checkpoint)
            if config is None:
                n = checkpoint_layers(checkpoint)
                self.model = TTSModel.load_model(language=BASE_LANGUAGES[n], temp=temp, checkpoint=checkpoint)
            else:
                self.model = TTSModel.load_model(config=config, temp=temp, checkpoint=checkpoint)
        else:
            assert config is not None, "need a config or a checkpoint"
            self.model = TTSModel.load_model(config=config, temp=temp)
        if device != "cpu":
            self.model.to(device)
        voice = voice or (default_voice(config) if config else None) or FALLBACK_VOICE
        self.voice_path = voice
        self.voice = self.model.get_state_for_audio_prompt(fetch(voice))
        # Romanize Devanagari input with the same lexicon the training transcripts used.
        lexicon = sibling(config, "lexicon_hi.tsv") if config else None
        self.frontend = HinglishFrontend(lexicon, numbers=numbers)

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def normalize(self, text: str) -> str:
        return self.frontend(text)

    def __call__(self, text: str) -> np.ndarray:
        audio = self.model.generate_audio(self.voice, self.normalize(text))
        return audio.detach().cpu().numpy().reshape(-1)  # mono

