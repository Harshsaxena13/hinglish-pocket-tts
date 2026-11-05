"""Audio helpers for data preparation: decode, resample, trim, write."""

from __future__ import annotations

import io
import os
import tempfile
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import soundfile

Wave: TypeAlias = npt.NDArray[np.float32]


def decode(data: bytes) -> tuple[Wave, int]:
    """Decode an encoded audio blob (wav/flac/ogg/mp3) to mono float32."""
    try:
        wav, sr = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return wav.mean(axis=1), int(sr)
    except Exception:  # noqa: BLE001 -- fall back to sphn (symphonia) for other codecs
        import sphn

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(data)
        try:
            wav, sr = sphn.read(tmp.name)
            return np.asarray(wav, dtype=np.float32).mean(axis=0), int(sr)
        finally:
            os.unlink(tmp.name)


def resample(wav: Wave, src: int, dst: int) -> Wave:
    if src == dst:
        return wav
    try:
        import soxr

        return soxr.resample(wav, src, dst, quality="HQ").astype(np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src, dst)
        return resample_poly(wav, dst // g, src // g).astype(np.float32)


def trim_silence(
    wav: Wave, sr: int, lead_sec: float = 0.25, tail_sec: float = 0.35, floor_db: float = -40.0
) -> Wave:
    """Cut leading/trailing silence, keeping short margins.

    Frames (20 ms) whose RMS is within `floor_db` of the loudest frame count as
    speech. The margins keep breaths and soft onsets; the training loader trims
    the tail precisely from the word alignment anyway.
    """
    hop = int(0.02 * sr)
    if len(wav) < 4 * hop:
        return wav
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    voiced = np.flatnonzero(20 * np.log10(rms / rms.max()) > floor_db)
    if voiced.size == 0:
        return wav
    start = max(0, voiced[0] * hop - int(lead_sec * sr))
    end = min(len(wav), (voiced[-1] + 1) * hop + int(tail_sec * sr))
    return wav[start:end]


def active_level_db(wav: Wave, sr: int, floor_db: float = -40.0) -> float:
    """RMS level (dBFS) of the speech frames only, so pauses do not skew it."""
    hop = min(int(0.02 * sr), len(wav)) or 1
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    active = rms[20 * np.log10(rms / rms.max()) > floor_db]
    return float(20 * np.log10(np.sqrt(np.mean(active**2)) + 1e-12))


def normalize_loudness(wav: Wave, sr: int, target_db: float = -20.0, ceiling: float = 0.95) -> Wave:
    """Scale to `target_db` active speech level, then limit the peak to `ceiling`.

    Corpora differ by 20+ dB; the model copies the prompt's level, so a level
    mismatch between sources would leak into generations.
    """
    gain = 10 ** ((target_db - active_level_db(wav, sr)) / 20)
    wav = wav * gain
    return fix_peak(wav, ceiling).astype(np.float32)


def loudness_ok(wav: Wave, min_dbfs: float = -45.0) -> bool:
    rms = float(np.sqrt(np.mean(wav**2) + 1e-12))
    return 20 * np.log10(rms) > min_dbfs


def fix_peak(wav: Wave, ceiling: float = 0.95) -> Wave:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return wav * (ceiling / peak) if peak > ceiling else wav


def write_flac(path: str | os.PathLike[str], wav: Wave, sr: int) -> None:
    tmp = f"{path}.tmp.flac"
    soundfile.write(tmp, wav, sr, subtype="PCM_16", format="FLAC")
    os.replace(tmp, path)
