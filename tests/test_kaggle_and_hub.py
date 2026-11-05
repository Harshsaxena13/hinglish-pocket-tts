import json
import tarfile
import time
from pathlib import Path

import pytest

from hinglish_tts.data.hub import pack, unpack
from hinglish_tts.kaggle_env import candidate_builds, driver_cuda_version
from hinglish_tts.pipeline import MODES, PREREQUISITES, STAGE_FUNCS, Settings

SMI = """
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
"""


def test_driver_cuda_version_and_build_choice() -> None:
    assert driver_cuda_version(SMI) == (12, 8)
    assert candidate_builds((12, 8)) == ["cu128", "cu126"]  # CUDA 12 minor-version compatibility
    assert candidate_builds((12, 2)) == ["cu128", "cu126"]
    assert candidate_builds((13, 0)) == ["cu130", "cu128", "cu126"]
    assert candidate_builds((11, 8)) == ["cpu"]
    assert candidate_builds(None) == ["cpu"]
    assert driver_cuda_version("no gpu here") is None


def _corpus(root: Path) -> None:
    man = root / "manifests"
    man.mkdir(parents=True)
    for split, n in (("train", 3), ("valid", 1), ("test", 1)):
        rows = []
        for i in range(n):
            wav = root / "audio" / "src" / "000" / f"{split}_{i}.flac"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"flac" * (i + 1))
            rows.append({"path": str(wav), "duration": 2.0, "transcript": "Aap kaise ho.", "words": []})
        (man / f"{split}_aligned.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (man / "clips_metadata.jsonl").write_text(json.dumps({"path": str(root / "audio/src/000/train_0.flac"), "source": "src"}) + "\n")
    (root / "STATS.md").write_text("# stats\n")


def test_pack_unpack_round_trip_relocates_paths(tmp_path: Path) -> None:
    src, staging, dst = tmp_path / "kaggle_a", tmp_path / "hub", tmp_path / "kaggle_b"
    _corpus(src)
    info = pack(src, staging, shard_bytes=10)  # tiny shards: exercises shard rollover
    assert info["clips"] == 5 and info["shards"] >= 2
    rel = json.loads((staging / "manifests" / "train_aligned.jsonl").read_text().splitlines()[0])["path"]
    assert rel == "audio/src/000/train_0.flac"  # repo manifests are machine-independent
    assert "Aap kaise ho" in (staging / "manifests" / "train_aligned.jsonl").read_text()
    counts = unpack(staging, dst)
    assert counts["train_aligned.jsonl"] == 3
    row = json.loads((dst / "manifests" / "train_aligned.jsonl").read_text().splitlines()[2])
    assert row["path"] == str((dst / "audio/src/000/train_2.flac").resolve()) and Path(row["path"]).read_bytes() == b"flac" * 3
    assert list(row) == ["path", "duration", "transcript", "words"]
    assert (dst / "STATS.md").exists()


def test_unpack_ignores_path_traversal(tmp_path: Path) -> None:
    staging, dst = tmp_path / "hub", tmp_path / "data"
    (staging / "audio").mkdir(parents=True)
    (staging / "manifests").mkdir()
    (staging / "manifests" / "train_aligned.jsonl").write_text("")
    evil = tmp_path / "evil.txt"
    evil.write_text("x")
    with tarfile.open(staging / "audio" / "x-000.tar", "w") as tar:
        tar.add(evil, arcname="../escaped.txt")
    unpack(staging, dst)
    assert not (tmp_path / "escaped.txt").exists()


def test_modes_and_hard_stop() -> None:
    assert set(STAGE_FUNCS) >= {st for stages in MODES.values() for st in stages}
    assert "data" not in MODES["train"] and "pull_data" in MODES["train"] and "push_data" in MODES["setup"]
    assert "smoke" in PREREQUISITES
    s = Settings(work=Path("/w"), out=Path("/o"), session_start=time.time() - 3 * 3600, hard_stop_hours=11.0,
                 teacher_share=0.65, teacher_hours=None, student_hours=None, smoke_steps=200, gpus=2, hours_scale=1.0,
                 hf_repo=None, dataset_repo="me/data", skip_student=False, eval_voices=4, force=False)  # fmt: skip
    assert s.train_hours_left() == pytest.approx(8.0, abs=0.01)
