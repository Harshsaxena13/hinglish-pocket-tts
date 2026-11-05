from pathlib import Path

import numpy as np
import yaml

from hinglish_tts.align import word_tokens
from hinglish_tts.data.audio import trim_silence
from hinglish_tts.evaluate import edit_distance, skeleton
from hinglish_tts.train_runner import render_config

ROOT = Path(__file__).resolve().parent.parent
MMS_VOCAB = {"<blank>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{c: 4 + i for i, c in enumerate("aientousrmkldghybpwcvjzf'qx")}}


def test_word_tokens_have_no_delimiter_and_skip_punctuation() -> None:
    tokens, word_of, norm = word_tokens(["Kya", "-", "hai?"], MMS_VOCAB)
    assert norm == ["kya", "", "hai"]
    assert word_of == [0, 0, 0, 1, 1, 1]
    assert len(tokens) == 6 and MMS_VOCAB["<blank>"] not in tokens


def test_skeleton_ignores_common_spelling_variation() -> None:
    assert skeleton("Bahut accha") == skeleton("bahut acha")
    assert skeleton("wahan") == skeleton("vahaan")
    assert edit_distance("kitten", "sitting") == 3


def test_trim_silence_keeps_margins() -> None:
    sr = 24000
    wav = np.zeros(sr * 3, dtype=np.float32)
    wav[sr : 2 * sr] = 0.3 * np.sin(np.arange(sr) / 5.0)
    out = trim_silence(wav, sr)
    assert 1.4 * sr < len(out) < 1.8 * sr


def test_training_configs_render(tmp_path: Path) -> None:
    for name in ("hinglish_ft_t4.yaml", "hinglish_distill_t4.yaml"):
        out = render_config(ROOT / "configs" / name, tmp_path / name, POCKET_TTS="/pt", DATA="/data", RUNS="/runs")
        cfg = yaml.safe_load(out.read_text())
        assert cfg["data"]["max_duration_sec"] == yaml.safe_load((ROOT / "configs" / "data_recipe.yaml").read_text())["max_sec"]
        world, accum, batch = 2, cfg["grad_accum_steps"], cfg["batch_size"]
        assert world * accum * batch == 64  # Kyutai's effective batch


TORCHAUDIO_MMS_DICT = {c: i for i, c in enumerate("-aientousrmkldghybpwcvjzf'qx")}  # MMS_FA.get_dict(star=None)


def test_hyphen_never_maps_to_the_torchaudio_blank() -> None:
    tokens, _, norm = word_tokens(["well-known", "-"], TORCHAUDIO_MMS_DICT)
    assert norm == ["wellknown", ""]
    assert TORCHAUDIO_MMS_DICT["-"] not in tokens


def test_loudness_normalization_hits_target_and_respects_ceiling() -> None:
    from hinglish_tts.data.audio import active_level_db, normalize_loudness

    sr = 24000
    t = np.arange(sr * 2) / sr
    wav = (0.01 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav[: sr // 2] = 0  # a pause must not drag the level down
    out = normalize_loudness(wav, sr, target_db=-20.0)
    assert abs(active_level_db(out, sr) + 20.0) < 0.5
    assert np.abs(normalize_loudness(wav * 50, sr, target_db=0.0)).max() <= 0.95 + 1e-6


def test_split_is_two_percent_and_speaker_disjoint_for_multi_speaker_sources() -> None:
    from hinglish_tts.data.prepare import split_clips

    rows = [{"path": f"a{i}", "source": "many", "speaker": f"s{i % 50}", "duration": 5.0} for i in range(1000)]
    rows += [{"path": f"b{i}", "source": "two", "speaker": f"t{i % 2}", "duration": 5.0} for i in range(500)]
    split = split_clips(rows, {"test_frac": 0.02, "valid_frac": 0.01, "valid_min": 10, "valid_max": 300, "seed": 1})
    test_many = [r for r in rows if r["source"] == "many" and split[r["path"]] == "test"]
    test_two = [r for r in rows if r["source"] == "two" and split[r["path"]] == "test"]
    assert 20 <= len(test_many) <= 30 and len(test_two) == 10
    held_out = {r["speaker"] for r in test_many}
    assert not any(r["speaker"] in held_out for r in rows if r["source"] == "many" and split[r["path"]] != "test")
    assert sum(v == "valid" for v in split.values()) == 15


def test_prompt_cut_rule_matches_the_loader() -> None:
    from hinglish_tts.data.prepare import _has_prompt_cut

    words = [{"word": "a", "start": 0.1, "end": 0.8}, {"word": "b", "start": 1.2, "end": 1.6}]
    assert _has_prompt_cut(words, 2.5)
    assert not _has_prompt_cut(words, 1.8)  # cut at 1.0 s leaves only 0.8 s of target
