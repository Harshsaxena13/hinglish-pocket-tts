"""hf_sync against an in-memory fake of the Hub API (no network)."""

import os
from pathlib import Path

import pytest

from hinglish_tts import hf_sync


class FakeHub:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commits = 0
        self.squashes = 0

    def list_repo_files(self, repo_id: str) -> list[str]:
        return sorted(self.files)

    def create_commit(self, repo_id: str, operations: list, commit_message: str) -> None:
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.files.pop(op.path_in_repo)
            else:
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        self.commits += 1

    def super_squash_history(self, repo_id: str) -> None:
        self.squashes += 1

    def create_repo(self, repo_id: str, private: bool, exist_ok: bool, repo_type: str = "model") -> None:
        assert private


def fake_checkpoint(run: Path, step: int) -> None:
    (run / "samples").mkdir(parents=True, exist_ok=True)
    (run / f"checkpoint_{step:08d}.pt").write_bytes(b"ckpt%d" % step)
    (run / f"optim_{step:08d}.pt").write_bytes(b"optim%d" % step)
    (run / "model.safetensors").write_bytes(b"export%d" % step)
    (run / "progress.jsonl").write_text(f'{{"step": {step}}}\n')
    (run / "samples" / f"step{step:08d}_0.wav").write_bytes(b"wav")


def test_push_keeps_only_the_latest_checkpoint(tmp_path: Path) -> None:
    hub, run = FakeHub(), tmp_path / "runs" / "teacher"
    fake_checkpoint(run, 1000)
    hf_sync.push_run("me/repo", run, tmp_path, api=hub, with_optim=True)
    fake_checkpoint(run, 2000)
    (run / "checkpoint_00001000.pt").unlink()  # the trainer keeps one checkpoint
    (run / "optim_00001000.pt").unlink()
    assert hf_sync.push_run("me/repo", run, tmp_path, api=hub, with_optim=True) == 2000
    assert sorted(hub.files) == [
        "runs/teacher/checkpoint_00002000.pt", "runs/teacher/model.safetensors", "runs/teacher/optim_00002000.pt",
        "runs/teacher/progress.jsonl", "runs/teacher/samples/step00001000_0.wav", "runs/teacher/samples/step00002000_0.wav",
    ]  # fmt: skip
    assert hub.files["runs/teacher/model.safetensors"] == b"export2000"
    assert hub.squashes == 2 and not (run / ".hf_staging").exists()


def test_staged_upload_survives_the_trainer_deleting_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub, run = FakeHub(), tmp_path / "runs" / "student"
    fake_checkpoint(run, 500)
    real_commit = hub.create_commit

    def commit_while_trainer_moves_on(**kw):  # the next checkpoint lands mid-upload
        (run / "checkpoint_00000500.pt").unlink()
        # The trainer exports via tmp + rename (checkpointing.export_pocket_safetensors): a new
        # inode, so the hard-linked staged copy keeps the old content.
        (run / "model.tmp").write_bytes(b"newer")
        os.replace(run / "model.tmp", run / "model.safetensors")
        real_commit(**kw)

    monkeypatch.setattr(hub, "create_commit", commit_while_trainer_moves_on)
    hf_sync.push_run("me/repo", run, tmp_path, api=hub, with_optim=False)
    assert hub.files["runs/student/checkpoint_00000500.pt"] == b"ckpt500"
    assert hub.files["runs/student/model.safetensors"] == b"export500"
    assert not any("optim" in f for f in hub.files)


def test_incomplete_checkpoints_are_not_pushed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    fake_checkpoint(run, 100)
    assert hf_sync._complete(run, 100, with_optim=True)
    (run / "model.safetensors.tmp").write_bytes(b"")  # export still being written
    assert not hf_sync._complete(run, 100, with_optim=True)


def test_restore_downloads_only_when_the_hub_is_ahead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    hub = FakeHub()
    src = tmp_path / "old_session"
    fake_checkpoint(src / "runs" / "teacher", 3000)
    hf_sync.push_run("me/repo", src / "runs" / "teacher", src, api=hub, with_optim=True)
    downloaded: list[str] = []

    def fake_download(repo_id: str, filename: str, local_dir: Path) -> str:
        downloaded.append(filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(hub.files[filename])
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    new = tmp_path / "new_session"
    run = new / "runs" / "teacher"
    assert hf_sync.restore("me/repo", run, new, "export", api=hub) == 3000
    assert (run / "model.safetensors").read_bytes() == b"export3000" and not (run / "checkpoint_00003000.pt").exists()
    assert hf_sync.restore("me/repo", run, new, "all", api=hub) == 3000
    assert hf_sync.latest_step(run) == 3000 and (run / "optim_00003000.pt").exists()
    n = len(downloaded)
    assert hf_sync.restore("me/repo", run, new, "all", api=hub) == 3000 and len(downloaded) == n  # up to date


def test_sync_is_off_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_sync.sync_repo() is None
    monkeypatch.setenv("HF_TOKEN", "x")
    assert hf_sync.sync_repo() == "harshsxna/pocket-tts-hinglish"
    monkeypatch.setenv("HF_SYNC_REPO", "")
    assert hf_sync.sync_repo() is None


def test_access_check_names_the_wrong_namespace() -> None:
    hub = FakeHub()
    hub.whoami = lambda: {"name": "alice", "orgs": [{"name": "lab"}], "auth": {"accessToken": {"role": "write"}}}
    assert "cannot write to 'bob'" in hf_sync.check_access("bob/repo", api=hub)
    assert hf_sync.check_access("alice/repo", api=hub) is None
    assert hf_sync.check_access("lab/repo", api=hub) is None
    hub.whoami = lambda: {"name": "alice", "orgs": [], "auth": {"accessToken": {"role": "read"}}}
    assert "read-only" in hf_sync.check_access("alice/repo", api=hub)
