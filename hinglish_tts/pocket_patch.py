"""Patch Kyutai's pocket-tts training code for Turing GPUs (Kaggle T4s).

The upstream trainer hard-codes bf16 autocast. T4s have no bf16 tensor cores, so
bf16 is either emulated (very slow) or unsupported. This patch:

1. picks the autocast dtype per GPU (bf16 on Ampere+, fp16 on older GPUs;
   POCKET_TTS_AMP_DTYPE overrides) and adds a GradScaler for fp16, which skips
   overflowing updates instead of aborting the run;
2. optionally runs the small LSD head loss in fp32 (POCKET_TTS_FP32_FLOW_LOSS=1),
   the fallback if fp16 overflows persist;
3. keeps the *teacher's* latent normalization stats when a distilled student is
   seeded from another checkpoint (`student_init_from`): the frozen sampler head
   comes from the teacher and reads the teacher's normalized space;
4. makes the latent-precompute batch size and decode workers configurable
   (POCKET_TTS_PRECOMPUTE_BATCH, POCKET_TTS_DECODE_WORKERS) and portable.

Every edit is an exact, unique string replacement against the pinned upstream
commit; a mismatch fails loudly instead of silently training unpatched code.
Re-running is a no-op.

    python -m hinglish_tts.pocket_patch /path/to/pocket-tts
"""

from __future__ import annotations

import sys
from pathlib import Path

POCKET_TTS_REPO = "https://github.com/kyutai-labs/pocket-tts.git"
POCKET_TTS_COMMIT = "2e117a1e7df06323c9df0315c671f3cbc3b8d9fc"
MARK = "[hinglish-t4]"

AMP_HELPER = '''

def _amp_dtype(device: torch.device) -> torch.dtype:
    """[hinglish-t4] Autocast dtype: bf16 on Ampere+, fp16 (with loss scaling) on older
    GPUs such as the T4, which have no bf16 tensor cores. POCKET_TTS_AMP_DTYPE overrides."""
    forced = os.environ.get("POCKET_TTS_AMP_DTYPE")
    if forced:
        return getattr(torch, forced)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return torch.float16
    return torch.bfloat16
'''

EDITS: list[tuple[str, str, str]] = [
    (
        "training/train.py",
        'VERBOSE_STEPS = 10  # log every step at the start of a run, then every log_freq\n',
        'VERBOSE_STEPS = 10  # log every step at the start of a run, then every log_freq\n' + AMP_HELPER,
    ),
    (
        "training/train.py",
        "    model, mimi, _config = build_models(args)\n",
        "    model, mimi, _config = build_models(args)\n"
        '    if os.environ.get("POCKET_TTS_FP32_FLOW_LOSS") == "1":  # [hinglish-t4]\n'
        "        model.flow.fp32_loss = True\n",
    ),
    (
        "training/train.py",
        '''    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    refl_rewards = ''',
        '''    amp_dtype = _amp_dtype(device)  # [hinglish-t4]
    autocast = torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda" and amp_dtype != torch.float32,
    )
    # fp16 needs loss scaling; with bf16/fp32 the scaler is a pass-through.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    nonfinite_steps = 0
    if rank == 0:
        logger.info(f"mixed precision {amp_dtype}, grad scaler {'on' if scaler.is_enabled() else 'off'}")
    refl_rewards = ''',
    ),
    (
        "training/train.py",
        """                with run.wrapped.no_sync():
                    scaled.backward()
            else:
                scaled.backward()
""",
        """                with run.wrapped.no_sync():
                    scaler.scale(scaled).backward()  # [hinglish-t4]
            else:
                scaler.scale(scaled).backward()
""",
    ),
    (
        "training/train.py",
        """        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)
        if not torch.isfinite(grad_norm):
            # Stop before the NaN reaches the weights: the run would otherwise
            # keep training and checkpointing garbage, and auto-resume reloads it.
            raise SystemExit(f"non-finite gradient at step {step}")
        optimizer.step()
        if ema is not None:
            ema.update(model)
""",
        """        scaler.unscale_(optimizer)  # [hinglish-t4]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)
        if not torch.isfinite(grad_norm):
            if not scaler.is_enabled():
                # Stop before the NaN reaches the weights: the run would otherwise
                # keep training and checkpointing garbage, and auto-resume reloads it.
                raise SystemExit(f"non-finite gradient at step {step}")
            # fp16 overflow: the scaler skips this update and lowers the loss scale.
            nonfinite_steps += 1
            if nonfinite_steps >= 50:
                raise SystemExit(
                    f"{nonfinite_steps} consecutive fp16 overflows at step {step}: rerun with "
                    "POCKET_TTS_FP32_FLOW_LOSS=1 or POCKET_TTS_AMP_DTYPE=float32"
                )
            if rank == 0:
                logger.warning(f"step {step + 1}: fp16 overflow, update skipped (scale {scaler.get_scale():.0f})")
        else:
            nonfinite_steps = 0
        scaler.step(optimizer)
        scaler.update()
        if ema is not None and nonfinite_steps == 0:
            ema.update(model)
""",
    ),
    (
        "training/train.py",
        '''    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    totals: dict[str, float] = {}''',
        '''    amp_dtype = _amp_dtype(device)  # [hinglish-t4]
    autocast = torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda" and amp_dtype != torch.float32,
    )
    totals: dict[str, float] = {}''',
    ),
    (
        "training/modules/builders.py",
        """            src = {k: v for k, v in src.items() if not k.startswith(("flow_net.", "out_eos."))}
""",
        """            src = {k: v for k, v in src.items() if not k.startswith(("flow_net.", "out_eos."))}
            # [hinglish-t4] The frozen heads come from the teacher and read its normalized
            # latent space, so the latent statistics must be the teacher's too.
            src = {k: v for k, v in src.items() if k not in ("emb_mean", "emb_std")}
""",
    ),
    (
        "training/train_utils.py",
        "import json\nimport logging\nimport math\n",
        "import json\nimport logging\nimport math\nimport os  # [hinglish-t4]\n",
    ),
    (
        "training/train_utils.py",
        """            audio_manifest,
            mimi,
            device,
            32,
""",
        """            audio_manifest,
            mimi,
            device,
            int(os.environ.get("POCKET_TTS_PRECOMPUTE_BATCH", "32")),  # [hinglish-t4]
""",
    ),
    (
        "training/scripts/precompute_latents.py",
        "    return max(4, len(os.sched_getaffinity(0)) - 2)\n",
        """    if os.environ.get("POCKET_TTS_DECODE_WORKERS"):  # [hinglish-t4]
        return int(os.environ["POCKET_TTS_DECODE_WORKERS"])
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4
    return max(4, cpus - 2)
""",
    ),
]


def apply(repo: str | Path) -> list[str]:
    """Apply every edit to the checkout at `repo`. Returns a log of what happened."""
    repo = Path(repo)
    log = []
    for rel, old, new in EDITS:
        path = repo / rel
        text = path.read_text()
        if new in text:
            log.append(f"already patched: {rel}")
            continue
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"{rel}: expected exactly one match for an edit, found {count}. "
                f"Is the checkout at commit {POCKET_TTS_COMMIT}?\n--- anchor ---\n{old}"
            )
        path.write_text(text.replace(old, new))
        log.append(f"patched: {rel}")
    return log


if __name__ == "__main__":
    for line in apply(sys.argv[1] if len(sys.argv) > 1 else "."):
        print(line)
