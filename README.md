# Hinglish Pocket TTS

Fine-tune [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) to speak **Indian Hinglish**:
romanized text in, Indian-accented speech out, switching smoothly between Hindi and English inside a
sentence. The result keeps Pocket TTS's architecture, English tokenizer and voice cloning, and loads in the
stock `pocket-tts` runtime on CPU.

```text
"Mera laptop hang ho gaya, isliye main meeting thodi der se join karunga."
"Aapka order ₹1,499 ka hai aur वो कल deliver हो जाएगा."   (the frontend handles Devanagari, digits, currency)
```

Everything (data, alignment, training, distillation, evaluation, export) runs from
[`kaggle/notebook.ipynb`](kaggle/notebook.ipynb) on **2x T4** within a 12-hour Kaggle session.

## Results (first run, 2026-11-05)

Model: **`hf://harshsxna/pocket-tts-hinglish/exports/hinglish-pocket-tts-24l/config.yaml`** (private; model
card in the repo). It's the 24-layer teacher after 8,415 steps (4.26 h on 2x T4, batch 64, fp16). The
6-layer student was not trained in this run.

- **Data:** 74.8 h after QA, 37,266 train clips: MUCS 26.2 h, IndicVoices-R 20.6 h, IndicTTS Hindi 17.2 h,
  IndicTTS English 8.3 h. Svarah contributed nothing, because its gated download failed.
- **Eval:** 104 generations per system; CER is MMS-based and lower is better. See *Evaluation* below.

  | CER | overall | code-mixed | English | Hindi | numbers | questions |
  |---|---|---|---|---|---|---|
  | English baseline (released 6-layer) | 0.252 | **0.228** | **0.079** | 0.248 | 0.530 | 0.291 |
  | Hinglish 24-layer | **0.243** | 0.267 | 0.190 | **0.130** | **0.499** | **0.202** |

  Hindi CER nearly halves, but English and code-mixed sentences regressed, and the overall CTC
  loss is worse (1.13 -> 1.66). Speaker similarity improves (0.931 -> 0.945). Likely causes: a short run,
  and only 11% English-only audio from 2 speakers. Next run: add Svarah, train longer, then distill.
- **CPU speed:** real-time factor 0.75 on an Apple M1 (4 threads), the same as the base 24-layer English
  model (0.72). See [`demo/`](demo/).

## How it works

```
 HF corpora ──► romanize + filter ──► 24 kHz, trim, loudness ──► MMS_FA alignment ──► QA ──► manifests + STATS.md
  (~100 h)      (Hinglish frontend)    (1-15 s clips)            (torchaudio)                      │
                                                                     private HF dataset ◄──────────┘ (reused by RUN_MODE="train")
   english_2026-09_24l ──► fine-tune 24-layer TEACHER (fp16, 2x T4) ──┐
                                                                       ├──► every checkpoint ──► private HF model repo
   english_2026-09 (6L) ──► distill STUDENT, CFG baked in ─────────────┘     (latest only; a new session resumes)
                                      │
               eval (baseline vs ours) ──► export (pocket-tts folder, voices, model card) ──► infer.py on any CPU
```

**Two-stage recipe, as Kyutai trains Pocket TTS.** A 24-layer teacher learns Hinglish. The 6-layer
student then learns to reproduce the teacher's classifier-free-guided output, so it gets guided quality
from a single forward pass, which is what makes it fast on CPU. `english_2026-09` was distilled from
`english_2026-09_24l`, so the pair is matched: the student starts from the released English student and
adapts in a few thousand steps.

| decision | why |
|---|---|
| Keep the English tokenizer | Romanized Hindi is Latin script, so every token already has an embedding and English words keep their pronunciation. A new tokenizer would reset the text embedding and need far more data. |
| Freeze the teacher's sampler head | Keeps the latent geometry the released student was distilled against, so the student can start from it. |
| Time-boxed, constant LR + EMA | Training stops at a wall-clock hard stop (STOP file -> checkpoint -> export), so a cosine schedule would never finish. The EMA weights act as the decay. |
| fp16 + GradScaler patch | Upstream hard-codes bf16, which T4s can't run efficiently. `hinglish_tts/pocket_patch.py` adds per-GPU autocast dtype and loss scaling (overflowing steps are skipped, not fatal), with exact-match edits against a pinned commit. |
| torchaudio MMS_FA on romanized text | One multilingual CTC aligner handles Hindi and English words alike. Its per-character loss flags clips whose transcript doesn't match the audio. |
| Romanize, don't transliterate | Training text follows how people type Hinglish (`accha`, `nahi`, `zindagi`, `computer`); 20% of Hindi words get a common alternative spelling, so `bahut` and `bohot` both work. |

## Run on Kaggle

1. Push this repository to GitHub (see *What's in the repo*). Set `REPO_URL` in the notebook's first cell
   if it isn't `github.com/Harshsaxena13/hinglish-pocket-tts`.
2. Kaggle: *File -> Import Notebook* -> `kaggle/notebook.ipynb`. *Settings*: Accelerator **GPU T4 x2**,
   Internet **on**.
3. *Add-ons -> Secrets*:
   - `HF_TOKEN`: a Hugging Face token **with write access** from an account that accepted the terms of
     [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) (gated, auto-approved). Accept
     [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) too for more Indian-English voices.
   - `GITHUB_TOKEN`: only if the GitHub repo is private.
4. First session: `RUN_MODE = "setup"`. Later sessions: `RUN_MODE = "train"`. Then *Save Version -> Save & Run
   All (Commit)*.

| cell | `RUN_MODE="setup"` | `RUN_MODE="train"` |
|---|---|---|
| GPU check, HF login (Kaggle secret), clone repo | yes | yes |
| `uv sync` with the torch build matching the CUDA driver, verified with `torch.cuda.is_available()` and an fp16 matmul on both GPUs inside the uv env | yes | yes |
| pocket-tts checkout at a pinned commit + T4 patch, gated-model access check | yes | yes |
| download + prepare (~1 h): clips, MMS_FA alignment, QA, `STATS.md` | yes | - |
| upload the corpus to the private HF dataset `HF_DATASET_REPO` | yes | pull it instead (minutes) |
| smoke test: 200 steps; must checkpoint with a finite loss; reports speed, memory, fp16 overflows | yes | yes |
| full training: teacher (~65% of the time left), then student; **hard stop at 11 h** (`HARD_STOP_HOURS`) | yes | yes |
| eval + export (in the final hour) | yes | yes |

Repos (private, set in the first cell): `HF_SYNC_REPO = harshsxna/pocket-tts-hinglish` (checkpoints, progress,
samples, final 6-layer export) and `HF_DATASET_REPO = harshsxna/hinglish-tts-data` (the prepared corpus).
The token's account must be able to write to both. The notebook warns otherwise, and the pipeline then
disables syncing rather than failing the run. Other settings in the same cell: `HARD_STOP_HOURS`, `SMOKE_STEPS`,
`DATA_HOURS_SCALE` (0.1 for a dry run), `SKIP_STUDENT`, and `TEACHER_HOURS` / `STUDENT_HOURS` to fix a
stage's budget instead of the automatic split (the hard stop still applies).

**Checkpoint sync and resume.** After every checkpoint (~every 1000 steps), a watcher thread uploads the
checkpoint, optimizer state, EMA export, `progress.jsonl` and new audio samples in one commit. It deletes
the previous checkpoint and squashes the history, so only the latest is stored (~6.5 GB for the teacher).
Uploads use hard-linked copies and never block or stop training. A new session with `RUN_MODE="train"`
restores `pipeline_state.json`, skips finished stages, restores the unfinished run, and the trainer
resumes. To do it by hand: `python scripts/hf_sync.py push|restore|watch --run-dir ... --root /kaggle/working`.

**If something fails:** the smoke test stops the notebook before the long run. CUDA OOM retries with half
the micro-batch and twice the accumulation (same effective batch), and the full run inherits whatever
batch the smoke test ended up with. Persistent fp16 overflows: set `POCKET_TTS_FP32_FLOW_LOSS=1` (or
`POCKET_TTS_AMP_DTYPE=float32`, much slower) in the first cell.

Outputs in `/kaggle/working`: `exports/hinglish-pocket-tts/` (6-layer CPU model), `exports/hinglish-pocket-tts-24l/`,
`eval/results.md` + `eval/index.html` (side-by-side listening page), `data/STATS.md`, `runs/*/` (logs,
`progress.jsonl`, `samples/`), `pipeline_state.json`.

## Use the model (Mac / Linux / Windows CPU)

```bash
uv sync --extra cpu
hf auth login                    # the export lives in your private HF repo
uv run infer.py --text "Kal ki meeting 10:30 बजे है, please time pe aana." --voice my_voice.wav --out out.wav
```

By default `infer.py` loads the 24-layer export from the private sync repo
(`hf://harshsxna/pocket-tts-hinglish/exports/hinglish-pocket-tts-24l/config.yaml`). Set
`HINGLISH_TTS_MODEL` or pass `--model` to use another. Other options:
- `--model <export folder or config.yaml>`;
- `--checkpoint <training .pt, local or hf://>` (6 vs 24 layers is detected automatically);
- `--numbers hi` to read digits in Hindi;
- `--text-file` for one utterance per line;
- `--threads` to set CPU threads.

Without `--voice` it uses the first voice shipped with the export. If the export has none, it falls back to
the Indian-English VCTK speaker `hf://kyutai/tts-voices/vctk/p248_023_enhanced.wav`. Use 5-10 s of clean,
full-band speech: the model copies the prompt's acoustics. It prints the normalized text and the real-time factor.

The export also works without this repo, but you must pass romanized text with numbers spelled out,
since this path skips the frontend:
`uvx pocket-tts generate --config hf://harshsxna/pocket-tts-hinglish/exports/hinglish-pocket-tts-24l/config.yaml --voice v.wav --text "..."`.

**Rebuilding the export from the Hub alone** (e.g. when Kaggle can't save its output):
`python scripts/export_from_hub.py --push` lists `runs/teacher`. It downloads only `model.safetensors`
and small text files, never optimizer state or data, and writes `exports/hinglish-pocket-tts-24l/`. Its
config points at those weights plus Kyutai's unchanged English tokenizer. It then uploads that folder
to the same repo.

## Data

| source | role | default budget | license |
|---|---|---|---|
| [MUCS 2021 Hinglish](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) | real code-switching (Hindi in Devanagari, English in Latin) | 40 h | CC-BY-4.0 |
| [IndicVoices-R Hindi](https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi) | many speakers, enhanced audio, SNR/C50 filtered | 35 h | CC-BY-4.0 |
| [IndicTTS Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) | studio Hindi (2 voices) | 18 h | see dataset |
| [IndicTTS English](https://huggingface.co/datasets/SPRINGLab/IndicTTS-English) | studio Indian-accented English | 9 h | see dataset |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (optional, gated) | Indian English, 117 speakers | 9 h | CC-BY-4.0 |

`python -m hinglish_tts.data.prepare --out <dir>` (the notebook's download and prepare cells) processes each
shard in `/kaggle/temp` and deletes it. Per clip:

1. transcript -> romanized Hinglish, English loanwords back to English (`मीटिंग` -> `meeting`), digits
   spelled out (Hindi words for Devanagari-majority text, English otherwise); other scripts, markup and
   texts under 3 words are dropped;
2. audio -> mono 24 kHz (Mimi's rate), silence trimmed, active speech level -20 dBFS, peak <= 0.95;
   kept if 1-15 s with a plausible speaking rate;
3. split: 2% test per source (whole speakers held out when a source has >= 20), ~1% valid, rest train;
4. word timestamps from **torchaudio MMS_FA**; unaligned clips are dropped, and so are train clips with
   outlier CTC loss. Train clips without a word boundary 1 s from both ends can't be cut into
   prompt + target by the trainer, so they are kept out of training only (`no_prompt_cut` in STATS.md).

Outputs: `manifests/{train,valid,test}_aligned.jsonl` in exactly the pocket-tts README format (`path`,
`duration`, `transcript`, `words`), `manifests/clips_metadata.jsonl` (source, speaker, raw text, alignment
score, drop reason), and **`STATS.md`** (hours per source and split, clips kept and dropped by reason,
transcript examples). Edit `configs/data_recipe.yaml` to change sources, budgets or filters.

## Text frontend (`hinglish_tts/text`)

Used for training transcripts and inference input alike:

- **Devanagari -> casual Roman** with Hindi schwa deletion (`समझना` -> `samajhna`) and chat-style vowels.
- **Lexicons:** ~300 frequent Hindi words (`main`, `mein`, `hoon`, `nahi`); ~180 English loanwords back to
  English; acronyms (`पीएम` -> `PM`); optionally ~1M attested spellings from
  [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar).
- **Numbers** in Indian grouping, English or Hindi words (`₹1,50,000` -> `one lakh fifty thousand rupees`
  / `ek lakh pachaas hazaar rupaye`), plus years, times, percentages, ordinals, phone numbers.

## Evaluation

`hinglish_tts/evaluate.py` runs the English baseline, the teacher and the student on curated sentences
(Hindi, code-mixed, Indian English, questions, numbers), with voice prompts from test-split clips. Speakers
are held out only for sources with at least 20 speakers; the IndicTTS voices also appear in training.

- **CTC NLL / CER:** an off-the-shelf ASR writes Hindi in Devanagari and can't score romanized input. Instead,
  MMS scores how well the audio matches the *input text* (per-character loss) and transcribes it greedily.
  CER is computed on a spelling-insensitive skeleton.
- **Speaker similarity** (WavLM-SV vs. the prompt), **UTMOS** if `utmos-pytorch` is installed, failure rate.
- `eval/index.html` puts baseline, teacher and student side by side for listening.

## Local development

```bash
uv sync --extra cpu                    # CPU torch on a laptop
uv run pytest -q                       # ~1 s
git clone https://github.com/kyutai-labs/pocket-tts /tmp/pocket-tts
uv run scripts/local_smoke_test.py --pocket-tts /tmp/pocket-tts --work /tmp/smoke
```

The smoke test fetches at most 50 clips through the datasets-server API (no shard downloads) and runs
every stage on CPU in ~10 min. It covers prepare, align, `STATS.md`, a dataset pack/unpack round trip,
training, distillation, eval, export and `infer.py`. It downloads ~1.5 GB of model weights. Without an
account that accepted kyutai/pocket-tts' terms, it uses Kyutai's ungated "without voice cloning" weights:
every code path runs, but the audio is meaningless. After editing the notebook cells, regenerate with
`python scripts/build_notebook.py`.

## What's in the repo

```
kaggle/notebook.ipynb   the Kaggle notebook, the only one (generated by scripts/build_notebook.py)
infer.py                text + voice prompt -> wav (CPU or GPU)
configs/                data_recipe.yaml, hinglish_ft_t4.yaml (24-layer fine-tune), hinglish_distill_t4.yaml (6-layer)
hinglish_tts/
  text/                 Hinglish frontend: devanagari.py, lexicon.py, numbers.py, variants.py, normalize.py
  data/prepare.py       corpus entry point: clips -> 2% test split -> alignment -> QA -> manifests + STATS.md
  data/build.py         shard-parallel clip extraction (download, filter, romanize, resample, trim, loudness)
  data/hub.py           upload / download the prepared corpus (private HF dataset, tar shards)
  align.py              torchaudio MMS_FA forced alignment (batched, fp16) + per-clip CTC loss
  pocket_patch.py       T4/fp16 patch for the pinned pocket-tts trainer
  train_runner.py       time-boxed torchrun launcher (STOP-file checkpointing)
  hf_sync.py            mirror checkpoints to a private HF repo (latest only), restore in a new session
  kaggle_env.py         uv sync with the torch build matching the CUDA driver, verified on the GPU
  pipeline.py           stages, run modes, smoke test, hard stop
  evaluate.py, export.py, synthesize.py
scripts/                build_notebook.py, hf_sync.py (CLI), export_from_hub.py, local_smoke_test.py
demo/                   demo sentences + CPU timings (the .wav files are generated locally, not committed)
tests/                  unit tests (no GPU, no network)
eval/                   evaluation sentences
pyproject.toml, uv.lock uv project; extras cpu / cu126 / cu128 / cu130 select the PyTorch build
```

No tokens are stored in the repo: the notebook reads `HF_TOKEN` (and `GITHUB_TOKEN`) from Kaggle secrets,
and local runs use `hf auth login`.

## Status and known risks

- First Kaggle run (2x T4, torch 2.14.1+cu130), all of it worked on real hardware:
  - data prep: 0.4 h;
  - dataset upload to the private HF dataset;
  - 200-step smoke test: 0.53 it/s, peak 10.5 GiB per GPU, no fp16 overflows;
  - teacher training: 4.26 h, 8,415 steps, with checkpoints synced to the Hub;
  - eval and export.

  The student stage didn't run. Kaggle then hung while saving the notebook output, so the 24-layer export
  was rebuilt from the Hub with `scripts/export_from_hub.py` and checked with stock `uvx pocket-tts generate`.
- English and code-mixed intelligibility regressed (see *Results*); Svarah wasn't used (gated download failed).
- MUCS audio is 16 kHz; prompting with a 16 kHz voice gives band-limited output.
- Heavy SMS-style spellings (`kch`, `h` for `hai`) are read less reliably.
- MUCS audio is 16 kHz; prompting with a 16 kHz voice gives band-limited output.
- Heavy SMS-style spellings (`kch`, `h` for `hai`) are read less reliably.

## Licenses and responsible use

Builds on Kyutai's pocket-tts training code (MIT); the base model is CC-BY-4.0. Each dataset has its own
terms (table above); the prepared-data repo is private for that reason. Check them before releasing or
commercially using a trained model. Voice cloning imitates real people: only clone voices with consent.

## References

- Kyutai, *Pocket TTS* ([code](https://github.com/kyutai-labs/pocket-tts), [tech report](https://kyutai.org/blog/2026-01-13-pocket-tts)); *Continuous Audio Language Models* ([arXiv:2509.06926](https://arxiv.org/abs/2509.06926))
- Diwan et al., *MUCS 2021: Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages*
- AI4Bharat, *IndicVoices-R*, *Svarah*, *Aksharantar*; IIT Madras, *Indic TTS*
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (MMS)
