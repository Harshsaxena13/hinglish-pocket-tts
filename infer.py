"""Hinglish text + voice prompt -> wav. Runs on a Mac/Linux/Windows CPU (or a GPU).

    uv sync --extra cpu
    uv run infer.py --text "Kal ki meeting 10:30 baje hai, please time pe aana." --voice my_voice.wav

Model (--model), any of:
  * an export folder or its config:  exports/hinglish-pocket-tts  (contains config_local.yaml)
  * a config on the Hub:             hf://harshsxna/pocket-tts-hinglish/exports/hinglish-pocket-tts-24l/config.yaml
                                     (the default; a private repo, so run `hf auth login` or set HF_TOKEN first)
  * a training checkpoint:           --checkpoint runs/student/checkpoint_00004000.pt (or hf://.../*.pt);
                                     6 vs 24 layers is detected and the matching base config used

Text may mix Devanagari, romanized Hindi, English, digits and currency; it is
normalized first (printed). The voice prompt should be 5-10 s of clean speech;
without --voice, the first voice shipped with the export is used, else an
Indian-English VCTK voice from kyutai/tts-voices.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = os.environ.get("HINGLISH_TTS_MODEL", "hf://harshsxna/pocket-tts-hinglish/exports/hinglish-pocket-tts-24l/config.yaml")


def resolve_config(model: str) -> str:
    p = Path(model)
    if p.is_dir():
        for name in ("config_local.yaml", "config.yaml"):
            if (p / name).exists():
                return str(p / name)
        raise FileNotFoundError(f"{model}: no config_local.yaml or config.yaml")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="text to speak (or use --text-file)")
    ap.add_argument("--text-file", type=Path, help="one utterance per line")
    ap.add_argument("--voice", help="voice prompt .wav (or an exported voice .safetensors)")
    ap.add_argument("--out", type=Path, default=Path("out.wav"), help="output .wav (with --text-file: a folder)")
    ap.add_argument("--model", default=None, help=f"export folder / config yaml / hf:// config (default {DEFAULT_MODEL})")
    ap.add_argument("--checkpoint", default=None, help="training checkpoint .pt instead of an export")
    ap.add_argument("--numbers", choices=["en", "hi"], default="en", help="read digits in English or Hindi words")
    ap.add_argument("--temp", type=float, default=None, help="sampling temperature (model default 0.3)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.text and not args.text_file:
        ap.error("give --text or --text-file")

    import soundfile
    import torch

    from hinglish_tts.synthesize import HinglishTTS

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    config = resolve_config(args.model) if args.model else (None if args.checkpoint else DEFAULT_MODEL)
    t0 = time.time()
    tts = HinglishTTS(config, args.voice, args.numbers, args.temp, args.device, args.checkpoint)
    print(f"loaded {args.checkpoint or config} in {time.time() - t0:.1f}s; voice {tts.voice_path}")
    texts = [args.text] if args.text else [l.strip() for l in args.text_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    outs = [args.out] if args.text else [args.out / f"{i:03d}.wav" for i in range(len(texts))]
    for text, out in zip(texts, outs, strict=True):
        print(f"text:       {text}\nnormalized: {tts.normalize(text)}")
        t0 = time.time()
        audio = tts(text)
        took = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(out, audio, tts.sample_rate)
        sec = len(audio) / tts.sample_rate
        print(f"wrote {out}: {sec:.2f} s of audio in {took:.2f} s (real-time factor {took / max(sec, 1e-6):.2f})")


if __name__ == "__main__":
    main()
