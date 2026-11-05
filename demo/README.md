# Demo: Hinglish 24-layer vs. base English 24-layer, CPU

Both models read [`sentences.txt`](sentences.txt) through `infer.py`'s Hinglish frontend, with the same
voice prompt and seed. The voice is `hf://kyutai/tts-voices/vctk/p248_023_enhanced.wav`, an Indian-English
VCTK speaker who isn't in the training data. Machine: Apple M1, CPU, 4 torch threads, 2026-11-05.

| # | text (as typed -> normalized) | Hinglish 24L: audio s / RTF | base English 24L: audio s / RTF |
|---|---|---|---|
| 0 | Mera laptop hang ho gaya, isliye main meeting thodi der se join karunga. | 5.84 / 0.79 | 6.96 / 0.72 |
| 1 | Kal ki meeting 10:30 बजे है, please time pe aana. -> *Kal ki meeting ten thirty baje hai, please time pe aana.* | 4.32 / 0.75 | 5.60 / 0.77 |
| 2 | Bachpan mein hum garmiyon ki chhuttiyon mein naani ke ghar jaate the. | 4.48 / 0.76 | 6.80 / 0.69 |
| 3 | Aapka order ₹1,499 ka hai aur woh kal shaam tak deliver ho jayega. -> *... one thousand four hundred ninety nine rupees ...* | 7.60 / 0.70 | 8.88 / 0.70 |
| | **total** | 22.2 s in 16.6 s: **RTF 0.75** | 28.2 s in 20.3 s: **RTF 0.72** |

RTF = compute time / audio duration; lower is faster, and below 1 is faster than real time. Both models have
the same size, so the speed is the same. The Hinglish model's speech is shorter: 22 s vs 28 s for the same
text. Listen to the files to judge quality.

Regenerate (writes `hinglish_24l/000-003.wav` and `base_english_24l/000-003.wav`; needs `hf auth login`):

```bash
uv sync --extra cpu
V=hf://kyutai/tts-voices/vctk/p248_023_enhanced.wav
uv run infer.py --voice $V --text-file demo/sentences.txt --out demo/hinglish_24l
uv run infer.py --voice $V --text-file demo/sentences.txt --out demo/base_english_24l \
    --model "$(uv run python -c 'import pocket_tts, pathlib; print(pathlib.Path(pocket_tts.__file__).parent / "config/english_2026-09_24l.yaml")')"
```

`uvx_check.wav` is the export check: stock `uvx pocket-tts generate --config hf://.../hinglish-pocket-tts-24l/config.yaml`
with the default voice. It produced 5.4 s of audio in 6.0 s.
