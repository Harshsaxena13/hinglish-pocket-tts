"""Spelling augmentation for romanized Hindi.

Hinglish has no standard orthography: the same word shows up as "bahut",
"bohot" and "bahot"; "baat" and "bat"; "zindagi" and "jindagi". Training only on
our canonical spellings would make the model brittle to how people actually
type. So a fraction of romanized-Hindi words in the training transcripts get a
plausible alternative spelling (deterministic per clip, so every epoch sees the
same text for the same audio). English words are never touched.
"""

from __future__ import annotations

import random
import re

from .romanize import Token

# (pattern, replacement): each one is a common real-world spelling alternation.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"aa"), "a"),  # baat -> bat
    (re.compile(r"ee"), "i"),  # jeet -> jit
    (re.compile(r"oo"), "u"),  # phool -> phul
    (re.compile(r"chh"), "ch"),  # chhota -> chota
    (re.compile(r"ph"), "f"),  # phir -> fir
    (re.compile(r"w"), "v"),  # wahan -> vahan
    (re.compile(r"(?<![sc])v"), "w"),  # vishay -> wishay (but never "sv"/"cv")
    (re.compile(r"q"), "k"),  # waqt -> wakt
    (re.compile(r"z"), "j"),  # zindagi -> jindagi
]


def vary_word(word: str, rng: random.Random) -> str:
    applicable = [(p, r) for p, r in _RULES if p.search(word)]
    if not applicable:
        return word
    pattern, repl = rng.choice(applicable)
    return pattern.sub(repl, word)


def apply_variants(tokens: list[Token], rng: random.Random, prob: float) -> list[Token]:
    """Respell each romanized-Hindi token with probability `prob`."""
    if prob <= 0:
        return tokens
    out = []
    for tok in tokens:
        if tok.origin == "hi" and rng.random() < prob:
            text = rng.choice(tok.variants) if tok.variants else vary_word(tok.text, rng)
            tok = Token(text, tok.origin, tok.variants)
        out.append(tok)
    return out
