"""Text frontend shared by data preparation and inference.

Inference:  any mix of Devanagari / Roman Hinglish / English / digits
            -> clean, sentence-cased romanized Hinglish the model was trained on.
Training:   a corpus transcript -> the same representation, or None when the
            transcript cannot be trusted (digits with ambiguous readings, other
            scripts, markup), plus optional spelling augmentation.
"""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from .devanagari import DEVANAGARI_DIGITS, is_devanagari
from .numbers import verbalize_numbers
from .romanize import Romanizer, Token
from .variants import apply_variants

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'",
    "“": "", "”": "", "„": "", '"': "", "«": "", "»": "",
    "–": ",", "—": ",", "―": ",", "…": "...", ";": ",",
    "।": ".", "॥": ".", "\u00a0": " ",
}  # fmt: skip
_KEEP_PUNCT = set(".,?!'-:")
_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_MARKUP = set("()[]{}<>*#@/\\|_=+~^$%&")
_SENTENCE_END = re.compile(r"([.?!]+)(\s+)([a-z])")


def _clean_chars(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(_PUNCT_MAP.get(c, c) for c in text)
    text = "".join(DEVANAGARI_DIGITS.get(c, c) for c in text)
    # Drop emoji, symbols and control characters.
    return "".join(
        c for c in text if not unicodedata.category(c).startswith(("So", "Sk", "Cs", "Cc", "Co"))
        or c in "\n\t"
    )


def _render(tokens: list[Token]) -> str:
    parts: list[str] = []
    for tok in tokens:
        if tok.origin == "punct" and tok.text not in _KEEP_PUNCT:
            parts.append(" ")
            continue
        text = tok.text
        if tok.origin == "en" and text == "i":
            text = "I"
        parts.append(text)
    text = "".join(parts)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,?!:])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([,:])(?=[^\s\d])", r"\1 ", text)
    text = re.sub(r"([.?!])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"([?!])[?!.]+", r"\1", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"^[\s,.:!?-]+", "", text).strip()
    # Sentence case: the model was trained on cased, punctuated text.
    text = _SENTENCE_END.sub(lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)
    return text[:1].upper() + text[1:]


class HinglishFrontend:
    """Callable text frontend. `numbers`: read digits as "en" or "hi" words."""

    def __init__(
        self, external_lexicon: str | Path | dict[str, str] | None = None, numbers: str = "en"
    ):
        self.romanizer = Romanizer(external_lexicon)
        self.numbers = numbers

    def __call__(self, text: str) -> str:
        """Inference: normalize arbitrary user text. Returns "" for empty input."""
        text = _clean_chars(text).replace("&", " and ")
        text = _TAGS.sub(" ", text)
        text = verbalize_numbers(text, self.numbers)
        return _render(self.romanizer.tokens(text))

    def training_text(
        self, text: str, rng: random.Random | None = None, variant_prob: float = 0.0
    ) -> str | None:
        """Corpus transcript -> training text, or None if it should be dropped.

        Digits are spelled out in the transcript's dominant language (Hindi words
        for Devanagari-majority text, English otherwise); a speaker who read them
        the other way gets a high alignment loss and is filtered out later.
        """
        text = _TAGS.sub(" ", _clean_chars(text))
        for c in text:
            if c.isalpha() and not (c.isascii() or is_devanagari(c)):
                return None  # another script (Urdu, Bengali, ...)
            if unicodedata.category(c) == "Mn" and not is_devanagari(c):
                return None
        deva = sum(c.isalpha() and is_devanagari(c) for c in text)
        latin = sum(c.isascii() and c.isalpha() for c in text)
        lang = "hi" if deva > latin else "en"
        text = verbalize_numbers(text, lang).replace("&", " aur " if lang == "hi" else " and ")
        if any(c.isdigit() for c in text) or any(c in _MARKUP for c in text):
            return None
        tokens = self.romanizer.tokens(text)
        if rng is not None:
            tokens = apply_variants(tokens, rng, variant_prob)
        out = _render(tokens)
        if len(out.split()) < 3:
            return None
        if out[-1] not in ".?!":
            out += "."
        return out
