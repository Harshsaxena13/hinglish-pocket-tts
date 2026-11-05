"""Turn mixed Devanagari / Latin text into romanized Hinglish tokens.

Each Devanagari word is resolved in priority order:
    curated Hindi lexicon -> English loanwords -> acronyms -> external lexicon -> rules
Latin words (English, or Hinglish already typed in Roman) pass through untouched.
Tokens keep their origin so training-time spelling augmentation only touches
romanized Hindi, never English words.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import devanagari, lexicon

# Devanagari block (letters + combining marks) and joiners; digits are split out.
_DEVA_WORD = r"[ऀ-ॣ॰-ॿ꣠-ꣿ‌‍]+"
_TOKEN = re.compile(
    rf"(?P<deva>{_DEVA_WORD})"
    r"|(?P<latin>[A-Za-z]+(?:['’][A-Za-z]+)*)"
    r"|(?P<num>[0-9०-९]+(?:[.,:][0-9०-९]+)*)"
    r"|(?P<space>\s+)"
    r"|(?P<other>.)",
    re.DOTALL,
)


@dataclass
class Token:
    text: str
    # "hi": romanized Hindi, "en": English (Latin input or loanword), "acr": acronym,
    # "num": digits, "space", "punct"
    origin: str
    variants: tuple[str, ...] = field(default=())


class Romanizer:
    def __init__(self, external_lexicon: dict[str, str] | str | Path | None = None):
        if isinstance(external_lexicon, (str, Path)):
            external_lexicon = lexicon.load_lexicon_tsv(external_lexicon)
        self.external = external_lexicon or {}

    def word(self, native: str) -> Token:
        native = unicodedata.normalize("NFC", native)
        if native in lexicon.HINDI:
            canonical, *variants = lexicon.HINDI[native]
            return Token(canonical, "hi", tuple(variants))
        if native in lexicon.ENGLISH_LOANWORDS:
            return Token(lexicon.ENGLISH_LOANWORDS[native][0], "en")
        acronym = lexicon.acronym_of(native)
        if acronym:
            return Token(acronym, "acr")
        if native in self.external:
            return Token(self.external[native], "hi")
        return Token(devanagari.transliterate_word(native), "hi")

    def tokens(self, text: str) -> list[Token]:
        out: list[Token] = []
        for m in _TOKEN.finditer(unicodedata.normalize("NFC", text)):
            kind, value = m.lastgroup, m.group()
            if kind == "deva":
                tok = self.word(value)
                if tok.text:
                    out.append(tok)
            elif kind == "latin":
                out.append(Token(value, "en"))
            elif kind == "num":
                out.append(Token(devanagari.transliterate(value), "num"))
            elif kind == "space":
                out.append(Token(" ", "space"))
            elif value in (devanagari.DANDA, devanagari.DOUBLE_DANDA):
                out.append(Token(".", "punct"))
            else:
                out.append(Token(value, "punct"))
        return out

    def __call__(self, text: str) -> str:
        return "".join(t.text for t in self.tokens(text))
