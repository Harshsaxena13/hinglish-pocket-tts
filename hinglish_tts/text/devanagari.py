"""Rule-based Devanagari -> casual Roman (Hinglish) transliteration.

The goal is the spelling an Indian typing Hinglish would plausibly use, not a
reversible scholarly scheme: "samajhna" for समझना, "accha" for अच्छा, "kamla"
for कमला. Two things make that work:

* Hindi schwa deletion. Every consonant carries an inherent "a" that is often
  not pronounced. We delete it word-finally and, scanning right to left, in the
  V C a C V context (Ohala's rule), which covers the bulk of real words.
* Position-dependent vowel spelling. Long vowels are doubled inside a word
  ("baat", "jeet", "phool") and written single at the end ("kya", "nahi", "aalu"),
  the dominant convention in chat Hinglish.

Frequent words whose casual spelling the rules cannot predict ("main", "mein",
"hoon", "nahi") are handled by the lexicon in `lexicon.py`, which takes priority.

Combining marks are written as escapes below so the source stays readable.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

VIRAMA = "\u094d"
NUKTA = "\u093c"
ANUSVARA = "\u0902"
CHANDRABINDU = "\u0901"
VISARGA = "\u0903"
AVAGRAHA = "ऽ"
DANDA = "।"
DOUBLE_DANDA = "॥"
OM = "ॐ"
ZERO_WIDTH = {"\u200c", "\u200d"}
GYA = "\ue000"  # private-use placeholder for the ज्ञ ligature ("gy" in Hindi)

CONSONANTS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "ऩ": "n", "ऱ": "r", "ऴ": "l",  # ऩ ऱ ऴ (precomposed nukta forms)
}  # fmt: skip
# Consonant + nukta: Perso-Arabic and flapped sounds (क़ -> q, ज़ -> z, ड़ -> d, फ़ -> f).
NUKTA_FORMS: dict[str, str] = {
    "क": "q", "ख": "kh", "ग": "gh", "ज": "z", "झ": "zh",
    "ड": "d", "ढ": "dh", "फ": "f", "य": "y", "र": "r",
}  # fmt: skip
# Precomposed nukta letters are Unicode composition exclusions: NFC keeps them
# split, but text in the wild uses both forms.
PRECOMPOSED_NUKTA = {
    "क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज",
    "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य",
}  # fmt: skip
# (spelling inside a word, spelling at the end of a word)
INDEPENDENT_VOWELS: dict[str, tuple[str, str]] = {
    "अ": ("a", "a"), "आ": ("aa", "aa"), "इ": ("i", "i"), "ई": ("ee", "i"),
    "उ": ("u", "u"), "ऊ": ("oo", "u"), "ऋ": ("ri", "ri"), "ॠ": ("ri", "ri"),
    "ऌ": ("li", "li"), "ए": ("e", "e"), "ऐ": ("ai", "ai"), "ओ": ("o", "o"),
    "औ": ("au", "au"), "ऑ": ("o", "o"), "ऍ": ("e", "e"), "ऎ": ("e", "e"), "ऒ": ("o", "o"),
}  # fmt: skip
MATRAS: dict[str, tuple[str, str]] = {
    "\u093e": ("aa", "a"),  # aa sign
    "\u093f": ("i", "i"),  # short i
    "\u0940": ("ee", "i"),  # long ii
    "\u0941": ("u", "u"),  # short u
    "\u0942": ("oo", "u"),  # long uu
    "\u0943": ("ri", "ri"),  # vocalic r
    "\u0944": ("ri", "ri"),  # vocalic rr
    "\u0945": ("e", "e"),  # candra e
    "\u0946": ("e", "e"),  # short e
    "\u0947": ("e", "e"),  # e
    "\u0948": ("ai", "ai"),  # ai
    "\u0949": ("o", "o"),  # candra o (English loanwords: "doctor")
    "\u094a": ("o", "o"),  # short o
    "\u094b": ("o", "o"),  # o
    "\u094c": ("au", "au"),  # au
    "\u0962": ("li", "li"),  # vocalic l
    "\u0963": ("li", "li"),  # vocalic ll
}
# A vowel letter right after another vowel glides: गए -> "gaye", आइए -> "aaiye".
GLIDE_AFTER_VOWEL = {"ए": "ye", "इ": "i", "ई": "i", "ऊ": "u", "उ": "u", "ओ": "o"}
DEVANAGARI_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
LABIALS = ("p", "b", "m", "f")
SEMIVOWEL_CODAS = {"r", "y", "v"}  # "mitra", "satya", "vishva": final schwa survives


def is_devanagari(ch: str) -> bool:
    return "\u0900" <= ch <= "ॿ" or "\ua8e0" <= ch <= "\ua8ff"


def has_devanagari(text: str) -> bool:
    return any(is_devanagari(c) for c in text)


@dataclass
class _Unit:
    """One orthographic syllable: an optional consonant and its vowel."""

    cons: str | None  # roman consonant, None for a bare vowel letter
    vowel: str  # independent vowel letter or matra; "" for schwa / virama
    kind: str  # "schwa" (inherent vowel) | "none" (virama) | "matra" | "independent"
    nasal: bool = False
    visarga: bool = False
    deleted: bool = False  # inherent schwa dropped by schwa deletion

    @property
    def has_vowel(self) -> bool:
        return self.kind != "none" and not self.deleted


def _parse(word: str) -> list[_Unit]:
    seq = "".join(
        PRECOMPOSED_NUKTA[c] + NUKTA if c in PRECOMPOSED_NUKTA else c for c in word
    ).replace("ज" + VIRAMA + "ञ", GYA)
    units: list[_Unit] = []
    i = 0
    while i < len(seq):
        c = seq[i]
        if c in CONSONANTS or c == GYA:
            roman = "gy" if c == GYA else CONSONANTS[c]
            if i + 1 < len(seq) and seq[i + 1] == NUKTA:
                roman = NUKTA_FORMS.get(c, roman)
                i += 1
            unit = _Unit(cons=roman, vowel="", kind="schwa")
            nxt = seq[i + 1] if i + 1 < len(seq) else ""
            if nxt == VIRAMA:
                unit.kind = "none"
                i += 1
            elif nxt in MATRAS:
                unit.vowel, unit.kind = nxt, "matra"
                i += 1
            units.append(unit)
        elif c in INDEPENDENT_VOWELS:
            units.append(_Unit(cons=None, vowel=c, kind="independent"))
        elif c in (ANUSVARA, CHANDRABINDU):
            if units:
                units[-1].nasal = True
            else:
                units.append(_Unit(cons="n", vowel="", kind="none"))
        elif c == VISARGA and units:
            units[-1].visarga = True
        # Anything else (zero-width joiners, avagraha, stray combining marks) is silent.
        i += 1
    return units


def _delete_schwas(units: list[_Unit]) -> None:
    n = len(units)
    if n < 2:
        return
    last = units[-1]
    # Word-final schwa: dropped ("ghar", "bahut") unless the word ends in a
    # consonant cluster closed by r/y/v ("mitra", "satya", "vishva").
    if last.kind == "schwa" and not last.nasal:
        cluster_final = units[-2].kind == "none"
        if not (cluster_final and last.cons in SEMIVOWEL_CODAS):
            last.deleted = True
    # Medial schwa (Ohala): V C a C V, right to left, never creating a 3-consonant cluster.
    for i in range(n - 2, 0, -1):
        u = units[i]
        if u.kind != "schwa" or u.nasal or u.visarga:
            continue
        prev, nxt = units[i - 1], units[i + 1]
        if not prev.has_vowel or prev.nasal or prev.visarga:
            continue  # the previous syllable is closed already: "zindagi", not "zindgi"
        if nxt.cons is None or not nxt.has_vowel:
            continue
        u.deleted = True


def _vowel_spelling(unit: _Unit, is_final: bool, prev_has_vowel: bool) -> str:
    if unit.kind == "schwa":
        return "" if unit.deleted else "a"
    if unit.kind == "none":
        return ""
    if unit.kind == "independent":
        if unit.cons is None and prev_has_vowel and unit.vowel in GLIDE_AFTER_VOWEL:
            return GLIDE_AFTER_VOWEL[unit.vowel]
        medial, final = INDEPENDENT_VOWELS[unit.vowel]
    else:
        medial, final = MATRAS[unit.vowel]
    return final if is_final else medial


def transliterate_word(word: str) -> str:
    """Transliterate one Devanagari word (no spaces) to casual Roman."""
    units = _parse(unicodedata.normalize("NFC", word))
    if not units:
        return ""
    _delete_schwas(units)
    out: list[str] = []
    n = len(units)
    for i, u in enumerate(units):
        is_final = i == n - 1
        prev = units[i - 1] if i > 0 else None
        cons = u.cons or ""
        if prev is not None and prev.kind == "none":
            # च्छ -> "cch" ("accha", "iccha") rather than "chchh".
            if cons == "chh" and prev.cons == "ch":
                out[-1] = "c"
                cons = "ch"
            # व as the second member of a cluster is a "w" ("swagat", "vishwas", "dwara").
            elif cons == "v":
                cons = "w"
        vowel = _vowel_spelling(u, is_final, prev is not None and prev.has_vowel)
        if u.nasal:
            if u.kind == "schwa":
                vowel = "a"  # a nasalised schwa is always pronounced ("hans", "ant")
            nxt = units[i + 1].cons if i + 1 < n else None
            if nxt and nxt.startswith(LABIALS):
                vowel += "m"  # "sambandh", "champa"
            elif is_final and vowel == "e":
                vowel += "in"  # करें -> "karein", में -> "mein"
            else:
                vowel += "n"
        if u.visarga:
            vowel += "h"
        out.append(cons + vowel)
    return "".join(out)


def transliterate(text: str) -> str:
    """Transliterate every Devanagari run in `text`, leaving other characters alone."""
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(transliterate_word("".join(buf)))
            buf.clear()

    for ch in text:
        if ch in DEVANAGARI_DIGITS:
            flush()
            out.append(DEVANAGARI_DIGITS[ch])
        elif ch in (DANDA, DOUBLE_DANDA):
            flush()
            out.append(".")
        elif ch == OM:
            flush()
            out.append("om")
        elif is_devanagari(ch) or ch in ZERO_WIDTH:
            buf.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)
