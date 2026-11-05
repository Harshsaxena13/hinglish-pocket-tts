"""Spell out numbers the way Indians read them: Indian grouping (lakh, crore),
in English words ("one lakh fifty thousand") or romanized Hindi ("dedh lakh" is
too idiomatic; we say "ek lakh pachaas hazaar").

The model never saw digits during training (utterances with digits are dropped,
since their reading is ambiguous), so every digit must be verbalized first.
"""

from __future__ import annotations

import re

_EN_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

# Hindi has an irregular word for every number below 100.
_HI_0_99 = (
    "shoonya ek do teen chaar paanch chhah saat aath nau das gyaarah baarah terah chaudah "
    "pandrah solah satrah athaarah unnees bees ikkees baais teis chaubees pachchees "
    "chhabbees sattaais atthaais untees tees iktees battees taintees chauntees paintees "
    "chhattees saintees adtees untaalees chaalees iktaalees bayaalees taintaalees "
    "chauvaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas ikyaavan "
    "baavan tirpan chauvan pachpan chhappan sattaavan atthaavan unsath saath iksath "
    "baasath tirsath chausath painsath chhiyaasath sadsath adsath unhattar sattar "
    "ikhattar bahattar tihattar chauhattar pachhattar chhihattar sathattar athhattar "
    "unaasi assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sattaasi attaasi "
    "navaasi nabbe ikyaanave baanave tiraanave chauraanave pachaanave chhiyaanave "
    "sattaanave atthaanave ninyaanave"
).split()
assert len(_HI_0_99) == 100

_EN_ORDINAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}  # fmt: skip


def _en_below_100(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    tens, ones = divmod(n, 10)
    return _EN_TENS[tens] + (f" {_EN_ONES[ones]}" if ones else "")


def _en_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(f"{_EN_ONES[hundreds]} hundred")
    if rest:
        parts.append(_en_below_100(rest))
    return " ".join(parts)


def _indian_groups(n: int) -> list[tuple[int, str]]:
    """Split into (crore, lakh, thousand, rest) with the Indian place values."""
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, rest = divmod(n, 1000)
    return [(crore, "crore"), (lakh, "lakh"), (thousand, "thousand"), (rest, "")]


def number_to_words(n: int, lang: str = "en") -> str:
    """Cardinal number in words with Indian grouping. lang: "en" or "hi"."""
    if n < 0:
        return ("minus " if lang == "en" else "maainas ") + number_to_words(-n, lang)
    if n == 0:
        return "zero" if lang == "en" else "shoonya"
    words: list[str] = []
    for value, unit in _indian_groups(n):
        if not value:
            continue
        if unit == "crore" and value >= 100:
            spelled = number_to_words(value, lang)  # "one hundred twenty crore"
        elif lang == "en":
            spelled = _en_below_1000(value)
        else:
            hundreds, rest = divmod(value, 100)
            spelled = " ".join(
                p for p in (f"{_HI_0_99[hundreds]} sau" if hundreds else "", _HI_0_99[rest] if rest else "") if p
            )
        unit_word = {"thousand": "hazaar"}.get(unit, unit) if lang == "hi" else unit
        words.append(f"{spelled} {unit_word}".strip())
    return " ".join(words)


def year_to_words(n: int, lang: str = "en") -> str:
    """1998 -> "nineteen ninety eight"; 2024 -> "two thousand twenty four"."""
    if 1100 <= n <= 1999 and n % 100 != 0:
        hi, lo = divmod(n, 100)
        if lang == "en":
            return f"{_en_below_100(hi)} {_en_below_100(lo) if lo >= 10 else 'oh ' + _EN_ONES[lo]}"
        return f"{_HI_0_99[hi]} sau {_HI_0_99[lo]}"
    return number_to_words(n, lang)


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n, "en").split()
    last = words[-1]
    if last in _EN_ORDINAL:
        words[-1] = _EN_ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def digits_to_words(digits: str, lang: str = "en") -> str:
    table = _EN_ONES if lang == "en" else _HI_0_99
    return " ".join(table[int(d)] for d in digits if d.isdigit())


def _decimal(int_part: str, frac: str, lang: str) -> str:
    return f"{number_to_words(int(int_part), lang)} point {digits_to_words(frac, lang)}"


_CURRENCY = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _read_number(token: str, lang: str) -> str:
    plain = token.replace(",", "")
    if "." in plain:
        int_part, frac = plain.split(".", 1)
        return _decimal(int_part or "0", frac, lang)
    if len(plain) >= 7 and "," not in token:
        return digits_to_words(plain, lang)  # phone numbers, codes: digit by digit
    if len(plain) > 1 and plain.startswith("0"):
        return digits_to_words(plain, lang)
    value = int(plain)
    if len(plain) == 4 and "," not in token and 1100 <= value <= 2099:
        return year_to_words(value, lang)
    return number_to_words(value, lang)


def verbalize_numbers(text: str, lang: str = "en") -> str:
    """Replace every number in `text` by words. lang: "en" or "hi"."""
    rupees = "rupees" if lang == "en" else "rupaye"
    text = _CURRENCY.sub(lambda m: f"{_read_number(m.group(1), lang)} {rupees}", text)
    text = _PERCENT.sub(lambda m: f"{_read_number(m.group(1), lang)} percent", text)
    text = _ORDINAL.sub(lambda m: ordinal_to_words(int(m.group(1))), text)

    def time(m: re.Match[str]) -> str:
        h, mins = int(m.group(1)), int(m.group(2))
        hour = number_to_words(h, lang)
        if mins == 0:
            return f"{hour} baje" if lang == "hi" else f"{hour} o'clock"
        return f"{hour} {number_to_words(mins, lang) if mins >= 10 else ('oh ' if lang == 'en' else '') + number_to_words(mins, lang)}"

    text = _TIME.sub(time, text)
    return _NUMBER.sub(lambda m: _read_number(m.group(0), lang), text)
