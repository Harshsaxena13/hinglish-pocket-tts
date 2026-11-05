import random

import pytest

from hinglish_tts.text import HinglishFrontend, number_to_words, verbalize_numbers
from hinglish_tts.text.devanagari import transliterate, transliterate_word
from hinglish_tts.text.lexicon import acronym_of
from hinglish_tts.text.romanize import Romanizer
from hinglish_tts.text.variants import apply_variants


@pytest.mark.parametrize(
    "native, roman",
    [
        ("समझना", "samajhna"),  # medial schwa deletion
        ("कमला", "kamla"),
        ("बहुत", "bahut"),  # final schwa deletion
        ("नमस्ते", "namaste"),  # no deletion before a cluster
        ("ज़िंदगी", "zindagi"),  # a nasal closes the syllable: keep the schwa
        ("मित्र", "mitra"),  # final schwa kept after a cluster ending in r
        ("अच्छा", "accha"),  # cch, final aa -> a
        ("ज्ञान", "gyaan"),
        ("लड़का", "ladka"),  # nukta flap
        ("स्वागत", "swaagat"),  # v after a consonant -> w
        ("गए", "gaye"),  # vowel glide
        ("संबंध", "sambandh"),  # anusvara before a labial -> m
        ("करें", "karein"),
        ("फ़िल्म", "film"),
        ("क्षेत्र", "kshetra"),
    ],
)
def test_transliterate_word(native: str, roman: str) -> None:
    assert transliterate_word(native) == roman


def test_transliterate_keeps_non_devanagari_and_maps_danda_and_digits() -> None:
    assert transliterate("मेरा phone ३ बजे।") == "mera phone 3 baje."


def test_lexicon_overrides_rules() -> None:
    r = Romanizer()
    assert r("मैं नहीं जानता कि वह कहाँ है") == "main nahi jaanta ki woh kahan hai"


def test_loanwords_and_acronyms() -> None:
    r = Romanizer()
    assert r("मेरा कंप्यूटर") == "mera computer"
    assert acronym_of("पीएम") == "PM"
    assert acronym_of("आईआईटी") == "IIT"
    assert acronym_of("जी") is None  # a single letter name is the Hindi "ji"
    assert acronym_of("बीबी") is None  # real word


def test_external_lexicon_has_lower_priority_than_curated() -> None:
    r = Romanizer({"मौसम": "mosam", "किताब": "kitaab"})
    assert r("मौसम किताब") == "mausam kitaab"


def test_numbers_indian_system() -> None:
    assert number_to_words(150000) == "one lakh fifty thousand"
    assert number_to_words(150000, "hi") == "ek lakh pachaas hazaar"
    assert number_to_words(10**9) == "one hundred crore"
    assert number_to_words(99, "hi") == "ninyaanave"
    assert verbalize_numbers("₹1,499") == "one thousand four hundred ninety nine rupees"
    assert verbalize_numbers("2024") == "two thousand twenty four"
    assert verbalize_numbers("1998", "hi") == "unnees sau atthaanave"
    assert verbalize_numbers("50%") == "fifty percent"
    assert verbalize_numbers("21st") == "twenty first"
    assert verbalize_numbers("10:30") == "ten thirty"
    assert verbalize_numbers("9876543210") == "nine eight seven six five four three two one zero"


def test_frontend_inference() -> None:
    f = HinglishFrontend()
    out = f("kal मीटिंग 10:30 बजे है!!! 😊 please time पे आना")
    assert out == "Kal meeting ten thirty baje hai! Please time pe aana"
    assert f("i think वो सही है") == "I think woh sahi hai"


def test_frontend_training_rejects_untrustworthy_text() -> None:
    f = HinglishFrontend()
    assert f.training_text("मुझे 3 किताबें चाहिए") == "Mujhe teen kitaabein chahiye."  # Hindi-majority: Hindi digits
    assert f.training_text("I have 2 cats at home") == "I have two cats at home."
    assert f.training_text("price is ₹500 only") == "Price is five hundred rupees only."
    assert f.training_text("یہ اردو ہے بھائی") is None  # another script
    assert f.training_text("हाँ") is None  # too short to cut at a word boundary
    out = f.training_text("[noise] आज मौसम बहुत अच्छा है")
    assert out == "Aaj mausam bahut accha hai."


def test_variants_touch_only_romanized_hindi() -> None:
    tokens = Romanizer().tokens("मेरा laptop बहुत slow है")
    varied = apply_variants(tokens, random.Random(0), prob=1.0)
    by_origin = {t.origin: [] for t in varied}
    for t in varied:
        by_origin[t.origin].append(t.text)
    assert by_origin["en"] == ["laptop", "slow"]
    assert "".join(t.text for t in varied) != "".join(t.text for t in tokens)
