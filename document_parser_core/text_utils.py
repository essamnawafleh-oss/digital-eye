"""Text normalisation and cheap text-quality signals (no ML).

The quality signals exist because a non-empty PDF text layer is not necessarily
correct. Defects reproduced with PyMuPDF 2.0 during inspection:
  * visual-order Arabic extracted reversed ("ةصخرلا" for "الرخصة"),
  * Arabic Presentation Forms code points instead of base letters,
  * broken ToUnicode maps producing foreign characters inside Arabic words.
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_DECIMAL_SEP = "٫"  # ٫
ARABIC_THOUSANDS_SEP = "٬"  # ٬
TATWEEL = "ـ"

_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓۺ-ۿ]")
_ARABIC_BLOCKS = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_DIACRITICS = re.compile(r"[ً-ٰٟ]")


# Persian/Urdu letter forms that look identical to Arabic ones. The Arabic OCR model
# emits them (benchmark 2026-10-24: 'ليلى' read as 'لیلى' with U+06CC), and they break
# matching and comparison. Final Farsi yeh is the Arabic alef maqsura.
_FARSI_YEH_FINAL = re.compile("\u06cc(?![\u0621-\u064a\u06cc])")


def normalize_text(text: str) -> str:
    """Normalise for storage: NFKC folds Arabic presentation forms to base letters;
    Persian yeh/keheh become the Arabic letters they look like."""
    text = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
    if "\u06cc" in text or "\u06a9" in text:
        text = _FARSI_YEH_FINAL.sub("\u0649", text).replace("\u06cc", "\u064a").replace("\u06a9", "\u0643")
    return text


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation for label matching only (never for output values)."""
    text = normalize_text(text).casefold()
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    return re.sub(r"\s+", " ", text).strip()


def to_ascii_digits(text: str) -> str:
    return text.translate(ARABIC_INDIC)


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER.search(text))


def _is_garbage_char(ch: str) -> bool:
    if ch == "�":
        return True
    cp = ord(ch)
    if 0xE000 <= cp <= 0xF8FF:  # private use area: unmapped glyphs
        return True
    cat = unicodedata.category(ch)
    return cat in ("Cc", "Co", "Cn") and ch not in "\n\t\r"


def garbage_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(_is_garbage_char(c) for c in chars) / len(chars)


def presentation_form_ratio(text: str) -> float:
    arabic = len(_ARABIC_BLOCKS.findall(text)) + len(_PRESENTATION.findall(text))
    if not arabic:
        return 0.0
    return len(_PRESENTATION.findall(text)) / arabic


def arabic_char_ratio(text: str) -> float:
    chars = [c for c in normalize_text(text) if c.isalpha()]
    if not chars:
        return 0.0
    return sum(bool(_ARABIC_LETTER.match(c)) for c in chars) / len(chars)


def suspicious_token_ratio(tokens: list[str]) -> float:
    """Share of Arabic tokens that also contain letters of another script inside the
    word (typical of a broken glyph-to-Unicode map: lam-alef -> 'Ǆ' or CJK '㒡')."""
    arabic_tokens = [normalize_text(t) for t in tokens if has_arabic(normalize_text(t))]
    if not arabic_tokens:
        return 0.0
    bad = sum(1 for t in arabic_tokens if any(_is_foreign_letter(c) for c in t))
    return bad / len(arabic_tokens)


def _is_foreign_letter(ch: str) -> bool:
    if not unicodedata.category(ch).startswith(("L", "Co", "Cn")) and ch != "\ufffd":
        return False
    return not _ARABIC_BLOCKS.match(ch)


def reversed_arabic_score(tokens: list[str]) -> float | None:
    """Heuristic: share of reversed-order cues among Arabic words.

    Logical-order cues: word starts with the article 'ال'; ta marbuta 'ة' or alef
    maqsura 'ى' at word end. Reversed cues are their mirror images.
    Returns None if there is too little Arabic to judge.
    """
    normal = rev = 0
    for tok in tokens:
        word = re.sub(r"[^ء-ي]", "", normalize_text(tok))
        if len(word) < 3:
            continue
        if word.startswith("ال"):
            normal += 1
        if word.endswith("لا"):
            rev += 1
        if word[-1] in "ةى":
            normal += 1
        if word[0] in "ةى":
            rev += 1
    if normal + rev < 3:
        return None
    return rev / (normal + rev)


def tokens_of(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text) if t]
