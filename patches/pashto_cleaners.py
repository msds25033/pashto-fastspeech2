

# ---------------------------------------------------------------------------
# Pashto. Append-only addition: nothing above is changed.
# Must behave exactly like normalize_words() in make_lexicon.py, because the
# lexicon keys and the aligner's .lab files were built with that function.
# ---------------------------------------------------------------------------
import unicodedata

# Written abbreviations that are *spoken* as words.
_PASHTO_SPOKEN_FORMS = {"\ufdfa": "صلی الله علیه وسلم"}  # ﷺ


def _pashto_is_annotation_mark(ch):
    o = ord(ch)  # Quranic recitation marks: not pronunciation
    return 0x0610 <= o <= 0x061A or 0x06D6 <= o <= 0x06ED


def pashto_words(text):
    """Pashto text -> list of words, punctuation removed (see make_lexicon.py)."""
    for written, spoken in _PASHTO_SPOKEN_FORMS.items():
        text = text.replace(written, f" {spoken} ")
    out = []
    for token in unicodedata.normalize("NFC", text).split():
        chars = []
        for ch in token:
            cat = unicodedata.category(ch)
            if cat[0] in "PS":                               # punctuation / symbols
                chars.append(" ")
            elif cat in ("Cf", "Cc") or ch == "\u0640":      # zero-width, bidi, tatweel
                continue
            elif cat == "Mn" and _pashto_is_annotation_mark(ch):
                continue
            else:
                chars.append(ch)
        out.extend("".join(chars).split())
    return out


def pashto_cleaners(text):
    '''Pipeline for Pashto script: keeps the Pashto letters, removes punctuation.'''
    return " ".join(pashto_words(text))
