""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from .numbers import normalize_numbers
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text



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
