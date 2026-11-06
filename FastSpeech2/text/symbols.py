""" Pashto phone set, written by fs2_pashto_tools.py from the lexicon's phone list.
The original English file is saved next to this one as symbols.py.orig. """

_pad = "_"
_special = "-"
_punctuation = "!'(),.:;? "
_silences = ["@sp", "@spn", "@sil"]

# "@" prefix = phone symbol (same convention as the ARPAbet symbols this replaces)
_pashto = ["@" + s for s in ["a", "ɑ", "e", "i", "d", "n", "r", "l", "o", "m", "k", "h", "w", "t", "ə", "u", "b", "s", "p", "j", "t͡ʃ", "x", "ɡ", "z", "ʃ", "q", "ɽ", "f", "ɣ", "t͡s", "d͡z", "d͡ʒ", "ʈ", "ɖ", "ŋ", "ʒ", "ɳ"]]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + _pashto + _silences
