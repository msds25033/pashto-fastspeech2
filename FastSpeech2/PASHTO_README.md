# Bundled FastSpeech2 (patched for Pashto)

This folder is [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) at commit
`d4e79eb52e8b01d24703b2dfc0385544092958f3` (MIT licence, see LICENSE), included so the
repo works without cloning anything. It has already been through
`python fs2_pashto_tools.py patch`, `modernize` and `symbols`. The exact changes are in
`../patches/fastspeech2_pashto.diff`, and the reasons are listed in the main README.

Removed from the upstream checkout because the Pashto pipeline doesn't use them:
`demo/`, `img/`, `index.html`, the English/Mandarin lexicons, the LJSpeech/LibriTTS/AISHELL3
file lists in `preprocessed_data/`, and the LJSpeech-only vocoder
`hifigan/generator_LJSpeech.pth.tar.zip` (config/Pashto uses the universal vocoder).

`pipeline.py setup` is still the first step: it re-applies the patches (a no-op here),
unzips the vocoder, and copies `config/Pashto` and the lexicon in here. Don't edit
`config/Pashto` or `lexicon/` inside this folder; edit the copies at the repo root.
