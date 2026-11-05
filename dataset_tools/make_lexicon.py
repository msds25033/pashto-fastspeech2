"""
Build a Pashto pronunciation lexicon (MFA / FastSpeech2 format) from the
dataset written by make_chunks_metadata.py.

    <dataset>/metadata.csv   <clip file>|<pashto text>
        --normalize--> unique words
        --nemo-pashto-g2p (one word per line)--> glued IPA string
        --split into phones--> lexicon.txt   (word <TAB> p1 p2 p3 ...)

Also writes <lexicon>.phones.txt: every phone used and how often. That list
is what you need for FastSpeech2's text/symbols.py.

Optionally (--write-corpus) writes the Montreal Forced Aligner corpus:
    <corpus>/<speaker>/<clip>.wav   (mono, 22.05 kHz)
    <corpus>/<speaker>/<clip>.lab   (text normalised EXACTLY like the lexicon keys)
Lexicon keys and .lab words must match character for character or MFA treats
the word as unknown, which is why both come from the same normalize_words().
Use that same function when you synthesise new text.

Usage (Colab, from the nemo-pashto-g2p-conformer folder; needs its inference.py
and the packages in its requirements.txt, nothing else):

    python make_lexicon.py --dataset pashto_dataset \
        --out lexicon/pashto-lexicon.txt --write-corpus raw_data/Pashto
"""

import argparse
import collections
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Text normalisation (shared by lexicon keys and .lab files)
# ---------------------------------------------------------------------------

# Written abbreviations that are *spoken* as words. Without this the audio
# contains a phrase the transcript doesn't, which hurts alignment. Edit freely.
SPOKEN_FORMS = {"\ufdfa": "صلی الله علیه وسلم"}  # ﷺ

TIE_BARS = {"\u0361", "\u035c"}


def _is_annotation_mark(ch):
    """Quranic recitation marks (e.g. U+0615). Not pronunciation, and the G2P
    model has never seen them."""
    o = ord(ch)
    return 0x0610 <= o <= 0x061A or 0x06D6 <= o <= 0x06ED


def normalize_words(text, strip_diacritics=False):
    """Pashto text -> list of words with punctuation removed.

    Punctuation becomes a space (transcripts often glue words together after
    '۔' or '-'). Zero-width/bidi characters are deleted. Vowel marks such as
    fatha are kept unless strip_diacritics=True.
    """
    for written, spoken in SPOKEN_FORMS.items():
        text = text.replace(written, f" {spoken} ")
    out = []
    for token in unicodedata.normalize("NFC", text).split():
        chars = []
        for ch in token:
            cat = unicodedata.category(ch)
            if cat[0] in "PS":                      # punctuation / symbols
                chars.append(" ")
            elif cat in ("Cf", "Cc") or ch == "\u0640":   # zero-width, bidi, tatweel
                continue
            elif cat == "Mn" and (strip_diacritics or _is_annotation_mark(ch)):
                continue
            else:
                chars.append(ch)
        out.extend("".join(chars).split())
    return out


def has_latin_or_digit(word):
    return any(ch.isdigit() or ("a" <= ch.lower() <= "z") for ch in word)


# ---------------------------------------------------------------------------
# Glued IPA string -> list of phones
# ---------------------------------------------------------------------------

def split_phones(ipa):
    """'t͡ʃe' -> ['t͡ʃ', 'e'].  Each phone is a base character plus any
    combining marks / modifier letters (ː ʰ ʷ ...); a tie bar joins the base
    character on its left with the one on its right (affricates)."""
    ipa = unicodedata.normalize("NFC", ipa).replace("g", "\u0261")  # ASCII g -> ɡ
    ipa = ipa.replace("\u035c", "\u0361")                          # one kind of tie bar
    phones, i, n = [], 0, len(ipa)
    while i < n:
        if ipa[i].isspace():
            i += 1
            continue
        j = i + 1
        while j < n:
            ch = ipa[j]
            if ch in TIE_BARS:
                j += 2 if j + 1 < n else 1          # swallow the tie bar and the next base
            elif unicodedata.combining(ch) or unicodedata.category(ch) in ("Lm", "Sk"):
                j += 1
            else:
                break
        phones.append(ipa[i:j])
        i = j
    return phones


def read_extra_lexicon(path):
    """Hand-written pronunciations: one 'word<TAB>p1 p2 ...' per line ('#' = comment)."""
    entries = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        word, _, ph = ln.partition("\t")
        if not ph.strip():                       # tolerate spaces instead of a tab
            parts = ln.split()
            word, ph = parts[0], " ".join(parts[1:])
        phones = ph.split()
        if word.strip() and phones:
            entries[unicodedata.normalize("NFC", word.strip())] = phones
    if not entries:
        raise SystemExit(f"--extra-lexicon: no entries found in {path}")
    return entries


def parse_phone_map(spec):
    """'ħ=h,ð=z,ʰ=' -> {'ħ': 'h', 'ð': 'z', 'ʰ': ''}  (empty value = delete the phone)."""
    mapping = {}
    for pair in filter(None, (x.strip() for x in (spec or "").split(","))):
        src, _, dst = pair.partition("=")
        canon = lambda x: unicodedata.normalize("NFC", x.strip()).replace("\u035c", "\u0361")  # noqa: E731
        mapping[canon(src).replace("g", "\u0261")] = canon(dst).replace("g", "\u0261")
    return mapping


# ---------------------------------------------------------------------------
# G2P (same approach as run_g2p in docx_to_alignment.py, but self-contained)
# ---------------------------------------------------------------------------

def run_g2p(lines, g2p_repo_dir, tmp_dir):
    """Run the repo's inference.py on one text per line; return the phoneme
    strings in the same order."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for old in tmp_dir.glob("*.json"):      # never pick up output from an earlier run
        old.unlink()
    text_file = tmp_dir / "words.txt"
    text_file.write_text("\n".join(lines), encoding="utf-8")

    started = time.time() - 1
    cmd = [sys.executable, "inference.py", "--text-file", str(text_file.resolve()),
           "--batch-size", "32", "--json"]
    result = subprocess.run(cmd, cwd=g2p_repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"G2P inference.py failed:\n{result.stderr[-3000:]}")

    # Where --text-file + --json writes its output isn't documented: look for a
    # JSON file created by THIS run, then fall back to parsing stdout.
    fresh = [f for f in list(tmp_dir.glob("*.json")) + list(Path(g2p_repo_dir).glob("*.json"))
             if f.stat().st_mtime >= started]
    data = None
    if fresh:
        data = json.loads(sorted(fresh, key=lambda f: f.stat().st_mtime)[-1]
                          .read_text(encoding="utf-8"))
    else:
        out = result.stdout.strip()
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            rows = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
            if rows:
                data = [json.loads(ln) for ln in rows]
    if data is None:
        raise SystemExit("Could not find the G2P output. stdout was:\n"
                         f"{result.stdout[-2000:]}\nstderr was:\n{result.stderr[-2000:]}")
    if isinstance(data, dict):               # {"results": [...]} style
        data = next(v for v in data.values() if isinstance(v, list))
    return [row["phonemes"] for row in data]


# ---------------------------------------------------------------------------
# metadata.csv / corpus
# ---------------------------------------------------------------------------

def load_metadata(path, delimiter="|"):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cols = line.split(delimiter)
            rows.append((cols[0], cols[1] if len(cols) > 1 else ""))
    return rows


def find_audio(audio_dir, name):
    p = Path(audio_dir) / name
    if p.exists():
        return p
    for ext in (".mp3", ".wav", ".flac"):
        q = Path(audio_dir) / (Path(name).stem + ext)
        if q.exists():
            return q
    raise FileNotFoundError(f"no audio for {name!r} in {audio_dir}")


def keep_existing_audio(rows, row_words, audio_dir):
    """Drop metadata rows whose audio file isn't in audio_dir (e.g. you uploaded only a few clips)."""
    keep = []
    for i, (name, _) in enumerate(rows):
        try:
            find_audio(audio_dir, name)
            keep.append(i)
        except FileNotFoundError:
            pass
    print(f"    audio found for {len(keep)} of {len(rows)} clips in {audio_dir}; using those")
    if not keep:
        raise SystemExit(f"None of the {len(rows)} clips in metadata.csv exist in {audio_dir}. "
                         "Check --audio-dir and the clip names.")
    return [rows[i] for i in keep], [row_words[i] for i in keep]


def write_corpus(rows, audio_dir, out_dir, speaker, sample_rate, strip_diacritics):
    from math import gcd

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    spk_dir = Path(out_dir) / speaker
    spk_dir.mkdir(parents=True, exist_ok=True)
    for n, (name, text) in enumerate(rows, start=1):
        audio, sr = sf.read(find_audio(audio_dir, name), dtype="float32", always_2d=True)
        wav = audio.mean(axis=1)
        if sr != sample_rate:
            g = gcd(sr, sample_rate)
            wav = resample_poly(wav, sample_rate // g, sr // g)
        peak = float(np.abs(wav).max())
        if peak > 0:
            wav = wav / peak * 0.95
        stem = Path(name).stem
        sf.write(spk_dir / f"{stem}.wav", wav, sample_rate, subtype="PCM_16")
        (spk_dir / f"{stem}.lab").write_text(
            " ".join(normalize_words(text, strip_diacritics)), encoding="utf-8")
        if n % 200 == 0:
            print(f"    {n}/{len(rows)} clips")


# ---------------------------------------------------------------------------

def select_clips(rows, row_words, missing, args):
    """Keep every clip by default. --skip-unknown-clips drops clips with ANY word that has no
    lexicon entry; --max-unknown-tokens N drops only clips with more than N such words."""
    limit = args.max_unknown_tokens
    if limit is None and args.skip_unknown_clips:
        limit = 0
    if limit is None:
        return rows
    kept = [r for r, ws in zip(rows, row_words) if sum(w in missing for w in ws) <= limit]
    print(f"    leaving out {len(rows) - len(kept)} clips with more than {limit} unknown word(s); "
          f"{len(kept)} remain")
    return kept


def report_missing(missing, row_words, top=12):
    """Which words without a lexicon entry cost the most clips."""
    clips = collections.Counter()
    tokens = collections.Counter()
    for ws in row_words:
        for w in ws:
            if w in missing:
                tokens[w] += 1
        for w in set(ws) & missing:
            clips[w] += 1
    if not clips:
        return
    print("    unknown words by number of clips they appear in:")
    for w, c in clips.most_common(top):
        print(f"      {w!r}: {c} clips, {tokens[w]} tokens")


def corpus_from_existing_lexicon(args, dataset, rows, row_words, counts):
    """Build the MFA corpus using a lexicon that already exists (no G2P needed)."""
    lex_path = Path(args.out)
    if not lex_path.exists():
        raise SystemExit(f"--use-existing-lexicon: {lex_path} not found")
    lex_lines = {}
    for ln in lex_path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            w, _, ph = ln.partition("\t")
            lex_lines[w] = ph
    if args.extra_lexicon:
        extra = read_extra_lexicon(args.extra_lexicon)
        lex_lines.update({w: " ".join(p) for w, p in extra.items()})
        with open(lex_path, "w", encoding="utf-8", newline="\n") as f:
            for w in sorted(lex_lines):
                f.write(f"{w}\t{lex_lines[w]}\n")
        print(f"    merged {len(extra)} hand-written entries into {lex_path}")
        phones_file = lex_path.with_suffix(".phones.txt")
        if phones_file.exists():
            have = {ln.split("\t")[0] for ln in phones_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
            new = sorted({p for ps in extra.values() for p in ps} - have)
            if new:
                print(f"    WARNING: phones {new} are not in {phones_file}: add them there and re-run "
                      "fs2_pashto_tools.py symbols, or they will be missing from FastSpeech2's symbol list")
    known = set(lex_lines)
    missing = {w for w in counts if w not in known}
    audio_dir = args.audio_dir or str(dataset / "clips")
    if args.write_corpus and args.skip_missing_audio:
        rows, row_words = keep_existing_audio(rows, row_words, audio_dir)
    n_bad = sum(1 for ws in row_words if missing & set(ws))
    print(f"Existing lexicon {lex_path}: {len(known)} entries; {len(missing)} words in the "
          f"transcripts are not in it (found in {n_bad} of {len(rows)} clips)")
    report_missing(missing, row_words)
    if not args.write_corpus:
        return
    corpus_rows = select_clips(rows, row_words, missing, args)
    print(f"Writing the MFA corpus to {args.write_corpus} ...")
    write_corpus(corpus_rows, audio_dir, args.write_corpus,
                 args.speaker, args.sample_rate, args.strip_diacritics)
    print("Done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="pashto_dataset",
                    help="folder containing metadata.csv (and clips/ if you use --write-corpus)")
    ap.add_argument("--out", default="lexicon/pashto-lexicon.txt")
    ap.add_argument("--g2p-repo", default=".", help="cloned nemo-pashto-g2p-conformer folder")
    ap.add_argument("--tmp-dir", default="./_g2p_tmp_lexicon")
    ap.add_argument("--delimiter", default="|")
    ap.add_argument("--strip-diacritics", action="store_true",
                    help="drop vowel marks (fatha etc.) so دَ and د are one word")
    ap.add_argument("--map-phones", default="",
                    help='merge/delete rare phones, e.g. "ħ=h,ð=z,tʰ=t,ʰ=" (empty = delete the phone)')
    ap.add_argument("--drop-words", default="",
                    help="comma-separated words to leave out of the lexicon (e.g. words the G2P "
                         "model gets badly wrong); they count as unknown words")
    ap.add_argument("--skip-unknown-clips", action="store_true",
                    help="with --write-corpus: leave out every clip that contains ANY word without a "
                         "lexicon entry. Strict; see --max-unknown-tokens for a gentler rule")
    ap.add_argument("--extra-lexicon", metavar="FILE",
                    help="hand-written pronunciations, one 'word<TAB>p1 p2 ...' per line; merged into "
                         "the lexicon (for fillers or words G2P can't handle)")
    ap.add_argument("--skip-missing-audio", action="store_true",
                    help="with --write-corpus: ignore metadata rows whose audio file isn't in the "
                         "clip folder, instead of stopping (handy when you upload only a few clips)")
    ap.add_argument("--max-unknown-tokens", type=int, default=None, metavar="N",
                    help="with --write-corpus: leave out only clips with MORE than N words that have "
                         "no lexicon entry. By default every clip is kept (the aligner marks such "
                         "words as spoken-noise 'spn')")
    ap.add_argument("--use-existing-lexicon", action="store_true",
                    help="don't run G2P: read the lexicon already at --out and only build the "
                         "corpus (needs no NeMo; use it in the FastSpeech2/MFA session)")
    ap.add_argument("--write-corpus", metavar="DIR",
                    help="also write the MFA corpus (wav + .lab) into DIR")
    ap.add_argument("--audio-dir", help="clip folder (default: <dataset>/clips)")
    ap.add_argument("--speaker", default="pashto", help="speaker sub-folder inside the corpus")
    ap.add_argument("--sample-rate", type=int, default=22050, help="corpus wav sample rate")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    rows = load_metadata(dataset / "metadata.csv", args.delimiter)

    # 1. unique words ------------------------------------------------------
    row_words = [normalize_words(text, args.strip_diacritics) for _, text in rows]
    counts = collections.Counter(w for ws in row_words for w in ws)
    print(f"{len(rows)} clips, {sum(counts.values())} word tokens, {len(counts)} unique words")
    if args.use_existing_lexicon:
        return corpus_from_existing_lexicon(args, dataset, rows, row_words, counts)

    # The Pashto G2P model can't pronounce Latin letters or digits (it returns
    # junk such as 414 -> "i r d i"), so those words are kept out of the lexicon.
    unknown = sorted((w for w in counts if has_latin_or_digit(w)), key=lambda w: -counts[w])
    words = sorted(w for w in counts if not has_latin_or_digit(w))
    drop = {unicodedata.normalize("NFC", w.strip()) for w in args.drop_words.split(",") if w.strip()}
    dropped = [w for w in words if w in drop]
    words = [w for w in words if w not in drop]
    if drop - set(dropped):
        print(f"    note: --drop-words not found in the data: {sorted(drop - set(dropped))}")
    if dropped:
        print(f"    dropped as requested: {dropped}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if unknown:
        hit = {w for w in unknown}
        n_clips = sum(1 for ws in row_words if hit & set(ws))
        unk_file = out.with_suffix(".unknown.txt")
        unk_file.write_text("".join(f"{w}\t{counts[w]}\n" for w in unknown), encoding="utf-8")
        print(f"    {len(unknown)} words with Latin letters/digits left out of the lexicon "
              f"({sum(counts[w] for w in unknown)} tokens in {n_clips} clips), listed in {unk_file}")
        print(f"    e.g. {unknown[:8]}")

    # 2. G2P, one word per line ------------------------------------------
    print("Running G2P on the unique words...")
    ipa = run_g2p(words, args.g2p_repo, args.tmp_dir)
    if len(ipa) != len(words):
        raise SystemExit(f"G2P returned {len(ipa)} rows for {len(words)} words (an old output "
                         f"file may have been picked up) - delete {args.tmp_dir} and retry")

    # 3. lexicon -----------------------------------------------------------
    pmap = parse_phone_map(args.map_phones)
    lexicon, phone_counts, empty = {}, collections.Counter(), []
    for word, s in zip(words, ipa):
        phones = []                       # spaces in the output just separate words; drop them
        for ph in split_phones(s):
            phones.extend(pmap[ph].split() if ph in pmap else [ph])
        if not phones:
            empty.append(word)
            continue
        lexicon[word] = phones
        phone_counts.update(phones * counts[word])  # weight by how often the word occurs

    extra = read_extra_lexicon(args.extra_lexicon) if args.extra_lexicon else {}
    for word, phones in extra.items():
        lexicon[word] = phones
        phone_counts.update(phones * counts.get(word, 0))
    empty = [w for w in empty if w not in extra]
    if extra:
        print(f"    merged {len(extra)} hand-written entries from {args.extra_lexicon}")

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(lexicon):
            f.write(f"{word}\t{' '.join(lexicon[word])}\n")
    phones_file = out.with_suffix(".phones.txt")
    phones_file.write_text(
        "".join(f"{p}\t{c}\n" for p, c in sorted(phone_counts.items(), key=lambda x: -x[1])),
        encoding="utf-8")

    print(f"Wrote {len(lexicon)} entries to {out}")
    print(f"{len(phone_counts)} distinct phones (also in {phones_file}):")
    print("    " + "  ".join(f"{p}:{c}" for p, c in phone_counts.most_common()))
    rare = [p for p, c in phone_counts.items() if c < 10]
    if rare:
        print(f"    rare (<10 uses), worth a look in case they're G2P noise: {rare}")
    if empty:
        print(f"WARNING: G2P gave no phones for {len(empty)} words, left out: {empty[:10]}")

    # 4. MFA corpus ----------------------------------------------------------
    if args.write_corpus:
        audio_dir = args.audio_dir or str(dataset / "clips")
        missing = (set(unknown) | set(empty) | set(dropped)) - set(extra)
        if args.skip_missing_audio:
            rows, row_words = keep_existing_audio(rows, row_words, audio_dir)
        report_missing(missing, row_words)
        corpus_rows = select_clips(rows, row_words, missing, args)
        print(f"Writing the MFA corpus to {args.write_corpus} ...")
        write_corpus(corpus_rows, audio_dir, args.write_corpus, args.speaker,
                     args.sample_rate, args.strip_diacritics)
        print("Done.")


if __name__ == "__main__":
    main()
