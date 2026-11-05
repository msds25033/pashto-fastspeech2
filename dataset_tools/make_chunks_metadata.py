"""
Cut a long Pashto recording into ~30 s chunks and write a two-column
metadata file (clip file name | Pashto transcript), using the phoneme TextGrid
produced by docx_to_alignment.py.

    (docx transcript, mp3 audio, TextGrid)  ->  clips/*.mp3 + metadata.csv

Produces (in --out-dir), e.g. for --audio 1_crop.mp3:
    clips/1_crop-001.mp3     mono clips, source sample rate by default
    metadata.csv             1_crop-001.mp3|<pashto text>   (no header)
    segments.csv             1_crop-001.mp3,<source audio>,<start>,<end>,<duration>  (for QA)
    g2p_cache/1_crop.json    cached G2P output, so G2P only runs once per file

How the transcript is mapped onto time
--------------------------------------
The TextGrid only has a "phonemes" tier: one interval per phoneme, with empty
intervals in between, and no word or sentence labels. To know which words fall
inside a chunk we rebuild the exact token sequence the aligner was given:

    docx -> docx_to_sentences() -> G2P (one phoneme string per sentence,
    words separated by spaces) -> segment each word into wav2vec2 vocab tokens

so every transcript word gets its own phoneme tokens. Lining that token
sequence up with the TextGrid's non-empty interval labels (normally identical;
a sequence diff absorbs small G2P differences) gives every word a start/end
time. The gap between one word's last phoneme and the next word's first
phoneme is a pause. If G2P returns a different number of words than the transcript for
some sentence, that sentence is kept whole as one unit instead (its words
can't be timed individually, but it still has start/end times).

How chunks are cut
------------------
Starting from the current chunk start, look at every word boundary where the
finished clip (sound edges + --edge-pad included) would be between --min-len
and --max-len seconds long, and cut at the longest one that stays within
--max-len (default: --target, 30 s -- clips never exceed it) and that both has
an aligned pause of at least --min-pause seconds and is actually quiet in the
audio (below --silence-db).
CTC alignments can be a few hundred ms off, so the cut goes at the quietest
point near the boundary, and each clip's edges are trimmed to --edge-pad of
silence around where the sound really starts and stops. If no such boundary
exists in the window, weaker options are tried (with a warning), preferring a
shorter chunk that ends in a real pause over a cut in the middle of speech.

Usage (on Colab, from the nemo-pashto-g2p-conformer folder that also holds
docx_to_alignment.py -- the same place preprocessing.ipynb ran it; one run per
source file; each run adds or replaces only its own <audio name>-NNN rows in
metadata.csv, so you can build the dataset up file by file):

    python make_chunks_metadata.py --docx 1.docx --audio 1_crop.mp3 \
        --textgrid 1.TextGrid --out-dir pashto_dataset

Needs the same environment as docx_to_alignment.py (torch, torchaudio,
transformers, python-docx, textgrid, and the G2P repo's requirements), plus
soundfile (>= 0.12, for mp3 read/write) and scipy. After the first run the
G2P output is cached, so the G2P repo is only needed when there is no cache yet.
"""

import argparse
import difflib
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import textgrid
from scipy.signal import resample_poly
from transformers import Wav2Vec2PhonemeCTCTokenizer

from docx_to_alignment import (
    ALIGN_MODEL_ID,
    build_vocab_symbols,
    docx_to_sentences,
    run_g2p,
    segment_into_vocab_tokens,
)


# --------------------------------------------------------------------------
# 1. Transcript -> phoneme count per word (G2P output cached on disk)
# --------------------------------------------------------------------------

def load_or_run_g2p(sentences, cache_path, g2p_repo, tmp_dir):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("sentences") == sentences:
            print(f"    using cached G2P output: {cache_path}")
            return cached["phonemes"]
        print(f"    cache {cache_path} was made from a different transcript, re-running G2P")

    if not (Path(g2p_repo) / "inference.py").exists():
        raise SystemExit(
            f"No usable G2P cache at {cache_path} and no inference.py in {g2p_repo!r} -- "
            "run this from the cloned nemo-pashto-g2p-conformer folder or pass --g2p-repo."
        )
    phonemes = run_g2p(sentences, g2p_repo, tmp_dir)
    if len(phonemes) != len(sentences):
        raise SystemExit(
            f"G2P returned {len(phonemes)} rows for {len(sentences)} sentences; "
            "cannot map words onto the TextGrid."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"sentences": sentences, "phonemes": phonemes}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return phonemes


def is_spoken(word):
    # Tokens with no letters/digits (e.g. a lone "-" or "،") produce no
    # phonemes; they're attached to a neighbouring word instead.
    return any(ch.isalnum() for ch in word)


def group_text_words(sentence):
    groups = []
    pending_prefix = []
    for w in sentence.split():
        if is_spoken(w):
            groups.append(" ".join(pending_prefix + [w]))
            pending_prefix = []
        elif groups:
            groups[-1] += " " + w
        else:
            pending_prefix.append(w)
    if pending_prefix:  # sentence with no spoken word at all
        groups.append(" ".join(pending_prefix))
    return groups


def build_units(sentences, phoneme_strings, symbols):
    """Return [(text, phoneme_tokens)] in transcript order, plus how many
    sentences had to be kept whole because word counts didn't line up."""
    units, fallback = [], 0
    for sentence, phon in zip(sentences, phoneme_strings):
        word_tokens = [segment_into_vocab_tokens(w, symbols)[0] for w in phon.split()]
        word_tokens = [t for t in word_tokens if t]
        text_words = group_text_words(sentence)
        if len(word_tokens) == len(text_words):
            units.extend(zip(text_words, word_tokens))
        else:
            fallback += 1
            units.append((" ".join(sentence.split()), [t for w in word_tokens for t in w]))
    return units, fallback


# --------------------------------------------------------------------------
# 2. Units -> start/end times from the TextGrid phoneme intervals
# --------------------------------------------------------------------------

def load_phone_intervals(textgrid_path, tier_name="phonemes"):
    tg = textgrid.TextGrid.fromFile(str(textgrid_path))
    tier = next(t for t in tg.tiers if t.name == tier_name)
    return [(iv.minTime, iv.maxTime, iv.mark.strip()) for iv in tier if iv.mark.strip()]


def map_tokens_to_intervals(tokens, phone_intervals, min_match):
    """For each re-derived token, the index of the TextGrid interval it was
    aligned to (or None). Normally the two sequences are identical; if G2P
    output drifted slightly since the TextGrid was made (different NeMo
    version, CPU vs GPU, ...) a sequence diff still lines them up."""
    labels = [mark for _, _, mark in phone_intervals]
    if tokens == labels:
        return list(range(len(tokens)))

    sm = difflib.SequenceMatcher(None, tokens, labels, autojunk=False)
    mapping = [None] * len(tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    matched = sum(m is not None for m in mapping)
    print(f"    re-derived phonemes differ slightly from the TextGrid "
          f"({len(tokens)} vs {len(labels)} tokens); aligned {matched}/{len(tokens)}")
    if matched < min_match * len(tokens):
        raise SystemExit(
            f"Only {matched}/{len(tokens)} phoneme tokens match the TextGrid. Make sure "
            "this TextGrid was generated from this exact docx (and the same G2P model)."
        )
    return mapping


def time_units(units, phone_intervals, min_match=0.95):
    tokens = [t for _, toks in units for t in toks]
    mapping = map_tokens_to_intervals(tokens, phone_intervals, min_match)

    timed, i = [], 0
    for text, toks in units:
        idx = [m for m in mapping[i:i + len(toks)] if m is not None]
        i += len(toks)
        if not idx:
            # No timed phonemes (e.g. only symbols G2P/aligner dropped): glue
            # the text to the previous unit so it isn't lost.
            if timed:
                timed[-1]["text"] += " " + text
            else:
                timed.append({"text": text, "start": None, "end": None})
            continue
        start, end = phone_intervals[idx[0]][0], phone_intervals[idx[-1]][1]
        if timed and timed[-1]["start"] is None:  # leading text-only unit
            text = timed.pop()["text"] + " " + text
        timed.append({"text": text, "start": start, "end": end})
    return timed


# --------------------------------------------------------------------------
# 3. Pick cut points at pauses closest to the target length
# --------------------------------------------------------------------------

class Loudness:
    """Frame-level RMS of the source audio, relative to the file's overall RMS.

    Used to double-check the aligner: a word boundary only counts as a pause
    if the audio around it is actually quiet, and clip edges are trimmed to
    where the sound really starts/stops (CTC alignments are 'peaky', so
    phoneme times can be a few hundred ms off)."""

    def __init__(self, mono, sr, hop=0.01, smooth=0.15):
        self.hop = hop
        n = int(hop * sr)
        n_frames = len(mono) // n
        frames = mono[: n_frames * n].reshape(n_frames, n)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        rms /= max(float(np.sqrt((mono ** 2).mean())), 1e-9)
        self.rms = rms
        k = max(1, int(smooth / hop))
        self.smooth = np.convolve(rms, np.ones(k) / k, mode="same")

    def _range(self, t0, t1):
        a = max(0, int(t0 / self.hop))
        b = min(len(self.rms), max(a + 1, int(t1 / self.hop)))
        return a, b

    def quietest(self, t0, t1):
        """(time, level) of the quietest smoothed point in [t0, t1]."""
        a, b = self._range(t0, t1)
        i = a + int(np.argmin(self.smooth[a:b]))
        return (i + 0.5) * self.hop, float(self.smooth[i])

    def first_loud(self, t0, t1, threshold):
        a, b = self._range(t0, t1)
        loud = np.nonzero(self.rms[a:b] > threshold)[0]
        return (a + loud[0]) * self.hop if len(loud) else None

    def last_loud(self, t0, t1, threshold):
        a, b = self._range(t0, t1)
        loud = np.nonzero(self.rms[a:b] > threshold)[0]
        return (a + loud[-1] + 1) * self.hop if len(loud) else None


def boundary_info(units, k, loudness, margin=0.3):
    """Cut time and loudness at the boundary between units k and k+1. The
    search reaches `margin` s into the neighbouring words to absorb alignment
    timing errors."""
    t0 = max(units[k]["end"] - margin, units[k]["start"])
    t1 = min(units[k + 1]["start"] + margin, units[k + 1]["end"])
    return loudness.quietest(t0, t1)


def clip_span(units, first, last, lo, hi, loudness, silence, edge_pad):
    """(start_sec, end_sec) of the clip holding units first..last, between the
    previous cut `lo` and this cut `hi`: from where the sound really starts to
    where it really stops, plus `edge_pad` of silence either side.

    Used both to *choose* cuts and to write the clips, so the length that is
    checked against --max-len is exactly the length of the finished clip."""
    onset = loudness.first_loud(lo, units[first]["start"] + 0.3, silence)
    offset = loudness.last_loud(units[last]["end"] - 0.3, hi, silence)
    onset = units[first]["start"] if onset is None else min(onset, units[first]["start"])
    offset = units[last]["end"] if offset is None else max(offset, units[last]["end"])
    return max(lo, onset - edge_pad), min(hi, offset + edge_pad)


def choose_chunks(units, loudness, target, min_len, max_len, min_pause, silence,
                  edge_pad, total_duration):
    """Return [(first_unit, last_unit, cut_time_after)] (last cut is None).

    max_len (<= target) is a hard limit on the finished clip, not just on word
    times: every candidate is measured with clip_span(), edge padding included."""
    chunks, i, n = [], 0, len(units)
    lo = 0.0  # previous cut
    while i < n:
        chunk_start = units[i]["start"]
        if i == n - 1:
            chunks.append((i, n - 1, None))
            break
        s_, e_ = clip_span(units, i, n - 1, lo, total_duration, loudness, silence, edge_pad)
        if e_ - s_ <= max_len:  # everything left fits in one clip
            chunks.append((i, n - 1, None))
            break

        # Candidate boundaries: cut after unit k, i.e. between k and k+1.
        candidates = []
        for k in range(i, n - 1):
            if units[k]["end"] - chunk_start > max_len + 2.0:  # cheap early exit
                break
            cut, level = boundary_info(units, k, loudness)
            s_, e_ = clip_span(units, i, k, lo, cut, loudness, silence, edge_pad)
            length = e_ - s_
            if length > max_len:
                break
            if length < 1.0:
                continue
            pause = units[k + 1]["start"] - units[k]["end"]
            candidates.append({"k": k, "len": length, "pause": pause, "cut": cut, "level": level,
                               "quiet": level < silence, "window": length >= min_len})

        def closest(cs):
            # Every candidate is <= max_len <= target, so (target - len) is never
            # negative: no abs() needed, and the longest chunk that still fits wins
            # (ties -> the longer pause).
            return min(cs, key=lambda c: (target - c["len"], -c["pause"]))

        # Preference order; the first non-empty tier wins.
        tiers = [
            ("", lambda c: c["window"] and c["quiet"] and c["pause"] >= min_pause, closest),
            ("quiet boundary with a short aligned gap",
             lambda c: c["window"] and c["quiet"], closest),
            # Better a shorter chunk ending in a real pause than a cut mid-speech.
            ("cutting early at a quiet pause",
             lambda c: c["quiet"] and c["pause"] >= min_pause,
             lambda cs: max(cs, key=lambda c: c["len"])),
            ("aligned pause that isn't quiet in the audio",
             lambda c: c["window"] and c["pause"] >= min_pause, closest),
            ("cutting at the quietest boundary", lambda c: c["window"],
             lambda cs: min(cs, key=lambda c: c["level"])),
            # Last resort that still respects the limit, e.g. when the next unit
            # is too long to fit: end right before it with the longest chunk
            # that does fit.
            ("cutting early, no good boundary left", lambda c: True,
             lambda cs: max(cs, key=lambda c: c["len"])),
        ]
        chosen = None
        for note, keep, pick in tiers:
            pool = [c for c in candidates if keep(c)]
            if pool:
                chosen = pick(pool)
                if note:
                    print(f"    WARNING: no quiet pause in the {min_len:g}-{max_len:g}s window "
                          f"after {chunk_start:.1f}s; {note} ({chosen['len']:.1f}s chunk)",
                          file=sys.stderr)
                break

        if chosen is None:
            # Nothing fits: a single unit (e.g. a sentence kept whole because
            # G2P and the transcript disagreed on word count) is already longer
            # than max_len and can't be split. Cut right after it, which keeps
            # the overshoot as small as possible.
            k = next((k for k in range(i, n - 1)
                      if units[k]["end"] - chunk_start >= 1.0), n - 2)
            cut = boundary_info(units, k, loudness)[0]
            s_, e_ = clip_span(units, i, k, lo, cut, loudness, silence, edge_pad)
            chosen = {"k": k, "cut": cut}
            print(f"    WARNING: no word boundary fits inside {max_len:g}s after "
                  f"{chunk_start:.1f}s (one unit is too long to split); "
                  f"this chunk will be {e_ - s_:.1f}s, over the limit", file=sys.stderr)

        chunks.append((i, chosen["k"], chosen["cut"]))
        lo = chosen["cut"]
        i = chosen["k"] + 1
    return chunks


def chunk_spans(units, chunks, loudness, silence, edge_pad, total_duration):
    """(start_sec, end_sec) per chunk: from cut to cut, with silence at the
    edges trimmed to `edge_pad` around where the sound actually is."""
    spans = []
    for j, (first, last, cut) in enumerate(chunks):
        lo = chunks[j - 1][2] if j > 0 else 0.0
        hi = cut if cut is not None else total_duration
        spans.append(clip_span(units, first, last, lo, hi, loudness, silence, edge_pad))
    return spans


# --------------------------------------------------------------------------
# 4. Write clips + metadata.csv
# --------------------------------------------------------------------------

def clean_field(text):
    return " ".join(text.replace("|", " ").split())


def is_own_clip(name, prefix):
    stem, _, num = Path(name).stem.rpartition("-")
    return stem == prefix and num.isdigit()


def merge_rows(path, prefix, new_rows, delimiter):
    """Keep existing rows from other source files, replace this file's rows.
    Fields never contain the delimiter, so lines are written as-is (no CSV
    quoting/escaping that would alter the transcript text)."""
    path = Path(path)
    lines = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not is_own_clip(line.split(delimiter, 1)[0], prefix):
                lines.append(line)
    lines.extend(delimiter.join(row) for row in new_rows)
    lines.sort(key=lambda line: line.split(delimiter, 1)[0])
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docx", required=True, help="Word transcript used to make the TextGrid")
    ap.add_argument("--audio", required=True, help="source audio (mp3/wav/flac)")
    ap.add_argument("--textgrid", required=True, help="full-file TextGrid from docx_to_alignment.py")
    ap.add_argument("--out-dir", default="pashto_dataset")
    ap.add_argument("--id-prefix", default=None,
                    help="clip name prefix, unique per source file (default: audio file "
                         "name without extension, giving e.g. 1_crop-001.mp3)")
    ap.add_argument("--g2p-repo", default=".",
                    help="cloned nemo-pashto-g2p-conformer folder (only used when there is no G2P cache)")
    ap.add_argument("--g2p-cache", default=None,
                    help="G2P cache file (default: <out-dir>/g2p_cache/<prefix>.json)")
    ap.add_argument("--tmp-dir", default="./_g2p_tmp")
    ap.add_argument("--target", type=float, default=30.0,
                    help="preferred chunk length (s); chunks get as close to it as possible "
                         "without going over --max-len")
    ap.add_argument("--min-len", type=float, default=25.0,
                    help="shortest preferred chunk (s); shorter only if there's no quiet pause")
    ap.add_argument("--max-len", type=float, default=None,
                    help="hard limit on clip length (s), measured on the finished clip; "
                         "defaults to --target and can't be larger than it")
    ap.add_argument("--min-pause", type=float, default=0.3,
                    help="silence between words (s) that counts as a pause")
    ap.add_argument("--silence-db", type=float, default=-20.0,
                    help="audio this many dB below the file's average level counts as silence")
    ap.add_argument("--edge-pad", type=float, default=0.15,
                    help="silence kept before the first / after the last sound of a clip (s)")
    ap.add_argument("--format", choices=["mp3", "wav", "flac"], default="mp3",
                    help="clip file format")
    ap.add_argument("--sample-rate", type=int, default=0,
                    help="resample clips to this rate; 0 keeps the source rate")
    ap.add_argument("--delimiter", default="|",
                    help="metadata.csv column separator ('|' avoids clashes with commas in text)")
    args = ap.parse_args()

    if args.max_len is None:
        args.max_len = args.target
    if not args.min_len <= args.max_len <= args.target:
        ap.error("need --min-len <= --max-len <= --target (--max-len defaults to --target)")
    prefix = args.id_prefix or Path(args.audio).stem
    if args.delimiter in prefix:
        ap.error(f"clip prefix {prefix!r} must not contain the delimiter {args.delimiter!r}")

    out_dir = Path(args.out_dir)
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.g2p_cache or out_dir / "g2p_cache" / f"{prefix}.json"

    print("1/5 Extracting sentences from the Word doc...")
    sentences = docx_to_sentences(args.docx)
    print(f"    {len(sentences)} sentences")

    print("2/5 Getting phonemes per sentence (G2P) and counting tokens per word...")
    phoneme_strings = load_or_run_g2p(sentences, cache_path, args.g2p_repo, args.tmp_dir)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(ALIGN_MODEL_ID, do_phonemize=False)
    symbols = build_vocab_symbols(tokenizer)
    units, fallback = build_units(sentences, phoneme_strings, symbols)
    print(f"    {len(units)} timing units")
    if fallback:
        print(f"    {fallback}/{len(sentences)} sentences had a G2P/transcript word-count "
              "mismatch and are kept whole (can only be cut at their edges)")

    print("3/5 Timing every word from the TextGrid...")
    units = time_units(units, load_phone_intervals(args.textgrid))

    print("4/5 Loading audio and choosing cut points...")
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    del audio
    total_duration = len(mono) / sr
    loudness = Loudness(mono, sr)
    silence = 10 ** (args.silence_db / 20)
    chunks = choose_chunks(units, loudness, args.target, args.min_len, args.max_len,
                           args.min_pause, silence, args.edge_pad, total_duration)
    spans = chunk_spans(units, chunks, loudness, silence, args.edge_pad, total_duration)

    print(f"5/5 Writing {len(chunks)} clips...")
    for old in clip_dir.iterdir():  # clips from a previous run of this file
        if is_own_clip(old.name, prefix):
            old.unlink()
    out_sr = args.sample_rate or sr
    g = gcd(out_sr, sr)
    width = max(3, len(str(len(chunks))))
    meta_rows, seg_rows = [], []
    for idx, ((first, last, _), (start, end)) in enumerate(zip(chunks, spans), start=1):
        clip_name = f"{prefix}-{idx:0{width}d}.{args.format}"
        clip = mono[int(start * sr):int(end * sr)]
        if out_sr != sr:
            clip = resample_poly(clip, out_sr // g, sr // g).astype(np.float32)
        sf.write(clip_dir / clip_name, np.clip(clip, -1.0, 1.0), out_sr,
                 **({"subtype": "PCM_16"} if args.format == "wav" else {}))

        text = clean_field(" ".join(u["text"] for u in units[first:last + 1]))
        meta_rows.append([clip_name, text.replace(args.delimiter, " ")])
        seg_rows.append([clip_name, Path(args.audio).name, f"{start:.3f}", f"{end:.3f}",
                         f"{end - start:.3f}"])

    merge_rows(out_dir / "metadata.csv", prefix, meta_rows, args.delimiter)
    merge_rows(out_dir / "segments.csv", prefix, seg_rows, ",")

    lengths = [float(r[4]) for r in seg_rows]
    too_long = [r[0] for r in seg_rows if float(r[4]) > args.max_len + 1e-3]
    if too_long:
        print(f"WARNING: {len(too_long)} clip(s) are longer than {args.max_len:g}s because a "
              f"single unit can't be split: {', '.join(too_long)}", file=sys.stderr)
    print(f"Done: {len(meta_rows)} clips in {clip_dir}, "
          f"length min/mean/max = {min(lengths):.1f}/{np.mean(lengths):.1f}/{max(lengths):.1f}s, "
          f"{sum(lengths) / 60:.1f} min of {total_duration / 60:.1f} min audio kept")


if __name__ == "__main__":
    main()
