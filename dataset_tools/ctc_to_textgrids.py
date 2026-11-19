"""
Turn the wav2vec2 CTC alignments (one 20 ms "spike" per phoneme, written per clip
by make_chunks_metadata.py) into FastSpeech2-ready TextGrids with real durations.

    textgrids/<clip>.TextGrid (tier "phonemes", spikes + empty gaps)
  + clips/<clip>.mp3 + metadata.csv + the lexicon
  ->  <out>/<clip>.TextGrid with tiers "words" and "phones" (MFA layout)

Why: FastSpeech2 learns each phone's duration from the "phones" tier, and the
phones must be exactly the lexicon phones that synthesis uses (tts.text_to_phones).
The CTC TextGrids have neither: every phone lasts 20 ms and the labels are the
aligner's vocabulary tokens ("ja", "əl", "ts", ...), not the lexicon's phones.

How
  1. Each clip's transcript -> words (normalize_words, as the corpus stage does)
     -> lexicon phones. Words missing from the lexicon become one "spn".
  2. The lexicon phones and the CTC tokens are both spelled out as plain IPA
     characters and lined up with a sequence diff; every lexicon phone whose
     characters match a CTC token takes that token's time. Phones without a match
     (G2P differences between the lexicon and the aligner run) are interpolated.
  3. A phone lasts from its time to the next phone's time, EXCEPT where the audio
     between them has a quiet stretch (RMS below the clip's 95th percentile minus
     --quiet-db for at least --min-pause s): that stretch becomes "sp", the phone
     before it ends where the quiet starts and the next phone starts where it
     ends (CTC spikes sit a little after a phone's onset). Leading and trailing
     audio is "sp" too, up to where the sound starts/stops.

Usage (from the repo root):
    python dataset_tools/ctc_to_textgrids.py --ctc-dir ../pashto_dataset/textgrids \
        --out outputs/ctc_textgrids [--clips 1_crop-00*] [--zip outputs/pashto_aligned_ctc.zip]
"""

import argparse
import collections
import difflib
import fnmatch
import sys
import unicodedata
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_lexicon import load_metadata, normalize_words, read_extra_lexicon  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SR, HOP = 16000, 160          # energy analysis: 10 ms frames
MIN_PHONE = 0.012             # >= one FastSpeech2 mel frame (256 / 22050 s)
SPIKE = 0.02                  # length of one wav2vec2 frame

# Spelling differences between the aligner vocabulary and the lexicon phones that
# would otherwise stop characters from matching. Marks are dropped, a few symbols
# are folded onto the lexicon's (the same direction as build_dataset.MAP_PHONES).
DROP_CHARS = "͜͡ːˑʰʲʷ"
FOLD = {"g": "ɡ", "y": "j", "ħ": "h", "ɪ": "i", "ʊ": "u", "ʔ": "", "ð": "z",
        "θ": "s", "v": "w", "ɔ": "o", "ɛ": "e"}


def spell(symbol):
    s = unicodedata.normalize("NFC", symbol)
    return "".join(FOLD.get(c, c) for c in s if c not in DROP_CHARS)


def read_lexicon(lex_dir):
    lex = {}
    for line in (lex_dir / "pashto-lexicon.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            word, _, phones = line.partition("\t")
            lex[word] = phones.split()
    overrides = lex_dir / "pron_overrides.txt"
    if overrides.exists():
        lex.update(read_extra_lexicon(overrides))
    return lex


def read_ctc(path):
    import tgt
    tg = tgt.io.read_textgrid(str(path))          # drops the empty gaps
    return [(i.text, i.start_time, i.end_time) for i in tg.tiers[0].intervals if i.text.strip()]


def quiet_frames(audio_path, quiet_db):
    import librosa
    y, _ = librosa.load(str(audio_path), sr=SR)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0] + 1e-6)
    return db < np.percentile(db, 95) - quiet_db, len(y) / SR, y


def voiced(y, a, b):
    """Voicing flags (10 ms frames) for y[a:b] seconds."""
    import librosa
    seg = y[int(a * SR):int(b * SR)]
    if len(seg) < 1024:
        return np.zeros(0, bool)
    _, flags, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP, center=False)
    return flags


def fix_long_phones(iv, y, args):
    """A phone longer than --check-phone usually swallowed a pause the aligner left
    between it and the next phone. If that pause is unvoiced (breath, noise), it
    becomes "sp". If it is voiced speech, the transcript is missing words there:
    returns the (start, end) of such spans as bad."""
    out, bad = [], []
    for s, e, p, wi in iv:
        if p in ("sp", "spn") or e - s <= args.check_phone:
            out.append((s, e, p, wi))
            continue
        a = s + 0.08
        v = voiced(y, a, e)
        u = None
        for i in range(len(v)):
            if not v[i:i + 12].any() and len(v[i:i + 12]) == 12:    # >= 120 ms unvoiced
                u = a + i * HOP / SR
                break
        if u is not None:
            out.append((s, u, p, wi))
            out.append((u, e, "sp", None))
            rest = v[int((u - a) * SR / HOP):]
            if len(rest) and rest.mean() > 0.4:
                bad.append((u, e))
        else:
            out.append((s, e, p, wi))
        if (u or e) - s > args.max_phone:
            bad.append((s, e))
    return out, bad


def runs(mask, a, b, min_len):
    """Quiet runs of >= min_len seconds inside [a, b] -> list of (start, end) seconds."""
    fa, fb = max(0, int(np.ceil(a * 100))), min(len(mask), int(b * 100))
    out, i = [], fa
    while i < fb:
        if mask[i]:
            j = i
            while j < fb and mask[j]:
                j += 1
            if (j - i) / 100 >= min_len:
                out.append((i / 100, j / 100))
            i = j
        else:
            i += 1
    return out


def convert(text, ctc, quiet, duration, lex, args):
    # 1. lexicon phones, with word index
    units = []                                   # (label, word_idx)
    words = normalize_words(text)
    for wi, w in enumerate(words):
        pron = lex.get(w) or lex.get(w.lower())
        if pron:
            units += [(p, wi) for p in pron]
        else:
            units.append(("spn", wi))
    if not units or not ctc:
        return None, 0.0

    # 2. character-level match against the CTC tokens
    a_chars, a_time = [], []
    for tok, s, e in ctc:
        sp = spell(tok)
        for k, c in enumerate(sp):
            a_chars.append(c)
            a_time.append(s + k * (e - s) / len(sp))
    b_chars, b_owner = [], []
    for ui, (p, _) in enumerate(units):
        if p != "spn":
            for c in spell(p):
                b_chars.append(c)
                b_owner.append(ui)
    hits = collections.defaultdict(list)
    sm = difflib.SequenceMatcher(None, a_chars, b_chars, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            hits[b_owner[blk.b + k]].append(a_time[blk.a + k])
    # A phone spelled with several characters (affricates: t͡ʃ) may have matched
    # tokens that are far apart ("t" ... "ʃ"); the later one is the real phone.
    t = [None] * len(units)
    for ui, ts in hits.items():
        t[ui] = ts[-1] if ts[-1] - ts[0] > 0.1 else ts[0]
    anchored = sum(x is not None for x in t) / len(t)

    # interpolate / extrapolate the unmatched phones
    idx = [i for i, x in enumerate(t) if x is not None]
    if not idx:
        return None, 0.0
    t = np.array([np.nan if x is None else x for x in t], float)
    known = np.array(idx)
    t = np.interp(np.arange(len(t)), known, t[known])
    first, last = idx[0], idx[-1]
    for i in range(first - 1, -1, -1):
        t[i] = t[i + 1] - 0.06
    for i in range(last + 1, len(t)):
        t[i] = t[i - 1] + 0.06
    # strictly increasing, inside the clip
    t[0] = max(t[0], 0.0)
    for i in range(1, len(t)):
        t[i] = max(t[i], t[i - 1] + MIN_PHONE)
    top = duration - MIN_PHONE
    for i in range(len(t) - 1, -1, -1):
        top = min(t[i], top)
        t[i] = top
        top -= MIN_PHONE
    if t[0] < 0:
        return None, anchored

    # 3. intervals: phones + "sp" at quiet stretches
    iv = []                                      # (start, end, label, word_idx or None)
    lead = runs(quiet, 0, t[0], 0.0)
    onset = lead[-1][1] if lead and lead[-1][1] >= t[0] - args.max_shift else t[0]
    onset = min(max(onset, t[0] - args.max_shift), t[0])
    start = onset if onset > 0.005 else 0.0
    if start > 0:
        iv.append((0.0, start, "sp", None))
    for i, (p, wi) in enumerate(units):
        if i + 1 < len(units):
            nxt = t[i + 1]
            same_word = units[i + 1][1] == wi
            q = runs(quiet, t[i] + SPIKE, nxt, args.min_pause * (1.5 if same_word else 1.0))
            if q:
                q0, q1 = q[0][0], q[-1][1]
                q1 = max(q1, nxt - args.max_shift)
                end = max(q0, start + MIN_PHONE)
                iv.append((start, end, p, wi))
                if q1 - end >= MIN_PHONE:
                    iv.append((end, q1, "sp", None))
                    start = q1
                else:
                    start = end
            else:
                end = max(nxt, start + MIN_PHONE)
                iv.append((start, end, p, wi))
                start = end
        else:
            tail = runs(quiet, t[i] + SPIKE, duration, args.min_pause)
            end = tail[0][0] if tail else min(duration, t[i] + args.max_last)
            end = min(max(end, start + MIN_PHONE), duration)
            iv.append((start, end, p, wi))
            if duration - end > 0.005:
                iv.append((end, duration, "sp", None))
    return (iv, words), anchored


def keep_longest_good(iv, bad, duration, min_keep):
    """Keep the longest stretch (cut at pauses) that overlaps no bad span; label the
    rest "sp". FastSpeech2's preprocess trims leading/trailing sp from the audio, so
    the bad parts are simply left out - no new audio files needed."""
    segs, cur = [], []                     # segment = phones up to and including a pause
    for x in iv:
        cur.append(x)
        if x[2] == "sp" and x[1] - x[0] >= 0.15:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    ok = [not any(a < seg[-1][1] and b > seg[0][0] for a, b in bad)
          and any(x[2] not in ("sp", "spn") for x in seg) for seg in segs]
    best, run = (0.0, None), None
    for i, good in enumerate(ok + [False]):
        if good and run is None:
            run = i
        elif not good and run is not None:
            s, e = segs[run][0][0], segs[i - 1][-1][1]
            if e - s > best[0]:
                best = (e - s, (run, i))
            run = None
    if best[1] is None or best[0] < min_keep:
        return None
    a, b = best[1]
    kept = [x for seg in segs[a:b] for x in seg]
    s, e = kept[0][0], kept[-1][1]
    return ([(0.0, s, "sp", None)] if s > 0 else []) + kept + ([(e, duration, "sp", None)] if e < duration else [])


def write_textgrid(iv, words, duration, path):
    import tgt
    tg = tgt.TextGrid()
    phones = tgt.IntervalTier(0, duration, "phones")
    wt = tgt.IntervalTier(0, duration, "words")
    spans = collections.OrderedDict()
    for s, e, p, wi in iv:
        if e - s <= 1e-6:
            continue
        phones.add_interval(tgt.Interval(s, e, p))
        if wi is not None:
            a, b = spans.get(wi, (s, e))
            spans[wi] = (min(a, s), max(b, e))
    for wi, (s, e) in spans.items():
        wt.add_interval(tgt.Interval(s, e, words[wi]))
    tg.add_tier(wt)
    tg.add_tier(phones)
    tgt.io.write_to_file(tg, str(path), format="long")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctc-dir", required=True, help="per-clip CTC TextGrids (make_chunks_metadata.py)")
    ap.add_argument("--data-dir", default=REPO / "data", type=Path, help="clips/, metadata.csv, lexicon/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clips", nargs="*", help="only these clip names / glob patterns")
    ap.add_argument("--quiet-db", type=float, default=24, help="quiet = this far below the clip's loud level")
    ap.add_argument("--min-pause", type=float, default=0.08, help="shortest quiet stretch that becomes sp (s)")
    ap.add_argument("--max-shift", type=float, default=0.3,
                    help="a phone may start at most this much before its CTC time (s)")
    ap.add_argument("--max-last", type=float, default=0.3, help="last phone length when no quiet follows")
    ap.add_argument("--min-anchored", type=float, default=0.6,
                    help="skip clips where fewer lexicon phones than this matched a CTC token")
    ap.add_argument("--check-phone", type=float, default=0.4,
                    help="phones longer than this (s) are checked for a swallowed pause")
    ap.add_argument("--max-phone", type=float, default=0.6,
                    help="a phone still longer than this (s) marks a transcript/audio mismatch")
    ap.add_argument("--min-keep", type=float, default=3.0,
                    help="clips with a bad span keep their longest clean stretch if it is this long (s)")
    ap.add_argument("--zip", type=Path, help="also write a zip laid out like pashto_aligned.zip")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    lex = read_lexicon(args.data_dir / "lexicon")
    rows = load_metadata(args.data_dir / "metadata.csv")
    if args.clips:
        rows = [r for r in rows if any(fnmatch.fnmatch(Path(r[0]).stem, p) for p in args.clips)]
    args.out.mkdir(parents=True, exist_ok=True)
    ctc_dir = Path(args.ctc_dir)
    stats = collections.Counter()
    anchored_all, low, labels, bad_spans = [], [], collections.Counter(), {}
    kept_time = [0.0, 0.0]                       # speech kept, speech in total (s)
    for n, (name, text) in enumerate(rows, 1):
        stem = Path(name).stem
        ctc_path = ctc_dir / f"{stem}.TextGrid"
        if not ctc_path.exists():
            stats["no CTC TextGrid"] += 1
            continue
        quiet, duration, y = quiet_frames(args.data_dir / "clips" / name, args.quiet_db)
        res, anchored = convert(text, read_ctc(ctc_path), quiet, duration, lex, args)
        anchored_all.append(anchored)
        if res is None or anchored < args.min_anchored:
            low.append((stem, anchored))
            stats["skipped (poor match)"] += 1
            (args.out / f"{stem}.TextGrid").unlink(missing_ok=True)
            continue
        iv, words = res
        iv, bad = fix_long_phones(iv, y, args)
        speech = sum(e - s for s, e, p, _ in iv if p != "sp")
        if bad:
            # speech the transcript does not have, sung/recited passages, CTC failures
            bad_spans[stem] = bad
            iv = keep_longest_good(iv, bad, duration, args.min_keep)
            if iv is None:
                stats["skipped (no clean stretch)"] += 1
                (args.out / f"{stem}.TextGrid").unlink(missing_ok=True)
                kept_time[1] += speech
                continue
            stats["trimmed to the longest clean stretch"] += 1
        kept = sum(e - s for s, e, p, _ in iv if p != "sp")
        kept_time[0] += kept
        kept_time[1] += speech
        labels.update(p for _, _, p, _ in iv)
        write_textgrid(iv, words, duration, args.out / f"{stem}.TextGrid")
        stats["written"] += 1
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    print(dict(stats))
    if kept_time[1]:
        print(f"speech kept: {kept_time[0] / 3600:.2f} h of {kept_time[1] / 3600:.2f} h "
              f"({kept_time[0] / kept_time[1] * 100:.0f}%)")
    if anchored_all:
        a = np.array(anchored_all)
        print(f"lexicon phones matched to a CTC token: median {np.median(a)*100:.0f}%, "
              f"10th pct {np.percentile(a, 10)*100:.0f}%")
    if low:
        print("skipped (match %):", ", ".join(f"{s} ({x*100:.0f}%)" for s, x in low[:20]))
    if bad_spans:
        print("bad spans:", "; ".join(f"{s} " + ",".join(f"{a:.1f}-{b:.1f}" for a, b in v)
                                      for s, v in list(bad_spans.items())[:30]))
    phones_file = args.data_dir / "lexicon" / "pashto-lexicon.phones.txt"
    allowed = {ln.split("\t")[0] for ln in phones_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    bad = {p: c for p, c in labels.items() if p not in allowed | {"sp", "spn", "sil"}}
    print("labels outside the phone set:", bad or "none")
    print(f"spn (words missing from the lexicon): {labels['spn']}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(args.out.glob("*.TextGrid")):
                z.write(f, f"preprocessed_data/Pashto/TextGrid/pashto/{f.name}")
        print(f"zip: {args.zip}")


if __name__ == "__main__":
    main()
