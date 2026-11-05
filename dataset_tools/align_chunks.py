"""
Generate one phoneme TextGrid per audio clip listed in metadata.csv, using
the same G2P model and forced aligner as docx_to_alignment.py:

    chunk transcript --sentence split--> pieces --nemo-pashto-g2p--> IPA
    IPA --vocab match--> wav2vec2 phoneme tokens
    clip audio + tokens --forced_align--> <clip>.TextGrid  (tier "phonemes")

Reads the dataset written by make_chunks_metadata.py:
    <dataset>/metadata.csv        <clip file>|<pashto text>
    <dataset>/clips/<clip file>

Writes:
    <dataset>/textgrids/<clip name without extension>.TextGrid
    <dataset>/g2p_cache/chunks.json   G2P output per text piece, so edited
                                      transcripts only re-run G2P for the
                                      pieces that changed

Usage (on Colab, from the nemo-pashto-g2p-conformer folder that also holds
docx_to_alignment.py -- same place preprocessing.ipynb ran it):

    python align_chunks.py --dataset pashto_dataset

Existing TextGrids are skipped (so an interrupted run can be resumed);
pass --overwrite to redo them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

from docx_to_alignment import (
    ALIGN_MODEL_ID,
    TARGET_SR,
    build_vocab_symbols,
    forced_align,
    run_g2p,
    sentences_to_phoneme_tokens,
    split_into_sentences,
    split_long_sentence,
    write_textgrid,
)


def text_to_pieces(text):
    """Split a chunk transcript exactly like docx_to_sentences() splits a
    paragraph, so every piece fits the G2P model's input limit."""
    pieces = []
    for sentence in split_into_sentences(text):
        pieces.extend(split_long_sentence(sentence))
    return pieces


def load_metadata(path, delimiter):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, text = line.split(delimiter, 1)
            rows.append((name, text))
    return rows


def g2p_with_cache(pieces, cache_path, g2p_repo, tmp_dir):
    """Return {piece: phonemes}, running G2P only on pieces not cached yet."""
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = sorted({p for p in pieces if p not in cache})
    if todo:
        if not (Path(g2p_repo) / "inference.py").exists():
            raise SystemExit(
                f"{len(todo)} text pieces need G2P but there's no inference.py in "
                f"{g2p_repo!r} -- run this from the nemo-pashto-g2p-conformer folder "
                "or pass --g2p-repo."
            )
        print(f"    running G2P on {len(todo)} new pieces ({len(pieces) - len(todo)} cached)")
        phonemes = run_g2p(todo, g2p_repo, tmp_dir)
        if len(phonemes) != len(todo):
            raise SystemExit(f"G2P returned {len(phonemes)} rows for {len(todo)} inputs")
        cache.update(zip(todo, phonemes))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        print(f"    all {len(pieces)} pieces already cached")
    return cache


def load_aligner(device):
    # Built from its parts with do_phonemize=False: we feed ready-made
    # phoneme tokens, so the tokenizer doesn't need phonemizer/espeak.
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(ALIGN_MODEL_ID, do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ALIGN_MODEL_ID)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL_ID, use_safetensors=True)
    model.eval().to(device)
    return processor, model


def load_clip(path):
    # soundfile reads mp3 directly, so no ffmpeg/torchcodec needed.
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.mean(axis=1)))
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="pashto_dataset",
                    help="folder with metadata.csv and clips/ from make_chunks_metadata.py")
    ap.add_argument("--g2p-repo", default=".",
                    help="cloned nemo-pashto-g2p-conformer folder (only used for uncached text)")
    ap.add_argument("--tmp-dir", default="./_g2p_tmp")
    ap.add_argument("--delimiter", default="|", help="metadata.csv column separator")
    ap.add_argument("--overwrite", action="store_true", help="redo existing TextGrids")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    clip_dir = dataset / "clips"
    out_dir = dataset / "textgrids"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(dataset / "metadata.csv", args.delimiter)
    todo = [(name, text) for name, text in rows
            if args.overwrite or not (out_dir / f"{Path(name).stem}.TextGrid").exists()]
    print(f"{len(rows)} clips in metadata.csv, {len(todo)} to align")
    if not todo:
        return

    print("1/3 Running G2P on the chunk transcripts...")
    pieces_per_clip = [text_to_pieces(text) for _, text in todo]
    g2p = g2p_with_cache([p for ps in pieces_per_clip for p in ps],
                         dataset / "g2p_cache" / "chunks.json", args.g2p_repo, args.tmp_dir)

    print("2/3 Loading the alignment model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_aligner(device)
    symbols = build_vocab_symbols(processor.tokenizer)
    print(f"    using device: {device}")

    print("3/3 Aligning each clip and writing TextGrids...")
    unmatched_all, failed = set(), []
    for n, ((name, _), pieces) in enumerate(zip(todo, pieces_per_clip), start=1):
        tokens, unmatched = sentences_to_phoneme_tokens([g2p[p] for p in pieces], symbols)
        unmatched_all |= unmatched
        waveform = load_clip(clip_dir / name)
        duration = waveform.shape[0] / TARGET_SR
        try:
            segments = forced_align(processor, model, waveform, tokens, device)
        except RuntimeError as e:  # e.g. more phonemes than audio frames
            failed.append(name)
            print(f"    [{n}/{len(todo)}] {name}: alignment FAILED ({e})", file=sys.stderr)
            continue
        write_textgrid(segments, duration, str(out_dir / f"{Path(name).stem}.TextGrid"))
        print(f"    [{n}/{len(todo)}] {name}: {len(tokens)} phonemes, {duration:.1f}s")

    if unmatched_all:
        print(f"WARNING: IPA symbols not in the aligner vocab were dropped: {sorted(unmatched_all)}")
    print(f"Done: {len(todo) - len(failed)} TextGrids in {out_dir}"
          + (f"; {len(failed)} failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
