"""
End-to-end pipeline for Pashto forced alignment.

    Word doc (transcript)  --sentence split-->  sentences.txt
    sentences.txt          --nemo-pashto-g2p--> IPA phoneme strings
    IPA phoneme strings    --vocab match-->      wav2vec2 phoneme tokens
    audio + phoneme tokens --forced_align-->     Praat TextGrid

Run this from inside the cloned G2P repo, AFTER you've already done in Colab:

    from huggingface_hub import snapshot_download
    snapshot_download(
        "TBOGamer22/nemo-pashto-g2p-conformer",
        local_dir="/content/nemo-pashto-g2p-conformer",
    )
    %cd /content/nemo-pashto-g2p-conformer
    !pip install -r requirements.txt
    !pip install phonemizer python-docx torch torchaudio transformers textgrid

Then:

    !python docx_to_alignment.py --docx /content/1.docx --audio /content/audio.mp3 --out /content/output.TextGrid

(copy this file into /content/nemo-pashto-g2p-conformer/ first, or pass
--g2p-repo pointing at that folder)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import docx
import torch
import torchaudio
import textgrid
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ALIGN_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
TARGET_SR = 16000

# --------------------------------------------------------------------------
# 1. Word doc -> sentence-split text (matches the shape of your 1.txt sample)
# --------------------------------------------------------------------------

# Pashto/Urdu sentence terminators. Runs of the same/mixed terminator
# (e.g. "۔۔۔") are treated as ONE boundary, not split into empty sentences.
SENT_BOUNDARY = re.compile(r"([۔؟!]+)")


def split_into_sentences(paragraph_text):
    parts = SENT_BOUNDARY.split(paragraph_text)
    sentences, buf = [], ""
    for part in parts:
        buf += part
        if SENT_BOUNDARY.fullmatch(part):
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


# The G2P model rejects inputs beyond ~768 chars after an internal 2x
# repeat, i.e. roughly 384 raw chars — some transcript "sentences" have no
# terminating punctuation for a long stretch and blow past that. Split those
# further: first on commas, then (if a clause still has none) on word
# boundaries. This only affects how many lines go into the G2P step; the
# resulting phoneme strings are concatenated back into one sequence anyway.
MAX_G2P_CHARS = 250
CLAUSE_BOUNDARY = re.compile(r"([،,]+)")


def split_long_sentence(sentence, max_chars=MAX_G2P_CHARS):
    if len(sentence) <= max_chars:
        return [sentence]

    parts = CLAUSE_BOUNDARY.split(sentence)
    chunks, buf = [], ""
    for part in parts:
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf.strip())
            buf = part
        else:
            buf += part
    if buf.strip():
        chunks.append(buf.strip())

    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue
        words, wbuf = c.split(), ""
        for w in words:
            if wbuf and len(wbuf) + len(w) + 1 > max_chars:
                final.append(wbuf.strip())
                wbuf = w
            else:
                wbuf = (wbuf + " " + w).strip()
        if wbuf:
            final.append(wbuf.strip())
    return final


def docx_to_sentences(docx_path):
    doc = docx.Document(docx_path)
    sentences = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        for s in split_into_sentences(text):
            sentences.extend(split_long_sentence(s))
    return sentences


# --------------------------------------------------------------------------
# 2. Sentences -> phonemes, via the NeMo Pashto G2P model's inference.py
# --------------------------------------------------------------------------

def run_g2p(sentences, g2p_repo_dir, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    text_file = tmp_dir / "sentences.txt"
    text_file.write_text("\n".join(sentences), encoding="utf-8")

    cmd = [
        sys.executable, "inference.py",
        "--text-file", str(text_file.resolve()),
        "--batch-size", "32",
        "--json",
    ]
    result = subprocess.run(cmd, cwd=g2p_repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"G2P inference.py failed:\n{result.stderr}")

    # NOTE: the exact output location for --text-file + --json isn't
    # documented, so this tries a couple of likely spots, then falls back
    # to parsing stdout. If none of these hit, run the command manually
    # once, see where/how it prints or saves, and adjust this function.
    candidates = (
        list(tmp_dir.glob("*phoneme*.json"))
        + list(Path(g2p_repo_dir).glob("*phoneme*.json"))
        + list(tmp_dir.glob("sentences*.json"))
    )
    data = None
    if candidates:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
    else:
        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            lines = [ln for ln in stdout.splitlines() if ln.strip().startswith("{")]
            if lines:
                data = [json.loads(ln) for ln in lines]

    if data is None:
        raise RuntimeError(
            "Could not locate the G2P JSON output automatically.\n"
            f"stdout was:\n{result.stdout}\n\nstderr was:\n{result.stderr}\n"
            "Run the inference.py command by hand once, see where it writes "
            "its JSON (or how it prints it), and point run_g2p() at that."
        )

    phoneme_strings = [row["phonemes"] for row in data]
    if len(phoneme_strings) != len(sentences):
        print(
            f"WARNING: got {len(phoneme_strings)} phoneme rows for "
            f"{len(sentences)} input sentences — check they line up.",
            file=sys.stderr,
        )
    return phoneme_strings


# --------------------------------------------------------------------------
# 3. IPA phoneme strings -> individual tokens matching wav2vec2's vocab
#
# The G2P model emits one glued IPA string per sentence, e.g. "ka muŋ gwɑɽu"
# with no separator between phones within a word. The aligner's vocab is a
# set of discrete phoneme symbols (some multi-character, e.g. affricates
# like "d͡ʒ"). Greedily match the longest known vocab symbol at each
# position so the glued string becomes a token sequence the aligner can use.
# --------------------------------------------------------------------------

def build_vocab_symbols(tokenizer):
    special = set(tokenizer.all_special_tokens)
    symbols = [s for s in tokenizer.get_vocab() if s not in special and s.strip()]
    symbols.sort(key=len, reverse=True)
    return symbols


def normalize_ipa(ipa_string):
    # G2P output sometimes uses plain ASCII 'g' where the aligner's vocab
    # uses the IPA script-g 'ɡ' (U+0261) — normalize so it matches.
    return ipa_string.replace("g", "ɡ")


def segment_into_vocab_tokens(ipa_string, symbols):
    ipa_string = normalize_ipa(ipa_string)
    tokens, unmatched = [], []
    i, n = 0, len(ipa_string)
    while i < n:
        ch = ipa_string[i]
        if ch.isspace():
            i += 1
            continue
        matched = None
        for sym in symbols:
            if ipa_string.startswith(sym, i):
                matched = sym
                break
        if matched:
            tokens.append(matched)
            i += len(matched)
        else:
            unmatched.append(ch)
            i += 1
    return tokens, unmatched


def sentences_to_phoneme_tokens(phoneme_strings, symbols):
    all_tokens, all_unmatched = [], set()
    for s in phoneme_strings:
        toks, unmatched = segment_into_vocab_tokens(s, symbols)
        all_tokens.extend(toks)
        all_unmatched.update(unmatched)
    return all_tokens, all_unmatched


# --------------------------------------------------------------------------
# 4. Forced alignment
# --------------------------------------------------------------------------

def load_align_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(ALIGN_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL_ID, use_safetensors=True)
    model.eval().to(device)
    return processor, model, device


def load_audio(path, target_sr=TARGET_SR):
    waveform, sr = torchaudio.load(path)  # mp3 needs ffmpeg (present on Colab by default)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)


# Wav2Vec2's self-attention cost grows quadratically with audio length, so a
# single forward pass over a long recording is slow/memory-hungry (this is
# almost certainly what your run was about to hit). Instead, run the model
# over fixed-length chunks and concatenate the resulting log-probs — the
# actual forced_align() call still runs once, over the full concatenated
# sequence and the full phoneme target, so alignment stays globally correct.
CHUNK_SECONDS = 20


def compute_log_probs_chunked(processor, model, waveform, device, chunk_seconds=CHUNK_SECONDS):
    chunk_samples = int(chunk_seconds * TARGET_SR)
    n_chunks = max(1, -(-waveform.shape[0] // chunk_samples))  # ceil division
    all_log_probs = []
    for idx, start in enumerate(range(0, waveform.shape[0], chunk_samples), start=1):
        chunk = waveform[start:start + chunk_samples]
        if chunk.shape[0] == 0:
            continue
        with torch.no_grad():
            input_values = processor(
                chunk.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_values.to(device)
            logits = model(input_values).logits  # (1, T_chunk, V)
            log_probs = torch.log_softmax(logits, dim=-1)
        all_log_probs.append(log_probs[0].cpu())
        print(f"    chunk {idx}/{n_chunks} processed", flush=True)
    return torch.cat(all_log_probs, dim=0).unsqueeze(0)  # (1, T_total, V)


def forced_align(processor, model, waveform, phoneme_tokens, device):
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    targets = torch.tensor([[vocab[p] for p in phoneme_tokens]], dtype=torch.int32)

    log_probs = compute_log_probs_chunked(processor, model, waveform, device)

    blank_id = tokenizer.pad_token_id
    aligned_tokens, _ = torchaudio.functional.forced_align(log_probs, targets, blank=blank_id)
    aligned_tokens = aligned_tokens[0].tolist()

    frame_duration = waveform.shape[0] / TARGET_SR / log_probs.shape[1]
    id_to_token = {v: k for k, v in vocab.items()}

    segments, prev_tok, start = [], None, 0
    for i, tok in enumerate(aligned_tokens):
        if tok != prev_tok:
            if prev_tok is not None and prev_tok != blank_id:
                segments.append((start * frame_duration, i * frame_duration, id_to_token[prev_tok]))
            start, prev_tok = i, tok
    if prev_tok is not None and prev_tok != blank_id:
        segments.append((start * frame_duration, len(aligned_tokens) * frame_duration, id_to_token[prev_tok]))

    return [{"start": s, "end": e, "phoneme": p} for s, e, p in segments]


# --------------------------------------------------------------------------
# 5. Write TextGrid
# --------------------------------------------------------------------------

def write_textgrid(segments, duration, out_path, tier_name="phonemes"):
    tg = textgrid.TextGrid(maxTime=duration)
    tier = textgrid.IntervalTier(name=tier_name, maxTime=duration)
    prev_end = 0.0
    for seg in segments:
        if seg["start"] > prev_end:
            tier.add(prev_end, seg["start"], "")
        tier.add(seg["start"], seg["end"], seg["phoneme"])
        prev_end = seg["end"]
    if prev_end < duration:
        tier.add(prev_end, duration, "")
    tg.append(tier)
    tg.write(out_path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True, help="path to the Word transcript, e.g. 1.docx")
    ap.add_argument("--audio", required=True, help="path to the audio file, e.g. audio.mp3")
    ap.add_argument("--g2p-repo", default=".", help="path to the cloned nemo-pashto-g2p-conformer dir")
    ap.add_argument("--tmp-dir", default="./_g2p_tmp")
    ap.add_argument("--out", default="output.TextGrid")
    args = ap.parse_args()

    print("1/4 Extracting + splitting sentences from the Word doc...")
    sentences = docx_to_sentences(args.docx)
    print(f"    {len(sentences)} sentences")

    print("2/4 Running G2P to get phonemes for each sentence...")
    phoneme_strings = run_g2p(sentences, args.g2p_repo, args.tmp_dir)

    print("3/4 Loading the alignment model and mapping phonemes onto its vocab...")
    processor, align_model, device = load_align_model()
    print(f"    using device: {device}")
    symbols = build_vocab_symbols(processor.tokenizer)
    phoneme_tokens, unmatched = sentences_to_phoneme_tokens(phoneme_strings, symbols)
    if unmatched:
        print(f"    WARNING: {len(unmatched)} unmatched IPA symbols, dropped: {sorted(unmatched)}")
    print(f"    {len(phoneme_tokens)} phoneme tokens total")

    print("4/4 Running forced alignment and writing the TextGrid...")
    waveform = load_audio(args.audio)
    segments = forced_align(processor, align_model, waveform, phoneme_tokens, device)
    duration = waveform.shape[0] / TARGET_SR
    write_textgrid(segments, duration, args.out)

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
