#!/usr/bin/env python3
"""
Overfitting test: regenerate the clips the model was trained on and compare them
with the originals. Run through `python pipeline.py overfit-test`, which calls
this from inside FastSpeech2/ (it imports FastSpeech2's own modules).

For every utterance in train.txt / val.txt it writes, into --out:
    <clip>_1_original.wav         the recording (aligner corpus copy, 22.05 kHz)
    <clip>_2_vocoder.wav          real mel-spectrogram -> HiFi-GAN: the best this vocoder can do
    <clip>_3_teacher_forced.wav   model given the real durations / pitch / energy
    <clip>_4_real_durations.wav   model given only the real durations (pitch / energy predicted)
    <clip>_5_from_phones.wav      model alone, from the training phone sequence (with pauses)
    <clip>_6_from_text.wav        model alone, from the Pashto transcript (pauses at punctuation)
    <clip>_mels.png               the spectrograms above, one under the other
plus metrics.csv / metrics.json:
    mel_l1_dtw     mean |log-mel difference| to the real spectrogram, after DTW lines up the
                   frames (same scale as the training mel loss; lower = closer to the original)
    length_ratio   generated length / real length (1.0 = same speaking rate)
Reading the versions: 3 vs 2 = how well the decoder memorised the spectrograms; 4 vs 3 = the
cost of predicted pitch/energy; 5 vs 4 = the cost of predicted durations; 6 vs 5 = the text
front end (lexicon pronunciations, pauses only at punctuation).
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile

sys.path.insert(0, os.getcwd())  # FastSpeech2/
from text import text_to_sequence  # noqa: E402
from utils.model import get_model, get_vocoder, vocoder_infer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_list(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                basename, speaker, phones, raw_text = line.split("|", 3)
                rows.append((basename, speaker, phones, raw_text))
    return rows


def run_model(model, speaker_id, ids, duration=None, pitch=None, energy=None):
    """-> predicted mel (frames x 80) as numpy. Targets that are given replace the model's
    own predictions (teacher forcing); the others are predicted."""
    spk = torch.tensor([speaker_id], dtype=torch.long, device=DEVICE)
    txt = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    lens = torch.tensor([len(ids)], device=DEVICE)
    tensor = lambda a: torch.from_numpy(a).float().unsqueeze(0).to(DEVICE)  # noqa: E731
    kwargs = {}
    if duration is not None:
        n_frames = int(duration.sum())
        kwargs.update(mel_lens=torch.tensor([n_frames], device=DEVICE), max_mel_len=n_frames,
                      d_targets=torch.from_numpy(duration).long().unsqueeze(0).to(DEVICE))
    if pitch is not None:
        kwargs["p_targets"] = tensor(pitch)
    if energy is not None:
        kwargs["e_targets"] = tensor(energy)
    with torch.no_grad():
        out = model(spk, txt, lens, len(ids), **kwargs)
    n = int(out[9][0])
    return out[1][0, :n].cpu().numpy()  # post-net mel


def to_wav(mel, vocoder, model_config, preprocess_config):
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    m = torch.from_numpy(mel.T).float().unsqueeze(0).to(DEVICE)
    return vocoder_infer(m, vocoder, model_config, preprocess_config, lengths=[mel.shape[0] * hop])[0]


def mel_l1_dtw(real, pred):
    import librosa
    _, path = librosa.sequence.dtw(X=real.T, Y=pred.T, metric="euclidean")
    return float(np.mean(np.abs(real[path[:, 0]] - pred[path[:, 1]])))


def plot(mels, titles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    width = max(m.shape[0] for m in mels)
    fig, axes = plt.subplots(len(mels), 1, figsize=(14, 2.2 * len(mels)), squeeze=False)
    for ax, mel, title in zip(axes[:, 0], mels, titles):
        ax.imshow(mel.T, origin="lower", aspect="auto", interpolation="none")
        ax.set_xlim(0, width)
        ax.set_title(title, fontsize=9)
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--preprocess_config", required=True)
    ap.add_argument("-m", "--model_config", required=True)
    ap.add_argument("-t", "--train_config", required=True)
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", help="only these clip names (default: all)")
    ap.add_argument("--metadata", help="metadata.csv: original transcripts with punctuation for the "
                                       "text version (default: the .lab text, which has none)")
    args = ap.parse_args()

    pre = yaml.safe_load(open(args.preprocess_config, encoding="utf-8"))
    mod = yaml.safe_load(open(args.model_config, encoding="utf-8"))
    trn = yaml.safe_load(open(args.train_config, encoding="utf-8"))
    pre_dir = Path(pre["path"]["preprocessed_path"])
    cleaners = pre["preprocessing"]["text"]["text_cleaners"]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = get_model(args, (pre, mod, trn), DEVICE, train=False)
    vocoder = get_vocoder(mod, DEVICE)
    speakers = json.load(open(pre_dir / "speakers.json", encoding="utf-8"))
    from synthesize import preprocess_pashto  # the same text front end as `synthesize`

    split = {}
    rows = {}
    for name in ("train", "val"):
        for r in read_list(pre_dir / f"{name}.txt"):
            rows[r[0]] = r
            split[r[0]] = "+".join(filter(None, [split.get(r[0]), name]))
    transcripts = {}
    if args.metadata:
        for line in Path(args.metadata).read_text(encoding="utf-8").splitlines():
            if "|" in line:
                name, text = line.split("|", 1)
                transcripts[Path(name).stem] = text
    if args.clips:
        rows = {k: v for k, v in rows.items() if k in args.clips or Path(k).stem in args.clips}
    if not rows:
        sys.exit(f"no utterances found in {pre_dir}/train.txt or val.txt")

    results = []
    for basename, speaker, phones, raw_text in sorted(rows.values()):
        print(f"\n{basename} ({split[basename]})")
        load = lambda kind: np.load(pre_dir / kind / f"{speaker}-{kind}-{basename}.npy")  # noqa: E731
        real_mel, pitch, energy, duration = load("mel"), load("pitch"), load("energy"), load("duration")
        sid = speakers[speaker]
        ids = text_to_sequence(phones, cleaners)

        text = transcripts.get(basename, raw_text)
        variants = {
            "2_vocoder": real_mel,
            "3_teacher_forced": run_model(model, sid, ids, duration, pitch, energy),
            "4_real_durations": run_model(model, sid, ids, duration),
            "5_from_phones": run_model(model, sid, ids),
            "6_from_text": run_model(model, sid, list(preprocess_pashto(text, pre))),
        }
        src = Path(pre["path"]["raw_path"]) / speaker / f"{basename}.wav"
        if src.exists():
            shutil.copy(src, out / f"{basename}_1_original.wav")
        for key, mel in variants.items():
            wavfile.write(out / f"{basename}_{key}.wav", sr, to_wav(mel, vocoder, mod, pre))
            row = {"clip": basename, "split": split[basename], "version": key,
                   "frames": mel.shape[0], "seconds": round(mel.shape[0] * pre["preprocessing"]["stft"]["hop_length"] / sr, 2),
                   "length_ratio": round(mel.shape[0] / real_mel.shape[0], 3),
                   "mel_l1_dtw": 0.0 if key == "2_vocoder" else round(mel_l1_dtw(real_mel, mel), 4)}
            results.append(row)
            print(f"  {key:18s} {row['seconds']:6.2f}s  length x{row['length_ratio']:.2f}  "
                  f"mel L1 (DTW) {row['mel_l1_dtw']:.3f}")
        plot(list(variants.values()),
             [f"{basename}: real mel (what 2_vocoder plays)", "3 teacher-forced (real durations/pitch/energy)",
              "4 real durations, predicted pitch/energy", "5 from training phones (model alone)",
              "6 from Pashto text via lexicon (model alone)"],
             out / f"{basename}_mels.png")

    with open(out / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    summary = {"checkpoint_step": args.restore_step, "clips": len(rows), "mean_by_version": {}}
    for key in variants:
        rs = [r for r in results if r["version"] == key]
        summary["mean_by_version"][key] = {
            "mel_l1_dtw": round(float(np.mean([r["mel_l1_dtw"] for r in rs])), 4),
            "length_ratio": round(float(np.mean([r["length_ratio"] for r in rs])), 3)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nmean over {len(rows)} clips:")
    for key, v in summary["mean_by_version"].items():
        print(f"  {key:18s} mel L1 (DTW) {v['mel_l1_dtw']:.3f}   length x{v['length_ratio']:.2f}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
