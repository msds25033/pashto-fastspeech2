#!/usr/bin/env python3
"""
Train FastSpeech2 on the whole dataset on Kaggle, over as many sessions as it takes
(a Kaggle GPU session ends after 12 hours). Used by kaggle_train.ipynb; every session
runs the same command:

    python kaggle_train.py

What it does:
  1. keeps everything worth saving in /kaggle/working (Kaggle's saved output):
     checkpoints + logs (output/), the preprocessed features (preprocessed_data/) and
     runs/<run-name>/ (configs, metrics, the aligner model);
  2. restores them from /kaggle/input when the previous session's output is attached
     as an input (the latest checkpoint, the logs and the features), so training resumes;
  3. first session only (no features found): installs the aligner, aligns all clips
     (training an aligner model on the full corpus, ~1 h), and preprocesses them;
  4. trains with a checkpoint at the end of every epoch, and stops by itself, with a
     checkpoint, before the session's time runs out (--hours), so the output is saved.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

START = time.time()
REPO = Path(__file__).resolve().parent
FS2 = REPO / "FastSpeech2"


def run(*cmd):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    if subprocess.run([str(c) for c in cmd], cwd=REPO).returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}")


def pipeline(*cmd):
    run(sys.executable, "pipeline.py", *cmd)


def hours_used():
    return (time.time() - START) / 3600


def ckpt_steps(ckpt_dir):
    return sorted(int(p.name.split(".")[0]) for p in Path(ckpt_dir).glob("*.pth.tar")
                  if p.name.split(".")[0].isdigit())


def link(repo_path, work_path):
    """repo_path becomes a link to work_path (in /kaggle/working, which Kaggle saves)."""
    work_path.mkdir(parents=True, exist_ok=True)
    if repo_path.is_symlink():
        repo_path.unlink()
    elif repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.symlink_to(work_path, target_is_directory=True)


def copy(src, dst):
    if src.exists():
        print(f"  {src} -> {dst}", flush=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def restore(inp, work, run_name):
    """Copy the previous session's output (attached as an input) into /kaggle/working."""
    ckpts = [(int(p.name.split(".")[0]), p) for p in inp.glob("**/output/ckpt/Pashto/*.pth.tar")
             if p.name.split(".")[0].isdigit()]
    feats = sorted(inp.glob("**/preprocessed_data/Pashto/train.txt"))
    if not ckpts and not feats:
        print(f"nothing to restore in {inp}: starting from scratch")
        return
    if ckpts:
        step, latest = max(ckpts)
        base = latest.parents[3]  # <base>/output/ckpt/Pashto/<step>.pth.tar
        print(f"restoring from {base} (latest checkpoint: step {step})")
        for p in sorted(base.glob("output/ckpt/Pashto/*.pth.tar")):  # all checkpoints, so the output keeps them
            copy(p, work / "output" / "ckpt" / "Pashto" / p.name)
        copy(base / "output" / "log", work / "output" / "log")
    else:
        base = feats[0].parents[2]  # <base>/preprocessed_data/Pashto/train.txt
        print(f"restoring the preprocessed features from {base} (no checkpoint yet)")
    copy(base / "preprocessed_data" / "Pashto", work / "preprocessed_data" / "Pashto")
    copy(base / "runs" / run_name, work / "runs" / run_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=11.5,
                    help="this session's time budget, counted from the start of this script "
                         "(Kaggle stops GPU sessions at 12 h; keep a margin)")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--steps", type=int, default=300000, help="total training steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-last", type=int, default=2,
                    help="epoch checkpoints to keep (every 10000th step is always kept)")
    ap.add_argument("--align", choices=["train", "existing"], default="train",
                    help="train an aligner model on all clips (default), or use data/mfa/pashto_acoustic.zip "
                         "(trained on 4 clips only, so it aligns the rest less well)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="aligner processes")
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    args = ap.parse_args()
    work = args.work

    # 1. setup, then point the generated folders at /kaggle/working
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")
    pipeline("setup")
    link(FS2 / "output", work / "output")
    link(FS2 / "preprocessed_data" / "Pashto", work / "preprocessed_data" / "Pashto")
    link(REPO / "runs", work / "runs")

    # 2. continue from the previous session
    if not ckpt_steps(work / "output" / "ckpt" / "Pashto") and args.input.exists():
        restore(args.input, work, args.run_name)

    # 3. first session: align + preprocess all clips
    if not (work / "preprocessed_data" / "Pashto" / "train.txt").exists():
        pipeline("install-mfa")
        pipeline("corpus")
        align = ["align", "--run-name", args.run_name, "--jobs", args.jobs]
        pipeline(*align, *(["--train-acoustic"] if args.align == "train" else []))
        pipeline("preprocess")
        print(f"\nalign + preprocess took {hours_used():.2f} h", flush=True)

    # 4. train until the session budget is used up
    left = args.hours - hours_used() - 0.25  # 15 min for the last epoch, validation and the metrics export
    if left < 0.25:
        sys.exit(f"only {args.hours - hours_used():.2f} h left in this session: run again to start training")
    steps = ckpt_steps(work / "output" / "ckpt" / "Pashto")
    train = ["train", "--run-name", args.run_name, "--steps", args.steps, "--batch-size", args.batch_size,
             "--save-every-epoch", "--keep-last", args.keep_last, "--time-limit", f"{left:.3f}"]
    if steps:
        train.append("--resume")
    pipeline(*train)

    steps = ckpt_steps(work / "output" / "ckpt" / "Pashto")
    print(f"\nlatest checkpoint: step {steps[-1] if steps else None}; checkpoints kept: {steps}")
    print(f"session time used: {hours_used():.2f} h. Saved output in {work}:")
    run("du", "-sh", *sorted(str(p) for p in work.iterdir()))


if __name__ == "__main__":
    main()
