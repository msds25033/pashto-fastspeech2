#!/usr/bin/env python3
"""
Train FastSpeech2 on the whole dataset on Kaggle, over as many sessions as it takes
(a Kaggle GPU session ends after 12 hours). Used by kaggle_train.ipynb.

Inputs (attached to the notebook with "Add Input"):
  * the alignments, made beforehand by colab_align.ipynb and uploaded as a Kaggle dataset
    (pashto_aligned.zip: preprocessed_data/Pashto/TextGrid/ + runs/full/pashto_acoustic.zip);
    the aligner runs on CPU only and is too slow to share a GPU session with training;
  * from the second session on: this notebook's own previous output.

What it does:
  1. keeps everything worth saving in /kaggle/working (Kaggle's saved output):
     checkpoints + logs (output/), the features (preprocessed_data/), runs/<run-name>/;
  2. restores the previous session's output, if attached (latest checkpoint, logs, features);
  3. no features yet: preprocesses all clips from the attached alignments (~30 min);
  4. trains with a checkpoint at the end of every epoch and stops by itself, with a
     checkpoint, before the session's time budget (--hours) runs out, so the output is saved.

--smoke: a ~15-minute check in an interactive session before committing: everything above,
but into /kaggle/temp/smoke (not saved), training only --smoke-minutes with frequent
logging / validation, then a list of what to check.
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
    print(f"\n[{hours_used():.2f} h] $ {' '.join(map(str, cmd))}", flush=True)
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
    """Copy what the attached inputs provide into the work folder, newest stage first:
    a previous session's checkpoints (+ its features), else features, else alignments."""
    pashto = work / "preprocessed_data" / "Pashto"
    ckpts = [(int(p.name.split(".")[0]), p) for p in inp.glob("**/output/ckpt/Pashto/*.pth.tar")
             if p.name.split(".")[0].isdigit()]
    if ckpts:
        step, latest = max(ckpts)
        base = latest.parents[3]  # <base>/output/ckpt/Pashto/<step>.pth.tar
        print(f"previous session found: {base} (latest checkpoint: step {step})")
        for p in sorted(base.glob("output/ckpt/Pashto/*.pth.tar")):  # all of them, so the output keeps them
            copy(p, work / "output" / "ckpt" / "Pashto" / p.name)
        copy(base / "output" / "log", work / "output" / "log")
        copy(base / "runs" / run_name, work / "runs" / run_name)
    if not (pashto / "train.txt").exists():
        feats = sorted(inp.glob("**/preprocessed_data/Pashto/train.txt"))
        if feats:
            print(f"features found: {feats[0].parent}")
            copy(feats[0].parent, pashto)
    if not (pashto / "train.txt").exists() and not any(pashto.glob("TextGrid/*/*.TextGrid")):
        grids = sorted(inp.glob("**/preprocessed_data/Pashto/TextGrid"))
        if grids:
            base = grids[0].parents[2]  # <base>/preprocessed_data/Pashto/TextGrid
            print(f"alignments found: {grids[0]} ({len(list(grids[0].glob('*/*.TextGrid')))} TextGrids)")
            copy(grids[0], pashto / "TextGrid")
            copy(base / "runs" / run_name, work / "runs" / run_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=11.5,
                    help="this session's time budget, counted from the start of this script "
                         "(Kaggle stops GPU sessions at 12 h, and earlier when the weekly quota runs out)")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--steps", type=int, default=300000, help="total training steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-last", type=int, default=2,
                    help="epoch checkpoints to keep (every 10000th step is always kept)")
    ap.add_argument("--smoke", action="store_true", help="short test run into /kaggle/temp/smoke (not saved)")
    ap.add_argument("--smoke-minutes", type=float, default=5, help="training time of the --smoke run")
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    args = ap.parse_args()
    work = Path("/kaggle/temp/smoke") if args.smoke else args.work
    if args.smoke:
        shutil.rmtree(work, ignore_errors=True)

    # 1. setup, then point the generated folders at the work folder
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")
    pipeline("setup")
    link(FS2 / "output", work / "output")
    link(FS2 / "preprocessed_data" / "Pashto", work / "preprocessed_data" / "Pashto")
    link(REPO / "runs", work / "runs")

    # 2. inputs: previous session / features / alignments
    pashto = work / "preprocessed_data" / "Pashto"
    if not ckpt_steps(work / "output" / "ckpt" / "Pashto") and args.input.exists():
        restore(args.input, work, args.run_name)

    # 3. features from the alignments (first session)
    if not (pashto / "train.txt").exists():
        if not any(pashto.glob("TextGrid/*/*.TextGrid")):
            sys.exit(f"no alignments, features or checkpoints found in {args.input}. Attach the dataset made "
                     "from colab_align.ipynb's pashto_aligned.zip (Add Input -> Your Work -> Datasets).")
        pipeline("corpus")      # the clips as 22.05 kHz wav + text, which preprocess reads
        pipeline("preprocess")
        print(f"\npreprocessing done after {hours_used():.2f} h", flush=True)

    # 4. train until the session budget is used up
    if args.smoke:
        limit = args.smoke_minutes / 60
        extra = ["--log-step", "10", "--val-step", "50", "--synth-step", "50"]
    else:
        limit = args.hours - hours_used() - 0.25  # 15 min for the last epoch, validation and the metrics export
        extra = []
        if limit < 0.25:
            sys.exit(f"only {args.hours - hours_used():.2f} h left in this session: run again to start training")
    resume = ["--resume"] if ckpt_steps(work / "output" / "ckpt" / "Pashto") else []
    pipeline("train", "--run-name", args.run_name, "--steps", args.steps, "--batch-size", args.batch_size,
             "--save-every-epoch", "--keep-last", args.keep_last, "--time-limit", f"{limit:.3f}", *resume, *extra)

    steps = ckpt_steps(work / "output" / "ckpt" / "Pashto")
    print(f"\nlatest checkpoint: step {steps[-1] if steps else None}; checkpoints kept: {steps}")
    print(f"session time used: {hours_used():.2f} h. Saved in {work}:")
    run("du", "-sh", *sorted(str(p) for p in work.iterdir()))
    if args.smoke:
        print(f"""
SMOKE TEST {'PASSED' if steps else 'FAILED: no checkpoint was saved'}. Check above that:
  - the alignments were found and preprocess printed ~720 training / ~38 validation utterances;
  - "Step N/..." loss lines appear and the losses are numbers (no nan) and mostly going down;
  - there is no "CUDA out of memory" (else set BATCH_SIZE = 8);
  - "time limit reached: stopped after step N" was printed, and checkpoints were kept: {steps};
  - steps per minute: about {steps[-1] / args.smoke_minutes if steps else 0:.0f}, so roughly
    {(steps[-1] / args.smoke_minutes * 60 if steps else 0):.0f} steps per hour of training.
Then set SMOKE_TEST = False and use Save Version -> Save & Run All (Commit).""")


if __name__ == "__main__":
    main()
