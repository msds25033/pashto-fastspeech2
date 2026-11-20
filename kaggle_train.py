#!/usr/bin/env python3
"""
Train FastSpeech2 on the whole dataset on Kaggle in sessions of at most --hours (default 6),
with everything a session needs to continue kept in a Hugging Face model repo
(--hf-repo, default speech-itu/pashto-fastspeech2). Used by kaggle_train.ipynb.

The Hugging Face repo (private; created on first use) holds:
    pashto_aligned.zip   the alignments; the first session uploads data/pashto_aligned.zip
                         from this git repo (dataset_tools/ctc_to_textgrids.py --zip)
    features.tar         the preprocessed features, pushed by the first session
    ckpt/<step>.pth.tar  checkpoints: every save_step-th (10000) one + the newest
    log/                 TensorBoard logs
    runs/full/           configs, metrics, loss curves, stats.json/speakers.json
The token comes from the HF_TOKEN environment variable (a Kaggle secret; write access to the org).

Each session:
  1. pulls the newest checkpoint, logs, runs/full/ and the features (or alignments);
  2. no features yet: preprocesses the clips from the alignments and pushes features.tar;
  3. trains, with a checkpoint at the end of every epoch (on disk), pushes the newest one to
     the Hub every --push-minutes, and stops itself (with a checkpoint, then a final push)
     before --hours runs out.

--smoke: a short check in an interactive session: everything above, but training only
--smoke-minutes, pushing every minute to smoke/ in the repo (deleted afterwards), then
checking a checkpoint downloads back, and printing how long the full training will take.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

START = time.time()
REPO = Path(__file__).resolve().parent
FS2 = REPO / "FastSpeech2"
STAGE = Path("/kaggle/temp/hf") if Path("/kaggle").exists() else REPO / "_hf_stage"


def hours_used():
    return (time.time() - START) / 3600


def run(*cmd):
    print(f"\n[{hours_used():.2f} h] $ {' '.join(map(str, cmd))}", flush=True)
    if subprocess.run([str(c) for c in cmd], cwd=REPO).returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}")


def pipeline(*cmd):
    run(sys.executable, "pipeline.py", *cmd)


def ckpt_steps(ckpt_dir):
    return sorted(int(p.name.split(".")[0]) for p in Path(ckpt_dir).glob("*.pth.tar")
                  if p.name.split(".")[0].isdigit())


def link(repo_path, work_path):
    """repo_path becomes a link to work_path."""
    work_path.mkdir(parents=True, exist_ok=True)
    if repo_path.is_symlink():
        repo_path.unlink()
    elif repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.symlink_to(work_path, target_is_directory=True)


class Hub:
    """The Hugging Face repo. `prefix` puts checkpoints / logs / runs under a subfolder
    (smoke/ for the smoke test); features.tar and pashto_aligned.zip are always at the top."""

    def __init__(self, repo_id, prefix, save_step, run_name):
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN is not set: add a Hugging Face token with write access to the org "
                     "as a Kaggle secret called HF_TOKEN (see the notebook).")
        self.api, self.repo, self.prefix = HfApi(token=token), repo_id, prefix
        self.save_step, self.run_name = save_step, run_name
        self.api.create_repo(repo_id, private=True, exist_ok=True)
        print(f"Hugging Face repo: https://huggingface.co/{repo_id}" + (f" (under {prefix})" if prefix else ""))
        self.pushed_step = None

    def files(self):
        return self.api.list_repo_files(self.repo)

    def seed_alignments(self, path):
        """First use of the repo: upload the alignments that ship with the git repo."""
        if "pashto_aligned.zip" in self.files() or not path.exists():
            return
        print(f"uploading {path.name} to the Hub (first use of the repo) ...", flush=True)
        self.api.upload_file(path_or_fileobj=str(path), path_in_repo="pashto_aligned.zip",
                             repo_id=self.repo, commit_message="alignments")

    def remote_ckpts(self, files):
        out = {}
        for f in files:
            name = f[len(self.prefix + "ckpt/"):] if f.startswith(self.prefix + "ckpt/") else ""
            if name.endswith(".pth.tar") and name.split(".")[0].isdigit():
                out[int(name.split(".")[0])] = f
        return out

    def get(self, path, dest):
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(self.repo, path, local_dir=STAGE / "pull", token=self.api.token)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(local, dest)
        return dest

    def pull(self, work):
        files = self.files()
        pashto = work / "preprocessed_data" / "Pashto"
        ckpts = self.remote_ckpts(files)
        if ckpts:
            step = max(ckpts)
            print(f"pulling checkpoint {step}, logs and runs/{self.run_name}/ ...", flush=True)
            self.get(ckpts[step], work / "output" / "ckpt" / "Pashto" / f"{step}.pth.tar")
            self.pushed_step = step
            for f in files:
                for remote, local in ((self.prefix + "log/", work / "output" / "log"),
                                      (self.prefix + f"runs/{self.run_name}/", work / "runs" / self.run_name)):
                    if f.startswith(remote):
                        self.get(f, local / f[len(remote):])
        else:
            print("no checkpoints on the Hub yet: training starts from scratch")
        if not (pashto / "train.txt").exists() and "features.tar" in files:
            print("pulling features.tar ...", flush=True)
            tar = self.get("features.tar", STAGE / "features.tar")
            with tarfile.open(tar) as t:
                t.extractall(pashto)
            tar.unlink()
        if not (pashto / "train.txt").exists() and not any(pashto.glob("TextGrid/*/*.TextGrid")) \
                and "pashto_aligned.zip" in files:
            print("pulling pashto_aligned.zip ...", flush=True)
            z = self.get("pashto_aligned.zip", STAGE / "pashto_aligned.zip")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(work)  # preprocessed_data/Pashto/TextGrid/... + runs/full/pashto_acoustic.zip
            z.unlink()

    def push_features(self, work):
        if "features.tar" in self.files():
            return
        pashto = work / "preprocessed_data" / "Pashto"
        tar = STAGE / "features.tar"
        tar.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{hours_used():.2f} h] pushing features.tar ...", flush=True)
        with tarfile.open(tar, "w") as t:
            for p in sorted(pashto.iterdir()):
                t.add(p, p.name)
        self.api.upload_file(path_or_fileobj=str(tar), path_in_repo="features.tar", repo_id=self.repo,
                             commit_message="preprocessed features")
        tar.unlink()

    def push(self, work, final=False):
        """Push the newest checkpoint (+ any new save_step one), the logs and runs/<run>/, and delete
        older non-save_step checkpoints. Old file versions are squashed away to keep the repo small."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        steps = ckpt_steps(work / "output" / "ckpt" / "Pashto")
        if not steps or (steps[-1] == self.pushed_step and not final):
            return
        stage = STAGE / "push"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        files = self.files()
        remote = self.remote_ckpts(files)
        keep = {s for s in steps if s % self.save_step == 0} | {steps[-1]}
        ops = []
        for s in sorted(keep - set(remote)):
            src, dst = work / "output" / "ckpt" / "Pashto" / f"{s}.pth.tar", stage / f"{s}.pth.tar"
            try:
                os.link(src, dst)  # a snapshot: training may delete the file meanwhile
            except OSError:
                shutil.copy2(src, dst)
            ops.append(CommitOperationAdd(f"{self.prefix}ckpt/{s}.pth.tar", str(dst)))
        ops += [CommitOperationDelete(path) for s, path in remote.items()
                if s % self.save_step != 0 and s != steps[-1]]
        for local, remote_dir in ((work / "output" / "log", "log"), (work / "runs" / self.run_name, f"runs/{self.run_name}")):
            if local.exists():
                snap = stage / remote_dir
                shutil.copytree(local, snap)  # a copy, so files training appends to don't change mid-upload
                ops += [CommitOperationAdd(f"{self.prefix}{remote_dir}/{p.relative_to(snap).as_posix()}", str(p))
                        for p in sorted(snap.rglob("*")) if p.is_file()]
        t = time.time()
        self.api.create_commit(self.repo, ops, commit_message=f"step {steps[-1]}")
        try:
            self.api.super_squash_history(self.repo, commit_message=f"step {steps[-1]}")
        except Exception as e:  # only keeps the storage down
            print(f"(history squash skipped: {e})")
        self.pushed_step = steps[-1]
        shutil.rmtree(stage, ignore_errors=True)
        print(f"\n[{hours_used():.2f} h] pushed checkpoint {steps[-1]} to the Hub in {time.time() - t:.0f} s", flush=True)

    def safe_push(self, work, final=False):
        for attempt in range(3 if final else 1):
            try:
                return self.push(work, final)
            except Exception as e:
                print(f"\npush to the Hub failed ({e}); " + ("retrying" if final and attempt < 2 else "will retry later"),
                      flush=True)
                time.sleep(10)


def restore_inputs(inp, work):
    """Alignments or features attached as a Kaggle dataset (instead of on the Hub)."""
    pashto = work / "preprocessed_data" / "Pashto"
    if not (pashto / "train.txt").exists():
        feats = sorted(inp.glob("**/preprocessed_data/Pashto/train.txt"))
        if feats:
            print(f"features found: {feats[0].parent}")
            shutil.copytree(feats[0].parent, pashto, dirs_exist_ok=True)
    if not (pashto / "train.txt").exists() and not any(pashto.glob("TextGrid/*/*.TextGrid")):
        grids = sorted(inp.glob("**/preprocessed_data/Pashto/TextGrid"))
        if grids:
            print(f"alignments found: {grids[0]}")
            shutil.copytree(grids[0], pashto / "TextGrid", dirs_exist_ok=True)


def estimate(work, setup_hours, args):
    """Training speed from the epoch checkpoints' save times -> how long the whole run takes."""
    ck = work / "output" / "ckpt" / "Pashto"
    pts = sorted((s, (ck / f"{s}.pth.tar").stat().st_mtime) for s in ckpt_steps(ck))
    if len(pts) < 3:
        print("\nTIME ESTIMATE: too few epochs finished; run the smoke test with a larger SMOKE_MINUTES.")
        return
    (s0, t0), (s1, t1) = pts[1], pts[-1]  # from the 2nd checkpoint: skips start-up
    per_hour = (s1 - s0) / (t1 - t0) * 3600
    per_session = max(0.0, args.hours - setup_hours - 0.4) * per_hour  # 0.4 h: final epoch, push, metrics
    print(f"\nTIME ESTIMATE (from {len(pts)} epochs; with T4 x2 and batch size {args.batch_size}):")
    print(f"  speed: {per_hour:,.0f} steps/hour ({3600 / per_hour:.2f} s/step); epoch = {pts[-1][0] - pts[-2][0]} steps")
    print(f"  start-up this session: {setup_hours * 60:.0f} min (later sessions skip preprocessing)")
    print(f"  a {args.hours:g} h session trains ~{per_session:,.0f} steps")
    for target in sorted({20000, 50000, 100000, args.steps}):
        h = target / per_hour
        sessions = target / per_session if per_session else float("inf")
        print(f"  {target:>7,} steps: {h:5.1f} h of training = {sessions:4.1f} sessions of {args.hours:g} h"
              f" = {h / 30:.1f} weeks of the 30 h weekly GPU quota")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=6.0, help="this session's maximum length (from the start of this script)")
    ap.add_argument("--hf-repo", default="speech-itu/pashto-fastspeech2", help="Hugging Face model repo")
    ap.add_argument("--push-minutes", type=float, default=30, help="push the newest checkpoint this often")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--steps", type=int, default=100000, help="total training steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-last", type=int, default=2, help="epoch checkpoints kept on disk")
    ap.add_argument("--smoke", action="store_true", help="short test run (see above)")
    ap.add_argument("--smoke-minutes", type=float, default=10, help="training time of the --smoke run")
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
    import yaml
    save_step = yaml.safe_load(open(REPO / "config" / "Pashto" / "train.yaml", encoding="utf-8"))["step"]["save_step"]
    hub = Hub(args.hf_repo, "smoke/" if args.smoke else "", save_step, args.run_name)

    # 2. pull the newest state (+ features or alignments)
    hub.seed_alignments(REPO / "data" / "pashto_aligned.zip")
    hub.pull(work)
    if args.input.exists():
        restore_inputs(args.input, work)
    pashto = work / "preprocessed_data" / "Pashto"

    # 3. first session: features from the alignments
    if not (pashto / "train.txt").exists():
        if not any(pashto.glob("TextGrid/*/*.TextGrid")):
            sys.exit(f"no alignments: commit data/pashto_aligned.zip (dataset_tools/ctc_to_textgrids.py "
                     f"--zip) to the git repo, upload it to https://huggingface.co/{args.hf_repo}, "
                     f"or attach it as a Kaggle dataset.")
        pipeline("corpus")      # the clips as 22.05 kHz wav + text, which preprocess reads
        pipeline("preprocess")
        hub.push_features(work)

    # 4. train, pushing to the Hub as it goes
    setup_hours = hours_used()
    if args.smoke:
        limit, interval = args.smoke_minutes / 60, 1.0
        extra = ["--log-step", "20", "--val-step", "200", "--synth-step", "200"]
        keep_last = 1000  # keep every epoch checkpoint: their times give the speed estimate
    else:
        limit, interval, extra, keep_last = args.hours - hours_used() - 0.4, args.push_minutes, [], args.keep_last
        if limit < 0.25:
            sys.exit(f"only {args.hours - hours_used():.2f} h left in this session")
    resume = ["--resume"] if ckpt_steps(work / "output" / "ckpt" / "Pashto") else []
    cmd = [sys.executable, "pipeline.py", "train", "--run-name", args.run_name, "--steps", str(args.steps),
           "--batch-size", str(args.batch_size), "--save-every-epoch", "--keep-last", str(keep_last),
           "--time-limit", f"{limit:.3f}", *resume, *extra]
    print(f"\n[{hours_used():.2f} h] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO)
    next_push = time.time() + interval * 60
    while proc.poll() is None:
        time.sleep(5)
        if time.time() >= next_push:
            hub.safe_push(work)
            next_push = time.time() + interval * 60
    hub.safe_push(work, final=True)
    if proc.returncode != 0:
        sys.exit(f"training failed with exit code {proc.returncode} (the newest checkpoint was still pushed)")

    steps = ckpt_steps(work / "output" / "ckpt" / "Pashto")
    print(f"\nsession time used: {hours_used():.2f} h; newest checkpoint: step {steps[-1] if steps else None}, "
          f"on https://huggingface.co/{args.hf_repo}")

    if args.smoke:
        files = hub.files()
        remote = hub.remote_ckpts(files)
        ok = bool(remote) and max(remote) == steps[-1]
        if ok:  # pull it back: the next session's first step
            back = hub.get(remote[max(remote)], STAGE / "check.pth.tar")
            local = work / "output" / "ckpt" / "Pashto" / f"{steps[-1]}.pth.tar"
            ok = back.stat().st_size == local.stat().st_size
            back.unlink()
        try:
            hub.api.delete_folder("smoke", repo_id=args.hf_repo, commit_message="remove smoke test")
            hub.api.super_squash_history(args.hf_repo, commit_message="remove smoke test")
        except Exception as e:
            print(f"(could not remove smoke/ from the repo: {e})")
        estimate(work, setup_hours, args)
        print(f"""
SMOKE TEST {'PASSED' if ok and steps else 'FAILED'}. Check above that:
  - the features / alignments were found (first time: preprocess printed ~720 training / ~38 validation);
  - "Step N/..." loss lines appear, the losses are numbers (no nan) and mostly going down;
  - there is no "CUDA out of memory" (else set BATCH_SIZE = 8);
  - "pushed checkpoint N to the Hub" appeared about every minute, and the checkpoint downloaded back: {ok};
  - "time limit reached: stopped after step N" was printed.
Then set SMOKE_TEST = False and use Save Version -> Save & Run All (Commit).""")


if __name__ == "__main__":
    main()
