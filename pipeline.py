#!/usr/bin/env python3
"""
Pashto text-to-speech with ming024/FastSpeech2 -- the Colab notebook steps as
one command-line tool. Run the stages in order, or all of them with `all`:

  setup        prepare the bundled FastSpeech2/ (already patched for Pashto; the
               patches are re-applied idempotently), unzip the HiFi-GAN vocoder,
               install the configs, lexicon and phone set
  install-mfa  (Linux / Colab) install Montreal Forced Aligner with micromamba
  corpus       write the aligner corpus (22.05 kHz wav + normalised .lab) for
               the selected clips
  align        Montreal Forced Aligner -> phone TextGrids for FastSpeech2
  preprocess   FastSpeech2 preprocess.py (mel, pitch, energy, durations)
  train        FastSpeech2 train.py, then export the training metrics
  metrics      export losses / learning rate (CSV), curves (PNG), summary and
               the logged sample spectrograms + audio from TensorBoard
  synthesize   Pashto text -> wav with a trained checkpoint
  overfit-test regenerate the training clips with a checkpoint and compare them
               with the originals (audio side by side, spectrograms, metrics)

Everything for one experiment (metrics, curves, logs, configs, audio) is
collected in runs/<run-name>/. FastSpeech2/ is ming024/FastSpeech2, bundled and
patched; the corpus, features and checkpoints it generates are git-ignored.

Demo on the 4 sample clips (see README.md for Colab):
    python pipeline.py all --clips "1_crop-00[1-4]" --steps 3000
"""

import argparse
import fnmatch
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "dataset_tools"))

DATASET = "Pashto"
LOSS_NAMES = ["total_loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss",
              "duration_loss"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None):
    """Run a command, streaming its output; stop the pipeline if it fails."""
    print(f"\n$ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(f"command failed with exit code {result.returncode}")


def tools(*args):
    run([sys.executable, REPO / "fs2_pashto_tools.py", *args])


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(path, data):
    import yaml
    Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                          encoding="utf-8")


def fs2_config(args, name):
    return args.fs2_dir / "config" / DATASET / f"{name}.yaml"


def corpus_dir(args):
    return args.fs2_dir / "raw_data" / DATASET


def textgrid_dir(args):
    return args.fs2_dir / "preprocessed_data" / DATASET / "TextGrid"


def run_dir(args):
    d = REPO / "runs" / args.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints(args):
    ckpt_dir = args.fs2_dir / load_yaml(fs2_config(args, "train"))["path"]["ckpt_path"]
    return sorted((int(p.name.split(".")[0]) for p in ckpt_dir.glob("*.pth.tar")
                   if p.name.split(".")[0].isdigit()))


def require_setup(args):
    if not fs2_config(args, "preprocess").exists():
        sys.exit(f"{args.fs2_dir} is not set up yet -- run `python pipeline.py setup` first")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def merged_lexicon(args):
    """data lexicon + hand-written pronunciations (pron_overrides.txt win)."""
    from make_lexicon import read_extra_lexicon
    lex = {}
    for line in (args.data_dir / "lexicon" / "pashto-lexicon.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            word, _, phones = line.partition("\t")
            lex[word] = phones.split()
    overrides = args.data_dir / "lexicon" / "pron_overrides.txt"
    if overrides.exists():
        lex.update(read_extra_lexicon(overrides))
    return lex


def stage_setup(args):
    fs2 = args.fs2_dir
    if not (fs2 / "train.py").exists():
        sys.exit(f"{fs2} not found: FastSpeech2 is bundled in this repo, so check that the "
                 "FastSpeech2/ folder was cloned along with it (or pass --fs2-dir).")
    print(f"using the bundled {fs2}")

    tools("patch", "--repo", fs2)
    tools("modernize", "--repo", fs2)
    tools("symbols", "--repo", fs2, "--phones", args.data_dir / "lexicon" / "pashto-lexicon.phones.txt")

    vocoder = fs2 / "hifigan" / "generator_universal.pth.tar"
    if not vocoder.exists():
        with zipfile.ZipFile(fs2 / "hifigan" / "generator_universal.pth.tar.zip") as z:
            z.extractall(fs2 / "hifigan")
        print(f"unzipped {vocoder.name}")

    lex = merged_lexicon(args)
    (fs2 / "lexicon").mkdir(exist_ok=True)
    with open(fs2 / "lexicon" / "pashto-lexicon.txt", "w", encoding="utf-8", newline="\n") as f:
        for word in sorted(lex):
            f.write(f"{word}\t{' '.join(lex[word])}\n")
    print(f"lexicon: {len(lex)} entries -> {fs2 / 'lexicon' / 'pashto-lexicon.txt'}")

    cfg_dir = fs2 / "config" / DATASET
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name in ("preprocess", "model", "train"):
        shutil.copy(REPO / "config" / DATASET / f"{name}.yaml", cfg_dir / f"{name}.yaml")
    print(f"configs copied to {cfg_dir}")


# ---------------------------------------------------------------------------
# install-mfa
# ---------------------------------------------------------------------------

def stage_install_mfa(args):
    if platform.system() != "Linux":
        sys.exit("install-mfa automates the Linux/Colab install only. On Windows/macOS install it with\n"
                 "    conda create -n aligner -c conda-forge montreal-forced-aligner\n"
                 "and pass  --mfa \"conda run -n aligner mfa\"  to the align stage.")
    mm = REPO / "bin" / "micromamba"
    if not mm.exists():
        run(["bash", "-c", f"cd {shlex.quote(str(REPO))} && "
             "wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"])
    run([mm, "create", "-y", "-q", "-n", "aligner", "-c", "conda-forge", "montreal-forced-aligner"])
    run([mm, "run", "-n", "aligner", "mfa", "version"])


def mfa_command(args):
    if args.mfa:
        return shlex.split(args.mfa, posix=os.name != "nt")
    if shutil.which("mfa"):
        return ["mfa"]
    mm = REPO / "bin" / "micromamba"
    if mm.exists():
        return [str(mm), "run", "-n", "aligner", "mfa"]
    sys.exit("Montreal Forced Aligner not found. Run `python pipeline.py install-mfa` (Linux/Colab) "
             "or pass --mfa with the command that runs it.")


# ---------------------------------------------------------------------------
# corpus / align
# ---------------------------------------------------------------------------

def select_rows(args):
    from make_lexicon import load_metadata
    rows = load_metadata(args.data_dir / "metadata.csv")
    if args.clips:
        rows = [r for r in rows
                if any(fnmatch.fnmatch(Path(r[0]).stem, pat) or r[0] == pat for pat in args.clips)]
    if not rows:
        sys.exit(f"no clips in metadata.csv match {args.clips}")
    return rows


def stage_corpus(args):
    require_setup(args)
    from make_lexicon import normalize_words, write_corpus
    rows = select_rows(args)
    known = set(merged_lexicon(args))
    unknown = sorted({w for _, text in rows for w in normalize_words(text) if w not in known})
    out = corpus_dir(args)
    if out.exists():
        shutil.rmtree(out)
    print(f"Writing {len(rows)} clips to the aligner corpus {out} ...")
    write_corpus(rows, args.data_dir / "clips", out, args.speaker, 22050, strip_diacritics=False)
    if unknown:
        print(f"note: {len(unknown)} words have no lexicon entry; the aligner labels them as "
              f"spoken noise (spn): {unknown[:15]}")
    print(f"corpus ready: {len(rows)} clips")


def stage_align(args):
    require_setup(args)
    corpus = corpus_dir(args)
    if not any(corpus.rglob("*.lab")):
        sys.exit(f"{corpus} has no clips -- run the corpus stage first")
    out = textgrid_dir(args)
    if out.exists():
        shutil.rmtree(out)
    lexicon = args.fs2_dir / "lexicon" / "pashto-lexicon.txt"
    mfa = mfa_command(args)
    # Options go AFTER the positional arguments: MFA passes unknown options such as
    # --beam through to its config, and before the positionals their values would
    # be taken as the corpus directory.
    if args.train_acoustic:
        model_out = run_dir(args) / "pashto_acoustic.zip"
        run([*mfa, "train", corpus, lexicon, model_out, "--output_directory", out,
             "--clean", "-j", args.jobs])
        print(f"new acoustic model saved to {model_out} (copy it to data/mfa/ to reuse it)")
    else:
        run([*mfa, "align", corpus, lexicon, args.acoustic_model, out,
             "--clean", "-j", args.jobs, "--beam", "100", "--retry_beam", "400"])

    # FastSpeech2 expects TextGrid/<speaker>/<clip>.TextGrid
    (out / args.speaker).mkdir(parents=True, exist_ok=True)
    for tg in out.glob("*.TextGrid"):
        tg.replace(out / args.speaker / tg.name)
    n = len(list((out / args.speaker).glob("*.TextGrid")))
    n_clips = len(list(corpus.rglob("*.lab")))
    print(f"{n} TextGrids for {n_clips} clips")
    if n < n_clips:
        print(f"WARNING: {n_clips - n} clips could not be aligned and will be skipped by preprocess")

    tools("fix-textgrids", "--textgrid-dir", out,
          "--phones", args.data_dir / "lexicon" / "pashto-lexicon.phones.txt")


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def stage_preprocess(args):
    require_setup(args)
    n = len(list((textgrid_dir(args) / args.speaker).glob("*.TextGrid")))
    if n == 0:
        sys.exit("no TextGrids -- run the align stage first")
    val_size = args.val_size if args.val_size is not None else (1 if n < 20 else max(1, round(0.05 * n)))
    train_on_all = args.train_on_all if args.train_on_all is not None else n < 20

    cfg = load_yaml(fs2_config(args, "preprocess"))
    cfg["preprocessing"]["val_size"] = val_size
    save_yaml(fs2_config(args, "preprocess"), cfg)

    out = args.fs2_dir / cfg["path"]["preprocessed_path"]
    for sub in ("mel", "pitch", "energy", "duration"):
        shutil.rmtree(out / sub, ignore_errors=True)
    run([sys.executable, "preprocess.py", f"config/{DATASET}/preprocess.yaml"], cwd=args.fs2_dir)

    train, val = out / "train.txt", out / "val.txt"
    train_lines = train.read_text(encoding="utf-8").splitlines()
    val_lines = val.read_text(encoding="utf-8").splitlines()
    if train_on_all:
        train_lines += [ln for ln in val_lines if ln not in train_lines]
        train.write_text("".join(ln + "\n" for ln in train_lines), encoding="utf-8")
        print("train-on-all: the validation clip(s) were added to train.txt too, so validation "
              "loss measures fit, not generalisation")
    print(f"preprocessed: {len(train_lines)} training / {len(val_lines)} validation utterances")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def stage_train(args):
    require_setup(args)
    pre = load_yaml(fs2_config(args, "preprocess"))
    train_txt = args.fs2_dir / pre["path"]["preprocessed_path"] / "train.txt"
    if not train_txt.exists():
        sys.exit("no train.txt -- run the preprocess stage first")
    n_train = len([ln for ln in train_txt.read_text(encoding="utf-8").splitlines() if ln.strip()])

    cfg = load_yaml(fs2_config(args, "train"))
    overrides = {
        ("optimizer", "batch_size"): args.batch_size,
        ("optimizer", "warm_up_step"): args.warmup,
        ("step", "total_step"): args.steps,
        ("step", "log_step"): args.log_step,
        ("step", "val_step"): args.val_step,
        ("step", "synth_step"): args.synth_step,
        ("step", "save_step"): args.save_step,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value
    if cfg["optimizer"]["batch_size"] > n_train:
        print(f"batch_size {cfg['optimizer']['batch_size']} > {n_train} training utterances; "
              f"using {n_train}")
        cfg["optimizer"]["batch_size"] = n_train
    save_yaml(fs2_config(args, "train"), cfg)
    print("training config: " + json.dumps({**cfg["optimizer"], **cfg["step"]}))

    rd = run_dir(args)
    (rd / "config").mkdir(exist_ok=True)
    for name in ("preprocess", "model", "train"):
        shutil.copy(fs2_config(args, name), rd / "config" / f"{name}.yaml")

    cmd = [sys.executable, "train.py", "-p", f"config/{DATASET}/preprocess.yaml",
           "-m", f"config/{DATASET}/model.yaml", "-t", f"config/{DATASET}/train.yaml"]
    if args.restore_step:
        cmd += ["--restore_step", args.restore_step]
    try:
        run(cmd, cwd=args.fs2_dir)
    finally:  # export whatever was logged, also when training stops early or is interrupted
        stage_metrics(args)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def read_tensorboard(log_dir):
    """-> scalars {tag: {step: value}}, images [(tag, step, png)], audio [(tag, step, wav)]"""
    from tensorboard.backend.event_processing import event_accumulator as ea
    acc = ea.EventAccumulator(str(log_dir), size_guidance={ea.SCALARS: 0, ea.IMAGES: 0, ea.AUDIO: 0})
    acc.Reload()
    tags = acc.Tags()
    scalars = {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags.get("scalars", [])}
    images = [(t, e.step, e.encoded_image_string) for t in tags.get("images", []) for e in acc.Images(t)]
    audio = [(t, e.step, e.encoded_audio_string) for t in tags.get("audio", []) for e in acc.Audio(t)]
    return scalars, images, audio


def stage_metrics(args):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_yaml(fs2_config(args, "train"))
    log_root = args.fs2_dir / cfg["path"]["log_path"]
    if not (log_root / "train").exists():
        print(f"no training logs in {log_root} yet")
        return
    out = run_dir(args) / "metrics"
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    series = {}  # (split, name) -> {step: value}
    for split in ("train", "val"):
        if not (log_root / split).exists():
            continue
        scalars, images, audio = read_tensorboard(log_root / split)
        for tag, values in scalars.items():
            series[(split, tag.split("/")[-1])] = values
        for tag, step, data in images:
            (samples / f"{split}_{tag.replace('/', '_')}.png").write_bytes(data)
        for tag, step, data in audio:
            (samples / f"{split}_{tag.replace('/', '_')}.wav").write_bytes(data)
        if (log_root / split / "log.txt").exists():
            shutil.copy(log_root / split / "log.txt", out / f"{split}_log.txt")

    # losses.csv: one row per (split, step)
    columns = LOSS_NAMES + ["learning_rate"]
    with open(out / "losses.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "step"] + columns)
        for split in ("train", "val"):
            steps = sorted({s for (sp, _), vals in series.items() if sp == split for s in vals})
            for s in steps:
                w.writerow([split, s] + [series.get((split, c), {}).get(s, "") for c in columns])

    # curves: every loss, training vs validation
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, LOSS_NAMES):
        for split, style in (("train", dict(alpha=0.6, lw=1)), ("val", dict(marker="o", ms=3, lw=1.5))):
            vals = series.get((split, name))
            if vals:
                steps = sorted(vals)
                ax.plot(steps, [vals[s] for s in steps], label=split, **style)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend()
    fig.suptitle(f"FastSpeech2 Pashto -- {args.run_name}: training vs validation losses (log scale)")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=120)
    plt.close(fig)

    lr = series.get(("train", "learning_rate"))
    if lr:
        fig, ax = plt.subplots(figsize=(7, 4))
        steps = sorted(lr)
        ax.plot(steps, [lr[s] for s in steps])
        ax.set_title("learning rate")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "learning_rate.png", dpi=120)
        plt.close(fig)

    summary = {"run": args.run_name, "checkpoints": checkpoints(args)}
    for split in ("train", "val"):
        total = series.get((split, "total_loss"))
        if total:
            last = max(total)
            summary[f"{split}_last_step"] = last
            summary[f"{split}_last"] = {n: series[(split, n)].get(last) for n in LOSS_NAMES
                                        if (split, n) in series}
            best = min(total, key=total.get)
            summary[f"{split}_best_total_loss"] = {"step": best, "value": total[best]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nmetrics exported to {out}: losses.csv, loss_curves.png, learning_rate.png, "
          f"summary.json, *_log.txt, samples/ ({len(list(samples.iterdir()))} files)")
    for split in ("train", "val"):
        if f"{split}_last" in summary:
            print(f"  {split} @ step {summary[f'{split}_last_step']}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in summary[f"{split}_last"].items()))


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

def read_sentences(args):
    items = []
    for t in args.text or []:
        items.append((None, t))
    if args.sentences or not items:
        path = Path(args.sentences or REPO / "data" / "demo_sentences.txt")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, sep, text = line.partition("|")
                items.append((name.strip(), text.strip()) if sep else (None, line))
    return [(name or f"sentence_{i:02d}", text) for i, (name, text) in enumerate(items, start=1)]


def stage_synthesize(args):
    require_setup(args)
    ckpts = checkpoints(args)
    step = args.step or (ckpts[-1] if ckpts else None)
    if step is None or step not in ckpts:
        sys.exit(f"no checkpoint {'for step %s ' % step if step else ''}found (have: {ckpts}) -- "
                 "train first, or pass --step")
    result_dir = args.fs2_dir / load_yaml(fs2_config(args, "train"))["path"]["result_path"]
    out = run_dir(args) / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in read_sentences(args):
        run([sys.executable, "synthesize.py", "--restore_step", step, "--mode", "single",
             "--text", text, "--name", name,
             "--duration_control", args.duration_control,
             "-p", f"config/{DATASET}/preprocess.yaml", "-m", f"config/{DATASET}/model.yaml",
             "-t", f"config/{DATASET}/train.yaml"], cwd=args.fs2_dir)
        for ext in (".wav", ".png"):
            src = result_dir / f"{name}{ext}"
            if src.exists():
                shutil.copy(src, out / f"{name}_step{step}{ext}")
        (out / f"{name}_step{step}.txt").write_text(text + "\n", encoding="utf-8")
        print(f"-> {out / f'{name}_step{step}.wav'}")


# ---------------------------------------------------------------------------
# overfit-test
# ---------------------------------------------------------------------------

def stage_overfit_test(args):
    """Regenerate the training clips with a checkpoint and compare them with the originals."""
    require_setup(args)
    pre = load_yaml(fs2_config(args, "preprocess"))
    pre_dir = args.fs2_dir / pre["path"]["preprocessed_path"]
    if not (pre_dir / "train.txt").exists() or not (pre_dir / "stats.json").exists():
        sys.exit(f"no preprocessed data in {pre_dir} -- run corpus, align and preprocess first "
                 "(with the same --clips as for training)")
    ckpts = checkpoints(args)
    step = args.step or (ckpts[-1] if ckpts else None)
    if step is None or step not in ckpts:
        sys.exit(f"no checkpoint {'for step %s ' % step if step else ''}found (have: {ckpts})")
    out = run_dir(args) / "overfit_test" / f"step{step}"
    cmd = [sys.executable, REPO / "fs2_overfit_test.py", "--restore_step", step, "--out", out,
           "-p", f"config/{DATASET}/preprocess.yaml", "-m", f"config/{DATASET}/model.yaml",
           "-t", f"config/{DATASET}/train.yaml"]
    if args.clips:
        cmd += ["--clips", *args.clips]
    run(cmd, cwd=args.fs2_dir)


# ---------------------------------------------------------------------------

def stage_all(args):
    stage_setup(args)
    stage_corpus(args)
    stage_align(args)
    stage_preprocess(args)
    stage_train(args)
    stage_synthesize(args)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fs2-dir", type=Path, default=REPO / "FastSpeech2",
                        help="FastSpeech2 folder (bundled in the repo)")
    common.add_argument("--data-dir", type=Path, default=REPO / "data")
    common.add_argument("--run-name", default="pashto",
                        help="experiment name; results go to runs/<run-name>/")
    common.add_argument("--speaker", default="pashto")

    corpus_opts = argparse.ArgumentParser(add_help=False)
    corpus_opts.add_argument("--clips", nargs="+",
                             help='clips to use: names or patterns, e.g. "1_crop-00[1-4]" (default: all)')

    align_opts = argparse.ArgumentParser(add_help=False)
    align_opts.add_argument("--mfa", help='command that runs MFA, e.g. "conda run -n aligner mfa" '
                                          "(default: mfa on PATH, else the install-mfa environment)")
    align_opts.add_argument("--acoustic-model", type=Path, default=REPO / "data" / "mfa" / "pashto_acoustic.zip")
    align_opts.add_argument("--train-acoustic", action="store_true",
                            help="train a new MFA acoustic model on the corpus instead of using "
                                 "--acoustic-model (do this when you change the clips substantially)")
    align_opts.add_argument("--jobs", type=int, default=2)

    pre_opts = argparse.ArgumentParser(add_help=False)
    pre_opts.add_argument("--val-size", type=int, help="validation clips (default: 1 below 20 clips, else 5%%)")
    pre_opts.add_argument("--train-on-all", action=argparse.BooleanOptionalAction, default=None,
                          help="also train on the validation clips (default: on below 20 clips)")

    train_opts = argparse.ArgumentParser(add_help=False)
    train_opts.add_argument("--steps", type=int, help="total training steps (config: total_step)")
    train_opts.add_argument("--batch-size", type=int)
    train_opts.add_argument("--warmup", type=int, help="learning-rate warm-up steps")
    train_opts.add_argument("--log-step", type=int, help="log training losses every N steps")
    train_opts.add_argument("--val-step", type=int, help="validate every N steps")
    train_opts.add_argument("--synth-step", type=int, help="log a sample spectrogram + audio every N steps")
    train_opts.add_argument("--save-step", type=int, help="save a checkpoint every N steps")
    train_opts.add_argument("--restore-step", type=int, help="continue from this checkpoint")

    synth_opts = argparse.ArgumentParser(add_help=False)
    synth_opts.add_argument("--step", type=int, help="checkpoint to use (default: latest)")
    synth_opts.add_argument("--text", action="append", help="Pashto sentence (repeatable)")
    synth_opts.add_argument("--sentences", help="file with one 'name|text' (or just text) per line "
                                                "(default: data/demo_sentences.txt)")
    synth_opts.add_argument("--duration-control", type=float, default=1.0,
                            help="speaking-rate factor (>1 = slower)")

    overfit_opts = argparse.ArgumentParser(add_help=False)
    overfit_opts.add_argument("--step", type=int, help="checkpoint to test (default: latest)")
    overfit_opts.add_argument("--clips", nargs="+", help="only these clips (default: all training clips)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    stages = {
        "setup": (stage_setup, []),
        "install-mfa": (stage_install_mfa, []),
        "corpus": (stage_corpus, [corpus_opts]),
        "align": (stage_align, [align_opts]),
        "preprocess": (stage_preprocess, [pre_opts]),
        "train": (stage_train, [train_opts]),
        "metrics": (stage_metrics, []),
        "synthesize": (stage_synthesize, [synth_opts]),
        "overfit-test": (stage_overfit_test, [overfit_opts]),
        "all": (stage_all, [corpus_opts, align_opts, pre_opts, train_opts, synth_opts]),
    }
    for name, (fn, parents) in stages.items():
        p = sub.add_parser(name, parents=[common, *parents], help=fn.__doc__)
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    args.fs2_dir = args.fs2_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.fn(args)


if __name__ == "__main__":
    main()
