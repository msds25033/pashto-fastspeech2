# Pashto FastSpeech2

Pashto text-to-speech with [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2):
a ~5.7-hour single-speaker Pashto dataset (759 clips of ~30 s), a pronunciation
lexicon, a Montreal Forced Aligner (MFA) model, and one script (`pipeline.py`)
that runs every step from alignment to synthesized audio, on Colab or locally.

## What's in the repo

```
pipeline.py              all steps: setup -> corpus -> align -> preprocess -> train -> synthesize / overfit-test
FastSpeech2/             ming024/FastSpeech2 (commit d4e79eb), bundled and already patched for Pashto:
                         train.py, synthesize.py, preprocess.py, model/, hifigan/ ... (see FastSpeech2/PASHTO_README.md)
fs2_pashto_tools.py      the patches applied to FastSpeech2/ (setup re-applies them; no-op when already in)
fs2_overfit_test.py      regenerates the training clips and compares them with the originals
colab_train.ipynb        Colab notebook: install -> data -> train -> synthesize -> overfit test
colab_generate.ipynb     Colab notebook: load a trained checkpoint from Google Drive and generate speech (no training)
requirements.txt         Python packages for training / synthesis
config/Pashto/           FastSpeech2 configs (preprocess / model / train)
patches/pashto_cleaners.py   Pashto text cleaner appended to FastSpeech2's text/cleaners.py
patches/fastspeech2_pashto.diff  every change made to upstream FastSpeech2, as a diff
data/
  clips/                 759 mp3 clips (1_crop-001.mp3 ... 9_crop-035.mp3)
  metadata.csv           clip|Pashto transcript
  segments.csv           where each clip comes from in the long recording
  lexicon/               pashto-lexicon.txt (word<TAB>phones), phone list, unknown words,
                         pron_overrides.txt (hand-written pronunciations)
  mfa/pashto_acoustic.zip  MFA acoustic model (trained on the 4 demo clips only)
  demo_sentences.txt     the sentences `synthesize` reads by default
  sources/               the original Word transcripts (the long recordings aren't in git)
dataset_tools/           how data/ was built (needs the NeMo Pashto G2P model; see below)
```

Nothing else is cloned: FastSpeech2's code, including `FastSpeech2/synthesize.py` and
`FastSpeech2/train.py`, is part of this repo. Created when you run the pipeline
(git-ignored): inside `FastSpeech2/`, the aligner corpus (`raw_data/`), features
(`preprocessed_data/Pashto/`), checkpoints and logs (`output/`), the unzipped vocoder
and the copied configs/lexicon; and `bin/` (micromamba). Results go to `runs/<run-name>/`.

## Quick start on Colab (the 4-clip demo)

1. Push this folder to your git host, or zip it.
2. Open `colab_train.ipynb` in Colab and pick a GPU runtime.
3. Edit `REPO_URL` in the first cell, or upload the zip to `/content`.
   For a private repo, first add a Colab secret `GITHUB_TOKEN`: a fine-grained GitHub token
   with *Contents: Read-only* on this repo. Turn on its *Notebook access*.
4. Run all cells.

The notebook only calls `pipeline.py`. The same steps as commands:

```bash
pip install -r requirements.txt
python pipeline.py setup                       # bundled FastSpeech2: vocoder, configs, lexicon
python pipeline.py install-mfa                 # Montreal Forced Aligner (Linux/Colab)
python pipeline.py corpus --clips "1_crop-00[1-4]"
python pipeline.py align
python pipeline.py preprocess
python pipeline.py train --run-name demo4 --steps 3000 --warmup 1000 --batch-size 4 \
    --log-step 10 --val-step 100 --synth-step 500 --save-step 1000
python pipeline.py synthesize --run-name demo4
```

The two demo outputs are written to `runs/demo4/audio/`:
- `seen_1_crop-001_step3000.wav`: a sentence from a training clip.
- `unseen_10_crop-001_step3000.wav`: a sentence the model never heard.

Each has a matching spectrogram image and text file.

**What to expect from 4 clips (2 minutes of speech).** The model mostly memorizes
them. The seen sentence should be recognisable; the unseen one will likely be
garbled. Good quality needs the full dataset.

## Pipeline stages

| Stage | What it does |
|---|---|
| `setup` | Prepares the bundled `FastSpeech2/`: re-applies `fs2_pashto_tools.py patch`, `modernize` and `symbols` (no-ops, since they are already applied). Unzips the HiFi-GAN universal vocoder. Merges `pron_overrides.txt` into the lexicon. Copies `config/Pashto/`. Safe to re-run. |
| `install-mfa` | Linux/Colab: micromamba plus the conda-forge `montreal-forced-aligner` environment. |
| `corpus --clips ...` | Writes 22.05 kHz wav and normalised `.lab` text for the selected clips. Clips are names or patterns; leave `--clips` out to use all 759. |
| `align` | `mfa align` with `data/mfa/pashto_acoustic.zip`, or `--train-acoustic` to train a new aligner model on the corpus. Then labels silences `sp` and checks every phone. |
| `preprocess` | Runs FastSpeech2's `preprocess.py`. Below 20 clips it holds out 1 clip for validation and also trains on it (`--val-size`, `--[no-]train-on-all`). |
| `train` | Runs FastSpeech2's `train.py`. Flags override `train.yaml`: `--steps --batch-size --warmup --log-step --val-step --synth-step --save-step --restore-step`; `--resume` continues from the latest checkpoint with the run's saved settings and only on the same clips; `--overfit` switches dropout off. Exports metrics when it ends. |
| `metrics` | Re-exports metrics at any time, e.g. while a long run is in progress. |
| `synthesize` | Turns `data/demo_sentences.txt`, `--sentences FILE` (`name|text` lines) or `--text "..."` into wav, through `FastSpeech2/synthesize.py`. Uses the latest checkpoint, or `--step N`. |
| `overfit-test` | Regenerates every training clip with a checkpoint and compares it with the original: audio side by side, spectrograms, metrics. `--step N`, `--clips ...`. See below. |

`all` runs setup → corpus → align → preprocess → train → synthesize in one go.

## What training records

Everything is written to `runs/<run-name>/`:

| File | Contents |
|---|---|
| `metrics/losses.csv` | Every logged step, training and validation. Columns: total loss, mel loss, mel post-net loss, pitch loss, energy loss, duration loss, learning rate. |
| `metrics/loss_curves.png` | Training vs validation curve for each of the 6 losses. |
| `metrics/learning_rate.png` | The learning-rate schedule. |
| `metrics/summary.json` | Last values, best validation step, list of checkpoints. |
| `metrics/train_log.txt`, `metrics/val_log.txt` | FastSpeech2's text logs. |
| `metrics/samples/` | At each synth step and validation step: a spectrogram/pitch/energy plot, the audio rebuilt from the real spectrogram (`*_reconstructed.wav`), and the model's prediction (`*_synthesized.wav`). |
| `config/` | The exact configs used. |
| `audio/` | Output of `synthesize`. |

There is no "accuracy": TTS models are judged by these losses, which measure how
close the predicted spectrogram, pitch, energy and phone durations are to the
real ones, and by listening to the samples.

Live view: `tensorboard --logdir FastSpeech2/output/log/Pashto`. Checkpoints
(about 350 MB each) are in `FastSpeech2/output/ckpt/Pashto/`.

## Overfitting test: regenerate the training clips

```bash
python pipeline.py overfit-test --run-name demo4 --step 3000
```

Needs the checkpoint plus the preprocessed data of the clips it was trained on. On Colab
with the checkpoints on Google Drive, use **`colab_generate.ipynb`**: it links the Drive folder,
recomputes the clips' data (deterministic) and runs this stage. No training is involved. Output goes to `runs/<run-name>/overfit_test/step<N>/`, with six
versions of every clip:

| File | What it is | What it tells you |
|---|---|---|
| `<clip>_1_original.wav` | the recording | the target |
| `<clip>_2_vocoder.wav` | real spectrogram → HiFi-GAN | the best this vocoder can do on this voice |
| `<clip>_3_teacher_forced.wav` | model given the real durations, pitch and energy | how well the spectrogram decoder memorised the clip |
| `<clip>_4_real_durations.wav` | model given only the real durations (pitch/energy predicted) | 4 vs 3 = the pitch/energy predictors |
| `<clip>_5_from_phones.wav` | model alone, from the training phone sequence (with pauses) | the real overfitting test; 5 vs 4 = the duration predictor |
| `<clip>_6_from_text.wav` | model alone, from the Pashto transcript via the lexicon, with pauses at punctuation | 6 vs 5 = the text front end |

Each clip also gets `<clip>_mels.png`, its spectrograms one under the other.

`metrics.csv` / `metrics.json` give, per version:
- **`mel_l1_dtw`:** the mean absolute log-mel difference from the real spectrogram, after
  dynamic time warping lines the frames up. It's on the same scale as the training mel
  loss; lower is closer.
- **`length_ratio`:** generated length ÷ real length.

A well-overfitted model has versions 3–5 close to version 2, both by ear and by
`mel_l1_dtw`, and `length_ratio` near 1.

**Making it overfit.** Continue the same run with dropout off, until the predictors
memorise the clips too:

```bash
python pipeline.py train --run-name demo4 --resume --overfit --steps 10000 \
    --save-step 1000 --val-step 250 --synth-step 1000
python pipeline.py overfit-test --run-name demo4          # latest checkpoint; compare with step 3000
```

- **`--resume`** continues from the newest checkpoint, using the run's saved settings
  (warm-up, batch size), so the learning-rate schedule carries on. It refuses to start if
  `train.txt` holds different clips from the ones recorded in `runs/<run>/train_clips.txt`.
- **`--overfit`** sets the encoder, decoder and duration/pitch/energy-predictor dropout to 0.
  The default config uses 0.2 / 0.2 / 0.5, which works against memorisation; dropout has no
  weights, so the checkpoint loads either way.

## Training on the full dataset

- **Aligner:** `data/mfa/pashto_acoustic.zip` was trained on the 4 demo clips only. For
  all 759 clips, train a new one: `python pipeline.py align --train-acoustic`.
  It's saved to `runs/<run-name>/pashto_acoustic.zip`.
- **Training settings:** use the full config values, e.g.
  `python pipeline.py train --run-name full --steps 100000 --batch-size 16`.
  If the GPU runs out of memory with 30 s clips, lower `--batch-size` to 8.
- **Colab sessions end.** Put `FastSpeech2/output` and `runs` on Google Drive
  (notebook step 3). In each new session, rerun setup/corpus/align/preprocess
  (or keep `FastSpeech2/preprocessed_data` on Drive too), then continue with
  `--restore-step <last saved step>`.
- **Rough timing on a T4:** about 1–1.5 s per step at batch size 16, so 15–35 hours for
  50k–100k steps. A usable voice often appears around 20k–30k steps.
  On a CPU it is about 22 s per step even with only 4 clips, so train on a GPU.

## Changes to FastSpeech2 and the configs

The bundled `FastSpeech2/` already contains these changes. The full diff against upstream
is `patches/fastspeech2_pashto.diff`. `fs2_pashto_tools.py patch` applies them to a fresh
upstream checkout; it is idempotent and aborts if the code doesn't match what it expects:

- **`text/cleaners.py`:** adds `pashto_cleaners`, the same word normalisation the
  lexicon and aligner used.
- **`text/symbols.py`:** replaced by the 37 Pashto phones plus `sp`/`spn`/`sil`
  (the `symbols` command).
- **librosa calls:** `modernize` rewrites them to keyword arguments for
  librosa ≥ 0.10 / 1.0, including `pad_center`, the call that crashed `preprocess.py`.
- **`train.py`:** runs on a handful of clips (the original requires
  `batch_size × 4 < clips`) and logs the learning rate.
- **`synthesize.py`:** adds a Pashto (`language: "ps"`) text front end and `--name`. The front
  end inserts a pause (`sp`, the aligner's silence label) at punctuation; without it, text
  input has no pauses at all, while this speaker pauses a lot.
- **`utils/tools.py`:** energy targets are cast to float32 like pitch. Without this,
  normalised energy is float64 under NumPy 2, and training on a GPU fails with
  `Found dtype Double but expected Float`.
- **`utils/model.py`, `audio/stft.py`:** checkpoints load on CPU and with
  PyTorch ≥ 2.6 (`weights_only=False` for FastSpeech2's own checkpoints), and the
  STFT no longer requires CUDA.

Config: pitch and energy are now `phoneme_level` with normalisation and
`linear` quantisation, as in FastSpeech2's default LJSpeech config. With the
earlier `frame_level` + `log` settings, silent frames keep pitch 0, and the
model's log-spaced pitch bins start at log(0), which is NaN.

## Running locally (no GPU)

Every stage runs on CPU; only training is slow.

```bash
pip install -r requirements.txt            # plus a CPU build of torch
conda create -n aligner -c conda-forge montreal-forced-aligner
python pipeline.py setup
python pipeline.py corpus --clips "1_crop-00[1-4]"
python pipeline.py align --mfa "conda run -n aligner mfa"
python pipeline.py preprocess
python pipeline.py train --run-name cpu_test --steps 30 --log-step 5 --val-step 10 --synth-step 20 --save-step 30
python pipeline.py synthesize --run-name cpu_test
```

## Rebuilding the dataset (`dataset_tools/`)

This is how `data/` was made from the 18 long recordings and their Word transcripts. It
needs the NeMo Pashto G2P model
([TBOGamer22/nemo-pashto-g2p-conformer](https://huggingface.co/TBOGamer22/nemo-pashto-g2p-conformer)),
so run it on Colab from that model's folder:

```bash
python dataset_tools/build_dataset.py --sources <folder with n.docx + n_crop.mp3> --out-dir pashto_dataset
```

It runs, per recording:
1. `docx_to_alignment.py`: G2P plus wav2vec2 forced alignment, giving a phoneme TextGrid.
2. `make_chunks_metadata.py`: cuts the recording into ~30 s clips at real pauses and writes `metadata.csv`.

Then, once for everything, `make_lexicon.py` builds the word→phone lexicon,
mapping rare G2P phones into the 37-phone set.
`align_chunks.py` (per-clip wav2vec2 TextGrids) is kept for reference; the
FastSpeech2 pipeline uses MFA instead.
