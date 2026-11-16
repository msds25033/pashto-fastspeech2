#!/usr/bin/env python3
"""
Speak with a trained checkpoint, from Python (used by colab_inference.ipynb):

    import tts
    tts.load(step=None)                       # latest checkpoint in FastSpeech2/output/ckpt/Pashto
    wav = tts.speak_phones("{d a sp k o r}")  # phone sequence -> int16 audio at tts.SR Hz
    phones = tts.text_to_phones("پښتو متن")    # Pashto text -> "{p h o n e s}"
    wav = tts.speak_text("پښتو متن")           # text -> phones -> audio

Run `python pipeline.py setup` first. The model also needs the pitch/energy statistics of
its training data (stats.json, speakers.json): they are taken from runs/<run>/ (saved there
by `pipeline.py train`) or data/model/<run>/ (committed for the demo4 run).
"""

import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent
FS2 = REPO / "FastSpeech2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 22050
_m = {}


def load(step=None, run_name="demo4"):
    """Load checkpoint `step` (None = latest) and the HiFi-GAN vocoder."""
    os.chdir(FS2)  # FastSpeech2's configs and modules use paths relative to it
    sys.path.insert(0, str(FS2))
    from utils.model import get_model, get_vocoder

    cfg = [yaml.safe_load(open(f"config/Pashto/{n}.yaml", encoding="utf-8")) for n in ("preprocess", "model", "train")]
    pre_dir = Path(cfg[0]["path"]["preprocessed_path"])
    pre_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stats.json", "speakers.json"):
        if not (pre_dir / name).exists():
            src = next((d / name for d in (REPO / "runs" / run_name, REPO / "data" / "model" / run_name)
                        if (d / name).exists()), None)
            if src is None:
                sys.exit(f"no {name} for run '{run_name}' in runs/ or data/model/ -- run the preprocess stage")
            shutil.copy(src, pre_dir / name)
    if step is None:
        steps = [int(p.name.split(".")[0]) for p in Path(cfg[2]["path"]["ckpt_path"]).glob("*.pth.tar")]
        if not steps:
            sys.exit(f"no checkpoints in {FS2 / cfg[2]['path']['ckpt_path']}")
        step = max(steps)
    _m.update(cfg=cfg, model=get_model(type("A", (), {"restore_step": step}), cfg, DEVICE, train=False),
              vocoder=get_vocoder(cfg[1], DEVICE))
    print(f"loaded checkpoint {step} on {DEVICE}")


def text_to_phones(text):
    """Pashto text -> phone sequence "{...}" (lexicon pronunciations, sp at punctuation)."""
    from synthesize import preprocess_pashto
    from text import sequence_to_text
    return sequence_to_text(preprocess_pashto(text, _m["cfg"][0]))


def speak_phones(phones):
    """Phone sequence, e.g. "{d a sp k o r}" (braces optional) -> int16 audio at SR Hz."""
    from text import text_to_sequence
    from utils.model import vocoder_infer
    phones = "{" + phones.strip().strip("{}") + "}"
    ids = text_to_sequence(phones, _m["cfg"][0]["preprocessing"]["text"]["text_cleaners"])
    if not ids:
        raise ValueError(f"no known phones in {phones!r}")
    with torch.no_grad():
        out = _m["model"](torch.tensor([0], device=DEVICE), torch.tensor([ids], device=DEVICE),
                          torch.tensor([len(ids)], device=DEVICE), len(ids))
    n = int(out[9][0])
    mel = out[1][:, :n].transpose(1, 2)  # (1, 80, frames)
    return vocoder_infer(mel, _m["vocoder"], _m["cfg"][1], _m["cfg"][0], lengths=[n * 256])[0]


def speak_text(text):
    """Pashto text -> phones -> int16 audio at SR Hz."""
    return speak_phones(text_to_phones(text))
