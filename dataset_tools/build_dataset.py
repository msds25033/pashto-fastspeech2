#!/usr/bin/env python3
"""
Rebuild data/ (clips, metadata.csv, lexicon) from the long source recordings --
the dataset cells of the original Colab notebook as one script.

For every <n>.docx with an audio file <n>.mp3 / <n>_*.mp3 / <n>.mpeg next to it:
    1. docx_to_alignment.py    -> <n>.TextGrid   (G2P + wav2vec2 forced alignment)
    2. make_chunks_metadata.py -> ~30 s clips + rows in metadata.csv
Then, once for everything:
    3. make_lexicon.py         -> lexicon/pashto-lexicon.txt (+ .phones.txt)
       with the phone clean-up used for the published dataset (--map-phones /
       --drop-words below)

Needs the NeMo G2P model: run it on Colab from the cloned
nemo-pashto-g2p-conformer folder (see dataset_tools/requirements.txt), e.g.

    python /content/pashto-fastspeech2/dataset_tools/build_dataset.py \
        --sources /content/sources --out-dir /content/pashto_dataset

Steps whose output already exists are skipped, so an interrupted run can be resumed.
Add --dry-run to only print the commands.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Rare / noisy phones from the G2P model, merged into the 37-phone set (notebook cell 34)
MAP_PHONES = ("tʰː=t,tʰ=t,ʈʰ=ʈ,pʰ=p,ʰ=h,ħ=h,ʔ=,ɪ=i,y=j,ɔ=o,-=,͡=,d͡=d͡ʒ,d͡a=d͡ʒ a,"
              "d͡d=d͡ʒ d,d͡ʃ=d͡ʒ ʃ,r͡z=r z,u͡z=u d͡z,aː=a b")
DROP_WORDS = "هں,یٰاَیُّهَا"


def find_pairs(sources):
    pairs = []
    for docx in sorted(Path(sources).glob("*.docx"), key=lambda p: (len(p.stem), p.stem)):
        audio = []
        for pattern in (f"{docx.stem}.mp3", f"{docx.stem}_*.mp3", f"{docx.stem}.mpeg", f"{docx.stem}.wav"):
            audio += sorted(docx.parent.glob(pattern))
        pairs.append((docx, audio[0] if audio else None))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True, help="folder with <n>.docx + <n>[_crop].mp3 pairs")
    ap.add_argument("--out-dir", default="pashto_dataset", help="dataset folder to write")
    ap.add_argument("--g2p-repo", default=".", help="cloned nemo-pashto-g2p-conformer folder")
    ap.add_argument("--skip-lexicon", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def run(cmd):
        print("\n$ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
        if not args.dry_run and subprocess.run([str(c) for c in cmd]).returncode != 0:
            sys.exit("command failed")

    out = Path(args.out_dir)
    py = sys.executable
    for docx, audio in find_pairs(args.sources):
        if audio is None:
            print(f"!! {docx.name}: no audio file found next to it, skipping")
            continue
        print(f"\n===== {docx.name} + {audio.name} =====")
        tg = docx.with_suffix(".TextGrid")
        if not tg.exists():
            run([py, HERE / "docx_to_alignment.py", "--docx", docx, "--audio", audio,
                 "--out", tg, "--g2p-repo", args.g2p_repo])
        if not (out / "clips" / f"{audio.stem}-001.mp3").exists():
            run([py, HERE / "make_chunks_metadata.py", "--docx", docx, "--audio", audio,
                 "--textgrid", tg, "--out-dir", out, "--g2p-repo", args.g2p_repo])

    if not args.skip_lexicon:
        run([py, HERE / "make_lexicon.py", "--dataset", out, "--out", out / "lexicon" / "pashto-lexicon.txt",
             "--g2p-repo", args.g2p_repo, "--map-phones", MAP_PHONES, "--drop-words", DROP_WORDS])
    print(f"\nDone. Copy {out}/clips, metadata.csv, segments.csv and lexicon/ into the repo's data/ folder.")


if __name__ == "__main__":
    main()
