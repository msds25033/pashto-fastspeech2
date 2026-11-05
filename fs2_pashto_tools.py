#!/usr/bin/env python3
"""
Helpers for training ming024/FastSpeech2 on Pashto. Four sub-commands:

  patch          Make the repo work for Pashto and with current libraries:
                 append pashto_cleaners to text/cleaners.py, let train.py run on a
                 handful of clips, log the learning rate, add a Pashto ("ps")
                 branch and --name option to synthesize.py, load checkpoints on
                 CPU-only machines and with PyTorch >= 2.6. Safe to re-run.

  symbols        Replace text/symbols.py (English ARPAbet + pinyin) with your
                 Pashto phone set, then smoke-test text_to_sequence().
  modernize      Rewrite old-style librosa calls (positional arguments) into
                 the keyword form that current librosa requires.
  fix-textgrids  After Montreal Forced Aligner: label silences "sp" (newer MFA
                 leaves them empty, which breaks preprocess.py) and check that
                 every phone label is in your phone set.

Examples (Colab):
  python fs2_pashto_tools.py symbols --repo /content/FastSpeech2 \
      --phones /content/nemo-pashto-g2p-conformer/lexicon/pashto-lexicon.phones.txt
  python fs2_pashto_tools.py modernize --repo /content/FastSpeech2
  python fs2_pashto_tools.py fix-textgrids \
      --textgrid-dir /content/FastSpeech2/preprocessed_data/Pashto/TextGrid \
      --phones /content/nemo-pashto-g2p-conformer/lexicon/pashto-lexicon.phones.txt
"""

import argparse
import ast
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

SILENCES = ["sp", "spn", "sil"]  # what FastSpeech2's preprocessor treats as silence


def read_phones(path):
    """First column of the phones file written by make_lexicon.py."""
    phones = [ln.split("\t")[0] for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not phones:
        sys.exit(f"{path} has no phones")
    return phones


# ---------------------------------------------------------------------------
# symbols
# ---------------------------------------------------------------------------

SYMBOLS_TEMPLATE = '''""" Pashto phone set, written by fs2_pashto_tools.py from the lexicon's phone list.
The original English file is saved next to this one as symbols.py.orig. """

_pad = "_"
_special = "-"
_punctuation = "!'(),.:;? "
_silences = ["@sp", "@spn", "@sil"]

# "@" prefix = phone symbol (same convention as the ARPAbet symbols this replaces)
_pashto = ["@" + s for s in {phones}]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + _pashto + _silences
'''


def cmd_symbols(args):
    repo = Path(args.repo)
    target = repo / "text" / "symbols.py"
    if not target.exists():
        sys.exit(f"{target} not found - is --repo the FastSpeech2 folder?")
    phones = read_phones(args.phones)
    backup = target.with_name("symbols.py.orig")
    if not backup.exists():
        shutil.copy(target, backup)
        print(f"saved the original as {backup}")
    target.write_text(SYMBOLS_TEMPLATE.format(phones=json.dumps(phones, ensure_ascii=False)),
                      encoding="utf-8")
    print(f"wrote {target} with {len(phones)} Pashto phones + {len(SILENCES)} silence symbols")

    # Smoke test with the repo's own text_to_sequence and the Pashto cleaner.
    probe = "{" + " ".join(phones[:3] + ["sp"]) + "}"
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from text import text_to_sequence, cleaners\n"
        "from text.symbols import symbols\n"
        "assert hasattr(cleaners, 'pashto_cleaners'), 'pashto_cleaners missing from text/cleaners.py'\n"
        f"ids = text_to_sequence({probe!r}, ['pashto_cleaners'])\n"
        "print(len(symbols), 'symbols;', ids)\n"
        "assert len(ids) == 4 and len(set(ids)) == 4, 'a phone was dropped or duplicated'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        print("SMOKE TEST FAILED:\n" + (r.stderr or r.stdout)[-1500:])
        sys.exit(1)
    print("smoke test passed:", r.stdout.strip())


# ---------------------------------------------------------------------------
# modernize
# ---------------------------------------------------------------------------

# (matcher on the dotted call name, full positional parameter order, positional args to keep)
RULES = [
    (lambda n: n.endswith("librosa_mel_fn") or n.endswith("filters.mel"),
     ["sr", "n_fft", "n_mels", "fmin", "fmax", "htk", "norm", "dtype"], 0),
    (lambda n: n.endswith("librosa.load"),
     ["path", "sr", "mono", "offset", "duration", "dtype", "res_type"], 1),
    (lambda n: n.endswith("librosa.resample"),
     ["y", "orig_sr", "target_sr"], 1),
    # librosa >= 0.10 (and 1.0): pad_center(data, *, size) -- the call preprocess.py crashed on
    (lambda n: n.endswith("pad_center"),
     ["data", "size", "axis"], 1),
]


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def modernize_source(src):
    """-> (new_source, [(lineno, old, new), ...])"""
    tree = ast.parse(src)
    raw = src.encode("utf-8")
    line_start = [0]
    for ln in raw.split(b"\n")[:-1]:
        line_start.append(line_start[-1] + len(ln) + 1)

    def span(node):
        return (line_start[node.lineno - 1] + node.col_offset,
                line_start[node.end_lineno - 1] + node.end_col_offset)

    def seg(node):
        a, b = span(node)
        return raw[a:b].decode("utf-8")

    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func) or ""
        rule = next((r for r in RULES if r[0](name)), None)
        if not rule:
            continue
        _, order, keep = rule
        if len(node.args) <= keep or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if len(node.args) > len(order):
            continue
        have = {k.arg for k in node.keywords if k.arg}
        extra = {order[i] for i in range(keep, len(node.args))}
        if extra & have:
            continue
        parts = [seg(a) for a in node.args[:keep]]
        parts += [f"{order[i]}={seg(node.args[i])}" for i in range(keep, len(node.args))]
        parts += [(f"{k.arg}={seg(k.value)}" if k.arg else f"**{seg(k.value)}") for k in node.keywords]
        new = f"{seg(node.func)}({', '.join(parts)})"
        edits.append((span(node), node.lineno, seg(node), new))

    edits.sort(key=lambda e: e[0][0])
    applied, last_end = [], -1
    for (a, b), lineno, old, new in edits:          # outermost first, skip overlaps
        if a >= last_end:
            applied.append((a, b, lineno, old, new))
            last_end = b
    for a, b, _, _, new in reversed(applied):
        raw = raw[:a] + new.encode("utf-8") + raw[b:]
    return raw.decode("utf-8"), [(ln, " ".join(o.split()), n) for _, _, ln, o, n in applied]


def cmd_modernize(args):
    total = 0
    for path in sorted(Path(args.repo).rglob("*.py")):
        if ".git" in path.parts or path.name == Path(__file__).name:
            continue
        try:
            src = path.read_text(encoding="utf-8")
            new_src, changes = modernize_source(src)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"skipped {path}: {e}")
            continue
        for lineno, old, new in changes:
            print(f"{path.relative_to(args.repo)}:{lineno}\n    - {old}\n    + {new}")
        if changes and not args.dry_run:
            path.write_text(new_src, encoding="utf-8")
        total += len(changes)
    print(f"\n{total} call(s) {'would be ' if args.dry_run else ''}rewritten. "
          "(Undo any time with:  git -C <repo> checkout .)")


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------

PASHTO_SYNTH_FN = '''

def preprocess_pashto(text, preprocess_config):
    """Pashto text -> phone IDs, through the same word normalisation and lexicon
    the training data was aligned with. Words missing from the lexicon are skipped."""
    from text.cleaners import pashto_words

    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phones, unknown = [], []
    for w in pashto_words(text):
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            unknown.append(w)
    if unknown:
        print("WARNING: words not in the lexicon were skipped: {}".format(unknown))
    phones = "{" + " ".join(phones) + "}"

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)

'''

# (file, marker that shows the patch is already in, [(old, new), ...])
PATCHES = [
    ("train.py", "Pashto patch: tiny datasets", [(
        "    assert batch_size * group_size < len(dataset)\n",
        "    # Pashto patch: tiny datasets (a handful of clips) can't fill 4 groups of batches\n"
        "    group_size = max(1, min(group_size, len(dataset) // batch_size))\n"
        "    assert batch_size * group_size <= len(dataset), (\n"
        "        f\"batch_size {batch_size} is larger than the {len(dataset)} training utterances\")\n",
    ), (
        # also record the learning rate next to the losses
        "                    log(train_logger, step, losses=losses)\n",
        "                    log(train_logger, step, losses=losses)\n"
        "                    train_logger.add_scalar(\n"
        "                        \"Optim/learning_rate\", optimizer._optimizer.param_groups[0][\"lr\"], step\n"
        "                    )\n",
    )]),
    ("audio/stft.py", "Pashto patch: CPU fallback", [
        ("from librosa.filters import mel as librosa_mel_fn\n",
         "from librosa.filters import mel as librosa_mel_fn\n\n"
         "_DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")  "
         "# Pashto patch: CPU fallback\n"),
        ("            input_data.cuda(),\n"
         "            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),\n",
         "            input_data.to(_DEVICE),\n"
         "            torch.autograd.Variable(self.forward_basis, requires_grad=False).to(_DEVICE),\n"),
    ]),
    # map_location: CPU-only machines. weights_only=False: PyTorch >= 2.6 refuses the
    # numpy scalars inside FastSpeech2's own checkpoints (needed for synthesis and --restore_step).
    ("utils/model.py", "weights_only=False", [
        ("ckpt = torch.load(ckpt_path)",
         "ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)"),
        ('torch.load("hifigan/generator_LJSpeech.pth.tar")',
         'torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)'),
        ('torch.load("hifigan/generator_universal.pth.tar")',
         'torch.load("hifigan/generator_universal.pth.tar", map_location=device)'),
    ]),
    ("synthesize.py", "def preprocess_pashto", [
        ("    with open(lex_path) as f:", '    with open(lex_path, encoding="utf-8") as f:'),
        ("\n\ndef synthesize(", PASHTO_SYNTH_FN + "\ndef synthesize("),
        ('        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":\n'
         '            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])\n',
         '        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":\n'
         '            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])\n'
         '        elif preprocess_config["preprocessing"]["text"]["language"] == "ps":\n'
         '            texts = np.array([preprocess_pashto(args.text, preprocess_config)])\n'),
        ("        ids = raw_texts = [args.text[:100]]\n",
         "        raw_texts = [args.text[:100]]\n"
         "        ids = [args.name] if args.name else raw_texts\n"),
        ('    parser.add_argument(\n        "--speaker_id",',
         '    parser.add_argument(\n        "--name",\n        type=str,\n        default=None,\n'
         '        help="output file name (without .wav) for single mode; default: the text",\n'
         '    )\n    parser.add_argument(\n        "--speaker_id",'),
    ]),
]


def cmd_patch(args):
    repo = Path(args.repo)
    snippet = Path(args.cleaners_snippet)
    cleaners = repo / "text" / "cleaners.py"
    src = cleaners.read_text(encoding="utf-8")
    if "def pashto_cleaners" in src:
        print("text/cleaners.py: already has pashto_cleaners")
    else:
        cleaners.write_text(src.rstrip("\n") + "\n\n" + snippet.read_text(encoding="utf-8"),
                            encoding="utf-8")
        print(f"text/cleaners.py: appended {snippet.name}")

    for rel, marker, edits in PATCHES:
        path = repo / rel
        src = path.read_text(encoding="utf-8")
        if marker in src:
            print(f"{rel}: already patched")
            continue
        for old, new in edits:
            if src.count(old) != 1:
                sys.exit(f"{rel}: expected exactly one occurrence of {old.strip()[:60]!r} "
                         f"(found {src.count(old)}). Is this the ming024/FastSpeech2 master branch?")
            src = src.replace(old, new)
        path.write_text(src, encoding="utf-8")
        print(f"{rel}: patched ({len(edits)} edit(s))")


# ---------------------------------------------------------------------------
# fix-textgrids
# ---------------------------------------------------------------------------

def cmd_fix_textgrids(args):
    try:
        import tgt
    except ImportError:
        sys.exit("needs the 'tgt' package:  pip install tgt")
    files = sorted(Path(args.textgrid_dir).rglob("*.TextGrid"))
    if not files:
        sys.exit(f"no .TextGrid files under {args.textgrid_dir}")
    allowed = set(read_phones(args.phones)) | set(SILENCES) if args.phones else None
    labels, fixed, changed_files, unknown_in = collections.Counter(), 0, 0, collections.defaultdict(list)
    for f in files:
        # tgt silently DROPS empty-label intervals unless asked not to - that is exactly
        # what newer MFA uses for silence, so they must be read in to be relabelled.
        tg = tgt.io.read_textgrid(str(f), include_empty_intervals=True)
        if args.tier not in tg.get_tier_names():
            sys.exit(f"{f.name} has no tier {args.tier!r}; it has {tg.get_tier_names()}")
        changed = False
        for iv in tg.get_tier_by_name(args.tier).intervals:
            if not iv.text.strip():
                iv.text, changed = args.silence, True
                fixed += 1
            labels[iv.text] += 1
            if allowed is not None and iv.text not in allowed and len(unknown_in[iv.text]) < 3:
                unknown_in[iv.text].append(f.name)
        if changed:
            tgt.io.write_to_file(tg, str(f), format="long")
            changed_files += 1
    print(f"{len(files)} TextGrids; {fixed} empty intervals relabelled '{args.silence}' in {changed_files} files")
    print(f"{len(labels)} distinct labels in the '{args.tier}' tier")
    if allowed is not None:
        bad = [l for l in labels if l not in allowed]
        if bad:
            print("\nWARNING: labels that are NOT in your phone set (preprocessing would break):")
            for l in bad:
                print(f"    {l!r}: {labels[l]} times, e.g. {unknown_in[l]}")
            sys.exit(1)
        print("all labels are in your phone set (plus sp/spn/sil): OK")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("symbols")
    s.add_argument("--repo", required=True, help="FastSpeech2 folder")
    s.add_argument("--phones", required=True, help="pashto-lexicon.phones.txt")
    s.set_defaults(fn=cmd_symbols)

    m = sub.add_parser("modernize")
    m.add_argument("--repo", required=True)
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(fn=cmd_modernize)

    p = sub.add_parser("patch")
    p.add_argument("--repo", required=True, help="FastSpeech2 folder")
    p.add_argument("--cleaners-snippet", default=str(Path(__file__).parent / "patches" / "pashto_cleaners.py"),
                   help="code appended to text/cleaners.py (defines pashto_cleaners)")
    p.set_defaults(fn=cmd_patch)

    t = sub.add_parser("fix-textgrids")
    t.add_argument("--textgrid-dir", required=True)
    t.add_argument("--phones", help="pashto-lexicon.phones.txt, to check every label")
    t.add_argument("--tier", default="phones")
    t.add_argument("--silence", default="sp")
    t.set_defaults(fn=cmd_fix_textgrids)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
