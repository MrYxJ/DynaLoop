"""Download every evaluation dataset into ``<repo>/data``.

DynaLoop never touches the network at evaluation time: each dataset is
fetched once, stored as parquet/jsonl, and read locally afterwards.  This
script is idempotent — existing files are skipped — and will try an
``hf-mirror.com`` URL before falling back to ``huggingface.co``, which
matters a lot if you are behind the Great Firewall.

    python scripts/fetch_datasets.py          # all four
    python scripts/fetch_datasets.py gsm8k    # or just one

Set ``DYNALOOP_DATA_DIR`` to store the data somewhere else (for example on
a shared disk): the loader in ``dynaloop/eval/datasets.py`` honours the same
variable.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("DYNALOOP_DATA_DIR") or os.path.join(ROOT, "data")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _urls(*candidates: str) -> list:
    """Mirror-first ordering, with the official host as the last resort."""
    out = []
    for c in candidates:
        out.append(c)
    return out


# (target file, [urls], min expected size in bytes)
TARGETS = {
    "gsm8k": [(
        os.path.join("gsm8k", "test.parquet"),
        _urls("https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/"
              "test-00000-of-00001.parquet",
              "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/"
              "test-00000-of-00001.parquet"),
        300_000,
    )],
    "math500": [(
        os.path.join("math500", "test.jsonl"),
        _urls("https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/"
              "resolve/main/test.jsonl",
              "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
              "resolve/main/test.jsonl"),
        300_000,
    )],
    "arcc": [
        (os.path.join("allenai_ai2_arc_parquet_ARC-Challenge_test_0.parquet"),
         _urls("https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/test-00000-of-00001.parquet",
               "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/test-00000-of-00001.parquet"),
         150_000),
        (os.path.join("allenai_ai2_arc_parquet_ARC-Challenge_train_0.parquet"),
         _urls("https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/train-00000-of-00001.parquet",
               "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/train-00000-of-00001.parquet"),
         150_000),
        (os.path.join(
            "allenai_ai2_arc_parquet_ARC-Challenge_validation_0.parquet"),
         _urls("https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/validation-00000-of-00001.parquet",
               "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/"
               "ARC-Challenge/validation-00000-of-00001.parquet"),
         50_000),
    ],
    "hellaswag": [
        ("hs_train.parquet",
         _urls("https://hf-mirror.com/datasets/Rowan/hellaswag/resolve/main/"
               "data/train-00000-of-00001.parquet",
               "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/"
               "data/train-00000-of-00001.parquet"),
         20_000_000),
        ("hs_val.parquet",
         _urls("https://hf-mirror.com/datasets/Rowan/hellaswag/resolve/main/"
               "data/validation-00000-of-00001.parquet",
               "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/"
               "data/validation-00000-of-00001.parquet"),
         5_000_000),
    ],
}


def try_download(url: str, dest: str, expect: int, timeout: int = 300) -> bool:
    """Download ``url`` to ``dest``; verify it really is a parquet/jsonl."""
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print("    FAIL %s: %s" % (type(exc).__name__, exc))
        return False
    if not data:
        print("    FAIL empty payload")
        return False
    head_ok = data[:4] == b"PAR1" or data[:1] in (b"{", b"[")
    if not head_ok:
        print("    FAIL not parquet/jsonl, head=%r" % data[:60])
        return False
    if len(data) < expect:
        print("    FAIL truncated: %d bytes < %d expected"
              % (len(data), expect))
        return False
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    print("    OK %d bytes -> %s" % (len(data), dest))
    return True


def main() -> int:
    wanted = sys.argv[1:] or list(TARGETS)
    os.makedirs(DATA, exist_ok=True)
    rc = 0
    for key in wanted:
        if key not in TARGETS:
            print("[skip] unknown dataset %r" % key)
            continue
        for rel, urls, expect in TARGETS[key]:
            dest = os.path.join(DATA, rel)
            if os.path.exists(dest) and os.path.getsize(dest) >= expect:
                print("[skip] %s exists (%d bytes)"
                      % (rel, os.path.getsize(dest)))
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print("[get ] %s" % rel)
            for url in urls:
                print("    <- %s" % url)
                if try_download(url, dest, expect):
                    break
            else:
                print("[FAIL] %s: all URLs failed — if you are on a slow "
                      "link, retry the script; it resumes per file." % rel)
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
