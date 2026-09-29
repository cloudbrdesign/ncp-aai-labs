"""Step 1: clean, filter and deduplicate the product manuals before they are indexed.

    python m04/clean.py              # reads m04/data/manuals/, writes m04/state/clean/
    python m04/clean.py --jaccard    # also print the most similar pairs of documents

The same stages as a NeMo Curator text pipeline, in plain Python for a handful of files:

  1. normalise   Unicode NFKC (full-width "Ｅ７３" becomes "E73", the "ﬁ" ligature becomes
                 "fi", non-breaking spaces become spaces), trailing spaces and extra blank
                 lines removed, boilerplate lines ("Page 2 of 4", "Printed copy: ...") dropped.
                 Without this, a keyword search for E73 never finds "Ｅ７３".
  2. filter      heuristic quality filters, no model needed: at least MIN_WORDS words (the
                 idea of Curator's WordCountFilter) and at most MAX_SYMBOLS of the
                 non-space characters that are not letters or digits (the idea of its
                 NonAlphaNumericFilter). Catches near-empty pages and garbled scans.
  3. exact dedup MD5 of the normalised text. Same hash = same document; keep the first.
  4. near dedup  character 5-gram shingles and Jaccard similarity (shared shingles / all
                 shingles). Over NEAR_DUP_THRESHOLD, two documents are near-duplicates, such
                 as two revisions of one manual; we keep the newer revision.
                 Curator does fuzzy dedup at scale with MinHash + LSH on a GPU, which
                 estimates the same Jaccard similarity without comparing every pair.

Duplicates matter for retrieval: two copies of a passage take two of the top-3 places,
and the answer gets less evidence. See it with: python m04/ingest.py --raw (README step 4).
"""
import argparse
import hashlib
import itertools
import os
import pathlib
import re
import shutil
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "manuals"
STATE_DIR = pathlib.Path(os.environ.get("M04_STATE_DIR", HERE / "state"))
CLEAN_DIR = STATE_DIR / "clean"

MIN_WORDS = 50              # Curator's WordCountFilter uses min_words=50 by default
MAX_SYMBOLS = 0.25          # share of non-space characters that are not letters or digits
SHINGLE = 5                 # characters per shingle
NEAR_DUP_THRESHOLD = 0.8    # Jaccard similarity at or above this = near-duplicate
BOILERPLATE = re.compile(r"^(Page \d+ of \d+|Printed copy:.*)$", re.I)
REVISION = re.compile(r"^Revision: (\S+)", re.M)


def normalise(text: str) -> tuple[str, list[str], int]:
    """Return (clean text, the characters NFKC changed, boilerplate lines removed)."""
    changed = sorted({c for c in text if unicodedata.normalize("NFKC", c) != c})
    text = unicodedata.normalize("NFKC", text)
    lines = [l.rstrip() for l in text.splitlines()]
    kept = [l for l in lines if not BOILERPLATE.match(l.strip())]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip() + "\n"
    return text, changed, len(lines) - len(kept)


def show_char(c: str) -> str:
    n = unicodedata.normalize("NFKC", c)
    return f"{'NBSP' if c == chr(0xA0) else c}->{'space' if n == ' ' else n}"


def quality_problems(text: str) -> list[str]:
    """The heuristic filters. An empty list means the document passes."""
    words = len(text.split())
    chars = [c for c in text if not c.isspace()]
    symbols = sum(1 for c in chars if not c.isalnum()) / max(len(chars), 1)
    found = []
    if words < MIN_WORDS:
        found.append(f"too short ({words} words < {MIN_WORDS})")
    if symbols > MAX_SYMBOLS:
        found.append(f"garbled ({symbols:.0%} symbols > {MAX_SYMBOLS:.0%})")
    return found


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def revision(text: str) -> str:
    m = REVISION.search(text)
    return m.group(1) if m else ""


def clean(raw_dir: pathlib.Path = RAW_DIR, out_dir: pathlib.Path = CLEAN_DIR, show_pairs: bool = False,
          say=print) -> dict:
    """Run the four stages. Returns {"kept": [names], "dropped": {name: reason}}."""
    docs, dropped = {}, {}
    files = sorted(raw_dir.glob("*.md"))
    say(f"[INFO] {len(files)} files in {raw_dir.relative_to(HERE.parent)}")

    say("== 1. Normalise")
    for f in files:
        text, changed, boiler = normalise(f.read_text(encoding="utf-8"))
        docs[f.name] = text
        notes = []
        if changed:
            notes.append("NFKC " + ", ".join(show_char(c) for c in changed))
        if boiler:
            notes.append(f"{boiler} boilerplate line{'s' if boiler > 1 else ''} removed")
        say(f"[normalise] {f.name}: {'; '.join(notes) or 'no change'}")

    say("== 2. Quality filters")
    for name in list(docs):
        problems = quality_problems(docs[name])
        if problems:
            dropped[name] = "; ".join(problems)
            say(f"[filter] DROP {name}: {dropped[name]}")
            del docs[name]
    say(f"[filter] {len(docs)} documents pass (min {MIN_WORDS} words, max {MAX_SYMBOLS:.0%} symbols)")

    say("== 3. Exact dedup (MD5 of the normalised text)")
    seen = {}
    for name in list(docs):
        h = md5(docs[name])
        if h in seen:
            dropped[name] = f"exact duplicate of {seen[h]}"
            say(f"[exact] DROP {name}: same MD5 as {seen[h]} ({h[:8]})")
            del docs[name]
        else:
            seen[h] = name
    say(f"[exact] {len(docs)} unique documents")

    say(f"== 4. Near dedup ({SHINGLE}-character shingles, Jaccard >= {NEAR_DUP_THRESHOLD})")
    sh = {name: shingles(text) for name, text in docs.items()}
    pairs = sorted(((jaccard(sh[a], sh[b]), a, b) for a, b in itertools.combinations(sh, 2)), reverse=True)
    if show_pairs:
        for score, a, b in pairs[:5]:
            say(f"[near-dup] {score:.2f}  {a} ~ {b}")
    for score, a, b in pairs:
        if score < NEAR_DUP_THRESHOLD or a not in docs or b not in docs:
            continue
        old, new = sorted([a, b], key=lambda n: revision(docs[n]))
        dropped[old] = f"near-duplicate of {new} (Jaccard {score:.2f}), older revision"
        say(f"[near-dup] DROP {old}: Jaccard {score:.2f} with {new}; "
            f"revision {revision(docs[old]) or '?'} < {revision(docs[new]) or '?'}")
        del docs[old]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, text in docs.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    say(f"[INFO] kept {len(docs)} of {len(files)}: {', '.join(docs)}")
    say(f"[INFO] wrote {out_dir.relative_to(HERE.parent)}/")
    return {"kept": list(docs), "dropped": dropped}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", action="store_true", help="print the 5 most similar document pairs")
    a = ap.parse_args()
    clean(show_pairs=a.jaccard)


if __name__ == "__main__":
    main()
