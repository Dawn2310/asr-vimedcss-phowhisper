"""Fixed external medical lexicon for `evaluate.py --bias retrieve --lexicon`: NLM MeSH 2026 descriptor terms, chosen
before any biased decode of TEST/HARD and independent of the evaluated splits.

    python scripts/build_lexicon.py [--source desc2026.gz]

Keeps normalized terms of at most 3 words without a comma, plus their single words of at least 4 letters ('Waldeyer's
ring' -> 'waldeyer'), a-z only. Writes data/lexicons/mesh2026.txt and prints its sha256.
"""
import argparse
import gzip
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import normalize_text

URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
ENTRY = re.compile(r"[a-z ]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="local desc2026.gz (default: download from NLM)")
    args = ap.parse_args()
    path = args.source
    if path is None:
        path, _ = urllib.request.urlretrieve(URL)
    terms, term_ui = set(), False
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if term_ui and "<String>" in line:
                s = line.split("<String>", 1)[1].split("</String>", 1)[0]
                norm = normalize_text(s)
                entries = [w for w in norm.split() if len(w) >= 4 and w.isalpha()]
                if "," not in s and "&amp;" not in s:
                    entries.append(norm)
                terms.update(e for e in entries if e and len(e.split()) <= 3 and ENTRY.fullmatch(e))
            term_ui = "<TermUI>" in line
    out = ROOT / "data/lexicons/mesh2026.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sorted(terms)) + "\n", encoding="utf-8")
    print(out, len(terms), "terms, sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
