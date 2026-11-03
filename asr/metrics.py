"""Text normalization and ASR metrics (WER/CER, code-switch term recall)."""
import re
import unicodedata

import jiwer

_NON_WORD = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """NFC, lowercase, punctuation -> space, collapse whitespace. Same rule for train targets and scoring."""
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


def split_terms(cs_terms: str) -> list[str]:
    """';'-separated cs_terms field -> normalized non-empty terms."""
    return [t for t in (normalize_text(x) for x in cs_terms.split(";")) if t]


def term_matches(hyp: str, cs_terms: str) -> list[tuple[str, bool, bool]]:
    """(term, substring hit, whole-word hit) per annotated occurrence. Substring also accepts spelling variants
    ('vaccin' in 'vaccine') and false matches, whole-word rejects both: they bracket the true recall."""
    h, words, out, seen = normalize_text(hyp), normalize_text(hyp).split(), [], {}
    for t in split_terms(cs_terms):
        w = t.split()
        seen[t] = seen.get(t, 0) + 1
        k = sum(words[i:i + len(w)] == w for i in range(len(words) - len(w) + 1))
        out.append((t, t in h, seen[t] <= k))
    return out


def lexicon_counts(text: str, lexicon: set[str], max_words: int = 3) -> dict:
    """Occurrences of lexicon entries (up to max_words words) as whole words of the normalized text."""
    words, out = normalize_text(text).split(), {}
    for n in range(1, max_words + 1):
        for i in range(len(words) - n + 1):
            g = " ".join(words[i:i + n])
            if g in lexicon:
                out[g] = out.get(g, 0) + 1
    return out


def cs_error_counts(ref: str, hyp: str, cs_terms: str) -> dict:
    """Word errors split into code-switched (inside an annotated cs_term) and other reference words, our reading of
    the ViMedCSS paper's CS-WER / N-WER (it gives no formula): jiwer alignment of normalized words; a substitution or
    deletion is charged to its reference word, an insertion to the reference word before it (the first if at the start)."""
    rt, ht = normalize_text(ref).split(), normalize_text(hyp).split()
    mask = [False] * len(rt)
    for t in (x.split() for x in split_terms(cs_terms)):
        for i in range(len(rt) - len(t) + 1):
            if rt[i:i + len(t)] == t:
                mask[i:i + len(t)] = [True] * len(t)
    out = dict(cs_err=0, cs_words=sum(mask), n_err=0, n_words=len(rt) - sum(mask))
    if not rt:
        return out
    for ch in jiwer.process_words(" ".join(rt), " ".join(ht)).alignments[0]:
        if ch.type in ("substitute", "delete"):
            for i in range(ch.ref_start_idx, ch.ref_end_idx):
                out["cs_err" if mask[i] else "n_err"] += 1
        elif ch.type == "insert":
            p = ch.ref_start_idx
            out["cs_err" if (mask[p - 1] if p > 0 else mask[0]) else "n_err"] += ch.hyp_end_idx - ch.hyp_start_idx
    return out


def compute_metrics(refs: list[str], hyps: list[str], cs_terms: list[str] | None = None) -> dict:
    refs = [normalize_text(r) for r in refs]
    hyps = [normalize_text(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]  # jiwer cannot score empty references
    r, h = [refs[i] for i in keep], [hyps[i] for i in keep]
    words = jiwer.process_words(r, h)
    out = {
        "n_utts": len(keep),
        "n_empty_refs_skipped": len(refs) - len(keep),
        "wer": words.wer,
        "cer": jiwer.cer(r, h),
        "ref_words": words.hits + words.substitutions + words.deletions,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
    }
    if cs_terms and any(cs_terms):
        # Approximate: a term counts as recognized if its normalized form occurs in the hypothesis.
        total = found = 0
        for terms, hyp in zip(cs_terms, hyps):
            for term in filter(None, (normalize_text(t) for t in terms.split(";"))):
                total += 1
                found += term in hyp
        out.update(cs_terms_total=total, cs_term_recall=found / total if total else None)
    return out
