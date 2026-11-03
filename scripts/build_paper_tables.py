"""Paper tables from experiments/<ID>/eval (Colab runs of colab_paper.ipynb) -> reports/paper/{tables,figures}. CPU only.

    python scripts/build_paper_tables.py [--probe-name probe] [--boot 2000]

A  leakage study: WER of each clean/leaky pair on probe_seen and probe_unseen, the gap clean - leaky, and the
   difference-in-differences (gap on seen minus gap on unseen); 95% CIs from a paired bootstrap over videos.
B  official ViMedCSS TEST/HARD: WER, CER, CS-WER, term recall (+ VIVOS/VietMed/FLEURS TEST WER) for every system,
   next to the paper's numbers, and paired video-bootstrap CIs for the ablation contrasts.
   A_balance.csv compares what the two TRAINs and the two probe groups contain (no model needed).
C  term recall by TRAIN frequency for O3W and C1 with each term list, as two bounds (substring / whole word), terms
   generated but not spoken, and paired CIs against the same model without a list. "unseen" = not annotated in
   ViMedCSS TRAIN but present in some TRAIN transcript; "unseen_any" = in no TRAIN transcript of the four sources (the
   backbone's own training data cannot be checked).
Rows whose predictions do not exist yet are left out, so the script runs at any stage.
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import cs_error_counts, lexicon_counts, normalize_text, split_terms, term_matches  # noqa: E402

EXP, DATA = ROOT / "experiments", ROOT / "data"
OUT = ROOT / "reports/paper"
TAB, FIG = OUT / "tables", OUT / "figures"
PAPER = {  # arXiv 2602.12911 v2 (read 2026-10-30): WER, CER, CS-WER; same values as build_same_metric_table.py
    "Paper · PhoWhisper-small zero-shot": dict(TEST=(36.31, 22.64, 62.55), HARD=(46.27, 28.11, 66.01)),
    "Paper · PhoWhisper-large zero-shot": dict(TEST=(31.24, 19.25, 55.05), HARD=(37.37, 23.02, 57.37)),
    "Paper · Whisper-large-v3 zero-shot": dict(TEST=(34.47, 24.61, 46.69), HARD=(39.35, 27.46, 50.10)),
    "Paper · PhoWhisper-small + LoRA": dict(TEST=(27.13, 18.01, 30.26), HARD=(37.27, 24.55, 60.71)),
    "Paper · PhoWhisper-small + AG (best)": dict(TEST=(23.67, 14.73, 19.50), HARD=(33.73, 21.48, 57.29)),
}
OFFICIAL = [  # label, experiment, eval suffix ("" greedy; "_beam5"), description
    ("B0", "B0_phowhisper_small_zeroshot", "", "PhoWhisper-small zero-shot"),
    ("B2", "B2_phowhisper_large_zeroshot", "", "PhoWhisper-large zero-shot"),
    ("B3", "B3_whisper_large_v3_zeroshot", "", "Whisper-large-v3 zero-shot"),
    ("D1", "D1_small_lora_vimedcss_official", "", "small + LoRA, ViMedCSS only (paper setting)"),
    ("D2", "D2_small_lora_vimedcss_vietmed_official", "", "small + LoRA, ViMedCSS + VietMed"),
    ("D3", "D3_small_lora_all4_official", "", "small + LoRA, 4 sources"),
    ("D4", "D4_whisper_large_v3_lora_vimedcss_official", "", "large-v3 + LoRA, ViMedCSS only"),
    ("O2", "O2_whisper_large_v3_lora_official", "", "large-v3 + LoRA, 4 sources"),
    ("O2R", "O2R_whisper_large_v3_lora_official_randos", "", "O2 + as many random clips repeated as O3W (same steps)"),
    ("O3W", "O3W_whisper_large_v3_lora_official_termos", "", "O2 + rare-term oversampling"),
    ("O3W-b5", "O3W_whisper_large_v3_lora_official_termos", "_beam5", "O3W, beam 5 (frozen final system)"),
    ("C1", "C1_whisper_large_v3_lora_official_termos_prompt", "", "O3W + term-prompt training, no list at test"),
]
CONTRASTS = [("D1", "D3", "extra sources, small"), ("D4", "O2", "extra sources, large-v3"), ("D3", "O2", "backbone small -> large-v3"),
             ("O2", "O3W", "oversampling recipe (2.8% more steps)"), ("O2R", "O3W", "rare-term targeting, same steps"),
             ("O3W", "C1", "term-prompt training (no list at test)")]
TERM_LISTS = ["oracle10", "oracle30", "oracle50", "distract30", "video", "retrieve30", "retrieve30-mesh2026"]  # eval name after _bias-


def pairs():
    for b, seeds in [("small", [13, 14, 15]), ("medium", [13]), ("largev3", [13])]:
        for s in seeds:
            yield b, s, f"L_{b}_clean_s{s}", f"L_{b}_leak_s{s}"


_groups = {}


def group_of(ids):
    """id -> source video / recording (group_id), read once from the parquets that can be evaluated."""
    if not _groups:
        for p in DATA.glob("*/*.parquet"):
            if p.stem != "train":
                t = pq.read_table(p, columns=["id", "group_id"])
                _groups.update(zip(t.column("id").to_pylist(), t.column("group_id").to_pylist()))
    return [_groups[i] for i in ids]


def preds(exp, name, source_prefix=None):
    p = EXP / exp / "eval" / name / "predictions.jsonl"
    if not p.is_file():  # laptop runs: local/runs/<exp>/<run-id>/eval/<name>
        found = sorted((ROOT / "local/runs" / exp).glob(f"*/eval/{name}/predictions.jsonl"))
        assert len(found) <= 1, found
        if not found:
            return None
        p = found[0]
    df = pd.read_json(p, lines=True, dtype={"id": str})
    if source_prefix:
        df = df[df.source.str.startswith(source_prefix)]
    df = df[df.text.map(normalize_text) != ""].copy()  # jiwer skips empty references too
    df["group"] = group_of(df.id)
    m = df.apply(lambda r: jiwer.process_words(normalize_text(r.text), normalize_text(r.hyp)), axis=1)
    df["err"] = [w.substitutions + w.deletions + w.insertions for w in m]
    df["words"] = [w.hits + w.substitutions + w.deletions for w in m]
    return df


def wer(df):
    return 100 * df.err.sum() / df.words.sum()


def boot(per_group_sets, stat, n, seed=13):
    """Bootstrap over groups. per_group_sets: list of (G_k x C) arrays of per-group sums, resampled independently
    (one array per stratum); stat maps the list of resampled column sums to a number. Returns (low, high) 95%."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(stat([a[rng.integers(0, len(a), len(a))].sum(axis=0) for a in per_group_sets]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def leak_study(n_boot, probe):
    rows, small_strata = [], []
    for b, seed, c_exp, l_exp in pairs():
        c, l = preds(c_exp, probe), preds(l_exp, probe)
        if c is None or l is None:
            continue
        m = c.merge(l[["id", "err"]], on="id", suffixes=("_c", "_l"))
        assert len(m) == len(c) == len(l), "clean and leaky runs scored different clips"
        strata = {}
        for sub in ("seen", "unseen"):
            s = m[m.source == f"ViMedCSS_probe_{sub}"]
            strata[sub] = s.groupby("group")[["err_c", "err_l", "words"]].sum().to_numpy(float)
        gap = lambda a: 100 * (a[0] - a[1]) / a[2]
        r = dict(backbone=b, seed=seed)
        for sub, a in strata.items():
            tot = a.sum(axis=0)
            r |= {f"{sub}_clips": int((m.source == f"ViMedCSS_probe_{sub}").sum()), f"{sub}_videos": len(a),
                  f"{sub}_wer_clean": 100 * tot[0] / tot[2], f"{sub}_wer_leak": 100 * tot[1] / tot[2], f"{sub}_gap": gap(tot)}
            r[f"{sub}_gap_lo"], r[f"{sub}_gap_hi"] = boot([a], lambda s: gap(s[0]), n_boot)
        r["did"] = r["seen_gap"] - r["unseen_gap"]
        r["did_lo"], r["did_hi"] = boot([strata["seen"], strata["unseen"]], lambda s: gap(s[0]) - gap(s[1]), n_boot)
        r["seen_rel_inflation_pct"] = 100 * r["seen_gap"] / r["seen_wer_clean"]
        rows.append(r)
        if b == "small":
            small_strata.append(strata)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    small = df[df.backbone == "small"]
    if len(small) > 1:
        # Same videos in every seed, resampled together: CI of the mean over seeds. Seed spread is reported next to it,
        # because the bootstrap covers the sampling of videos only, not the training run.
        joint = [np.hstack([x[sub] for x in small_strata]) for sub in ("seen", "unseen")]
        k = range(0, joint[0].shape[1], 3)
        did = lambda s: np.mean([gap(s[0][i:i + 3]) - gap(s[1][i:i + 3]) for i in k])
        lo, hi = boot(joint, did, n_boot)
        points = [c for c in small.columns if c not in ("backbone", "seed") and not c.endswith(("_lo", "_hi"))]
        df = pd.concat([df, pd.DataFrame([dict(backbone="small", seed=f"pooled ({len(small)} seeds)") | small[points].mean().to_dict()
                                          | dict(did_lo=lo, did_hi=hi, did_sd_seeds=small.did.std())])])
    df.round(3).to_csv(TAB / "A_leak_study.csv", index=False)

    one = df[df.seed.map(lambda s: isinstance(s, (int, np.integer)))].groupby("backbone", sort=False).first().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, sub, title in zip(ax, ("seen", "unseen"), ("probe_seen: video có clip trong Train rò rỉ", "probe_unseen: đối chứng")):
        x = np.arange(len(one))
        a.bar(x - .2, one[f"{sub}_wer_clean"], .4, label="Train sạch", color="#4C78A8")
        a.bar(x + .2, one[f"{sub}_wer_leak"], .4, label="Train rò rỉ (cùng số clip)", color="#E45756")
        for i, r in one.iterrows():
            a.text(i, max(r[f"{sub}_wer_clean"], r[f"{sub}_wer_leak"]) + .4, f"{r[f'{sub}_gap']:+.2f}", ha="center", fontsize=9)
        a.set_xticks(x, one.backbone)
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("WER %")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rò rỉ video: cùng tập chấm, hai bộ Train chỉ khác nguồn gốc 722 clip (số trên cột: sạch − rò rỉ)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "A_leak_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def official_tables(n_boot):
    rows, by_src, cache = [], [], {}
    for split in ("TEST", "HARD"):
        for name, (t, c, cs) in [(k, v[split]) for k, v in PAPER.items()]:
            rows.append(dict(split=split, system=name, wer=t, cer=c, cs_wer=cs))
        for label, exp, sfx, desc in OFFICIAL:
            df = preds(exp, split.lower() + sfx, "ViMedCSS")
            if df is None:
                continue
            cache[(split, label)] = df
            cs = [cs_error_counts(r.text, r.hyp, r.cs_terms) for r in df.itertuples()]
            refs, hyps = [normalize_text(x) for x in df.text], [normalize_text(x) for x in df.hyp]
            hits = [x for r in df.itertuples() for x in term_matches(r.hyp, r.cs_terms)]
            lo, hi = boot([df.groupby("group")[["err", "words"]].sum().to_numpy(float)], lambda s: 100 * s[0][0] / s[0][1], n_boot)
            rows.append(dict(split=split, system=f"{label} · {desc}", wer=wer(df), wer_lo=lo, wer_hi=hi,
                             cer=100 * jiwer.cer(refs, hyps),
                             cs_wer=100 * sum(x["cs_err"] for x in cs) / sum(x["cs_words"] for x in cs),
                             term_recall=100 * sum(x[1] for x in hits) / len(hits),
                             term_recall_word=100 * sum(x[2] for x in hits) / len(hits), n_clips=len(df)))
    for label, exp, sfx, desc in OFFICIAL:
        p = EXP / exp / "eval" / f"test{sfx}" / "metrics.json"
        if p.is_file():
            m = json.loads(p.read_text(encoding="utf-8"))["by_source"]
            by_src.append(dict(system=label, description=desc, **{f"wer_{s}": 100 * v["wer"] for s, v in m.items()}))
    pd.DataFrame(rows).round(3).to_csv(TAB / "B_official_vimedcss.csv", index=False)
    pd.DataFrame(by_src).round(3).to_csv(TAB / "B_official_test_by_source.csv", index=False)

    con = []
    for split in ("TEST", "HARD"):
        for a, b, what in CONTRASTS:
            if (split, a) not in cache or (split, b) not in cache:
                continue
            m = cache[(split, a)].merge(cache[(split, b)][["id", "err"]], on="id", suffixes=("_a", "_b"))
            assert len(m) == len(cache[(split, a)]) == len(cache[(split, b)]), f"{a} and {b} scored different clips on {split}"
            g = m.groupby("group")[["err_a", "err_b", "words"]].sum().to_numpy(float)
            d = lambda s: 100 * (s[0][0] - s[0][1]) / s[0][2]
            lo, hi = boot([g], d, n_boot)
            tot = g.sum(axis=0)
            con.append(dict(split=split, contrast=f"{a} -> {b}", what=what, wer_a=100 * tot[0] / tot[2], wer_b=100 * tot[1] / tot[2],
                            gain=d([tot]), gain_lo=lo, gain_hi=hi))
    pd.DataFrame(con).round(3).to_csv(TAB / "B_contrasts.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(con)


def read_text(path):
    return [" " + normalize_text(x) + " " for x in pq.read_table(path, columns=["text"]).column("text").to_pylist()]


def term_buckets():
    """term -> frequency bucket; unseen terms are checked against every TRAIN transcript of the four sources."""
    counts = {}
    for s in pq.read_table(DATA / "ViMedCSS/train.parquet", columns=["cs_terms"]).column("cs_terms").to_pylist():
        for t in split_terms(s):
            counts[t] = counts.get(t, 0) + 1
    blob = "\n".join(x for src in ("ViMedCSS", "VietMed", "VIVOS", "FLEURS") for x in read_text(DATA / src / "train.parquet"))
    memo = {}

    def bucket(t):
        if t in counts:
            return "1-5" if counts[t] <= 5 else ">5"
        if t not in memo:
            memo[t] = "unseen" if f" {t} " in blob else "unseen_any"
        return memo[t]
    return bucket


def clip_terms(df, bucket, vocab):
    """Per clip: whole-word hits of unseen terms and terms generated beyond the reference. vocab = every annotated term
    of ViMedCSS + the clip's own list, so it covers terms a prompt could have pushed in."""
    d = df[["id", "group", "err", "words"]].copy()
    hit, n, false = [], [], []
    for r in df.itertuples():
        m = [x for x in term_matches(r.hyp, r.cs_terms) if bucket(x[0]).startswith("unseen")]
        hit.append(sum(x[2] for x in m))
        n.append(len(m))
        lex = vocab | set(getattr(r, "prompt_terms", []) or [])
        ref = lexicon_counts(r.text, lex)
        false.append(sum(max(0, c - ref.get(t, 0)) for t, c in lexicon_counts(r.hyp, lex).items()))
    d["hit"], d["n"], d["false"] = hit, n, false
    return d


def term_tables(n_boot):
    bucket = term_buckets()
    vocab = {t for split in ("train", "dev", "test", "hard")
             for s in pq.read_table(DATA / f"ViMedCSS/{split}.parquet", columns=["cs_terms"]).column("cs_terms").to_pylist()
             for t in split_terms(s)}
    rows = []
    for label, exp in [("O3W", "O3W_whisper_large_v3_lora_official_termos"), ("C1", "C1_whisper_large_v3_lora_official_termos_prompt")]:
        for split in ("test", "hard"):
            distract = preds(exp, f"{split}_bias-distract30_ViMedCSS")
            base = preds(exp, split, "ViMedCSS")
            base_c = clip_terms(base, bucket, vocab) if base is not None else None
            for lst, name in [("none", split)] + [(x, f"{split}_bias-{x}_ViMedCSS") for x in TERM_LISTS]:
                df = base if lst == "none" else preds(exp, name, "ViMedCSS")
                if df is None:
                    continue
                keys = ("unseen_any", "unseen", "1-5", ">5")
                hit = {k: [0, 0, 0] for k in keys}
                for r in df.itertuples():
                    for t, sub, word in term_matches(r.hyp, r.cs_terms):
                        hit[bucket(t)][0] += sub
                        hit[bucket(t)][1] += word
                        hit[bucket(t)][2] += 1
                # false insertions: listed distractors (distract30 lists) present in the hypothesis but not in the reference;
                # for the unprompted decode the same lists measure the base rate
                fa = n_listed = 0
                if distract is not None:
                    lists = dict(zip(distract.id, distract.prompt_terms))
                    for r in df.itertuples():
                        for t in lists.get(r.id, []):
                            n_listed += 1
                            fa += t in normalize_text(r.hyp) and t not in normalize_text(r.text)
                cs = [cs_error_counts(r.text, r.hyp, r.cs_terms) for r in df.itertuples()]
                listed = golds = 0  # how often the list held the clip's term: the ceiling a term list can reach
                got = [[0, 0], [0, 0]]  # whole-word hits of terms the list held / did not hold
                if "prompt_terms" in df:
                    for r in df.itertuples():
                        for t, _, word in term_matches(r.hyp, r.cs_terms):
                            golds, listed = golds + 1, listed + (t in r.prompt_terms)
                            got[t not in r.prompt_terms][0] += word
                            got[t not in r.prompt_terms][1] += 1
                cur = clip_terms(df, bucket, vocab)
                row = dict(system=label, split=split.upper(), term_list=lst, wer=wer(df),
                           cs_wer=100 * sum(x["cs_err"] for x in cs) / sum(x["cs_words"] for x in cs),
                           gold_in_list=100 * listed / golds if golds else None,
                           recall_word_listed=100 * got[0][0] / got[0][1] if got[0][1] else None,
                           recall_word_unlisted=100 * got[1][0] / got[1][1] if got[1][1] else None,
                           **{f"recall_{k}": 100 * v[0] / v[2] if v[2] else None for k, v in hit.items()},
                           **{f"recall_word_{k}": 100 * v[1] / v[2] if v[2] else None for k, v in hit.items()},
                           **{f"n_{k}": v[2] for k, v in hit.items()},
                           false_terms_per_100_clips=100 * cur["false"].sum() / len(cur),
                           false_insert_per_100_listed=100 * fa / n_listed if n_listed else None)
                if lst != "none" and base_c is not None:  # paired over videos against the same model without a list
                    m = cur.merge(base_c, on="id", suffixes=("", "_b"))
                    assert len(m) == len(cur) == len(base_c), f"{lst} and the unprompted decode scored different clips"
                    g = m.groupby("group")[["hit", "hit_b", "n", "err", "err_b", "words", "false", "false_b"]].sum().to_numpy(float)
                    d_recall = lambda s: 100 * (s[0][0] - s[0][1]) / s[0][2]
                    d_wer = lambda s: 100 * (s[0][3] - s[0][4]) / s[0][5]
                    tot = [g.sum(axis=0)]
                    lo, hi = boot([g], d_recall, n_boot)
                    wlo, whi = boot([g], d_wer, n_boot)
                    row |= dict(d_recall_word_unseen=d_recall(tot), d_recall_lo=lo, d_recall_hi=hi, d_wer=d_wer(tot), d_wer_lo=wlo, d_wer_hi=whi,
                                d_false_per_100_clips=100 * (tot[0][6] - tot[0][7]) / len(m))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(TAB / "C_term_lists.csv", index=False)
    return df


def leak_balance():
    """What the 722 added and 722 dropped clips hold, and text/term overlap of each TRAIN with the probe groups."""
    cols = ["id", "text", "cs_terms", "duration"]
    vs = pq.read_table(DATA / "ViMedCSS_vsplit/train.parquet", columns=cols).to_pylist()
    added = pq.read_table(DATA / "ViMedCSS_leak/train.parquet", columns=cols).to_pylist()
    probe = pq.read_table(DATA / "ViMedCSS_leak/probe.parquet", columns=cols + ["source"]).to_pylist()
    excl = json.loads((DATA / "leak_train_exclude.json").read_text(encoding="utf-8"))["ids"]
    dropped = [r for r in vs if excl.get(r["id"]) == "leak-study size match"]
    leaky = [r for r in vs if r["id"] not in excl] + added
    clean = [r for r in vs if r["id"] not in excl] + dropped
    texts = lambda rows: {normalize_text(r["text"]) for r in rows}
    terms = lambda rows: {t for r in rows for t in split_terms(r["cs_terms"])}
    clean_text, leaky_text, clean_terms, leaky_terms = texts(clean), texts(leaky), terms(clean), terms(leaky)

    def row(name, rows, **extra):
        w = [len(normalize_text(r["text"]).split()) for r in rows]
        secs = sum(r["duration"] for r in rows)
        occ = [t for r in rows for t in split_terms(r["cs_terms"])]
        return dict(group=name, clips=len(rows), hours=secs / 3600, words_per_clip=np.mean(w), words_per_sec=sum(w) / secs,
                    term_occ_per_clip=len(occ) / len(rows), distinct_terms=len(set(occ))) | extra

    out = [row("added to leaky TRAIN", added), row("dropped from leaky TRAIN", dropped)]
    for sub in ("seen", "unseen"):
        p = [r for r in probe if r["source"] == f"ViMedCSS_probe_{sub}"]
        occ = [t for r in p for t in split_terms(r["cs_terms"])]
        ts = [normalize_text(r["text"]) for r in p]
        out.append(row(f"probe_{sub}", p, same_text_in_clean=sum(t in clean_text for t in ts), same_text_in_leaky=sum(t in leaky_text for t in ts),
                       term_occ_in_clean_vocab=sum(t in clean_terms for t in occ), term_occ_in_leaky_vocab=sum(t in leaky_terms for t in occ)))
    df = pd.DataFrame(out)
    df.round(3).to_csv(TAB / "A_balance.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-name", default="probe", help="eval name of the probe decode (e.g. probe_sub20 for a dry run)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--leak-only", action="store_true", help="part A only (interim look at the leakage study)")
    args = ap.parse_args()
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("A balance\n", leak_balance().round(2).to_string(index=False))
    a = leak_study(args.boot, args.probe_name)
    print("A leakage study\n", a.round(2).to_string(index=False) if not a.empty else "(no probe decodes yet)")
    if args.leak_only:
        return
    b, con = official_tables(args.boot)
    print("\nB official ViMedCSS\n", b.round(2).to_string(index=False))
    print("\nB contrasts (gain = WER a - WER b)\n", con.round(2).to_string(index=False) if not con.empty else "(none yet)")
    c = term_tables(args.boot)
    print("\nC term lists\n", c.round(2).to_string(index=False) if not c.empty else "(no biased decodes yet)")


if __name__ == "__main__":
    main()
