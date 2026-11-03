"""Score our ViMedCSS predictions with the ViMedCSS paper's metric set (WER, CER, CS-WER, N-WER) and tabulate next to the paper.

The paper defines CS-WER as WER "computed only on tokens inside code-switched regions" and N-WER as WER outside them, but
gives no formulas, alignment rule or insertion rule. Our reading (documented in the output): align normalised reference and
hypothesis words with jiwer; reference words covered by an annotated cs_term are code-switched; a substitution or deletion
is charged to its reference word, an insertion to the reference word before it (the first word if at the start).
CS-WER and N-WER then split the total error count exactly, so they average to WER. The paper's own numbers do not (its WER
lies below both CS-WER and N-WER for LoRA), so its N-WER is computed on some other basis and is not reproducible here.

Run from ASR/:  ..\\.venv-asr\\Scripts\\python.exe scripts/build_same_metric_table.py
"""
import json
import sys
from pathlib import Path

import jiwer
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import normalize_text  # noqa: E402

RUNS = ROOT / "local/runs"
TAB, FIG = ROOT / "reports/task2_20261030/tables", ROOT / "reports/task2_20261030/figures"
SRC = "ViMedCSS_vsplit"

OURS = {  # name -> (run dir, split)
    "B0V": ("B0V_phowhisper_small_zeroshot_vsplit/{s}_guard", "PhoWhisper-small zero-shot"),
    "V3": ("V3_phowhisper_small_lora_vsplit/{s}_local_v3", "small + LoRA"),
    "V3B": ("V3B_phowhisper_small_lora_vsplit_beam5/{s}_beam5", "small + LoRA, beam 5"),
    "V4": ("V4_phowhisper_small_lora_vsplit_specaug/{s}_local_v4", "small + LoRA + SpecAugment"),
    "V4B": ("V4B_phowhisper_small_lora_vsplit_specaug_beam5/{s}_beam5", "small + LoRA + SpecAugment, beam 5"),
    "V5": ("V5_phowhisper_medium_lora_vsplit_specaug/{s}_local_v5", "medium + LoRA + SpecAugment"),
    "V5B": ("V5B_phowhisper_medium_lora_vsplit_specaug_beam5/{s}_beam5", "medium + LoRA + SpecAugment, beam 5"),
}


def cs_mask(ref_tokens, terms):
    mask = [False] * len(ref_tokens)
    for t in filter(None, (normalize_text(a).split() for a in terms.split(";"))):
        for i in range(len(ref_tokens) - len(t) + 1):
            if ref_tokens[i:i + len(t)] == t:
                for j in range(i, i + len(t)):
                    mask[j] = True
    return mask


def score(df):
    refs = [normalize_text(t) for t in df.text]
    hyps = [normalize_text(h) for h in df.hyp]
    n = dict(cs=0, ncs=0)
    err = dict(cs_sd=0, ncs_sd=0, cs_ins=0, ncs_ins=0)
    covered = total_terms = 0
    for r, h, terms in zip(refs, hyps, df.cs_terms):
        if not r:
            continue
        rt = r.split()
        mask = cs_mask(rt, terms)
        for t in filter(None, (normalize_text(a) for a in terms.split(";"))):
            total_terms += 1
            covered += t in r
        n["cs"] += sum(mask)
        n["ncs"] += len(rt) - sum(mask)
        for ch in jiwer.process_words(r, h).alignments[0]:
            if ch.type in ("substitute", "delete"):
                for i in range(ch.ref_start_idx, ch.ref_end_idx):
                    err["cs_sd" if mask[i] else "ncs_sd"] += 1
            elif ch.type == "insert":
                k = ch.hyp_end_idx - ch.hyp_start_idx
                p = ch.ref_start_idx
                cls = mask[p - 1] if p > 0 else mask[0]
                err["cs_ins" if cls else "ncs_ins"] += k
    keep = [i for i, r in enumerate(refs) if r]
    out = dict(
        n_clips=len(keep), cs_words=n["cs"], term_coverage=covered / max(1, total_terms),
        wer=100 * jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep]),
        cer=100 * jiwer.cer([refs[i] for i in keep], [hyps[i] for i in keep]),
        cs_wer=100 * (err["cs_sd"] + err["cs_ins"]) / n["cs"],
        cs_wer_no_ins=100 * err["cs_sd"] / n["cs"],
        n_wer=100 * (err["ncs_sd"] + err["ncs_ins"]) / n["ncs"],
    )
    return out


rows = []
for split in ("dev", "test", "hard"):
    for name, (pattern, desc) in OURS.items():
        path = RUNS / pattern.format(s=split) / f"eval/{split}/predictions.jsonl"
        if not path.is_file():
            continue
        df = pd.read_json(path, lines=True)
        df = df[df.source == SRC]
        rows.append(dict(split=split.upper(), system=name, description=desc, **score(df)))
ours = pd.DataFrame(rows)

# paper (arXiv 2602.12911 v2, read 2026-10-30): WER, CER, CS-WER, N-WER per split
PAPER = {
    "Whisper-Small zero-shot": dict(TEST=(50.03, 34.34, 61.25, 51.35), HARD=(69.20, 44.72, 66.75, 71.81)),
    "PhoWhisper-Small zero-shot": dict(TEST=(36.31, 22.64, 62.55, 37.59), HARD=(46.27, 28.11, 66.01, 48.13)),
    "Whisper-Large-v3 zero-shot": dict(TEST=(34.47, 24.61, 46.69, 35.26), HARD=(39.35, 27.46, 50.10, 40.37)),
    "PhoWhisper-Large zero-shot": dict(TEST=(31.24, 19.25, 55.05, 32.36), HARD=(37.37, 23.02, 57.37, 38.67)),
    "VietASR zero-shot": dict(TEST=(27.56, 20.38, 58.38, 28.25), HARD=(34.43, 25.28, 60.71, 35.34)),
    "PhoWhisper-Small + LoRA": dict(TEST=(27.13, 18.01, 30.26, 27.90), HARD=(37.27, 24.55, 60.71, 38.69)),
    "PhoWhisper-Small + AG (best)": dict(TEST=(23.67, 14.73, 19.50, 24.40), HARD=(33.73, 21.48, 57.29, 34.80)),
}
prow = []
for name, d in PAPER.items():
    for split, v in d.items():
        prow.append(dict(split=split, system=name, wer=v[0], cer=v[1], cs_wer=v[2], n_wer=v[3]))
paper = pd.DataFrame(prow)
ours.round(3).to_csv(TAB / "same_metric_ours.csv", index=False)
paper.to_csv(TAB / "same_metric_paper.csv", index=False)

# relative reduction vs each side's own PhoWhisper-small zero-shot on the same split (removes most of the test-set difficulty gap)
def rel(df, base):
    out = []
    for _, r in df.iterrows():
        b = base[(base.split == r["split"])].iloc[0]
        out.append(dict(system=r["system"], split=r["split"], wer_red=100 * (1 - r["wer"] / b["wer"]),
                        cs_red=100 * (1 - r["cs_wer"] / b["cs_wer"])))
    return pd.DataFrame(out)


rel_ours = rel(ours, ours[ours.system == "B0V"])
rel_paper = rel(paper, paper[paper.system == "PhoWhisper-Small zero-shot"])
rel_ours.round(2).to_csv(TAB / "same_metric_relative_ours.csv", index=False)
rel_paper.round(2).to_csv(TAB / "same_metric_relative_paper.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print(ours.round(2).to_string(index=False))
print(rel_ours.round(1).to_string(index=False))
print(rel_paper[rel_paper.system.isin(["PhoWhisper-Small + LoRA", "PhoWhisper-Small + AG (best)"])].round(1).to_string(index=False))

# figure: WER (same definition) and relative CS-WER reduction vs the same zero-shot baseline
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
pz = lambda sp, m: paper[(paper.split == sp) & (paper.system == "PhoWhisper-Small zero-shot")][m].iloc[0]
items = []
for name in ["PhoWhisper-Small zero-shot", "PhoWhisper-Small + LoRA", "PhoWhisper-Small + AG (best)"]:
    r = paper[(paper.split == "TEST") & (paper.system == name)].iloc[0]
    items.append(("Paper: " + name.replace("PhoWhisper-Small", "small"), r.wer, 100 * (1 - r.cs_wer / pz("TEST", "cs_wer")), "#9ecae9"))
for sp, tag, col, names in [("TEST", "TEST", "#F58518", ["B0V", "V3", "V4B", "V5B"])]:
    base = ours[(ours.split == sp) & (ours.system == "B0V")].iloc[0]
    for name in names:
        r = ours[(ours.split == sp) & (ours.system == name)].iloc[0]
        items.append((f"Nhóm {tag}: {name}", r.wer, 100 * (1 - r.cs_wer / base.cs_wer), col))
for a_, idx, title in zip(ax, [1, 2], ["WER % (cùng định nghĩa, khác tập test)", "Giảm CS-WER so với zero-shot cùng tập (%)"]):
    a_.barh([i[0] for i in items][::-1], [i[idx] for i in items][::-1], color=[i[3] for i in items][::-1])
    for k, it in enumerate(items[::-1]):
        a_.text(it[idx] + 0.8, k, f"{it[idx]:.1f}", va="center", fontsize=9)
    a_.set_title(title)
    a_.set_xlim(0, max(i[idx] for i in items) * 1.18)
fig.suptitle("ViMedCSS: paper (split chính thức) so với nhóm (chia theo video); CS-WER của nhóm theo cách tính riêng nên chỉ so tương đối", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig13_same_metric_vs_paper.png", bbox_inches="tight", dpi=150)
print("saved")

# ------------------------------------------------------------------ official split (Colab A100 runs, experiments/<ID>/eval)
# Same TEST (1,614 clips) and HARD (658) as the paper. O3W was frozen on official DEV before TEST
# (reports/final_freeze_official/freeze.json). "unseen" = clips whose source video is absent from ViMedCSS TRAIN.
import pyarrow.parquet as pq  # noqa: E402

EXP = ROOT / "experiments"
OFFICIAL = {  # name -> (experiment, eval suffix, description, paper row reproduced)
    "B0": ("B0_phowhisper_small_zeroshot", "", "PhoWhisper-small zero-shot", "PhoWhisper-Small zero-shot"),
    "B2": ("B2_phowhisper_large_zeroshot", "", "PhoWhisper-large zero-shot", "PhoWhisper-Large zero-shot"),
    "B3": ("B3_whisper_large_v3_zeroshot", "", "Whisper-large-v3 zero-shot", "Whisper-Large-v3 zero-shot"),
    "O3W": ("O3W_whisper_large_v3_lora_official_termos", "_beam5",
            "Whisper-large-v3 + LoRA + SpecAugment + rare-term oversampling, beam 5", None),
}
D = ROOT / "data/ViMedCSS"
train_videos = set(pq.read_table(D / "train.parquet", columns=["group_id"]).column("group_id").to_pylist())
video = {}
for sp in ("test", "hard"):
    t = pq.read_table(D / f"{sp}.parquet", columns=["id", "group_id"]).to_pydict()
    video.update(zip(t["id"], t["group_id"]))

orow = []
for split in ("test", "hard"):
    for name, (exp, sfx, desc, paper_row) in OFFICIAL.items():
        path = EXP / exp / f"eval/{split}{sfx}/predictions.jsonl"
        if not path.is_file():
            continue
        df = pd.read_json(path, lines=True)
        df = df[df.source == "ViMedCSS"]
        unseen = df[~df.id.map(lambda i: video[i] in train_videos)]
        m = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        orow.append(dict(split=split.upper(), system=name, description=desc, **score(df),
                         term_recall=100 * m["by_source"]["ViMedCSS"]["cs_term_recall"],
                         n_unseen=len(unseen), wer_unseen=score(unseen)["wer"],
                         paper_row=paper_row))
official = pd.DataFrame(orow)
for sp in ("TEST", "HARD"):
    base = official[(official.split == sp) & (official.system == "B0")].iloc[0]
    sel = official.split == sp
    official.loc[sel, "wer_red_vs_B0"] = 100 * (1 - official.loc[sel, "wer"] / base.wer)
    official.loc[sel, "cs_red_vs_B0"] = 100 * (1 - official.loc[sel, "cs_wer"] / base.cs_wer)
official.round(4).to_csv(TAB / "same_metric_official.csv", index=False)

# other corpora on the same TEST decode (paper rows: PhoWhisper VIVOS, VietMed XLSR-53-Viet)
crow = []
for name, (exp, sfx, desc, _) in OFFICIAL.items():
    p = EXP / exp / f"eval/test{sfx}/metrics.json"
    if p.is_file():
        m = json.loads(p.read_text(encoding="utf-8"))
        crow.append(dict(system=name, description=desc, **{f"wer_{s}": 100 * v["wer"] for s, v in m["by_source"].items()},
                         wer_all=100 * m["overall"]["wer"]))
pd.DataFrame(crow).round(4).to_csv(TAB / "official_test_by_source.csv", index=False)
print(official.drop(columns=["description", "paper_row"]).round(2).to_string(index=False))
print(pd.DataFrame(crow).round(2).to_string(index=False))

# figure: official TEST/HARD WER, paper vs our reproduction vs O3W
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for a_, sp in zip(ax, ("TEST", "HARD")):
    items = []
    for pname, label in [("PhoWhisper-Small zero-shot", "small zero-shot"), ("PhoWhisper-Large zero-shot", "PhoWhisper-large zero-shot"),
                         ("Whisper-Large-v3 zero-shot", "Whisper-large-v3 zero-shot"), ("PhoWhisper-Small + LoRA", "small + LoRA"),
                         ("PhoWhisper-Small + AG (best)", "small + Attention Guide (tốt nhất)")]:
        r = paper[(paper.split == sp) & (paper.system == pname)].iloc[0]
        items.append(("Paper: " + label, r.wer, "#9ecae9"))
    for name, label in [("B0", "small zero-shot"), ("B2", "PhoWhisper-large zero-shot"), ("B3", "Whisper-large-v3 zero-shot")]:
        r = official[(official.split == sp) & (official.system == name)].iloc[0]
        items.append((f"Nhóm chạy lại: {label}", r.wer, "#bdbdbd"))
    r = official[(official.split == sp) & (official.system == "O3W")].iloc[0]
    items.append(("Nhóm: O3W (hệ cuối)", r.wer, "#d94f2b"))
    a_.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1], color=[i[2] for i in items][::-1])
    for k, it in enumerate(items[::-1]):
        a_.text(it[1] + 0.6, k, f"{it[1]:.2f}", va="center", fontsize=9)
    a_.set_title(f"ViMedCSS {sp} chính thức ({'1.614' if sp == 'TEST' else '658'} clip), WER %")
    a_.set_xlim(0, max(i[1] for i in items) * 1.18)
ax[1].set_yticklabels([])
fig.suptitle("Cùng tập test với paper ViMedCSS. Cách chấm của nhóm cho zero-shot thấp hơn paper 5–10 điểm (xám so với xanh)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig14_official_vs_paper.png", bbox_inches="tight", dpi=150)
print("saved official")
