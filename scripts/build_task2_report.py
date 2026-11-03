"""Build the Task 2 report assets (tables + figures) from the data manifests and local run outputs. CPU only.

Run from ASR/:  ..\.venv-asr\Scripts\python.exe scripts/build_task2_report.py
Output: reports/task2_20261030/{tables,figures}
"""
import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import jiwer
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import normalize_text  # noqa: E402

OUT = ROOT / "reports/task2_20261030"
TAB, FIG = OUT / "tables", OUT / "figures"
RUNS = ROOT / "local/runs"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})
C = {"VIVOS": "#4C78A8", "FLEURS": "#F58518", "VietMed": "#54A24B", "ViMedCSS": "#B279A2"}
COLS = ["id", "source", "split", "group_id", "speaker_id", "text", "cs_terms", "duration"]


def words(text):
    return len(normalize_text(text).split())


def load(source, split):
    path = ROOT / f"data/{source}/{split}.parquet"
    return pd.read_parquet(path, columns=COLS) if path.is_file() else None


def terms_of(df):
    c = collections.Counter()
    for t in df.cs_terms:
        for x in filter(None, (normalize_text(a) for a in t.split(";"))):
            c[x] += 1
    return c


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1.1 / 1.3 data statistics
SRC = ["VIVOS", "FLEURS", "VietMed", "ViMedCSS"]
rows = []
frames = {}
for s in SRC + ["ViMedCSS_vsplit"]:
    for sp in ["train", "dev", "test", "hard"]:
        df = load(s, sp)
        if df is None:
            continue
        frames[(s, sp)] = df
        rows.append(dict(source=s, split=sp, clips=len(df), hours=df.duration.sum() / 3600,
                         words=int(df.text.map(words).sum()), groups=df.group_id.nunique(),
                         speakers=df.speaker_id.replace("", pd.NA).nunique(), mean_dur=df.duration.mean()))
stats = pd.DataFrame(rows)
stats.round(3).to_csv(TAB / "data_stats_by_source_split.csv", index=False)

official = stats[stats.source.isin(SRC) & stats.split.isin(["train", "dev", "test"])]
split_tot = official.groupby("split")[["clips", "hours", "words"]].sum()
split_pct = (100 * split_tot / split_tot.sum()).round(2)
split_tot.join(split_pct, rsuffix="_pct").round(2).to_csv(TAB / "split_ratio_official.csv")

# V3 effective data (ViMedCSS re-split by video) for the split ratio actually used by V3+
v3 = stats[stats.source.isin(["VIVOS", "FLEURS", "VietMed", "ViMedCSS_vsplit"]) & stats.split.isin(["train", "dev", "test"])]
v3_tot = v3.groupby("split")[["clips", "hours", "words"]].sum()
v3_tot.join((100 * v3_tot / v3_tot.sum()).round(2), rsuffix="_pct").round(2).to_csv(TAB / "split_ratio_v3.csv")

# --- fig 1: composition of train / dev / test by source (hours and words)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, sp in zip(ax, ["train", "dev", "test"]):
    d = official[official.split == sp].set_index("source").reindex(SRC)
    share = 100 * d["words"] / d["words"].sum()
    left = 0
    for s in SRC:
        a.barh(0, share[s], left=left, color=C[s], label=s)
        a.text(left + share[s] / 2, 0, f"{share[s]:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += share[s]
    hshare = 100 * d["hours"] / d["hours"].sum()
    left = 0
    for s in SRC:
        a.barh(1, hshare[s], left=left, color=C[s])
        a.text(left + hshare[s] / 2, 1, f"{hshare[s]:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += hshare[s]
    a.set_yticks([0, 1], ["số từ", "số giờ"])
    a.set_xlim(0, 100)
    a.set_title(sp.upper())
    a.grid(False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", frameon=False)
fig.suptitle("Cơ cấu dữ liệu theo nguồn (split chính thức của từng corpus)")
fig.tight_layout(rect=[0, .08, 1, .95])
fig.savefig(FIG / "fig1_source_composition.png", bbox_inches="tight")
plt.close(fig)

# --- fig 2: clips per split per source
fig, ax = plt.subplots(figsize=(6.5, 3.4))
piv = official.pivot(index="source", columns="split", values="clips").reindex(SRC)[["train", "dev", "test"]]
piv.plot.bar(ax=ax, color=["#4C78A8", "#F58518", "#54A24B"], width=.75)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=7)
ax.set_ylabel("số clip")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.set_title("Số clip Train / Dev / Test theo nguồn")
save(fig, "fig2_split_by_source.png")

# ------------------------------------------------------------------ 1.3 imbalance: term long tail + seen/unseen
tr_off, dv_off = load("ViMedCSS", "train"), load("ViMedCSS", "dev")
tr_v, dv_v = frames[("ViMedCSS_vsplit", "train")], frames[("ViMedCSS_vsplit", "dev")]
ct, cd = terms_of(tr_v), terms_of(dv_v)
tot_occ = sum(ct.values())
ranked = ct.most_common()
imb = dict(
    train_term_occurrences=tot_occ, train_term_distinct=len(ct),
    terms_seen_once=sum(1 for v in ct.values() if v == 1),
    terms_seen_lt5=sum(1 for v in ct.values() if v < 5),
    top10_share_pct=100 * sum(v for _, v in ranked[:10]) / tot_occ,
    top50_share_pct=100 * sum(v for _, v in ranked[:50]) / tot_occ,
    top1_term=ranked[0][0], top1_count=ranked[0][1], top1_share_pct=100 * ranked[0][1] / tot_occ,
    dev_term_occurrences=sum(cd.values()), dev_term_distinct=len(cd),
    dev_distinct_unseen=sum(1 for k in cd if k not in ct),
    dev_occ_seen_share_pct=100 * sum(v for k, v in cd.items() if k in ct) / sum(cd.values()),
)
dev_words = {s: int(stats[(stats.source == s) & (stats.split == "dev")].words.iloc[0]) for s in
             ["VIVOS", "FLEURS", "VietMed", "ViMedCSS_vsplit"]}
train_words = {s: int(stats[(stats.source == s) & (stats.split == "train")].words.iloc[0]) for s in
               ["VIVOS", "FLEURS", "VietMed", "ViMedCSS_vsplit"]}
imb["dev_word_share_pct"] = {k: 100 * v / sum(dev_words.values()) for k, v in dev_words.items()}
imb["train_word_share_pct"] = {k: 100 * v / sum(train_words.values()) for k, v in train_words.items()}
(TAB / "imbalance.json").write_text(json.dumps(imb, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
cnt = [v for _, v in ranked]
ax[0].bar(range(len(cnt)), cnt, width=1.0, color="#B279A2")
ax[0].set_yscale("log")
ax[0].set_xlabel(f"thuật ngữ (xếp hạng theo tần suất, {len(cnt)} thuật ngữ)")
ax[0].set_ylabel("số lần xuất hiện trong TRAIN (log)")
ax[0].set_title(f"Phân bố đuôi dài: top-10 chiếm {imb['top10_share_pct']:.1f}%, {imb['terms_seen_once']} thuật ngữ chỉ 1 lần")
top = ranked[:12]
ax[1].barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#B279A2")
ax[1].set_title("12 thuật ngữ phổ biến nhất (ViMedCSS TRAIN)")
save(fig, "fig3_term_long_tail.png")

# --- fig 4: DEV word share vs TRAIN hours share by source (dominance of VietMed in the DEV metric)
names = ["VIVOS", "FLEURS", "VietMed", "ViMedCSS_vsplit"]
fig, ax = plt.subplots(figsize=(7, 3.5))
x = range(len(names))
w = .38
ax.bar([i - w / 2 for i in x], [imb["train_word_share_pct"][n] for n in names], w, label="TRAIN (số từ)", color="#9ecae9")
ax.bar([i + w / 2 for i in x], [imb["dev_word_share_pct"][n] for n in names], w, label="DEV (số từ)", color="#F58518")
for i, n in enumerate(names):
    ax.text(i - w / 2, imb["train_word_share_pct"][n] + .5, f"{imb['train_word_share_pct'][n]:.1f}%", ha="center", fontsize=8)
    ax.text(i + w / 2, imb["dev_word_share_pct"][n] + .5, f"{imb['dev_word_share_pct'][n]:.1f}%", ha="center", fontsize=8)
ax.set_xticks(list(x), ["VIVOS", "FLEURS", "VietMed", "ViMedCSS\n(re-split)"])
ax.set_ylabel("% tổng số từ")
ax.legend(frameon=False)
ax.set_title("Mất cân bằng nguồn: VietMed ít trong TRAIN nhưng áp đảo DEV")
save(fig, "fig4_source_imbalance.png")

# ------------------------------------------------------------------ 1.4 imbalance handling: leakage before / after
audit = json.loads((ROOT / "reports/handover_20261021/split_overlap_audit.json").read_text(encoding="utf-8"))
vsplit_rep = json.loads((ROOT / "data/ViMedCSS_vsplit/split_report.json").read_text(encoding="utf-8"))
g_tr_off = set(tr_off.group_id)
before = dict(train_clips=len(tr_off), dev_clips=len(dv_off), test_clips=len(load("ViMedCSS", "test")),
              hard_clips=len(load("ViMedCSS", "hard")),
              train_videos=tr_off.group_id.nunique(), dev_videos=dv_off.group_id.nunique(),
              dev_videos_shared_with_train=len(g_tr_off & set(dv_off.group_id)),
              dev_clips_sharing_video=int(dv_off.group_id.isin(g_tr_off).sum()),
              test_videos_shared_with_train=len(g_tr_off & set(load("ViMedCSS", "test").group_id)))
te_v = frames[("ViMedCSS_vsplit", "test")]
g_tr_v = set(tr_v.group_id)
after = dict(train_clips=len(tr_v), dev_clips=len(dv_v), test_clips=len(te_v),
             hard_clips=len(frames[("ViMedCSS_vsplit", "hard")]),
             train_videos=tr_v.group_id.nunique(), dev_videos=dv_v.group_id.nunique(),
             dev_videos_shared_with_train=len(g_tr_v & set(dv_v.group_id)),
             dev_clips_sharing_video=int(dv_v.group_id.isin(g_tr_v).sum()),
             test_videos_shared_with_train=len(g_tr_v & set(te_v.group_id)),
             dropped_duplicates=len(vsplit_rep["dropped_duplicate_ids"]), dropped_silent=len(vsplit_rep["dropped_silent_ids"]))
exclude = json.loads((ROOT / "data/v3_train_exclude.json").read_text(encoding="utf-8"))
after["train_exclude_ids"] = len(exclude["ids"])
(TAB / "leakage_before_after.json").write_text(json.dumps(dict(before=before, after=after), indent=2), encoding="utf-8")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
lab = ["Trước\n(split chính thức)", "Sau\n(chia lại theo video)"]
shared = [100 * before["dev_clips_sharing_video"] / before["dev_clips"], 100 * after["dev_clips_sharing_video"] / after["dev_clips"]]
ax[0].bar(lab, shared, color=["#d62728", "#2ca02c"], width=.5)
for i, v in enumerate(shared):
    ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("% clip DEV có video trùng với TRAIN")
ax[0].set_title("Rò rỉ video Train → Dev (ViMedCSS)")
ax[0].grid(False)
sp = ["train", "dev", "test", "hard"]
bx = [before[f"{s}_clips"] for s in sp]
ax2 = ax[1]
xx = range(4)
ax2.bar([i - .2 for i in xx], bx, .4, label="trước", color="#9ecae9")
ax2.bar([i + .2 for i in xx], [after[f"{s}_clips"] for s in sp], .4, label="sau", color="#F58518")
for i, s in enumerate(sp):
    ax2.text(i - .2, bx[i] + 150, f"{bx[i]:,}", ha="center", fontsize=8)
    ax2.text(i + .2, after[f"{s}_clips"] + 150, f"{after[f'{s}_clips']:,}", ha="center", fontsize=8)
ax2.set_xticks(list(xx), [s.upper() for s in sp])
ax2.set_yscale("log")
ax2.set_ylabel("số clip (log)")
ax2.legend(frameon=False)
ax2.set_title("Kích thước các split ViMedCSS")
save(fig, "fig5_leakage_before_after.png")

# ------------------------------------------------------------------ run metrics (ablation / error / latency)
DEV_OLD = [("B0", "B0_phowhisper_small_zeroshot/dev_local_v1", "PhoWhisper-small zero-shot", "greedy", "không"),
           ("B1", "B1_whisper_small_zeroshot/dev_local_v1", "Whisper-small zero-shot", "greedy", "không"),
           ("B0G", "B0G_phowhisper_small_zeroshot_guard/dev_guard", "PhoWhisper-small zero-shot", "greedy", "guard"),
           ("B1G", "B1G_whisper_small_zeroshot_guard/dev_guard", "Whisper-small zero-shot", "greedy", "guard"),
           ("V1", "V1_phowhisper_small_lora/dev_local_v1", "PhoWhisper-small + LoRA", "greedy", "không"),
           ("V1G", "V1G_phowhisper_small_lora_guard/dev_guard", "PhoWhisper-small + LoRA", "greedy", "guard"),
           ("V2", "V2_phowhisper_small_lora_cont/dev_local_v2", "V1 → huấn luyện tiếp", "greedy", "không"),
           ("V2G", "V2G_phowhisper_small_lora_cont_guard/dev_guard", "V1 → huấn luyện tiếp", "greedy", "guard")]
DEV_V3 = [("B0V", "B0V_phowhisper_small_zeroshot_vsplit/dev_guard", "PhoWhisper-small zero-shot", "greedy"),
          ("V3", "V3_phowhisper_small_lora_vsplit/dev_local_v3", "small + LoRA, chia theo video", "greedy"),
          ("V3B", "V3B_phowhisper_small_lora_vsplit_beam5/dev_beam5", "V3 + beam 5", "beam 5"),
          ("V4", "V4_phowhisper_small_lora_vsplit_specaug/dev_local_v4", "V3 + SpecAugment", "greedy"),
          ("V4B", "V4B_phowhisper_small_lora_vsplit_specaug_beam5/dev_beam5", "V4 + beam 5", "beam 5"),
          ("V5", "V5_phowhisper_medium_lora_vsplit_specaug/dev_local_v5", "PhoWhisper-medium + LoRA + SpecAugment", "greedy"),
          ("V5B", "V5B_phowhisper_medium_lora_vsplit_specaug_beam5/dev_beam5", "V5 + beam 5", "beam 5")]


def metrics(rel, split="dev"):
    p = RUNS / rel / f"eval/{split}/metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def row(name, rel, desc, dec, split="dev", **extra):
    m = metrics(rel, split)
    if m is None:
        return None
    o = m["overall"]
    r = dict(run=name, description=desc, decode=dec, n_utts=o["n_utts"], wer=100 * o["wer"], cer=100 * o["cer"],
             sub=o["substitutions"], dele=o["deletions"], ins=o["insertions"], ref_words=o["ref_words"],
             term_recall=100 * o["cs_term_recall"] if o.get("cs_term_recall") is not None else None,
             n_truncated=o.get("n_truncated", 0), rtf=m.get("rtf_decode_only"), **extra)
    for s, v in m["by_source"].items():
        r["wer_" + s.replace("_vsplit", "")] = 100 * v["wer"]
    return r


old = pd.DataFrame([row(n, p, d, dec, guard=g) for n, p, d, dec, g in DEV_OLD if row(n, p, d, dec)])
new = pd.DataFrame([row(n, p, d, dec) for n, p, d, dec in DEV_V3 if row(n, p, d, dec)])
old.round(6).to_csv(TAB / "ablation_dev_official.csv", index=False)
new.round(6).to_csv(TAB / "ablation_dev_v3.csv", index=False)

# TEST / HARD tables (V3 split; TEST already seen, reported for the paper comparison only)
tt = []
for split, suf in [("test", "test"), ("hard", "hard")]:
    for n, p in [("B0V", f"B0V_phowhisper_small_zeroshot_vsplit/{split}_guard"),
                 ("V3", f"V3_phowhisper_small_lora_vsplit/{split}_local_v3"),
                 ("V3B", f"V3B_phowhisper_small_lora_vsplit_beam5/{split}_beam5"),
                 ("V4", f"V4_phowhisper_small_lora_vsplit_specaug/{split}_local_v4"),
                 ("V4B", f"V4B_phowhisper_small_lora_vsplit_specaug_beam5/{split}_beam5"),
                 ("V5", f"V5_phowhisper_medium_lora_vsplit_specaug/{split}_local_v5"),
                 ("V5B", f"V5B_phowhisper_medium_lora_vsplit_specaug_beam5/{split}_beam5")]:
        r = row(n, p, "", "", split=split)
        if r:
            r["split"] = split
            tt.append(r)
pd.DataFrame(tt).round(3).to_csv(TAB / "test_hard_v3.csv", index=False)

# --- fig 6: ablation progression (DEV V3)
fig, ax = plt.subplots(figsize=(8.2, 3.9))
order = new.run.tolist()
vals = new.wer.tolist()
colors = ["#bbbbbb"] + ["#4C78A8"] * 4 + ["#F58518"] * 2
bars = ax.bar(order, vals, color=colors[:len(order)], width=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .4, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("WER % (DEV chia theo video, 4.978 clip)")
ax.set_ylim(0, max(vals) + 4)
ax.set_title("Tiến trình các phiên bản trên cùng DEV: B0V → V5B")
ax.grid(axis="x", visible=False)
ax.text(.99, .95, "xám: baseline · xanh: small · cam: medium", transform=ax.transAxes, ha="right", va="top", fontsize=8)
save(fig, "fig6_ablation_dev_v3.png")

# --- fig 7: S/D/I breakdown per run (DEV V3) normalised by ref words
fig, ax = plt.subplots(figsize=(8.2, 3.9))
bot = [0] * len(new)
for key, lab, col in [("sub", "thay thế", "#4C78A8"), ("dele", "xóa", "#E45756"), ("ins", "chèn", "#F2CF5B")]:
    v = (100 * new[key] / new.ref_words).tolist()
    ax.bar(new.run, v, bottom=bot, label=lab, color=col, width=.6)
    bot = [a + b for a, b in zip(bot, v)]
ax.set_ylabel("% số từ tham chiếu")
ax.legend(frameon=False, ncol=3)
ax.set_title("Cơ cấu lỗi (thay thế / xóa / chèn) theo phiên bản – DEV")
ax.grid(axis="x", visible=False)
save(fig, "fig7_error_breakdown.png")

# --- fig 8: WER by source (DEV V3)
fig, ax = plt.subplots(figsize=(9, 3.8))
srcs = ["FLEURS", "VIVOS", "ViMedCSS", "VietMed"]
w = .8 / len(new)
for i, (_, r) in enumerate(new.iterrows()):
    ax.bar([k + i * w for k in range(4)], [r["wer_" + s] for s in srcs], w, label=r.run)
ax.set_xticks([k + .4 - w / 2 for k in range(4)], srcs)
ax.set_ylabel("WER %")
ax.legend(ncol=7, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1))
ax.set_title("WER theo nguồn trên DEV")
ax.grid(axis="x", visible=False)
save(fig, "fig8_wer_by_source.png")

# ------------------------------------------------------------------ training curves
curves = {}
for name, rel in [("V1", "V1_phowhisper_small_lora/local_v1"), ("V2", "V2_phowhisper_small_lora_cont/local_v2"),
                  ("V3", "V3_phowhisper_small_lora_vsplit/local_v3"), ("V4", "V4_phowhisper_small_lora_vsplit_specaug/local_v4"),
                  ("V5", "V5_phowhisper_medium_lora_vsplit_specaug/local_v5")]:
    ck = sorted((RUNS / rel / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ck:
        continue
    state = json.loads((ck[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    log = state["log_history"]
    curves[name] = dict(train=[(e["step"], e["loss"]) for e in log if "loss" in e],
                        evl=[(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e],
                        evw=[(e["step"], 100 * e["eval_wer"]) for e in log if "eval_wer" in e])
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
cc = dict(zip(curves, ["#999999", "#bbbb55", "#4C78A8", "#54A24B", "#F58518"]))
for n, c in curves.items():
    if c["train"]:
        s, v = zip(*c["train"])
        ax[0].plot(s, v, label=n, color=cc[n], lw=1.2)
    if c["evl"]:
        s, v = zip(*c["evl"])
        ax[1].plot(s, v, "o-", label=n, color=cc[n], ms=3)
    if c["evw"]:
        s, v = zip(*c["evw"])
        ax[2].plot(s, v, "o-", label=n, color=cc[n], ms=3)
for a, t, yl in zip(ax, ["Train loss", "Eval loss (DEV-subset)", "Eval WER % (DEV-subset 600 clip)"], ["loss", "loss", "WER %"]):
    a.set_title(t)
    a.set_xlabel("bước tối ưu")
    a.set_ylabel(yl)
ax[0].set_ylim(0, 1.2)
ax[1].set_ylim(0, 0.8)
ax[2].set_ylim(0, 40)
ax[2].annotate("V1 phân kỳ (lr 1e-3):\nWER tới 320% ở bước 700", (700, 36), fontsize=8, color="#555555")
ax[2].legend(frameon=False, loc="upper right")
fig.suptitle("Đường cong huấn luyện (V1 dùng lịch lr khác; V1/V2 đánh giá trên DEV-subset của split chính thức)", fontsize=9)
save(fig, "fig9_training_curves.png")
pd.DataFrame([dict(run=n, step=s, eval_wer=v) for n, c in curves.items() for s, v in c["evw"]]).to_csv(TAB / "eval_curves.csv", index=False)

# compact two-panel version for slides
fig, ax = plt.subplots(2, 1, figsize=(5.8, 4.4))
for n, c in curves.items():
    s_, v_ = zip(*c["train"])
    ax[0].plot(s_, v_, label=n, color=cc[n], lw=1.2)
    s_, v_ = zip(*c["evw"])
    ax[1].plot(s_, v_, "o-", label=n, color=cc[n], ms=3)
ax[0].set_ylim(0, 1.2)
ax[0].set_ylabel("train loss")
ax[1].set_ylim(0, 40)
ax[1].set_ylabel("WER % (DEV-subset)")
ax[1].set_xlabel("bước tối ưu")
ax[1].annotate("V1 phân kỳ (lr 1e-3): WER tới 320%", (650, 33), fontsize=9, color="#555555")
ax[0].legend(frameon=False, ncol=5, fontsize=9, loc="upper right")
save(fig, "fig9b_curves_compact.png")

# ------------------------------------------------------------------ failure analysis (V5B on DEV V3)
pred = pd.read_json(RUNS / "V5B_phowhisper_medium_lora_vsplit_specaug_beam5/dev_beam5/eval/dev/predictions.jsonl", lines=True)
pred["ref"] = pred.text.map(normalize_text)
pred["hyp_n"] = pred.hyp.map(normalize_text)
pred = pred[pred.ref != ""].reset_index(drop=True)
subs = collections.Counter()
recs = []
for i, r in pred.iterrows():
    o = jiwer.process_words(r.ref, r.hyp_n)
    n = max(1, len(r.ref.split()))
    first_del = last_del = 0
    for ch in o.alignments[0]:
        if ch.type == "substitute":
            for a, b in zip(range(ch.ref_start_idx, ch.ref_end_idx), range(ch.hyp_start_idx, ch.hyp_end_idx)):
                subs[(r.ref.split()[a], r.hyp_n.split()[b])] += 1
    chunks = o.alignments[0]
    head = chunks[0] if chunks else None
    tail = chunks[-1] if chunks else None
    edge_err = 0
    for ch in (head, tail):
        if ch is not None and ch.type in ("delete", "insert"):
            edge_err += max(ch.ref_end_idx - ch.ref_start_idx, ch.hyp_end_idx - ch.hyp_start_idx)
    recs.append(dict(i=i, id=r.id, source=r.source, ref_words=n, S=o.substitutions, D=o.deletions, I=o.insertions,
                     wer=(o.substitutions + o.deletions + o.insertions) / n, edge_err=edge_err,
                     truncated=bool(r.truncated), duration=r.duration))
err = pd.DataFrame(recs)
err["errors"] = err.S + err.D + err.I
total_err = int(err.errors.sum())
cat = pd.Series(index=err.index, dtype=object)
cat[:] = "thay thế rải rác"
cat[err.D > err.S + err.I] = "xóa là chính (mất đoạn)"
cat[err.I > err.S + err.D] = "chèn là chính (thừa/lặp)"
cat[(err.edge_err >= 3) & (err.edge_err >= .4 * err.errors)] = "lệch biên đầu/cuối clip"
cat[err.errors == 0] = "đúng hoàn toàn"
err["category"] = cat
by_cat = err.groupby("category").agg(clips=("id", "count"), errors=("errors", "sum")).assign(
    err_share_pct=lambda d: 100 * d.errors / total_err).sort_values("errors", ascending=False)
by_cat.round(2).to_csv(TAB / "failure_categories.csv")
by_src_cat = err.pivot_table(index="source", columns="category", values="errors", aggfunc="sum", fill_value=0)
by_src_cat.to_csv(TAB / "failure_by_source_category.csv")
src_err = err.groupby("source").errors.sum()
(TAB / "failure_summary.json").write_text(json.dumps(dict(
    total_errors=total_err, clips=len(err), perfect_clips=int((err.errors == 0).sum()),
    source_error_share_pct={k: 100 * v / total_err for k, v in src_err.items()},
    clips_wer_gt_50=int((err.wer > .5).sum()), errors_in_clips_wer_gt_50_pct=100 * err[err.wer > .5].errors.sum() / total_err,
    top_edge_errors_vietmed_pct=100 * err[err.source == "VietMed"].edge_err.sum() / max(1, err[err.source == "VietMed"].errors.sum())),
    ensure_ascii=False, indent=2), encoding="utf-8")

worst = err.sort_values("errors", ascending=False).head(60)
mix = pred.join(err.set_index("i")[["S", "D", "I", "wer", "category"]])
ex = []
for src in ["VietMed", "ViMedCSS_vsplit", "FLEURS"]:
    d = mix[(mix.source == src) & (mix.ref.str.split().str.len().between(6, 28))].sort_values("wer", ascending=False).head(25)
    ex.append(d)
pd.concat(ex)[["id", "source", "category", "wer", "S", "D", "I", "ref", "hyp_n"]].round(3).to_csv(TAB / "failure_examples.csv", index=False)
mix.sort_values("wer", ascending=False).head(0)

# confusion (substitution pairs), restricted to frequent pairs
top_sub = subs.most_common(20)
pd.DataFrame([dict(ref=a, hyp=b, count=c) for (a, b), c in subs.most_common(60)]).to_csv(TAB / "top_substitutions.csv", index=False)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
lab = [f"{a} → {b}" for (a, b), _ in top_sub][::-1]
ax.barh(lab, [c for _, c in top_sub][::-1], color="#4C78A8")
ax.set_xlabel("số lần")
ax.set_title("20 cặp nhầm lẫn từ phổ biến nhất của V5B (DEV)")
save(fig, "fig10_top_confusions.png")

# --- fig 11: failure category breakdown
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
bc = by_cat.drop(index="đúng hoàn toàn", errors="ignore")
ax[0].barh(bc.index[::-1], bc.err_share_pct[::-1], color="#E45756")
for i, v in enumerate(bc.err_share_pct[::-1]):
    ax[0].text(v + .5, i, f"{v:.1f}%", va="center", fontsize=9)
ax[0].set_xlabel("% tổng số lỗi từ")
ax[0].set_title(f"Loại clip theo lỗi chiếm (V5B, {total_err:,} lỗi)")
sh = (100 * src_err / total_err).reindex(["VietMed", "ViMedCSS_vsplit", "FLEURS", "VIVOS"])
ax[1].bar(["VietMed", "ViMedCSS", "FLEURS", "VIVOS"], sh.values, color=["#54A24B", "#B279A2", "#F58518", "#4C78A8"])
for i, v in enumerate(sh.values):
    ax[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
ax[1].set_ylabel("% tổng số lỗi")
ax[1].set_title("Đóng góp lỗi theo nguồn")
ax[1].grid(axis="x", visible=False)
save(fig, "fig11_failure_categories.png")

# --- term-level failures (ViMedCSS)
miss = collections.Counter()
seen_t = collections.Counter()
for _, r in pred[pred.source == "ViMedCSS_vsplit"].iterrows():
    for t in filter(None, (normalize_text(a) for a in r.cs_terms.split(";"))):
        seen_t[t] += 1
        if t not in r.hyp_n:
            miss[t] += 1
rows = [dict(term=t, occurrences=seen_t[t], missed=miss[t], recall_pct=100 * (1 - miss[t] / seen_t[t]), in_train=t in ct)
        for t in seen_t]
tm = pd.DataFrame(rows).sort_values("missed", ascending=False)
tm.round(2).to_csv(TAB / "term_errors.csv", index=False)
seen_mask = tm.in_train
term_summary = dict(
    occ_total=int(tm.occurrences.sum()), missed_total=int(tm.missed.sum()),
    recall_seen_pct=100 * (1 - tm[seen_mask].missed.sum() / tm[seen_mask].occurrences.sum()),
    recall_unseen_pct=100 * (1 - tm[~seen_mask].missed.sum() / max(1, tm[~seen_mask].occurrences.sum())),
    occ_seen=int(tm[seen_mask].occurrences.sum()), occ_unseen=int(tm[~seen_mask].occurrences.sum()),
    missed_unseen_share_pct=100 * tm[~seen_mask].missed.sum() / tm.missed.sum(),
    top_missed=tm.head(12)[["term", "occurrences", "missed"]].to_dict("records"))
(TAB / "term_summary.json").write_text(json.dumps(term_summary, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ latency / size / VRAM
def peak(rel):
    p = RUNS / rel / "resources.jsonl"
    if not p.is_file():
        return None, None
    rr = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(r["gpu_total_used_gb"] for r in rr), max(r["host_rss_gb"] for r in rr)


rows = []
for name, rel_eval, rel_train, params, adapter in [
        ("V4B (small)", "V4B_phowhisper_small_lora_vsplit_specaug_beam5/dev_beam5", "V4_phowhisper_small_lora_vsplit_specaug/local_v4", 244, None),
        ("V4 (small)", "V4_phowhisper_small_lora_vsplit_specaug/dev_local_v4", "V4_phowhisper_small_lora_vsplit_specaug/local_v4", 244, None),
        ("V5 (medium)", "V5_phowhisper_medium_lora_vsplit_specaug/dev_local_v5", "V5_phowhisper_medium_lora_vsplit_specaug/local_v5", 769, None),
        ("V5B (medium)", "V5B_phowhisper_medium_lora_vsplit_specaug_beam5/dev_beam5", "V5_phowhisper_medium_lora_vsplit_specaug/local_v5", 769, None)]:
    m = metrics(rel_eval)
    ad = RUNS / rel_train / "final/adapter_model.safetensors"
    vr, rss = peak(rel_eval)
    tr_vr, tr_rss = peak(rel_train)
    rows.append(dict(system=name, params_M=params, adapter_MB=ad.stat().st_size / 1e6 if ad.is_file() else None,
                     dev_wer=100 * m["overall"]["wer"], rtf=m["rtf_decode_only"], audio_h=m["overall"]["audio_hours"],
                     decode_time_min=m["overall"]["audio_hours"] * 60 * m["rtf_decode_only"],
                     eval_gpu_gb=vr, eval_ram_gb=rss, train_gpu_gb=tr_vr, train_ram_gb=tr_rss))
lat = pd.DataFrame(rows)
lat.round(6).to_csv(TAB / "latency_size.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cols = ["#54A24B", "#4C78A8", "#F58518", "#E45756"]
ax[0].scatter(lat.rtf, lat.dev_wer, s=120, c=cols[:len(lat)])
for _, r in lat.iterrows():
    ax[0].annotate(r.system, (r.rtf, r.dev_wer), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax[0].set_xlabel("RTF (thời gian giải mã / thời lượng audio), RTX 4050 6GB")
ax[0].set_ylabel("WER % trên DEV")
ax[0].set_title("Đánh đổi chính xác – tốc độ")
ax[0].axvline(1.0, color="grey", ls="--", lw=1)
ax[0].set_xlim(0, .45)
ax[1].bar(lat.system, lat.adapter_MB, color=cols[:len(lat)])
for i, v in enumerate(lat.adapter_MB):
    ax[1].text(i, v + 1, f"{v:.0f} MB", ha="center", fontsize=9)
ax[1].set_ylabel("dung lượng adapter LoRA (MB)")
ax[1].set_title("Kích thước phần huấn luyện thêm")
ax[1].tick_params(axis="x", labelsize=8, rotation=10)
ax[1].grid(axis="x", visible=False)
save(fig, "fig12_latency_size.png")

print("done:", OUT)
print(split_pct.to_string())
print(json.dumps(imb, ensure_ascii=False, indent=1)[:1500])
print(json.dumps(before), json.dumps(after))
print(by_cat.round(2).to_string())
print(lat.round(3).to_string())
