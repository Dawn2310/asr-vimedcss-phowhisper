"""Data + terrain image for the PowerPoint build (same log-mel and algorithm as the web cover)."""
import csv, json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

S = Path(__file__).parent
T = Path(r"D:\1.FPT_FALL26\DPL - Deep Learning\ASR\reports\task2_20261030\tables")
curves = defaultdict(list)
for r in csv.DictReader(open(T / "eval_curves.csv", encoding="utf-8")):
    curves[r["run"]].append([int(r["step"]), round(float(r["eval_wer"]), 2)])
subs = [[r["ref"], r["hyp"], int(r["count"])] for r in csv.DictReader(open(T / "top_substitutions.csv", encoding="utf-8"))]
mt = json.load(open(S.parent / "mt_samples.json", encoding="utf-8"))["rows"]
json.dump(dict(curves=curves, subs=subs, mt=mt), open(S / "data.json", "w", encoding="utf-8"), ensure_ascii=False)

m = json.load(open(S.parent / "mel.json", encoding="utf-8"))
f, t, v = m["f"], m["t"], m["v"]
SC = 2  # supersample
W, H = 1920 * SC, 1080 * SC
im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
g = ImageDraw.Draw(im)
x0, dx, top, gap, sk, amp = 290, (1920 - 290 - 30) / t, 470, 11, -5.2, 150
for r in range(f - 1, -1, -1):
    by = top + (f - 1 - r) * gap
    ox = x0 + (f - 1 - r) * sk
    pts = [(ox, by + 40)] + [(ox + k * dx, by - (v[r * t + k] / 99) ** 2.3 * amp) for k in range(t)] + [(ox + (t - 1) * dx, by + 40)]
    g.polygon([(a * SC, b * SC) for a, b in pts], fill=(10, 58, 66, 255))
    for k in range(1, t):
        a = v[r * t + k] / 99
        y0 = by - (v[r * t + k - 1] / 99) ** 2.3 * amp
        y1 = by - a ** 2.3 * amp
        col = (255, 178, 26) if a > .72 else (143, 196, 191) if a > .5 else (44, 118, 128)
        g.line([((ox + (k - 1) * dx) * SC, y0 * SC), ((ox + k * dx) * SC, y1 * SC)], fill=col, width=int((2.2 if a > .72 else 1.4) * SC))
im = im.resize((1920, 1080), Image.LANCZOS)
im.save(S / "terrain.png")
print("ok", {k: len(x) for k, x in curves.items()})
