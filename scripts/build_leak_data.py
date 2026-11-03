"""Leakage study data (paper part A): same test clips, two equally sized TRAIN sets that differ only in whether other
clips of the test clips' videos are in them. Needs data/ViMedCSS_vsplit (scripts/build_v3_data.py). Existing files
are not modified.

    python scripts/build_leak_data.py [--data-dir data]

Pool = videos of the vsplit DEV and TEST with >= 2 clips (no clip of them is in the vsplit TRAIN). Each pool video is
assigned by sha256(seed:leak:video) to
- S (seen-in-leaky): half of its clips (sha256 order) go to the leak set L, the rest to probe (source ViMedCSS_probe_seen);
- U (control): all its clips go to probe (source ViMedCSS_probe_unseen).
Clean TRAIN = vsplit TRAIN. Leaky TRAIN = vsplit TRAIN minus |L| clips (sha256 order) + L, so both have the same clip
count, the other three sources are shared and only the video origin of |L| clips differs. A leaky model should beat
the clean one on probe_seen but not on probe_unseen (difference-in-differences). In the official ViMedCSS split 64%
of a test clip's video is in TRAIN on average; here 50% (conservative).

Writes data/ViMedCSS_leak/{train,probe}.parquet (train = L), data/leak_train_exclude.json (v3 exclusions + the dropped
vsplit TRAIN clips; used by the leaky experiments only) and data/ViMedCSS_leak/split_report.json (id hashes, so a copy
rebuilt elsewhere can be checked against this one), then refreshes data/registry.json.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import normalize_text
from build_v3_data import MAX_TOKENS_PER_SECOND, SEED, batches
from prepare_data import SCHEMA, write_registry

MAX_DURATION = 30.0  # data.max_duration: longer clips never reach training, so they cannot leak


def h(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest(), 16)


def id_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, SCHEMA) as w:
        for i in range(0, len(rows), 256):
            w.write_table(pa.Table.from_pylist(rows[i:i + 256], SCHEMA))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    data = Path(ap.parse_args().data_dir)
    vs, out = data / "ViMedCSS_vsplit", data / "ViMedCSS_leak"
    out.mkdir(exist_ok=True)
    v3_exclude = json.loads((data / "v3_train_exclude.json").read_text(encoding="utf-8"))

    from transformers import WhisperTokenizer  # same tokenizer and label-density rule as v3_train_exclude.json
    tok = WhisperTokenizer.from_pretrained("vinai/PhoWhisper-small")

    def trainable(r) -> bool:
        if not r["duration"] or not 0 < r["duration"] <= MAX_DURATION:
            return False
        n = max(len(tok(t, add_special_tokens=False).input_ids) for t in (r["text"], normalize_text(r["text"])))
        return n / r["duration"] <= MAX_TOKENS_PER_SECOND

    pool = defaultdict(list)
    for split in ("dev", "test"):
        for r in batches(vs / f"{split}.parquet"):
            pool[r["group_id"]].append(r)
    pool = {v: rows for v, rows in pool.items() if len(rows) >= 2}

    leak, probe, skipped = [], [], []
    for video in sorted(pool):
        rows = sorted(pool[video], key=lambda r: h("clip", r["id"]))
        if h("leak", video) / 16 ** 64 < 0.5:
            half = len(rows) // 2
            mine = [r | {"source": "ViMedCSS_leak", "split": "train"} for r in rows[:half] if trainable(r)]
            if not mine:  # nothing of this video could be trained on, so its probe clips would not be "seen"
                skipped.append(video); continue
            leak += mine
            probe += [r | {"source": "ViMedCSS_probe_seen", "split": "probe"} for r in rows[half:]]
        else:
            probe += [r | {"source": "ViMedCSS_probe_unseen", "split": "probe"} for r in rows]

    # size match: drop |L| vsplit TRAIN clips that would otherwise be trained on
    candidates = [r for r in batches(vs / "train.parquet", ["id", "duration"])
                  if r["id"] not in v3_exclude["ids"] and r["duration"] and 0 < r["duration"] <= MAX_DURATION]
    dropped = sorted(candidates, key=lambda r: h("drop", r["id"]))[:len(leak)]

    write(out / "train.parquet", leak)
    write(out / "probe.parquet", probe)
    (data / "leak_train_exclude.json").write_text(json.dumps(
        {"rule": v3_exclude["rule"] + f"; plus {len(dropped)} ViMedCSS_vsplit TRAIN clips dropped so the leaky TRAIN "
                                      "(+ ViMedCSS_leak) has as many clips as the clean one (scripts/build_leak_data.py)",
         "counts": v3_exclude["counts"] | {"leak_size_match": len(dropped)},
         "ids": v3_exclude["ids"] | {r["id"]: "leak-study size match" for r in dropped}}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    train_videos = {r["group_id"] for r in batches(vs / "train.parquet", ["group_id"])}
    leak_videos = {r["group_id"] for r in leak}
    seen = [r for r in probe if r["source"] == "ViMedCSS_probe_seen"]
    unseen = [r for r in probe if r["source"] == "ViMedCSS_probe_unseen"]
    assert not ({r["group_id"] for r in probe} & train_videos), "probe video in the clean TRAIN"
    assert not ({r["group_id"] for r in unseen} & leak_videos), "control video leaked"
    assert not ({r["id"] for r in probe} & {r["id"] for r in leak}), "probe clip in L"
    assert {r["group_id"] for r in seen} <= leak_videos, "probe_seen video without a leaked clip"

    def stats(rows):
        return {"clips": len(rows), "videos": len({r["group_id"] for r in rows}),
                "hours": round(sum(r["duration"] for r in rows) / 3600, 3)}
    report = {
        "seed": SEED, "pool": "vsplit dev+test videos with >= 2 clips", "leak_share_of_seen_videos": 0.5,
        "leak_train": stats(leak), "probe_seen": stats(seen), "probe_unseen": stats(unseen),
        "seen_videos_skipped_no_trainable_leak_clip": skipped,
        "dropped_from_vsplit_train": {"clips": len(dropped), "hours": round(sum(r["duration"] for r in dropped) / 3600, 3)},
        "id_sha256": {"leak_train": id_hash(r["id"] for r in leak), "probe": id_hash(r["id"] for r in probe),
                      "dropped": id_hash(r["id"] for r in dropped),
                      **{f"vsplit_{s}": id_hash(r["id"] for r in batches(vs / f"{s}.parquet", ["id"]))
                         for s in ("train", "dev", "test")}},
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_registry(data)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
