"""Copy the results of local runs (local/runs, not committed: it also holds 6+ GB of weights) into results/local_runs,
so the repository carries every local number without any model file.

    python scripts/export_local_results.py

Per run (local/runs/<ID>/<tag>/): manifest.json, runs.jsonl, eval/<split>/{metrics.json,predictions.jsonl} and the
trainer_state.json of the last checkpoint (full loss/eval history). Smoke and validation tags (pipeline tests on a few
clips) and resources.jsonl (GPU telemetry) are skipped. Also copies local/runs/comparison.csv.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "local/runs", ROOT / "results/local_runs"
SKIP_TAG = re.compile(r"smoke|validation")


def last_trainer_state(run):
    states = [(int(p.parent.name.split("-")[1]), p) for p in run.glob("checkpoints/checkpoint-*/trainer_state.json")]
    return max(states)[1] if states else None


def main():
    if DST.exists():
        shutil.rmtree(DST)
    files = []
    for run in sorted(p for p in SRC.glob("*/*") if p.is_dir() and not SKIP_TAG.search(p.name)):
        picks = [run / "manifest.json", run / "runs.jsonl", *sorted(run.glob("eval/*/metrics.json")),
                 *sorted(run.glob("eval/*/predictions.jsonl"))]
        files += [(p, p.relative_to(SRC)) for p in picks if p.exists()]
        state = last_trainer_state(run)
        if state:
            files.append((state, run.relative_to(SRC) / "trainer_state.json"))
    files.append((SRC / "comparison.csv", Path("comparison.csv")))
    for src, rel in files:
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, DST / rel)
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"{len(files)} files, {size / 1e6:.1f} MB -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
