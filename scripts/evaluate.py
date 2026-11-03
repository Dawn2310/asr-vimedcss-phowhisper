"""Decode one split and score it. Output: experiments/<ID>/eval/<split>/{predictions.jsonl, metrics.json}

    python scripts/evaluate.py B0_phowhisper_small_zeroshot --split dev
    python scripts/evaluate.py V1_phowhisper_small_lora --split test --allow-test
    python scripts/evaluate.py O1_phowhisper_large_lora_official --num-beams 5   # -> eval/dev_beam5
    python scripts/evaluate.py C_whisper_large_v3_lora_official_termprompt --split hard --allow-test \
        --bias oracle --distractors 30 --sources ViMedCSS                        # -> eval/hard_bias-oracle30_ViMedCSS

--bias decodes one clip at a time with a term-list prompt (asr/data.py::TermPrompter); distractors come from the
experiment's TRAIN term vocabulary, seeded per clip id:
  oracle    the clip's annotated terms + N distractors (the usual contextual-biasing protocol: a known term list)
  distract  N distractors only (no annotated term of the clip): measures false insertions
  video     every annotated term of the clip's video within the evaluated split (a per-recording keyword list)
  retrieve  no annotation of the clip: two-pass decoding. The N lexicon terms closest to this model's unprompted
            hypothesis (eval/<split>/predictions.jsonl, must exist) by character trigrams, ignoring accents and spaces.
            Lexicon = TRAIN vocabulary + --lexicon (a fixed external list). Without --lexicon: TRAIN vocabulary + all
            terms annotated in the evaluated split, a closed list that holds every right answer (upper bound only).
"""
import argparse
import heapq
import json
import math
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.config import ROOT, load_experiment, log_run
from asr.data import SR, TermPrompter, decode_audio, load_split, train_terms
from asr.metrics import compute_metrics, split_terms
from asr.model import load_model

LOCKED_SPLITS = {"test", "hard", "probe"}  # final evaluation only, never for model selection
BIAS_MAX_TOKENS = 160  # prompt length when the experiment sets no data.term_prompt.max_tokens


def token_cap(ev: dict, duration: float) -> int:
    """Decode guard: max_new_tokens grows with clip length (eval.max_tokens_per_second, fit on TRAIN refs)."""
    if ev.get("max_tokens_per_second") is None:
        return ev["max_new_tokens"]
    return min(ev["max_new_tokens"], math.ceil(ev["max_tokens_per_second"] * duration) + ev["max_tokens_offset"])


def trigrams(text: str) -> set[str]:
    """Character trigrams without accents and spaces: 'vắc xin' and 'vacxin' share all of theirs."""
    s = "".join(c for c in unicodedata.normalize("NFD", text.replace("đ", "d")) if not unicodedata.combining(c)).replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)} or {s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--checkpoint", help="default: experiments/<ID>/final if the experiment trains")
    ap.add_argument("--max-per-source", type=int, help="quick look on a subset")
    ap.add_argument("--data-dir")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--num-beams", type=int, help="override eval.num_beams; output goes to eval/<split>_beam<N>")
    ap.add_argument("--bias", choices=["oracle", "distract", "video", "retrieve"], help="term-list prompt (see module doc)")
    ap.add_argument("--distractors", type=int, default=30)
    ap.add_argument("--lexicon", help="--bias retrieve: external term list, one per line, path relative to ASR/")
    ap.add_argument("--sources", nargs="+", help="override data.eval_sources (output name gets the source list)")
    args = ap.parse_args()
    if args.split in LOCKED_SPLITS and not args.allow_test:
        sys.exit(f"'{args.split}' is locked. Freeze model/config first, then pass --allow-test.")

    cfg, exp_dir = load_experiment(args.exp_id, args.data_dir)
    ev = cfg["eval"]
    if args.num_beams:
        ev["num_beams"] = args.num_beams
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif ev.get("checkpoint"):  # evaluation-only experiment of an existing adapter (path relative to ASR/)
        ckpt = ROOT / ev["checkpoint"]
    else:
        ckpt = exp_dir / "final" if cfg["train"] else None
    if ckpt is not None and not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, processor = load_model(cfg, checkpoint=ckpt)
    model.to(device, dtype).eval()

    if args.sources:
        cfg["data"]["eval_sources"] = args.sources
    ds = load_split(cfg, args.split, args.max_per_source)
    prompter = None
    if args.bias:
        ev["batch_size"] = 1  # generate() takes one prompt for the whole batch
        prompter = TermPrompter(processor, train_terms(cfg),
                                {"max_tokens": (cfg["data"].get("term_prompt") or {}).get("max_tokens", BIAS_MAX_TOKENS)})
        video_terms = {}
        for g, s in zip(ds["group_id"], ds["cs_terms"]):
            video_terms.setdefault(g, {}).update(dict.fromkeys(split_terms(s)))
    if args.bias == "retrieve":
        first_pass = exp_dir / "eval" / args.split / "predictions.jsonl"
        if not first_pass.is_file():
            sys.exit(f"--bias retrieve needs the unprompted decode: {first_pass}")
        with open(first_pass, encoding="utf-8") as f:
            first_hyp = {r["id"]: r["hyp"] for r in map(json.loads, f)}
        listed = (ROOT / args.lexicon).read_text(encoding="utf-8").split("\n") if args.lexicon else \
            [t for ts in video_terms.values() for t in ts]
        lexicon = [(t, trigrams(t)) for t in sorted(set(filter(None, listed)) | set(prompter.vocab))]

    def bias_terms(clip_id, cs_terms, group):
        rng = random.Random(f"{args.bias}:{clip_id}")
        gold = list(dict.fromkeys(split_terms(cs_terms)))
        if args.bias == "oracle":
            terms = gold + prompter.distractors(gold, args.distractors, rng)
        elif args.bias == "distract":
            terms = prompter.distractors(gold, args.distractors, rng)
        elif args.bias == "retrieve":
            hyp = trigrams(first_hyp[clip_id])
            terms = [t for t, _ in heapq.nsmallest(args.distractors, lexicon, key=lambda x: (-len(x[1] & hyp) / len(x[1]), x[0]))]
        else:
            terms = list(video_terms[group])
        rng.shuffle(terms)
        return terms

    silence_peak = ev.get("silence_peak")  # decode guard: max |wav| below this -> empty hyp, model not run
    eos, sot = model.generation_config.eos_token_id, model.generation_config.decoder_start_token_id
    rows, compute_s = [], 0.0
    for i, batch in enumerate(ds.iter(batch_size=ev["batch_size"])):
        waves = [decode_audio(a) for a in batch["audio"]]
        silent = [silence_peak is not None and (len(w) == 0 or float(np.abs(w).max()) < silence_peak) for w in waves]
        keep = [k for k, s in enumerate(silent) if not s]
        hyps, truncated = [""] * len(waves), [False] * len(waves)
        prompt_kw, kept = {}, []
        if prompter is not None:
            pids, kept = prompter.build(bias_terms(batch["id"][0], batch["cs_terms"][0], batch["group_id"][0]))
            prompt_kw["prompt_ids"] = torch.tensor(pids, device=device)
        if keep:
            feats = processor.feature_extractor([waves[k] for k in keep], sampling_rate=SR,
                                                return_tensors="pt").input_features.to(device, dtype)
            cap = max(token_cap(ev, batch["duration"][k]) for k in keep)
            start = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(feats, language=cfg["model"]["language"], task=cfg["model"]["task"],
                                     num_beams=ev["num_beams"], max_new_tokens=cap, **prompt_kw)
            compute_s += time.perf_counter() - start
            if prompt_kw:  # drop any echoed prompt: keep the sequence from its last <|startoftranscript|>
                ids = [seq[(seq == sot).nonzero()[-1, 0]:] if (seq == sot).any() else seq for seq in ids]
            for k, seq, hyp in zip(keep, ids, processor.batch_decode(ids, skip_special_tokens=True)):
                # Text token ids are < eos (specials start there); `cap` text tokens means EOS never came.
                hyps[k], truncated[k] = hyp, int((seq < eos).sum()) >= cap
        for k in range(len(waves)):
            rows.append({key: batch[key][k] for key in ("id", "source", "text", "cs_terms", "duration")}
                        | {"hyp": hyps[k].strip(), "silent": silent[k], "truncated": truncated[k]}
                        | ({"prompt_terms": kept} if prompter is not None else {}))
        print(f"\r{len(rows)}/{len(ds)}", end="", flush=True)
    print()

    def score(subset):
        m = compute_metrics([r["text"] for r in subset], [r["hyp"] for r in subset], [r["cs_terms"] for r in subset])
        m["audio_hours"] = sum(r["duration"] for r in subset) / 3600
        m["n_silent"] = sum(r["silent"] for r in subset)
        m["n_truncated"] = sum(r["truncated"] for r in subset)
        return m

    metrics = {"overall": score(rows),
               "by_source": {s: score([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
               "rtf_decode_only": compute_s / sum(r["duration"] for r in rows),
               "checkpoint": str(ckpt) if ckpt else cfg["model"]["name"],
               "decode": {"num_beams": ev["num_beams"], "batch_size": ev["batch_size"], "dtype": str(dtype), "device": device,
                          **{k: ev.get(k) for k in ("silence_peak", "max_tokens_per_second", "max_tokens_offset")},
                          "bias": args.bias, "distractors": args.distractors if args.bias not in (None, "video") else None,
                          "lexicon": args.lexicon}}

    name = args.split + (f"_beam{args.num_beams}" if args.num_beams else "")
    if args.bias:
        name += f"_bias-{args.bias}" + (str(args.distractors) if args.bias != "video" else "")
        if args.lexicon:
            name += "-" + Path(args.lexicon).stem
    if args.sources:
        name += "_" + "-".join(args.sources)
    out = exp_dir / "eval" / (name if not args.max_per_source else f"{name}_sub{args.max_per_source}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log_run(cfg, exp_dir, f"eval:{out.name}", checkpoint=metrics["checkpoint"])
    print(json.dumps({"overall_wer": metrics["overall"]["wer"],
                      "by_source_wer": {s: m["wer"] for s, m in metrics["by_source"].items()}}, indent=2))


if __name__ == "__main__":
    main()
