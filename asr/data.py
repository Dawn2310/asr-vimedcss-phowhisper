"""Load prepared parquet splits and turn them into Whisper training batches."""
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from datasets import Dataset, concatenate_datasets, load_dataset

from asr.metrics import normalize_text, split_terms

SR = 16000
MAX_TARGET_POSITIONS = 448  # Whisper decoder length: prompt + prefix + text + EOT must fit


def decode_audio(flac_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
    assert sr == SR, f"expected {SR} Hz, got {sr} (re-run prepare_data.py)"
    return wav


def _cap_hours(ds: Dataset, max_hours: float, seed: int) -> Dataset:
    ds = ds.shuffle(seed=seed)
    n = int(np.searchsorted(np.cumsum(ds["duration"]), max_hours * 3600))
    return ds.select(range(n))


def _oversample_rare_terms(ds: Dataset, opt: dict) -> Dataset:
    """data.term_oversample {source, max_count, repeat}: clips holding a cs_term seen <= max_count times in this
    source's TRAIN appear `repeat` times in total (resampling against the long tail of medical terms).
    random_control: true repeats as many random clips instead, same step count without the term targeting."""
    terms = [split_terms(s) for s in ds["cs_terms"]]
    counts = {}
    for ts in terms:
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
    rare = [i for i, ts in enumerate(terms) if any(counts[t] <= opt["max_count"] for t in ts)]
    print(f"Oversample: {len(rare)} clips with a term seen <= {opt['max_count']} times, x{opt['repeat']}", flush=True)
    if opt.get("random_control"):
        rare = random.Random(13).sample(range(len(terms)), len(rare))
    return concatenate_datasets([ds] + [ds.select(rare)] * (opt["repeat"] - 1)) if rare else ds


def load_split(cfg: dict, split: str, max_per_source: int | None = None) -> Dataset:
    """train: data.train_sources {source: max_hours|null}; other splits: data.eval_sources (missing files skipped)."""
    d, seed = cfg["data"], cfg["seed"]
    sources = d["train_sources"] if split == "train" else dict.fromkeys(d["eval_sources"])
    exclude = set()
    if split == "train" and d.get("train_exclude"):  # json {"ids": {id: reason}} next to the parquet sources
        exclude = set(json.loads((Path(d["dir"]) / d["train_exclude"]).read_text(encoding="utf-8"))["ids"])
    parts = []
    for source, max_hours in sources.items():
        path = Path(d["dir"]) / source / f"{split}.parquet"
        if not path.exists():
            if split == "train":
                raise FileNotFoundError(path)
            continue
        ds = load_dataset("parquet", data_files=str(path), split="train")
        if exclude:
            n = len(ds)
            ds = ds.filter(lambda i: i not in exclude, input_columns="id")
            print(f"Exclude {source}/{split}: {n - len(ds)} rows listed in {d['train_exclude']}", flush=True)
        empty_indices = [i for i, dur in enumerate(ds["duration"]) if dur <= 0]
        if empty_indices:
            print(f"Exclude {source}/{split}: {len(empty_indices)} empty-audio rows "
                  f"(duration <= 0): {ds.select(empty_indices)['id']}", flush=True)
        ds = ds.filter(lambda dur: 0 < dur <= d["max_duration"], input_columns="duration")
        if max_hours is not None:
            ds = _cap_hours(ds, max_hours, seed)
        if split == "train" and source == (d.get("term_oversample") or {}).get("source"):
            ds = _oversample_rare_terms(ds, d["term_oversample"])
        if max_per_source and len(ds) > max_per_source:
            ds = ds.shuffle(seed=seed).select(range(max_per_source))
        parts.append(ds)
    if not parts:
        raise FileNotFoundError(f"No '{split}' parquet for {list(sources)} in {d['dir']}")
    return concatenate_datasets(parts)


def train_terms(cfg: dict) -> list[str]:
    """Distinct normalized cs_terms in the TRAIN parquet of every train source (the term-prompt vocabulary)."""
    d, terms = cfg["data"], set()
    for source in d["train_sources"]:
        for s in pq.read_table(Path(d["dir"]) / source / "train.parquet", columns=["cs_terms"]).column("cs_terms").to_pylist():
            terms.update(split_terms(s))
    return sorted(terms)


class TermPrompter:
    """Whisper previous-text prompt listing medical terms: ' term, term, ...' after <|startofprev|>.

    Training (data.term_prompt {p, max_distractors, gold_keep, max_tokens}): with probability p a clip gets its own
    terms (each kept with probability gold_keep) mixed with 0..max_distractors random TRAIN-vocabulary terms, so the
    decoder learns to copy a listed term when it is spoken and to ignore listed terms that are not. Evaluation builds
    the list itself (scripts/evaluate.py --bias)."""

    def __init__(self, processor, vocab: list[str], opt: dict):
        self.tok, self.vocab, self.opt = processor.tokenizer, vocab, opt
        self.prev = self.tok.convert_tokens_to_ids("<|startofprev|>")

    def build(self, terms: list[str]) -> tuple[list[int], list[str]]:
        """Prompt token ids and the terms they hold; terms that would push it past max_tokens are left out
        (callers shuffle first, so which ones is random)."""
        out, kept = [self.prev], []
        for t in terms:
            piece = self.tok((", " if kept else " ") + t, add_special_tokens=False).input_ids
            if len(out) + len(piece) <= self.opt["max_tokens"]:
                out += piece
                kept.append(t)
        return out, kept

    def distractors(self, exclude: list[str], k: int, rng) -> list[str]:
        pool = [t for t in self.vocab if t not in exclude]
        return rng.sample(pool, min(k, len(pool)))

    def sample(self, cs_terms: str, rng) -> list[int] | None:
        o = self.opt
        if rng.random() >= o["p"]:
            return None
        gold = [t for t in dict.fromkeys(split_terms(cs_terms)) if rng.random() < o["gold_keep"]]
        terms = gold + self.distractors(gold, rng.randint(0, o["max_distractors"]), rng)
        rng.shuffle(terms)
        return self.build(terms)[0]


class Featurizer:
    """On-the-fly transform (ds.with_transform): audio -> log-mel, text -> label ids.

    With a TermPrompter the decoder input is [prompt] + [SOT, lang, task, notimestamps] + text and the loss skips the
    prompt and SOT, so it also returns decoder_input_ids (Collator then uses them instead of shifting the labels)."""

    def __init__(self, processor, normalize: bool, prompter: TermPrompter | None = None):
        self.processor, self.normalize, self.prompter = processor, normalize, prompter

    def __call__(self, batch: dict) -> dict:
        feats = self.processor.feature_extractor(
            [decode_audio(b) for b in batch["audio"]], sampling_rate=SR
        ).input_features
        texts = [normalize_text(t) if self.normalize else t for t in batch["text"]]
        labels = self.processor.tokenizer(texts).input_ids
        if self.prompter is None:
            return {"input_features": feats, "labels": labels}
        dec, lab = [], []
        for seq, terms in zip(labels, batch["cs_terms"]):
            prompt = self.prompter.sample(terms, random) or []  # dataloader workers seed `random` per worker
            if len(prompt) + len(seq) > MAX_TARGET_POSITIONS:
                prompt = []
            dec.append(prompt + seq[:-1])
            lab.append([-100] * len(prompt) + seq[1:])
        return {"input_features": feats, "decoder_input_ids": dec, "labels": lab}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        if "decoder_input_ids" in features[0]:  # term-prompt batches: explicit decoder input, prompt masked in labels
            n, pad = max(len(f["labels"]) for f in features), self.processor.tokenizer.pad_token_id
            batch["decoder_input_ids"] = torch.tensor([f["decoder_input_ids"] + [pad] * (n - len(f["labels"])) for f in features])
            batch["labels"] = torch.tensor([f["labels"] + [-100] * (n - len(f["labels"])) for f in features])
            return batch
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all():  # model prepends it itself
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def to_device_features(processor, audios: list[bytes], device: str, dtype: torch.dtype) -> torch.Tensor:
    feats = processor.feature_extractor([decode_audio(a) for a in audios], sampling_rate=SR, return_tensors="pt")
    return feats.input_features.to(device, dtype)
