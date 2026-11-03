"""Train one experiment. Checkpoints go to experiments/<ID>/checkpoints and resume automatically.

    python scripts/train.py V1_phowhisper_small_lora [--data-dir /content/data] [--smoke]

--smoke: 2 optimizer steps on 8 clips per source, outputs under experiments/<ID>/smoke (checks the pipeline only).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.config import load_experiment, log_run
from asr.data import Collator, Featurizer, TermPrompter, load_split, train_terms
from asr.metrics import compute_metrics
from asr.model import load_model


class ResumableTrainer(Seq2SeqTrainer):
    def _load_rng_state(self, checkpoint):
        # transformers 4.46 predates torch >= 2.6 defaulting torch.load to weights_only=True, which rejects
        # rng_state.pth (numpy state) and breaks every resume. Only states written by this run are deserialised.
        if checkpoint is None:
            return
        path = Path(checkpoint).resolve()
        if not path.is_relative_to(Path(self.args.output_dir).resolve()):
            raise ValueError("RNG checkpoint must belong to this experiment")
        rng_file = path / "rng_state.pth"
        if not rng_file.is_file():
            return
        state = torch.load(rng_file, map_location="cpu", weights_only=False)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["cpu"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.random.set_rng_state(state["cuda"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--data-dir", help="override data.dir (e.g. local copy on Colab)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-workers", type=int, help="override train.num_workers (several trainings sharing one machine)")
    args = ap.parse_args()

    cfg, exp_dir = load_experiment(args.exp_id, args.data_dir)
    t, ev = cfg["train"], cfg["eval"]
    if not t:
        sys.exit(f"{args.exp_id} has no 'train' section -> evaluation-only, use scripts/evaluate.py")
    set_seed(cfg["seed"])
    out_dir = exp_dir / "smoke" if args.smoke else exp_dir

    model, processor = load_model(cfg, for_training=True)
    featurize = Featurizer(processor, cfg["data"]["normalize_text"])
    term_prompt = cfg["data"].get("term_prompt")  # in-training dev eval stays unprompted
    prompter = TermPrompter(processor, train_terms(cfg), term_prompt) if term_prompt else None
    train_ds = load_split(cfg, "train", 8 if args.smoke else None).with_transform(
        Featurizer(processor, cfg["data"]["normalize_text"], prompter))
    dev_ds = load_split(cfg, "dev", 2 if args.smoke else cfg["data"]["dev_max_per_source"]).with_transform(featurize)
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    pad_id = processor.tokenizer.pad_token_id

    def metrics_fn(pred):
        preds = np.where(pred.predictions == -100, pad_id, pred.predictions)
        labels = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
        m = compute_metrics(processor.batch_decode(labels, skip_special_tokens=True),
                            processor.batch_decode(preds, skip_special_tokens=True))
        return {"wer": m["wer"], "cer": m["cer"]}

    # train.eval_strategy "epoch" evaluates (and saves) after every epoch, including the last one, which a
    # steps interval only does when it divides the total step count.
    strategy = "steps" if args.smoke else t.get("eval_strategy", "steps")
    steps = {} if strategy == "epoch" else {"eval_steps": 1 if args.smoke else t["eval_steps"],
                                            "save_steps": 1 if args.smoke else t["save_steps"]}
    ckpt_dir = out_dir / "checkpoints"
    workers = 0 if args.smoke else args.num_workers or t["num_workers"]  # 8 spawned workers hang on the Windows laptop
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        seed=cfg["seed"], data_seed=cfg["seed"],
        num_train_epochs=t["epochs"], max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=2 if args.smoke else t["batch_size"],
        per_device_eval_batch_size=t.get("eval_batch_size", ev["batch_size"]),
        gradient_accumulation_steps=1 if args.smoke else t["grad_accum"],
        learning_rate=t["lr"], warmup_ratio=t["warmup_ratio"], lr_scheduler_type="linear",
        fp16=t["fp16"] and not t.get("bf16"), bf16=bool(t.get("bf16")),
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=strategy, save_strategy=strategy, **steps, save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=ev["max_new_tokens"], generation_num_beams=1,
        logging_steps=1 if args.smoke else t["logging_steps"], report_to="none",
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=workers, dataloader_persistent_workers=workers > 0,
        dataloader_prefetch_factor=4 if workers else None,
    )
    trainer = ResumableTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=metrics_fn, processing_class=processor.feature_extractor,
    )

    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    log_run(cfg, exp_dir, "train-smoke" if args.smoke else "train", resume_from=last, n_train=len(train_ds), n_dev=len(dev_ds))
    print(f"resume_from_checkpoint={last}")
    trainer.train(resume_from_checkpoint=last)

    final = out_dir / "final"  # best checkpoint by dev WER (LoRA: adapter only)
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print(f"saved best model -> {final}")


if __name__ == "__main__":
    main()
