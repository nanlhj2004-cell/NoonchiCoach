import argparse
import json
import hashlib
import math
from pathlib import Path
import tempfile
from importlib.metadata import version

from datasets import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from packaging.version import Version
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from train import metrics
from common import LABEL_TO_ID, SPLITS


def load_splits(path):
    frame = pd.read_parquet(path)
    frame["labels"] = frame.label.map(LABEL_TO_ID)
    for column in ("context", "response"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Missing or blank {column} in preprocessed data")
    if frame.labels.isna().any() or frame.duplicated(["context", "response"]).any():
        raise ValueError("Unknown labels or duplicate inputs in preprocessed data")
    if set(frame.split.unique()) != set(SPLITS):
        raise ValueError("Expected nonempty train, validation and test splits")
    if any(group.labels.nunique() != 2 for _, group in frame.groupby("split")):
        raise ValueError("Each split must contain both classes")
    return {split: frame[frame.split.eq(split)].reset_index(drop=True) for split in SPLITS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("koelectra_results"))
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), default=8)
    args = parser.parse_args()
    if Version(torch.__version__.split("+", 1)[0]) < Version("2.6"):
        raise RuntimeError("Use a Colab GPU runtime with PyTorch 2.6 or newer to load the KoELECTRA checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running this experiment.")
    subsets = load_splits(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    with tempfile.TemporaryDirectory(prefix="noonchi-checkpoints-") as scratch:
        train_and_evaluate(args, subsets, scratch)


def train_and_evaluate(args, subsets, scratch):
    checkpoint = "monologg/koelectra-base-v3-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    distributions, lengths = {}, {}
    for split, subset in subsets.items():
        ids = tokenizer(list(subset.context), list(subset.response), truncation=False, verbose=False)["input_ids"]
        lengths[split] = pd.Series([len(x) for x in ids])
    max_length = min(tokenizer.model_max_length, 512,
                     max(64, 8 * math.ceil(lengths["train"].quantile(.95) / 8)))
    for split, values in lengths.items():
        distributions[split] = dict(summary=values.describe(percentiles=[.5, .9, .95, .99]).to_dict(),
                                    truncated=int(values.gt(max_length).sum()))
    (args.output / "token_lengths.json").write_text(json.dumps(
        dict(max_length=max_length, splits=distributions), indent=2), encoding="utf-8")
    for split, values in lengths.items():
        plt.hist(values, bins=35, histtype="step", density=True, label=split)
    plt.axvline(max_length, linestyle="--", label="Train-derived cutoff")
    plt.xlabel("KoELECTRA input tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "token_lengths.png")
    plt.close()

    def encode(batch):
        return tokenizer(batch["context"], batch["response"], truncation=True, max_length=max_length)

    encoded = {}
    for split, subset in subsets.items():
        dataset = Dataset.from_pandas(subset[["context", "response", "labels"]], preserve_index=False)
        encoded[split] = dataset.map(encode, batched=True, remove_columns=["context", "response"])
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, label2id=LABEL_TO_ID,
        id2label={value: name for name, value in LABEL_TO_ID.items()})
    training = TrainingArguments(
        output_dir=scratch, num_train_epochs=3, save_safetensors=True,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=16 // args.batch_size,
        learning_rate=2e-5, weight_decay=.01, warmup_ratio=.1, fp16=True, logging_steps=50,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        seed=42, report_to="none",
    )

    def compute(pred):
        return {k: v for k, v in metrics(pred.label_ids, pred.predictions.argmax(axis=-1)).items()
                if k != "confusion"}

    trainer = Trainer(model=model, args=training, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute)
    trainer.train()
    trainer.save_model(str(args.output / "final"))
    tokenizer.save_pretrained(str(args.output / "final"))
    pred = trainer.predict(encoded["test"])
    predicted = pred.predictions.argmax(axis=-1)
    errors = subsets["test"][["source_row_zero_based", "label"]].copy()
    errors["prediction"] = [model.config.id2label[int(label)] for label in predicted]
    errors["sarcasm_score"] = torch.as_tensor(pred.predictions).float().softmax(dim=-1)[:, 1].numpy()
    errors["error_type"] = [("FP" if guess == 1 else "FN") if guess != actual else ""
                            for guess, actual in zip(predicted, pred.label_ids, strict=True)]
    errors.loc[predicted != pred.label_ids].to_csv(args.output / "errors_test.csv", index=False)
    report = dict(test=metrics(pred.label_ids, predicted), log=trainer.state.log_history,
                  checkpoint=checkpoint, max_length=max_length, best_validation_macro_f1=trainer.state.best_metric,
                  best_checkpoint=Path(trainer.state.best_model_checkpoint).name,
                  input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  label_to_id=LABEL_TO_ID, gpu=torch.cuda.get_device_name(),
                  batch_size=args.batch_size, effective_batch_size=16, seed=42,
                  package_versions={name: version(name) for name in
                                    ("torch", "transformers", "datasets", "accelerate")})
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(pred.label_ids, predicted, labels=[0, 1],
                                           display_labels=list(LABEL_TO_ID))
    plt.tight_layout()
    plt.savefig(args.output / "confusion.png")
    plt.close()
    evaluations = [row for row in trainer.state.log_history if "eval_macro_f1" in row]
    plt.plot([row["epoch"] for row in evaluations], [row["eval_macro_f1"] for row in evaluations], "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro F1")
    plt.tight_layout()
    plt.savefig(args.output / "validation.png")
    plt.close()
    for field in ("loss", "eval_loss"):
        points = [(row["step"], row[field]) for row in trainer.state.log_history if field in row]
        if points:
            steps, values = zip(*points)
            plt.plot(steps, values, label=field)
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "loss.png")
    plt.close()
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
