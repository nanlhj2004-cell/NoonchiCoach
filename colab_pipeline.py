"""Dataset preparation and held-out evaluation for the integrated notebook."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from common import LABEL_TO_ID, SPLITS, normalize
from train import metrics


def download(root):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download("YuminKim/KoCoSa", repo_type="dataset",
                                   allow_patterns=["data/*.parquet"], cache_dir=root / "hub")) / "data"
    for split in SPLITS:
        if not (source / f"{split}-00000-of-00001.parquet").is_file():
            raise FileNotFoundError(f"KoCoSa split missing: {split}")
    external = load_dataset("SpellOnYou/kor_sarcasm", cache_dir=str(root / "datasets"))
    if {s: len(d) for s, d in external.items()} != {"train": 9000, "test": 301}:
        raise ValueError("Unexpected published Kocasm split sizes")
    for dataset in external.values():
        if getattr(dataset.features["label"], "names", None) != ["no_sarcasm", "sarcasm"]:
            raise ValueError("Unexpected Kocasm ClassLabel metadata; verify label semantics before evaluation")
    expected = {"train": 10466, "validation": 1321, "test": 1037}
    import pyarrow.parquet as parquet
    counts = {s: parquet.ParquetFile(source / f"{s}-00000-of-00001.parquet").metadata.num_rows
              for s in SPLITS}
    if counts != expected:
        raise ValueError(f"Unexpected KoCoSa split sizes: {counts}")
    return source, {s: d.to_pandas() for s, d in external.items()}


def prepare_external(splits, out):
    audit, frames = {}, []
    for split, raw in splits.items():
        text = raw.tokens.map(lambda value: normalize(value) if isinstance(value, str) else "")
        frame = pd.DataFrame(dict(response=text, label=raw.label, split=split,
                                  source_row_zero_based=range(len(raw))))
        valid = frame[frame.response.ne("") & frame.label.isin(LABEL_TO_ID.values())]
        groups = valid.groupby("response").label
        audit[split] = dict(rows=len(raw), null=raw.isna().sum().to_dict(),
                            empty_strings=int(raw.tokens.eq("").sum()),
                            whitespace_only=int((raw.tokens.notna() & raw.tokens.ne("") & text.eq("")).sum()),
                            exact_extra=int(raw.duplicated().sum()),
                            normalized_extra=int(valid.response.duplicated().sum()),
                            conflicts=int(groups.nunique().gt(1).sum()),
                            classes=raw.label.value_counts().to_dict(),
                            chars=text.str.len().describe().to_dict())
        frames.append(valid)
    data = pd.concat(frames, ignore_index=True)
    shared = set(data.loc[data.split.eq("train"), "response"]) & set(data.loc[data.split.eq("test"), "response"])
    conflicts = data.groupby("response").label.nunique()
    conflicts = conflicts[conflicts.gt(1)].index
    clean = (data[~data.response.isin(conflicts)].sort_values(["split", "source_row_zero_based"])
             .drop_duplicates("response"))
    audit["between_splits"] = dict(shared_inputs=len(shared), conflicting_inputs=len(conflicts),
                                   rule="Drop conflicting labels; for consistent duplicates keep test first.")
    for split in splits:
        audit[split]["after_classes"] = clean.loc[clean.split.eq(split), "label"].value_counts().to_dict()
    (out / "kocasm_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame({f"{split}/{stage}": audit[split][key] for split in splits
                  for stage, key in (("before", "classes"), ("after", "after_classes"))}).T.to_csv(
                      out / "kocasm_preprocessing_counts.csv")
    ax = pd.read_csv(out / "kocasm_preprocessing_counts.csv", index_col=0).plot.bar(rot=25)
    ax.set_ylabel("Rows")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "kocasm_classes.png")
    plt.close(ax.figure)
    return clean


def compare(kocosa, kocasm, out):
    from inference import load_predictor
    from transformers import AutoTokenizer

    baseline = json.loads((out / "baseline/metrics.json").read_text())["context"]
    electra = json.loads((out / "koelectra/metrics.json").read_text())
    candidates = {
        "baseline": (out / "baseline/baseline.joblib", baseline["test"], baseline["selection"]["validation_macro_f1"]),
        "koelectra": (out / "koelectra/final", electra["test"], electra["best_validation_macro_f1"]),
    }
    winner = max(candidates, key=lambda key: candidates[key][2])
    selection = {"model": winner, "rule": "Highest KoCoSa validation Macro F1; ties favor baseline",
                 "scores": {name: value[2] for name, value in candidates.items()}}
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    # Exclude exact normalized text already seen by training or validation.
    seen = kocosa.loc[kocosa.split.isin(["train", "validation"])]
    seen_text = set(seen.response) | set(seen.context) | set(seen.context + " " + seen.response)
    external = kocasm[kocasm.split.eq("test") & ~kocasm.response.isin(seen_text)].copy()
    if external.label.nunique() != 2:
        raise ValueError("External evaluation must contain both classes")
    tokenizer = AutoTokenizer.from_pretrained(out / "koelectra/final", local_files_only=True)
    stats = {}
    for split, frame in kocasm.groupby("split"):
        inputs = tokenizer([""] * len(frame), frame.response.tolist(), truncation=False, verbose=False)
        lengths = pd.Series([len(ids) for ids in inputs["input_ids"]])
        stats[split] = lengths.describe(percentiles=[.5, .9, .95, .99]).to_dict()
        stats[split]["above_kocosa_cutoff"] = int(lengths.gt(electra["max_length"]).sum())
        plt.hist(lengths, bins=30, histtype="step", density=True, label=split)
    plt.xlabel("KoELECTRA input tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "kocasm_token_lengths.png")
    plt.close()
    (out / "kocasm_token_lengths.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    rows, analysis = [], {}
    for name, (model_path, internal, _) in candidates.items():
        predict = load_predictor(name, model_path)
        scores = predict([""] * len(external), external.response.tolist())
        predicted = (scores > .5).astype(int)
        score = metrics(external.label, predicted)
        for dataset, values in (("KoCoSa Test", internal), ("Kocasm Test", score)):
            rows.append(dict(model=name, dataset=dataset, **{k: v for k, v in values.items() if k != "confusion"}))
        errors = external[["source_row_zero_based", "label"]].assign(prediction=predicted, sarcasm_score=scores)
        errors = errors[errors.label.ne(errors.prediction)].copy()
        errors["error_type"] = errors.prediction.map({0: "FN", 1: "FP"})
        errors["chars"] = external.loc[errors.index, "response"].str.len()
        errors["has_user_marker"] = external.loc[errors.index, "response"].str.contains("@user", regex=False)
        errors.to_csv(out / f"kocasm_errors_{name}.csv", index=False)
        analysis[name] = dict(external_rows=len(external), external_raw_rows=301,
                              external_removed_for_kocosa_overlap=int(
                                  kocasm[kocasm.split.eq("test")].response.isin(seen_text).sum()),
                              macro_f1_delta=score["macro_f1"] - internal["macro_f1"],
                              error_groups=errors.groupby("error_type").agg(
                                  rows=("chars", "size"), mean_chars=("chars", "mean"),
                                  user_marker_rate=("has_user_marker", "mean")).to_dict("index"))
        ConfusionMatrixDisplay.from_predictions(external.label, predicted, labels=[0, 1])
        plt.tight_layout()
        plt.savefig(out / f"kocasm_confusion_{name}.png")
        plt.close()
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
    (out / "domain_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return selection
