"""Post-training threshold analysis. Never fits or modifies a model."""
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from common import LABEL_TO_ID
from inference import load_predictor
from train import metrics
from colab_pipeline import download, prepare_external


def select_threshold(labels, scores, out):
    sweep = pd.DataFrame([
        dict(threshold=value, **metrics(labels, scores > value))
        for value in np.arange(5, 96, 1) / 100
    ])
    reference = sweep.loc[sweep.threshold.eq(.5)].iloc[0]
    eligible = sweep[sweep.macro_f1.ge(reference.macro_f1 - .01 - 1e-12)
                     & sweep.fpr.le(reference.fpr)]
    selected = eligible.sort_values(
        ["fpr", "macro_f1", "recall", "threshold"], ascending=[True, False, False, True]
    ).iloc[0]
    sweep.drop(columns="confusion").to_csv(out / "validation_thresholds.csv", index=False)
    ax = sweep.set_index("threshold")[["precision", "recall", "fpr", "macro_f1"]].plot()
    ax.axvline(selected.threshold, color="black", linestyle="--", label="Selected on validation")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(out / "validation_thresholds.png")
    plt.close(ax.figure)
    return float(selected.threshold)


def evaluate_saved(run, prepared, cache, out):
    run, prepared, out = map(Path, (run, prepared, out))
    metadata = json.loads((run / "koelectra/metrics.json").read_text(encoding="utf-8"))
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != metadata["input_sha256"]:
        raise ValueError("Use the exact preprocessed Parquet from this training run.")
    classifier = load_predictor("koelectra", run / "koelectra/final")
    frame = pd.read_parquet(prepared)
    frame["target"] = frame.label.map(LABEL_TO_ID)
    out.mkdir(parents=True, exist_ok=False)
    validation = frame[frame.split.eq("validation")]
    if set(validation.target) != {0, 1}:
        raise ValueError("KoCoSa Validation must contain both mapped labels.")
    scores = classifier(validation.context.tolist(), validation.response.tolist())
    threshold = select_threshold(validation.target, scores, out)
    validation[["source_row_zero_based", "target"]].assign(score=scores).to_csv(
        out / "validation_scores.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "threshold": threshold, "reference_threshold": .5,
        "rule": "Minimize validation FPR with Macro F1 within 0.01 of the 0.50 reference and FPR no higher. Ties favor F1, recall, then lower threshold.",
        "model_path": str(run / "koelectra/final"),
    }, indent=2), encoding="utf-8")

    _, raw_external = download(Path(cache))
    clean = prepare_external(raw_external, out)
    seen = frame[frame.split.isin(["train", "validation"])]
    seen_text = set(seen.response) | set(seen.context) | set(seen.context + " " + seen.response)
    external = clean[clean.split.eq("test") & ~clean.response.isin(seen_text)].copy()
    external["context"], external["target"] = "", external.label
    (out / "kocasm_input_check.json").write_text(json.dumps({
        "label_mapping": {"0": "no_sarcasm", "1": "sarcasm"},
        "metadata_verified_by_download": True, "empty_context": bool(external.context.eq("").all()),
        "raw_test_rows": len(raw_external["test"]), "evaluated_test_rows": len(external),
        "class_counts": external.target.value_counts().to_dict(),
    }, indent=2), encoding="utf-8")
    rows = []
    for name, test in (("kocosa", frame[frame.split.eq("test")]), ("kocasm", external)):
        if set(test.target) != {0, 1}:
            raise ValueError(f"{name} Test must contain both mapped labels.")
        scores = classifier(test.context.tolist(), test.response.tolist())
        for tag, cutoff in (("default", .5), ("selected", threshold)):
            predicted = (scores > cutoff).astype(int)
            values = metrics(test.target, predicted)
            rows.append(dict(dataset=name, policy=tag, threshold=cutoff, n=len(test),
                             predicted_positive=int(predicted.sum()),
                             **{k: v for k, v in values.items() if k != "confusion"}))
            errors = test[["source_row_zero_based", "target"]].assign(
                score=scores, prediction=predicted, chars=test.response.str.len(),
                has_user_marker=test.response.str.contains("@user", regex=False))
            errors = errors[errors.target.ne(errors.prediction)].copy()
            errors["error_type"] = errors.prediction.map({0: "FN", 1: "FP"})
            errors["distance_to_threshold"] = (errors.score - cutoff).abs()
            errors.sort_values("distance_to_threshold").to_csv(out / f"{name}_{tag}_errors.csv", index=False)
            errors.groupby("error_type").agg(count=("score", "size"), mean_score=("score", "mean"),
                mean_chars=("chars", "mean"), user_marker_rate=("has_user_marker", "mean")).to_csv(
                    out / f"{name}_{tag}_error_summary.csv")
            ConfusionMatrixDisplay.from_predictions(test.target, predicted, labels=[0, 1],
                                                   display_labels=list(LABEL_TO_ID))
            plt.tight_layout()
            plt.savefig(out / f"{name}_{tag}_confusion.png")
            plt.close()
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_comparison.csv", index=False)
    ax = table.set_index(["dataset", "policy"])[["macro_f1", "precision", "recall", "fpr"]].plot.bar(rot=20)
    ax.set_ylim(0, 1.02)
    ax.figure.tight_layout()
    ax.figure.savefig(out / "test_comparison.png")
    plt.close(ax.figure)
    return table
