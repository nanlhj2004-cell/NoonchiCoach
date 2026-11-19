import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline

from common import LABEL_TO_ID, SPLITS, format_input


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                fpr=float(fp / (tn + fp)), confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])


def run(source, audit_file=None, output="results"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    frame["target"] = frame.label.map(LABEL_TO_ID)
    assert frame.target.notna().all() and not frame.duplicated(["context", "response"]).any()
    train, val, test = [frame[frame.split.eq(s)].reset_index(drop=True) for s in SPLITS]
    results = {}
    for mode in ("context", "response_only"):
        def features(rows):
            return [format_input(c if mode == "context" else "", r) for c, r in zip(rows.context, rows.response)]

        x_train, x_val, x_test = map(features, (train, val, test))
        trials = []
        best = None
        for c in (.25, 1, 4):
            for weight in (None, "balanced"):
                model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                                       max_features=85000, sublinear_tf=True),
                                      LogisticRegression(C=c, class_weight=weight, max_iter=500, random_state=42))
                model.fit(x_train, train.target)
                score = f1_score(val.target, model.predict(x_val), average="macro")
                trials.append(dict(C=c, class_weight=weight, validation_macro_f1=score))
                if best is None or score > best[0]:
                    best = (score, model, trials[-1])
        _, model, selection = best
        pred = model.predict(x_test)
        results[mode] = dict(selection=selection, trials=trials, test=metrics(test.target, pred))
        errors = test.loc[test.target.ne(pred), ["source_row_zero_based", "label"]].copy()
        errors["prediction"] = pred[test.target.ne(pred)]
        errors["score"] = model.predict_proba(x_test)[test.target.ne(pred), 1]
        errors.to_csv(out / f"errors_{mode}.csv", index=False)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(test.target, pred), display_labels=["Non-Sarcasm", "Sarcasm"]).plot(ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{mode}.png", dpi=135)
        plt.close(fig)
        if mode == "context":
            joblib.dump(model, out / "baseline.joblib", compress=3)
    fig, ax = plt.subplots()
    for mode, result in results.items():
        for weight in (None, "balanced"):
            trials = [x for x in result["trials"] if x["class_weight"] == weight]
            ax.plot([x["C"] for x in trials], [x["validation_macro_f1"] for x in trials],
                    "o-", label=f"{mode} / {weight or 'unweighted'}")
    ax.set(xscale="log", xlabel="Logistic regression C", ylabel="Validation Macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=135)
    plt.close(fig)
    if audit_file:
        audit = json.loads(Path(audit_file).read_text(encoding="utf-8"))
        after = pd.crosstab(frame.split, frame.label)
        before = pd.DataFrame({s: data["labels"] for s, data in audit["splits"].items()}).T
        comparison = pd.concat({"before": before, "after": after}, names=["stage", "split"])
        comparison.to_csv(out / "preprocessing_counts.csv")
        ax = comparison.plot.bar(rot=30)
        ax.set_ylabel("Rows")
        ax.figure.tight_layout()
        ax.figure.savefig(out / "preprocessing.png", dpi=135)
        plt.close(ax.figure)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run(*sys.argv[1:])
