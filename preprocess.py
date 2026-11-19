import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata
import sys

import pandas as pd

root = Path.cwd()
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent
fields = ["context", "response", "label", "sarcasm_explanation"]
labels = ["Sarcasm", "Non-Sarcasm"]
frames = []
report = {"files": {}, "splits": {}, "overlap": {}}


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", value).split()) if isinstance(value, str) else ""


def distribution(series):
    return {key: round(float(value), 2) for key, value in {
        "min": series.min(), "mean": series.mean(), "p50": series.median(),
        "p90": series.quantile(.90), "p95": series.quantile(.95),
        "p99": series.quantile(.99), "max": series.max()
    }.items()}


def lengths(frame):
    context, response = frame["ctx"], frame["resp"]
    combined = (context + " " + response).str.strip()
    return {
        "context_chars": distribution(context.str.len()),
        "response_chars": distribution(response.str.len()),
        "combined_chars": distribution(combined.str.len()),
        "whitespace_tokens": distribution(combined.str.split().str.len()),
        "speaker_turns": distribution(combined.map(lambda s: len(re.findall(r"(?<!\S)[AB]\s*:", s)))),
    }


for split in ("train", "validation", "test"):
    path = source / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path, engine="pyarrow")
    report["files"][split] = {
        "file": path.name, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "read_success": True,
    }
    missing = {f: {
        "null": int(frame[f].isna().sum()), "empty": int(frame[f].eq("").sum()),
        "whitespace_only": int(frame[f].map(lambda x: isinstance(x, str) and bool(x) and not x.strip()).sum())
    } for f in fields}
    exact = int(frame.duplicated().sum())
    frame = frame.assign(ctx=frame.context.map(normalize), resp=frame.response.map(normalize),
                         lab=frame.label.map(normalize), split=split, row=range(len(frame)))
    frame["valid"] = frame.ctx.ne("") & frame.resp.ne("") & frame.lab.isin(labels)
    valid = frame[frame.valid]
    keys = ["ctx", "resp"]
    counts = valid.groupby(keys).lab.agg(["size", "nunique"])
    report["splits"][split] = {
        "rows": len(frame), "labels": frame.lab.value_counts().to_dict(), "missing": missing,
        "exact_extra": exact, "normalized_extra_valid": int((counts["size"] - 1).sum()),
        "normalized_duplicate_groups_valid": int(counts["size"].gt(1).sum()),
        "conflict_groups_valid": int(counts["nunique"].gt(1).sum()),
        "invalid_required_rows": int((~frame.valid).sum()),
        "normalized_changed_fields": {a: int((frame[a].notna() & frame[a].ne(frame[b])).sum())
                                       for a, b in [("context", "ctx"), ("response", "resp"), ("label", "lab")]},
        "explanation_blank_by_label": frame.assign(blank=frame.sarcasm_explanation.map(normalize).eq(""))
            .groupby("lab").blank.sum().to_dict(),
        "lengths_valid": lengths(valid),
    }
    frames.append(frame)

all_rows = pd.concat(frames, ignore_index=True)
valid = all_rows[all_rows.valid].copy()
counts = valid.groupby(keys).lab.agg(["size", "nunique"])
report["totals"] = {
    "rows": len(all_rows), "bytes": sum(f["bytes"] for f in report["files"].values()),
    "labels": all_rows.lab.value_counts().to_dict(),
    "exact_extra": int(all_rows[fields].duplicated().sum()),
    "normalized_extra_valid": int((counts["size"] - 1).sum()),
    "conflict_groups_valid": int(counts["nunique"].gt(1).sum()),
    "lengths_valid": lengths(valid),
}
for a, b in itertools.combinations(frames, 2):
    av, bv = a[a.valid], b[b.valid]
    pair = {}
    for name, columns in [("exact_rows", fields), ("input", keys), ("context", ["ctx"]), ("response", ["resp"])]:
        left, right = (a, b) if name == "exact_rows" else (av, bv)
        if name == "exact_rows":
            left, right = left.fillna("<NULL>"), right.fillna("<NULL>")
        ka = pd.MultiIndex.from_frame(left[columns])
        kb = pd.MultiIndex.from_frame(right[columns])
        common = ka.unique().intersection(kb.unique())
        pair[name] = {"keys": len(common), "left_rows": int(ka.isin(common).sum()),
                      "right_rows": int(kb.isin(common).sum())}
    def neutralize_speakers(s):
        return re.sub(r"(?<!\S)[AB]\s*:", "S:", s)

    for name, columns in [("speaker_neutral_input", keys), ("speaker_neutral_context", ["ctx"])]:
        ka = pd.MultiIndex.from_frame(av[columns].apply(lambda s: s.map(neutralize_speakers)))
        kb = pd.MultiIndex.from_frame(bv[columns].apply(lambda s: s.map(neutralize_speakers)))
        common = ka.unique().intersection(kb.unique())
        pair[name] = {"keys": len(common), "left_rows": int(ka.isin(common).sum()),
                      "right_rows": int(kb.isin(common).sum())}
    report["overlap"][a.split.iloc[0] + "/" + b.split.iloc[0]] = pair

conflicts = counts[counts["nunique"].gt(1)].index
valid["conflict"] = pd.MultiIndex.from_frame(valid[keys]).isin(conflicts)
# Preserve held-out examples ahead of development and training rows.
priority = {"test": 0, "validation": 1, "train": 2}
clean = valid[~valid.conflict].copy()
clean = clean.assign(priority=clean.split.map(priority)).sort_values(["priority", "row"])
clean = clean.drop_duplicates(keys)
report["preprocessed"] = clean.split.value_counts().to_dict()
report["kept_rows"] = len(clean)
report["removed_rows"] = len(all_rows) - len(clean)
report["removed_breakdown"] = {
    "invalid_required": int((~all_rows.valid).sum()),
    "conflicting_labels": int(valid.conflict.sum()),
    "duplicate_input": int(len(valid) - valid.conflict.sum() - len(clean)),
}
out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "outputs"
out.mkdir(exist_ok=True)
clean[["split", "row", "ctx", "resp", "lab"]].rename(
    columns={"row": "source_row_zero_based", "ctx": "context", "resp": "response", "lab": "label"}
).to_parquet(out / "kocosa_preprocessed.parquet", index=False)
report["provenance"] = "All three input Parquet files were read; see files for SHA-256 and byte counts."
report["tokenization"] = "Whitespace token counts are measured. KoELECTRA subword counts are unavailable without its vocabulary/tokenizer."
report["redistribution"] = "Local analysis and preprocessing only. Dataset license and redistribution permission remain unverified."
(out / "kocosa_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True))
