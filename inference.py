import json
from pathlib import Path

import joblib

from common import format_input, normalize


def load_predictor(backend, path):
    path = Path(path)
    if backend == "baseline":
        model = joblib.load(path)

        def scores(contexts, responses):
            inputs = [format_input(c, r) for c, r in zip(contexts, responses, strict=True)]
            return model.predict_proba(inputs)[:, list(model.classes_).index(1)]

        return scores
    if backend != "koelectra":
        raise ValueError("Model must be baseline or koelectra.")
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    max_length = json.loads((path.parent / "token_lengths.json").read_text(encoding="utf-8"))["max_length"]

    def scores(contexts, responses):
        pairs = list(zip(contexts, responses, strict=True))
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 16):
                c, r = zip(*pairs[start:start + 16])
                batch = tokenizer(list(map(normalize, c)), list(map(normalize, r)), padding=True,
                                  truncation=True, max_length=max_length, return_tensors="pt")
                chunks.append(model(**batch).logits.softmax(dim=-1)[:, 1].numpy())
        return np.concatenate(chunks) if chunks else np.empty(0)

    return scores
