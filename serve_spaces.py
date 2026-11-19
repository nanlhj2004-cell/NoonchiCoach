"""Download private model artifacts and start the KoELECTRA WSGI service."""
import json
import os
from pathlib import Path
import sys


def prepare_model():
    if os.environ.get("NOONCHI_MODEL", "koelectra") != "koelectra":
        raise RuntimeError("This Space requires KoELECTRA.")
    repo = os.environ.get("MODEL_REPO_ID")
    revision = os.environ.get("MODEL_REVISION")
    if not repo or not revision:
        raise RuntimeError("Set MODEL_REPO_ID and MODEL_REVISION to the private model repository and commit.")
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=repo, revision=revision, token=os.environ.get("HF_TOKEN"),
        allow_patterns=["final/model.safetensors", "final/config.json", "final/tokenizer.json",
                        "final/tokenizer_config.json", "final/special_tokens_map.json",
                        "final/vocab.txt", "token_lengths.json"],
    ))
    final = snapshot / "final"
    required = ["model.safetensors", "config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    if not all((final / name).is_file() for name in required):
        raise FileNotFoundError("Private model repository is missing required artifacts.")
    settings = json.loads((snapshot / "token_lengths.json").read_text(encoding="utf-8"))
    config = json.loads((final / "config.json").read_text(encoding="utf-8"))
    if settings["max_length"] != 152 or config["id2label"] != {"0": "Non-Sarcasm", "1": "Sarcasm"}:
        raise ValueError("Model settings differ from the validated KoELECTRA configuration.")
    os.environ["NOONCHI_MODEL"] = "koelectra"
    os.environ["NOONCHI_MODEL_PATH"] = str(final)
    print("Starting KoELECTRA; max_length=152; threshold > 0.50", flush=True)


if __name__ == "__main__":
    prepare_model()
    os.execv(sys.executable, [
        sys.executable, "-m", "gunicorn", "--bind", "0.0.0.0:7860",
        "--workers", "1", "--threads", "2", "--timeout", "180",
        "--access-logfile", "-", "--error-logfile", "-", "app:app",
    ])
