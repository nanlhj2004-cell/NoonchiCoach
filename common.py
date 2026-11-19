import unicodedata

LABEL_TO_ID = {"Non-Sarcasm": 0, "Sarcasm": 1}
SPLITS = ("train", "validation", "test")


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def format_input(context, response):
    return f"맥락: {normalize(context)}\n마지막 발언: {normalize(response)}"
