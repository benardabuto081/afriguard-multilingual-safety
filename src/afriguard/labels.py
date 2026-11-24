"""Label sets and normalisation (mirrors the official scorer's rules)."""
import re
import unicodedata

LANGUAGES = ["amharic", "hausa", "igbo", "oromo", "shona", "swahili", "twi", "wolof", "yoruba", "zulu"]
TASK_DIRS = {"A": "task_a_safety", "B": "task_b_category"}
TASK_A_LABELS = ["safe", "unsafe"]
TASK_B_LABELS = [
    "Criminal Activity & Exploitation",
    "Political & Societal Manipulation",
    "Violence & Physical Harm",
    "Systemic Risks",
    "Sexual & Exploitative Content",
    "Unsafe or High-Risk Advice",
    "Privacy & Data Misuse",
    "Hate, Harassment & Discrimination",
    "Ethical & Moral Risk",
    "Intellectual Property & Content Integrity",
]
LETTERS = "ABCDEFGHIJ"
UNPARSED = "__unparsed__"


def labels_for(task: str) -> list[str]:
    return TASK_A_LABELS if task == "A" else TASK_B_LABELS


def normalize_key(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())
