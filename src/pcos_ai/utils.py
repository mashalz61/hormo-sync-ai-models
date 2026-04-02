from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


LOGGER_NAME = "pcos_ai"


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clean_column_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    return text.strip()


def slugify_column_name(name: str) -> str:
    value = clean_column_name(name).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def probability_to_level(probability: float, low_upper: float = 0.33, medium_upper: float = 0.66) -> str:
    if probability < low_upper:
        return "Low"
    if probability < medium_upper:
        return "Medium"
    return "High"


def write_markdown(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)

