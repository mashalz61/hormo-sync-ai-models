from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any]
    config_path: Path

    @property
    def random_seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def models_dir(self) -> Path:
        return Path(self.raw["paths"]["models_dir"])

    @property
    def reports_dir(self) -> Path:
        return Path(self.raw["paths"]["reports_dir"])

    @property
    def target_aliases(self) -> dict[str, list[str]]:
        return self.raw["data"]["target_aliases"]

    @property
    def dropped_columns(self) -> list[str]:
        return self.raw["data"]["dropped_columns"]

    @property
    def positive_labels(self) -> set[str]:
        return {str(value).strip().lower() for value in self.raw["data"]["positive_labels"]}

    @property
    def negative_labels(self) -> set[str]:
        return {str(value).strip().lower() for value in self.raw["data"]["negative_labels"]}

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def reporting(self) -> dict[str, Any]:
        return self.raw["reporting"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return AppConfig(raw=loaded, config_path=path)
