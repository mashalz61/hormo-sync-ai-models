from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import AppConfig
from .utils import clean_column_name, slugify_column_name


def find_target_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized_to_original = {slugify_column_name(column): column for column in columns}
    for alias in aliases:
        normalized_alias = slugify_column_name(alias)
        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]

    alias_set = {slugify_column_name(alias) for alias in aliases}
    for column in columns:
        normalized = slugify_column_name(column)
        if normalized in alias_set:
            return column
    return None


def drop_configured_columns(data_frame: pd.DataFrame, dropped_columns: list[str]) -> pd.DataFrame:
    existing = []
    normalized_lookup = {slugify_column_name(column): column for column in data_frame.columns}
    for candidate in dropped_columns:
        match = normalized_lookup.get(slugify_column_name(candidate))
        if match:
            existing.append(match)
    return data_frame.drop(columns=existing, errors="ignore")


def coerce_dirty_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^0-9a-zA-Z.\-+/ ]", "", regex=True)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    non_null_ratio = float(numeric.notna().mean()) if len(series) else 0.0
    return numeric if non_null_ratio >= 0.7 else series


def normalize_binary_target(series: pd.Series, config: AppConfig) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower()
    positive = config.positive_labels
    negative = config.negative_labels

    def _map(value: str) -> int | None:
        if value in positive:
            return 1
        if value in negative:
            return 0
        return None

    normalized = mapped.map(_map)
    if normalized.isna().any():
        unknown_values = sorted(set(series[normalized.isna()].astype(str)))
        raise ValueError(f"Unable to normalize target labels. Unknown values: {unknown_values}")
    return normalized.astype(int)


def prepare_feature_frame(
    data_frame: pd.DataFrame,
    config: AppConfig,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    working = data_frame.copy()
    working.columns = [clean_column_name(column) for column in working.columns]
    working = drop_configured_columns(working, config.dropped_columns)

    target = None
    if target_column and target_column in working.columns:
        target = normalize_binary_target(working[target_column], config)
        working = working.drop(columns=[target_column])

    for column in working.columns:
        working[column] = coerce_dirty_numeric_series(working[column])

    return working, target

