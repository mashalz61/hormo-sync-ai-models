from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from .utils import slugify_column_name


REQUIRED_COLUMNS = {"meal_name", "grams_estimate", "calories_actual"}

_COLUMN_ALIASES = {
    "meal_name": ("meal_name", "meal", "food", "name"),
    "grams_estimate": ("grams_estimate", "grams", "gram", "weight_g", "weight_grams", "weight"),
    "calories_actual": ("calories_actual", "calories", "calorie", "kcal"),
}


@dataclass(frozen=True)
class MealStats:
    canonical_meal_name: str
    normalized_meal_name: str
    calories_per_gram: float
    mean_grams: float
    mean_calories: float
    sample_size: int


class CaloriePredictor:
    def __init__(self, meal_stats: dict[str, MealStats]) -> None:
        if not meal_stats:
            raise ValueError("Calorie predictor requires at least one meal record.")
        self._meal_stats = meal_stats

    @classmethod
    def from_csv(cls, path: str | Path) -> "CaloriePredictor":
        return cls.from_dataframe(pd.read_csv(path))

    @classmethod
    def from_joblib(cls, path: str | Path) -> "CaloriePredictor":
        loaded = joblib.load(path)
        if not isinstance(loaded, CaloriePredictor):
            raise TypeError("Joblib file does not contain a CaloriePredictor instance.")
        return loaded

    def to_joblib(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def from_dataframe(cls, data_frame: pd.DataFrame) -> "CaloriePredictor":
        normalized_columns = {slugify_column_name(column): column for column in data_frame.columns}
        rename_map: dict[str, str] = {}
        for required in REQUIRED_COLUMNS:
            if required in data_frame.columns:
                continue
            for alias in _COLUMN_ALIASES.get(required, ()):
                candidate = normalized_columns.get(slugify_column_name(alias))
                if candidate is not None:
                    rename_map[candidate] = required
                    break

        coerced = data_frame.rename(columns=rename_map)
        missing_columns = REQUIRED_COLUMNS.difference(coerced.columns)
        if missing_columns:
            raise ValueError(
                "Meal dataset is missing required columns. "
                f"Expected {sorted(REQUIRED_COLUMNS)}, got {sorted(coerced.columns)}."
            )

        cleaned = coerced.copy()
        cleaned["meal_name"] = cleaned["meal_name"].astype(str).str.strip()
        cleaned["grams_estimate"] = pd.to_numeric(cleaned["grams_estimate"], errors="coerce")
        cleaned["calories_actual"] = pd.to_numeric(cleaned["calories_actual"], errors="coerce")
        cleaned = cleaned.dropna(subset=["meal_name", "grams_estimate", "calories_actual"])
        cleaned = cleaned[
            (cleaned["meal_name"] != "")
            & (cleaned["grams_estimate"] > 0)
            & (cleaned["calories_actual"] >= 0)
        ]
        if cleaned.empty:
            raise ValueError("Meal dataset does not contain valid rows for calorie prediction.")

        cleaned["normalized_meal_name"] = cleaned["meal_name"].map(_normalize_meal_name)
        cleaned["calories_per_gram"] = cleaned["calories_actual"] / cleaned["grams_estimate"]

        grouped = (
            cleaned.groupby("normalized_meal_name", as_index=False)
            .agg(
                canonical_meal_name=("meal_name", _canonical_name),
                mean_grams=("grams_estimate", "mean"),
                mean_calories=("calories_actual", "mean"),
                calories_per_gram=("calories_per_gram", "mean"),
                sample_size=("meal_name", "size"),
            )
        )

        meal_stats = {
            row["normalized_meal_name"]: MealStats(
                canonical_meal_name=str(row["canonical_meal_name"]),
                normalized_meal_name=str(row["normalized_meal_name"]),
                calories_per_gram=float(row["calories_per_gram"]),
                mean_grams=float(row["mean_grams"]),
                mean_calories=float(row["mean_calories"]),
                sample_size=int(row["sample_size"]),
            )
            for _, row in grouped.iterrows()
        }
        return cls(meal_stats)

    def estimate(
        self,
        meal_name: str,
        grams: Optional[float] = None,
        portion_count: Optional[float] = None,
    ) -> dict[str, Any]:
        normalized_meal_name = _normalize_meal_name(meal_name)
        if not normalized_meal_name:
            raise ValueError("Meal name must not be empty.")

        if (grams is None and portion_count is None) or (grams is not None and portion_count is not None):
            raise ValueError("Provide exactly one of `grams` or `portion_count`.")

        matched_stats, similarity, match_type = self._match(normalized_meal_name)
        response: dict[str, Any] = {
            "meal_name": meal_name,
            "matched_meal_name": matched_stats.canonical_meal_name,
            "match_type": match_type,
            "sample_size": matched_stats.sample_size,
            "reference_mean_grams": round(matched_stats.mean_grams, 2),
            "reference_mean_calories": round(matched_stats.mean_calories, 2),
            "calories_per_gram": round(matched_stats.calories_per_gram, 4),
            "calories_per_100g": round(matched_stats.calories_per_gram * 100.0, 2),
        }

        if grams is not None:
            if grams <= 0:
                raise ValueError("Grams must be greater than zero.")
            estimated_calories = matched_stats.calories_per_gram * float(grams)
            response.update(
                {
                    "input_mode": "grams",
                    "input_grams": float(grams),
                    "estimated_calories": round(estimated_calories, 2),
                    "formula": "estimated_calories = learned_calories_per_gram * input_grams",
                }
            )
        else:
            if portion_count is None or portion_count <= 0:
                raise ValueError("Portion count must be greater than zero.")
            estimated_calories = matched_stats.mean_calories * float(portion_count)
            response.update(
                {
                    "input_mode": "portion_count",
                    "portion_count": float(portion_count),
                    "estimated_calories": round(estimated_calories, 2),
                    "formula": "estimated_calories = reference_mean_calories * portion_count",
                    "estimated_grams": round(matched_stats.mean_grams * float(portion_count), 2),
                }
            )

        if match_type == "fuzzy":
            response["name_similarity"] = round(similarity, 3)
        return response

    def _match(self, normalized_meal_name: str) -> tuple[MealStats, float, str]:
        direct_match = self._meal_stats.get(normalized_meal_name)
        if direct_match is not None:
            return direct_match, 1.0, "exact"

        best_key: str | None = None
        best_score = 0.0
        for candidate_key in self._meal_stats:
            score = SequenceMatcher(None, normalized_meal_name, candidate_key).ratio()
            if score > best_score:
                best_score = score
                best_key = candidate_key

        if best_key is None or best_score < 0.72:
            raise KeyError(f"No close meal match found for '{normalized_meal_name}'.")
        return self._meal_stats[best_key], best_score, "fuzzy"


def _normalize_meal_name(value: str) -> str:
    return slugify_column_name(value).replace("_", " ")


def _canonical_name(series: pd.Series) -> str:
    modes = series.mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return str(series.iloc[0])
