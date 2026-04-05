from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .utils import slugify_column_name


REQUIRED_COLUMNS = {
    "Name of Exercise",
    "Sets",
    "Reps",
    "Benefit",
    "Burns Calories (per 30 min)",
    "Target Muscle Group",
    "Equipment Needed",
    "Difficulty Level",
}


@dataclass(frozen=True)
class ExerciseRecord:
    exercise_name: str
    normalized_exercise_name: str
    sets: int
    reps: int
    benefit: str
    calories_per_30_min: float
    target_muscle_group: str
    equipment_needed: str
    difficulty_level: str


class ExercisePredictor:
    def __init__(self, exercises: list[ExerciseRecord]) -> None:
        if not exercises:
            raise ValueError("Exercise predictor requires at least one exercise record.")
        self._exercises = exercises
        self._by_name = {exercise.normalized_exercise_name: exercise for exercise in exercises}

    @classmethod
    def from_csv(cls, path: str | Path) -> "ExercisePredictor":
        return cls.from_dataframe(pd.read_csv(path))

    @classmethod
    def from_dataframe(cls, data_frame: pd.DataFrame) -> "ExercisePredictor":
        missing_columns = REQUIRED_COLUMNS.difference(data_frame.columns)
        if missing_columns:
            raise ValueError(f"Exercise dataset is missing required columns: {sorted(missing_columns)}")

        cleaned = data_frame.copy()
        cleaned["Name of Exercise"] = cleaned["Name of Exercise"].astype(str).str.strip()
        cleaned["Sets"] = pd.to_numeric(cleaned["Sets"], errors="coerce")
        cleaned["Reps"] = pd.to_numeric(cleaned["Reps"], errors="coerce")
        cleaned["Burns Calories (per 30 min)"] = pd.to_numeric(
            cleaned["Burns Calories (per 30 min)"],
            errors="coerce",
        )
        text_columns = [
            "Benefit",
            "Target Muscle Group",
            "Equipment Needed",
            "Difficulty Level",
        ]
        for column in text_columns:
            cleaned[column] = cleaned[column].fillna("None").astype(str).str.strip()

        cleaned = cleaned.dropna(subset=["Name of Exercise", "Sets", "Reps", "Burns Calories (per 30 min)"])
        cleaned = cleaned[
            (cleaned["Name of Exercise"] != "")
            & (cleaned["Sets"] > 0)
            & (cleaned["Reps"] > 0)
            & (cleaned["Burns Calories (per 30 min)"] >= 0)
        ]
        if cleaned.empty:
            raise ValueError("Exercise dataset does not contain valid rows for recommendations.")

        exercises = [
            ExerciseRecord(
                exercise_name=str(row["Name of Exercise"]),
                normalized_exercise_name=_normalize_value(str(row["Name of Exercise"])),
                sets=int(row["Sets"]),
                reps=int(row["Reps"]),
                benefit=str(row["Benefit"]),
                calories_per_30_min=float(row["Burns Calories (per 30 min)"]),
                target_muscle_group=str(row["Target Muscle Group"]),
                equipment_needed=str(row["Equipment Needed"]),
                difficulty_level=str(row["Difficulty Level"]),
            )
            for _, row in cleaned.iterrows()
        ]
        return cls(exercises)

    def recommend(
        self,
        *,
        exercise_name: Optional[str] = None,
        duration_minutes: float = 30.0,
        difficulty_level: Optional[str] = None,
        target_muscle_group: Optional[str] = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        if duration_minutes <= 0:
            raise ValueError("`duration_minutes` must be greater than zero.")
        if limit <= 0:
            raise ValueError("`limit` must be greater than zero.")

        if exercise_name:
            match, similarity, match_type = self._match_exercise(exercise_name)
            response = self._serialize_exercise(match, duration_minutes)
            response["match_type"] = match_type
            if match_type == "fuzzy":
                response["name_similarity"] = round(similarity, 3)
            return response

        normalized_difficulty = _normalize_value(difficulty_level) if difficulty_level else None
        normalized_muscle_group = _normalize_value(target_muscle_group) if target_muscle_group else None

        filtered = self._exercises
        if normalized_difficulty:
            filtered = [
                exercise
                for exercise in filtered
                if _normalize_value(exercise.difficulty_level) == normalized_difficulty
            ]
        if normalized_muscle_group:
            filtered = [
                exercise
                for exercise in filtered
                if normalized_muscle_group in _normalize_value(exercise.target_muscle_group)
            ]

        if not filtered:
            raise KeyError("No exercise recommendations matched the supplied filters.")

        ranked = sorted(filtered, key=lambda exercise: exercise.calories_per_30_min, reverse=True)[:limit]
        return {
            "available": True,
            "request_mode": "recommendation_list",
            "duration_minutes": float(duration_minutes),
            "recommendations": [self._serialize_exercise(exercise, duration_minutes) for exercise in ranked],
        }

    def _match_exercise(self, exercise_name: str) -> tuple[ExerciseRecord, float, str]:
        normalized_name = _normalize_value(exercise_name)
        if not normalized_name:
            raise ValueError("`exercise_name` must not be empty.")

        direct_match = self._by_name.get(normalized_name)
        if direct_match is not None:
            return direct_match, 1.0, "exact"

        best_match: ExerciseRecord | None = None
        best_score = 0.0
        for exercise in self._exercises:
            score = SequenceMatcher(None, normalized_name, exercise.normalized_exercise_name).ratio()
            if score > best_score:
                best_score = score
                best_match = exercise

        if best_match is None or best_score < 0.72:
            raise KeyError(f"No close exercise match found for '{exercise_name}'.")
        return best_match, best_score, "fuzzy"

    @staticmethod
    def _serialize_exercise(exercise: ExerciseRecord, duration_minutes: float) -> dict[str, Any]:
        estimated_burn = exercise.calories_per_30_min * (duration_minutes / 30.0)
        return {
            "available": True,
            "request_mode": "single_exercise",
            "exercise_name": exercise.exercise_name,
            "sets": exercise.sets,
            "reps": exercise.reps,
            "benefit": exercise.benefit,
            "difficulty_level": exercise.difficulty_level,
            "target_muscle_group": exercise.target_muscle_group,
            "equipment_needed": exercise.equipment_needed,
            "duration_minutes": float(duration_minutes),
            "calories_burned_per_30_min": round(exercise.calories_per_30_min, 2),
            "estimated_calories_burned": round(estimated_burn, 2),
        }


def _normalize_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return slugify_column_name(value).replace("_", " ")
