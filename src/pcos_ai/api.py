from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, model_validator

from .calorie_predictor import CaloriePredictor
from .exercise_predictor import ExercisePredictor
from .predict import load_model_bundle, predict_from_dict
from .utils import slugify_column_name


LOGGER = logging.getLogger("pcos_ai.api")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class CaloriePredictionRequest(BaseModel):
    meal_name: str = Field(min_length=1)
    grams: Optional[float] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("grams", "gram", "weight_grams", "weight_g"),
    )
    portion_count: Optional[float] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("portion_count", "portion_size", "portion"),
    )

    @model_validator(mode="after")
    def validate_input_mode(self) -> "CaloriePredictionRequest":
        if (self.grams is None and self.portion_count is None) or (
            self.grams is not None and self.portion_count is not None
        ):
            raise ValueError("Provide exactly one of `grams` or `portion_count`.")
        return self


class ExercisePredictionRequest(BaseModel):
    exercise_name: Optional[str] = Field(default=None)
    duration_minutes: float = Field(default=30.0, gt=0)
    difficulty_level: Optional[str] = Field(default=None)
    target_muscle_group: Optional[str] = Field(default=None)
    limit: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_request(self) -> "ExercisePredictionRequest":
        if self.exercise_name is None and self.difficulty_level is None and self.target_muscle_group is None:
            raise ValueError(
                "Provide `exercise_name` for a direct lookup or at least one filter such as "
                "`difficulty_level` or `target_muscle_group`."
            )
        return self


class ApiState:
    def __init__(self) -> None:
        self.pcos_bundle: dict[str, Any] | None = None
        self.ir_bundle: dict[str, Any] | None = None
        self.calorie_predictor: CaloriePredictor | None = None
        self.exercise_predictor: ExercisePredictor | None = None
        self.ir_available = False
        self.ir_status_message = "IR model unavailable. Train and save an IR model to enable this endpoint."
        self.calorie_status_message = (
            "Calorie model unavailable. Train it with `python -m src.pcos_ai.train_calories` "
            "or ensure a meals CSV exists at data/calory_predictor/meal.csv (or meals.csv)."
        )
        self.exercise_status_message = (
            "Exercise recommendations unavailable. Ensure the exercises CSV exists at "
            "data/exercise_predictor/exercises.csv."
        )

state = ApiState()

PCOS_ALIAS_MAP = {
    "marriage_status": "Marraige Status (Yrs)",
}


def _coerce_payload_key(key: str, canonical_by_slug: dict[str, str]) -> str | None:
    direct_match = canonical_by_slug.get(slugify_column_name(key))
    if direct_match:
        return direct_match

    alias_match = PCOS_ALIAS_MAP.get(slugify_column_name(key))
    if alias_match in canonical_by_slug.values():
        return alias_match
    return None


def _pcos_value_warnings(features: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    systolic = features.get("BP _Systolic (mmHg)")
    diastolic = features.get("BP _Diastolic (mmHg)")
    if isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float)):
        if systolic == diastolic:
            warnings.append("Systolic and diastolic blood pressure are identical. Please double-check those values.")
        elif systolic < diastolic:
            warnings.append("Systolic blood pressure is lower than diastolic blood pressure. Please verify the reading.")

    abortions = features.get("No. of aborptions")
    if isinstance(abortions, (int, float)) and abortions >= 5:
        warnings.append("`No. of aborptions` is unusually high. Please confirm that value is correct.")

    cycle_value = features.get("Cycle(R/I)")
    if isinstance(cycle_value, (int, float)) and cycle_value not in {2, 4}:
        warnings.append("`Cycle(R/I)` is usually sent as `R`/`I` or the numeric values `2`/`4` used by the dataset.")
    return warnings


def normalize_pcos_payload(bundle: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    feature_columns = bundle.get("feature_columns", [])
    if not feature_columns:
        return payload, []

    canonical_by_slug = {slugify_column_name(column): column for column in feature_columns}
    normalized: dict[str, Any] = {}
    warnings: list[str] = []

    for key, value in payload.items():
        canonical_key = _coerce_payload_key(key, canonical_by_slug)
        if canonical_key is None:
            warnings.append(f"Ignored unused field `{key}`.")
            continue
        normalized[canonical_key] = value

    for expected_key in feature_columns:
        normalized.setdefault(expected_key, None)

    warnings.extend(_pcos_value_warnings(payload))
    return normalized, warnings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path(filename: str) -> Path:
    if "insulin" in filename:
        return _project_root() / "models" / "insulin_resistance" / filename
    return _project_root() / "models" / "pcos" / filename


def _default_calorie_data_path() -> Path:
    data_dir = _project_root() / "data" / "calory_predictor"
    for candidate in ("meal.csv", "meals.csv"):
        path = data_dir / candidate
        if path.exists():
            return path
    return data_dir / "meal.csv"


def _default_calorie_model_path() -> Path:
    return _project_root() / "models" / "calory_predictor" / "calorie_predictor.joblib"


def _default_exercise_data_path() -> Path:
    return _project_root() / "data" / "exercise_predictor" / "exercises.csv"


def _safe_load_bundle(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        LOGGER.warning("Model file not found: %s", path)
        return None
    try:
        return load_model_bundle(path)
    except Exception as exc:
        LOGGER.exception("Failed to load model bundle from %s: %s", path, exc)
        return None


def _safe_load_calorie_predictor(path: Path) -> CaloriePredictor | None:
    try:
        model_path = _default_calorie_model_path()
        if model_path.exists():
            return CaloriePredictor.from_joblib(model_path)
        if not path.exists():
            LOGGER.warning("Calorie data file not found: %s", path)
            return None
        return CaloriePredictor.from_csv(path)
    except Exception as exc:
        LOGGER.exception("Failed to build calorie predictor: %s", exc)
        return None


def _safe_load_exercise_predictor(path: Path) -> ExercisePredictor | None:
    if not path.exists():
        LOGGER.warning("Exercise data file not found: %s", path)
        return None
    try:
        return ExercisePredictor.from_csv(path)
    except Exception as exc:
        LOGGER.exception("Failed to build exercise predictor from %s: %s", path, exc)
        return None


def load_models() -> None:
    state.pcos_bundle = _safe_load_bundle(_default_model_path("best_pcos_model.joblib"))
    state.ir_bundle = _safe_load_bundle(_default_model_path("best_insulin_resistance_model.joblib"))
    state.calorie_predictor = _safe_load_calorie_predictor(_default_calorie_data_path())
    state.exercise_predictor = _safe_load_exercise_predictor(_default_exercise_data_path())
    state.ir_available = state.ir_bundle is not None
    if state.pcos_bundle is None:
        LOGGER.warning("PCOS model is not loaded. Train the PCOS model before using the API.")
    if not state.ir_available:
        LOGGER.info(state.ir_status_message)
    if state.calorie_predictor is None:
        LOGGER.info(state.calorie_status_message)
    if state.exercise_predictor is None:
        LOGGER.info(state.exercise_status_message)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield


app = FastAPI(title="PCOS AI API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "pcos_model_loaded": state.pcos_bundle is not None,
        "ir_model_loaded": state.ir_available,
        "calorie_model_loaded": state.calorie_predictor is not None,
        "exercise_model_loaded": state.exercise_predictor is not None,
    }


@app.post("/predict/pcos")
def predict_pcos(request: PredictionRequest) -> dict[str, Any]:
    if state.pcos_bundle is None:
        raise HTTPException(status_code=503, detail="PCOS model is not available. Train and save the PCOS model first.")
    if not request.features:
        raise HTTPException(status_code=422, detail="Request must include a non-empty 'features' object.")
    normalized_features, warnings = normalize_pcos_payload(state.pcos_bundle, request.features)
    result = predict_from_dict(state.pcos_bundle, normalized_features)
    if warnings:
        result["warnings"] = warnings
    return result


@app.post("/predict/ir")
def predict_ir(request: PredictionRequest) -> dict[str, Any]:
    if not state.ir_available or state.ir_bundle is None:
        return {
            "available": False,
            "message": state.ir_status_message,
        }
    if not request.features:
        raise HTTPException(status_code=422, detail="Request must include a non-empty 'features' object.")
    result = predict_from_dict(state.ir_bundle, request.features)
    result["available"] = True
    return result


@app.post("/predict/calories")
def predict_calories(request: CaloriePredictionRequest) -> dict[str, Any]:
    if state.calorie_predictor is None:
        raise HTTPException(status_code=503, detail=state.calorie_status_message)
    try:
        result = state.calorie_predictor.estimate(
            meal_name=request.meal_name,
            grams=request.grams,
            portion_count=request.portion_count,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["available"] = True
    return result


@app.post("/predict/exercise")
def predict_exercise(request: ExercisePredictionRequest) -> dict[str, Any]:
    if state.exercise_predictor is None:
        raise HTTPException(status_code=503, detail=state.exercise_status_message)
    try:
        return state.exercise_predictor.recommend(
            exercise_name=request.exercise_name,
            duration_minutes=request.duration_minutes,
            difficulty_level=request.difficulty_level,
            target_muscle_group=request.target_muscle_group,
            limit=request.limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
