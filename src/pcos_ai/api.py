from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_model_bundle, predict_from_dict
from .utils import slugify_column_name


LOGGER = logging.getLogger("pcos_ai.api")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class ApiState:
    def __init__(self) -> None:
        self.pcos_bundle: dict[str, Any] | None = None
        self.ir_bundle: dict[str, Any] | None = None
        self.ir_available = False
        self.ir_status_message = "IR model unavailable. Train and save an IR model to enable this endpoint."

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


def _safe_load_bundle(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        LOGGER.warning("Model file not found: %s", path)
        return None
    try:
        return load_model_bundle(path)
    except Exception as exc:
        LOGGER.exception("Failed to load model bundle from %s: %s", path, exc)
        return None


def load_models() -> None:
    state.pcos_bundle = _safe_load_bundle(_default_model_path("best_pcos_model.joblib"))
    state.ir_bundle = _safe_load_bundle(_default_model_path("best_insulin_resistance_model.joblib"))
    state.ir_available = state.ir_bundle is not None
    if state.pcos_bundle is None:
        LOGGER.warning("PCOS model is not loaded. Train the PCOS model before using the API.")
    if not state.ir_available:
        LOGGER.info(state.ir_status_message)


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
