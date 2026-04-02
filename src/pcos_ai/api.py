from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_model_bundle, predict_from_dict


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
    return predict_from_dict(state.pcos_bundle, request.features)


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
