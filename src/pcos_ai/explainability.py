from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_value(frame: pd.DataFrame, column: str) -> Any:
    value = frame.iloc[0][column]
    if pd.isna(value):
        return None
    return _json_safe(value)


def _build_predict_fn(bundle: dict[str, Any]):
    pipeline = bundle["pipeline"]
    feature_columns = list(bundle.get("feature_columns", []))

    def predict_positive_probability(frame_like: Any) -> np.ndarray:
        if isinstance(frame_like, pd.DataFrame):
            frame = frame_like.copy()
        else:
            frame = pd.DataFrame(frame_like, columns=feature_columns or None)

        from .predict import align_features_to_training_schema, normalize_inference_inputs

        frame = normalize_inference_inputs(frame)
        frame = align_features_to_training_schema(bundle, frame)
        return pipeline.predict_proba(frame)[:, 1]

    return predict_positive_probability


def explain_prediction_with_shap(
    bundle: dict[str, Any],
    data_frame: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    background_frame = bundle.get("explainability_background")
    if background_frame is None:
        return {
            "available": False,
            "provider": "shap",
            "reason": "This model artifact was saved without SHAP background data.",
        }

    try:
        import shap  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "provider": "shap",
            "reason": f"SHAP is not installed: {exc}",
        }

    background = pd.DataFrame(background_frame).copy()
    predict_positive_probability = _build_predict_fn(bundle)
    feature_columns = list(data_frame.columns)
    max_evals = max((2 * len(feature_columns)) + 1, 101)
    explainer = shap.Explainer(
        predict_positive_probability,
        background,
        algorithm="permutation",
        feature_names=feature_columns,
    )
    explanation = explainer(data_frame, max_evals=max_evals)
    shap_values = np.asarray(explanation.values[0], dtype=float)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

    top_features: list[dict[str, Any]] = []
    for index in top_indices:
        contribution = float(shap_values[index])
        top_features.append(
            {
                "feature": feature_columns[index],
                "feature_value": _feature_value(data_frame, feature_columns[index]),
                "shap_value": contribution,
                "impact": "increases_risk" if contribution >= 0 else "decreases_risk",
            }
        )

    return {
        "available": True,
        "provider": "shap",
        "base_value": base_value,
        "feature_count": len(feature_columns),
        "top_features": top_features,
    }


def summarize_shap_importance(
    bundle: dict[str, Any],
    data_frame: pd.DataFrame,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    explanation, evaluation_frame = compute_global_shap_explanation(
        bundle,
        data_frame,
        max_rows=max_rows,
    )
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    summary = pd.DataFrame(
        {
            "feature": list(evaluation_frame.columns),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
        }
    )
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    return summary.loc[:, ["rank", "feature", "mean_abs_shap", "mean_shap"]]


def compute_global_shap_explanation(
    bundle: dict[str, Any],
    data_frame: pd.DataFrame,
    *,
    max_rows: int = 100,
):
    background_frame = bundle.get("explainability_background")
    if background_frame is None:
        raise ValueError("This model artifact was saved without SHAP background data.")

    try:
        import shap  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"SHAP is not installed: {exc}") from exc

    evaluation_frame = data_frame.copy().reset_index(drop=True)
    if len(evaluation_frame) > max_rows:
        evaluation_frame = evaluation_frame.iloc[:max_rows].copy()
    background = pd.DataFrame(background_frame).copy()
    predict_positive_probability = _build_predict_fn(bundle)
    feature_columns = list(evaluation_frame.columns)
    max_evals = max((2 * len(feature_columns)) + 1, 101)
    explainer = shap.Explainer(
        predict_positive_probability,
        background,
        algorithm="permutation",
        feature_names=feature_columns,
    )
    explanation = explainer(evaluation_frame, max_evals=max_evals)
    return explanation, evaluation_frame
