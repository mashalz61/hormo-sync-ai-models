from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .feature_utils import coerce_dirty_numeric_series
from .utils import pretty_json, probability_to_level, slugify_column_name


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Model artifact is not a valid bundle with a pipeline.")
    return bundle


def _prediction_key(prefix: str, predicted_positive: bool) -> tuple[str, str]:
    if predicted_positive:
        return f"{prefix}_present", f"{prefix}_level"
    return f"{prefix}_present", f"{prefix}_risk_level"


def normalize_inference_inputs(data_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = data_frame.copy()
    yes_no_map = {
        "y": 1,
        "yes": 1,
        "true": 1,
        "present": 1,
        "n": 0,
        "no": 0,
        "false": 0,
        "absent": 0,
    }
    cycle_map = {
        "r": 2,
        "regular": 2,
        "i": 4,
        "ir": 4,
        "irregular": 4,
    }

    for column in normalized.columns:
        slug = slugify_column_name(column)
        series = normalized[column]
        if series.dtype == object:
            lowered = series.astype(str).str.strip().str.lower()
            if "y_n" in slug or "pregnant" in slug or "exercise" in slug:
                normalized[column] = lowered.map(lambda value: yes_no_map.get(value, value))
            elif "cycle_r_i" in slug:
                normalized[column] = lowered.map(lambda value: cycle_map.get(value, value))
        normalized[column] = coerce_dirty_numeric_series(normalized[column])
    return normalized


def align_features_to_training_schema(bundle: dict[str, Any], data_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        return data_frame

    aligned = data_frame.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.loc[:, feature_columns]


def predict_from_dataframe(bundle: dict[str, Any], data_frame: pd.DataFrame) -> dict[str, Any]:
    if len(data_frame) != 1:
        raise ValueError("Prediction expects exactly one input row.")
    data_frame = normalize_inference_inputs(data_frame)
    data_frame = align_features_to_training_schema(bundle, data_frame)
    pipeline = bundle["pipeline"]
    condition_name = bundle.get("condition_name", "condition")
    probabilities = pipeline.predict_proba(data_frame)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))
    probability = float(probabilities[0])
    predicted_positive = probability >= threshold
    level = probability_to_level(probability)
    present_key, level_key = _prediction_key(condition_name, predicted_positive)

    response: dict[str, Any] = {
        present_key: bool(predicted_positive),
        level_key: level,
    }
    if predicted_positive:
        response[f"{condition_name}_probability"] = probability
    else:
        response[f"{condition_name}_probability_of_developing"] = probability
    return response


def predict_from_dict(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return predict_from_dataframe(bundle, pd.DataFrame([payload]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction with a saved PCOS or IR model bundle.")
    parser.add_argument("--model", required=True, help="Path to a saved .joblib model bundle.")
    parser.add_argument("--input-json", help="JSON string representing a single input row.")
    parser.add_argument("--input-csv", help="CSV file containing a single row.")
    args = parser.parse_args()

    bundle = load_model_bundle(args.model)
    if args.input_json:
        import json

        payload = json.loads(args.input_json)
        result = predict_from_dict(bundle, payload)
    elif args.input_csv:
        data_frame = pd.read_csv(args.input_csv)
        result = predict_from_dataframe(bundle, data_frame)
    else:
        raise ValueError("Provide either --input-json or --input-csv.")

    print(pretty_json(result))


if __name__ == "__main__":
    main()
