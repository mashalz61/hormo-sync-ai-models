from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.pcos_ai import predict as predict_module
from src.pcos_ai.explainability import summarize_shap_importance
from src.pcos_ai.predict import predict_from_dataframe
from src.pcos_ai.preprocessing import build_preprocessor


def test_prediction_schema_for_positive_or_negative_output() -> None:
    training_frame = pd.DataFrame(
        {
            "Age (yrs)": [25, 30, 35, 28],
            "Cycle(R/I)": ["R", "I", "I", "R"],
            "Weight (Kg)": [60, 70, 80, 65],
        }
    )
    target = [0, 1, 1, 0]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(training_frame)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(training_frame, target)
    bundle = {"condition_name": "pcos", "pipeline": pipeline, "threshold": 0.5}

    result = predict_from_dataframe(bundle, training_frame.iloc[[0]])
    assert "pcos_present" in result
    assert "pcos_level" in result or "pcos_risk_level" in result
    assert "pcos_probability" in result or "pcos_probability_of_developing" in result


def test_prediction_schema_includes_shap_explanation(monkeypatch) -> None:
    training_frame = pd.DataFrame(
        {
            "Age (yrs)": [25, 30, 35, 28],
            "Cycle(R/I)": ["R", "I", "I", "R"],
            "Weight (Kg)": [60, 70, 80, 65],
        }
    )
    target = [0, 1, 1, 0]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(training_frame)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(training_frame, target)
    bundle = {
        "condition_name": "pcos",
        "pipeline": pipeline,
        "threshold": 0.5,
        "feature_columns": list(training_frame.columns),
        "explainability_background": training_frame,
    }

    monkeypatch.setattr(
        predict_module,
        "explain_prediction_with_shap",
        lambda bundle, data_frame: {
            "available": True,
            "provider": "shap",
            "feature_count": len(data_frame.columns),
            "top_features": [
                {
                    "feature": "Weight (Kg)",
                    "feature_value": 60,
                    "shap_value": 0.12,
                    "impact": "increases_risk",
                }
            ],
        },
    )

    result = predict_from_dataframe(bundle, training_frame.iloc[[0]])
    assert "shap_explanation" in result
    assert result["shap_explanation"]["available"] is True
    assert result["shap_explanation"]["top_features"][0]["feature"] == "Weight (Kg)"


def test_summarize_shap_importance_orders_features(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Age (yrs)": [25, 30],
            "Weight (Kg)": [60, 80],
        }
    )
    bundle = {
        "pipeline": object(),
        "feature_columns": list(frame.columns),
        "explainability_background": frame,
    }

    class DummyExplanation:
        values = [[0.1, -0.4], [0.2, -0.6]]

    class DummyExplainer:
        def __call__(self, data_frame, max_evals):
            return DummyExplanation()

    monkeypatch.setitem(__import__("sys").modules, "shap", SimpleNamespace(Explainer=lambda *args, **kwargs: DummyExplainer()))

    summary = summarize_shap_importance(bundle, frame)
    assert list(summary["feature"]) == ["Weight (Kg)", "Age (yrs)"]
    assert list(summary["rank"]) == [1, 2]
