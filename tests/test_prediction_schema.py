from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

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
