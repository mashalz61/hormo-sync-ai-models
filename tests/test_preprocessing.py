from __future__ import annotations

import pandas as pd

from src.pcos_ai.preprocessing import build_preprocessor, split_feature_types


def test_split_feature_types_handles_numeric_and_categorical() -> None:
    data_frame = pd.DataFrame(
        {
            "Age (yrs)": [25, 29, 31],
            "Cycle(R/I)": ["R", "I", "R"],
            "Weight (Kg)": [60.0, None, 75.5],
        }
    )
    numeric_columns, categorical_columns = split_feature_types(data_frame)
    assert "Age (yrs)" in numeric_columns
    assert "Weight (Kg)" in numeric_columns
    assert "Cycle(R/I)" in categorical_columns


def test_preprocessor_transforms_without_leakage_shape_errors() -> None:
    data_frame = pd.DataFrame(
        {
            "Age (yrs)": [25, 29, 31],
            "Cycle(R/I)": ["R", "I", None],
            "Weight (Kg)": [60.0, None, 75.5],
        }
    )
    preprocessor = build_preprocessor(data_frame)
    transformed = preprocessor.fit_transform(data_frame)
    assert transformed.shape[0] == 3

