from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_categorical_missing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace({None: np.nan})


def split_feature_types(data_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = list(data_frame.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in data_frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(data_frame: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_columns, categorical_columns = split_feature_types(data_frame)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "normalize_missing",
                FunctionTransformer(normalize_categorical_missing, validate=False),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
