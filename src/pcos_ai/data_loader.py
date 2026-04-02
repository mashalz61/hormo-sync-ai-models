from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import clean_column_name


def load_excel_data(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found: {file_path}")

    data_frame = pd.read_excel(file_path, engine="openpyxl")
    if data_frame.empty:
        raise ValueError(f"Input file contains no rows: {file_path}")

    cleaned_columns = [clean_column_name(column) for column in data_frame.columns]
    data_frame.columns = cleaned_columns
    data_frame = data_frame.loc[:, ~data_frame.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    return data_frame

