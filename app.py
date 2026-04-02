from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.pcos_ai.predict import load_model_bundle, predict_from_dataframe, predict_from_dict


def _read_uploaded_csv(file: Any) -> pd.DataFrame:
    return pd.read_csv(file)


def _manual_input_to_dict(raw_text: str) -> dict[str, Any]:
    if not raw_text.strip():
        return {}
    return json.loads(raw_text)


def _coerce_form_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return None
    try:
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return value


def _render_manual_form(feature_columns: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in feature_columns:
        user_value = st.text_input(column, key=f"field_{column}")
        payload[column] = _coerce_form_value(user_value)
    return payload


def main() -> None:
    st.set_page_config(page_title="PCOS AI Project", layout="wide")
    st.title("PCOS AI Project")
    st.caption("Decision-support predictions for PCOS and optional insulin resistance models.")

    pcos_model_file = st.file_uploader("Upload trained PCOS model (.joblib)", type=["joblib"], key="pcos_model")
    ir_model_file = st.file_uploader("Upload trained IR model (.joblib, optional)", type=["joblib"], key="ir_model")

    if not pcos_model_file:
        st.info("Upload a trained PCOS model to begin.")
        return

    temp_dir = Path(".streamlit_tmp")
    temp_dir.mkdir(exist_ok=True)

    pcos_path = temp_dir / "uploaded_pcos_model.joblib"
    pcos_path.write_bytes(pcos_model_file.read())
    pcos_bundle = load_model_bundle(pcos_path)

    ir_bundle = None
    if ir_model_file:
        ir_path = temp_dir / "uploaded_ir_model.joblib"
        ir_path.write_bytes(ir_model_file.read())
        ir_bundle = load_model_bundle(ir_path)

    input_mode = st.radio("Choose input mode", ["Manual Form", "Manual JSON", "One-row CSV"], horizontal=True)

    data_frame: pd.DataFrame | None = None
    feature_columns = list(pcos_bundle.get("feature_columns", []))
    if input_mode == "Manual Form":
        if not feature_columns:
            st.error("The uploaded model artifact does not include feature column metadata for form rendering.")
            return
        form_payload = _render_manual_form(feature_columns)
        if st.button("Run Prediction"):
            pcos_prediction = predict_from_dict(pcos_bundle, form_payload)
            st.subheader("PCOS Result")
            st.json(pcos_prediction)

            if ir_bundle is not None:
                st.subheader("Insulin Resistance Result")
                st.json(predict_from_dict(ir_bundle, form_payload))
            else:
                st.warning("IR model unavailable. Train and upload a valid IR model artifact to enable IR prediction.")
    elif input_mode == "Manual JSON":
        raw_json = st.text_area(
            "Enter one patient record as JSON",
            value='{"Age (yrs)": 28, "Weight (Kg)": 68, "Cycle(R/I)": "I"}',
            height=180,
        )
        if st.button("Run Prediction"):
            data = _manual_input_to_dict(raw_json)
            pcos_prediction = predict_from_dict(pcos_bundle, data)
            st.subheader("PCOS Result")
            st.json(pcos_prediction)

            if ir_bundle is not None:
                st.subheader("Insulin Resistance Result")
                st.json(predict_from_dict(ir_bundle, data))
            else:
                st.warning("IR model unavailable. Train and upload a valid IR model artifact to enable IR prediction.")
    else:
        csv_file = st.file_uploader("Upload a one-row CSV file", type=["csv"], key="csv_input")
        if csv_file:
            data_frame = _read_uploaded_csv(csv_file)
            st.dataframe(data_frame)

        if data_frame is not None and st.button("Run Prediction"):
            pcos_prediction = predict_from_dataframe(pcos_bundle, data_frame)
            st.subheader("PCOS Result")
            st.json(pcos_prediction)

            if ir_bundle is not None:
                st.subheader("Insulin Resistance Result")
                st.json(predict_from_dataframe(ir_bundle, data_frame))
            else:
                st.warning("IR model unavailable. Train and upload a valid IR model artifact to enable IR prediction.")


if __name__ == "__main__":
    main()
