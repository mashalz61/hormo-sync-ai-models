from __future__ import annotations

from fastapi.testclient import TestClient

from src.pcos_ai import api


def test_health_endpoint_reflects_model_state(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pcos_model_loaded"] is True
    assert body["ir_model_loaded"] is False


def test_predict_ir_returns_unavailable_when_model_missing(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    with TestClient(api.app) as client:
        response = client.post("/predict/ir", json={"features": {"Age (yrs)": 28}})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "IR model unavailable" in body["message"]


def test_normalize_pcos_payload_maps_aliases_and_fills_missing_fields() -> None:
    bundle = {
        "feature_columns": [
            "Age (yrs)",
            "Marraige Status (Yrs)",
            "Cycle(R/I)",
        ]
    }

    normalized, warnings = api.normalize_pcos_payload(
        bundle,
        {
            "Age (yrs)": 28,
            "Marriage Status": 0,
            "Cycle(R/I)": 2,
            "Unused Field": "drop me",
        },
    )

    assert normalized == {
        "Age (yrs)": 28,
        "Marraige Status (Yrs)": 0,
        "Cycle(R/I)": 2,
    }
    assert "Ignored unused field `Unused Field`." in warnings


def test_predict_pcos_returns_warnings_for_normalized_payload(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {
        "pipeline": object(),
        "feature_columns": [
            "Age (yrs)",
            "Marraige Status (Yrs)",
            "BP _Systolic (mmHg)",
            "BP _Diastolic (mmHg)",
        ],
    }
    api.state.ir_bundle = None
    api.state.ir_available = False
    captured: dict[str, object] = {}

    def fake_predict(bundle: dict[str, object], features: dict[str, object]) -> dict[str, object]:
        captured["bundle"] = bundle
        captured["features"] = features
        return {"pcos_present": False}

    monkeypatch.setattr(api, "predict_from_dict", fake_predict)
    with TestClient(api.app) as client:
        response = client.post(
            "/predict/pcos",
            json={
                "features": {
                    "Age (yrs)": 28,
                    "Marriage Status": 0,
                    "BP _Systolic (mmHg)": 118,
                    "BP _Diastolic (mmHg)": 118,
                    "Unused Field": 123,
                }
            },
        )

    assert response.status_code == 200
    assert captured["features"] == {
        "Age (yrs)": 28,
        "Marraige Status (Yrs)": 0,
        "BP _Systolic (mmHg)": 118,
        "BP _Diastolic (mmHg)": 118,
    }
    body = response.json()
    assert body["pcos_present"] is False
    assert "warnings" in body
    assert "Ignored unused field `Unused Field`." in body["warnings"]
    assert "Systolic and diastolic blood pressure are identical. Please double-check those values." in body["warnings"]
