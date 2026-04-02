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
