from __future__ import annotations

from fastapi.testclient import TestClient

from src.pcos_ai import api
from src.pcos_ai.calorie_predictor import CaloriePredictor
from src.pcos_ai.exercise_predictor import ExercisePredictor


def _build_calorie_predictor() -> CaloriePredictor:
    return CaloriePredictor.from_csv("tests/fixtures/meal.csv")


def _build_exercise_predictor() -> ExercisePredictor:
    return ExercisePredictor.from_csv("data/exercise_predictor/exercises.csv")


def test_health_endpoint_reflects_model_state(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = object()
    api.state.exercise_predictor = object()
    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pcos_model_loaded"] is True
    assert body["ir_model_loaded"] is False
    assert body["calorie_model_loaded"] is True
    assert body["exercise_model_loaded"] is True


def test_predict_ir_returns_unavailable_when_model_missing(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = None
    api.state.exercise_predictor = None
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
    api.state.calorie_predictor = None
    api.state.exercise_predictor = None
    captured: dict[str, object] = {}

    def fake_predict(bundle: dict[str, object], features: dict[str, object]) -> dict[str, object]:
        captured["bundle"] = bundle
        captured["features"] = features
        return {
            "pcos_present": False,
            "shap_explanation": {
                "available": True,
                "provider": "shap",
                "top_features": [
                    {
                        "feature": "Age (yrs)",
                        "feature_value": 28,
                        "shap_value": 0.18,
                        "impact": "increases_risk",
                    }
                ],
            },
        }

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
    assert body["shap_explanation"]["available"] is True
    assert body["shap_explanation"]["top_features"][0]["feature"] == "Age (yrs)"
    assert "warnings" in body
    assert "Ignored unused field `Unused Field`." in body["warnings"]
    assert "Systolic and diastolic blood pressure are identical. Please double-check those values." in body["warnings"]


def test_predict_calories_supports_grams(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = _build_calorie_predictor()
    api.state.exercise_predictor = None

    with TestClient(api.app) as client:
        response = client.post("/predict/calories", json={"meal_name": "aloo gosht", "grams": 200})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["input_mode"] == "grams"
    assert body["matched_meal_name"] == "aloo gosht"
    assert body["estimated_calories"] == 458.33


def test_predict_calories_supports_portion_count(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = _build_calorie_predictor()
    api.state.exercise_predictor = None

    with TestClient(api.app) as client:
        response = client.post("/predict/calories", json={"meal_name": "aloo gosht", "portion_count": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["input_mode"] == "portion_count"
    assert body["portion_count"] == 2.0
    assert body["estimated_calories"] == 800.0
    assert "estimated_grams" in body


def test_predict_calories_accepts_portion_size_alias(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = _build_calorie_predictor()
    api.state.exercise_predictor = None

    with TestClient(api.app) as client:
        response = client.post("/predict/calories", json={"meal_name": "aloo gosht", "portion_size": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["input_mode"] == "portion_count"
    assert body["estimated_calories"] == 800.0


def test_predict_calories_rejects_dual_input_modes(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = _build_calorie_predictor()
    api.state.exercise_predictor = None

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/calories",
            json={"meal_name": "aloo gosht", "grams": 200, "portion_count": 1},
        )

    assert response.status_code == 422


def test_predict_exercise_supports_name_lookup(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = None
    api.state.exercise_predictor = _build_exercise_predictor()

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/exercise",
            json={"exercise_name": "push ups", "duration_minutes": 45},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["exercise_name"] == "Push-ups"
    assert body["match_type"] == "exact"
    assert body["duration_minutes"] == 45.0
    assert body["estimated_calories_burned"] == 300.0


def test_predict_exercise_supports_filtered_recommendations(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = None
    api.state.exercise_predictor = _build_exercise_predictor()

    with TestClient(api.app) as client:
        response = client.post(
            "/predict/exercise",
            json={"difficulty_level": "Beginner", "target_muscle_group": "glutes", "limit": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["request_mode"] == "recommendation_list"
    assert len(body["recommendations"]) == 2
    assert body["recommendations"][0]["exercise_name"] == "Lunges"


def test_predict_exercise_requires_name_or_filters(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_models", lambda: None)
    api.state.pcos_bundle = {"pipeline": object()}
    api.state.ir_bundle = None
    api.state.ir_available = False
    api.state.calorie_predictor = None
    api.state.exercise_predictor = _build_exercise_predictor()

    with TestClient(api.app) as client:
        response = client.post("/predict/exercise", json={})

    assert response.status_code == 422
