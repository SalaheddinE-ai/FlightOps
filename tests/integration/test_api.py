
import pytest
from fastapi.testclient import TestClient

PAYLOAD = {"type": "M", "air_temperature": 298.1, "process_temperature": 308.6,
           "rotational_speed": 1551, "torque": 42.8, "tool_wear": 120}


@pytest.fixture()
def client(trained_session, monkeypatch):
    monkeypatch.setenv("FLIGHTOPS_ROOT", str(trained_session["root"]))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(trained_session['root'] / 'mlflow.db').as_posix()}")
    from api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health_ready_and_model_info(client, trained_session):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200
    info = client.get("/model/info").json()
    assert info["model_version"] == trained_session["version"] and info["alias"] == "champion"
    assert info["run_id"] and 0 < info["threshold"] < 1
    assert [f["name"] for f in info["feature_schema"]][0] == "type"


def test_predict_returns_registered_version_and_valid_fields(client, trained_session):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    b = r.json()
    assert b["model_version"] == trained_session["version"]
    assert 0 <= b["failure_probability"] <= 1 and b["prediction"] in (0, 1)
    assert b["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert b["prediction"] == int(b["failure_probability"] >= b["threshold"])


def test_high_risk_conditions_score_higher_than_benign(client):
    benign = client.post("/predict", json=PAYLOAD).json()["failure_probability"]
    risky = client.post("/predict", json={**PAYLOAD, "torque": 70.0, "tool_wear": 240}).json()["failure_probability"]
    assert risky > benign


def test_batch_prediction(client):
    r = client.post("/predict/batch", json={"observations": [PAYLOAD, {**PAYLOAD, "type": "H"}]})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert client.post("/predict/batch", json={"observations": []}).status_code == 422


@pytest.mark.parametrize("patch", [{"torque": -1}, {"type": "X"}, {"tool_wear": -5}, {"air_temperature": 0},
                                   {"unknown_field": 1}])
def test_validation_errors_are_clear_422(client, patch):
    r = client.post("/predict", json={**PAYLOAD, **patch})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_missing_field_rejected(client):
    bad = {k: v for k, v in PAYLOAD.items() if k != "torque"}
    assert client.post("/predict", json=bad).status_code == 422


def test_shap_explanation_optional(client):
    b = client.post("/predict?explain=true", json=PAYLOAD).json()
    if b["explanation"] is not None:
        assert b["explanation"]["contributions"]


def test_predictions_are_logged_and_metrics_exposed(client):
    client.post("/predict", json=PAYLOAD)
    client.post("/predict", json={**PAYLOAD, "torque": -1})
    text = client.get("/metrics").text
    assert "flightops_predictions_total" in text and "flightops_invalid_inputs_total 1" in text
    assert "flightops_model_loaded 1" in text
    from flightops_ml.monitoring.prediction_log import PredictionLog

    log = PredictionLog()
    assert len(log.predictions()) >= 1 and len(log.errors()) >= 1


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("FLIGHTOPS_API_KEY", "s3cret")
    assert client.post("/predict", json=PAYLOAD).status_code == 401
    assert client.post("/predict", json=PAYLOAD, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # probes stay open


def test_no_promotion_endpoint_is_exposed(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert not any(k in p for p in paths for k in ("promote", "deploy", "rollback", "approve"))


def test_service_without_model_reports_not_ready(isolated_root):
    from api.main import create_app

    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=PAYLOAD).status_code == 503
