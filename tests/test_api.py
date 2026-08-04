"""API contract tests.

Run against the real trained artefact when one exists, and against a
purpose-trained tiny model otherwise, so the suite passes on a fresh clone
before `python -m src.train` has been run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config
from src.api import app, bundle
from src.data_generation import generate_dataset
from src.data_prep import prepare_dataset
from src.features import build_pipeline

EXAMPLE_PAYLOAD = {
    "tenure_months": 14,
    "orders": 8,
    "average_order_value": 72.50,
    "days_since_last_order": 63,
    "support_tickets": 4,
}


@pytest.fixture(scope="module")
def client():
    if not config.MODEL_PATH.exists():
        X, y = prepare_dataset(generate_dataset(n_customers=1200, seed=11))
        bundle.pipeline = build_pipeline(
            LogisticRegression(max_iter=800, class_weight="balanced", random_state=0)
        ).fit(X, y)
        bundle.metadata = {
            "model_name": "test_stub",
            "trained_at": "test",
            "decision_threshold": 0.43,
            "risk_bands": {"high_risk": 0.43, "medium_risk": 0.215},
        }
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_predict_matches_the_brief_example_contract(client):
    """The exact payload from the specification must work as written."""
    response = client.post("/predict", json=EXAMPLE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"low_risk", "medium_risk", "high_risk"}
    assert isinstance(body["will_churn"], bool)
    assert body["model_version"]


def test_response_declares_which_fields_were_imputed(client):
    body = client.post("/predict", json=EXAMPLE_PAYLOAD).json()
    assert "country" in body["imputed_fields"]
    assert "orders" not in body["imputed_fields"]


def test_predict_returns_actionable_reasons(client):
    body = client.post("/predict", json=EXAMPLE_PAYLOAD).json()
    assert len(body["top_reasons"]) >= 1
    for reason in body["top_reasons"]:
        assert reason["direction"] in {"increases_risk", "decreases_risk"}
        assert reason["label"]


def test_risk_band_is_consistent_with_the_threshold(client):
    body = client.post("/predict", json=EXAMPLE_PAYLOAD).json()
    above = body["churn_probability"] >= body["decision_threshold"]
    assert body["will_churn"] is above
    assert (body["prediction"] == "high_risk") is above


def test_a_dormant_customer_scores_higher_than_an_active_one(client):
    dormant = client.post("/predict", json={**EXAMPLE_PAYLOAD, "days_since_last_order": 300}).json()
    active = client.post("/predict", json={**EXAMPLE_PAYLOAD, "days_since_last_order": 3}).json()
    assert dormant["churn_probability"] > active["churn_probability"]


@pytest.mark.parametrize(
    "payload",
    [
        {"tenure_months": 14},  # missing the mandatory `orders`
        {"tenure_months": 14, "orders": -1},  # negative count
        {"tenure_months": -5, "orders": 2},  # negative tenure
        {"tenure_months": 14, "orders": 2, "discount_order_share": 1.4},  # not a proportion
        {"tenure_months": 14, "orders": 2, "unexpected_field": 1},  # typo protection
    ],
)
def test_invalid_payloads_are_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_minimal_payload_is_accepted(client):
    """Only tenure and orders are mandatory; the rest is imputed."""
    response = client.post("/predict", json={"tenure_months": 3, "orders": 1})
    assert response.status_code == 200


def test_batch_endpoint_scores_every_customer_and_echoes_ids(client):
    payload = {
        "customers": [
            {**EXAMPLE_PAYLOAD, "customer_id": "CUST-000001"},
            {**EXAMPLE_PAYLOAD, "customer_id": "CUST-000002", "days_since_last_order": 5},
        ]
    }
    body = client.post("/predict/batch", json=payload).json()
    assert body["count"] == 2
    assert [p["customer_id"] for p in body["predictions"]] == ["CUST-000001", "CUST-000002"]


def test_batch_and_single_predictions_agree(client):
    single = client.post("/predict", json=EXAMPLE_PAYLOAD).json()["churn_probability"]
    batched = client.post("/predict/batch", json={"customers": [EXAMPLE_PAYLOAD]}).json()
    assert batched["predictions"][0]["churn_probability"] == pytest.approx(single, abs=1e-6)


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_info_exposes_the_audit_trail(client):
    body = client.get("/model-info").json()
    assert "decision_threshold" in body
    assert "model_name" in body


def test_missing_model_returns_503_not_500():
    """An un-trained deployment must fail loudly and correctly, not crash."""
    saved_pipeline, saved_metadata = bundle.pipeline, bundle.metadata
    bundle.pipeline, bundle.metadata = None, {}
    try:
        with TestClient(app) as c:
            if bundle.pipeline is not None:  # lifespan reloaded a real artefact
                bundle.pipeline = None
            assert c.post("/predict", json=EXAMPLE_PAYLOAD).status_code == 503
            assert c.get("/health").json()["model_loaded"] is False
    finally:
        bundle.pipeline, bundle.metadata = saved_pipeline, saved_metadata
