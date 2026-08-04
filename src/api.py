"""FastAPI prediction service.

Design notes:

* The model is loaded once at startup and cached in a small `ModelBundle`
  holder, not on every request. `/health` reports whether the load succeeded, so
  a container orchestrator can refuse to route traffic to a broken pod.
* The saved artefact is the *entire* pipeline, so there is no serving-side
  feature code that can drift away from training. `to_frame` only assembles
  columns; it never transforms values.
* The decision threshold and risk bands come from the model metadata written at
  training time, not from a constant in this file. Retraining can move the
  threshold without an API deployment.
* Requests are validated by pydantic before they reach the model, so a negative
  `orders` returns a 422 with a field-level message rather than a silent
  nonsense prediction.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src import config, explain
from src.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    to_frame,
)

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class ModelBundle:
    """Holds the fitted pipeline and its training-time metadata."""

    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def version(self) -> str:
        return f"{self.metadata.get('model_name', 'unknown')}@{self.metadata.get('trained_at', 'unknown')}"

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def load(self) -> None:
        if not config.MODEL_PATH.exists():
            logger.warning("No model at %s - run `python -m src.train` first.", config.MODEL_PATH)
            return
        self.pipeline = joblib.load(config.MODEL_PATH)
        if config.METADATA_PATH.exists():
            self.metadata = json.loads(config.METADATA_PATH.read_text())
        logger.info("Loaded model %s (threshold %.2f)", self.version, self.threshold)

    def risk_band(self, probability: float) -> str:
        bands = self.metadata.get("risk_bands", {})
        high = float(bands.get("high_risk", self.threshold))
        medium = float(bands.get("medium_risk", self.threshold / 2))
        if probability >= high:
            return "high_risk"
        if probability >= medium:
            return "medium_risk"
        return "low_risk"


bundle = ModelBundle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle.load()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Scores an e-commerce customer's probability of placing no order in the next 90 days. "
        "Returns a calibrated probability, a risk band and the main drivers behind the score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    if not bundle.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` to produce artifacts/churn_model.joblib.",
        )
    return bundle.pipeline


def _score(customers: list[CustomerFeatures], with_reasons: bool) -> list[PredictionResponse]:
    pipeline = _require_model()
    frame = to_frame(customers)
    probabilities = pipeline.predict_proba(frame)[:, 1]

    responses = []
    for i, customer in enumerate(customers):
        probability = float(probabilities[i])
        reasons = explain.top_reasons_for_customer(pipeline, frame.iloc[[i]]) if with_reasons else []
        responses.append(
            PredictionResponse(
                churn_probability=round(probability, 4),
                prediction=bundle.risk_band(probability),
                will_churn=probability >= bundle.threshold,
                decision_threshold=bundle.threshold,
                top_reasons=reasons,
                imputed_fields=customer.missing_fields(),
                model_version=bundle.version,
                customer_id=customer.customer_id,
            )
        )
    return responses


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe. 200 with `model_loaded: false` means do not route traffic here."""
    return HealthResponse(
        status="ok" if bundle.loaded else "degraded",
        model_loaded=bundle.loaded,
        model_version=bundle.version if bundle.loaded else None,
    )


@app.get("/model-info", tags=["ops"])
def model_info() -> JSONResponse:
    """Everything needed to reproduce or audit the deployed model."""
    _require_model()
    return JSONResponse(bundle.metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score one customer.

    Only `tenure_months` and `orders` are mandatory; every other field improves
    the estimate but is imputed if absent, and the response lists what was
    imputed so callers can see how much of the score rests on defaults.
    """
    started = time.perf_counter()
    response = _score([customer], with_reasons=True)[0]
    logger.info(
        "predict p=%.3f band=%s imputed=%d in %.1fms",
        response.churn_probability,
        response.prediction,
        len(response.imputed_fields),
        1000 * (time.perf_counter() - started),
    )
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(request: BatchRequest) -> BatchPredictionResponse:
    """Score up to 5,000 customers in one call.

    Per-customer explanations are omitted here - they are the expensive part, and
    batch callers are usually building a ranked campaign list, not a conversation
    script. Use `/predict` for the explained single-customer path.
    """
    responses = _score(request.customers, with_reasons=False)
    return BatchPredictionResponse(predictions=responses, count=len(responses))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
