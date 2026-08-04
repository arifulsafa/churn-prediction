"""Request/response contracts for the prediction API.

The brief's example request carries only five fields, but the model was trained
on more. Rather than reject those requests or silently guess, every extra field
is optional: anything not supplied is passed through as missing and handled by
the same imputer the model was fitted with, and the response names exactly which
fields were imputed. Callers can therefore start with five fields and improve
their accuracy by sending more, without a breaking change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src import config


class CustomerFeatures(BaseModel):
    """A single customer at scoring time."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "tenure_months": 14,
                "orders": 8,
                "average_order_value": 72.50,
                "days_since_last_order": 63,
                "support_tickets": 4,
            }
        },
    )

    # --- core behavioural fields (the brief's example payload) ---------------
    tenure_months: float = Field(..., ge=0, le=600, description="Months since account creation.")
    orders: int = Field(..., ge=0, description="Lifetime completed orders.")
    average_order_value: float | None = Field(None, ge=0, description="Mean order value in GBP.")
    days_since_last_order: float | None = Field(
        None, ge=-1, description="Days since the most recent order; -1 or null if the customer never ordered."
    )
    support_tickets: int | None = Field(None, ge=0, description="Lifetime support tickets.")

    # --- optional enrichment -------------------------------------------------
    age: int | None = Field(None, ge=0, le=120)
    orders_last_90d: int | None = Field(None, ge=0)
    order_value_std: float | None = Field(None, ge=0)
    support_tickets_last_180d: int | None = Field(None, ge=0)
    returns: int | None = Field(None, ge=0)
    discount_order_share: float | None = Field(None, ge=0, le=1)
    newsletter_opens_last_90d: int | None = Field(None, ge=0)
    subscription_type: str | None = Field(None, examples=["free", "basic", "premium"])
    country: str | None = Field(None, examples=["United Kingdom", "DE"])
    acquisition_channel: str | None = Field(None, examples=["organic", "paid_search", "social"])
    primary_device: str | None = Field(None, examples=["mobile", "desktop", "tablet"])

    customer_id: str | None = Field(None, description="Echoed back for joining; not used as a feature.")

    def missing_fields(self) -> list[str]:
        supplied = self.model_dump(exclude_none=True)
        expected = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
        return [f for f in expected if f not in supplied]


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=5000)


class Reason(BaseModel):
    feature: str
    label: str
    direction: str
    contribution: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(..., description="Calibrated probability of no order in the next 90 days.")
    prediction: str = Field(..., description="low_risk | medium_risk | high_risk")
    will_churn: bool = Field(..., description="True when the probability is at or above the decision threshold.")
    decision_threshold: float
    top_reasons: list[Reason] = []
    imputed_fields: list[str] = Field(
        default=[], description="Fields not supplied in the request and filled by the model's imputer."
    )
    model_version: str
    customer_id: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None


def to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    """Build the exact column set the pipeline expects.

    Absent fields become `np.nan` (not `None`): sklearn's `SimpleImputer` detects
    NaN in object columns but not `None`, so using `None` here would silently
    push an un-imputed null into the one-hot encoder.
    """
    expected = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    rows = []
    for customer in customers:
        payload = customer.model_dump()
        rows.append({col: (payload.get(col) if payload.get(col) is not None else np.nan) for col in expected})
    return pd.DataFrame(rows, columns=expected)
