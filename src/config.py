"""Central configuration: paths, constants and modelling conventions.

Keeping these in one place means the data generator, training pipeline, tests and
API all agree on file locations, the churn definition and the feature contract.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "customers_raw.csv"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
METRICS_PATH = REPORT_DIR / "evaluation.json"

RANDOM_SEED = 42

# --- Churn definition -------------------------------------------------------
# A customer is churned if they place no order in the OUTCOME_WINDOW_DAYS that
# follow the snapshot date. Features are only ever built from behaviour strictly
# before the snapshot date, so the label lives entirely in the future.
SNAPSHOT_DATE = "2026-01-01"
OUTCOME_WINDOW_DAYS = 90
OBSERVATION_WINDOW_DAYS = 365

# --- Split strategy ---------------------------------------------------------
TEST_SIZE = 0.20  # held out once, touched only for the final report
VALIDATION_FOLDS = 5  # stratified K-fold on the remaining 80% for selection/tuning

# --- Business cost assumptions (used to pick the decision threshold) ---------
# Retention offer sent to a flagged customer.
COST_PER_INTERVENTION = 8.0
# Expected margin lost when a churner is missed (avg 12-month forward margin).
COST_PER_MISSED_CHURNER = 120.0
# Fraction of correctly-flagged churners actually saved by the offer.
INTERVENTION_SUCCESS_RATE = 0.30

# --- Feature contract -------------------------------------------------------
# Columns the API accepts / the pipeline consumes. Engineered features are
# derived inside the pipeline so the API request stays small and human-friendly.
NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "orders",
    "orders_last_90d",
    "average_order_value",
    "order_value_std",
    "days_since_last_order",
    "support_tickets",
    "support_tickets_last_180d",
    "returns",
    "discount_order_share",
    "newsletter_opens_last_90d",
]

CATEGORICAL_FEATURES = [
    "subscription_type",
    "country",
    "acquisition_channel",
    "primary_device",
]

TARGET = "churned"
ID_COLUMN = "customer_id"

# Columns that must never reach the model. `subscription_cancelled_at_export`
# is recorded when the CRM extract is taken (i.e. AFTER the outcome window) and
# is therefore a textbook leak; see README "Data leakage".
LEAKY_COLUMNS = [
    "subscription_cancelled_at_export",
    "orders_in_outcome_window",
    "snapshot_date",
]

RISK_BANDS = {"low_risk": 0.0, "medium_risk": 0.35, "high_risk": 0.60}
