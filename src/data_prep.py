"""Cleaning and feature engineering.

Two levels, kept deliberately separate:

* **Dataset-level** (`prepare_dataset`): things that only make sense on a whole
  extract - de-duplication, dropping leaky columns. Training only.
* **Row-level** (`RowCleaner`, `FeatureEngineer`): things that must happen
  identically at training time and at scoring time. These are sklearn
  transformers and live *inside* the saved pipeline, so the API physically
  cannot apply a different transformation from the one the model was fitted on.

That split is the main defence against train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

# Free-text country values arriving from three source systems.
COUNTRY_ALIASES = {
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "de": "Germany",
    "germany": "Germany",
    "fr": "France",
    "france": "France",
    "es": "Spain",
    "spain": "Spain",
    "nl": "Netherlands",
    "netherlands": "Netherlands",
    "ie": "Ireland",
    "ireland": "Ireland",
    "pl": "Poland",
    "poland": "Poland",
}

VALID_AGE_RANGE = (16, 100)
# Above this an `average_order_value` is assumed to be a cents/pence export bug.
AOV_CENTS_THRESHOLD = 1500.0


def canonicalise_country(value: object) -> object:
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    return COUNTRY_ALIASES.get(key, value.strip().title())


class RowCleaner(TransformerMixin, BaseEstimator):
    """Row-independent normalisation. Safe to run on a single API request.

    Fixes, in order:
      1. category strings -> canonical form (case, whitespace, ISO aliases);
      2. impossible ages -> NaN (imputed downstream, never silently clipped);
      3. `-1` recency sentinel -> NaN plus an explicit `never_ordered` flag;
      4. cents/pence unit errors on `average_order_value` -> divided by 100;
      5. negative counts (returns, orders, tickets) -> NaN.

    Nothing here is fitted from the data, so there is no leakage risk and the
    transform is identical for a training batch and a single live customer.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        if "country" in df:
            df["country"] = df["country"].map(canonicalise_country)
        for col in ("subscription_type", "acquisition_channel", "primary_device"):
            if col in df:
                df[col] = df[col].astype("object").map(lambda v: v.strip().lower() if isinstance(v, str) else v)

        if "age" in df:
            age = pd.to_numeric(df["age"], errors="coerce")
            df["age"] = age.where(age.between(*VALID_AGE_RANGE))

        if "days_since_last_order" in df:
            recency = pd.to_numeric(df["days_since_last_order"], errors="coerce")
            df["never_ordered"] = ((recency < 0) | recency.isna()).astype(int)
            df["days_since_last_order"] = recency.where(recency >= 0)

        if "average_order_value" in df:
            aov = pd.to_numeric(df["average_order_value"], errors="coerce")
            df["average_order_value"] = aov.where(aov < AOV_CENTS_THRESHOLD, aov / 100.0)

        for col in ("orders", "orders_last_90d", "support_tickets", "support_tickets_last_180d", "returns"):
            if col in df:
                series = pd.to_numeric(df[col], errors="coerce")
                df[col] = series.where(series >= 0)

        return df


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Ratio/intensity features - the ones that actually carry churn signal.

    Raw counts are confounded by tenure: 10 orders means something very
    different for a 2-month-old account than for a 3-year-old one. Every feature
    below normalises a count by the exposure that produced it, or compares
    recent behaviour against that customer's own baseline.

    `recency_vs_habit` is the important one: days since the last order divided by
    that customer's typical gap between orders. A weekly buyer silent for 40 days
    is alarming; a twice-a-year buyer silent for 40 days is normal. A model given
    only `days_since_last_order` cannot tell those apart.

    A log-recency term was tried and removed: it was collinear with raw recency,
    added nothing to CV PR-AUC (0.8391 vs 0.8387) and took a negative coefficient
    that made "days since last order" read as risk-*reducing* in the explanation
    table. An uninterpretable feature that buys no accuracy is a net loss.
    """

    engineered_features = [
        "orders_per_month",
        "expected_order_gap_days",
        "recency_vs_habit",
        "recent_order_share",
        "estimated_lifetime_value",
        "support_tickets_per_order",
        "support_recency_share",
        "return_rate",
        "order_value_cv",
        "never_ordered",
    ]

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        if "never_ordered" not in df:
            df["never_ordered"] = df["orders"].fillna(0).le(0).astype(int)

        tenure = df["tenure_months"].clip(lower=0.5)
        orders = df["orders"].fillna(0)
        recency = df["days_since_last_order"]

        df["orders_per_month"] = orders / tenure
        df["expected_order_gap_days"] = (tenure * 30.44) / orders.clip(lower=1)
        df["recency_vs_habit"] = recency / df["expected_order_gap_days"].clip(lower=1)
        df["recent_order_share"] = df["orders_last_90d"].fillna(0) / orders.clip(lower=1)
        df["estimated_lifetime_value"] = orders * df["average_order_value"].fillna(0.0)
        df["support_tickets_per_order"] = df["support_tickets"].fillna(0) / orders.clip(lower=1)
        df["support_recency_share"] = df["support_tickets_last_180d"].fillna(0) / df["support_tickets"].fillna(0).clip(
            lower=1
        )
        df["return_rate"] = df["returns"].fillna(0) / orders.clip(lower=1)
        df["order_value_cv"] = df["order_value_std"].fillna(0) / df["average_order_value"].clip(lower=1.0)
        return df.replace([np.inf, -np.inf], np.nan)


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_dataset(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset-level preparation: de-duplicate, drop leaks, split off the target."""
    n_before = len(df)
    df = df.drop_duplicates(subset=config.ID_COLUMN, keep="first").reset_index(drop=True)
    if verbose:
        print(f"  de-duplicated on {config.ID_COLUMN}: {n_before:,} -> {len(df):,} rows")

    dropped = [c for c in config.LEAKY_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    if verbose and dropped:
        print(f"  dropped leaky/non-feature columns: {dropped}")

    y = df[config.TARGET].astype(int)
    X = df.drop(columns=[config.TARGET, config.ID_COLUMN], errors="ignore")
    return X, y
