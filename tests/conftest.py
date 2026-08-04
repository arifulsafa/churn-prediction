"""Shared fixtures. A small generated dataset keeps the suite under ~30s."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_generation import generate_dataset
from src.data_prep import prepare_dataset


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return generate_dataset(n_customers=1500, seed=7)


@pytest.fixture(scope="session")
def prepared(raw_df):
    return prepare_dataset(raw_df)


@pytest.fixture()
def messy_row() -> pd.DataFrame:
    """One row containing every data-quality defect the cleaner is meant to fix."""
    return pd.DataFrame(
        [
            {
                "age": 214,  # impossible
                "tenure_months": 12.0,
                "orders": 0,  # never ordered
                "orders_last_90d": 0,
                "average_order_value": 7250.0,  # exported in cents
                "order_value_std": None,
                "days_since_last_order": -1,  # sentinel, not a real recency
                "support_tickets": None,  # NULL meaning 0
                "support_tickets_last_180d": 0,
                "returns": -2,  # impossible count
                "discount_order_share": 0.3,
                "newsletter_opens_last_90d": 1,
                "subscription_type": "  PREMIUM ",
                "country": "uk",
                "acquisition_channel": "Organic",
                "primary_device": "Mobile",
            }
        ]
    )
