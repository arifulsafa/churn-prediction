"""Cleaning and feature-engineering behaviour.

These are the tests that matter most: every defect below, left unfixed, produces
a model that trains happily and is wrong in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_prep import FeatureEngineer, RowCleaner, canonicalise_country, prepare_dataset


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("uk", "United Kingdom"), ("UK", "United Kingdom"), ("GB", "United Kingdom"),
     ("United Kingdom ", "United Kingdom"), ("germany", "Germany"), ("DE", "Germany")],
)
def test_country_aliases_collapse_to_one_value(raw, expected):
    assert canonicalise_country(raw) == expected


def test_country_unknown_value_is_title_cased_not_dropped():
    assert canonicalise_country(" portugal ") == "Portugal"


def test_row_cleaner_fixes_every_known_defect(messy_row):
    cleaned = RowCleaner().fit_transform(messy_row).iloc[0]

    assert pd.isna(cleaned["age"]), "age 214 must become NaN, not be clipped to 100"
    assert cleaned["average_order_value"] == pytest.approx(72.50), "cents export must be divided by 100"
    assert pd.isna(cleaned["days_since_last_order"]), "-1 sentinel must not survive as a real recency"
    assert cleaned["never_ordered"] == 1
    assert pd.isna(cleaned["returns"]), "negative counts are impossible and must become NaN"
    assert cleaned["subscription_type"] == "premium"
    assert cleaned["country"] == "United Kingdom"


def test_sentinel_recency_is_never_treated_as_recent(messy_row):
    """The single most damaging defect: -1 would make dormant customers look active."""
    cleaned = RowCleaner().fit_transform(messy_row)
    assert not (cleaned["days_since_last_order"].fillna(999) < 0).any()


def test_row_cleaner_is_row_independent(raw_df):
    """Cleaning one row alone must give the same result as cleaning the batch.

    If this fails, the API and the training job disagree - the definition of
    train/serve skew.
    """
    cleaner = RowCleaner()
    batch = cleaner.fit_transform(raw_df.head(50))
    single = cleaner.fit_transform(raw_df.head(50).iloc[[7]])
    pd.testing.assert_frame_equal(
        batch.iloc[[7]].reset_index(drop=True), single.reset_index(drop=True), check_dtype=False
    )


def test_feature_engineer_produces_no_infinities(raw_df):
    """Zero-order customers make every ratio a division by zero if unguarded."""
    engineered = FeatureEngineer().fit_transform(RowCleaner().fit_transform(raw_df))
    numeric = engineered.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_engineered_features_are_all_present(raw_df):
    engineered = FeatureEngineer().fit_transform(RowCleaner().fit_transform(raw_df))
    for name in FeatureEngineer.engineered_features:
        assert name in engineered.columns


def test_recency_vs_habit_distinguishes_shopper_cadences():
    """40 days of silence means different things to a weekly and a yearly buyer."""
    frame = pd.DataFrame(
        [
            {"tenure_months": 12, "orders": 52, "days_since_last_order": 40},  # weekly buyer, alarming
            {"tenure_months": 12, "orders": 2, "days_since_last_order": 40},  # occasional buyer, normal
        ]
    )
    for col in ("orders_last_90d", "average_order_value", "support_tickets",
                "support_tickets_last_180d", "returns", "order_value_std"):
        frame[col] = 0
    out = FeatureEngineer().fit_transform(frame)
    assert out["recency_vs_habit"].iloc[0] > out["recency_vs_habit"].iloc[1]


def test_prepare_dataset_removes_duplicate_customers(raw_df):
    X, _ = prepare_dataset(raw_df)
    assert raw_df[config.ID_COLUMN].duplicated().any(), "fixture should contain the injected duplicates"
    assert len(X) == raw_df[config.ID_COLUMN].nunique()


def test_prepare_dataset_drops_every_leaky_column(raw_df):
    X, y = prepare_dataset(raw_df)
    for col in config.LEAKY_COLUMNS + [config.TARGET, config.ID_COLUMN]:
        assert col not in X.columns
    assert set(y.unique()) <= {0, 1}
