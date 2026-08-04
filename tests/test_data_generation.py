"""Properties the synthetic generator must hold.

The dataset is part of the deliverable, so it gets the same scrutiny as the
model: reproducible, plausibly balanced, and - critically - carrying a label the
model cannot trivially reconstruct from a single column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_generation import generate_dataset


def test_generation_is_reproducible_for_a_fixed_seed():
    a = generate_dataset(n_customers=300, seed=5)
    b = generate_dataset(n_customers=300, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = generate_dataset(n_customers=300, seed=5)
    b = generate_dataset(n_customers=300, seed=6)
    assert not a["days_since_last_order"].equals(b["days_since_last_order"])


def test_churn_rate_is_imbalanced_but_learnable(raw_df):
    rate = raw_df[config.TARGET].mean()
    assert 0.15 < rate < 0.40, f"churn rate {rate:.2%} outside the intended range"


def test_label_is_not_a_deterministic_function_of_recency(raw_df):
    """Guards against the circular dataset that makes every model look perfect.

    If churn were defined as a recency cut-off, a single threshold would separate
    the classes exactly. It must not.
    """
    recency = raw_df["days_since_last_order"].replace(-1, np.nan)
    churned = raw_df[config.TARGET] == 1
    overlap_low = recency[churned].quantile(0.10)
    overlap_high = recency[~churned].quantile(0.90)
    assert overlap_low < overlap_high, "class distributions must overlap in recency"


def test_expected_data_quality_defects_are_present(raw_df):
    assert raw_df[config.ID_COLUMN].duplicated().any()
    assert raw_df["age"].isna().any()
    assert (raw_df["age"] > 100).any()
    assert (raw_df["days_since_last_order"] == -1).any()
    assert (raw_df["average_order_value"] > 1500).any()
    assert raw_df["country"].nunique() > len(set(raw_df["country"].str.strip().str.lower()))


def test_leaky_column_exists_so_the_pipeline_has_something_to_drop(raw_df):
    assert "subscription_cancelled_at_export" in raw_df.columns
    correlation = raw_df["subscription_cancelled_at_export"].corr(raw_df[config.TARGET])
    assert correlation > 0.5, "the demonstration column must actually be leaky"


def test_counts_are_internally_consistent(raw_df):
    assert (raw_df["orders_last_90d"] <= raw_df["orders"]).all()
    assert (raw_df["returns"] <= raw_df["orders"]).all()
    assert (raw_df["support_tickets_last_180d"] <= raw_df["support_tickets"].fillna(np.inf)).all()


def test_never_ordered_customers_have_no_order_value(raw_df):
    never = raw_df["orders"] == 0
    assert raw_df.loc[never, "average_order_value"].isna().all()
