"""Metric and threshold-selection arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.evaluate import choose_threshold, compute_metrics, expected_cost, lift_at_k


def test_metrics_match_a_hand_computed_confusion_matrix():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.1, 0.05])
    m = compute_metrics(y_true, y_proba, threshold=0.5)

    assert (m.tp, m.fn, m.fp, m.tn) == (2, 1, 1, 4)
    assert m.precision == 2 / 3  # 2 true positives out of 3 flagged
    assert m.recall == 2 / 3  # 2 of the 3 real churners caught
    assert m.f1 == 2 / 3
    assert m.accuracy == 0.75


def test_expected_cost_uses_the_stated_business_assumptions():
    y_true = np.array([1, 0])
    y_pred = np.array([0, 0])  # one missed churner, nothing flagged
    assert expected_cost(y_true, y_pred) == config.COST_PER_MISSED_CHURNER / 2


def test_missing_a_churner_costs_more_than_flagging_a_loyal_customer():
    """The asymmetry is the whole reason recall is prioritised over precision."""
    y_true = np.array([1, 0, 0, 0])
    miss = expected_cost(y_true, np.array([0, 0, 0, 0]))
    false_alarm = expected_cost(y_true, np.array([1, 1, 0, 0]))
    assert miss > false_alarm


def test_choose_threshold_returns_the_cost_minimising_cut_off():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.27, size=2000)
    y_proba = np.clip(rng.normal(0.3 + 0.35 * y_true, 0.18), 0, 1)

    threshold, sweep = choose_threshold(y_true, y_proba)
    costs = [row["cost"] for row in sweep]
    assert min(costs) == next(r["cost"] for r in sweep if r["threshold"] == threshold)


def test_cost_optimal_threshold_favours_recall():
    """With a 15:1 cost ratio the optimum must sit below the naive 0.5."""
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.27, size=3000)
    y_proba = np.clip(rng.normal(0.3 + 0.3 * y_true, 0.2), 0, 1)
    threshold, sweep = choose_threshold(y_true, y_proba)
    chosen = next(r for r in sweep if r["threshold"] == threshold)
    assert threshold < 0.5
    assert chosen["recall"] > chosen["precision"]


def test_lift_is_one_for_random_scores_and_higher_for_useful_ones():
    rng = np.random.default_rng(3)
    y_true = rng.binomial(1, 0.25, size=4000)
    assert lift_at_k(y_true, rng.random(4000)) == pytest.approx(1.0, abs=0.35)
    informative = y_true + rng.normal(0, 0.3, size=4000)
    assert lift_at_k(y_true, informative) > 2.0


def test_perfect_and_random_rankers_bracket_the_auc_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    assert compute_metrics(y_true, y_true.astype(float), 0.5).roc_auc == 1.0
    assert compute_metrics(y_true, 1 - y_true.astype(float), 0.5).roc_auc == 0.0
