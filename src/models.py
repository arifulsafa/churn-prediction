"""Candidate models and their search spaces.

Four candidates, chosen to span the complexity range rather than to win a
leaderboard:

| Candidate            | Why it is here                                             |
|----------------------|------------------------------------------------------------|
| `majority`           | Sanity floor. 73% accuracy while catching zero churners -   |
|                      | the single best argument against reporting accuracy.        |
| `recency_rule`       | The heuristic the business already uses ("no order in 60    |
|                      | days = at risk"). A model that cannot beat this is not      |
|                      | worth deploying.                                            |
| `logistic_regression`| Interpretable, calibrated, fast, monotone coefficients that |
|                      | a stakeholder can read. Cannot express interactions.        |
| `gradient_boosting`  | Handles non-linearities and interactions (e.g. high support |
|                      | volume only matters for low-frequency buyers), tolerates    |
|                      | skewed features. Costs interpretability and needs           |
|                      | regularisation to avoid overfitting a 12k-row dataset.      |
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.features import build_pipeline, build_rule_pipeline


class RecencyRuleClassifier(ClassifierMixin, BaseEstimator):
    """The incumbent business rule, expressed as an sklearn estimator.

    "Flag anyone who has not ordered in `threshold_days`." Scores are mapped to a
    pseudo-probability so it can be compared on ROC-AUC/PR-AUC alongside the
    learned models. This is the bar to beat, not a strawman: recency alone is a
    genuinely strong churn signal.
    """

    def __init__(self, threshold_days: float = 60.0):
        self.threshold_days = threshold_days

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _recency(self, X) -> np.ndarray:
        return np.asarray(X["days_since_last_order"], dtype=float)

    def predict_proba(self, X) -> np.ndarray:
        recency = np.nan_to_num(self._recency(X), nan=365.0)
        # Logistic ramp centred on the threshold so ranking metrics are meaningful.
        p = 1.0 / (1.0 + np.exp(-(recency - self.threshold_days) / 20.0))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def candidate_models() -> dict[str, dict]:
    """name -> {pipeline, param_grid, family}."""
    return {
        "majority_baseline": {
            "pipeline": build_pipeline(DummyClassifier(strategy="prior"), scale=False),
            "param_grid": {},
            "family": "baseline",
        },
        "recency_rule_baseline": {
            # Operates on the cleaned frame directly - no encoding needed.
            "pipeline": build_rule_pipeline(RecencyRuleClassifier()),
            "param_grid": {"model__threshold_days": [45, 60, 90]},
            "family": "baseline",
        },
        "logistic_regression": {
            "pipeline": build_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=config.RANDOM_SEED,
                ),
                scale=True,
            ),
            "param_grid": {
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                # sklearn >=1.8 expresses the penalty as l1_ratio: 0.0 = L2, 1.0 = L1.
                "model__l1_ratio": [0.0, 1.0],
            },
            "family": "linear",
        },
        "gradient_boosting": {
            "pipeline": build_pipeline(
                HistGradientBoostingClassifier(
                    random_state=config.RANDOM_SEED,
                    early_stopping=False,
                ),
                scale=False,
            ),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20, 50],
                "model__l2_regularization": [0.0, 1.0],
                "model__max_iter": [200, 400],
            },
            "family": "tree_ensemble",
        },
    }
