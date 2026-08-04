"""Pipeline, model and leakage-guard tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src import config
from src.features import build_pipeline, get_feature_names
from src.models import RecencyRuleClassifier, candidate_models
from src.train import select_model


def _fit_logistic(prepared):
    X, y = prepared
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0))
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def test_pipeline_emits_no_nans_to_the_estimator(prepared):
    """Trees tolerate NaN; logistic regression silently fails on it. Assert it never gets one."""
    X, y = prepared
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)
    encoded = pipe[:-1].transform(X)
    assert not np.isnan(np.asarray(encoded, dtype=float)).any()


def test_pipeline_beats_the_majority_baseline(prepared):
    pipe, X_test, y_test = _fit_logistic(prepared)
    model_ap = average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])

    X, y = prepared
    dummy = build_pipeline(DummyClassifier(strategy="prior"), scale=False).fit(X, y)
    dummy_ap = average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1])

    assert model_ap > dummy_ap + 0.15, f"model AP {model_ap:.3f} vs baseline {dummy_ap:.3f}"


def test_unseen_category_does_not_crash_the_pipeline(prepared):
    """A new country appearing in production must degrade, not 500."""
    pipe, X_test, _ = _fit_logistic(prepared)
    novel = X_test.head(1).copy()
    novel.loc[:, "country"] = "Antarctica"
    novel.loc[:, "subscription_type"] = "platinum_unlaunched_tier"
    probability = pipe.predict_proba(novel)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_missing_optional_fields_are_imputed_not_rejected(prepared):
    pipe, X_test, _ = _fit_logistic(prepared)
    sparse = X_test.head(1).copy()
    for col in ("age", "support_tickets", "discount_order_share", "country"):
        sparse[col] = np.nan
    assert 0.0 <= pipe.predict_proba(sparse)[0, 1] <= 1.0


def test_average_risk_increases_with_recency(prepared):
    """Direction sanity check on the population, not on one row.

    Recency enters the model through three correlated columns (raw, log, and
    recency-relative-to-habit), so an individual customer's curve can wobble;
    the population average must still rise with dormancy or the model is wrong
    in a way no aggregate metric would reveal.
    """
    pipe, X_test, _ = _fit_logistic(prepared)
    cohort = X_test.head(200)
    means = []
    for days in (5, 30, 90, 240):
        counterfactual = cohort.copy()
        counterfactual.loc[:, "days_since_last_order"] = days
        means.append(float(pipe.predict_proba(counterfactual)[:, 1].mean()))
    assert means == sorted(means), f"non-monotone in recency: {means}"


def test_preprocessing_is_fitted_inside_cross_validation_folds(prepared):
    """The leakage guard.

    If the imputer were fitted on the whole dataset, its median would be
    identical across folds. Fitting on two disjoint halves must give different
    statistics - proof that fold-local fitting is actually happening.
    """
    X, y = prepared
    half = len(X) // 2
    medians = []
    for subset in (X.iloc[:half], X.iloc[half:]):
        pipe = build_pipeline(LogisticRegression(max_iter=200)).fit(subset, y.loc[subset.index])
        imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
        medians.append(imputer.statistics_)
    assert not np.allclose(medians[0], medians[1])


def test_feature_names_match_the_encoded_matrix_width(prepared):
    pipe, _, _ = _fit_logistic(prepared)
    X, _ = prepared
    assert len(get_feature_names(pipe)) == pipe[:-1].transform(X.head(5)).shape[1]


def test_recency_rule_baseline_ranks_dormant_customers_higher():
    rule = RecencyRuleClassifier(threshold_days=60).fit(None)
    frame = pd.DataFrame({"days_since_last_order": [1.0, 30.0, 120.0, np.nan]})
    probs = rule.predict_proba(frame)[:, 1]
    assert probs[0] < probs[1] < probs[2]
    assert probs[3] > 0.9, "unknown recency should be treated as long-dormant, not as a fresh buyer"


def test_all_candidate_models_are_fittable(prepared):
    X, y = prepared
    small_X, small_y = X.head(400), y.head(400)
    for name, spec in candidate_models().items():
        spec["pipeline"].fit(small_X, small_y)
        probs = spec["pipeline"].predict_proba(small_X)[:, 1]
        assert probs.shape == (len(small_X),), name
        assert ((probs >= 0) & (probs <= 1)).all(), name


def test_one_standard_error_rule_prefers_the_simpler_tied_model():
    results = {
        "logistic_regression": {"cv_pr_auc": 0.830, "cv_pr_auc_std": 0.012, "family": "linear"},
        "gradient_boosting": {"cv_pr_auc": 0.835, "cv_pr_auc_std": 0.011, "family": "tree_ensemble"},
    }
    assert select_model(results) == "logistic_regression"


def test_one_standard_error_rule_still_picks_a_clearly_better_model():
    results = {
        "logistic_regression": {"cv_pr_auc": 0.700, "cv_pr_auc_std": 0.010, "family": "linear"},
        "gradient_boosting": {"cv_pr_auc": 0.850, "cv_pr_auc_std": 0.010, "family": "tree_ensemble"},
    }
    assert select_model(results) == "gradient_boosting"


def test_config_feature_lists_do_not_overlap_with_leaky_columns():
    declared = set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    assert declared.isdisjoint(config.LEAKY_COLUMNS)
    assert config.TARGET not in declared
