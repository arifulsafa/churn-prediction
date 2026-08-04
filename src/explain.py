"""Global and per-customer explanations.

Three complementary views, because each answers a different question:

* **Permutation importance** - "which features does the deployed model actually
  rely on?" Model-agnostic, measured on held-out data, in units of PR-AUC lost.
* **Logistic coefficients / odds ratios** - "which direction, and how strongly?"
  Read straight off the linear model, easy to put in front of a stakeholder.
* **SHAP** - "why this particular customer?" Needed for the retention agent who
  has to open a conversation with a reason, not a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.features import get_feature_names

# Human-readable labels for the stakeholder-facing table.
FRIENDLY_NAMES = {
    "recency_vs_habit": "Silence relative to their own normal buying rhythm",
    "log_days_since_last_order": "Days since last order (log scale)",
    "days_since_last_order": "Days since last order",
    "orders_per_month": "Order frequency",
    "recent_order_share": "Share of lifetime orders placed in the last 90 days",
    "orders_last_90d": "Orders in the last 90 days",
    "return_rate": "Proportion of orders returned",
    "support_tickets_per_order": "Support tickets per order",
    "support_tickets_last_180d": "Recent support tickets",
    "discount_order_share": "Reliance on discounts",
    "newsletter_opens_last_90d": "Recent newsletter engagement",
    "estimated_lifetime_value": "Lifetime spend to date",
    "average_order_value": "Average order value",
    "tenure_months": "Account tenure",
    "never_ordered": "Never placed an order",
    "subscription_type_premium": "Premium subscriber",
    "subscription_type_free": "Free-tier account",
    "subscription_type_basic": "Basic-tier account",
    "orders": "Lifetime orders",
    "returns": "Lifetime returns",
    "support_tickets": "Lifetime support tickets",
    "order_value_std": "Variability of order values",
    "order_value_cv": "Order-value consistency",
    "expected_order_gap_days": "Typical gap between orders",
    "support_recency_share": "Share of support tickets raised recently",
    "age": "Age",
    "country": "Country",
    "acquisition_channel": "How the customer was acquired",
    "primary_device": "Main shopping device",
    "missingindicator_age": "Age not provided",
    "missingindicator_average_order_value": "No order value on file",
    "missingindicator_support_tickets": "Support history missing",
    "missingindicator_days_since_last_order": "No recorded last order",
}


def friendly(name: str) -> str:
    return FRIENDLY_NAMES.get(name, name.replace("_", " "))


def permutation_report(pipeline, X, y, n_repeats: int = 10, top_n: int = 15) -> pd.DataFrame:
    """Permutation importance over the RAW input columns.

    Deliberately permuted on the raw frame rather than the encoded matrix: that
    way one row of the report equals one thing the business can act on, instead
    of a one-hot fragment. Scored with average precision, matching the metric
    used for model selection.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "label": [friendly(c) for c in X.columns],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)


def coefficient_report(pipeline, top_n: int = 20) -> pd.DataFrame | None:
    """Odds ratios from a fitted logistic-regression pipeline (None otherwise).

    Numeric features are standardised inside the pipeline, so each odds ratio is
    the multiplicative effect of a **one-standard-deviation** change in that
    feature - not of one unit. Reporting these as per-unit effects is a common
    and badly misleading mistake.
    """
    model = pipeline.named_steps.get("model")
    if not hasattr(model, "coef_"):
        return None
    names = get_feature_names(pipeline)
    coefs = model.coef_.ravel()
    df = pd.DataFrame(
        {
            "feature": names,
            "label": [friendly(n) for n in names],
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).head(top_n).drop(columns="abs_coefficient")


def shap_report(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_display: int = 15):
    """Mean |SHAP| per encoded feature for a tree-based pipeline.

    Returns (DataFrame, shap_values) or (None, None) if SHAP is unavailable or
    the estimator is not tree-based.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover - optional dependency
        return None, None

    model = pipeline.named_steps.get("model")
    pre = pipeline[:-1]
    try:
        explainer = shap.TreeExplainer(model)
        encoded = pre.transform(X_explain)
        values = explainer.shap_values(encoded)
    except Exception:  # pragma: no cover - non-tree model or version mismatch
        return None, None

    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    names = get_feature_names(pipeline)
    df = pd.DataFrame(
        {
            "feature": names,
            "label": [friendly(n) for n in names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).head(max_display).reset_index(drop=True)
    return df, values


def _linear_contributions(pipeline, row: pd.DataFrame) -> np.ndarray | None:
    """coefficient x standardised feature value = that customer's log-odds push.

    For a linear model this is exact, not an approximation: the contributions sum
    to `logit(p) - intercept`. No SHAP dependency needed.
    """
    model = pipeline.named_steps.get("model")
    if not hasattr(model, "coef_"):
        return None
    encoded = np.asarray(pipeline[:-1].transform(row)).reshape(-1)
    return model.coef_.ravel() * encoded


def top_reasons_for_customer(pipeline, row: pd.DataFrame, k: int = 3) -> list[dict]:
    """Per-customer drivers, returned by the API alongside the probability.

    A retention agent needs a reason to open a conversation, not a bare score.
    Uses exact linear contributions for the logistic model and TreeSHAP for the
    boosted model; returns an empty list rather than raising if neither applies.
    """
    values = _linear_contributions(pipeline, row)

    if values is None:
        try:
            import shap

            encoded = pipeline[:-1].transform(row)
            explainer = shap.TreeExplainer(pipeline.named_steps["model"])
            values = np.asarray(explainer.shap_values(encoded))
            if values.ndim == 3:
                values = values[:, :, -1]
            values = values.reshape(-1)
        except Exception:  # pragma: no cover - explainer unavailable
            return []

    names = get_feature_names(pipeline)
    if len(names) != len(values):  # pragma: no cover - defensive
        return []
    order = np.argsort(-np.abs(values))[:k]
    return [
        {
            "feature": names[i],
            "label": friendly(names[i]),
            "direction": "increases_risk" if values[i] > 0 else "decreases_risk",
            "contribution": round(float(values[i]), 4),
        }
        for i in order
    ]
