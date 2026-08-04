"""Pipeline construction.

Everything - cleaning, feature engineering, imputation, scaling, encoding and the
estimator - is wrapped in a single `Pipeline`. That matters for two reasons:

1. **No leakage.** Imputer medians, scaler means and encoder categories are
   fitted inside each cross-validation fold, never on the full dataset.
2. **No train/serve skew.** The API loads one object and calls `predict_proba`;
   there is no separate serving-side preprocessing code that can drift.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data_prep import FeatureEngineer, RowCleaner

MODEL_NUMERIC_FEATURES = config.NUMERIC_FEATURES + FeatureEngineer.engineered_features
MODEL_CATEGORICAL_FEATURES = config.CATEGORICAL_FEATURES


def build_preprocessor(scale: bool = True, extra_numeric: list[str] | None = None) -> ColumnTransformer:
    """Impute + (optionally) scale numerics, impute + one-hot encode categoricals.

    * Median imputation for numerics: robust to the long right tails on order
      value and recency. `add_indicator=True` keeps "this was missing" as its own
      signal, which matters because age is missing non-randomly.
    * `handle_unknown="ignore"` on the encoder so an unseen country from the API
      degrades to an all-zero block instead of raising.
    * Scaling is on for logistic regression (needs it for the L2 penalty to be
      meaningful) and off for trees (they are invariant to monotone rescaling).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ]
    )

    numeric_columns = MODEL_NUMERIC_FEATURES + list(extra_numeric or [])
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_columns),
            ("cat", categorical, MODEL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale: bool = True, extra_numeric: list[str] | None = None) -> Pipeline:
    """`extra_numeric` is only used by the leakage demonstration in train.py."""
    return Pipeline(
        [
            ("clean", RowCleaner()),
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor(scale=scale, extra_numeric=extra_numeric)),
            ("model", estimator),
        ]
    )


def build_rule_pipeline(estimator) -> Pipeline:
    """Cleaning + engineering only, no encoding.

    Used by the heuristic baseline, which reads one named column and would be
    broken by the ColumnTransformer turning the frame into a positional array.
    """
    return Pipeline([("clean", RowCleaner()), ("engineer", FeatureEngineer()), ("model", estimator)])


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
