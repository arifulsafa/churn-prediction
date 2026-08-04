# CLAUDE.md — churn-prediction

Working notes for Claude Code (and for me) on this repository. Read before changing anything.

## What this is

Take-home submission: an end-to-end customer-churn ML solution — synthetic data generation, EDA,
leakage-safe training, model comparison, cost-based thresholding, explainability, a FastAPI service,
tests and a production architecture. Judged on **reasoning and engineering judgement, not on model
score**. Every non-obvious decision must be defensible out loud in a follow-up interview.

## Commands

```bash
source .venv/bin/activate         # or use .venv/bin/<tool> directly
python -m src.data_generation     # regenerate data/customers_raw.csv
python -m src.eda                 # regenerate reports/eda_report.md + eda_overview.png
python -m src.train               # full training run, writes artifacts/ + reports/  (~30s)
python -m pytest                  # 59 tests, ~2s
uvicorn src.api:app --reload      # API on :8000, docs at /docs
python docs/make_architecture_diagram.py
make all                          # data -> eda -> train -> test
```

## Architecture in one paragraph

`src/config.py` holds every constant (paths, churn definition, split policy, cost assumptions,
feature contract, leaky-column list). `data_generation.py` simulates behaviour and injects defects.
`data_prep.py` splits work into **dataset-level** (`prepare_dataset`: de-dupe, drop leaks) and
**row-level** sklearn transformers (`RowCleaner`, `FeatureEngineer`) that live *inside* the saved
pipeline. `features.py` assembles clean → engineer → impute/scale/encode → estimator.
`train.py` orchestrates split → tune → select → threshold → test → persist. `api.py` loads the one
serialised pipeline; there is no serving-side feature code.

## Invariants — do not break these

1. **The test set is touched exactly once**, in `train.main`, after every decision is frozen.
   Model selection and threshold choice both use out-of-fold predictions on TRAIN.
2. **All preprocessing lives inside the `Pipeline`.** Never fit an imputer/scaler/encoder outside
   cross-validation. `test_preprocessing_is_fitted_inside_cross_validation_folds` guards this.
3. **`config.LEAKY_COLUMNS` stays dropped.** `subscription_cancelled_at_export` is written after the
   outcome window closes; keeping it inflates CV PR-AUC 0.839 → 0.958 and is worthless in
   production. `train.leakage_demonstration` exists to show that on purpose — do not "fix" it by
   deleting the column from the generator.
4. **`RowCleaner` must stay row-independent** (no fitted state, no batch statistics), or the API and
   the training job diverge. `test_row_cleaner_is_row_independent` guards this.
5. **De-duplicate before splitting**, never after.
6. **Sklearn mixin order is `(TransformerMixin, BaseEstimator)` / `(ClassifierMixin, BaseEstimator)`.**
   Reversed, sklearn ≥1.9 loses the estimator type and scorers pass a 2-column array to
   `average_precision`, failing with "y should be a 1d array".
7. **Numeric features are standardised**, so logistic coefficients are per **standard deviation**,
   not per unit. Any write-up that says otherwise is wrong.

## Decisions already made (with the reasoning) — don't silently reverse them

- **Churn = no order in the 90 days after the 2026-01-01 snapshot.** Observable, forward-looking,
  actionable. Known weakness: mislabels genuinely slow-cadence customers — which is *why*
  `recency_vs_habit` exists.
- **Simulated behaviour, not a rule-defined label.** A rule-defined label makes any model look
  perfect and demonstrates nothing.
- **Logistic regression selected over gradient boosting** by the one-standard-error rule
  (0.8387 vs 0.8407, fold std 0.013 → statistically tied, take the simpler one). This is a
  tie-breaker, not a blanket preference; `test_one_standard_error_rule_still_picks_a_clearly_better_model`
  pins the other direction.
- **PR-AUC is the selection metric; recall is the business priority.** A missed churner costs ~15x a
  false alarm, hence a 0.43 threshold rather than 0.5. Accuracy is reported and never used to decide
  anything (the majority baseline gets 72.9%).
- **`log_days_since_last_order` was removed** after an ablation: 0.8391 → 0.8387 (nothing), and it
  took a negative coefficient that made recency read as risk-*reducing*. Do not re-add it.
- **No SMOTE / resampling.** 27% positives is not extreme; `class_weight="balanced"` plus a
  cost-based threshold keeps calibration intact, which SMOTE would damage.
- **API: only `tenure_months` and `orders` are required.** The brief's 5-field example must keep
  working verbatim; everything else is imputed and reported in `imputed_fields`.

## Conventions

- Comments explain **why**, never what the line does. If a comment restates the code, delete it.
- Test names are sentences describing the property being asserted
  (`test_sentinel_recency_is_never_treated_as_recent`), so `pytest -v` reads as a specification.
- Numbers quoted in README.md / notebook / docs come from `reports/evaluation.json`. **After any
  retrain, re-check them** — stale figures are the easiest way to look careless in an interview.
- British spelling in prose; `-` not `—` inside code and docstrings.

## Regeneration order (matters)

`data_generation` → `eda` → `train` → re-execute the notebook → sync numbers in `README.md`.
Changing the generator invalidates every downstream number, including the defect counts in the
README data-quality table.

## Current results (test set, 2,400 customers, scored once)

PR-AUC 0.850 · ROC-AUC 0.912 · precision 0.601 · recall 0.851 · F1 0.704 · Brier 0.117 ·
lift@10% 3.66x · threshold 0.43 · confusion matrix TN 1381 / FP 368 / FN 97 / TP 554.
Baselines: majority PR-AUC 0.271, recency rule PR-AUC 0.795.

## Open items (deliberately not done — scope was 3–4 hours)

Temporal validation across multiple snapshots; randomised holdout → uplift modelling; survival
model; Dockerfile + CI; MLflow registry integration; API auth and rate limiting; fairness audit
across country and tenure bands.
