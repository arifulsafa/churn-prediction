"""End-to-end training: split -> tune -> select -> threshold -> test -> persist.

Validation strategy, and why it is arranged this way:

    raw extract
      |-- de-duplicate, drop leaky columns
      |-- stratified 80/20 split  ->  TEST (sealed until the very last step)
            |
            +-- TRAIN (80%)
                  |-- 5-fold stratified CV: hyperparameter search per candidate
                  |-- 5-fold out-of-fold predictions: model selection (PR-AUC)
                  |-- same out-of-fold predictions: decision-threshold choice
                  |-- refit best config on all of TRAIN
      |-- score TEST exactly once, report

The test set is touched by exactly one line of this file, after every choice has
already been made. Preprocessing is inside the pipeline, so fold statistics never
see the validation fold.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from src import config, evaluate, explain
from src.data_prep import load_raw, prepare_dataset
from src.models import candidate_models


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.VALIDATION_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


def tune_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Grid-search every candidate on the training folds only."""
    results, leaderboard = {}, []

    for name, spec in candidate_models().items():
        started = time.perf_counter()
        search = GridSearchCV(
            spec["pipeline"],
            spec["param_grid"],
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            refit="pr_auc",
            cv=_cv(),
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        idx = search.best_index_
        cvres = search.cv_results_

        results[name] = {
            "estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_pr_auc": float(cvres["mean_test_pr_auc"][idx]),
            "cv_pr_auc_std": float(cvres["std_test_pr_auc"][idx]),
            "cv_roc_auc": float(cvres["mean_test_roc_auc"][idx]),
            "family": spec["family"],
        }
        leaderboard.append(
            {
                "model": name,
                "family": spec["family"],
                "cv_pr_auc": results[name]["cv_pr_auc"],
                "cv_pr_auc_std": results[name]["cv_pr_auc_std"],
                "cv_roc_auc": results[name]["cv_roc_auc"],
                "fit_seconds": round(time.perf_counter() - started, 1),
                "best_params": json.dumps(search.best_params_),
            }
        )
        print(
            f"  {name:<24} PR-AUC {results[name]['cv_pr_auc']:.4f} "
            f"(+/-{results[name]['cv_pr_auc_std']:.4f})  ROC-AUC {results[name]['cv_roc_auc']:.4f}"
        )

    board = pd.DataFrame(leaderboard).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    return results, board


def select_model(results: dict) -> str:
    """One-standard-error rule: prefer the simplest model statistically tied with the best.

    A gradient booster that beats logistic regression by less than one CV
    standard error is not actually better - it is noise, and it costs
    interpretability, latency and operational surface area. Complexity has to
    earn its place.
    """
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["cv_pr_auc"])
    best_name, best = ranked[0]
    cutoff = best["cv_pr_auc"] - best["cv_pr_auc_std"]

    simplicity = {"baseline": 0, "linear": 1, "tree_ensemble": 2}
    tied = [(n, r) for n, r in results.items() if r["cv_pr_auc"] >= cutoff and r["family"] != "baseline"]
    if not tied:
        return best_name
    tied.sort(key=lambda kv: (simplicity[kv[1]["family"]], -kv[1]["cv_pr_auc"]))
    chosen = tied[0][0]
    if chosen != best_name:
        print(
            f"  1-SE rule: '{chosen}' is within one standard error of '{best_name}' "
            f"and is simpler -> selecting '{chosen}'"
        )
    return chosen


def leakage_demonstration(df_raw: pd.DataFrame, X_train, y_train) -> dict:
    """Quantify what the leaky CRM column would have bought us, had we kept it.

    `subscription_cancelled_at_export` is populated when the extract is taken -
    i.e. after the outcome window has already closed. Keeping it produces a model
    that looks excellent offline and is worthless in production, because at
    scoring time the field is always empty for the customers we care about.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from src.features import build_pipeline

    leaky_col = "subscription_cancelled_at_export"
    if leaky_col not in df_raw.columns:
        return {}

    # `prepare_dataset` de-duplicates then resets the index, so X_train.index
    # indexes into the de-duplicated frame - re-derive it the same way.
    deduped = df_raw.drop_duplicates(subset=config.ID_COLUMN, keep="first").reset_index(drop=True)
    X_leaky = X_train.copy()
    X_leaky[leaky_col] = deduped.loc[X_train.index, leaky_col].to_numpy()

    pipe = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        extra_numeric=[leaky_col],
    )
    scores = cross_val_score(pipe, X_leaky, y_train, cv=_cv(), scoring="average_precision")

    return {
        "leaky_column": leaky_col,
        "cv_pr_auc_with_leak": float(scores.mean()),
        "note": (
            "Inflated because the column is written after the outcome window closes. "
            "It is dropped in src/config.LEAKY_COLUMNS."
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--data", default=str(config.RAW_DATA_PATH))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-leakage-demo", action="store_true")
    args = parser.parse_args(argv)

    print("1. Loading and preparing data")
    df_raw = load_raw(args.data)
    X, y = prepare_dataset(df_raw, verbose=True)
    print(f"  {len(X):,} customers, {y.mean():.1%} churn rate")

    print("\n2. Holding out the test set (never used for selection or tuning)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"  train {len(X_train):,} | test {len(X_test):,}")

    print(f"\n3. {config.VALIDATION_FOLDS}-fold CV hyperparameter search on the training set")
    results, leaderboard = tune_candidates(X_train, y_train)

    print("\n4. Model selection")
    best_name = select_model(results)
    best = results[best_name]
    print(f"  selected: {best_name}  params={best['best_params']}")

    print("\n5. Choosing the decision threshold on out-of-fold validation predictions")
    oof_proba = cross_val_predict(best["estimator"], X_train, y_train, cv=_cv(), method="predict_proba", n_jobs=-1)[:, 1]
    threshold, sweep = evaluate.choose_threshold(y_train, oof_proba)
    oof_metrics = evaluate.compute_metrics(y_train, oof_proba, threshold)
    print(
        f"  threshold={threshold:.2f} (min expected cost "
        f"{oof_metrics.expected_cost_per_customer:.2f}/customer)  "
        f"OOF precision={oof_metrics.precision:.3f} recall={oof_metrics.recall:.3f}"
    )

    print("\n6. Refitting on the full training set and scoring the sealed test set")
    final_model = best["estimator"].fit(X_train, y_train)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.compute_metrics(y_test, test_proba, threshold)
    print(
        f"  TEST  PR-AUC={test_metrics.pr_auc:.4f}  ROC-AUC={test_metrics.roc_auc:.4f}  "
        f"precision={test_metrics.precision:.3f}  recall={test_metrics.recall:.3f}  "
        f"F1={test_metrics.f1:.3f}  lift@10%={test_metrics.lift_at_10pct:.2f}x"
    )

    # Baselines on the same test set, for an honest "is this worth deploying?"
    baseline_test = {}
    for name in ("majority_baseline", "recency_rule_baseline"):
        if name in results:
            proba = results[name]["estimator"].fit(X_train, y_train).predict_proba(X_test)[:, 1]
            baseline_test[name] = evaluate.compute_metrics(y_test, proba, 0.5).as_dict()

    print("\n7. Explainability")
    perm = explain.permutation_report(final_model, X_test, y_test)
    print(perm[["label", "importance_mean"]].head(8).to_string(index=False))
    coefs = explain.coefficient_report(final_model)
    shap_df, _ = explain.shap_report(final_model, X_train, X_test.head(1000))

    leakage = {} if args.skip_leakage_demo else leakage_demonstration(df_raw, X_train, y_train)
    if leakage:
        print(
            f"\n8. Leakage check: keeping '{leakage['leaky_column']}' would have shown "
            f"CV PR-AUC {leakage['cv_pr_auc_with_leak']:.4f} vs the honest {best['cv_pr_auc']:.4f}"
        )

    # --- persist ----------------------------------------------------------
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "model_family": best["family"],
        "best_params": {k: str(v) for k, v in best["best_params"].items()},
        "decision_threshold": threshold,
        "risk_bands": {
            "high_risk": threshold,
            "medium_risk": round(threshold * 0.5, 3),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "cv_pr_auc": best["cv_pr_auc"],
        "test_metrics": test_metrics.as_dict(),
        "input_features": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "random_seed": config.RANDOM_SEED,
    }
    evaluate.save_json(metadata, config.METADATA_PATH)

    report = {
        "churn_definition": (
            f"No order in the {config.OUTCOME_WINDOW_DAYS} days following the "
            f"{config.SNAPSHOT_DATE} snapshot."
        ),
        "selected_model": best_name,
        "leaderboard_cv": leaderboard.to_dict(orient="records"),
        "out_of_fold_metrics": oof_metrics.as_dict(),
        "test_metrics": test_metrics.as_dict(),
        "baseline_test_metrics": baseline_test,
        "threshold_sweep": sweep,
        "permutation_importance": perm.to_dict(orient="records"),
        "logistic_coefficients": coefs.to_dict(orient="records") if coefs is not None else None,
        "shap_importance": shap_df.to_dict(orient="records") if shap_df is not None else None,
        "leakage_demonstration": leakage,
    }
    evaluate.save_json(report, config.METRICS_PATH)
    leaderboard.to_csv(config.REPORT_DIR / "model_leaderboard.csv", index=False)

    if not args.no_plots:
        for p in evaluate.make_evaluation_plots(y_test, test_proba, threshold):
            print(f"  wrote {p}")

    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")
    return report


if __name__ == "__main__":
    main()
