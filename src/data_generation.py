"""Synthetic e-commerce customer generator.

Why simulate instead of sampling a static table of features + a hand-made label?
Because a hand-made label is circular: you decide `churn = days_since_last_order
> 60`, the model rediscovers your rule and every metric looks fantastic. That
proves nothing.

Instead we simulate the *behaviour* and read the label off the future:

1. Every customer has a latent purchase rate `lambda` and a latent dormancy time
   `D` (the moment they quietly stop caring). `D` is driven by covariates -
   support friction, discount dependence, returns, subscription tier, tenure.
2. Orders are drawn as a Poisson process at rate `lambda` while `t < D`.
3. Features are computed from orders BEFORE the snapshot date.
4. The label is computed from orders in the 90 days AFTER the snapshot date.

So `days_since_last_order` is informative but not deterministic - exactly the
irreducible noise a real churn problem has, and the reason the ceiling on ROC-AUC
is well below 1.0.

Realistic data-quality problems are injected on purpose (duplicates, missing
values, sentinel values, unit errors, inconsistent category spellings) so the
preparation step has something real to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

COUNTRY_WEIGHTS = {
    "United Kingdom": 0.34,
    "Germany": 0.18,
    "France": 0.14,
    "Spain": 0.10,
    "Netherlands": 0.09,
    "Ireland": 0.08,
    "Poland": 0.07,
}
SUBSCRIPTION_WEIGHTS = {"free": 0.55, "basic": 0.30, "premium": 0.15}
CHANNEL_WEIGHTS = {"organic": 0.30, "paid_search": 0.26, "social": 0.22, "affiliate": 0.14, "referral": 0.08}
DEVICE_WEIGHTS = {"mobile": 0.62, "desktop": 0.30, "tablet": 0.08}


def _choice(rng: np.random.Generator, weights: dict[str, float], size: int) -> np.ndarray:
    keys = list(weights)
    return rng.choice(keys, size=size, p=np.array([weights[k] for k in keys]))


def _simulate_population(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Latent customer state + covariates, before any orders are drawn."""
    tenure_days = rng.integers(30, 365 * 3, size=n)

    country = _choice(rng, COUNTRY_WEIGHTS, n)
    subscription = _choice(rng, SUBSCRIPTION_WEIGHTS, n)
    channel = _choice(rng, CHANNEL_WEIGHTS, n)
    device = _choice(rng, DEVICE_WEIGHTS, n)

    age = np.clip(rng.normal(38, 12, size=n), 18, 88).round().astype(int)

    # Latent monthly purchase rate. Premium subscribers buy more often; older
    # cohorts buy slightly more per month.
    tier_rate_mult = pd.Series(subscription).map({"free": 0.55, "basic": 1.0, "premium": 1.9}).to_numpy()
    lam_monthly = rng.lognormal(mean=0.20, sigma=0.60, size=n) * tier_rate_mult
    lam_daily = lam_monthly / 30.0

    # Basket size: premium and desktop shoppers spend more per order.
    device_mult = pd.Series(device).map({"mobile": 0.92, "desktop": 1.16, "tablet": 1.0}).to_numpy()
    basket_mu = np.log(48) + 0.35 * (subscription == "premium") + 0.12 * (subscription == "basic")
    basket_mean = np.exp(basket_mu + rng.normal(0, 0.30, size=n)) * device_mult

    # Behavioural friction signals.
    support_rate = rng.gamma(shape=0.8, scale=0.9, size=n)  # tickets per 180d
    discount_share = np.clip(rng.beta(2.0, 5.0, size=n), 0, 1)
    return_propensity = np.clip(rng.beta(1.4, 12.0, size=n), 0, 1)
    newsletter_affinity = np.clip(rng.beta(2.2, 3.0, size=n), 0, 1)

    # --- Dormancy hazard -----------------------------------------------------
    # Log-hazard of "quietly stopping". These coefficients ARE the ground truth
    # the model has to recover; they are deliberately non-obvious combinations.
    log_hazard = (
        -8.40
        + 0.42 * support_rate
        + 1.05 * discount_share
        + 2.10 * return_propensity
        - 0.85 * newsletter_affinity
        - 0.55 * np.log1p(lam_monthly)
        - 0.30 * (subscription == "premium")
        + 0.25 * (subscription == "free")
        + 0.22 * (channel == "paid_search")
        + 0.30 * (channel == "social")
        - 0.20 * (channel == "referral")
        - 0.0035 * (age - 38)
        + rng.normal(0, 0.45, size=n)  # unobserved heterogeneity -> noise floor
    )
    daily_hazard = np.exp(log_hazard)
    # Time (days after signup) at which the customer becomes dormant.
    dormancy_day = rng.exponential(1.0 / daily_hazard)

    return pd.DataFrame(
        {
            "tenure_days": tenure_days,
            "age": age,
            "country": country,
            "subscription_type": subscription,
            "acquisition_channel": channel,
            "primary_device": device,
            "lam_daily": lam_daily,
            "basket_mean": basket_mean,
            "support_rate": support_rate,
            "discount_share": discount_share,
            "return_propensity": return_propensity,
            "newsletter_affinity": newsletter_affinity,
            "dormancy_day": dormancy_day,
        }
    )


def _simulate_orders(pop: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw order counts before/after the snapshot from the latent process."""
    n = len(pop)
    tenure = pop["tenure_days"].to_numpy()
    lam = pop["lam_daily"].to_numpy()
    dormancy = pop["dormancy_day"].to_numpy()

    # Active days inside each window (a customer only buys while t < dormancy).
    active_before = np.clip(np.minimum(tenure, dormancy), 0, None)
    active_last_90 = np.clip(np.minimum(tenure, dormancy) - np.maximum(tenure - 90, 0), 0, None)
    active_after = np.clip(np.minimum(tenure + config.OUTCOME_WINDOW_DAYS, dormancy) - tenure, 0, None)

    orders_total = rng.poisson(lam * active_before)
    orders_last_90d = np.minimum(rng.poisson(lam * active_last_90), orders_total)
    orders_after = rng.poisson(lam * active_after)

    # Recency: time of last order before the snapshot. Approximated by the
    # uniform-order-times property of a Poisson process on the active span.
    days_since_last_order = np.full(n, np.nan)
    has_orders = orders_total > 0
    # max of k uniforms on [0, active_before] -> Beta(k, 1) scaled.
    u_max = rng.beta(np.maximum(orders_total[has_orders], 1), 1.0)
    last_order_day = u_max * active_before[has_orders]
    days_since_last_order[has_orders] = tenure[has_orders] - last_order_day
    days_since_last_order = np.where(has_orders, np.maximum(days_since_last_order, 0), np.nan)

    basket = pop["basket_mean"].to_numpy()
    aov = np.where(has_orders, basket * rng.lognormal(0, 0.18, size=n), np.nan)
    order_value_std = np.where(
        orders_total > 1, aov * np.clip(rng.normal(0.32, 0.10, size=n), 0.05, None), np.where(has_orders, 0.0, np.nan)
    )

    support_tickets_last_180d = rng.poisson(pop["support_rate"].to_numpy())
    support_tickets = support_tickets_last_180d + rng.poisson(
        pop["support_rate"].to_numpy() * np.clip(tenure - 180, 0, None) / 180.0
    )
    returns = rng.binomial(orders_total, pop["return_propensity"].to_numpy())
    newsletter_opens = rng.poisson(pop["newsletter_affinity"].to_numpy() * 9.0)
    discount_order_share = np.where(
        has_orders, np.clip(rng.normal(pop["discount_share"].to_numpy(), 0.06), 0, 1), np.nan
    )

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n + 1)],
            "age": pop["age"].to_numpy(),
            "tenure_months": np.round(tenure / 30.44, 1),
            "orders": orders_total,
            "orders_last_90d": orders_last_90d,
            "average_order_value": np.round(aov, 2),
            "order_value_std": np.round(order_value_std, 2),
            "days_since_last_order": np.round(days_since_last_order, 0),
            "support_tickets": support_tickets,
            "support_tickets_last_180d": support_tickets_last_180d,
            "returns": returns,
            "discount_order_share": np.round(discount_order_share, 3),
            "newsletter_opens_last_90d": newsletter_opens,
            "subscription_type": pop["subscription_type"].to_numpy(),
            "country": pop["country"].to_numpy(),
            "acquisition_channel": pop["acquisition_channel"].to_numpy(),
            "primary_device": pop["primary_device"].to_numpy(),
            "snapshot_date": config.SNAPSHOT_DATE,
            "orders_in_outcome_window": orders_after,
            config.TARGET: (orders_after == 0).astype(int),
        }
    )


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make the extract look like a real CRM export rather than a clean matrix."""
    df = df.copy()
    n = len(df)

    # 1. Inconsistent free-text country spellings from multiple source systems.
    spelling_map = {"United Kingdom": ["UK", "uk", "United Kingdom ", "GB"], "Germany": ["DE", "germany"]}
    for canonical, variants in spelling_map.items():
        idx = df.index[df["country"] == canonical]
        pick = rng.random(len(idx)) < 0.30
        chosen = rng.choice(variants, size=int(pick.sum()))
        df.loc[idx[pick], "country"] = chosen

    # 2. Mixed casing / stray whitespace on the subscription tier.
    idx = rng.choice(n, size=int(0.08 * n), replace=False)
    df.loc[df.index[idx], "subscription_type"] = df.loc[df.index[idx], "subscription_type"].str.upper()

    # 3. Age is self-reported: missing at ~7%, plus impossible values.
    idx = rng.choice(n, size=int(0.07 * n), replace=False)
    df.loc[df.index[idx], "age"] = np.nan
    n_bad_age = max(2, round(0.002 * n))
    idx = rng.choice(n, size=n_bad_age, replace=False)
    df.loc[df.index[idx], "age"] = rng.choice([0, 3, 129, 214], size=n_bad_age)

    # 4. Sentinel value for "never ordered" instead of NULL (classic legacy ETL).
    never = df["orders"] == 0
    df.loc[never, "days_since_last_order"] = -1

    # 5. Unit errors: a subset of AOV rows exported in cents.
    candidates = df.index[df["average_order_value"].notna()]
    idx = rng.choice(candidates, size=max(2, round(0.003 * len(candidates))), replace=False)
    df.loc[idx, "average_order_value"] = df.loc[idx, "average_order_value"] * 100

    # 6. Support-ticket column occasionally NULL rather than 0.
    candidates = df.index[df["support_tickets"] == 0]
    idx = rng.choice(candidates, size=max(2, round(0.08 * len(candidates))), replace=False)
    df.loc[idx, "support_tickets"] = np.nan

    # 7. Duplicated rows from a re-run of the export job.
    dupes = df.sample(frac=0.025, random_state=config.RANDOM_SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    # 8. Leaky column: CRM status read at export time, i.e. AFTER the outcome
    #    window closed. Highly correlated with the label, useless at scoring time.
    cancelled = (df[config.TARGET] == 1) & (rng.random(len(df)) < 0.72)
    df["subscription_cancelled_at_export"] = cancelled.astype(int)

    return df.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)


def generate_dataset(n_customers: int = 12_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pop = _simulate_population(n_customers, rng)
    df = _simulate_orders(pop, rng)
    return _inject_data_quality_issues(df, rng)


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    rate = df[config.TARGET].mean()
    print(f"Wrote {len(df):,} rows -> {config.RAW_DATA_PATH}")
    print(f"Churn rate: {rate:.1%}  |  columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
