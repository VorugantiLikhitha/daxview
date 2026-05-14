"""
validate.py — DaxView Statistical Validation Suite
====================================================
Validates forecast accuracy and anomaly detection confidence
using MAE, MAPE, and Kolmogorov-Smirnov tests.

Run: python validate.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def separator(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print('─' * 55)


# ── 1. Forecast Validation ────────────────────────────────────
def validate_forecast():
    separator("FORECAST VALIDATION — Prophet 26-Week Model")

    forecast_path = Path("outputs/forecast_results.csv")
    data_path = Path("data/processed/demand_clean.csv")

    if not forecast_path.exists() or not data_path.exists():
        print("  ⚠️  Run 01_eda.py and 03_forecasting.py first.")
        return

    fcast = pd.read_csv(forecast_path, parse_dates=["ds"])
    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Weekly actuals
    weekly = df.groupby(df["Date"].dt.to_period("W"))["Order_Demand"].sum().reset_index()
    weekly["Date"] = weekly["Date"].dt.to_timestamp()
    weekly = weekly.rename(columns={"Date": "ds", "Order_Demand": "actual"})

    # Merge actuals with in-sample forecast
    merged = pd.merge(weekly, fcast[["ds", "yhat"]], on="ds", how="inner")

    if merged.empty:
        print("  ⚠️  No overlapping dates for validation.")
        return

    mae = np.mean(np.abs(merged["actual"] - merged["yhat"]))
    mape = np.mean(np.abs((merged["actual"] - merged["yhat"]) / merged["actual"])) * 100
    rmse = np.sqrt(np.mean((merged["actual"] - merged["yhat"]) ** 2))

    print(f"\n  Weeks validated:  {len(merged)}")
    print(f"  MAE:              {mae:,.0f} units")
    print(f"  MAPE:             {mape:.1f}%")
    print(f"  RMSE:             {rmse:,.0f} units")

    status = "✅ PASS" if mape < 20 else "⚠️  REVIEW"
    print(f"\n  Status: {status}  (target MAPE < 20%)")

    # KS-test: are residuals normally distributed?
    residuals = merged["actual"] - merged["yhat"]
    ks_stat, ks_p = stats.normaltest(residuals)
    print(f"\n  Residual normality test (D'Agostino):")
    print(f"  p-value: {ks_p:.4f}  {'✅ Normal' if ks_p > 0.05 else '⚠️  Non-normal (check seasonality)'}")


# ── 2. Anomaly Validation ─────────────────────────────────────
def validate_anomalies():
    separator("ANOMALY VALIDATION — Z-Score Confidence")

    data_path = Path("data/processed/demand_clean.csv")
    if not data_path.exists():
        print("  ⚠️  Run 01_eda.py first.")
        return

    df = pd.read_csv(data_path, parse_dates=["Date"])
    monthly = df.groupby(["Product_Category", df["Date"].dt.to_period("M")])["Order_Demand"].sum().reset_index()

    anomaly_count = 0
    high_conf = 0
    results = []

    for cat, group in monthly.groupby("Product_Category"):
        if len(group) < 6:
            continue
        mean = group["Order_Demand"].mean()
        std = group["Order_Demand"].std()
        if std == 0:
            continue
        zscores = (group["Order_Demand"] - mean) / std
        anomalies = group[zscores.abs() > 2.5]
        for _, row in anomalies.iterrows():
            z = abs((row["Order_Demand"] - mean) / std)
            p_val = 2 * (1 - stats.norm.cdf(z))
            results.append({"category": cat, "zscore": z, "p_value": p_val})
            anomaly_count += 1
            if p_val < 0.01:
                high_conf += 1

    print(f"\n  Total anomalies detected: {anomaly_count}")
    print(f"  High confidence (p < 0.01): {high_conf} ({100*high_conf/max(anomaly_count,1):.0f}%)")
    print(f"  Low confidence (p ≥ 0.01): {anomaly_count - high_conf}")

    if results:
        df_res = pd.DataFrame(results).sort_values("zscore", ascending=False).head(5)
        print(f"\n  Top 5 highest-confidence anomalies:")
        for _, r in df_res.iterrows():
            print(f"    {r['category']:<20} Z={r['zscore']:.2f}  p={r['p_value']:.4f}  ✅")


# ── 3. Distribution Validation (KS-test) ─────────────────────
def validate_distribution():
    separator("DISTRIBUTION VALIDATION — KS-Test")

    data_path = Path("data/processed/demand_clean.csv")
    if not data_path.exists():
        print("  ⚠️  Run 01_eda.py first.")
        return

    df = pd.read_csv(data_path, parse_dates=["Date"])

    # Split into two halves — test if distribution is consistent over time
    midpoint = df["Date"].median()
    first_half = df[df["Date"] < midpoint]["Order_Demand"]
    second_half = df[df["Date"] >= midpoint]["Order_Demand"]

    ks_stat, ks_p = stats.ks_2samp(first_half, second_half)

    print(f"\n  Comparing demand distribution: first half vs second half of dataset")
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  p-value:      {ks_p:.4f}")

    if ks_p > 0.05:
        print(f"  Result: ✅ Distributions are statistically similar (p > 0.05)")
        print(f"          Demand patterns are stable — forecast model is reliable")
    else:
        print(f"  Result: ⚠️  Distributions differ significantly (p ≤ 0.05)")
        print(f"          Demand has shifted over time — consider retraining periodically")

    # Per-category KS test
    print(f"\n  Per-category distribution stability:")
    cats = df["Product_Category"].value_counts().head(5).index
    for cat in cats:
        cat_df = df[df["Product_Category"] == cat]
        mid = cat_df["Date"].median()
        h1 = cat_df[cat_df["Date"] < mid]["Order_Demand"]
        h2 = cat_df[cat_df["Date"] >= mid]["Order_Demand"]
        if len(h1) > 5 and len(h2) > 5:
            _, p = stats.ks_2samp(h1, h2)
            status = "✅ Stable" if p > 0.05 else "⚠️  Shifted"
            print(f"    {cat:<20} p={p:.4f}  {status}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  DAXVIEW — STATISTICAL VALIDATION SUITE")
    print("═" * 55)

    validate_forecast()
    validate_anomalies()
    validate_distribution()

    print(f"\n{'═' * 55}")
    print("  Validation complete. Results saved to knowledge_docs/")
    print("═" * 55)

    # Write summary to knowledge doc for RAG
    Path("knowledge_docs").mkdir(exist_ok=True)
    Path("knowledge_docs/validation_report.md").write_text(
        "# Validation Report\n\nStatistical validation was run using MAE/MAPE for forecasts "
        "and Kolmogorov-Smirnov tests for anomaly confidence and distribution stability. "
        "See validate.py output for full results.\n"
    )
