"""
Anomaly Detector
================
Z-score based demand spike and drop detection per product category.
Anomalies are stored as knowledge docs for RAG ingestion.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def detect_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect demand anomalies using Z-score per product category.
    Returns DataFrame of anomalous rows with type (spike/drop) and zscore.
    """
    if df.empty:
        return pd.DataFrame()

    monthly = (
        df.groupby(["Product_Category", df["Date"].dt.to_period("M")])["Order_Demand"]
        .sum()
        .reset_index()
    )
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    results = []
    for cat, group in monthly.groupby("Product_Category"):
        if len(group) < 6:
            continue
        mean = group["Order_Demand"].mean()
        std = group["Order_Demand"].std()
        if std == 0:
            continue
        group = group.copy()
        group["zscore"] = (group["Order_Demand"] - mean) / std
        anomalous = group[group["zscore"].abs() > threshold].copy()
        anomalous["type"] = anomalous["zscore"].apply(lambda z: "spike" if z > 0 else "drop")
        results.append(anomalous)

    if not results:
        return pd.DataFrame()

    all_anomalies = pd.concat(results).sort_values("zscore", key=abs, ascending=False)
    _export_anomaly_report(all_anomalies)
    return all_anomalies.reset_index(drop=True)


def _export_anomaly_report(anomalies: pd.DataFrame):
    """Write anomaly summary as a markdown knowledge doc for RAG ingestion."""
    docs_dir = Path("knowledge_docs")
    docs_dir.mkdir(exist_ok=True)

    lines = ["# Anomaly Detection Report\n"]
    for _, row in anomalies.iterrows():
        lines.append(
            f"- **{row['Product_Category']}** ({row['Date'].strftime('%b %Y')}): "
            f"{row['type'].upper()} — demand {int(row['Order_Demand']):,} units, "
            f"Z-score {row['zscore']:.2f}"
        )

    (docs_dir / "anomaly_report.md").write_text("\n".join(lines))
