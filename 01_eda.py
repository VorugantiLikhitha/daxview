"""
01_eda.py — EDA & Preprocessing
Run: python 01_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import calendar
from pathlib import Path

sns.set_theme(style='darkgrid', palette='viridis')
Path('outputs/charts').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('knowledge_docs').mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/raw/historical_product_demand.csv')
print(f"Shape: {df.shape}")
print(df.dtypes)

# ── Clean ─────────────────────────────────────────────────────────────────────
print("\nCleaning data...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Order_Demand'] = (
    df['Order_Demand']
    .astype(str)
    .str.replace(r'[()\s]', '', regex=True)
    .pipe(pd.to_numeric, errors='coerce')
)
df = df.dropna(subset=['Date', 'Order_Demand'])
df = df[df['Order_Demand'] > 0]
df['Order_Demand'] = df['Order_Demand'].abs()
print(f"Cleaned shape: {df.shape}")
print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")

# ── EDA Plots ─────────────────────────────────────────────────────────────────
print("\nGenerating EDA plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

monthly = df.groupby(df['Date'].dt.to_period('M'))['Order_Demand'].sum()
monthly.index = monthly.index.to_timestamp()
monthly.plot(ax=axes[0], title='Monthly Demand Trend', color='#00ff87')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Total Demand')

top_cats = df.groupby('Product_Category')['Order_Demand'].sum().sort_values(ascending=False).head(10)
top_cats.plot(kind='barh', ax=axes[1], title='Top 10 Categories', color='#ff6b35')
axes[1].set_xlabel('Total Demand')

plt.tight_layout()
plt.savefig('outputs/charts/eda_overview.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/charts/eda_overview.png")
plt.show()

# ── Save processed CSV ────────────────────────────────────────────────────────
df.to_csv('data/processed/demand_clean.csv', index=False)
print("Saved: data/processed/demand_clean.csv")

# ── Write to SQLite ───────────────────────────────────────────────────────────
conn = sqlite3.connect('data/processed/demand.db')
df.to_sql('demand', conn, if_exists='replace', index=False)
conn.close()
print("Saved: data/processed/demand.db")

# ── Generate RAG Knowledge Doc ────────────────────────────────────────────────
print("\nGenerating knowledge doc...")
top5 = top_cats.head(5)
date_range = f"{df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}"
total_units = int(df['Order_Demand'].sum())
monthly_avg = df.groupby(df['Date'].dt.month)['Order_Demand'].mean()
peak_month = monthly_avg.idxmax()
peak_month_name = calendar.month_name[peak_month]

doc = f"""# EDA Summary — Daxwell Demand Dataset

## Dataset Overview
- Date range: {date_range}
- Total demand units: {total_units:,}
- Unique products (SKUs): {df['Product_Code'].nunique()}
- Warehouses: {df['Warehouse'].nunique()}
- Product categories: {df['Product_Category'].nunique()}

## Top Product Categories by Total Demand
{chr(10).join([f'- {cat}: {int(val):,} units' for cat, val in top5.items()])}

## Seasonality
- Peak demand month: {peak_month_name}
- Monthly demand range: {int(monthly_avg.min()):,} to {int(monthly_avg.max()):,} avg units/month

## Warehouse Distribution
{df.groupby('Warehouse')['Order_Demand'].sum().sort_values(ascending=False).to_string()}
"""

Path('knowledge_docs/eda_summary.md').write_text(doc)
print("Saved: knowledge_docs/eda_summary.md")
print("\n✅ EDA complete! Run 03_forecasting.py next.")
