"""
03_forecasting.py — Demand Forecasting with Prophet
Run: python 03_forecasting.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path

Path('outputs/charts').mkdir(parents=True, exist_ok=True)
Path('knowledge_docs').mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading processed data...")
df = pd.read_csv('data/processed/demand_clean.csv', parse_dates=['Date'])

# Aggregate weekly for Prophet
weekly = df.groupby(df['Date'].dt.to_period('W'))['Order_Demand'].sum().reset_index()
weekly['Date'] = weekly['Date'].dt.to_timestamp()
weekly = weekly.rename(columns={'Date': 'ds', 'Order_Demand': 'y'})
print(f"Training on {len(weekly)} weekly observations")

# ── Train Prophet ─────────────────────────────────────────────────────────────
print("\nTraining Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(weekly)

future = model.make_future_dataframe(periods=26, freq='W')
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    'outputs/forecast_results.csv', index=False
)
print("Saved: outputs/forecast_results.csv")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\nGenerating forecast plots...")
fig = model.plot(forecast, figsize=(14, 5))
plt.title('Daxwell Demand Forecast — 26 Week Horizon')
plt.tight_layout()
plt.savefig('outputs/charts/forecast.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/charts/forecast.png")
plt.show()

fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.savefig('outputs/charts/forecast_components.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/charts/forecast_components.png")
plt.show()

# ── Generate RAG Knowledge Doc ────────────────────────────────────────────────
print("\nGenerating knowledge doc...")
future_only = forecast[forecast['ds'] > weekly['ds'].max()]
avg_forecast = int(future_only['yhat'].mean())
peak_week = future_only.loc[future_only['yhat'].idxmax()]
low_week = future_only.loc[future_only['yhat'].idxmin()]

doc = f"""# Demand Forecast Report — 26-Week Horizon

## Model
- Algorithm: Facebook Prophet with yearly seasonality
- Training data: {len(weekly)} weekly observations
- Forecast horizon: 26 weeks ahead

## Forecast Summary
- Average forecasted weekly demand: {avg_forecast:,} units
- Peak week: {peak_week['ds'].strftime('%b %d, %Y')} — {int(peak_week['yhat']):,} units
- Trough week: {low_week['ds'].strftime('%b %d, %Y')} — {int(low_week['yhat']):,} units
- Confidence band width (avg): {int((future_only['yhat_upper'] - future_only['yhat_lower']).mean()):,} units

## Procurement Implication
Procurement teams should align orders 4-6 weeks ahead of peak week ({peak_week['ds'].strftime('%b %d, %Y')}).
Safety stock should be sized to cover the upper confidence interval of {int(peak_week['yhat_upper']):,} units.
"""

Path('knowledge_docs/forecast_report.md').write_text(doc)
print("Saved: knowledge_docs/forecast_report.md")
print("\n✅ Forecasting complete! Run: streamlit run app.py")
