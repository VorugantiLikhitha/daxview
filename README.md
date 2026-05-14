# 📦 DaxView — Demand Intelligence Platform

> **Daxwell processes millions of single-use disposable units across healthcare, food service, and industrial verticals. A 5% demand forecasting error at that scale means stockouts, lost contracts, and emergency procurement costs.**
>
> DaxView replaces gut-feel procurement with a live AI analyst — combining statistical forecasting, anomaly detection, and a RAG chatbot that answers supply chain questions in plain English.

---

## 💰 Business Impact

| Problem | DaxView Solution | Estimated Value |
|---|---|---|
| Reactive procurement (stockouts) | 26-week Prophet forecast per category | Avoid $200K–$1M emergency order premiums |
| Manual anomaly review | Z-score spike/drop detection across all SKUs | Save 10–15 analyst hours/week |
| Data locked in spreadsheets | Natural language SQL agent | Answers in seconds vs. hours |
| Siloed institutional knowledge | RAG over EDA + forecast + anomaly docs | Retained and queryable anytime |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                        │
│                                                             │
│  Raw CSV (1M+ rows)                                         │
│       │                                                     │
│       ▼                                                     │
│  pandas cleaning ──► SQLite DB ──► ChromaDB (embeddings)   │
│       │                                                     │
│       ▼                                                     │
│  Prophet Forecast ──► forecast_results.csv                  │
│       │                                                     │
│       ▼                                                     │
│  Z-score Anomaly Detector ──► anomaly_report.md             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        RAG ENGINE                           │
│                                                             │
│   User Question                                             │
│        ├──► ChromaDB vector search (local embeddings)       │
│        ├──► SQL Agent (Gemini generates + runs SQLite SQL)  │
│        └──► Gemini 2.0 Flash synthesizes both sources       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DAXVIEW APP                             │
│                                                             │
│  📊 Overview      Metrics + trend + category breakdown      │
│  📈 Forecasts     26-week Prophet + confidence intervals    │
│  🔍 Anomalies     Z-score spikes/drops + KS validation      │
│  🤖 AI Analyst    RAG chatbot + voice interface             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Statistical Validation

Forecast and anomaly outputs are validated using **Kolmogorov-Smirnov tests**.

| Validation | Method | Threshold |
|---|---|---|
| Forecast accuracy | MAE + MAPE per category | < 15% MAPE target |
| Anomaly confidence | Z-score with p-value | Z > 2.5 (p < 0.01) |
| Demand distribution | KS-test per feature | p > 0.05 = valid |

```bash
python validate.py
```

---

## 🎙️ Voice Interface

Ask questions out loud — DaxView hears you and speaks the answer back. Built with browser-native Web Speech API — no extra API key needed.

---

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/daxview.git
cd daxview
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add dataset (kaggle.com/datasets/felixzhao/productdemandforecasting)
mv "Historical Product Demand.csv" data/raw/historical_product_demand.csv

echo "GEMINI_API_KEY=your_key_here" > .env  # aistudio.google.com

python 01_eda.py
python 03_forecasting.py
python validate.py
streamlit run app.py
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| App Framework | Streamlit |
| Forecasting | Prophet (Meta) |
| Anomaly Detection | Z-score + KS-test validation |
| Vector DB | ChromaDB (local embeddings) |
| LLM / SQL Agent | Google Gemini 2.0 Flash |
| Voice Interface | Web Speech API + SpeechSynthesis |
| Data | pandas, SQLite, Plotly |

---

*Built for Daxwell — a single-use disposables company serving healthcare, food service, and industrial customers.*
