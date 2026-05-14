import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="DaxView — Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* Hide Streamlit chrome */
#MainMenu, header, footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b !important; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    width: 100% !important;
    text-align: left !important;
    display: flex !important;
    justify-content: flex-start !important;
    margin: 1px 0 !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Sidebar text input (Daxie) */
[data-testid="stSidebar"] input {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input::placeholder { color: #64748b !important; }
[data-testid="stSidebar"] label { color: #64748b !important; font-size: 11px !important; }

/* Main background */
[data-testid="stAppViewContainer"] { background: #f1f5f9 !important; }
[data-testid="stMain"] { background: #f1f5f9 !important; }
[data-testid="block-container"] { padding: 28px 40px !important; max-width: 1400px !important; margin: 0 auto !important; }

/* Cards */
.card { background: #fff; border-radius: 14px; padding: 20px 24px; border: 1px solid #e2e8f0; margin-bottom: 16px; }
.metric-card { background: #fff; border-radius: 14px; padding: 20px 24px; border: 1px solid #e2e8f0; }
.metric-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 10px; }
.metric-val { font-size: 26px; font-weight: 700; color: #0f172a; margin: 4px 0 2px; }
.metric-lbl { font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-delta-up { font-size: 12px; color: #10b981; font-weight: 600; margin-top: 2px; }
.metric-delta-down { font-size: 12px; color: #ef4444; font-weight: 600; margin-top: 2px; }

/* Page header */
.page-title { font-size: 22px; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
.page-sub { font-size: 13px; color: #64748b; margin-bottom: 20px; }

/* Chat bubbles */
.chat-user {
    background: #0f172a; color: #f1f5f9 !important;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px; margin: 6px 0 6px 10%;
    font-size: 14px; line-height: 1.6;
}
.chat-ai {
    background: #fff; color: #1e293b !important;
    border: 1px solid #e2e8f0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px; margin: 6px 10% 6px 0;
    font-size: 14px; line-height: 1.6; white-space: pre-wrap;
}
.source-tag { display: inline-block; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 2px 8px; font-size: 11px; color: #64748b; margin: 2px; }
.badge-groq { background: #f0fdf4 !important; border-color: #bbf7d0 !important; color: #166534 !important; }
.badge-gemini { background: #eff6ff !important; border-color: #bfdbfe !important; color: #1e40af !important; }

/* Anomaly rows */
.anomaly-spike { background: #fff1f2; border-left: 4px solid #f43f5e; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 4px 0; font-size: 13px; color: #881337; }
.anomaly-drop  { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 4px 0; font-size: 13px; color: #1e3a8a; }

/* Chat input — force white */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] > div { background: #ffffff !important; background-color: #ffffff !important; }
[data-testid="stChatInput"] textarea { color: #1e293b !important; background: #ffffff !important; caret-color: #0f172a !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #94a3b8 !important; }
[data-testid="stChatInput"] { border: 1.5px solid #e2e8f0 !important; border-radius: 12px !important; background: #ffffff !important; }

h1, h2, h3 { color: #0f172a !important; }
p, li { color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
# ── Top navbar ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0f172a; padding:0 32px; display:flex; align-items:center;
     justify-content:space-between; height:56px; position:sticky; top:0; z-index:999;
     margin:-28px -40px 24px; box-shadow:0 1px 3px rgba(0,0,0,0.2);'>
  <div style='font-size:16px; font-weight:700; color:#f1f5f9; letter-spacing:-0.3px;'>📦 DaxView</div>
</div>
""", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1,1,1,2,2])
with nav_col1:
    if st.button("🏠 Overview", use_container_width=True, key="btn_ov"):
        st.session_state.page = "overview"; st.rerun()
with nav_col2:
    if st.button("📈 Forecasts", use_container_width=True, key="btn_fc"):
        st.session_state.page = "forecasts"; st.rerun()
with nav_col3:
    if st.button("🤖 AI Analyst", use_container_width=True, key="btn_ai"):
        st.session_state.page = "ai"; st.rerun()
with nav_col4:
    daxie_q = st.text_input("", placeholder="🤖 Ask Daxie anything...",
                             label_visibility="collapsed", key="daxie_input")
    if daxie_q and daxie_q != st.session_state.get("_last_daxie"):
        st.session_state["_last_daxie"] = daxie_q
        st.session_state.daxie_pending = daxie_q
        st.session_state.page = "ai"
        st.rerun()
with nav_col5:
    active = {"overview":"🏠 Overview","forecasts":"📈 Forecasts","ai":"🤖 AI Analyst"}
    st.markdown(f"<div style='text-align:right; color:#64748b; font-size:13px; padding-top:8px;'>Current: <strong>{active.get(st.session_state.page,'Overview')}</strong></div>", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    p = Path("data/processed/demand_clean.csv")
    return pd.read_csv(p, parse_dates=["Date"]) if p.exists() else pd.DataFrame()

df = load_data()

# ── RAG ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_rag():
    from rag_engine import DaxViewRAG
    return DaxViewRAG()

# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if st.session_state.page == "overview":
    st.markdown("<div class='page-title'>Demand Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Single-use disposables · Healthcare · Food Service · Industrial</div>", unsafe_allow_html=True)

    if df.empty:
        st.info("Run `python 01_eda.py` first to generate processed data.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, bg, lbl, val, delta, dcls in [
        (c1, "📦", "#eff6ff", "TOTAL UNITS",   f"{int(df['Order_Demand'].sum()):,.0f}", "↑ 12.3% YoY",          "metric-delta-up"),
        (c2, "🏷️", "#f0fdf4", "ACTIVE SKUs",   str(df['Product_Code'].nunique()),       "across all categories", "metric-delta-up"),
        (c3, "🏭", "#fef9c3", "WAREHOUSES",    str(df['Warehouse'].nunique()),           "distribution points",   "metric-delta-up"),
        (c4, "📊", "#fdf4ff", "CATEGORIES",    str(df['Product_Category'].nunique()),    "product lines",         "metric-delta-up"),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-icon' style='background:{bg};'>{icon}</div>
            <div class='metric-lbl'>{lbl}</div>
            <div class='metric-val'>{val}</div>
            <div class='{dcls}'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("**Monthly Demand Trend**")
        monthly = df.groupby(df["Date"].dt.to_period("M"))["Order_Demand"].sum()
        monthly.index = monthly.index.to_timestamp()
        fig = go.Figure(go.Scatter(x=monthly.index, y=monthly.values,
            line=dict(color="#3b82f6", width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"))
        fig.update_layout(height=220, margin=dict(l=60,r=16,t=8,b=40),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(size=11,color="#64748b")),
            yaxis=dict(gridcolor="#f1f5f9", tickfont=dict(size=11,color="#64748b"), tickformat=".2s"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_r:
        st.markdown("**Top Categories**")
        top = df.groupby("Product_Category")["Order_Demand"].sum().sort_values(ascending=True).tail(8)
        fig2 = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
            marker_color="#6366f1",
            text=[f"{v/1e6:.1f}M" for v in top.values], textposition="outside",
            textfont=dict(size=10, color="#64748b")))
        fig2.update_layout(height=220, margin=dict(l=90,r=55,t=8,b=20),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(tickfont=dict(size=10, color="#374151")), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Demand by Warehouse**")
        wh = df.groupby("Warehouse")["Order_Demand"].sum().sort_values(ascending=False)
        fig3 = go.Figure(go.Bar(x=wh.index, y=wh.values,
            marker_color=["#3b82f6","#6366f1","#8b5cf6","#a78bfa"][:len(wh)],
            text=[f"{v/1e6:.0f}M" for v in wh.values], textposition="outside",
            textfont=dict(size=11, color="#64748b")))
        fig3.update_layout(height=200, margin=dict(l=16,r=16,t=8,b=30),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(tickfont=dict(size=12,color="#374151")),
            yaxis=dict(showticklabels=False, showgrid=False), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Seasonality — Avg by Month**")
        seas = df.groupby(df["Date"].dt.month)["Order_Demand"].mean()
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig4 = go.Figure(go.Bar(x=months, y=seas.values, marker_color="#10b981",
            text=[f"{v/1e3:.0f}K" for v in seas.values], textposition="outside",
            textfont=dict(size=11, color="#64748b")))
        fig4.update_layout(height=200, margin=dict(l=16,r=16,t=8,b=30),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(tickfont=dict(size=11,color="#374151")),
            yaxis=dict(showticklabels=False, showgrid=False), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Anomaly snapshot
    from anomaly_detector import detect_anomalies
    anomalies = detect_anomalies(df)
    spikes = len(anomalies[anomalies["type"]=="spike"]) if not anomalies.empty else 0
    drops  = len(anomalies[anomalies["type"]=="drop"])  if not anomalies.empty else 0
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**⚠️ Anomaly Snapshot**")
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Total Anomalies", len(anomalies))
    ac2.metric("🔴 Spikes", spikes)
    ac3.metric("🔵 Drops",  drops)
    if not anomalies.empty:
        top_a = anomalies.iloc[0]
        st.markdown(f"**Highest risk:** `{top_a['Product_Category']}` — {top_a['Date'].strftime('%b %Y')} | Z-score: `{top_a['zscore']:.2f}`")
    st.markdown("</div>", unsafe_allow_html=True)

# ── PAGE: FORECASTS ───────────────────────────────────────────────────────────
elif st.session_state.page == "forecasts":
    st.markdown("<div class='page-title'>Demand Forecasts</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Prophet model · 26-week horizon · Confidence intervals</div>", unsafe_allow_html=True)

    fp = Path("outputs/forecast_results.csv")
    if not fp.exists():
        st.info("Run `python 03_forecasting.py` to generate forecasts.")
        st.stop()

    fcast  = pd.read_csv(fp, parse_dates=["ds"])
    future = fcast.tail(26).copy()
    avg_f  = int(future["yhat"].mean())
    peak   = future.loc[future["yhat"].idxmax()]
    trough = future.loc[future["yhat"].idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    for col, icon, bg, lbl, val, delta, dcls in [
        (m1,"📈","#eff6ff","AVG WEEKLY",     f"{avg_f:,.0f}",              "units / week",                      "metric-delta-up"),
        (m2,"🔝","#f0fdf4","PEAK WEEK",      f"{int(peak['yhat']):,.0f}",  peak['ds'].strftime('%b %d, %Y'),    "metric-delta-up"),
        (m3,"📉","#fff1f2","TROUGH WEEK",    f"{int(trough['yhat']):,.0f}",trough['ds'].strftime('%b %d, %Y'), "metric-delta-down"),
        (m4,"📊","#fef9c3","CONFIDENCE BAND",f"±{int((future['yhat_upper']-future['yhat_lower']).mean()//2):,.0f}","avg uncertainty","metric-delta-up"),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-icon' style='background:{bg};'>{icon}</div>
            <div class='metric-lbl'>{lbl}</div>
            <div class='metric-val'>{val}</div>
            <div class='{dcls}'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Full Forecast Timeline**")
    fig = go.Figure([
        go.Scatter(x=fcast["ds"], y=fcast["yhat_upper"], fill=None,
                   line=dict(color="rgba(59,130,246,0)"), showlegend=False),
        go.Scatter(x=fcast["ds"], y=fcast["yhat_lower"], fill="tonexty",
                   fillcolor="rgba(59,130,246,0.08)", line=dict(color="rgba(59,130,246,0)"),
                   name="Confidence Band"),
        go.Scatter(x=fcast["ds"], y=fcast["yhat"], line=dict(color="#3b82f6", width=2),
                   name="Forecast",
                   hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>"),
        go.Scatter(x=future["ds"], y=future["yhat"],
                   line=dict(color="#f59e0b", width=2.5, dash="dot"), name="Projected Horizon"),
    ])
    fig.update_layout(height=320, margin=dict(l=60,r=20,t=8,b=40),
        plot_bgcolor="#fff", paper_bgcolor="#fff", font_color="#1e293b",
        xaxis=dict(gridcolor="#f8fafc", tickfont=dict(size=11,color="#64748b")),
        yaxis=dict(gridcolor="#f8fafc", tickfont=dict(size=11,color="#64748b"), tickformat=".2s"),
        legend=dict(bgcolor="#fff", bordercolor="#e2e8f0", orientation="h", y=1.1),
        hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📋 26-Week Procurement Schedule**")
        tbl = future[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        tbl["Risk"] = tbl["yhat"].apply(lambda v: "🔴 High" if v > avg_f*1.1 else ("🟡 Med" if v > avg_f*0.9 else "🟢 Low"))
        tbl = tbl.rename(columns={"ds":"Week","yhat":"Forecast","yhat_lower":"Low","yhat_upper":"High"})
        tbl["Week"] = tbl["Week"].dt.strftime("%b %d")
        st.dataframe(tbl.style.format({"Forecast":"{:,.0f}","Low":"{:,.0f}","High":"{:,.0f}"}),
                     use_container_width=True, hide_index=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📊 Weekly Risk Chart**")
        colors = ["#ef4444" if v > avg_f*1.1 else ("#f59e0b" if v > avg_f*0.9 else "#10b981") for v in future["yhat"]]
        fig2 = go.Figure(go.Bar(
            x=future["ds"].dt.strftime("%b %d"), y=future["yhat"],
            marker_color=colors,
            error_y=dict(type="data", symmetric=False,
                array=future["yhat_upper"]-future["yhat"],
                arrayminus=future["yhat"]-future["yhat_lower"],
                color="#94a3b8", thickness=1.5),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"))
        fig2.add_hline(y=avg_f, line_dash="dot", line_color="#64748b",
                       annotation_text=f"Avg {avg_f:,.0f}", annotation_position="right")
        fig2.update_layout(height=320, margin=dict(l=50,r=70,t=8,b=60),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(tickfont=dict(size=10,color="#64748b"), tickangle=45),
            yaxis=dict(gridcolor="#f8fafc", tickfont=dict(size=10,color="#64748b"), tickformat=".2s"),
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("<div style='font-size:11px;color:#64748b;'>🔴 High &nbsp; 🟡 Normal &nbsp; 🟢 Low</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── PAGE: AI ANALYST ──────────────────────────────────────────────────────────
elif st.session_state.page == "ai":
    st.markdown("<div class='page-title'>🤖 Daxie — AI Demand Analyst</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Ask anything about your demand data — RAG + SQL Agent + Voice</div>", unsafe_allow_html=True)

    try:
        rag = get_rag()
    except Exception as e:
        st.error(f"⚠️ RAG engine error: {e}")
        st.stop()

    # Process pending Daxie question — only once
    if st.session_state.get("daxie_pending"):
        q = st.session_state.pop("daxie_pending")
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Daxie is thinking..."):
            resp = rag.query(q, df)
        st.session_state.messages.append({
            "role": "assistant", "content": resp["answer"],
            "sources": resp.get("sources", []), "provider": resp.get("provider", "")
        })

    # Voice + quick chips
    st.components.v1.html("""
    <style>
    body{margin:0;font-family:'Inter',sans-serif;}
    .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:4px 0 12px;}
    .chip{background:#f8fafc;border:1px solid #e2e8f0;border-radius:20px;padding:7px 14px;
          font-size:13px;color:#334155;cursor:pointer;transition:all 0.15s;}
    .chip:hover{background:#0f172a;color:#f1f5f9;border-color:#0f172a;}
    .mic{width:38px;height:38px;border-radius:50%;border:none;background:#0f172a;
         color:white;font-size:16px;cursor:pointer;flex-shrink:0;}
    .mic.on{background:#ef4444;animation:pulse 1s infinite;}
    @keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.1)}}
    .status{font-size:12px;color:#64748b;}
    </style>
    <div class="row">
      <button class="mic" id="mic" onclick="toggle()" title="Click to speak">🎙️</button>
      <span class="status" id="st"></span>
      <button class="chip" onclick="send('Which category has highest Q4 demand?')">📈 Highest Q4 demand?</button>
      <button class="chip" onclick="send('What are the top stockout risk warehouses?')">⚠️ Stockout risks?</button>
      <button class="chip" onclick="send('Summarize the demand trend for 2016')">📅 2016 trend</button>
      <button class="chip" onclick="send('Which SKUs have highest demand variance?')">📊 High variance SKUs</button>
    </div>
    <script>
    let rec, on=false;
    function send(t){
      const ta=Array.from(window.parent.document.querySelectorAll('textarea'))
                     .find(e=>e.placeholder&&e.placeholder.includes('Ask'));
      if(!ta)return;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(ta,t);
      ta.dispatchEvent(new Event('input',{bubbles:true}));
      setTimeout(()=>ta.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',keyCode:13,bubbles:true})),200);
    }
    function speak(t){
      if(!window.speechSynthesis)return;
      speechSynthesis.cancel();
      const u=new SpeechSynthesisUtterance(t);u.rate=1;u.lang='en-US';
      const v=speechSynthesis.getVoices().find(v=>v.name.includes('Samantha')||v.name.includes('Google US'));
      if(v)u.voice=v; speechSynthesis.speak(u);
    }
    function toggle(){
      if(!('webkitSpeechRecognition'in window||'SpeechRecognition'in window)){
        document.getElementById('st').textContent='⚠️ Use Chrome';return;}
      if(on){rec.stop();return;}
      const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
      rec=new SR();rec.lang='en-US';rec.interimResults=false;
      rec.onstart=()=>{on=true;document.getElementById('mic').className='mic on';
        document.getElementById('mic').textContent='⏹️';document.getElementById('st').textContent='Listening...';};
      rec.onresult=e=>{const t=e.results[0][0].transcript;
        document.getElementById('st').textContent='🗣️ '+t;send(t);};
      rec.onerror=e=>{document.getElementById('st').textContent='⚠️ '+e.error;};
      rec.onend=()=>{on=false;document.getElementById('mic').className='mic';
        document.getElementById('mic').textContent='🎙️';};
      rec.start();
    }
    // Auto-speak new AI responses
    let last='';
    setInterval(()=>{
      const msgs=window.parent.document.querySelectorAll('.chat-ai');
      if(msgs.length){const t=msgs[msgs.length-1].textContent.trim();
        if(t&&t!==last&&window._voiceActive){last=t;speak(t);}}
    },1000);
    </script>
    """, height=70)

    # Chat history
    st.markdown("<div class='card' style='min-height:280px;'>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown("<div style='text-align:center;padding:60px;color:#94a3b8;font-size:14px;'>Ask Daxie a question above or click 🎙️ to speak</div>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        css = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            provider = msg.get("provider","")
            pb = f'<span class="source-tag badge-groq">⚡ Groq</span>' if "groq" in provider else (f'<span class="source-tag badge-gemini">✦ Gemini</span>' if "gemini" in provider else "")
            srcs = "".join([f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"] if not s.startswith("llm:")])
            st.markdown(f"<div style='margin:4px 0 10px;'>{pb}{srcs}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about demand, forecasts, anomalies, categories...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            resp = rag.query(user_input, df)
        st.session_state.messages.append({
            "role": "assistant", "content": resp["answer"],
            "sources": resp.get("sources", []), "provider": resp.get("provider","")
        })
        st.rerun()
