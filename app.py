"""
S&P 500 Direction Predictor — Streamlit App
BiLSTM + Temporal Attention | Shivam Tiwari (065104)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0f1420; border-right: 1px solid #1e2d40; }
.metric-card {
    background: linear-gradient(135deg, #0f1e35 0%, #0d1829 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 8px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 600; color: #f1f5f9; line-height: 1; }
.metric-sub { font-size: 12px; color: #475569; margin-top: 6px; }
.direction-up { background: linear-gradient(135deg, #052e16 0%, #064e3b 100%); border: 1px solid #10b981; border-radius: 16px; padding: 28px; text-align: center; }
.direction-down { background: linear-gradient(135deg, #1c0a0a 0%, #2d1515 100%); border: 1px solid #ef4444; border-radius: 16px; padding: 28px; text-align: center; }
.direction-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 10px; }
.direction-arrow { font-size: 56px; line-height: 1; margin: 8px 0; }
.direction-prob { font-family: 'IBM Plex Mono', monospace; font-size: 42px; font-weight: 600; }
.direction-conf { font-size: 13px; margin-top: 8px; opacity: 0.7; }
.section-header { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #3b82f6; margin: 28px 0 16px; padding-bottom: 8px; border-bottom: 1px solid #1e2d40; }
.disclaimer { background: #1c1a0a; border: 1px solid #854d0e; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px 18px; font-size: 12px; color: #a16207; margin-top: 20px; }
.badge-live { display: inline-flex; align-items: center; gap: 6px; background: #052e16; border: 1px solid #10b981; color: #10b981; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 1px; padding: 3px 10px; border-radius: 20px; }
.dot-live { width: 6px; height: 6px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.stButton > button { background: linear-gradient(135deg, #1d4ed8, #1e40af); color: white; border: none; border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; letter-spacing: 1px; padding: 12px 28px; width: 100%; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8); transform: translateY(-1px); box-shadow: 0 4px 20px rgba(59,130,246,0.4); }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color: #94a3b8 !important; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ── DataPipeline stub ─────────────────────────────────────────────────────────
# Defined BEFORE the unpickler so it's in scope everywhere.
# The _SafeUnpickler below intercepts ALL module paths for this class name,
# so it doesn't matter whether the .pkl was saved from __main__, __mp_main__,
# ipykernel_launcher, or any Colab cell context.
class DataPipeline:
    """Minimal stub — matches the Colab DataPipeline attributes pickle needs."""
    def __init__(self, cfg=None):
        self.feature_scaler  = RobustScaler()
        self.feature_columns = []

    def split_and_scale(self, *a, **kw):
        raise NotImplementedError


# Register in every plausible module name pickle might look in
for _mod_name in ("__main__", "__mp_main__", "app", "streamlit_app"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = type(sys)(_mod_name)
    sys.modules[_mod_name].DataPipeline = DataPipeline


class _SafeUnpickler(pickle.Unpickler):
    """Intercepts class lookup: any module + 'DataPipeline' → our stub."""
    def find_class(self, module, name):
        if name == "DataPipeline":
            return DataPipeline
        return super().find_class(module, name)


def safe_load_pickle(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


# ── PyTorch model ─────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.3, output_size=1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = TemporalAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        x           = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        context, w  = self.attention(lstm_out)
        return self.head(context), w


# ── Feature engineering ───────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    tp    = (high + low + close) / 3

    for n in [9, 21, 50, 200]:
        df[f"EMA_{n}"] = close.ewm(span=n, adjust=False).mean()
    df["Price_vs_EMA9"]  = (close / df["EMA_9"])  - 1
    df["Price_vs_EMA21"] = (close / df["EMA_21"]) - 1
    df["Price_vs_EMA50"] = (close / df["EMA_50"]) - 1
    df["EMA9_vs_EMA21"]  = (df["EMA_9"]  / df["EMA_21"]) - 1
    df["EMA21_vs_EMA50"] = (df["EMA_21"] / df["EMA_50"]) - 1

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = (100 - (100 / (1 + gain / (loss + 1e-10)))) / 100

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_norm"]  = (ema12 - ema26) / (close + 1e-10)
    df["MACDs_norm"] = df["MACD_norm"].ewm(span=9, adjust=False).mean()
    df["MACDh_norm"] = df["MACD_norm"] - df["MACDs_norm"]

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["STOCHk"] = (close - low14) / (high14 - low14 + 1e-10)
    df["STOCHd"] = df["STOCHk"].rolling(3).mean()

    df["ROC_5"]    = close.pct_change(5)
    df["ROC_10"]   = close.pct_change(10)
    df["CCI_20"]   = ((tp - tp.rolling(20).mean()) /
                      (0.015 * tp.rolling(20).std() + 1e-10)) / 100
    df["WILLR_14"] = (high14 - close) / (high14 - low14 + 1e-10)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BBP"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    df["BBW"] = (4 * std20) / (sma20 + 1e-10)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14_norm"] = tr.rolling(14).mean() / (close + 1e-10)

    df["VOL_SMA20"]   = vol.rolling(20).mean()
    df["VOL_ratio"]   = vol / (df["VOL_SMA20"] + 1e-10)
    df["OBV"]         = (np.sign(close.diff()) * vol).cumsum()
    df["OBV_norm"]    = df["OBV"] / (df["OBV"].abs().rolling(20).mean() + 1e-10)

    df["Return_1d"]     = close.pct_change(1)
    df["Return_5d"]     = close.pct_change(5)
    df["Return_20d"]    = close.pct_change(20)
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["Volatility_5"]  = df["Return_1d"].rolling(5).std()

    high_52w = close.rolling(252).max()
    low_52w  = close.rolling(252).min()
    df["Dist_52w_high"] = (close - high_52w) / (high_52w + 1e-10)
    df["Dist_52w_low"]  = (close - low_52w)  / (low_52w  + 1e-10)

    df["Day_of_week"] = pd.to_datetime(df.index).dayofweek / 4
    df["Month"]       = pd.to_datetime(df.index).month / 11
    df["Quarter"]     = pd.to_datetime(df.index).quarter / 3

    return df


FEATURE_COLS = [
    "Price_vs_EMA9","Price_vs_EMA21","Price_vs_EMA50",
    "EMA9_vs_EMA21","EMA21_vs_EMA50",
    "RSI_14","MACD_norm","MACDs_norm","MACDh_norm",
    "STOCHk","STOCHd","ROC_5","ROC_10","CCI_20","WILLR_14",
    "BBP","BBW","ATR_14_norm",
    "VOL_ratio","OBV_norm",
    "Return_1d","Return_5d","Return_20d",
    "Volatility_20","Volatility_5",
    "Dist_52w_high","Dist_52w_low",
    "Day_of_week","Month","Quarter",
]


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(period: str = "2y") -> pd.DataFrame:
    last_err = None
    for attempt in range(4):
        try:
            raw = yf.download(
                "^GSPC", period=period,
                auto_adjust=True, progress=False,
                multi_level_index=False,
            )
            if raw.empty:
                raise ValueError("Empty dataframe returned from yfinance.")
            raw.columns = [c.capitalize() for c in raw.columns]
            return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            last_err = e
            is_rate = any(k in str(e) for k in ["Rate", "429", "Too Many"])
            if attempt < 3:
                time.sleep((5 * (attempt + 1)) if is_rate else 2)
    st.warning(
        f"⚠️ Could not fetch live data after 4 attempts: `{last_err}`\n\n"
        "yfinance may be rate-limiting this server — try again in a few minutes."
    )
    return pd.DataFrame()


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    def find(filename):
        for p in [Path(filename), Path("model") / filename]:
            if p.exists():
                return p
        return None

    model_path    = find("best_model.pt")
    pipeline_path = find("pipeline.pkl") or find("pipeline .pkl")
    config_path   = find("config.json")

    missing = [n for n, p in [
        ("best_model.pt", model_path),
        ("pipeline.pkl",  pipeline_path),
        ("config.json",   config_path),
    ] if p is None]

    if missing:
        return None, None, None, f"Missing: {', '.join(missing)}"

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Use our safe unpickler — intercepts DataPipeline from any module
        pipeline = safe_load_pickle(pipeline_path)

        input_size  = config["model"]["input_size"] or len(FEATURE_COLS)
        hidden_size = config["model"]["hidden_size"]
        num_layers  = config["model"]["num_layers"]
        dropout     = config["model"]["dropout"]

        model = LSTMStockPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        return model, pipeline, config, None

    except Exception as e:
        return None, None, None, str(e)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, pipeline, raw_df, seq_len=60):
    feat_df   = add_technical_indicators(raw_df.copy()).dropna()
    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X         = feat_df[available].values

    if len(X) < seq_len:
        return None, None, f"Need ≥{seq_len} rows, got {len(X)}."

    window = X[-seq_len:]
    scaled = pipeline.feature_scaler.transform(window)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    logit, attn_w = model(tensor)
    prob_up = torch.sigmoid(logit).item()
    return prob_up, attn_w.squeeze(0).numpy(), None


def confidence_label(prob):
    diff = abs(prob - 0.5)
    if diff < 0.02: return "Very Low Confidence", "#f59e0b"
    if diff < 0.04: return "Low Confidence",       "#f97316"
    if diff < 0.07: return "Moderate Confidence",  "#3b82f6"
    return                  "High Confidence",      "#10b981"


# ── Charts ────────────────────────────────────────────────────────────────────
def make_price_chart(df, days=120):
    df_plot = df.tail(days).copy()
    df_plot["EMA21"] = df_plot["Close"].ewm(span=21).mean()
    df_plot["EMA50"] = df_plot["Close"].ewm(span=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Close",
        line=dict(color="#3b82f6", width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA21"], name="EMA 21",
        line=dict(color="#f59e0b", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA50"], name="EMA 50",
        line=dict(color="#ec4899", width=1, dash="dot")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,32,0.8)",
        font=dict(family="IBM Plex Mono", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0), height=280, hovermode="x unified",
        xaxis=dict(gridcolor="#1e2d40"), yaxis=dict(gridcolor="#1e2d40", tickprefix="$"),
    )
    return fig


def make_attention_chart(attn, seq_len=60):
    tick_vals = list(range(0, seq_len, 5))
    tick_text = [f"t-{seq_len - i - 1}" for i in tick_vals]
    fig = go.Figure(go.Bar(
        x=list(range(seq_len)), y=attn,
        marker=dict(color=attn,
                    colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]],
                    showscale=False),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,32,0.8)",
        font=dict(family="IBM Plex Mono", size=10, color="#94a3b8"),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_text,
                   gridcolor="#1e2d40", title="Time Step"),
        yaxis=dict(gridcolor="#1e2d40", title="Weight"),
        margin=dict(l=0, r=0, t=10, b=0), height=180,
    )
    return fig


def make_gauge(prob):
    color = "#10b981" if prob >= 0.5 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number=dict(suffix="%", font=dict(family="IBM Plex Mono", size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(family="IBM Plex Mono", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(15,20,32,0.8)", bordercolor="#1e2d40",
            steps=[
                dict(range=[0,  35], color="rgba(239,68,68,0.15)"),
                dict(range=[35, 50], color="rgba(239,68,68,0.08)"),
                dict(range=[50, 65], color="rgba(16,185,129,0.08)"),
                dict(range=[65,100], color="rgba(16,185,129,0.15)"),
            ],
            threshold=dict(line=dict(color="#f8fafc", width=2), thickness=0.8, value=50),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                      margin=dict(l=20, r=20, t=20, b=20), height=220)
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
def main():
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown("""
        <h1 style='font-family:IBM Plex Mono;font-size:26px;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>
            S&P 500 Direction Predictor</h1>
        <p style='font-family:IBM Plex Mono;font-size:12px;color:#475569;letter-spacing:1px;'>
            BiLSTM + Temporal Attention &nbsp;·&nbsp; 44 Technical Indicators
            &nbsp;·&nbsp; 1986–2024 Training Data</p>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown("""
        <div style='text-align:right;padding-top:8px;'>
            <span class='badge-live'><span class='dot-live'></span> LIVE DATA</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    with st.sidebar:
        st.markdown("""<p style='font-family:IBM Plex Mono;font-size:11px;letter-spacing:2px;
            text-transform:uppercase;color:#3b82f6;margin-bottom:16px;'>⚙ Configuration</p>""",
            unsafe_allow_html=True)
        seq_len    = st.slider("Sequence Length (days)", 30, 90, 60, 5)
        chart_days = st.slider("Chart History (days)", 60, 365, 120, 30)
        show_attn  = st.toggle("Show Attention Weights", True)
        show_indic = st.toggle("Show Indicators Panel", True)
        st.divider()
        st.markdown("""
        <p style='font-family:IBM Plex Mono;font-size:10px;color:#334155;letter-spacing:1px;
            text-transform:uppercase;margin-bottom:8px;'>About</p>
        <p style='font-size:12px;color:#475569;line-height:1.6;'>
            Bidirectional LSTM with soft temporal attention.<br>
            Trained on 9,827 daily bars of ^GSPC.<br>
            Outputs P(close₊₁ > close₀).</p>
        <p style='font-size:11px;color:#334155;margin-top:12px;'>Shivam Tiwari · 065104</p>
        """, unsafe_allow_html=True)
        run_btn = st.button("▶  RUN PREDICTION", use_container_width=True)

    model, pipeline, config, err = load_artifacts()

    with st.spinner("Fetching market data..."):
        raw_df = fetch_data("2y")

    if raw_df.empty:
        st.error("Could not fetch S&P 500 data. Please try again later.")
        return

    last_close = float(raw_df["Close"].iloc[-1])
    last_date  = raw_df.index[-1]
    prev_close = float(raw_df["Close"].iloc[-2])
    day_return = (last_close - prev_close) / prev_close * 100

    c1, c2, c3, c4 = st.columns(4)
    status_sub = ("artifacts loaded" if model
                  else (err[:38] + "…" if err and len(err) > 38 else err or "check files"))
    for col, label, value, sub in [
        (c1, "Last Close",   f"{last_close:,.2f}",   last_date.strftime("%b %d, %Y")),
        (c2, "Day Return",   f"{day_return:+.2f}%",  "▲ positive" if day_return >= 0 else "▼ negative"),
        (c3, "Data Points",  f"{len(raw_df):,}",     "trading days loaded"),
        (c4, "Model Status", "READY" if model else "NO MODEL", status_sub),
    ]:
        with col:
            color = ("#10b981" if (label == "Day Return" and day_return >= 0) else
                     "#ef4444" if (label == "Day Return" and day_return <  0) else
                     "#10b981" if (label == "Model Status" and model)         else
                     "#ef4444" if (label == "Model Status" and not model)     else "#f1f5f9")
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{value}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>PRICE HISTORY</div>", unsafe_allow_html=True)
    st.plotly_chart(make_price_chart(raw_df, chart_days), width="stretch")

    if run_btn or st.session_state.get("has_prediction"):
        if model is None:
            st.error(f"⚠️ Model not loaded: {err}\n\n"
                     "Place `best_model.pt`, `pipeline.pkl`, and `config.json` "
                     "in the root or `model/` directory.")
        else:
            with st.spinner("Running inference..."):
                prob_up, attn, inf_err = predict(model, pipeline, raw_df, seq_len)

            if inf_err:
                st.error(f"Inference error: {inf_err}")
            else:
                st.session_state["has_prediction"] = True
                conf_label, conf_color = confidence_label(prob_up)
                direction = "UP" if prob_up >= 0.5 else "DOWN"
                dir_class = "direction-up" if direction == "UP" else "direction-down"
                dir_color = "#10b981" if direction == "UP" else "#ef4444"
                dir_arrow = "↑" if direction == "UP" else "↓"

                st.markdown("<div class='section-header'>NEXT-DAY FORECAST</div>",
                            unsafe_allow_html=True)
                pred_col, gauge_col = st.columns([1, 1])
                with pred_col:
                    st.markdown(f"""
                    <div class='{dir_class}'>
                        <div class='direction-label' style='color:{dir_color}'>PREDICTED DIRECTION</div>
                        <div class='direction-arrow' style='color:{dir_color}'>{dir_arrow}</div>
                        <div class='direction-prob' style='color:{dir_color}'>{direction}</div>
                        <div class='direction-conf'>
                            P(UP) = {prob_up:.1%} &nbsp;·&nbsp;
                            <span style='color:{conf_color}'>{conf_label}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with gauge_col:
                    st.plotly_chart(make_gauge(prob_up), width="stretch")

                if show_attn and attn is not None:
                    st.markdown("<div class='section-header'>TEMPORAL ATTENTION WEIGHTS</div>",
                                unsafe_allow_html=True)
                    st.caption("Which of the last 60 trading days influenced this prediction most.")
                    st.plotly_chart(make_attention_chart(attn, seq_len), width="stretch")

                if show_indic:
                    st.markdown("<div class='section-header'>CURRENT INDICATOR SNAPSHOT</div>",
                                unsafe_allow_html=True)
                    feat_df = add_technical_indicators(raw_df.copy()).dropna()
                    latest  = feat_df.iloc[-1]
                    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                    for col, lbl, val, sub in [
                        (ic1, "RSI 14",      f"{latest.get('RSI_14', 0)*100:.1f}",    "Overbought >70, Oversold <30"),
                        (ic2, "MACD",        f"{latest.get('MACD_norm', 0)*100:.3f}%","Normalised vs price"),
                        (ic3, "BB Position", f"{latest.get('BBP', 0)*100:.1f}%",      "0%=lower, 100%=upper band"),
                        (ic4, "Vol Ratio",   f"{latest.get('VOL_ratio', 0):.2f}×",    "vs 20-day avg volume"),
                        (ic5, "Stoch %K",    f"{latest.get('STOCHk', 0)*100:.1f}",    "14-day stochastic"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class='metric-card' style='padding:14px 16px'>
                                <div class='metric-label' style='font-size:10px'>{lbl}</div>
                                <div class='metric-value' style='font-size:22px'>{val}</div>
                                <div class='metric-sub' style='font-size:10px'>{sub}</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class='disclaimer'>
                    ⚠️ <strong>Not financial advice.</strong>
                    Educational ML demonstration. ~55% directional accuracy on held-out data
                    — marginally above chance. Do not make investment decisions from these predictions.
                </div>""", unsafe_allow_html=True)

    elif model is not None:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#334155;'>
            <p style='font-family:IBM Plex Mono;font-size:32px;'>◈</p>
            <p style='font-family:IBM Plex Mono;font-size:13px;letter-spacing:2px;'>
                PRESS RUN PREDICTION TO BEGIN</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("📁 Add your model artifacts to enable predictions.\n\n"
                "Required files: `best_model.pt`, `pipeline.pkl`, `config.json`\n\n"
                f"Debug — load error: `{err}`")


if __name__ == "__main__":
    main()
