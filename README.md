# S&P 500 Direction Predictor
### BiLSTM + Temporal Attention | Shivam Tiwari (065104)

A production-ready Streamlit app for next-day S&P 500 direction prediction using a
stacked LSTM with temporal attention trained on 38 years of daily OHLCV data.

Dseployment: https://s-p-500-lstm-stock-predictor-xrtibnlqfxrrrxzgkrjjcu.streamlit.app/

---

## Project Structure

```
sp500_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Streamlit theme + server config
└── model/                  # ← ADD YOUR FILES HERE
    ├── best_model.pt       # Trained model weights (from Colab)
    ├── pipeline.pkl        # Fitted RobustScaler (from Colab)
    └── config.json         # Model hyperparameters (from Colab)
```

---

## Quickstart (Local)

```bash
# 1. Clone / download this folder
cd sp500_app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your model artifacts to model/
#    Copy from your Colab /content/checkpoints/ folder:
#      best_model.pt, pipeline.pkl, config.json

# 5. Run
streamlit run app.py
```

## Features

- **Live data**: fetches real ^GSPC data from Yahoo Finance (cached 1hr)
- **44 technical indicators**: EMA, RSI, MACD, Stochastic, Bollinger Bands, ATR, OBV, etc.
- **Direction prediction**: P(UP) displayed as gauge + banner
- **Confidence levels**: Very Low / Low / Moderate / High based on distance from 0.5
- **Attention heatmap**: shows which of the 60 lookback days influenced the prediction
- **Indicator snapshot**: live values for RSI, MACD, BB Position, Volume Ratio, Stochastic
- **Configurable**: sequence length and chart window adjustable from sidebar

---

## Important Disclaimer

This is an **educational machine learning project**.
The model achieves ~55% directional accuracy on held-out data — marginally above chance.
**Do not make investment decisions based on these predictions.**

---

## Model Info

| Property | Value |
|---|---|
| Architecture | 2-layer LSTM + Temporal Attention |
| Input features | 44 technical indicators |
| Sequence length | 60 trading days |
| Training data | S&P 500 daily OHLCV, Jan 1986 – Dec 2024 |
| Test accuracy | ~55.04% directional |
| AUC-ROC | 0.5321 |
| Parameters | ~62,000 |
| Model size | ~254 KB |
