# S&P 500 Direction Predictor
### BiLSTM + Temporal Attention | Shivam Tiwari (065104)

A production-ready Streamlit app for next-day S&P 500 direction prediction using a
stacked LSTM with temporal attention trained on 38 years of daily OHLCV data.

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

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this entire folder to a **public GitHub repository**
   - Include the `model/` folder with all three artifact files
   - The model is only 254 KB so committing it directly is fine

2. Go to https://share.streamlit.io

3. Click **"New app"** → connect your GitHub account

4. Select:
   - **Repository**: your repo
   - **Branch**: main
   - **Main file path**: `app.py`

5. Click **Deploy** — takes ~2 minutes

6. Your app is live at `https://your-app-name.streamlit.app`

### Notes for Streamlit Cloud
- Free tier has 1 GB RAM — fine for this model (~250 MB PyTorch overhead)
- Apps spin down after ~7 days of inactivity (cold start ~30s)
- Data is cached for 1 hour (`@st.cache_data(ttl=3600)`)
- If yfinance fails, the app will show an error — add a retry or alternative source

---

## Deploy to HuggingFace Spaces (Alternative)

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Streamlit**
   - Visibility: Public

2. Clone the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/sp500-predictor
   ```

3. Copy all files into the cloned folder:
   ```bash
   cp -r sp500_app/* sp500-predictor/
   ```

4. Push:
   ```bash
   cd sp500-predictor
   git add .
   git commit -m "Initial deployment"
   git push
   ```

5. HuggingFace will build and deploy automatically (~3 minutes)

### Notes for HuggingFace Spaces
- For large model files (>50 MB), use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pt" "*.pkl"
  git add .gitattributes
  ```
- This model is only 254 KB so LFS is not needed
- Spaces stay awake longer than Streamlit Cloud free tier

---

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
