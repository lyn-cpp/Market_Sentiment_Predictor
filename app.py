import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "independent **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)


# 1. PyTorch Architecture Definition
class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.30):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        val, gate = self.fc2(h).chunk(2, dim=-1)
        return self.norm(x + (val * torch.sigmoid(gate)))


class MultimodalMarketNet(nn.Module):
    def __init__(self, macro_dim=45, news_dim=22, hidden_dim=64, dropout=0.32):
        super().__init__()
        self.macro_in = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            GatedResidualBlock(hidden_dim, dropout),
        )
        self.news_in = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            GatedResidualBlock(hidden_dim, dropout),
        )
        self.cross_attn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            GatedResidualBlock(hidden_dim, dropout),
            GatedResidualBlock(hidden_dim, dropout),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_m, x_n):
        hm = self.macro_in(x_m)
        hn = self.news_in(x_n)
        attn = self.cross_attn_gate(torch.cat([hm, hn], dim=-1))
        return self.fusion(torch.cat([hm, hn * attn], dim=-1))


# 2. Load Model Artifacts
@st.cache_resource
def load_artifacts():
    gbdt = joblib.load("gbdt_model.pkl")
    meta = joblib.load("meta_stacker_clf.pkl")
    s_macro = joblib.load("scaler_macro.pkl")
    s_news = joblib.load("scaler_news.pkl")
    tfidf = joblib.load("tfidf_anchor.pkl")

    macro_dim = getattr(s_macro, "n_features_in_", 45)
    news_dim = getattr(s_news, "n_features_in_", 22)

    pytorch_model = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
    pytorch_model.load_state_dict(
        torch.load("multimodal_market_net.pt", map_location=torch.device("cpu"))
    )
    pytorch_model.eval()
    return gbdt, meta, s_macro, s_news, tfidf, pytorch_model


gbdt_model, meta_clf, scaler_macro, scaler_news, tfidf_anchor, pytorch_model = load_artifacts()


# 3. Feature Builders (Macro = 45 dims, News = 22 dims)
@st.cache_data(ttl=900)
def fetch_live_macro_vector(target_ticker: str, expected_dim: int = 45) -> np.ndarray:
    """Fetches live market & cross-asset macro series via yfinance and aligns to scaler_macro."""
    tickers = [target_ticker, "^VIX", "^TNX", "DX-Y.NYB", "GC=F", "CL=F"]
    feats = []
    try:
        data = yf.download(tickers, period="3mo", interval="1d", progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.ffill().bfill()

        for col in tickers:
            if col in data.columns and len(data[col].dropna()) >= 21:
                s = data[col].dropna()
                ret_1d = s.pct_change(1).iloc[-1]
                ret_5d = s.pct_change(5).iloc[-1]
                ret_20d = s.pct_change(20).iloc[-1]
                vol_5d = s.pct_change().rolling(5).std().iloc[-1]
                vol_20d = s.pct_change().rolling(20).std().iloc[-1]
                ma_ratio_5 = (s.iloc[-1] / s.rolling(5).mean().iloc[-1]) - 1.0
                ma_ratio_20 = (s.iloc[-1] / s.rolling(20).mean().iloc[-1]) - 1.0
                feats.extend([ret_1d, ret_5d, ret_20d, vol_5d, vol_20d, ma_ratio_5, ma_ratio_20])
    except Exception:
        pass

    arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Pad with neutral training mean (if available) or truncate to match exact expected_dim (45)
    if len(arr) < expected_dim:
        pad_vals = (
            scaler_macro.mean_[len(arr) : expected_dim]
            if hasattr(scaler_macro, "mean_") and len(scaler_macro.mean_) == expected_dim
            else np.zeros(expected_dim - len(arr), dtype=np.float32)
        )
        arr = np.concatenate([arr, pad_vals])
    return arr[:expected_dim].reshape(1, -1)


def build_news_vector(headlines: list[str], expected_dim: int = 22) -> tuple[np.ndarray, float]:
    """Builds news feature vector using VADER distributions + saved tfidf_anchor."""
    scores = [sia.polarity_scores(h) for h in headlines]
    compounds = [s["compound"] for s in scores]
    pos_vals = [s["pos"] for s in scores]
    neg_vals = [s["neg"] for s in scores]
    neu_vals = [s["neu"] for s in scores]

    vader_stats = np.array(
        [
            np.mean(compounds),
            np.std(compounds),
            np.max(compounds),
            np.min(compounds),
            np.mean(pos_vals),
            np.mean(neg_vals),
            np.mean(neu_vals),
            np.sum(np.array(compounds) > 0.05) / len(headlines),
            np.sum(np.array(compounds) < -0.05) / len(headlines),
        ],
        dtype=np.float32,
    )

    combined_text = " ".join(headlines)
    tfidf_vec = tfidf_anchor.transform([combined_text]).toarray().flatten().astype(np.float32)

    # Combine TF-IDF anchor features with VADER stats depending on trained feature order
    if len(tfidf_vec) == expected_dim:
        raw_news = tfidf_vec
    elif len(tfidf_vec) + len(vader_stats) >= expected_dim:
        needed_vader = max(0, expected_dim - len(tfidf_vec))
        raw_news = np.concatenate([tfidf_vec[:expected_dim], vader_stats[:needed_vader]])
    else:
        raw_news = np.concatenate([tfidf_vec, vader_stats])
        if len(raw_news) < expected_dim:
            raw_news = np.pad(raw_news, (0, expected_dim - len(raw_news)))

    return raw_news[:expected_dim].reshape(1, -1), float(np.mean(compounds))


def run_ensemble_inference(x_macro_raw: np.ndarray, x_news_raw: np.ndarray) -> list[float]:
    """Executes scaler -> PyTorch MultimodalMarketNet + GBDT -> Meta Stacker Classifier."""
    # Scale inputs using trained Colab scalers
    x_m_scaled = scaler_macro.transform(x_macro_raw).astype(np.float32)
    x_n_scaled = scaler_news.transform(x_news_raw).astype(np.float32)

    # A. PyTorch Cross-Attention Forward Pass
    with torch.no_grad():
        t_m = torch.tensor(x_m_scaled, dtype=torch.float32)
        t_n = torch.tensor(x_n_scaled, dtype=torch.float32)
        nn_logits = pytorch_model(t_m, t_n)
        nn_probs = F.softmax(nn_logits, dim=-1).cpu().numpy()  # Shape: (1, 3)

    # B. GBDT Forward Pass (align input features dynamically)
    full_scaled = np.hstack([x_m_scaled, x_n_scaled])
    gbdt_dim = getattr(gbdt_model, "n_features_in_", full_scaled.shape[1])
    if gbdt_dim == x_m_scaled.shape[1]:
        gbdt_in = x_m_scaled
    elif gbdt_dim == x_n_scaled.shape[1]:
        gbdt_in = x_n_scaled
    else:
        gbdt_in = full_scaled[:, :gbdt_dim]
    gbdt_probs = gbdt_model.predict_proba(gbdt_in)  # Shape: (1, 3)

    # C. Meta-Stacker Classifier Forward Pass
    meta_features = np.hstack([gbdt_probs, nn_probs])  # Default 6-dim stack: [gbdt_p(3), nn_p(3)]
    meta_dim = getattr(meta_clf, "n_features_in_", meta_features.shape[1])
    if meta_dim != meta_features.shape[1]:
        # Fallback if meta-stacker also included raw features or logits
        extended = np.hstack([gbdt_probs, nn_probs, full_scaled])
        meta_features = extended[:, :meta_dim]

    final_probs = meta_clf.predict_proba(meta_features)[0]
    return final_probs.tolist()


# 4. User Input Interface
st.sidebar.header("News Scenario Prompt")
user_prompt = st.sidebar.text_area(
    "Enter custom headlines or market events (one per line):",
    value=(
        "Federal Reserve announces unexpected interest rate cut amid rising inflation pressures.\n"
        "China reports stronger than expected manufacturing GDP growth."
    ),
    height=180,
)

predict_btn = st.sidebar.button("Predict Across All Indices")

if predict_btn:
    with st.spinner("Fetching live macro context and running PyTorch + GBDT + Meta-Stacker..."):
        headlines = [h.strip() for h in user_prompt.split("\n") if h.strip()]
        if not headlines:
            headlines = ["Market stability expected amid ongoing economic updates."]

        macro_dim = getattr(scaler_macro, "n_features_in_", 45)
        news_dim = getattr(scaler_news, "n_features_in_", 22)

        x_news_raw, compound_score = build_news_vector(headlines, expected_dim=news_dim)

        index_map = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]

        results = []
        for display_name, ticker in index_map:
            x_macro_raw = fetch_live_macro_vector(ticker, expected_dim=macro_dim)
            probs = run_ensemble_inference(x_macro_raw, x_news_raw)
            results.append((display_name, probs))

    st.subheader("📊 Independent Regional Index Predictions (True Ensemble Inference)")
    cols = st.columns(4)
    regime_names = ["Down (0)", "Neutral (1)", "Up (2)"]

    for i, (idx_name, probs) in enumerate(results):
        pred_class_idx = int(np.argmax(probs))
        pred_regime = regime_names[pred_class_idx]
        conf = f"{probs[pred_class_idx] * 100:.1f}% Confidence"

        with cols[i]:
            st.markdown(f"### {idx_name}")
            st.metric("Regime", pred_regime, conf)
            prob_df = pd.DataFrame(
                {"Class": ["Down", "Neutral", "Up"], "Probability": probs}
            ).set_index("Class")
            st.bar_chart(prob_df, height=200)

    st.markdown("---")
    st.success(
        f"Ran live ensemble inference across {len(headlines)} headline(s)! "
        f"Mean VADER Compound: `{compound_score:.3f}`"
    )
else:
    st.info("👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**.")
