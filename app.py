import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "real-time **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)

# Exact 6 Domain-Anchored Macro Prototypes from Cell [41] of kaggle.ipynb
ANCHOR_TOPICS = {
    "Monetary_Fed_Rates": "federal reserve interest rate central bankecb boj inflation monetary policy treasury yield bond",
    "Global_Trade_China": "china trade export import tariff manufacturing gdp growth economy beijing asia supply chain",
    "Energy_Oil_Commodities": "oil crude opec energy gas barrel commodity gold copper prices production pipeline",
    "Banking_Credit_Markets": "bank Wall Street credit debt loan financial crisis hedge fund stock market investor",
    "Sovereign_FX_Fiscal": "currency dollar euro yen yuan debt ceiling budget deficit imf sovereign bailout tax",
    "Geopolitics_Security": "war military sanction nuclear missile conflict russia ukraine iran israel strike treaty",
}
N_CLUSTERS = len(ANCHOR_TOPICS)


# 1. Exact PyTorch Two-Tower Cross-Attention Architecture (Cell [42] of kaggle.ipynb)
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
        attn_weights = self.cross_attn_gate(torch.cat([hm, hn], dim=-1))
        hn_modulated = hn * attn_weights
        return self.fusion(torch.cat([hm, hn_modulated], dim=-1))


# 2. Load Exact FinBERT (yiyanghkust/finbert-tone from Cell [9]) & Trained Artifacts
@st.cache_resource
def load_models():
    # FinBERT from Cell [9]: output classes are strictly [0: Neutral, 1: Positive, 2: Negative]
    fb_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fb_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    fb_model.eval()

    gbdt = joblib.load("gbdt_model.pkl")
    meta = joblib.load("meta_stacker_clf.pkl")
    s_macro = joblib.load("scaler_macro.pkl")
    s_news = joblib.load("scaler_news.pkl")
    tfidf_loaded = joblib.load("tfidf_anchor.pkl")
    tfidf_vec = tfidf_loaded["vectorizer"] if isinstance(tfidf_loaded, dict) else tfidf_loaded

    macro_dim = int(getattr(s_macro, "n_features_in_", 45))
    news_dim = int(getattr(s_news, "n_features_in_", 22))

    nn_model = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
    nn_model.load_state_dict(
        torch.load("multimodal_market_net.pt", map_location=torch.device("cpu"))
    )
    nn_model.eval()

    # Load leak-free per-index Two-Tower ensembles if exported from updated Cell [35]
    reg_ensembles = None
    if os.path.exists("regional_ensembles.pkl"):
        raw_ens = joblib.load("regional_ensembles.pkl")
        reg_ensembles = {}
        for sym, pack in raw_ens.items():
            idx_nn = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
            idx_nn.load_state_dict(pack["nn_state_dict"])
            idx_nn.eval()
            reg_ensembles[sym] = {
                "nn": idx_nn,
                "gbdt": pack["gbdt"],
                "meta": pack["meta"],
                "mask_cols": pack.get("mask_cols", []),
            }

    return fb_tokenizer, fb_model, gbdt, meta, s_macro, s_news, tfidf_vec, nn_model, reg_ensembles


(
    finbert_tokenizer,
    finbert_model,
    gbdt_model,
    meta_clf,
    scaler_macro,
    scaler_news,
    tfidf_anchor,
    pytorch_model,
    regional_ensembles,
) = load_models()


# 3. Exact 22-Feature News Tensor (Matches Cell [9] + Cell [41] 1-to-1, Zero VADER)
def build_news_features_cell41(headlines: list[str]) -> tuple[np.ndarray, float, dict]:
    n_hl = len(headlines)
    rank_w = np.array([1.0 / np.log2(i + 1) for i in range(1, n_hl + 1)], dtype=np.float32)
    rank_w /= rank_w.sum()

    # A. Score each headline & daily concatenated block with yiyanghkust/finbert-tone
    with torch.no_grad():
        hl_inputs = finbert_tokenizer(
            headlines, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        hl_probs = F.softmax(finbert_model(**hl_inputs).logits, dim=-1).cpu().numpy()

        daily_text = " ".join(headlines)
        day_inputs = finbert_tokenizer(
            [daily_text], return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        day_probs = F.softmax(finbert_model(**day_inputs).logits, dim=-1).cpu().numpy()[0]

    # yiyanghkust/finbert-tone order: [0: Neutral, 1: Positive, 2: Negative]
    fb_neu = float(day_probs[0])
    fb_pos = float(day_probs[1])
    fb_neg = float(day_probs[2])
    fb_net = fb_pos - fb_neg

    # Per-headline neural polarity in [-1.0, +1.0]
    sents = (hl_probs[:, 1] - hl_probs[:, 2]).astype(np.float32)

    # B. Project onto the exact 6 ANCHOR_TOPICS via fitted tfidf_anchor (Cell [41])
    anchor_vecs = tfidf_anchor.transform(list(ANCHOR_TOPICS.values()))
    hl_tfidf = tfidf_anchor.transform(headlines)
    sims = cosine_similarity(hl_tfidf, anchor_vecs)  # Shape: (n_hl, 6)

    # Start from scaler_news.center_ for inactive topic slots so 0 cosine overlap maps to 0.0
    # after RobustScaler instead of shifting positive against negative r/worldnews medians
    centers = (
        np.asarray(scaler_news.center_, dtype=np.float32)
        if hasattr(scaler_news, "center_")
        else np.zeros(22, dtype=np.float32)
    )

    cluster_sent = centers[0:6].copy()
    cluster_mass = centers[6:12].copy()

    w_s = sents * rank_w
    for k in range(N_CLUSTERS):
        rel = sims[:, k]
        mass_k = float(np.sum(rank_w * rel))
        if mass_k > 1e-6:
            cluster_sent[k] = float(np.sum(w_s * rel))
            cluster_mass[k] = mass_k

    fin_rel_total = sims.max(axis=1)
    # Use learned median relevance (centers[16]) as floor when custom terms fall outside anchor strings
    eff_rel = np.maximum(fin_rel_total, float(max(centers[16], 0.12)))

    summary_stats = np.zeros(6, dtype=np.float32)
    summary_stats[0] = float(np.sum(sents * rank_w * (eff_rel + 0.05)))
    summary_stats[1] = float(np.std(sents)) if n_hl > 1 else float(centers[13])
    summary_stats[2] = float(np.min(sents))
    summary_stats[3] = float(np.max(sents))
    summary_stats[4] = float(np.mean(fin_rel_total)) if np.any(fin_rel_total > 0) else float(centers[16])
    summary_stats[5] = float(np.mean(sents < -0.3) - np.mean(sents > 0.3))

    fb_vec = np.array([fb_neu, fb_pos, fb_neg, fb_net], dtype=np.float32)

    # Exact 22-column order from Cell [41]: [cluster_sent(6), cluster_mass(6), summary_stats(6), finbert(4)]
    x_news_raw = np.concatenate([cluster_sent, cluster_mass, summary_stats, fb_vec]).reshape(1, -1)
    x_news_s = np.clip(scaler_news.transform(x_news_raw), -5.0, 5.0).astype(np.float32)

    diag = {
        "fb_neu": fb_neu,
        "fb_pos": fb_pos,
        "fb_neg": fb_neg,
        "fb_net": fb_net,
        "news_col12_raw": float(summary_stats[0]),
    }
    return x_news_s, float(summary_stats[0]), diag


# 4. Exact 45-Feature Macro Tensor (Matches Cell [40] + Cell [41] 1-to-1)
@st.cache_data(ttl=900)
def fetch_market_and_fred() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tickers = ["^DJI", "^GSPC", "^N225", "000001.SS", "^VIX"]
    mkt = {}
    for sym in tickers:
        df_t = yf.Ticker(sym).history(period="6mo", interval="1d", auto_adjust=True)
        if df_t is not None and not df_t.empty:
            if getattr(df_t.index, "tz", None) is not None:
                df_t.index = df_t.index.tz_localize(None)
            df_t.index = df_t.index.normalize()
            mkt[sym] = df_t[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill()

    fred_codes = ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]
    start_dt = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fred_cols = {}
    for c in fred_codes:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={c}&cosd={start_dt}"
        s = pd.read_csv(url, index_col=0, parse_dates=True)[c]
        fred_cols[c] = pd.to_numeric(s, errors="coerce").ffill()

    return mkt, pd.DataFrame(fred_cols).ffill().bfill()


def build_macro_features_cell40(
    mkt: dict[str, pd.DataFrame],
    fred_df: pd.DataFrame,
    news_roll3_val: float,
    focus_sym: str,
) -> np.ndarray:
    """Builds exact 44 market/FRED columns (Cell [40]) + 45th column `news_roll3` (Cell [41])."""
    vals = []

    # Swap primary ticker only when evaluating single-target model across non-DJI indices
    if regional_ensembles is not None:
        us_pairs = [("^DJI", "DJI"), ("^GSPC", "SPX")]
    else:
        sec = "^GSPC" if focus_sym != "^GSPC" else "^DJI"
        us_pairs = [(focus_sym, "DJI"), (sec, "SPX")]

    # Cols 0..21: Strictly lagged US features (t-1 = iloc[-1], t-2 = iloc[-2])
    for sym, _ in us_pairs:
        d = mkt[sym]
        c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        r = c.pct_change()
        vol_5d = float(r.rolling(5).std().iloc[-1])
        vol_20d = float(r.rolling(20).std().iloc[-1])
        hl_range = (h - l) / (c + 1e-6)
        vol_z = float((v.iloc[-1] - v.rolling(20).mean().iloc[-1]) / (v.rolling(20).std().iloc[-1] + 1e-6))

        vals.extend([
            float(r.iloc[-1]),                       # ret_lag1
            float(r.iloc[-2]),                       # ret_lag2
            float(r.rolling(5).mean().iloc[-1]),     # ret_ma5
            float(r.rolling(20).mean().iloc[-1]),    # ret_ma20
            vol_5d,                                  # vol_5d
            vol_20d,                                 # vol_20d
            vol_5d / (vol_20d + 1e-6),               # vol_ratio
            float(hl_range.iloc[-1]),                # hl_range_lag1
            float(hl_range.rolling(5).mean().iloc[-1]),  # hl_range_ma5
            0.0,                                     # open_gap_t (0.0 prior to next session open)
            vol_z,                                   # vol_z_lag1
        ])

    # Cols 22..25: VIX features
    vc = mkt["^VIX"]["Close"]
    vals.extend([
        float(vc.iloc[-1]),                                        # VIX_lag1
        float(vc.diff().iloc[-1]),                                 # VIX_diff1
        float(vc.iloc[-1] / (vc.rolling(20).mean().iloc[-1] + 1e-6)),  # VIX_ma_rat
        0.0,                                                       # VIX_open_t
    ])

    # Cols 26..33: Asian session features (NIKKEI cols 26..29, SSE cols 30..33)
    for sym in ["^N225", "000001.SS"]:
        ar = mkt[sym]["Close"].pct_change()
        pre_ret = 0.0 if sym == focus_sym else float(ar.iloc[-1])
        vals.extend([
            pre_ret,                                 # preUS_ret_t (masked when predicting own index)
            float(ar.iloc[-1]),                      # ret_lag1
            float(ar.rolling(5).mean().iloc[-1]),    # ret_ma5
            float(ar.rolling(10).std().iloc[-1]),    # vol_10d
        ])

    # Cols 34..43: FRED Macroeconomic Series (lag1 & chg5)
    for code in ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]:
        s = fred_df[code].dropna()
        vals.extend([
            float(s.iloc[-1]),               # {code}_lag1
            float(s.iloc[-1] - s.iloc[-6]),  # {code}_chg5
        ])

    # Col 44 (45th feature in Cell [41]): news_roll3
    vals.append(float(news_roll3_val))

    x_macro_raw = np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0).reshape(1, -1)
    x_macro_s = np.clip(scaler_macro.transform(x_macro_raw), -5.0, 5.0).astype(np.float32)
    return x_macro_s


# 5. Real-Time Inference via Two-Tower MultimodalMarketNet + GBDT + Meta-Stacker
def predict_regime(ticker: str, x_macro_s: np.ndarray, x_news_s: np.ndarray) -> list[float]:
    if regional_ensembles is not None and ticker in regional_ensembles:
        pack = regional_ensembles[ticker]
        xm = x_macro_s.copy()
        if pack["mask_cols"]:
            xm[:, pack["mask_cols"]] = 0.0
        with torch.no_grad():
            logits = pack["nn"](torch.tensor(xm), torch.tensor(x_news_s))
            nn_p = F.softmax(logits, dim=-1).cpu().numpy()
        gbdt_p = pack["gbdt"].predict_proba(np.hstack([xm, x_news_s]))
        # Exact Cell [43] stacking order: [nn_probs, gbdt_probs]
        meta_in = np.hstack([nn_p, gbdt_p])
        return pack["meta"].predict_proba(meta_in)[0].tolist()

    # Standalone artifacts fallback (multimodal_market_net.pt + gbdt_model.pkl + meta_stacker_clf.pkl)
    with torch.no_grad():
        logits = pytorch_model(torch.tensor(x_macro_s), torch.tensor(x_news_s))
        nn_probs = F.softmax(logits, dim=-1).cpu().numpy()

    x_full = np.hstack([x_macro_s, x_news_s])
    gbdt_probs = gbdt_model.predict_proba(x_full)

    # Cell [43] trained meta_clf on [nn_probs, gbdt_probs]
    meta_in = np.hstack([nn_probs, gbdt_probs])
    return meta_clf.predict_proba(meta_in)[0].tolist()


# 6. Streamlit UI
st.sidebar.header("News Scenario Prompt")
user_prompt = st.sidebar.text_area(
    "Enter custom headlines or market events (one per line):",
    value="Earthquake destroys Silicon Valley causing severe losses.",
    height=180,
)

predict_btn = st.sidebar.button("Predict Across All Indices")

if predict_btn:
    headlines = [h.strip() for h in user_prompt.split("\n") if h.strip()]
    if not headlines:
        st.warning("Please enter at least one headline.")
        st.stop()

    with st.spinner("Running FinBERT (yiyanghkust/finbert-tone) + TF-IDF Anchors + Two-Tower MultimodalMarketNet..."):
        mkt_data, fred_data = fetch_market_and_fred()
        x_news_s, news_col12, diag = build_news_features_cell41(headlines)

        indices = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]

        results = []
        for idx_label, sym in indices:
            x_macro_s = build_macro_features_cell40(
                mkt_data, fred_data, news_roll3_val=news_col12, focus_sym=sym
            )
            probs = predict_regime(sym, x_macro_s, x_news_s)
            results.append((idx_label, probs))

    st.subheader("📊 Independent Regional Index Predictions")
    cols = st.columns(4)
    regime_names = ["Down (0)", "Neutral (1)", "Up (2)"]

    for i, (idx_name, probs) in enumerate(results):
        pred_idx = int(np.argmax(probs))
        pred_regime = regime_names[pred_idx]
        conf = f"{probs[pred_idx] * 100:.1f}% Confidence"

        with cols[i]:
            st.markdown(f"### {idx_name}")
            st.metric("Regime", pred_regime, conf)
            prob_df = pd.DataFrame(
                {"Class": ["Down", "Neutral", "Up"], "Probability": probs}
            ).set_index("Class")
            st.bar_chart(prob_df, height=200)

    st.markdown("---")
    st.success(
        f"Executed `yiyanghkust/finbert-tone` + `MultimodalMarketNet` + `LightGBM` + `Meta-Stacker` | "
        f"FinBERT `[Neu: {diag['fb_neu']:.3f}, Pos: {diag['fb_pos']:.3f}, Neg: {diag['fb_neg']:.3f}, Net: {diag['fb_net']:+.3f}]`"
    )
else:
    st.info("👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**.")
