import io
import os
import urllib.request
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

# Prevent PyTorch OpenMP CPU thread deadlocks across repeated Streamlit reruns
torch.set_num_threads(1)

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "real-time **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)

DEFAULT_ANCHORS = {
    "Monetary_Fed_Rates": "federal reserve interest rate cut hike central bank ecb boj inflation monetary policy treasury yield bond",
    "Global_Trade_China": "china trade export import tariff manufacturing gdp growth economy beijing asia stimulus",
    "Tech_Equities_Earnings": "nvidia apple microsoft tech semiconductor ai chip silicon valley stocks rise rally surge optimistic earnings profit record bull",
    "Banking_Credit_Markets": "bank wall street credit debt loan financial crisis default bankruptcy plunge crash losses recession selloff bear",
    "Energy_Disasters_Supply": "oil crude opec energy commodity earthquake disaster hurricane supply chain shortage destruction halt",
    "Geopolitics_Security": "war military sanction nuclear missile conflict russia ukraine iran israel strike treaty peace",
}


# 1. Exact PyTorch Two-Tower Cross-Attention Architecture (Cell [40] of kaggle.ipynb)
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
    def __init__(self, macro_dim=45, news_dim=22, hidden_dim=64, dropout=0.30):
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
        return self.fusion(torch.cat([hm, hn * attn_weights], dim=-1))


# 2. Load FinBERT & Trained Colab Artifacts Once
@st.cache_resource
def load_all_models():
    fb_tok = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fb_mdl = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    fb_mdl.eval()

    gbdt = joblib.load("gbdt_model.pkl")
    meta = joblib.load("meta_stacker_clf.pkl")
    s_macro = joblib.load("scaler_macro.pkl")
    s_news = joblib.load("scaler_news.pkl")
    tfidf_pack = joblib.load("tfidf_anchor.pkl")

    if isinstance(tfidf_pack, dict):
        tfidf_vec = tfidf_pack["vectorizer"]
        anchors = tfidf_pack.get("anchors", DEFAULT_ANCHORS)
    else:
        tfidf_vec = tfidf_pack
        anchors = DEFAULT_ANCHORS

    macro_dim = int(getattr(s_macro, "n_features_in_", 45))
    news_dim = int(getattr(s_news, "n_features_in_", 22))

    nn_model = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
    nn_model.load_state_dict(
        torch.load("multimodal_market_net.pt", map_location=torch.device("cpu"))
    )
    nn_model.eval()

    reg_ensembles = None
    if os.path.exists("regional_ensembles.pkl"):
        raw_ens = joblib.load("regional_ensembles.pkl")
        reg_ensembles = {}
        for sym, pack in raw_ens.items():
            m = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
            m.load_state_dict(pack["nn_state_dict"])
            m.eval()
            reg_ensembles[sym] = {"nn": m, "gbdt": pack["gbdt"], "meta": pack["meta"]}

    return fb_tok, fb_mdl, gbdt, meta, s_macro, s_news, tfidf_vec, anchors, nn_model, reg_ensembles


(
    finbert_tokenizer,
    finbert_model,
    gbdt_model,
    meta_clf,
    scaler_macro,
    scaler_news,
    tfidf_anchor,
    anchor_topics,
    pytorch_model,
    regional_ensembles,
) = load_all_models()


# 3. Position-Invariant, Mini-Batched 22-Feature News Tensor (Matches Cell [52] 1-to-1)
def build_news_tensor(headlines: list[str]) -> tuple[np.ndarray, float, dict]:
    # Mini-batch FinBERT inference (batch_size=8, max_length=128) to prevent RAM spikes / hangs on many headlines
    probs_batches = []
    with torch.inference_mode():
        for start_i in range(0, len(headlines), 8):
            batch_hl = headlines[start_i : start_i + 8]
            hl_in = finbert_tokenizer(
                batch_hl, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            batch_p = F.softmax(finbert_model(**hl_in).logits, dim=-1).cpu().numpy()
            probs_batches.append(batch_p)

    hl_probs = np.vstack(probs_batches)  # Shape: (N_hl, 3) -> [0: Neutral, 1: Positive, 2: Negative]

    # Salience / conviction weighting instead of 1/log2(i+1) position decay:
    # Later headlines carry equal positional weight, and market-moving headlines aren't drowned out by filler lines
    salience_w = (1.0 - hl_probs[:, 0]) ** 1.5 + 0.05
    salience_w = salience_w / salience_w.sum()

    fb_neu = float(np.sum(hl_probs[:, 0] * salience_w))
    fb_pos = float(np.sum(hl_probs[:, 1] * salience_w))
    fb_neg = float(np.sum(hl_probs[:, 2] * salience_w))
    fb_net = fb_pos - fb_neg

    # Per-headline TF-IDF anchor projection weighted by salience (prevents TF-IDF dilution on long multi-line inputs)
    anchor_mat = tfidf_anchor.transform(list(anchor_topics.values()))
    hl_sims = cosine_similarity(tfidf_anchor.transform(headlines), anchor_mat)  # (N_hl, 6)
    sims = (salience_w.reshape(-1, 1) * hl_sims).sum(axis=0, keepdims=True).astype(np.float32)  # (1, 6)

    net_arr = np.array([[fb_net]], dtype=np.float32)
    c_sent = net_arr * (sims + 0.15)
    c_mass = sims
    max_sim = sims.max(axis=1, keepdims=True)
    mean_sim = sims.mean(axis=1, keepdims=True)
    weighted_net = net_arr * (max_sim + 0.25)
    conviction = np.array([[1.0 - fb_neu]], dtype=np.float32)
    pos_dom = np.array([[max(0.0, fb_pos - fb_neg)]], dtype=np.float32)
    neg_dom = np.array([[-max(0.0, fb_neg - fb_pos)]], dtype=np.float32)

    summary = np.hstack([weighted_net, conviction, pos_dom, neg_dom, mean_sim, net_arr])
    fb_block = np.array([[fb_neu, fb_pos, fb_neg, fb_net]], dtype=np.float32)

    x_news_raw = np.hstack([c_sent, c_mass, summary, fb_block]).astype(np.float32)
    x_news_s = np.clip(scaler_news.transform(x_news_raw), -4.5, 4.5).astype(np.float32)

    diag = {
        "n_headlines": len(headlines),
        "fb_neu": fb_neu,
        "fb_pos": fb_pos,
        "fb_neg": fb_neg,
        "fb_net": fb_net,
    }
    return x_news_s, float(weighted_net[0, 0]), diag


# 4. Fault-Tolerant Stationary 60-Day Macro Tensor (With Strict Network Timeouts)
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_stationary_macro() -> pd.DataFrame:
    tickers = ["^DJI", "^GSPC", "^N225", "000001.SS", "^VIX"]
    close_d, high_d, low_d, open_d, vol_d = {}, {}, {}, {}, {}

    for sym in tickers:
        try:
            h = yf.Ticker(sym).history(period="9mo", interval="1d", auto_adjust=True, timeout=8)
            if h is not None and not h.empty:
                if getattr(h.index, "tz", None) is not None:
                    h.index = h.index.tz_localize(None)
                h.index = h.index.normalize()
                close_d[sym] = h["Close"]
                high_d[sym] = h["High"]
                low_d[sym] = h["Low"]
                open_d[sym] = h["Open"]
                vol_d[sym] = h["Volume"]
        except Exception:
            pass

    if not close_d:
        return pd.DataFrame(np.zeros((5, 44), dtype=np.float32))

    close_px = pd.DataFrame(close_d).ffill().bfill()
    high_px = pd.DataFrame(high_d).reindex(close_px.index).ffill().bfill()
    low_px = pd.DataFrame(low_d).reindex(close_px.index).ffill().bfill()
    vol_px = pd.DataFrame(vol_d).reindex(close_px.index).ffill().bfill()

    rets = close_px.pct_change()
    mkt_feats = pd.DataFrame(index=close_px.index)

    for sym, tag in [("^DJI", "DJI"), ("^GSPC", "SPX")]:
        if sym in rets.columns:
            r_lag1 = rets[sym].shift(1)
            mkt_feats[f"{tag}_ret_lag1"] = r_lag1
            mkt_feats[f"{tag}_ret_lag2"] = rets[sym].shift(2)
            mkt_feats[f"{tag}_ret_ma5"] = r_lag1.rolling(5).mean()
            mkt_feats[f"{tag}_ret_ma20"] = r_lag1.rolling(20).mean()
            mkt_feats[f"{tag}_vol_5d"] = r_lag1.rolling(5).std()
            mkt_feats[f"{tag}_vol_20d"] = r_lag1.rolling(20).std()
            mkt_feats[f"{tag}_vol_ratio"] = mkt_feats[f"{tag}_vol_5d"] / (mkt_feats[f"{tag}_vol_20d"] + 1e-6)
            hl_range = ((high_px[sym] - low_px[sym]) / (close_px[sym] + 1e-6)).shift(1)
            mkt_feats[f"{tag}_hl_range_lag1"] = hl_range
            mkt_feats[f"{tag}_hl_range_ma5"] = hl_range.rolling(5).mean()
            mkt_feats[f"{tag}_open_gap_t"] = 0.0
            vol_s1 = vol_px[sym].shift(1)
            mkt_feats[f"{tag}_vol_z_lag1"] = (vol_s1 - vol_s1.rolling(20).mean()) / (vol_s1.rolling(20).std() + 1e-6)
        else:
            for sfx in [
                "ret_lag1", "ret_lag2", "ret_ma5", "ret_ma20", "vol_5d", "vol_20d",
                "vol_ratio", "hl_range_lag1", "hl_range_ma5", "open_gap_t", "vol_z_lag1",
            ]:
                mkt_feats[f"{tag}_{sfx}"] = 0.0

    if "^VIX" in close_px.columns:
        vix_c = close_px["^VIX"].shift(1)
        mkt_feats["VIX_lag1"] = vix_c
        mkt_feats["VIX_diff1"] = close_px["^VIX"].diff().shift(1)
        mkt_feats["VIX_ma_rat"] = vix_c / (vix_c.rolling(20).mean() + 1e-6)
        mkt_feats["VIX_open_t"] = 0.0
    else:
        for k in ["VIX_lag1", "VIX_diff1", "VIX_ma_rat", "VIX_open_t"]:
            mkt_feats[k] = 0.0

    for sym, tag in [("^N225", "NIKKEI"), ("000001.SS", "SSE")]:
        if sym in rets.columns:
            mkt_feats[f"{tag}_preUS_ret_t"] = 0.0
            mkt_feats[f"{tag}_ret_lag1"] = rets[sym].shift(1)
            mkt_feats[f"{tag}_ret_ma5"] = rets[sym].rolling(5).mean()
            mkt_feats[f"{tag}_vol_10d"] = rets[sym].rolling(10).std()
        else:
            for sfx in ["preUS_ret_t", "ret_lag1", "ret_ma5", "vol_10d"]:
                mkt_feats[f"{tag}_{sfx}"] = 0.0

    start_dt = (close_px.index.min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    for c in ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={c}&cosd={start_dt}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                csv_bytes = resp.read().decode("utf-8")
            s = pd.to_numeric(
                pd.read_csv(io.StringIO(csv_bytes), index_col=0, parse_dates=True)[c],
                errors="coerce",
            ).ffill()
            aligned = s.reindex(close_px.index).ffill().bfill()
            mkt_feats[f"{c}_lag1"] = aligned.shift(1)
            mkt_feats[f"{c}_chg5"] = aligned.diff(5).shift(1)
        except Exception:
            mkt_feats[f"{c}_lag1"] = 0.0
            mkt_feats[f"{c}_chg5"] = 0.0

    mkt_feats = mkt_feats.ffill().bfill().fillna(0.0)
    r_mean = mkt_feats.rolling(60, min_periods=10).mean().bfill()
    r_std = mkt_feats.rolling(60, min_periods=10).std().bfill().replace(0, 1.0).fillna(1.0)
    return ((mkt_feats - r_mean) / (r_std + 1e-6)).clip(-4.0, 4.0).fillna(0.0)


def build_macro_tensor(stationary_44: pd.DataFrame, news_col12: float) -> np.ndarray:
    last_44 = stationary_44.iloc[-1].values.astype(np.float32)
    x_macro_raw = np.append(last_44, np.float32(news_col12)).reshape(1, -1)
    return np.clip(scaler_macro.transform(x_macro_raw), -4.5, 4.5).astype(np.float32)


# 5. Trained Ensemble Forward Pass
def predict_index(sym: str, x_macro_s: np.ndarray, x_news_s: np.ndarray) -> list[float]:
    if regional_ensembles is not None and sym in regional_ensembles:
        pack = regional_ensembles[sym]
        with torch.inference_mode():
            nn_p = F.softmax(
                pack["nn"](torch.tensor(x_macro_s), torch.tensor(x_news_s)), dim=-1
            ).cpu().numpy()
        gb_p = pack["gbdt"].predict_proba(np.hstack([x_macro_s, x_news_s]))
        return pack["meta"].predict_proba(np.hstack([nn_p, gb_p]))[0].tolist()

    with torch.inference_mode():
        nn_p = F.softmax(
            pytorch_model(torch.tensor(x_macro_s), torch.tensor(x_news_s)), dim=-1
        ).cpu().numpy()
    gb_p = gbdt_model.predict_proba(np.hstack([x_macro_s, x_news_s]))
    return meta_clf.predict_proba(np.hstack([nn_p, gb_p]))[0].tolist()


# 6. Sidebar Form (Using st.form prevents partial mid-typing reruns in Streamlit)
with st.sidebar.form("prediction_form"):
    st.header("News Scenario Prompt")
    user_prompt = st.text_area(
        "Enter custom headlines or market events (one per line):",
        value="NVIDIA stocks rise a lot, very optimistic",
        height=200,
    )
    submitted = st.form_submit_button("Predict Across All Indices")

if submitted:
    headlines = [h.strip() for h in user_prompt.splitlines() if h.strip()]
    if not headlines:
        st.warning("Please enter at least one headline.")
        st.stop()

    with st.spinner(f"Scoring {len(headlines)} headline(s) via FinBERT + MultimodalMarketNet..."):
        stationary_44 = fetch_live_stationary_macro()
        x_news_s, news_col12, diag = build_news_tensor(headlines)
        x_macro_s = build_macro_tensor(stationary_44, news_col12)

        indices = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]
        results = [(label, predict_index(sym, x_macro_s, x_news_s)) for label, sym in indices]

    st.subheader("📊 Independent Regional Index Predictions")
    cols = st.columns(4)
    regime_names = ["Down (0)", "Neutral (1)", "Up (2)"]

    for i, (idx_name, probs) in enumerate(results):
        pred_idx = int(np.argmax(probs))
        with cols[i]:
            st.markdown(f"### {idx_name}")
            st.metric("Regime", regime_names[pred_idx], f"{probs[pred_idx] * 100:.1f}% Confidence")
            prob_df = pd.DataFrame(
                {"Class": ["Down", "Neutral", "Up"], "Probability": probs}
            ).set_index("Class")
            st.bar_chart(prob_df, height=200)

    st.markdown("---")
    st.success(
        f"Processed **{diag['n_headlines']}** headline(s) | "
        f"FinBERT Salience-Weighted `[Neu: {diag['fb_neu']:.3f}, Pos: {diag['fb_pos']:.3f}, "
        f"Neg: {diag['fb_neg']:.3f}, Net: {diag['fb_net']:+.3f}]`"
    )
else:
    st.info("👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**.")
