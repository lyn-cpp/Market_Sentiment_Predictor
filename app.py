import os

# Must be set BEFORE importing torch, lightgbm, and transformers on Streamlit Cloud Linux hosts
# to prevent OpenMP (libgomp vs libomp) & tokenizer thread deadlocks across repeated predictions
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import urllib.request
from pathlib import Path

import joblib
import lightgbm  # noqa: F401 - required before unpickling LGBMClassifier artifacts
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForSequenceClassification, BertTokenizer

torch.set_num_threads(1)
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "real-time **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)

DEFAULT_MACRO_COLS = [
    "DJI_ret_lag1", "DJI_ret_lag2", "DJI_ret_ma5", "DJI_ret_ma20", "DJI_vol_5d",
    "DJI_vol_20d", "DJI_vol_ratio", "DJI_hl_range_lag1", "DJI_hl_range_ma5",
    "DJI_open_gap_t", "DJI_vol_z_lag1",
    "SPX_ret_lag1", "SPX_ret_lag2", "SPX_ret_ma5", "SPX_ret_ma20", "SPX_vol_5d",
    "SPX_vol_20d", "SPX_vol_ratio", "SPX_hl_range_lag1", "SPX_hl_range_ma5",
    "SPX_open_gap_t", "SPX_vol_z_lag1",
    "VIX_lag1", "VIX_diff1", "VIX_ma_rat", "VIX_open_t",
    "NIKKEI_preUS_ret_t", "NIKKEI_ret_lag1", "NIKKEI_ret_ma5", "NIKKEI_vol_10d",
    "SSE_preUS_ret_t", "SSE_ret_lag1", "SSE_ret_ma5", "SSE_vol_10d",
    "T10Y2Y_lag1", "T10Y2Y_chg5", "BAA10Y_lag1", "BAA10Y_chg5",
    "DCOILWTICO_lag1", "DCOILWTICO_chg5", "DEXJPUS_lag1", "DEXJPUS_chg5",
    "DEXCHUS_lag1", "DEXCHUS_chg5",
]


# 1. Two-Tower Gated Cross-Attention Architecture (Exact match to Cell 3)
class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.30):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.fc1(x)))
        val, gate = self.fc2(h).chunk(2, dim=-1)
        return self.norm(x + val * torch.sigmoid(gate))


class MultimodalMarketNet(nn.Module):
    def __init__(
        self, macro_dim: int = 45, news_dim: int = 22, hidden_dim: int = 64, dropout: float = 0.30
    ):
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

    def forward(self, x_m: torch.Tensor, x_n: torch.Tensor) -> torch.Tensor:
        hm = self.macro_in(x_m)
        hn = self.news_in(x_n)
        attn_weights = self.cross_attn_gate(torch.cat([hm, hn], dim=-1))
        return self.fusion(torch.cat([hm, hn * attn_weights], dim=-1))


# 2. Load FinBERT & All Exported Artifacts Once in Shared Cache
@st.cache_resource(show_spinner="Loading FinBERT and trained Multimodal Ensemble weights...")
def load_pipeline():
    fb_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fb_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    fb_model.eval()

    s_macro = joblib.load(BASE_DIR / "scaler_macro.pkl")
    s_news = joblib.load(BASE_DIR / "scaler_news.pkl")
    tfidf_bundle = joblib.load(BASE_DIR / "tfidf_anchor.pkl")

    tfidf_vec = tfidf_bundle["vectorizer"]
    anchors_dict = tfidf_bundle["anchors"]
    macro_cols = tfidf_bundle.get("macro_cols") or DEFAULT_MACRO_COLS
    anchor_mat = tfidf_vec.transform(list(anchors_dict.values()))

    macro_dim = int(getattr(s_macro, "n_features_in_", 45))
    news_dim = int(getattr(s_news, "n_features_in_", 22))

    # Load per-index ensembles from regional_ensembles.pkl (with single-model fallback)
    ensembles = {}
    reg_path = BASE_DIR / "regional_ensembles.pkl"
    if reg_path.exists():
        raw_reg = joblib.load(reg_path)
        for sym, pack in raw_reg.items():
            nn_m = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
            nn_m.load_state_dict(pack["nn_state_dict"])
            nn_m.eval()

            gb_m = pack["gbdt"]
            gb_m.set_params(n_jobs=1)

            ensembles[sym] = {"nn": nn_m, "gbdt": gb_m, "meta": pack["meta"]}
    else:
        nn_m = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
        nn_m.load_state_dict(
            torch.load(BASE_DIR / "multimodal_market_net.pt", map_location="cpu")
        )
        nn_m.eval()

        gb_m = joblib.load(BASE_DIR / "gbdt_model.pkl")
        gb_m.set_params(n_jobs=1)
        meta_m = joblib.load(BASE_DIR / "meta_stacker_clf.pkl")

        for sym in ["^DJI", "^GSPC", "^N225", "000001.SS"]:
            ensembles[sym] = {"nn": nn_m, "gbdt": gb_m, "meta": meta_m}

    return fb_tokenizer, fb_model, s_macro, s_news, tfidf_vec, anchor_mat, macro_cols, ensembles


(
    finbert_tokenizer,
    finbert_model,
    scaler_macro,
    scaler_news,
    tfidf_v2,
    anchor_mat,
    saved_macro_cols,
    regional_ensembles,
) = load_pipeline()


# 3. Exact 22-Dim News Feature Builder (Matches Cell 2 `build_news_matrix_from_finbert_and_sims`)
def build_news_matrix_from_finbert_and_sims(
    fb_neu_a: np.ndarray, fb_pos_a: np.ndarray, fb_neg_a: np.ndarray, sims_a: np.ndarray
) -> np.ndarray:
    net = (fb_pos_a - fb_neg_a).reshape(-1, 1)
    c_sent = net * (sims_a + 0.15)                                    # Cols 0..5
    c_mass = sims_a                                                   # Cols 6..11
    max_sim = sims_a.max(axis=1, keepdims=True)
    mean_sim = sims_a.mean(axis=1, keepdims=True)
    weighted_net = net * (max_sim + 0.25)
    conviction = (1.0 - fb_neu_a).reshape(-1, 1)
    pos_dom = np.maximum(0.0, fb_pos_a - fb_neg_a).reshape(-1, 1)
    neg_dom = np.maximum(0.0, fb_neg_a - fb_pos_a).reshape(-1, 1)
    summary = np.hstack([weighted_net, conviction, pos_dom, -neg_dom, mean_sim, net])  # Cols 12..17
    fb_block = np.column_stack([fb_neu_a, fb_pos_a, fb_neg_a, fb_pos_a - fb_neg_a])    # Cols 18..21
    return np.hstack([c_sent, c_mass, summary, fb_block]).astype(np.float32)


def encode_user_headlines(headlines: list[str]) -> tuple[np.ndarray, float, dict]:
    # Mini-batch FinBERT scoring (prevents memory spikes on Streamlit Cloud's 1GB RAM limit)
    batches = []
    with torch.inference_mode():
        for i in range(0, len(headlines), 8):
            enc = finbert_tokenizer(
                headlines[i : i + 8],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            probs = F.softmax(finbert_model(**enc).logits, dim=-1).cpu().numpy()
            batches.append(probs)

    hl_probs = np.vstack(batches)  # [0: Neutral, 1: Positive, 2: Negative]

    # Salience weighting: identical to Cell 3 for 1 headline; position-invariant for multiple headlines
    salience_w = (1.0 - hl_probs[:, 0]) ** 1.5 + 0.05
    salience_w = (salience_w / salience_w.sum()).astype(np.float32)

    fb_neu = np.array([np.sum(hl_probs[:, 0] * salience_w)], dtype=np.float32)
    fb_pos = np.array([np.sum(hl_probs[:, 1] * salience_w)], dtype=np.float32)
    fb_neg = np.array([np.sum(hl_probs[:, 2] * salience_w)], dtype=np.float32)

    hl_sims = cosine_similarity(tfidf_v2.transform(headlines), anchor_mat)  # (N_hl, 6)
    sims = (salience_w.reshape(-1, 1) * hl_sims).sum(axis=0, keepdims=True).astype(np.float32)

    x_news_raw = build_news_matrix_from_finbert_and_sims(fb_neu, fb_pos, fb_neg, sims)
    news_col12_raw = float(x_news_raw[0, 12])
    x_news_s = np.clip(scaler_news.transform(x_news_raw), -4.5, 4.5).astype(np.float32)

    diag = {
        "count": len(headlines),
        "fb_neu": float(fb_neu[0]),
        "fb_pos": float(fb_pos[0]),
        "fb_neg": float(fb_neg[0]),
        "fb_net": float(fb_pos[0] - fb_neg[0]),
    }
    return x_news_s, news_col12_raw, diag


# 4. Exact 45-Dim Stationary Macro Builder (Matches Cell 1 & Cell 2)
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stationary_macro_44(expected_cols: tuple[str, ...]) -> tuple[np.ndarray, bool]:
    tickers = ["^DJI", "^GSPC", "^N225", "000001.SS", "^VIX"]
    close_d, high_d, low_d, vol_d = {}, {}, {}, {}

    for sym in tickers:
        try:
            hist = yf.Ticker(sym).history(period="9mo", interval="1d", auto_adjust=True, timeout=8)
            if hist is not None and not hist.empty:
                if getattr(hist.index, "tz", None) is not None:
                    hist.index = hist.index.tz_localize(None)
                hist.index = hist.index.normalize()
                close_d[sym] = hist["Close"]
                high_d[sym] = hist["High"]
                low_d[sym] = hist["Low"]
                vol_d[sym] = hist["Volume"]
        except Exception:
            continue

    if len(close_d) < 2:
        return np.zeros(len(expected_cols), dtype=np.float32), False

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
            mkt_feats[f"{tag}_vol_ratio"] = mkt_feats[f"{tag}_vol_5d"] / (
                mkt_feats[f"{tag}_vol_20d"] + 1e-6
            )
            hl_range = ((high_px[sym] - low_px[sym]) / (close_px[sym] + 1e-6)).shift(1)
            mkt_feats[f"{tag}_hl_range_lag1"] = hl_range
            mkt_feats[f"{tag}_hl_range_ma5"] = hl_range.rolling(5).mean()
            mkt_feats[f"{tag}_open_gap_t"] = 0.0
            vol_s1 = vol_px[sym].shift(1)
            mkt_feats[f"{tag}_vol_z_lag1"] = (vol_s1 - vol_s1.rolling(20).mean()) / (
                vol_s1.rolling(20).std() + 1e-6
            )

    if "^VIX" in close_px.columns:
        vix_c = close_px["^VIX"].shift(1)
        mkt_feats["VIX_lag1"] = vix_c
        mkt_feats["VIX_diff1"] = close_px["^VIX"].diff().shift(1)
        mkt_feats["VIX_ma_rat"] = vix_c / (vix_c.rolling(20).mean() + 1e-6)
        mkt_feats["VIX_open_t"] = 0.0

    for sym, tag in [("^N225", "NIKKEI"), ("000001.SS", "SSE")]:
        if sym in rets.columns:
            r_asia_lag1 = rets[sym].shift(1)
            mkt_feats[f"{tag}_preUS_ret_t"] = 0.0
            mkt_feats[f"{tag}_ret_lag1"] = r_asia_lag1
            mkt_feats[f"{tag}_ret_ma5"] = r_asia_lag1.rolling(5).mean()
            mkt_feats[f"{tag}_vol_10d"] = r_asia_lag1.rolling(10).std()

    start_dt = (close_px.index.min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    for c in ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={c}&cosd={start_dt}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                csv_txt = resp.read().decode("utf-8")
            s = pd.to_numeric(
                pd.read_csv(io.StringIO(csv_txt), index_col=0, parse_dates=True)[c],
                errors="coerce",
            ).ffill()
            aligned = s.reindex(close_px.index).ffill().bfill()
            mkt_feats[f"{c}_lag1"] = aligned.shift(1)
            mkt_feats[f"{c}_chg5"] = aligned.diff(5).shift(1)
        except Exception:
            mkt_feats[f"{c}_lag1"] = 0.0
            mkt_feats[f"{c}_chg5"] = 0.0

    # Strictly align columns to exact training order saved in tfidf_anchor.pkl
    mkt_feats = mkt_feats.reindex(columns=list(expected_cols), fill_value=0.0)
    mkt_feats = mkt_feats.ffill().bfill().fillna(0.0)

    # Exact 60-day rolling z-score normalization from Cell 2
    roll_mean = mkt_feats.rolling(60, min_periods=10).mean().bfill()
    roll_std = mkt_feats.rolling(60, min_periods=10).std().bfill().replace(0, 1.0).fillna(1.0)
    macro_stationary = ((mkt_feats - roll_mean) / (roll_std + 1e-6)).clip(-4.0, 4.0).fillna(0.0)

    return macro_stationary.iloc[-1].values.astype(np.float32), True


def build_macro_tensor(last_44: np.ndarray, news_col12_raw: float) -> np.ndarray:
    x_macro_raw = np.append(last_44, np.float32(news_col12_raw)).reshape(1, -1)
    return np.clip(scaler_macro.transform(x_macro_raw), -4.5, 4.5).astype(np.float32)


# 5. Trained Multimodal Ensemble Forward Pass (Matches Cell 3 `[nn_probs, gbdt_probs]`)
def run_index_ensemble(sym: str, x_macro_s: np.ndarray, x_news_s: np.ndarray) -> list[float]:
    pack = regional_ensembles[sym]
    with torch.inference_mode():
        logits = pack["nn"](
            torch.from_numpy(x_macro_s),
            torch.from_numpy(x_news_s),
        )
        p_nn = F.softmax(logits, dim=-1).cpu().numpy()

    p_gbdt = pack["gbdt"].predict_proba(np.hstack([x_macro_s, x_news_s]))
    p_meta = pack["meta"].predict_proba(np.hstack([p_nn, p_gbdt]))[0]
    return p_meta.tolist()


# 6. Streamlit Sidebar Form & Real-Time Predictions
with st.sidebar.form("scenario_form"):
    st.header("News Scenario Prompt")
    user_prompt = st.text_area(
        "Enter custom headlines or market events (one per line):",
        value="NVIDIA stocks rise a lot, very optimistic",
        height=200,
    )
    submit_btn = st.form_submit_button("Predict Across All Indices")

if submit_btn:
    headlines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
    if not headlines:
        st.warning("Please enter at least one news headline.")
        st.stop()

    with st.spinner("Running FinBERT, fetching live macro state, and executing Multimodal Ensembles..."):
        last_44, live_macro_ok = fetch_stationary_macro_44(tuple(saved_macro_cols))
        x_news_s, news_col12_raw, diag = encode_user_headlines(headlines)
        x_macro_s = build_macro_tensor(last_44, news_col12_raw)

        index_targets = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]
        results = [
            (label, run_index_ensemble(sym, x_macro_s, x_news_s))
            for label, sym in index_targets
        ]

    st.subheader("📊 Independent Regional Index Predictions")
    cols = st.columns(4)
    regime_labels = ["Down (0)", "Neutral (1)", "Up (2)"]

    for col_ui, (idx_name, probs) in zip(cols, results):
        pred_idx = int(np.argmax(probs))
        with col_ui:
            st.markdown(f"### {idx_name}")
            st.metric(
                "Regime",
                regime_labels[pred_idx],
                f"{probs[pred_idx] * 100:.1f}% Confidence",
            )
            prob_df = pd.DataFrame(
                {"Class": ["Down", "Neutral", "Up"], "Probability": probs}
            ).set_index("Class")
            st.bar_chart(prob_df, height=200)

    st.markdown("---")
    macro_status = "Live yfinance + FRED" if live_macro_ok else "Neutral Stationary Baseline"
    st.success(
        f"**Headlines:** `{diag['count']}` | **Macro Feed:** `{macro_status}` | "
        f"**FinBERT (`yiyanghkust/finbert-tone`):** "
        f"`[Neu: {diag['fb_neu']:.3f}, Pos: {diag['fb_pos']:.3f}, Neg: {diag['fb_neg']:.3f}, Net: {diag['fb_net']:+.3f}]`"
    )
else:
    st.info(
        "👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**."
    )
