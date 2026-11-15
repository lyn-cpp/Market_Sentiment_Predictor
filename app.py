import re
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


# 1. PyTorch Architecture Definition (Matches multimodal_market_net.pt)
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


# 2. Load Colab Artifacts
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
    state = torch.load("multimodal_market_net.pt", map_location=torch.device("cpu"))
    pytorch_model.load_state_dict(state)
    pytorch_model.eval()
    return gbdt, meta, s_macro, s_news, tfidf, pytorch_model


gbdt_model, meta_clf, scaler_macro, scaler_news, tfidf_anchor, pytorch_model = load_artifacts()


def get_feature_names(artifact, fallback_prefix: str, dim: int) -> list[str]:
    if hasattr(artifact, "feature_names_in_"):
        return [str(x) for x in artifact.feature_names_in_]
    if hasattr(artifact, "feature_name_"):  # LightGBM native attribute
        return [str(x) for x in artifact.feature_name_]
    return [f"{fallback_prefix}_{i}" for i in range(dim)]


# 3. Dynamic News Vector Builder (Maps directly to scaler_news schema + tfidf_anchor)
def extract_tfidf_signals(headlines: list[str], tfidf_obj) -> dict[str, np.ndarray]:
    """Handles TfidfVectorizer, Pipeline(TFIDF+SVD), or Dict(vectorizer + class anchors)."""
    joined = " ".join(headlines)
    out = {"vec": np.zeros(0, dtype=np.float32), "per_headline": None}

    if isinstance(tfidf_obj, dict):
        # Common Colab pattern: {"vectorizer": tfidf, "anchors": anchor_matrix, ...}
        vec_model = (
            tfidf_obj.get("vectorizer")
            or tfidf_obj.get("tfidf")
            or next((v for v in tfidf_obj.values() if hasattr(v, "transform")), None)
        )
        if vec_model is not None:
            mat = vec_model.transform(headlines)
            joined_mat = vec_model.transform([joined])
            anchors = (
                tfidf_obj.get("anchors")
                or tfidf_obj.get("centroids")
                or tfidf_obj.get("anchor_vecs")
            )
            if anchors is not None:
                anch_arr = anchors.toarray() if hasattr(anchors, "toarray") else np.asarray(anchors)
                sims = (mat @ anch_arr.T)
                sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
                out["per_headline"] = sims.astype(np.float32)
                out["vec"] = np.concatenate(
                    [sims.mean(axis=0), sims.max(axis=0), sims.std(axis=0)]
                ).astype(np.float32)
            else:
                arr = joined_mat.toarray().flatten() if hasattr(joined_mat, "toarray") else np.asarray(joined_mat).flatten()
                out["vec"] = arr.astype(np.float32)
    elif hasattr(tfidf_obj, "transform"):
        joined_mat = tfidf_obj.transform([joined])
        arr = joined_mat.toarray().flatten() if hasattr(joined_mat, "toarray") else np.asarray(joined_mat).flatten()
        out["vec"] = arr.astype(np.float32)
        h_mat = tfidf_obj.transform(headlines)
        out["per_headline"] = (h_mat.toarray() if hasattr(h_mat, "toarray") else np.asarray(h_mat)).astype(np.float32)

    return out


def build_news_dataframe(headlines: list[str]) -> tuple[pd.DataFrame, dict]:
    """Constructs exact 22-column news feature row matched to scaler_news.feature_names_in_."""
    news_dim = getattr(scaler_news, "n_features_in_", 22)
    col_names = get_feature_names(scaler_news, "news", news_dim)

    scores = [sia.polarity_scores(h) for h in headlines]
    comp = np.array([s["compound"] for s in scores], dtype=np.float32)
    pos = np.array([s["pos"] for s in scores], dtype=np.float32)
    neg = np.array([s["neg"] for s in scores], dtype=np.float32)
    neu = np.array([s["neu"] for s in scores], dtype=np.float32)
    lengths = np.array([len(h.split()) for h in headlines], dtype=np.float32)
    char_lens = np.array([len(h) for h in headlines], dtype=np.float32)

    tfidf_info = extract_tfidf_signals(headlines, tfidf_anchor)
    tfidf_vec = tfidf_info["vec"]

    # Precompute candidate statistics commonly engineered in 22-dim news pipelines
    bull_ratio = float(np.mean(comp > 0.05))
    bear_ratio = float(np.mean(comp < -0.05))
    net_breadth = bull_ratio - bear_ratio
    comp_mean = float(np.mean(comp))
    comp_std = float(np.std(comp)) if len(comp) > 1 else abs(comp_mean) * 0.25
    comp_max = float(np.max(comp))
    comp_min = float(np.min(comp))
    comp_skew = float(comp_max + comp_min - 2 * np.median(comp))

    has_real_names = hasattr(scaler_news, "feature_names_in_")
    row = {}

    if has_real_names:
        tfidf_cursor = 0
        for idx, col in enumerate(col_names):
            c = col.lower()
            if any(k in c for k in ["tfidf", "svd", "anchor", "topic", "emb", "sim", "cluster"]):
                # Extract explicit index if present (e.g., tfidf_3 or anchor_2)
                m = re.search(r"(\d+)$", c)
                t_idx = int(m.group(1)) if m and int(m.group(1)) < len(tfidf_vec) else tfidf_cursor
                val = float(tfidf_vec[t_idx]) if t_idx < len(tfidf_vec) else comp_mean
                tfidf_cursor += 1
            elif "comp" in c and "std" in c:
                val = comp_std
            elif "comp" in c and "max" in c:
                val = comp_max
            elif "comp" in c and "min" in c:
                val = comp_min
            elif "comp" in c or ("sent" in c and "std" not in c and "max" not in c and "min" not in c):
                val = comp_mean
            elif "pos" in c and "std" in c:
                val = float(np.std(pos))
            elif "pos" in c and "max" in c:
                val = float(np.max(pos))
            elif "pos" in c or "bull" in c:
                val = bull_ratio if "ratio" in c or "pct" in c or "share" in c else float(np.mean(pos))
            elif "neg" in c and "std" in c:
                val = float(np.std(neg))
            elif "neg" in c and "max" in c:
                val = float(np.max(neg))
            elif "neg" in c or "bear" in c:
                val = bear_ratio if "ratio" in c or "pct" in c or "share" in c else float(np.mean(neg))
            elif "neu" in c:
                val = float(np.mean(neu))
            elif "std" in c or "disp" in c or "vol" in c:
                val = comp_std
            elif "max" in c:
                val = comp_max
            elif "min" in c:
                val = comp_min
            elif "skew" in c or "asym" in c:
                val = comp_skew
            elif "net" in c or "diff" in c or "spread" in c or "pol" in c:
                val = net_breadth
            elif "word" in c or "len" in c:
                val = float(np.mean(lengths)) if "word" in c else float(np.mean(char_lens))
            elif "count" in c or "num" in c:
                val = float(len(headlines))
            else:
                # Fallback to training mean shifted by sentiment if column name is opaque
                base_m = float(scaler_news.mean_[idx]) if hasattr(scaler_news, "mean_") else 0.0
                base_s = float(scaler_news.scale_[idx]) if hasattr(scaler_news, "scale_") else 1.0
                val = base_m + comp_mean * base_s
            row[col] = val
    else:
        # No feature names saved in scaler_news: combine VADER stats FIRST + compact TF-IDF projection
        core_vader = [
            comp_mean,
            comp_std,
            comp_max,
            comp_min,
            float(np.mean(pos)),
            float(np.mean(neg)),
            float(np.mean(neu)),
            bull_ratio,
            bear_ratio,
            net_breadth,
            float(np.mean(lengths)),
            float(len(headlines)),
        ]
        remaining = max(0, news_dim - len(core_vader))
        if len(tfidf_vec) >= remaining and remaining > 0:
            # Bin/average across the entire TF-IDF vector instead of slicing the first N zeros!
            chunks = np.array_split(tfidf_vec, remaining)
            tfidf_compact = [float(np.mean(ch)) if len(ch) else 0.0 for ch in chunks]
        else:
            tfidf_compact = list(tfidf_vec) + [0.0] * max(0, remaining - len(tfidf_vec))
        combined = (core_vader + tfidf_compact)[:news_dim]
        row = {col_names[i]: combined[i] for i in range(news_dim)}

    df_news = pd.DataFrame([row], columns=col_names)
    meta_stats = {
        "compound_mean": comp_mean,
        "pos_mean": float(np.mean(pos)),
        "neg_mean": float(np.mean(neg)),
        "tfidf_nonzero": int(np.count_nonzero(tfidf_vec)),
        "tfidf_dim": int(len(tfidf_vec)),
    }
    return df_news, meta_stats


# 4. Live Macro Feature Builder (Maps directly to scaler_macro schema + Regional Index)
@st.cache_data(ttl=900)
def fetch_market_history() -> pd.DataFrame:
    tickers = [
        "^DJI", "^GSPC", "^IXIC", "^RUT", "^N225", "000001.SS", "^HSI", "^FTSE", "^GDAXI",
        "^VIX", "^TNX", "^IRX", "^FVX", "DX-Y.NYB", "GC=F", "CL=F", "JPY=X", "CNY=X", "EURUSD=X",
    ]
    data = yf.download(tickers, period="6mo", interval="1d", progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.ffill().bfill()


def build_macro_dataframe(target_ticker: str) -> pd.DataFrame:
    macro_dim = getattr(scaler_macro, "n_features_in_", 45)
    col_names = get_feature_names(scaler_macro, "macro", macro_dim)
    hist = fetch_market_history()

    s_target = hist[target_ticker].dropna() if target_ticker in hist.columns else pd.Series(dtype=float)

    def series_stats(s: pd.Series) -> dict[str, float]:
        if len(s) < 25:
            return {"r1": 0.0, "r5": 0.0, "r20": 0.0, "v5": 0.01, "v20": 0.01, "ma5": 0.0, "ma20": 0.0, "rsi": 50.0, "lvl": 0.0}
        ret = s.pct_change()
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / (loss + 1e-9)
        return {
            "r1": float(ret.iloc[-1]),
            "r5": float(s.pct_change(5).iloc[-1]),
            "r20": float(s.pct_change(20).iloc[-1]),
            "v5": float(ret.rolling(5).std().iloc[-1]),
            "v20": float(ret.rolling(20).std().iloc[-1]),
            "ma5": float((s.iloc[-1] / s.rolling(5).mean().iloc[-1]) - 1.0),
            "ma20": float((s.iloc[-1] / s.rolling(20).mean().iloc[-1]) - 1.0),
            "rsi": float(100.0 - (100.0 / (1.0 + rs))),
            "lvl": float(s.iloc[-1]),
        }

    tgt = series_stats(s_target)
    vix = series_stats(hist["^VIX"].dropna()) if "^VIX" in hist else tgt
    tnx = series_stats(hist["^TNX"].dropna()) if "^TNX" in hist else tgt
    dxy = series_stats(hist["DX-Y.NYB"].dropna()) if "DX-Y.NYB" in hist else tgt
    gold = series_stats(hist["GC=F"].dropna()) if "GC=F" in hist else tgt
    oil = series_stats(hist["CL=F"].dropna()) if "CL=F" in hist else tgt

    # Start from training mean so any unmapped macro indicator stays cleanly in-distribution (z-score = 0)
    base_means = (
        scaler_macro.mean_.copy().astype(np.float64)
        if hasattr(scaler_macro, "mean_") and len(scaler_macro.mean_) == macro_dim
        else np.zeros(macro_dim, dtype=np.float64)
    )

    if hasattr(scaler_macro, "feature_names_in_"):
        row = {}
        for i, col in enumerate(col_names):
            c = col.lower()
            src = tgt
            if "vix" in c:
                src = vix
            elif any(k in c for k in ["tnx", "10y", "yield", "rate", "bond"]):
                src = tnx
            elif any(k in c for k in ["dxy", "usd", "dollar"]):
                src = dxy
            elif "gold" in c or "gc" in c:
                src = gold
            elif "oil" in c or "cl" in c or "wti" in c:
                src = oil
            elif "dji" in c and "^DJI" in hist:
                src = series_stats(hist["^DJI"].dropna())
            elif ("gspc" in c or "sp500" in c or "spx" in c) and "^GSPC" in hist:
                src = series_stats(hist["^GSPC"].dropna())
            elif ("n225" in c or "nikkei" in c) and "^N225" in hist:
                src = series_stats(hist["^N225"].dropna())
            elif ("sse" in c or "000001" in c or "china" in c) and "000001.SS" in hist:
                src = series_stats(hist["000001.SS"].dropna())

            if "rsi" in c:
                val = src["rsi"]
            elif "vol" in c and "20" in c:
                val = src["v20"]
            elif "vol" in c:
                val = src["v5"]
            elif "ma" in c and "20" in c:
                val = src["ma20"]
            elif "ma" in c:
                val = src["ma5"]
            elif "20" in c:
                val = src["r20"]
            elif "5" in c:
                val = src["r5"]
            elif any(k in c for k in ["ret", "chg", "roc", "1d", "return"]):
                val = src["r1"]
            elif "vix" in c or "tnx" in c:
                val = src["lvl"]
            else:
                val = float(base_means[i])
            row[col] = val
        return pd.DataFrame([row], columns=col_names).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Fallback if scaler_macro has no saved column names: populate first slots with index + cross-asset stats
    raw_feats = [
        tgt["r1"], tgt["r5"], tgt["r20"], tgt["v5"], tgt["v20"], tgt["ma5"], tgt["ma20"], tgt["rsi"],
        vix["lvl"], vix["r1"], vix["r5"], tnx["lvl"], tnx["r1"], tnx["r5"],
        dxy["r1"], dxy["r5"], gold["r1"], gold["r5"], oil["r1"], oil["r5"],
    ]
    vec = base_means.copy()
    for idx, val in enumerate(raw_feats[:macro_dim]):
        if np.isfinite(val):
            vec[idx] = val
    return pd.DataFrame([vec], columns=col_names)


# 5. True Ensemble Forward Pass (PyTorch + GBDT -> Meta-Stacker)
def run_ensemble_inference(
    df_macro: pd.DataFrame, df_news: pd.DataFrame, blend_mode: str = "Full Stacked Ensemble"
) -> dict:
    x_m_scaled = scaler_macro.transform(df_macro).astype(np.float32)
    x_n_scaled = scaler_news.transform(df_news).astype(np.float32)

    # Clip extreme out-of-distribution z-scores so single-headline scaling doesn't saturate LayerNorm/Trees
    x_m_scaled = np.clip(np.nan_to_num(x_m_scaled, nan=0.0), -5.0, 5.0)
    x_n_scaled = np.clip(np.nan_to_num(x_n_scaled, nan=0.0), -5.0, 5.0)

    # 1) PyTorch MultimodalMarketNet Forward Pass
    with torch.no_grad():
        t_m = torch.tensor(x_m_scaled, dtype=torch.float32)
        t_n = torch.tensor(x_n_scaled, dtype=torch.float32)
        nn_logits = pytorch_model(t_m, t_n)
        nn_probs = F.softmax(nn_logits, dim=-1).cpu().numpy()[0]

    # 2) GBDT Forward Pass (Align exact feature names if stored in gbdt_model)
    combined_scaled_df = pd.DataFrame(
        np.hstack([x_m_scaled, x_n_scaled]),
        columns=list(df_macro.columns) + list(df_news.columns),
    )
    combined_raw_df = pd.concat([df_macro, df_news], axis=1)

    gbdt_names = getattr(gbdt_model, "feature_names_in_", getattr(gbdt_model, "feature_name_", None))
    gbdt_dim = getattr(gbdt_model, "n_features_in_", combined_scaled_df.shape[1])

    if gbdt_names is not None and all(c in combined_scaled_df.columns for c in gbdt_names):
        gbdt_in = combined_scaled_df[list(gbdt_names)]
    elif gbdt_dim == x_m_scaled.shape[1]:
        gbdt_in = x_m_scaled
    elif gbdt_dim == x_n_scaled.shape[1]:
        gbdt_in = x_n_scaled
    else:
        gbdt_in = combined_scaled_df.iloc[:, :gbdt_dim].values

    gbdt_probs = gbdt_model.predict_proba(gbdt_in)[0]

    # 3) Meta-Stacker Classifier Forward Pass
    # Inspect meta_clf feature names to detect exact stacking order ([nn, gbdt] vs [gbdt, nn])
    meta_names = [str(x).lower() for x in getattr(meta_clf, "feature_names_in_", [])]
    meta_dim = getattr(meta_clf, "n_features_in_", 6)

    if meta_names and ("nn" in meta_names[0] or "torch" in meta_names[0] or "dl" in meta_names[0]):
        base_stack = np.concatenate([nn_probs, gbdt_probs]).reshape(1, -1)
    else:
        base_stack = np.concatenate([gbdt_probs, nn_probs]).reshape(1, -1)

    if meta_dim != base_stack.shape[1]:
        extended = np.hstack([base_stack, x_m_scaled, x_n_scaled])
        meta_in = extended[:, :meta_dim]
    else:
        meta_in = base_stack

    meta_probs = meta_clf.predict_proba(meta_in)[0]

    if blend_mode == "PyTorch MultimodalNet Only":
        final_probs = nn_probs
    elif blend_mode == "GBDT Only":
        final_probs = gbdt_probs
    else:
        final_probs = meta_probs

    return {
        "final_probs": final_probs.tolist(),
        "nn_probs": nn_probs.tolist(),
        "gbdt_probs": gbdt_probs.tolist(),
        "meta_probs": meta_probs.tolist(),
        "news_z_norm": float(np.linalg.norm(x_n_scaled)),
        "macro_z_norm": float(np.linalg.norm(x_m_scaled)),
        "x_news_scaled": x_n_scaled[0],
    }


# 6. Sidebar UI
st.sidebar.header("News Scenario Prompt")
user_prompt = st.sidebar.text_area(
    "Enter custom headlines or market events (one per line):",
    value=(
        "Federal Reserve announces unexpected interest rate cut amid rising inflation pressures.\n"
        "China reports stronger than expected manufacturing GDP growth."
    ),
    height=180,
)

inference_mode = st.sidebar.radio(
    "Model Output Layer:",
    ["Full Stacked Ensemble", "PyTorch MultimodalNet Only", "GBDT Only"],
    help="Compare the Meta-Stacker against individual base models trained in Colab.",
)

predict_btn = st.sidebar.button("Predict Across All Indices")

if predict_btn:
    raw_headlines = [h.strip() for h in user_prompt.split("\n") if h.strip()]
    if not raw_headlines:
        st.warning("Please enter at least one headline in the sidebar.")
        st.stop()

    with st.spinner("Vectorizing headlines via TF-IDF + VADER, pulling live macro data, and running Colab models..."):
        df_news, news_stats = build_news_dataframe(raw_headlines)

        index_map = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]

        results = []
        for display_name, ticker in index_map:
            df_macro = build_macro_dataframe(ticker)
            out = run_ensemble_inference(df_macro, df_news, blend_mode=inference_mode)
            results.append((display_name, ticker, out))

    st.subheader(f"📊 Regional Index Predictions ({inference_mode})")
    cols = st.columns(4)
    regime_names = ["Down (0)", "Neutral (1)", "Up (2)"]

    for i, (idx_name, ticker, out) in enumerate(results):
        probs = out["final_probs"]
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
            st.caption(
                f"PyTorch Up: `{out['nn_probs'][2]:.1%}` | GBDT Up: `{out['gbdt_probs'][2]:.1%}`"
            )

    st.markdown("---")
    # Live Verification Panel: Proves the trained artifacts are executing and shows exact feature schemas
    with st.expander("🔍 Live Model & Artifact Diagnostics (Verify Trained Weights & Schemas)", expanded=True):
        st.markdown(
            f"**Input Headlines Processed:** `{len(raw_headlines)}` | "
            f"**VADER Mean Compound:** `{news_stats['compound_mean']:.3f}` | "
            f"**TF-IDF Vector Dim:** `{news_stats['tfidf_dim']}` (`{news_stats['tfidf_nonzero']}` active terms) | "
            f"**Scaled News L2 Norm:** `{results[0][2]['news_z_norm']:.3f}`"
        )

        diag_rows = []
        for idx_name, _, out in results:
            diag_rows.append(
                {
                    "Index": idx_name,
                    "PyTorch [Down, Neu, Up]": [round(x, 4) for x in out["nn_probs"]],
                    "GBDT [Down, Neu, Up]": [round(x, 4) for x in out["gbdt_probs"]],
                    "Meta-Stacker [Down, Neu, Up]": [round(x, 4) for x in out["meta_probs"]],
                }
            )
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**`scaler_news.pkl` Expected Columns & Current Scaled Values:**")
            news_debug = pd.DataFrame(
                {
                    "Feature": df_news.columns,
                    "Raw Value": df_news.iloc[0].values,
                    "Scaled Z-Score": results[0][2]["x_news_scaled"],
                }
            )
            st.dataframe(news_debug, height=220, use_container_width=True)
        with c2:
            st.markdown("**Artifact Shapes Loaded from Repo:**")
            st.json(
                {
                    "tfidf_anchor_type": type(tfidf_anchor).__name__,
                    "scaler_macro_n_features": getattr(scaler_macro, "n_features_in_", "unknown"),
                    "scaler_macro_has_names": hasattr(scaler_macro, "feature_names_in_"),
                    "scaler_news_n_features": getattr(scaler_news, "n_features_in_", "unknown"),
                    "scaler_news_has_names": hasattr(scaler_news, "feature_names_in_"),
                    "gbdt_type": type(gbdt_model).__name__,
                    "gbdt_n_features": getattr(gbdt_model, "n_features_in_", "unknown"),
                    "meta_clf_type": type(meta_clf).__name__,
                    "meta_clf_coef_or_features": getattr(meta_clf, "n_features_in_", "unknown"),
                }
            )
else:
    st.info("👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**.")
