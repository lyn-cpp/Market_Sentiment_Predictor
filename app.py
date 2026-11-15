import os
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
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "real-time **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)


# 1. Two-Tower Cross-Attention Architecture (Exact match to kaggle.ipynb)
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


# 2. Load Trained Model Artifacts & FinBERT
@st.cache_resource
def load_finbert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None


@st.cache_resource
def load_artifacts():
    gbdt = joblib.load("gbdt_model.pkl")
    meta = joblib.load("meta_stacker_clf.pkl")
    s_macro = joblib.load("scaler_macro.pkl")
    s_news = joblib.load("scaler_news.pkl")
    tfidf_raw = joblib.load("tfidf_anchor.pkl")
    reg_heads = joblib.load("regional_heads.pkl") if os.path.exists("regional_heads.pkl") else None

    macro_dim = int(getattr(s_macro, "n_features_in_", 45))
    news_dim = int(getattr(s_news, "n_features_in_", 22))

    pytorch_model = MultimodalMarketNet(macro_dim=macro_dim, news_dim=news_dim, hidden_dim=64)
    state_dict = torch.load("multimodal_market_net.pt", map_location=torch.device("cpu"))
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    return gbdt, meta, s_macro, s_news, tfidf_raw, reg_heads, pytorch_model


finbert_tok, finbert_mdl = load_finbert()
gbdt_model, meta_clf, scaler_macro, scaler_news, tfidf_obj, regional_heads, pytorch_model = load_artifacts()


# 3. Exact 22-Feature News Pipeline (FinBERT + VADER + TF-IDF Class Anchors + Topic Shares)
def score_finbert(headlines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns arrays of (p_pos, p_neg, p_neu) for each headline using ProsusAI/finbert."""
    if finbert_tok is not None and finbert_mdl is not None:
        inputs = finbert_tok(
            headlines, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits = finbert_mdl(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        # ProsusAI/finbert label mapping: {0: positive, 1: negative, 2: neutral}
        id2label = getattr(finbert_mdl.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})
        pos_idx = next((k for k, v in id2label.items() if "pos" in v.lower()), 0)
        neg_idx = next((k for k, v in id2label.items() if "neg" in v.lower()), 1)
        neu_idx = next((k for k, v in id2label.items() if "neu" in v.lower()), 2)
        return probs[:, pos_idx], probs[:, neg_idx], probs[:, neu_idx]

    # Graceful fallback to calibrated VADER probabilities only if `transformers` isn't installed
    vs = [sia.polarity_scores(h) for h in headlines]
    p_pos = np.array([v["pos"] for v in vs], dtype=np.float32)
    p_neg = np.array([v["neg"] for v in vs], dtype=np.float32)
    p_neu = np.array([v["neu"] for v in vs], dtype=np.float32)
    return p_pos, p_neg, p_neu


def build_exact_22d_news_features(headlines: list[str]) -> tuple[np.ndarray, dict]:
    """Builds the exact 22 news features engineered in Part 2 of kaggle.ipynb."""
    news_dim = int(getattr(scaler_news, "n_features_in_", 22))

    # A. FinBERT Headline Probabilities (9 features)
    fb_pos, fb_neg, fb_neu = score_finbert(headlines)
    fb_net = fb_pos - fb_neg

    fb_pos_mean = float(np.mean(fb_pos))
    fb_neg_mean = float(np.mean(fb_neg))
    fb_neu_mean = float(np.mean(fb_neu))
    fb_net_mean = float(np.mean(fb_net))
    fb_net_std = float(np.std(fb_net)) if len(fb_net) > 1 else 0.0
    fb_pos_max = float(np.max(fb_pos))
    fb_neg_max = float(np.max(fb_neg))
    fb_bull_ratio = float(np.mean(fb_net > 0.10))
    fb_bear_ratio = float(np.mean(fb_net < -0.10))

    # B. VADER Lexicon Features (7 features)
    v_scores = [sia.polarity_scores(h) for h in headlines]
    v_comp = np.array([s["compound"] for s in v_scores], dtype=np.float32)
    v_pos = np.array([s["pos"] for s in v_scores], dtype=np.float32)
    v_neg = np.array([s["neg"] for s in v_scores], dtype=np.float32)

    vader_comp_mean = float(np.mean(v_comp))
    vader_comp_std = float(np.std(v_comp)) if len(v_comp) > 1 else 0.0
    vader_comp_min = float(np.min(v_comp))
    vader_comp_max = float(np.max(v_comp))
    vader_pos_mean = float(np.mean(v_pos))
    vader_neg_mean = float(np.mean(v_neg))
    vader_net = float(np.mean(v_comp > 0.05) - np.mean(v_comp < -0.05))

    # C. TF-IDF Class-Anchor Cosine Similarity Features (4 features)
    joined_doc = " ".join(headlines)
    if isinstance(tfidf_obj, dict) and "vectorizer" in tfidf_obj:
        vec = tfidf_obj["vectorizer"]
        x_tf = vec.transform([joined_doc])
        c0 = np.asarray(tfidf_obj["c0"]).reshape(1, -1)
        c1 = np.asarray(tfidf_obj["c1"]).reshape(1, -1)
        c2 = np.asarray(tfidf_obj["c2"]).reshape(1, -1)
        sim_down = float(cosine_similarity(x_tf, c0)[0, 0])
        sim_neu = float(cosine_similarity(x_tf, c1)[0, 0])
        sim_up = float(cosine_similarity(x_tf, c2)[0, 0])
        sim_spread = sim_up - sim_down
        saved_news_cols = tfidf_obj.get("news_cols", [])
    elif hasattr(tfidf_obj, "transform"):
        # Backward compatibility if old tfidf_anchor.pkl (raw TfidfVectorizer) is still on disk:
        # Score bullish vs bearish vocabulary anchors directly inside the fitted TF-IDF space
        x_tf = tfidf_obj.transform([joined_doc])
        bull_anchor = tfidf_obj.transform(
            ["surge rally growth cut recovery boom beat profit record stimulus gain"]
        )
        bear_anchor = tfidf_obj.transform(
            ["crash recession crisis default inflation war plunge drop fear loss Cut"]
        )
        neu_anchor = tfidf_obj.transform(
            ["steady unchanged hold report meeting session market index"]
        )
        sim_down = float(cosine_similarity(x_tf, bear_anchor)[0, 0])
        sim_neu = float(cosine_similarity(x_tf, neu_anchor)[0, 0])
        sim_up = float(cosine_similarity(x_tf, bull_anchor)[0, 0])
        sim_spread = sim_up - sim_down
        saved_news_cols = []
    else:
        sim_down, sim_neu, sim_up, sim_spread = 0.0, 0.0, 0.0, 0.0
        saved_news_cols = []

    # D. Macro / Geopolitical Keyword Intensity Features (2 features -> Total = 22)
    low_doc = joined_doc.lower()
    macro_kw = sum(
        k in low_doc
        for k in ["fed", "rate", "inflation", "gdp", "jobs", "cpi", "central bank", "stimulus", "yield"]
    ) / max(len(headlines), 1)
    risk_kw = sum(
        k in low_doc
        for k in ["war", "crisis", "crash", "sanction", "default", "recession", "tariff", "bankrupt"]
    ) / max(len(headlines), 1)

    feature_lookup = {
        "fb_pos_mean": fb_pos_mean,
        "fb_neg_mean": fb_neg_mean,
        "fb_neu_mean": fb_neu_mean,
        "fb_net_mean": fb_net_mean,
        "fb_net_std": fb_net_std,
        "fb_pos_max": fb_pos_max,
        "fb_neg_max": fb_neg_max,
        "fb_bull_ratio": fb_bull_ratio,
        "fb_bear_ratio": fb_bear_ratio,
        "vader_comp_mean": vader_comp_mean,
        "vader_comp_std": vader_comp_std,
        "vader_comp_min": vader_comp_min,
        "vader_comp_max": vader_comp_max,
        "vader_pos_mean": vader_pos_mean,
        "vader_neg_mean": vader_neg_mean,
        "vader_net": vader_net,
        "tfidf_sim_down": sim_down,
        "tfidf_sim_neu": sim_neu,
        "tfidf_sim_up": sim_up,
        "tfidf_sim_spread": sim_spread,
        "macro_kw_intensity": float(macro_kw),
        "risk_kw_intensity": float(risk_kw),
    }

    default_22 = list(feature_lookup.values())

    # Map by exact column name if present in scaler_news or tfidf_bundle
    target_cols = list(getattr(scaler_news, "feature_names_in_", [])) or saved_news_cols
    if target_cols and len(target_cols) == news_dim:
        vec = []
        for idx, c in enumerate(target_cols):
            cl = c.lower()
            if c in feature_lookup:
                vec.append(feature_lookup[c])
            elif "finbert" in cl or cl.startswith("fb_"):
                if "pos" in cl and "max" in cl:
                    vec.append(fb_pos_max)
                elif "neg" in cl and "max" in cl:
                    vec.append(fb_neg_max)
                elif "pos" in cl:
                    vec.append(fb_pos_mean)
                elif "neg" in cl:
                    vec.append(fb_neg_mean)
                elif "neu" in cl:
                    vec.append(fb_neu_mean)
                elif "std" in cl:
                    vec.append(fb_net_std)
                elif "bull" in cl:
                    vec.append(fb_bull_ratio)
                elif "bear" in cl:
                    vec.append(fb_bear_ratio)
                else:
                    vec.append(fb_net_mean)
            elif "vader" in cl:
                if "std" in cl:
                    vec.append(vader_comp_std)
                elif "min" in cl:
                    vec.append(vader_comp_min)
                elif "max" in cl:
                    vec.append(vader_comp_max)
                elif "pos" in cl:
                    vec.append(vader_pos_mean)
                elif "neg" in cl:
                    vec.append(vader_neg_mean)
                elif "net" in cl or "diff" in cl:
                    vec.append(vader_net)
                else:
                    vec.append(vader_comp_mean)
            elif any(k in cl for k in ["tfidf", "anchor", "sim", "cent"]):
                if "down" in cl or "0" in cl or "bear" in cl:
                    vec.append(sim_down)
                elif "up" in cl or "2" in cl or "bull" in cl:
                    vec.append(sim_up)
                elif "neu" in cl or "1" in cl:
                    vec.append(sim_neu)
                else:
                    vec.append(sim_spread)
            else:
                vec.append(default_22[idx % len(default_22)])
        raw_arr = np.array(vec, dtype=np.float32).reshape(1, -1)
    else:
        raw_arr = np.array(default_22[:news_dim], dtype=np.float32).reshape(1, -1)
        if raw_arr.shape[1] < news_dim:
            raw_arr = np.pad(raw_arr, ((0, 0), (0, news_dim - raw_arr.shape[1])))

    if hasattr(scaler_news, "feature_names_in_"):
        raw_in = pd.DataFrame(raw_arr, columns=scaler_news.feature_names_in_)
    else:
        raw_in = raw_arr

    x_news_s = scaler_news.transform(raw_in).astype(np.float32)
    x_news_s = np.clip(np.nan_to_num(x_news_s, nan=0.0), -4.5, 4.5)

    diagnostics = {
        "finbert_active": finbert_mdl is not None,
        "fb_net": fb_net_mean,
        "vader_comp": vader_comp_mean,
        "tfidf_spread": sim_spread,
    }
    return x_news_s, diagnostics


# 4. Exact 45-Feature Live Market + FRED Pipeline (Matches Part 1 of kaggle.ipynb)
@st.cache_data(ttl=900)
def fetch_live_macro_and_market() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tickers = ["^DJI", "^GSPC", "^N225", "000001.SS", "^VIX"]
    mkt = {}
    for sym in tickers:
        try:
            h = yf.Ticker(sym).history(period="6mo", interval="1d", auto_adjust=True)
            if h is not None and not h.empty:
                if getattr(h.index, "tz", None) is not None:
                    h.index = h.index.tz_localize(None)
                h.index = h.index.normalize()
                mkt[sym] = h[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill()
        except Exception:
            pass

    # Fetch exact 5 FRED series from Part 1 directly from St. Louis Fed CSV endpoints (zero API key needed)
    fred_codes = ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]
    start_dt = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fred_dict = {}
    for code in fred_codes:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start_dt}"
            s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
            fred_dict[code] = pd.to_numeric(s, errors="coerce").ffill()
        except Exception:
            fred_dict[code] = pd.Series(dtype=float)

    fred_df = pd.DataFrame(fred_dict).ffill().bfill()
    return mkt, fred_df


def build_exact_45d_macro_features(
    mkt: dict[str, pd.DataFrame], fred_df: pd.DataFrame, focus_sym: str
) -> np.ndarray:
    """
    Constructs the exact 45 market + FRED macro features from Part 1 of kaggle.ipynb.
    When evaluating a non-DJI regional index (if regional_heads.pkl is not yet uploaded),
    adapts the primary equity block to `focus_sym` so each index reflects its own live tape.
    """
    macro_dim = int(getattr(scaler_macro, "n_features_in_", 45))
    feat_map = {}

    # Determine primary & secondary equity tickers
    primary_sym = "^DJI" if regional_heads is not None else focus_sym
    secondary_sym = "^GSPC" if primary_sym != "^GSPC" else "^DJI"

    for sym, tag in [(primary_sym, "DJI"), (secondary_sym, "SPX")]:
        df_s = mkt.get(sym)
        if df_s is not None and len(df_s) >= 25:
            c = df_s["Close"]
            h = df_s["High"]
            l = df_s["Low"]
            o = df_s["Open"]
            v = df_s["Volume"].replace(0, np.nan).ffill().fillna(1.0)
            r = c.pct_change()

            r_lag1 = r.shift(1)
            vol5 = float(r_lag1.rolling(5).std().iloc[-1])
            vol20 = float(r_lag1.rolling(20).std().iloc[-1])
            hl_range = (h - l) / (c + 1e-6)
            vol_s1 = v.shift(1)

            feat_map[f"{tag}_ret_lag1"] = float(r.iloc[-2])
            feat_map[f"{tag}_ret_lag2"] = float(r.iloc[-3])
            feat_map[f"{tag}_ret_ma5"] = float(r_lag1.rolling(5).mean().iloc[-1])
            feat_map[f"{tag}_ret_ma20"] = float(r_lag1.rolling(20).mean().iloc[-1])
            feat_map[f"{tag}_vol_5d"] = vol5
            feat_map[f"{tag}_vol_20d"] = vol20
            feat_map[f"{tag}_vol_ratio"] = vol5 / (vol20 + 1e-6)
            feat_map[f"{tag}_hl_range_lag1"] = float(hl_range.iloc[-2])
            feat_map[f"{tag}_hl_range_ma5"] = float(hl_range.shift(1).rolling(5).mean().iloc[-1])
            feat_map[f"{tag}_open_gap_t"] = float((o.iloc[-1] / (c.iloc[-2] + 1e-6)) - 1.0)
            feat_map[f"{tag}_vol_z_lag1"] = float(
                (vol_s1.iloc[-1] - vol_s1.rolling(20).mean().iloc[-1])
                / (vol_s1.rolling(20).std().iloc[-1] + 1e-6)
            )
        else:
            for sfx in [
                "ret_lag1", "ret_lag2", "ret_ma5", "ret_ma20", "vol_5d", "vol_20d",
                "vol_ratio", "hl_range_lag1", "hl_range_ma5", "open_gap_t", "vol_z_lag1",
            ]:
                feat_map[f"{tag}_{sfx}"] = 0.0

    # VIX features (4 features -> running total: 26)
    vix_df = mkt.get("^VIX")
    if vix_df is not None and len(vix_df) >= 25:
        vc = vix_df["Close"]
        vo = vix_df["Open"]
        feat_map["VIX_lag1"] = float(vc.iloc[-2])
        feat_map["VIX_diff1"] = float(vc.diff().iloc[-2])
        feat_map["VIX_ma_rat"] = float(vc.iloc[-2] / (vc.shift(1).rolling(20).mean().iloc[-1] + 1e-6))
        feat_map["VIX_open_t"] = float(vo.iloc[-1] / (vc.iloc[-2] + 1e-6) - 1.0)
    else:
        feat_map.update({"VIX_lag1": 18.0, "VIX_diff1": 0.0, "VIX_ma_rat": 1.0, "VIX_open_t": 0.0})

    # Pre-Market Asian Session Returns: Nikkei & SSE (8 features -> running total: 34)
    for sym, tag in [("^N225", "NIKKEI"), ("000001.SS", "SSE")]:
        adf = mkt.get(sym)
        if adf is not None and len(adf) >= 15:
            ar = adf["Close"].pct_change()
            feat_map[f"{tag}_preUS_ret_t"] = float(ar.iloc[-1])
            feat_map[f"{tag}_ret_lag1"] = float(ar.iloc[-2])
            feat_map[f"{tag}_ret_ma5"] = float(ar.rolling(5).mean().iloc[-1])
            feat_map[f"{tag}_vol_10d"] = float(ar.rolling(10).std().iloc[-1])
        else:
            feat_map.update(
                {f"{tag}_preUS_ret_t": 0.0, f"{tag}_ret_lag1": 0.0, f"{tag}_ret_ma5": 0.0, f"{tag}_vol_10d": 0.01}
            )

    # FRED Macroeconomic Series: 5 codes x 2 (lag1, chg5) = 10 features -> running total: 44
    for code in ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]:
        s = fred_df[code].dropna() if code in fred_df.columns else pd.Series(dtype=float)
        if len(s) >= 7:
            feat_map[f"{code}_lag1"] = float(s.iloc[-2])
            feat_map[f"{code}_chg5"] = float(s.iloc[-2] - s.iloc[-7])
        else:
            feat_map[f"{code}_lag1"] = 0.0
            feat_map[f"{code}_chg5"] = 0.0

    # 45th feature: DJI_RollVol20 (or calendar day-of-week if named in macro_cols)
    dji_df = mkt.get(primary_sym)
    feat_map["DJI_RollVol20"] = (
        float(dji_df["Close"].pct_change().shift(1).rolling(20).std().iloc[-1])
        if dji_df is not None and len(dji_df) >= 25
        else 0.01
    )

    saved_macro_cols = (
        list(getattr(scaler_macro, "feature_names_in_", []))
        or (tfidf_obj.get("macro_cols", []) if isinstance(tfidf_obj, dict) else [])
    )

    if saved_macro_cols and len(saved_macro_cols) == macro_dim:
        row = []
        for idx, c in enumerate(saved_macro_cols):
            if c in feat_map and np.isfinite(feat_map[c]):
                row.append(feat_map[c])
            elif hasattr(scaler_macro, "mean_"):
                row.append(float(scaler_macro.mean_[idx]))
            else:
                row.append(0.0)
        raw_arr = np.array(row, dtype=np.float32).reshape(1, -1)
    else:
        ordered_vals = list(feat_map.values())
        base = (
            scaler_macro.mean_.copy().astype(np.float32)
            if hasattr(scaler_macro, "mean_") and len(scaler_macro.mean_) == macro_dim
            else np.zeros(macro_dim, dtype=np.float32)
        )
        for i, val in enumerate(ordered_vals[:macro_dim]):
            if np.isfinite(val):
                base[i] = float(val)
        raw_arr = base.reshape(1, -1)

    if hasattr(scaler_macro, "feature_names_in_"):
        raw_in = pd.DataFrame(raw_arr, columns=scaler_macro.feature_names_in_)
    else:
        raw_in = raw_arr

    x_macro_s = scaler_macro.transform(raw_in).astype(np.float32)
    return np.clip(np.nan_to_num(x_macro_s, nan=0.0), -4.5, 4.5)


# 5. Real-Time Forward Pass Through Two-Tower MultimodalMarketNet + GBDT + Meta-Stacker
def predict_for_index(ticker: str, x_macro_s: np.ndarray, x_news_s: np.ndarray) -> list[float]:
    # A. PyTorch Two-Tower Cross-Attention Forward Pass
    with torch.no_grad():
        t_m = torch.tensor(x_macro_s, dtype=torch.float32)
        t_n = torch.tensor(x_news_s, dtype=torch.float32)
        nn_logits = pytorch_model(t_m, t_n)
        nn_probs = F.softmax(nn_logits, dim=-1).cpu().numpy()  # (1, 3)

    # B. GBDT Base Learner Forward Pass
    x_full = np.hstack([x_macro_s, x_news_s])
    gbdt_dim = int(getattr(gbdt_model, "n_features_in_", x_full.shape[1]))
    gbdt_in = x_full[:, :gbdt_dim]
    if hasattr(gbdt_model, "feature_names_in_"):
        gbdt_in = pd.DataFrame(gbdt_in, columns=gbdt_model.feature_names_in_)
    gbdt_probs = gbdt_model.predict_proba(gbdt_in)  # (1, 3)

    # C. If trained per-index regional heads exist (from updated Cell [35]), use the exact index head
    if isinstance(regional_heads, dict) and ticker in regional_heads:
        reg_in = np.hstack([nn_probs, gbdt_probs, x_news_s, x_macro_s])
        return regional_heads[ticker].predict_proba(reg_in)[0].tolist()

    # D. Meta-Stacker Classifier Forward Pass ([gbdt_probs, nn_probs])
    stacked = np.hstack([gbdt_probs, nn_probs])
    meta_dim = int(getattr(meta_clf, "n_features_in_", stacked.shape[1]))
    if meta_dim == stacked.shape[1]:
        meta_in = stacked
    elif meta_dim < stacked.shape[1]:
        meta_in = stacked[:, :meta_dim]
    else:
        ext = np.hstack([stacked, x_full])
        meta_in = ext[:, :meta_dim]

    if hasattr(meta_clf, "feature_names_in_"):
        meta_in = pd.DataFrame(meta_in, columns=meta_clf.feature_names_in_)

    return meta_clf.predict_proba(meta_in)[0].tolist()


# 6. Sidebar User Input & Real-Time Execution
st.sidebar.header("News Scenario Prompt")
user_prompt = st.sidebar.text_area(
    "Enter custom headlines or market events (one per line):",
    value=(
        "Federal Reserve announces unexpected interest rate cut amid easing inflation.\n"
        "China reports stronger than expected manufacturing GDP growth and stimulus."
    ),
    height=180,
)

predict_btn = st.sidebar.button("Predict Across All Indices")

if predict_btn:
    headlines = [h.strip() for h in user_prompt.split("\n") if h.strip()]
    if not headlines:
        st.warning("Please enter at least one headline in the sidebar.")
        st.stop()

    with st.spinner("Running FinBERT + VADER + TF-IDF Anchors, fetching live yfinance/FRED macro series, and executing Two-Tower MultimodalMarketNet..."):
        mkt_data, fred_data = fetch_live_macro_and_market()
        x_news_s, diag = build_exact_22d_news_features(headlines)

        indices = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]

        results = []
        for label, sym in indices:
            x_macro_s = build_exact_45d_macro_features(mkt_data, fred_data, focus_sym=sym)
            probs = predict_for_index(sym, x_macro_s, x_news_s)
            results.append((label, probs))

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
    fb_tag = "ProsusAI/finbert Active" if diag["finbert_active"] else "VADER Fallback (install `transformers` for FinBERT)"
    st.success(
        f"Live Multimodal Inference Complete ({fb_tag}) | "
        f"FinBERT Net Polarity: `{diag['fb_net']:+.3f}` | "
        f"VADER Compound: `{diag['vader_comp']:+.3f}` | "
        f"TF-IDF Anchor Spread (Up−Down): `{diag['tfidf_spread']:+.4f}`"
    )
else:
    st.info("👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**.")
