import os



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import urllib.request
from pathlib import Path

import joblib
import lightgbm  
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
BASEDIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Custom Prompt Multi-Index Market AI", layout="wide")
st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to evaluate "
    "real-time **Down (0)**, **Neutral (1)**, or **Up (2)** regime predictions "
    "across **Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)

DEFAULTMACROCOLS = [
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
        self, macrodim: int = 45, newsdim: int = 22, hiddendim: int = 64, dropout: float = 0.30
    ):
        super().__init__()
        self.macro_in = nn.Sequential(
            nn.Linear(macrodim, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.GELU(),
            GatedResidualBlock(hiddendim, dropout),
        )
        self.news_in = nn.Sequential(
            nn.Linear(newsdim, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.GELU(),
            GatedResidualBlock(hiddendim, dropout),
        )
        self.cross_attn_gate = nn.Sequential(
            nn.Linear(hiddendim * 2, hiddendim),
            nn.GELU(),
            nn.Linear(hiddendim, hiddendim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hiddendim * 2, hiddendim),
            nn.LayerNorm(hiddendim),
            nn.GELU(),
            GatedResidualBlock(hiddendim, dropout),
            GatedResidualBlock(hiddendim, dropout),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hiddendim, 3),
        )

    def forward(self, xm: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
        hm = self.macro_in(xm)
        hn = self.news_in(xn)
        attnweights = self.cross_attn_gate(torch.cat([hm, hn], dim=-1))
        return self.fusion(torch.cat([hm, hn * attnweights], dim=-1))



@st.cache_resource(show_spinner="Loading FinBERT and trained Multimodal Ensemble weights...")
def loadpipeline():
    fbtokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fbmodel = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    fbmodel.eval()

    smacro = joblib.load(BASEDIR / "scaler_macro.pkl")
    snews = joblib.load(BASEDIR / "scaler_news.pkl")
    tfidfbundle = joblib.load(BASEDIR / "tfidf_anchor.pkl")

    tfidfvec = tfidfbundle["vectorizer"]
    anchorsdict = tfidfbundle["anchors"]
    macrocols = tfidfbundle.get("macro_cols") or DEFAULTMACROCOLS
    anchormat = tfidfvec.transform(list(anchorsdict.values()))

    macrodim = int(getattr(smacro, "n_features_in_", 45))
    newsdim = int(getattr(snews, "n_features_in_", 22))

    
    ensembles = {}
    regpath = BASEDIR / "regional_ensembles.pkl"
    if regpath.exists():
        rawreg = joblib.load(regpath)
        for sym, pack in rawreg.items():
            nnm = MultimodalMarketNet(macrodim=macrodim, newsdim=newsdim, hiddendim=64)
            nnm.load_state_dict(pack["nn_state_dict"])
            nnm.eval()

            gbm = pack["gbdt"]
            gbm.set_params(n_jobs=1)

            ensembles[sym] = {"nn": nnm, "gbdt": gbm, "meta": pack["meta"]}
    else:
        nnm = MultimodalMarketNet(macrodim=macrodim, newsdim=newsdim, hiddendim=64)
        nnm.load_state_dict(
            torch.load(BASEDIR / "multimodal_market_net.pt", map_location="cpu")
        )
        nnm.eval()

        gbm = joblib.load(BASEDIR / "gbdt_model.pkl")
        gbm.set_params(n_jobs=1)
        metam = joblib.load(BASEDIR / "meta_stacker_clf.pkl")

        for sym in ["^DJI", "^GSPC", "^N225", "000001.SS"]:
            ensembles[sym] = {"nn": nnm, "gbdt": gbm, "meta": metam}

    return fbtokenizer, fbmodel, smacro, snews, tfidfvec, anchormat, macrocols, ensembles


(
    finberttokenizer,
    finbertmodel,
    scalermacro,
    scalernews,
    tfidfv2,
    anchormat,
    savedmacrocols,
    regionalensembles,
) = loadpipeline()



def buildnewsmatrixfromfinbertandsims(
    fbneua: np.ndarray, fbposa: np.ndarray, fbnega: np.ndarray, simsa: np.ndarray
) -> np.ndarray:
    net = (fbposa - fbnega).reshape(-1, 1)
    csent = net * (simsa + 0.15)                                    
    cmass = simsa                                                   
    maxsim = simsa.max(axis=1, keepdims=True)
    meansim = simsa.mean(axis=1, keepdims=True)
    weightednet = net * (maxsim + 0.25)
    conviction = (1.0 - fbneua).reshape(-1, 1)
    posdom = np.maximum(0.0, fbposa - fbnega).reshape(-1, 1)
    negdom = np.maximum(0.0, fbnega - fbposa).reshape(-1, 1)
    summary = np.hstack([weightednet, conviction, posdom, -negdom, meansim, net])  
    fbblock = np.column_stack([fbneua, fbposa, fbnega, fbposa - fbnega])    
    return np.hstack([csent, cmass, summary, fbblock]).astype(np.float32)


def encodeuserheadlines(headlines: list[str]) -> tuple[np.ndarray, float, dict]:
    
    batches = []
    with torch.inference_mode():
        for i in range(0, len(headlines), 8):
            enc = finberttokenizer(
                headlines[i : i + 8],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            probs = F.softmax(finbertmodel(**enc).logits, dim=-1).cpu().numpy()
            batches.append(probs)

    hlprobs = np.vstack(batches)  

    
    saliencew = (1.0 - hlprobs[:, 0]) ** 1.5 + 0.05
    saliencew = (saliencew / saliencew.sum()).astype(np.float32)

    fbneu = np.array([np.sum(hlprobs[:, 0] * saliencew)], dtype=np.float32)
    fbpos = np.array([np.sum(hlprobs[:, 1] * saliencew)], dtype=np.float32)
    fbneg = np.array([np.sum(hlprobs[:, 2] * saliencew)], dtype=np.float32)

    hlsims = cosine_similarity(tfidfv2.transform(headlines), anchormat)  
    sims = (saliencew.reshape(-1, 1) * hlsims).sum(axis=0, keepdims=True).astype(np.float32)

    xnewsraw = buildnewsmatrixfromfinbertandsims(fbneu, fbpos, fbneg, sims)
    newscol12raw = float(xnewsraw[0, 12])
    xnewss = np.clip(scalernews.transform(xnewsraw), -4.5, 4.5).astype(np.float32)

    diag = {
        "count": len(headlines),
        "fb_neu": float(fbneu[0]),
        "fb_pos": float(fbpos[0]),
        "fb_neg": float(fbneg[0]),
        "fb_net": float(fbpos[0] - fbneg[0]),
    }
    return xnewss, newscol12raw, diag



@st.cache_data(ttl=1800, show_spinner=False)
def fetchstationarymacro44(expectedcols: tuple[str, ...]) -> tuple[np.ndarray, bool]:
    tickers = ["^DJI", "^GSPC", "^N225", "000001.SS", "^VIX"]
    closed, highd, lowd, vold = {}, {}, {}, {}

    for sym in tickers:
        try:
            hist = yf.Ticker(sym).history(period="9mo", interval="1d", auto_adjust=True, timeout=8)
            if hist is not None and not hist.empty:
                if getattr(hist.index, "tz", None) is not None:
                    hist.index = hist.index.tz_localize(None)
                hist.index = hist.index.normalize()
                closed[sym] = hist["Close"]
                highd[sym] = hist["High"]
                lowd[sym] = hist["Low"]
                vold[sym] = hist["Volume"]
        except Exception:
            continue

    if len(closed) < 2:
        return np.zeros(len(expectedcols), dtype=np.float32), False

    closepx = pd.DataFrame(closed).ffill().bfill()
    highpx = pd.DataFrame(highd).reindex(closepx.index).ffill().bfill()
    lowpx = pd.DataFrame(lowd).reindex(closepx.index).ffill().bfill()
    volpx = pd.DataFrame(vold).reindex(closepx.index).ffill().bfill()
    rets = closepx.pct_change()

    mktfeats = pd.DataFrame(index=closepx.index)

    for sym, tag in [("^DJI", "DJI"), ("^GSPC", "SPX")]:
        if sym in rets.columns:
            rlag1 = rets[sym].shift(1)
            mktfeats[f"{tag}_ret_lag1"] = rlag1
            mktfeats[f"{tag}_ret_lag2"] = rets[sym].shift(2)
            mktfeats[f"{tag}_ret_ma5"] = rlag1.rolling(5).mean()
            mktfeats[f"{tag}_ret_ma20"] = rlag1.rolling(20).mean()
            mktfeats[f"{tag}_vol_5d"] = rlag1.rolling(5).std()
            mktfeats[f"{tag}_vol_20d"] = rlag1.rolling(20).std()
            mktfeats[f"{tag}_vol_ratio"] = mktfeats[f"{tag}_vol_5d"] / (
                mktfeats[f"{tag}_vol_20d"] + 1e-6
            )
            hlrange = ((highpx[sym] - lowpx[sym]) / (closepx[sym] + 1e-6)).shift(1)
            mktfeats[f"{tag}_hl_range_lag1"] = hlrange
            mktfeats[f"{tag}_hl_range_ma5"] = hlrange.rolling(5).mean()
            mktfeats[f"{tag}_open_gap_t"] = 0.0
            vols1 = volpx[sym].shift(1)
            mktfeats[f"{tag}_vol_z_lag1"] = (vols1 - vols1.rolling(20).mean()) / (
                vols1.rolling(20).std() + 1e-6
            )

    if "^VIX" in closepx.columns:
        vixc = closepx["^VIX"].shift(1)
        mktfeats["VIX_lag1"] = vixc
        mktfeats["VIX_diff1"] = closepx["^VIX"].diff().shift(1)
        mktfeats["VIX_ma_rat"] = vixc / (vixc.rolling(20).mean() + 1e-6)
        mktfeats["VIX_open_t"] = 0.0

    for sym, tag in [("^N225", "NIKKEI"), ("000001.SS", "SSE")]:
        if sym in rets.columns:
            rasialag1 = rets[sym].shift(1)
            mktfeats[f"{tag}_preUS_ret_t"] = 0.0
            mktfeats[f"{tag}_ret_lag1"] = rasialag1
            mktfeats[f"{tag}_ret_ma5"] = rasialag1.rolling(5).mean()
            mktfeats[f"{tag}_vol_10d"] = rasialag1.rolling(10).std()

    startdt = (closepx.index.min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    for c in ["T10Y2Y", "BAA10Y", "DCOILWTICO", "DEXJPUS", "DEXCHUS"]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={c}&cosd={startdt}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                csvtxt = resp.read().decode("utf-8")
            s = pd.to_numeric(
                pd.read_csv(io.StringIO(csvtxt), index_col=0, parse_dates=True)[c],
                errors="coerce",
            ).ffill()
            aligned = s.reindex(closepx.index).ffill().bfill()
            mktfeats[f"{c}_lag1"] = aligned.shift(1)
            mktfeats[f"{c}_chg5"] = aligned.diff(5).shift(1)
        except Exception:
            mktfeats[f"{c}_lag1"] = 0.0
            mktfeats[f"{c}_chg5"] = 0.0

    
    mktfeats = mktfeats.reindex(columns=list(expectedcols), fill_value=0.0)
    mktfeats = mktfeats.ffill().bfill().fillna(0.0)

    
    rollmean = mktfeats.rolling(60, min_periods=10).mean().bfill()
    rollstd = mktfeats.rolling(60, min_periods=10).std().bfill().replace(0, 1.0).fillna(1.0)
    macrostationary = ((mktfeats - rollmean) / (rollstd + 1e-6)).clip(-4.0, 4.0).fillna(0.0)

    return macrostationary.iloc[-1].values.astype(np.float32), True


def buildmacrotensor(last44: np.ndarray, newscol12raw: float) -> np.ndarray:
    xmacroraw = np.append(last44, np.float32(newscol12raw)).reshape(1, -1)
    return np.clip(scalermacro.transform(xmacroraw), -4.5, 4.5).astype(np.float32)



def runindexensemble(sym: str, xmacros: np.ndarray, xnewss: np.ndarray) -> list[float]:
    pack = regionalensembles[sym]
    with torch.inference_mode():
        logits = pack["nn"](
            torch.from_numpy(xmacros),
            torch.from_numpy(xnewss),
        )
        pnn = F.softmax(logits, dim=-1).cpu().numpy()

    pgbdt = pack["gbdt"].predict_proba(np.hstack([xmacros, xnewss]))
    pmeta = pack["meta"].predict_proba(np.hstack([pnn, pgbdt]))[0]
    return pmeta.tolist()



with st.sidebar.form("scenario_form"):
    st.header("News Scenario Prompt")
    userprompt = st.text_area(
        "Enter custom headlines or market events (one per line):",
        value="NVIDIA stocks rise a lot, very optimistic",
        height=200,
    )
    submitbtn = st.form_submit_button("Predict Across All Indices")

if submitbtn:
    headlines = [line.strip() for line in userprompt.splitlines() if line.strip()]
    if not headlines:
        st.warning("Please enter at least one news headline.")
        st.stop()

    with st.spinner("Running FinBERT, fetching live macro state, and executing Multimodal Ensembles..."):
        last44, livemacrook = fetchstationarymacro44(tuple(savedmacrocols))
        xnewss, newscol12raw, diag = encodeuserheadlines(headlines)
        xmacros = buildmacrotensor(last44, newscol12raw)

        indextargets = [
            ("Dow Jones (^DJI)", "^DJI"),
            ("S&P 500 (^GSPC)", "^GSPC"),
            ("Nikkei 225 (^N225)", "^N225"),
            ("China SSE (000001.SS)", "000001.SS"),
        ]
        results = [
            (label, runindexensemble(sym, xmacros, xnewss))
            for label, sym in indextargets
        ]

    st.subheader("📊 Independent Regional Index Predictions")
    cols = st.columns(4)
    regimelabels = ["Down (0)", "Neutral (1)", "Up (2)"]

    for colui, (idxname, probs) in zip(cols, results):
        predidx = int(np.argmax(probs))
        with colui:
            st.markdown(f"### {idxname}")
            st.metric(
                "Regime",
                regimelabels[predidx],
                f"{probs[predidx] * 100:.1f}% Confidence",
            )
            probdf = pd.DataFrame(
                {"Class": ["Down", "Neutral", "Up"], "Probability": probs}
            ).set_index("Class")
            st.bar_chart(probdf, height=200)

    st.markdown("---")
    macrostatus = "Live yfinance + FRED" if livemacrook else "Neutral Stationary Baseline"
    st.success(
        f"**Headlines:** `{diag['count']}` | **Macro Feed:** `{macrostatus}` | "
        f"**FinBERT (`yiyanghkust/finbert-tone`):** "
        f"`[Neu: {diag['fb_neu']:.3f}, Pos: {diag['fb_pos']:.3f}, Neg: {diag['fb_neg']:.3f}, Net: {diag['fb_net']:+.3f}]`"
    )
else:
    st.info(
        "👈 Type or paste your custom news headlines in the sidebar and click **Predict Across All Indices**."
    )

