import io
import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForSequenceClassification, BertTokenizer

# Lock CPU thread pools to prevent OpenMP/LightGBM deadlocks & RAM spikes on Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Multi-Index Market Sentiment AI",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Custom News Prompt & Multi-Index Market Predictor")
st.markdown(
    "Enter custom news headlines or macroeconomic scenarios below to run real-time "
    "inference through **FinBERT (`yiyanghkust/finbert-tone`)**, **Two-Tower "
    "`MultimodalMarketNet`**, **LightGBM**, and the **OOF Meta-Stacker** across "
    "**Dow Jones**, **S&P 500**, **Nikkei 225**, and **China SSE**."
)

# Exact 6 institutional anchor prototypes from kaggle.ipynb
DEFAULT_ANCHORS = {
    "Monetary_Fed_Rates": "federal reserve interest rate cut hike central bank ecb boj inflation monetary policy treasury yield bond",
    "Global_Trade_China": "china trade export import tariff manufacturing gdp growth economy beijing asia stimulus",
    "Tech_Equities_Earnings": "nvidia apple microsoft tech semiconductor ai chip silicon valley stocks rise rally surge optimistic earnings profit record bull",
    "Banking_Credit_Markets": "bank wall street credit debt loan financial crisis default bankruptcy plunge crash losses recession selloff bear",
    "Energy_Disasters_Supply": "oil crude opec energy commodity earthquake disaster hurricane supply chain shortage destruction halt",
    "Geopolitics_Security": "war military sanction nuclear missile conflict russia ukraine iran israel strike treaty peace",
}

EXACT_44_MACRO_COLS = [
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


# 1. Exact PyTorch Two-Tower Cross-Attention Architecture (Matches kaggle.ipynb)
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
    def __init__(self, macro_dim: int = 45, news_dim: int = 22, hidden_dim: int = 64, dropout: float = 0.30):
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


# 2. Cached Artifact & FinBERT Loader (Cloud-Safe Path & Memory Handling)
@st.cache_resource(show_spinner="Loading FinBERT & trained Multimodal Ensemble weights...")
def load_production_pipeline():
    required_files = [
        "multimodal_market_net.pt",
        "gbdt_model.pkl",
        "meta_stacker_clf.pkl",
        "scaler_macro.pkl",
        "scaler_news.pkl",
        "tfidf_anchor.pkl",
    ]
    missing = [f for f in required_files if not (BASE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s) in repository root: {', '.join(missing)}"
        )

    fb_tok = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fb_mdl = BertForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone", low_cpu_mem_usage=True
    )
    fb_mdl.eval()

    s_macro = joblib.load(BASE_DIR / "scaler_macro.pkl")
    s_news = joblib.load(BASE_DIR / "scaler_news.pkl")
    tfidf_pack = joblib.load(BASE_DIR / "tfidf_anchor.pkl")

    if isinstance(tfidf_pack, dict):
        tfidf_vec = tfidf_pack["vectorizer"]
        anchors = tfidf_pack.get("anchors", DEFAULT_ANCHORS)
        macro_cols = tfidf_pack.get("macro_cols", EXACT_44_MACRO_COLS)
    else:
        tfidf_vec = tfidf
