"""Streamlit demo: real saved LightGBM inference and separate text sentiment."""
from pathlib import Path
import io
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

st.set_page_config(page_title="News & Dow Jones Research", layout="wide")
ROOT = Path(__file__).resolve().parent
LABELS = {0: "Down", 1: "Neutral", 2: "Up"}

@st.cache_resource
def load_model():
    # Only load the repository's trusted trained artifact, never user pickle files.
    return joblib.load(ROOT / "gbdt_model.pkl")

@st.cache_resource
def sentiment_tool():
    if not nltk.download("vader_lexicon", quiet=True):
        raise RuntimeError("Could not download the VADER lexicon.")
    return SentimentIntensityAnalyzer()

def read_features(payload, model):
    frame = pd.read_csv(io.BytesIO(payload))
    expected = [f"f{i}" for i in range(model.n_features_in_)]
    if list(frame.columns) != ["Date", "Actual", *expected]:
        raise ValueError("Column names/order do not match. Use the Colab export below.")
    if frame.empty or len(frame) > 10000:
        raise ValueError("Upload between 1 and 10,000 daily rows.")
    dates = pd.to_datetime(frame["Date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be present, unique, and in increasing order.")
    x = frame[expected].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Features contain missing or infinite values.")
    if (np.abs(x) > 5.00001).any():
        raise ValueError("Expected already-scaled, clipped X_full_te values; do not upload raw data.")
    y = pd.to_numeric(frame["Actual"], errors="raise")
    if not y.isin([0, 1, 2]).all():
        raise ValueError("Actual labels must be 0, 1, or 2.")
    return dates, x, y.to_numpy(dtype=int)

st.title("News & Dow Jones Research")
st.caption("Research prototype. Not financial advice.")
st.write("Replay real predictions from the saved LightGBM model, or inspect the tone of custom headlines.")
st.info("Market predictions require the complete news and market features used in Colab. "
        "Text sentiment alone is not a market forecast. Only Dow Jones is supported here.")
replay, text_tab = st.tabs(["Real model predictions", "Headline sentiment"])

with replay:
    st.subheader("Historical test replay")
    st.write("This tab calls gbdt_model.pkl.predict_proba on exported Colab test features. "
             "It uses LightGBM alone, not the neural-network stack; the earlier 57.41% "
             "stacked accuracy does not apply to this tab.")
    with st.expander("Prepare the input in Colab", expanded=True):
        st.write("Run this after the final training cell in the SAME session. It exports the "
                 "existing scaled test inputs and the matching trained LightGBM. Replace "
                 "gbdt_model.pkl in this repository with the downloaded file from that run.")
        st.code('''import numpy as np
import pandas as pd
import joblib
from google.colab import files

X_export = np.asarray(X_full_te)
assert X_export.shape[0] == len(y_test_3c)
assert X_export.shape[1] == gbdt_full.n_features_in_
assert np.isfinite(X_export).all()

export = pd.DataFrame(
    X_export, columns=[f"f{i}" for i in range(X_export.shape[1])]
)
export.insert(0, "Actual", np.asarray(y_test_3c, dtype=int))
export.insert(0, "Date",
              pd.to_datetime(df.loc[test_mask, "Date"]).dt.strftime("%Y-%m-%d").to_numpy())
export.to_csv("test_features.csv", index=False)
joblib.dump(gbdt_full, "gbdt_model.pkl")

files.download("test_features.csv")
files.download("gbdt_model.pkl")
''', language="python")
    st.caption("No second scaling is applied: X_full_te has already been processed in Colab. "
               "Use matching features and model from one run, not files from different experiments.")
    upload = st.file_uploader("Upload test_features.csv", type=["csv"])
    bundled = ROOT / "test_features.csv"
    payload = upload.getvalue() if upload is not None else (bundled.read_bytes() if bundled.exists() else None)
    if payload is None:
        st.warning("Waiting for the exported test features. No market probabilities are generated without inputs.")
    else:
        try:
            gbdt = load_model()
            dates, x, actual = read_features(payload, gbdt)
            probs = np.asarray(gbdt.predict_proba(x))
            classes = np.asarray(gbdt.classes_, dtype=int)
            if set(classes.tolist()) != {0, 1, 2} or probs.shape != (len(x), 3):
                raise ValueError("Expected a trained three-class LightGBM.")
            probs = probs[:, [int(np.flatnonzero(classes == k)[0]) for k in range(3)]]
            if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1):
                raise ValueError("Model returned invalid probabilities.")
            predicted = probs.argmax(axis=1)
            a, b, c = st.columns(3)
            a.metric("LightGBM accuracy", f"{accuracy_score(actual, predicted):.2%}")
            b.metric("Macro F1", f"{f1_score(actual, predicted, labels=[0,1,2], average='macro', zero_division=0):.3f}")
            c.metric("Always-neutral baseline", f"{np.mean(actual == 1):.2%}")
            st.caption(f"{len(x)} uploaded dates: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}. "
                       "These are historical labels, not a live forecast or trading return.")
            selected = st.selectbox("Inspect a test date", range(len(x)),
                                    format_func=lambda i: dates.iloc[i].strftime("%Y-%m-%d"))
            st.write(f"Prediction: **{LABELS[int(predicted[selected])]}** · "
                     f"Actual: **{LABELS[int(actual[selected])]}**")
            st.bar_chart(pd.DataFrame({"Model probability": probs[selected]},
                                     index=["Down", "Neutral", "Up"]))
            st.caption("Model probabilities are not a verified probability of being correct.")
            st.write("Confusion matrix: rows are actual classes, columns are predictions.")
            st.dataframe(pd.DataFrame(confusion_matrix(actual, predicted, labels=[0,1,2]),
                                      index=["Actual Down","Actual Neutral","Actual Up"],
                                      columns=["Predicted Down","Predicted Neutral","Predicted Up"]))
            result = pd.DataFrame({"Date": dates.dt.strftime("%Y-%m-%d"),
                                   "Actual": actual, "Predicted": predicted,
                                   "P_Down": probs[:,0], "P_Neutral": probs[:,1], "P_Up": probs[:,2]})
            st.download_button("Download actual model predictions", result.to_csv(index=False),
                               "lightgbm_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Prediction unavailable: {exc}")

with text_tab:
    st.subheader("Inspect headline tone")
    st.write("This is VADER text sentiment only. It does not call the stock predictor and "
             "cannot establish whether a headline will move a market.")
    with st.form("headlines"):
        prompt = st.text_area("One headline per line", height=180)
        submitted = st.form_submit_button("Analyze sentiment")
    if submitted:
        headlines = [h.strip() for h in prompt.splitlines() if h.strip()]
        if not headlines:
            st.warning("Enter at least one headline.")
        elif len(headlines) > 25:
            st.warning("Enter at most 25 headlines. No lines have been silently removed.")
        else:
            try:
                sia = sentiment_tool()
                scores = [sia.polarity_scores(h)["compound"] for h in headlines]
                st.success(f"Analyzed {len(headlines)} headline(s). No filler headlines were added.")
                st.dataframe(pd.DataFrame({"Headline": headlines, "Text sentiment": scores}),
                             hide_index=True)
                st.caption("Positive language, including non-financial text, can receive a positive "
                           "sentiment score. It is never converted into an index probability.")
            except Exception as exc:
                st.error(f"Sentiment analysis unavailable: {exc}")

