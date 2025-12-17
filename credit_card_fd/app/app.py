

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

# Config

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_pipeline.pkl"
DATA_PATH = ROOT / "data" / "fraudTest.csv"   # REQUIRED (no fallback)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model(MODEL_PATH)


# Load dataset (REQUIRED)

@st.cache_data
def load_data(csv_path: Path):
    if not csv_path.exists():
        st.error(f"Dataset not found: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)

    # Clean merchants for UI (remove leakage prefix)
    df["merchant_clean"] = (
        df["merchant"].astype(str).str.replace(r"^fraud_", "", regex=True)
    )

    return df

df_data = load_data(DATA_PATH)


# Build dropdown options (only from data)

def top_k(series: pd.Series, k: int = 30):
    return series.value_counts().head(k).index.tolist()

merchant_choices = top_k(df_data["merchant_clean"], 40)
category_choices = top_k(df_data["category"], 30)
job_choices = top_k(df_data["job"], 30)
state_choices = sorted(df_data["state"].dropna().unique().tolist())



# Expected columns (from your training X)

EXPECTED_COLS = [
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "unix_time",
    "merch_lat",
    "merch_long",
    "trans_hour",
    "trans_dayofweek",
    "trans_month",
    "age",
]

# Hidden defaults (still required by your current model)
HIDDEN_DEFAULTS = {
    "first": "Unknown",
    "last": "Unknown",
    "street": "Unknown",
    "zip": "00000",
    "unix_time": 0,
}


# UI

st.title("💳 Credit Card Fraud Detection")
st.write("Clean UI. Dropdowns are populated only from your dataset (no fallback).")

# Sidebar threshold
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fraud threshold", 0.05, 0.90, 0.30, 0.01)
show_debug = st.sidebar.checkbox("Show debug (model input)", value=False)

st.subheader("Transaction input")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        trans_date = st.date_input("Transaction date", value=date(2020, 12, 1))
        trans_hour = st.slider("Transaction hour", 0, 23, 12)
        amt = st.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0)
        gender = st.selectbox("Gender", ["M", "F"])
        state = st.selectbox("State", state_choices)

    with col2:
        merchant = st.selectbox("Merchant", merchant_choices)  # cleaned
        category = st.selectbox("Category", category_choices)
        job = st.selectbox("Job", job_choices)
        city = st.text_input("City", value="New York")
        city_pop = st.number_input("City population", min_value=0, value=50000, step=1000)

    st.markdown("### Customer")
    age = st.slider("Customer age", min_value=18, max_value=95, value=35)

    st.markdown("### Location")
    lat = st.number_input("Customer latitude (lat)", value=float(df_data["lat"].median()))
    long_ = st.number_input("Customer longitude (long)", value=float(df_data["long"].median()))
    merch_lat = st.number_input("Merchant latitude (merch_lat)", value=float(df_data["merch_lat"].median()))
    merch_long = st.number_input("Merchant longitude (merch_long)", value=float(df_data["merch_long"].median()))

    submitted = st.form_submit_button("Predict fraud risk")


# Predict + Analysis

if submitted:
    trans_dayofweek = trans_date.weekday()
    trans_month = trans_date.month

    # Build features
    row = {
        "merchant": str(merchant),            # cleaned merchant
        "category": str(category),
        "amt": float(amt),
        "gender": str(gender),
        "city": str(city),
        "state": str(state),
        "lat": float(lat),
        "long": float(long_),
        "city_pop": int(city_pop),
        "job": str(job),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
        "trans_hour": int(trans_hour),
        "trans_dayofweek": int(trans_dayofweek),
        "trans_month": int(trans_month),
        "age": int(age),
    }

    X_input = pd.DataFrame([row])

    # Add required hidden columns for CURRENT trained model
    for col, val in HIDDEN_DEFAULTS.items():
        if col not in X_input.columns:
            X_input[col] = val

    # unix_time required by your model
    if "unix_time" not in X_input.columns:
        X_input["unix_time"] = 0

    # Ensure correct order
    missing = set(EXPECTED_COLS) - set(X_input.columns)
    if missing:
        st.error(f"Missing columns for the model: {missing}")
        st.stop()

    X_input = X_input[EXPECTED_COLS]

    try:
        prob = model.predict_proba(X_input)[0, 1]
        pred = int(prob >= threshold)

        st.subheader("Result")
        st.metric("Fraud probability", f"{prob * 100:.2f}%")
        st.caption(f"Threshold used: {threshold:.2f}")

        if pred == 1:
            st.error("⚠️ Likely FRAUD")
        else:
            st.success("✅ Likely LEGIT")

        st.subheader("📊 Prediction analysis")

        # 1) Probability chart
        fig1 = plt.figure()
        plt.bar(["Legit (1-p)", "Fraud (p)"], [1 - prob, prob])
        plt.axhline(y=threshold, linestyle="--")
        plt.ylim(0, 1)
        plt.title("Model output probability")
        plt.ylabel("Probability")
        st.pyplot(fig1)

        # 2) Amount distribution
        fig2 = plt.figure()
        clipped = df_data["amt"].clip(upper=500)
        plt.hist(clipped, bins=50)
        plt.axvline(x=amt, linestyle="--")
        plt.title("Transaction amount vs typical amounts (clipped at $500)")
        plt.xlabel("Amount ($)")
        plt.ylabel("Count")
        st.pyplot(fig2)

        # 3) Category fraud rates
        cat_stats = (
            df_data.groupby("category")["is_fraud"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig3 = plt.figure()
        plt.bar(cat_stats.index.astype(str), cat_stats.values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Top 10 categories by historical fraud rate")
        plt.ylabel("Fraud rate")
        st.pyplot(fig3)

        if show_debug:
            with st.expander("Debug: model input"):
                st.dataframe(X_input)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.info(
            "If you stripped 'fraud_' in UI but your model was trained with 'fraud_' merchants, "
            "retrain the model after removing the prefix (recommended)."
        )
