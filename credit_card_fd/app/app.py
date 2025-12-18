

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, datetime, time
import matplotlib.pyplot as plt
import math

# Config

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_pipeline.pkl"


# Load model

@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model(MODEL_PATH)


# Expected columns (from training)

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

# Hidden defaults required by current trained model
HIDDEN_DEFAULTS = {
    "first": "Unknown",
    "last": "Unknown",
    "street": "Unknown",
    "zip": "00000",
}


# Helpers

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle distance between two points on Earth (km).
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def make_unix_time(d: date, hour: int) -> int:
    """
    Build a stable unix_time from date + hour (approximate).
    """
    dt = datetime.combine(d, time(hour=hour, minute=0, second=0))
    return int(dt.timestamp())



# UI

st.title("💳 Credit Card Fraud Detection")
st.caption("This app loads a trained model and predicts fraud risk. No dataset (CSV) is used at runtime.")

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
        state = st.text_input("State (2-letter code)", value="NY", max_chars=2)

    with col2:
        # Free text inputs (no dataset, no dropdowns)
        merchant = st.text_input("Merchant (exact format your model expects)", value="fraud_Kirlin and Sons")
        category = st.text_input("Category", value="shopping_net")
        job = st.text_input("Job", value="Unknown")
        city = st.text_input("City", value="New York")
        city_pop = st.number_input("City population", min_value=0, value=50000, step=1000)

    st.markdown("### Customer")
    age = st.slider("Customer age", min_value=18, max_value=95, value=35)

    st.markdown("### Location")
    lat = st.number_input("Customer latitude (lat)", value=40.7128, format="%.6f")
    long_ = st.number_input("Customer longitude (long)", value=-74.0060, format="%.6f")
    merch_lat = st.number_input("Merchant latitude (merch_lat)", value=40.7128, format="%.6f")
    merch_long = st.number_input("Merchant longitude (merch_long)", value=-74.0060, format="%.6f")

    submitted = st.form_submit_button("Predict fraud risk")

# Predict + Analysis 

if submitted:
    trans_dayofweek = trans_date.weekday()
    trans_month = trans_date.month

    unix_time = make_unix_time(trans_date, trans_hour)

    # Build features
    row = {
        "merchant": str(merchant),
        "category": str(category),
        "amt": float(amt),
        "gender": str(gender),
        "city": str(city),
        "state": str(state).upper(),
        "zip": HIDDEN_DEFAULTS["zip"],  # required by model
        "lat": float(lat),
        "long": float(long_),
        "city_pop": int(city_pop),
        "job": str(job),
        "unix_time": int(unix_time),
        "merch_lat": float(merch_lat),
        "merch_long": float(merch_long),
        "trans_hour": int(trans_hour),
        "trans_dayofweek": int(trans_dayofweek),
        "trans_month": int(trans_month),
        "age": int(age),
    }

    # Add required hidden fields
    for col, val in HIDDEN_DEFAULTS.items():
        row.setdefault(col, val)

    X_input = pd.DataFrame([row])

    # Ensure correct columns/order
    missing = set(EXPECTED_COLS) - set(X_input.columns)
    if missing:
        st.error(f"Missing columns for the model: {sorted(missing)}")
        st.stop()
    X_input = X_input[EXPECTED_COLS]

    try:
        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(prob >= threshold)

        st.subheader("Result")
        st.metric("Fraud probability", f"{prob * 100:.2f}%")
        st.caption(f"Threshold used: {threshold:.2f}")

        if pred == 1:
            st.error("⚠️ Likely FRAUD")
        else:
            st.success("✅ Likely LEGIT")

        st.subheader("📊 Analysis (no dataset required)")

        # 1) Probability chart
        fig1 = plt.figure()
        plt.bar(["Legit (1-p)", "Fraud (p)"], [1 - prob, prob])
        plt.axhline(y=threshold, linestyle="--")
        plt.ylim(0, 1)
        plt.title("Model output probability")
        plt.ylabel("Probability")
        st.pyplot(fig1)
        plt.close(fig1)

        # 2) Distance feature insight (derived)
        dist_km = haversine_km(lat, long_, merch_lat, merch_long)
        st.write(f"**Customer ↔ Merchant distance:** {dist_km:.1f} km")

        fig2 = plt.figure()
        plt.bar(["Distance (km)"], [dist_km])
        plt.title("Customer–Merchant distance (derived)")
        plt.ylabel("km")
        st.pyplot(fig2)
        plt.close(fig2)

        # 3) Simple rule-based “signals” (derived)
        st.markdown("**Signals (derived checks)**")
        signals = []
        if amt >= 200:
            signals.append("High amount (≥ $200)")
        if trans_hour <= 5 or trans_hour >= 23:
            signals.append("Late-night transaction")
        if trans_dayofweek >= 5:
            signals.append("Weekend transaction")
        if dist_km >= 300:
            signals.append("Large distance (≥ 300 km)")

        if signals:
            for s in signals:
                st.warning(s)
        else:
            st.success("No obvious risk signals triggered by simple rules.")

        if show_debug:
            with st.expander("Debug: model input"):
                st.dataframe(X_input)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.info(
           
        )
