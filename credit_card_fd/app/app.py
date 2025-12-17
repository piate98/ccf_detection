

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered",
)



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

# Load dataset (OPTIONAL)

UI_COLS = [
    "merchant",
    "category",
    "job",
    "state",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "amt",
    "is_fraud",  # might not exist in some datasets; we'll handle it
]

@st.cache_data
def load_data_optional(csv_path: Path):
    if not csv_path.exists():
        return None

    # Read only existing columns 
    try:
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        st.warning(f"Could not read dataset header: {e}")
        return None

    usecols = [c for c in UI_COLS if c in header_cols]
    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")
        return None

    # Add merchant_clean and mapping to keep model-compatible merchants
    if "merchant" in df.columns:
        df["merchant"] = df["merchant"].astype(str)
        df["merchant_clean"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)
    else:
        df["merchant_clean"] = ""

    return df

df_data = load_data_optional(DATA_PATH)


def top_k(series: pd.Series, k: int = 30):
    return series.value_counts().head(k).index.tolist()

def safe_median(df: pd.DataFrame, col: str, default: float):
    if df is None or col not in df.columns or df[col].dropna().empty:
        return default
    return float(df[col].median())

# Build dropdowns + merchant mapping
merchant_choices = []
category_choices = []
job_choices = []
state_choices = []


merchant_clean_to_raw = {}

if df_data is not None and not df_data.empty:
    if "merchant_clean" in df_data.columns and "merchant" in df_data.columns:
        merchant_choices = top_k(df_data["merchant_clean"], 40)

        # For each cleaned merchant name, keep the most common raw merchant string
        # (usually "fraud_XXXX") so the model input matches training.
        tmp = (
            df_data.groupby("merchant_clean")["merchant"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        merchant_clean_to_raw = tmp

    if "category" in df_data.columns:
        category_choices = top_k(df_data["category"].astype(str), 30)

    if "job" in df_data.columns:
        job_choices = top_k(df_data["job"].astype(str), 30)

    if "state" in df_data.columns:
        state_choices = sorted(df_data["state"].dropna().astype(str).unique().tolist())

# Reasonable fallbacks if dataset missing
if not merchant_choices:
    merchant_choices = ["amazon", "walmart", "target"]
if not category_choices:
    category_choices = ["shopping_net", "misc_net", "grocery_pos"]
if not job_choices:
    job_choices = ["Unknown"]
if not state_choices:
    state_choices = ["NY", "CA", "TX"]


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


HIDDEN_DEFAULTS = {
    "first": "Unknown",
    "last": "Unknown",
    "street": "Unknown",
    "zip": 0,
    "unix_time": 0,
}

# UI
st.title("💳 Credit Card Fraud Detection")

if df_data is None:
    st.info(
        "Dataset file was not found in the repo, so dropdowns/charts use defaults. "
        "Predictions will still work if your model file is present."
    )
else:
    st.caption("Dropdowns are populated from your dataset (when available).")

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
        merchant_clean = st.selectbox("Merchant", merchant_choices)
        category = st.selectbox("Category", category_choices)
        job = st.selectbox("Job", job_choices)
        city = st.text_input("City", value="New York")
        city_pop = st.number_input("City population", min_value=0, value=50000, step=1000)

    st.markdown("### Customer")
    age = st.slider("Customer age", min_value=18, max_value=95, value=35)

    st.markdown("### Location")
    default_lat = safe_median(df_data, "lat", 40.7128)
    default_long = safe_median(df_data, "long", -74.0060)
    default_merch_lat = safe_median(df_data, "merch_lat", 40.7128)
    default_merch_long = safe_median(df_data, "merch_long", -74.0060)

    lat = st.number_input("Customer latitude (lat)", value=default_lat)
    long_ = st.number_input("Customer longitude (long)", value=default_long)
    merch_lat = st.number_input("Merchant latitude (merch_lat)", value=default_merch_lat)
    merch_long = st.number_input("Merchant longitude (merch_long)", value=default_merch_long)

    submitted = st.form_submit_button("Predict fraud risk")


# Predict

if submitted:
    trans_dayofweek = trans_date.weekday()
    trans_month = trans_date.month

    
    merchant_raw = merchant_clean_to_raw.get(str(merchant_clean), str(merchant_clean))

    row = {
        "merchant": str(merchant_raw),
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

    # Add required hidden columns for current trained model
    for col, val in HIDDEN_DEFAULTS.items():
        if col not in X_input.columns:
            X_input[col] = val

    # Ensure correct order and presence
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

        st.subheader("📊 Prediction analysis")

        # 1) Probability chart
        fig1 = plt.figure()
        plt.bar(["Legit (1-p)", "Fraud (p)"], [1 - prob, prob])
        plt.axhline(y=threshold, linestyle="--")
        plt.ylim(0, 1)
        plt.title("Model output probability")
        plt.ylabel("Probability")
        st.pyplot(fig1)
        plt.close(fig1)

        # 2) Amount distribution (only if dataset is available)
        if df_data is not None and "amt" in df_data.columns:
            fig2 = plt.figure()
            clipped = df_data["amt"].clip(upper=500)
            plt.hist(clipped, bins=50)
            plt.axvline(x=amt, linestyle="--")
            plt.title("Transaction amount vs typical amounts (clipped at $500)")
            plt.xlabel("Amount ($)")
            plt.ylabel("Count")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.caption("Amount distribution chart disabled (dataset not available).")

        # 3) Category fraud rates (only if dataset has is_fraud)
        if df_data is not None and "category" in df_data.columns and "is_fraud" in df_data.columns:
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
            plt.close(fig3)
        else:
            st.caption("Category fraud-rate chart disabled (dataset missing 'is_fraud').")

        if show_debug:
            with st.expander("Debug: model input"):
                st.dataframe(X_input)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.info(
            "If your model was trained with merchants like 'fraud_xxx', the app now passes the raw value "
            "even though the UI shows cleaned names."
        )
