import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# (Optional) helps unpickling but not strictly required if sklearn is installed
from sklearn.tree import DecisionTreeClassifier  #

st.set_page_config(page_title="Credit Default Prediction", page_icon="📊")

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "decision_tree_model.pkl"   # your pruned tree
DATA_INFO_PATH = HERE / "data_info.pkl"         # must contain expected_columns, etc.

# ---------- Load artifacts ----------
@st.cache_resource
def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n{e}")
    st.stop()

try:
    data_info = load_pickle(DATA_INFO_PATH)
except Exception as e:
    st.error(
        f"Could not load data_info at {DATA_INFO_PATH}.\n"
        f"Ensure data_info.pkl exists and includes expected_columns.\n{e}"
    )
    st.stop()

expected_columns = data_info["expected_columns"]

# These lists are only used to make nicer sliders; they won't change encoding
numeric_ranges = data_info.get("numeric_ranges", {})

# ---------- Code↔Label maps (UI shows labels; encoding uses codes) ----------
checking_balance_map = {
    "A11": "< 0 DM",
    "A12": "0 ≤ … < 200 DM",
    "A13": "≥ 200 DM",
    "A14": "no checking account",
}
credit_history_map = {
    "A30": "no credits / all paid duly",
    "A31": "all credits at this bank paid duly",
    "A32": "credits paid duly till now",
    "A33": "delay in past",
    "A34": "critical / other credits",
}
purpose_map = {
    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture/equipment",
    "A43": "radio/TV",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation (?)",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
}
savings_balance_map = {
    "A61": "< 100 DM",
    "A62": "100 ≤ … < 500 DM",
    "A63": "500 ≤ … < 1000 DM",
    "A64": "≥ 1000 DM",
    "A65": "no savings account",
}
other_credit_map = {
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
}
housing_map = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
}
job_map = {
    "A171": "unemployed / unskilled non-resident",
    "A172": "unskilled resident",
    "A173": "skilled employee / official",
    "A174": "management / self-employed / highly qualified / officer",
}
phone_map = {
    "A191": "none",
    "A192": "yes, registered",
}

# Ordinal mapping for employment_duration (training used ordinal, not OHE)
employment_duration_levels = ["unemployed", "< 1 yr", "1–4 yrs", "4–7 yrs", "≥ 7 yrs"]
employment_duration_ord = {
    "unemployed": 0,
    "< 1 yr": 1,
    "1–4 yrs": 2,
    "4–7 yrs": 3,
    "≥ 7 yrs": 4,
}

# Helper: label->code for UI selections
def label_to_code(selection_label: str, mapping: dict) -> str:
    # mapping is code->label; invert to label->code
    inv = {v: k for k, v in mapping.items()}
    return inv[selection_label]

# ---------- UI ----------
st.title("Credit Default Prediction")
st.caption("Encodings: employment_duration = ordinal; all other categoricals = one-hot (drop_first=True).")

st.header("Enter Customer Details")

def num_slider(name, default, lo, hi, step=1):
    r = numeric_ranges.get(name, {})
    lo = int(r.get("min", lo))
    hi = int(r.get("max", hi))
    val = int(r.get("default", default))
    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

# Numeric features
months_loan_duration = num_slider("months_loan_duration", 12, 6, 72)
amount               = num_slider("amount", 1000, 250, 20000)
percent_of_income    = num_slider("percent_of_income", 2, 1, 4)
years_at_residence   = num_slider("years_at_residence", 2, 1, 4)
age                  = num_slider("age", 30, 18, 75)
existing_loans_count = num_slider("existing_loans_count", 1, 0, 5)
dependents           = num_slider("dependents", 1, 0, 5)

st.subheader("Categorical Features (friendly labels)")

# Show labels, convert back to codes
checking_balance_label = st.selectbox("Checking Balance", list(checking_balance_map.values()))
checking_balance = label_to_code(checking_balance_label, checking_balance_map)

credit_history_label = st.selectbox("Credit History", list(credit_history_map.values()))
credit_history = label_to_code(credit_history_label, credit_history_map)

purpose_label = st.selectbox("Purpose", list(purpose_map.values()))
purpose = label_to_code(purpose_label, purpose_map)

savings_balance_label = st.selectbox("Savings Balance", list(savings_balance_map.values()))
savings_balance = label_to_code(savings_balance_label, savings_balance_map)

# Ordinal: keep label for UX, map to integer for the model
employment_duration_label = st.selectbox("Employment Duration", employment_duration_levels)
employment_duration = employment_duration_ord[employment_duration_label]

other_credit_label = st.selectbox("Other Credit", list(other_credit_map.values()))
other_credit = label_to_code(other_credit_label, other_credit_map)

housing_label = st.selectbox("Housing", list(housing_map.values()))
housing = label_to_code(housing_label, housing_map)

job_label = st.selectbox("Job", list(job_map.values()))
job = label_to_code(job_label, job_map)

phone_label = st.selectbox("Phone", list(phone_map.values()))
phone = label_to_code(phone_label, phone_map)

# ---------- Build raw row ----------
raw_row = {
    "months_loan_duration": months_loan_duration,
    "amount": amount,
    "percent_of_income": percent_of_income,
    "years_at_residence": years_at_residence,
    "age": age,
    "existing_loans_count": existing_loans_count,
    "dependents": dependents,
    # Categorical codes (as in training)
    "checking_balance": checking_balance,
    "credit_history": credit_history,
    "purpose": purpose,
    "savings_balance": savings_balance,
    # Ordinal numeric
    "employment_duration": employment_duration,
    # More categorical codes
    "other_credit": other_credit,
    "housing": housing,
    "job": job,
    "phone": phone,
}

raw_df = pd.DataFrame([raw_row])

# ---------- Encode EXACTLY like training ----------
# OHE only these categorical code columns; drop_first=True
ohe_cols = [
    "checking_balance", "credit_history", "purpose",
    "savings_balance", "other_credit", "housing", "job", "phone"
]

input_encoded = pd.get_dummies(raw_df, columns=ohe_cols, drop_first=True, dtype=int)

# Make sure all expected training columns exist and in the same order
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_columns]

st.divider()
if st.button("Predict"):
    try:
        pred = model.predict(input_encoded)
        proba = getattr(model, "predict_proba", None)

        st.subheader("Prediction Result")
        if pred[0] == 0:
            st.success("Prediction: No Default")
        else:
            st.error("Prediction: Default")

        if callable(proba):
            p = proba(input_encoded)[0]
            st.write(f"Probability of No Default: {p[0]:.2f}")
            st.write(f"Probability of Default: {p[1]:.2f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
