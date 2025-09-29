import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
import numpy as np # Import numpy for isnan check

# (Optional) helps unpickling but not strictly required if sklearn is installed
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="📊")

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

# No specific label->code maps needed for Titanic beyond 'Sex'
# Keep 'Sex' mapping for clarity in UI, even though it's simple
sex_map = {
    "male": "male",
    "female": "female",
}


# Helper: label->code for UI selections - Simplified for Sex
def label_to_code(selection_label: str, mapping: dict) -> str:
    # For Sex, label is the same as code
    return selection_label

# ---------- UI ----------
st.title("Titanic Survival Prediction")
st.caption("Predicting passenger survival on the Titanic.")

st.header("Enter Passenger Details")

def num_input(name, default, lo, hi, step=1):
    r = numeric_ranges.get(name, {})
    lo = int(r.get("min", lo)) if not np.isnan(lo) else lo # Handle potential NaN defaults
    hi = int(r.get("max", hi)) if not np.isnan(hi) else hi
    val = int(r.get("default", default)) if not np.isnan(default) else default
    # Use number_input for more flexibility with Pclass (integer input) and Age/Fare (float potentially)
    if name in ['Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']:
         return st.number_input(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)
    else: # Age and Fare can be floats
         # Adjusting step for Age and Fare for potentially finer input
         input_step = 1.0 if name == 'Age' else 0.1
         return st.number_input(name.replace("_", " ").title(), min_value=float(lo), max_value=float(hi), value=float(val), step=input_step)


# Numeric features for Titanic
pclass = num_input("Pclass", 3, 1, 3) # Pclass is treated as numerical by the model
age = num_input("Age", 30, 0, 80)
siblings_spouses = num_input("Siblings/Spouses Aboard", 0, 0, 8)
parents_children = num_input("Parents/Children Aboard", 0, 0, 6)
fare = num_input("Fare", 15, 0, 520, step=0.1)


st.subheader("Categorical Features")

# Only 'Sex' for Titanic
sex_label = st.selectbox("Sex", list(sex_map.values()))
sex = label_to_code(sex_label, sex_map)


# ---------- Build raw row ----------
raw_row = {
    "Pclass": pclass,
    "Sex": sex, # Original categorical value
    "Age": age,
    "Siblings/Spouses Aboard": siblings_spouses,
    "Parents/Children Aboard": parents_children,
    "Fare": fare,
}


raw_df = pd.DataFrame([raw_row])

# ---------- Encode EXACTLY like training ----------
# OHE only 'Sex'; drop_first=True
ohe_cols = ["Sex"]

input_encoded = pd.get_dummies(raw_df, columns=ohe_cols, drop_first=True, dtype=int)

# Make sure all expected training columns exist and in the same order
# Add missing columns with value 0 (for categories not present in the input row)
# and reorder columns to match the training data
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure the order of columns matches expected_columns
input_encoded = input_encoded[expected_columns]


st.divider()
if st.button("Predict Survival"):
    try:
        pred = model.predict(input_encoded)
        proba = getattr(model, "predict_proba", None)

        st.subheader("Prediction Result")
        if pred[0] == 1: # 1 for Survived, 0 for Did Not Survive
            st.success("Prediction: Survived")
        else:
            st.error("Prediction: Did Not Survive")

        if callable(proba):
            p = proba(input_encoded)[0]
            # Assuming the model's classes_ are in the order [0, 1] (Did Not Survive, Survived)
            st.write(f"Probability of Did Not Survive: {p[0]:.2f}")
            st.write(f"Probability of Survived: {p[1]:.2f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
