"""
Placement Readiness Predictor - input interface
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from train import FEATURES, build_and_train

st.set_page_config(page_title="Placement Readiness Predictor")


@st.cache_resource  # train once, then reuse while the app is running
def get_models():
    df, models, _ = build_and_train()
    return df, models


df, models = get_models()

st.title("Placement Readiness Predictor")
st.caption("Label is rule-based (CGPA of 7 or more and at most 1 backlog). "
           "A demo of the ML pipeline, not real placement advice.")

left, right = st.columns(2)
with left:
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.01)
    age = st.number_input("Age", 17, 35, 20)
    backlogs = st.selectbox("Backlogs", ["0", "1", "2", "3+"])
with right:
    course = st.selectbox("Course", sorted(df["course"].unique()))
    branch = st.selectbox("Branch", sorted(df["branch"].unique()))

model_name = st.radio("Model", ["Decision Tree", "Logistic Regression"],
                      horizontal=True)

if st.button("Predict", type="primary"):
    student = pd.DataFrame([{"cgpa": cgpa, "age": age, "backlogs": backlogs,
                             "course": course, "branch": branch}])[FEATURES]
    prob = models[model_name].predict_proba(student)[0, 1]
    if prob >= 0.5:
        st.success(f"Ready for placement ({prob:.0%} probability)")
    else:
        st.error(f"Not ready yet ({prob:.0%} probability of being ready)")

    reasons = []
    reasons.append(f"CGPA {cgpa:.2f} is "
                   + ("at or above" if cgpa >= 7 else "below") + " the 7.0 cutoff")
    reasons.append(f"{backlogs} backlog(s) is "
                   + ("within" if backlogs in ["0", "1"] else "over")
                   + " the limit of 1")
    st.write("Why: " + "; ".join(reasons) + ".")
