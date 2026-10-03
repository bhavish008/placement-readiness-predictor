"""
Placement Readiness Predictor - input interface
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from train import (FEATURES, MIN_CGPA, READY_CUTOFF, build_and_train,
                   readiness_score)

st.set_page_config(page_title="Placement Readiness Predictor")


@st.cache_resource  # train once, then reuse while the app is running
def get_models():
    df, models, _ = build_and_train()
    return df, models


df, models = get_models()

st.title("Placement Readiness Predictor")
st.caption("The label comes from a hand-made readiness score, not from real "
           "placement outcomes. A demo of the ML pipeline, not placement advice.")

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
    # The verdict comes from the trained model, the score from the formula.
    is_ready = models[model_name].predict(student)[0] == 1
    if is_ready:
        st.success(f"{model_name} says: Ready for placement")
    else:
        st.error(f"{model_name} says: Not ready yet")

    score = readiness_score(cgpa, backlogs)
    st.metric("Readiness score (formula)", f"{score:.0f} / 100",
              help=f"Ready means a score of {READY_CUTOFF} or more.")
    if cgpa < MIN_CGPA:
        st.write(f"Why: CGPA {cgpa:.2f} is below the minimum of {MIN_CGPA}.")
    elif backlogs not in ["0", "1"]:
        st.write(f"Why: {backlogs} backlogs is over the limit of 1.")
    else:
        side = "reaches" if score >= READY_CUTOFF else "falls short of"
        st.write(f"Why: eligible, and the score of {score:.0f} {side} the "
                 f"cutoff of {READY_CUTOFF}. One backlog costs 15 points, "
                 "so it needs a higher CGPA to make up for it.")
    if is_ready != (score >= READY_CUTOFF):
        st.warning("The model disagrees with the formula here. This student "
                   "sits close to a cutoff, where the model can be wrong.")
