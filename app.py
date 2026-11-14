"""Diabetes Risk Screening Tool - Streamlit app.   Run:  streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import utils  # noqa: F401  (needed to unpickle the saved pipeline)

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺")

YES_NO = {
    "HighBP": "Have you ever been told you have high blood pressure?",
    "HighChol": "Have you ever been told you have high cholesterol?",
    "CholCheck": "Have you had your cholesterol checked in the last 5 years?",
    "Smoker": "Have you smoked at least 100 cigarettes in your life?",
    "Stroke": "Have you ever had a stroke?",
    "HeartDiseaseorAttack": "Have you ever had coronary heart disease or a heart attack?",
    "PhysActivity": "Did you do any physical activity in the past 30 days (not counting your job)?",
    "Fruits": "Do you eat fruit at least once a day?",
    "Veggies": "Do you eat vegetables at least once a day?",
    "HvyAlcoholConsump": "Heavy drinker? (men: 14+ drinks/week, women: 7+ drinks/week)",
    "AnyHealthcare": "Do you have any kind of health coverage?",
    "NoDocbcCost": "In the past year, did cost stop you from seeing a doctor?",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
}
SHORT = {"HighBP": "high blood pressure", "HighChol": "high cholesterol", "BMI": "body mass index (BMI)",
         "Age": "age group", "GenHlth": "general health rating", "PhysHlth": "days of poor physical health",
         "MentHlth": "days of poor mental health", "DiffWalk": "difficulty walking", "Income": "income level",
         "Education": "education level", "HeartDiseaseorAttack": "heart disease / heart attack",
         "PhysActivity": "physical activity", "Sex": "sex", "Smoker": "smoking history",
         "Stroke": "stroke history", "Fruits": "fruit intake", "Veggies": "vegetable intake",
         "HvyAlcoholConsump": "heavy drinking", "CholCheck": "cholesterol check",
         "AnyHealthcare": "health coverage", "NoDocbcCost": "cost barrier to doctor visits"}
GEN_HLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION = {1: "Never attended / kindergarten only", 2: "Elementary", 3: "Some high school",
             4: "High school graduate", 5: "Some college / technical school", 6: "College graduate"}
INCOME = {1: "Lowest income bracket", 2: "Low", 3: "Lower-middle", 4: "Middle-low", 5: "Middle",
          6: "Middle-high", 7: "Upper-middle", 8: "Highest income bracket"}


def age_category(years):  # BRFSS 13-level age code: 1 = 18-24, 2 = 25-29, ... 13 = 80+
    return 1 if years < 25 else min(13, (years - 25) // 5 + 2)


@st.cache_resource
def load_bundle():
    path = Path(__file__).parent / "models" / "model_short.joblib"
    return joblib.load(path) if path.exists() else None


bundle = load_bundle()
st.title("🩺 Diabetes Risk Screening")
st.caption("Answer a few questions to see whether it may be worth getting tested.")
st.warning("**This is a screening aid, not a medical diagnosis.** Only a blood test and a doctor can tell "
           "whether you have diabetes.")

if bundle is None:
    st.error("Model file not found. Run `python train.py` first to create `models/model_short.joblib`.")
    st.stop()

pipe, features, thr, ref = bundle["pipeline"], bundle["features"], bundle["threshold"], bundle["reference"]

with st.form("questionnaire"):
    ans = {}
    for f in features:
        if f in YES_NO:
            ans[f] = float(st.radio(YES_NO[f], ["No", "Yes"], horizontal=True, key=f) == "Yes")
        elif f == "Sex":
            ans[f] = float(st.radio("Sex", ["Female", "Male"], horizontal=True, key=f) == "Male")
        elif f == "BMI":
            c1, c2 = st.columns(2)
            h = c1.number_input("Height (cm)", 100, 250, 170, key="h")
            w = c2.number_input("Weight (kg)", 30, 300, 70, key="w")
            ans[f] = w / (h / 100) ** 2
        elif f == "Age":
            ans[f] = float(age_category(st.number_input("Age (years)", 18, 100, 40, key="age")))
        elif f == "GenHlth":
            ans[f] = float(st.selectbox("How would you rate your general health?", list(GEN_HLTH),
                                        format_func=GEN_HLTH.get, index=2, key=f))
        elif f in ("PhysHlth", "MentHlth"):
            what = "physical" if f == "PhysHlth" else "mental"
            ans[f] = float(st.slider(f"In the past 30 days, how many days was your {what} health not good?",
                                     0, 30, 0, key=f))
        elif f == "Education":
            ans[f] = float(st.selectbox("Highest education level", list(EDUCATION), format_func=EDUCATION.get,
                                        index=3, key=f))
        elif f == "Income":
            ans[f] = float(st.selectbox("Household income level", list(INCOME), format_func=INCOME.get,
                                        index=4, key=f))
    go = st.form_submit_button("Check my risk")

if go:
    if "BMI" in ans and not 12 <= ans["BMI"] <= 60:
        st.info(f"Your BMI ({ans['BMI']:.1f}) is outside the usual range, so please double-check height and weight.")
    X = pd.DataFrame([ans])[features]
    p = float(pipe.predict_proba(X)[0, 1])
    level = "High" if p >= thr else "Moderate" if p >= thr / 2 else "Lower"
    icon = {"High": "🔴", "Moderate": "🟠", "Lower": "🟢"}[level]
    st.subheader(f"{icon} {level} risk")
    st.metric("Model risk score", f"{p:.0%}")
    st.write({"High": "Your answers look similar to people who have diabetes or prediabetes. "
                      "Consider asking a doctor or pharmacist about a blood sugar (HbA1c / fasting glucose) test.",
              "Moderate": "Some of your answers are associated with higher risk. A routine check-up "
                          "and a blood sugar test are a good idea.",
              "Lower": "Your answers are not strongly associated with diabetes. Keep up healthy habits "
                       "and get checked regularly."}[level])
    contrib = {}
    for f in features:  # how much does each answer move the score vs. a typical respondent?
        Xr = X.copy()
        Xr[f] = ref[f]
        contrib[f] = p - float(pipe.predict_proba(Xr)[0, 1])
    top = [(f, c) for f, c in sorted(contrib.items(), key=lambda t: -t[1]) if c > 0.005][:3]
    if top:
        st.markdown("**Answers raising your score** (compared with a typical survey respondent):")
        for f, c in top:
            st.markdown(f"- {SHORT.get(f, f)}  (+{c:.0%})")

with st.expander("About this tool and its limitations"):
    st.markdown(
        "- Trained on the CDC *Diabetes Health Indicators* survey (UCI), which is **US self-reported survey data**; "
        "results may not transfer to every population.\n"
        "- The risk score is a model estimate, not a clinical probability.\n"
        "- The tool is tuned to catch most true cases, so some people flagged will not have diabetes.\n"
        "- It does not replace medical advice.")
