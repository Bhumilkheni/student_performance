import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = r"C:\Data Science\Machine Learning\Project\Student_Performance\models\student_model.pkl"

font_px = 20

CSS = """
<style>
/* base text */
html, body, .stApp, .stMarkdown p, .stMarkdown li, .stCaption, [data-testid="stCaptionContainer"] {
    font-size: __FONT__px !important;
}
/* headings */
h1 {
    font-size: calc(__FONT__px * 1.45) !important;
    white-space: nowrap !important;
}
h2, h3 { font-size: calc(__FONT__px * 1.6) !important; }
/* widget labels (sliders, selectboxes, radios) */
[data-testid="stWidgetLabel"] p, label p {
    font-size: calc(__FONT__px * 1.05) !important;
}
/* selectbox text + dropdown options */
div[data-baseweb="select"] *, ul[role="listbox"] * {
    font-size: calc(__FONT__px * 0.95) !important;
}
/* slider value bubbles */
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
    font-size: calc(__FONT__px * 0.85) !important;
}
/* metrics (result numbers) */
[data-testid="stMetricValue"] { font-size: calc(__FONT__px * 2.2) !important; }
[data-testid="stMetricLabel"] p { font-size: calc(__FONT__px * 1.0) !important; }
/* button */
.stButton button p, button[kind="secondaryFormSubmit"] p {
    font-size: calc(__FONT__px * 1.1) !important;
}
/* alerts (success / warning / error) */
[data-testid="stAlert"] p { font-size: calc(__FONT__px * 1.0) !important; }
</style>
"""
st.markdown(CSS.replace("__FONT__", str(font_px)), unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def performance_level(score):
    if score >= 85:
        return "Excellent 🌟", "success"
    if score >= 70:
        return "Good 👍", "success"
    if score >= 50:
        return "Average 📘", "warning"
    return "Poor ⚠️", "error"


st.title("🎓 Student Performance Prediction")
st.write("Enter the student's details to predict the final exam score.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model not found. Run the Jupyter notebook first to create `models/student_model.pkl`."
    )
    st.stop()

model = load_model()

with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        study_hours = st.slider("Study hours / day", 0.0, 12.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 30, 100, 80)
        previous_score = st.slider("Previous exam score", 0, 100, 65)
        sleep_hours = st.slider("Sleep hours / night", 3.0, 10.0, 7.0, 0.5)
        assignments = st.slider("Assignments completed (of 10)", 0, 10, 7)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        parental_education = st.selectbox(
            "Parental education", ["High School", "Bachelor", "Master", "PhD"]
        )
        internet = st.radio("Internet access", ["Yes", "No"], horizontal=True)
        extracurricular = st.radio(
            "Extracurricular activities", ["Yes", "No"], horizontal=True
        )
        tutoring = st.radio("Extra tutoring", ["Yes", "No"], horizontal=True)
    submitted = st.form_submit_button("Predict Performance", use_container_width=True)

if submitted:
    row = pd.DataFrame(
        [
            {
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_score": previous_score,
                "sleep_hours": sleep_hours,
                "assignments_completed": assignments,
                "gender": gender,
                "parental_education": parental_education,
                "internet_access": internet,
                "extracurricular": extracurricular,
                "tutoring": tutoring,
            }
        ]
    )
    score = float(np.clip(model.predict(row)[0], 0, 100))
    level, kind = performance_level(score)

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Predicted Final Score", f"{score:.1f} / 100")
    m2.metric("Performance Level", level)
    st.progress(int(score))
    getattr(st, kind)(f"This student is predicted to perform at an **{level}** level.")

    tips = []
    if study_hours < 3:
        tips.append("Increase daily study time (aim for 3–5 hours).")
    if attendance < 75:
        tips.append("Improve attendance — it strongly affects results.")
    if assignments < 7:
        tips.append("Complete more assignments.")
    if sleep_hours < 6:
        tips.append("Get 7–8 hours of sleep for better focus.")
    if tips:
        st.markdown("**💡 Suggestions**")
        for t in tips:
            st.markdown(f"- {t}")

st.caption("Built with scikit-learn & Streamlit • Internship ML Project")
