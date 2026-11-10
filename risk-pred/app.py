# Build an interactive student performance and at-risk prediction application

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configure the Streamlit page
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "models" / "student_risk_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

# Application heading
st.title("🎓 Student Performance & At-Risk Prediction")
st.write(
    "An ML-based advisory tool to identify students "
    "who may need additional academic support."
)

st.divider()

# Student input section
st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    assignment = st.slider("Assignment Completion", 0, 100, 70)
    previous_score = st.slider("Previous Assessment Score", 0, 100, 65)

with col2:
    study_hours = st.slider("Daily Study Hours", 0.0, 8.0, 3.5)
    participation = st.slider("Participation Score", 0, 100, 60)
    late_submissions = st.slider("Late Submissions", 0, 8, 2)

# Prediction button
if st.button("Predict Student Risk", type="primary"):

    input_data = pd.DataFrame([{
        "attendance_percentage": attendance,
        "assignment_completion": assignment,
        "previous_score": previous_score,
        "study_hours": study_hours,
        "participation_score": participation,
        "late_submissions": late_submissions
    }])

    input_data = input_data[features]

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    st.metric(
        "At-Risk Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.warning("⚠️ At Risk — Human review recommended.")

        st.write("Suggested support actions:")
        st.write("- Review recent academic performance.")
        st.write("- Discuss possible learning difficulties.")
        st.write("- Consider additional academic support.")
        st.write("- Monitor progress in upcoming assessments.")

    else:
        st.success("No elevated risk flagged by the model.")

        st.write("Suggested action:")
        st.write("- Continue regular academic monitoring.")

    st.info(
        "This application provides advisory predictions only. "
        "Final decisions must involve human review."
    )