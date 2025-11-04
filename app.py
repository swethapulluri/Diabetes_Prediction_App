import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("diabetes_model.pkl","rb"))

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
# Add a top banner
st.markdown("""
    <div style="background-color:#FF4B4B;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">🧠 ML Project by Swetha — XGBoost-based Diabetes Predictor</h2>
    </div>
    """, unsafe_allow_html=True)


# Title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app uses a Machine Learning model (XGBoost) to predict whether a person is likely to have **diabetes** 
based on their medical information.
""")

# Sidebar
st.sidebar.header("Enter Patient Details:")
preg = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose Level", 0, 200)
bp = st.sidebar.number_input("Blood Pressure", 0, 150)
skin = st.sidebar.number_input("Skin Thickness", 0, 100)
insulin = st.sidebar.number_input("Insulin Level", 0, 900)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.sidebar.number_input("Age", 1, 120)

# Predict button
if st.sidebar.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    st.subheader("Prediction Result:")
    if result == 1:
        st.error(f"The person is likely to have Diabetes.\n\n**Confidence: {prob[1]*100:.2f}%**")
    else:
        st.success(f"The person is NOT likely to have Diabetes.\n\n**Confidence: {prob[0]*100:.2f}%**")

    # ---- Probability Bar Chart ----
    st.subheader("Prediction Probability")
    st.bar_chart({
        "Diabetes Probability": [prob[1]],
        "No Diabetes Probability": [prob[0]]
    })

