import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore

# Load the model
model = joblib.load("titanic_model_pipeline.pkl")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival:")

# Collect user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    outcome = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
    st.subheader(f"Prediction: {outcome}")
