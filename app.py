import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# Load saved pipeline and model
pipeline = joblib.load("artifacts/pipeline.pkl")
model = keras.models.load_model("artifacts/model.h5", compile=False)

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details below:")

# Example user inputs (adjust based on your dataset features)
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
age = st.number_input("Age", min_value=18, max_value=92, value=35)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider("Tenure (years)", 0, 10, 5)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Collect inputs into a dataframe
user_data = pd.DataFrame([{
    "Gender": gender,
    "Geography": geography,
    "Age": age,
    "Balance": balance,
    "CreditScore": credit_score,
    "Tenure": tenure,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

# if st.button("Predict Churn"):
#     # Apply preprocessing
#     X_processed = pipeline.transform(user_data)

#     # Predict probability
#     prob = model.predict(X_processed)[0][0]
#     result = "Churn ❌" if prob >= 0.5 else "No Churn ✅"

#     st.write(f"**Churn Probability:** {prob:.2%}")
#     st.write(f"**Prediction:** {result}")
