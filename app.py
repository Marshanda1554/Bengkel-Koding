import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load('model_churn_rf.pkl')

st.title("Prediksi Churn Pelanggan")
st.write("Masukkan data pelanggan untuk prediksi.")

# Input Sederhana
tenure = st.number_input("Tenure (Bulan)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# Karena model kamu butuh banyak kolom, kita isi kolom lainnya dengan nilai default
if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])
    
    res = model.predict(input_df)
    if res[0] == 'Yes':
        st.error("Pelanggan akan Churn")
    else:
        st.success("Pelanggan tetap bertahan")
