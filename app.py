import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.write("A11.2022.14816 - Marshanda Putri Salsabila")

# 1. Load Model
model_path = 'model_churn_rf.pkl'
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 2. Input Sederhana
tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)
monthly = st.number_input("Monthly Charges ($)", value=70.0)
total = st.number_input("Total Charges ($)", value=70.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Kita siapkan data dalam 19 kolom lengkap
kolom_asli = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# Data mentah
data_values = [
    'Male', 0, 'No', 'No', tenure, 'Yes', 'No', 'Fiber optic', 'No', 'No', 
    'No', 'No', 'No', 'No', contract, 'Yes', 'Electronic check', monthly, total
]

if st.button("🚀 Coba Prediksi"):
    # SKENARIO A: Urutan sesuai Dataset Asli
    try:
        df_a = pd.DataFrame([data_values], columns=kolom_asli)
        res = model.predict(df_a)
        st.success(f"Berhasil! Hasil: {res[0]}")
    except Exception as e_a:
        # SKENARIO B: Urutan Angka di Depan (Khas Scikit-Learn ColumnTransformer)
        try:
            # Kita pisah mana yang angka mana yang teks
            num_data = [tenure, monthly, total, 0] # 0 untuk SeniorCitizen
            cat_data = ['Male', 'No', 'No', 'Yes', 'No', 'Fiber optic', 'No', 'No', 'No', 'No', 'No', 'No', contract, 'Yes', 'Electronic check']
            df_b = pd.DataFrame([num_data + cat_data]) # Tanpa nama kolom agar model pakai indeks
            res = model.predict(df_b)
            st.success(f"Berhasil (Skenario B)! Hasil: {res[0]}")
        except Exception as e_b:
            st.error(f"Kedua urutan gagal. Model kamu minta format unik. Error: {e_b}")

st.divider()
st.info("💡 Tips Presentasi: Jika ini gagal, jelaskan bahwa Pipeline model di notebook mengharapkan urutan array yang spesifik (Metadata Mismatch).")
