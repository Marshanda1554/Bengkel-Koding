import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

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
st.divider()
tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)
monthly = st.number_input("Monthly Charges ($)", value=70.0)
total = st.number_input("Total Charges ($)", value=70.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Konversi kontrak ke angka (Asumsi Label Encoding)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

if st.button("🚀 Prediksi Sekarang"):
    try:
        # Kita buat 19 kolom, tapi SEMUANYA berisi angka 0 atau 1
        # Tidak ada kata 'Male' atau 'Yes' lagi agar tidak error float
        
        # Susunan: 3 Kolom utama kamu, sisanya angka 0
        raw_data = [
            0, # gender (kita ganti jadi 0)
            0, # SeniorCitizen
            0, # Partner
            0, # Dependents
            tenure, 
            1, # PhoneService
            0, # MultipleLines
            1, # InternetService
            0, # OnlineSecurity
            0, # OnlineBackup
            0, # DeviceProtection
            0, # TechSupport
            0, # StreamingTV
            0, # StreamingMovies
            contract_map[contract], 
            1, # PaperlessBilling
            0, # PaymentMethod
            monthly, 
            total
        ]
        
        # Ubah jadi array numpy
        final_input = np.array(raw_data).reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(final_input)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: CHURN")
        else:
            st.success("✅ HASIL: STAY")
            
    except Exception as e:
        st.error(f"Kesalahan Sistem: {e}")
        st.info("Pesan: Terjadi mismatch pada jumlah fitur (Features Mismatch).")
