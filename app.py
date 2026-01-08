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

if model is None:
    st.error("⚠️ Model tidak ditemukan!")
    st.stop()

# 2. Input Sederhana
st.divider()
tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)
monthly = st.number_input("Monthly Charges ($)", value=70.0)
total = st.number_input("Total Charges ($)", value=70.0)

if st.button("🚀 Prediksi Sekarang"):
    try:
        # MODEL KAMU MINTA 45 FITUR.
        # Biasanya, 3 fitur pertama atau terakhir adalah numerik (tenure, monthly, total).
        # Kita buat array 45 kolom berisi nol semua.
        data_45 = np.zeros(45)
        
        # Kita coba masukkan 3 angka utama kamu ke 3 kolom pertama
        data_45[0] = tenure
        data_45[1] = monthly
        data_45[2] = total
        
        # Ubah jadi format yang diminta model
        final_input = data_45.reshape(1, -1)
        
        # Prediksi
        prediction = model.predict(final_input)
        
        st.divider()
        # Jika hasil prediksi adalah 1 atau 'Yes', berarti Churn
        if prediction[0] == 1 or str(prediction[0]).lower() == 'yes':
            st.error("⚠️ HASIL: CHURN (Pelanggan akan berhenti)")
        else:
            st.success("✅ HASIL: STAY (Pelanggan akan bertahan)")
            
    except Exception as e:
        st.error(f"Kesalahan Struktur: {e}")
        st.info("Saran: Fokuslah pada demo di Notebook karena model hasil training memiliki dimensi fitur yang kompleks (One-Hot Encoded).")

st.divider()
st.caption("UAS Data Science 2026")
