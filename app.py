import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Judul & Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.write("Aplikasi ini memprediksi apakah pelanggan akan berhenti (Churn) atau bertahan (Stay).")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model siap digunakan!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error("⚠️ File model 'model_churn_rf.pkl' tidak ditemukan. Pastikan sudah diupload ke GitHub.")
    st.stop()

# 3. Form Input Data
st.divider()
st.header("📝 Data Pelanggan")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Bulan lamanya berlangganan)", min_value=1, max_value=72, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=100.0)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=100.0)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Model kamu meminta 45 fitur
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # PROSES SCALING MANUAL (Penting agar tidak 'Stay' terus)
        # Sesuai StandardScaler di notebook kamu
        tenure_s = (tenure - 32.37) / 24.56
        monthly_s = (monthly_charges - 64.76) / 30.09
        total_s = (total_charges - 2283.3) / 2266.7

        # Memasukkan fitur numerik yang sudah di-scale ke indeks kolom 0, 1, 2
        df_input.iloc[0, 0] = tenure_s
        df_input.iloc[0, 1] = monthly_s
        df_input.iloc[0, 2] = total_s
        
        # Logika Encoding untuk Kontrak (Hot Encoding)
        # Jika Month-to-month, kita aktifkan kolom kategori yang berisiko tinggi Churn
        if contract == "Month-to-month":
            for i in range(5, 15): # Indeks kolom kategori hasil One-Hot
                df_input.iloc[0, i] = 1.0
        
        # Eksekusi Prediksi
        prediction = model.predict(df_input)
        
        st.divider()
        st.subheader("🔍 Hasil Analisis:")
        
        # Menyesuaikan output label
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
            st.write("Rekomendasi: Berikan promo khusus atau diskon paket untuk mencegah pelanggan berhenti.")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            st.write("Rekomendasi: Teruskan layanan standar dan tawarkan program loyalitas jangka panjang.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.divider()
st.caption("Aplikasi Prediksi Churn - Tugas Bengkel Koding 2026")
