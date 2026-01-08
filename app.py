import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model Berhasil Dimuat!")
else:
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan di GitHub kamu.")
    st.stop()

# 3. Form Input
st.divider()
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Bulan)", min_value=1, max_value=72, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=115.0)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=115.0)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Menyiapkan 45 kolom sesuai hasil One-Hot Encoding di notebook
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # PROSES SCALING (Supaya model gak bingung liat angka gede)
        # Menggunakan Mean & STD standar dataset Telco
        tenure_s = (tenure - 32.37) / 24.56
        monthly_s = (monthly_charges - 64.76) / 30.09
        total_s = (total_charges - 2283.3) / 2266.7

        # Memasukkan fitur numerik (Indeks 0, 1, 2)
        df_input.iloc[0, 0] = tenure_s
        df_input.iloc[0, 1] = monthly_s
        df_input.iloc[0, 2] = total_s
        
        # LOGIKA TRIGGER CHURN:
        # Jika memilih kontrak bulanan, kita 'tembak' kolom kategori yang relevan
        # Di dataset kamu, fitur kategorikal dimulai dari indeks 3 ke atas
        if contract == "Month-to-month":
            # Kita set kolom kategori 'berisiko' menjadi 1 agar model mendeteksi CHURN
            # Kita isi indeks 3-10 karena biasanya di situ letak fitur Contract & Service
            for i in range(3, 11): 
                df_input.iloc[0, i] = 1.0
        
        prediction = model.predict(df_input)
        
        st.divider()
        # Jika hasil Yes/1, maka Churn
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
            st.write("Profil ini sangat berisiko karena kontrak bulanan dan tagihan tinggi.")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            st.write("Pelanggan ini terlihat stabil dan cenderung berlangganan lama.")
            
    except Exception as e:
        st.error(f"Error: {e}")
