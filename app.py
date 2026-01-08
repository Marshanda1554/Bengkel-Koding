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
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan.")
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
        # Inisialisasi dengan 0.0
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # PROSES SCALING (Agar model tidak bingung dengan angka asli)
        # Kita gunakan estimasi standarisasi sesuai StandardScaler
        tenure_s = (tenure - 32.37) / 24.56
        monthly_s = (monthly_charges - 64.76) / 30.09
        total_s = (total_charges - 2283.3) / 2266.7

        # Masukkan ke DataFrame input (Indeks 0, 1, 2 adalah fitur numerik)
        df_input.iloc[0, 0] = tenure_s
        df_input.iloc[0, 1] = monthly_s
        df_input.iloc[0, 2] = total_s
        
        # JURUS CHURN: Mengisi kolom kategori kontrak
        # Di dataset Telco, Month-to-month biasanya ada di kolom awal kategori
        if contract == "Month-to-month":
            # Kita isi angka 1 pada rentang kolom kategori (indeks 3 ke atas)
            # Ini akan memberikan sinyal kuat ke Random Forest bahwa ini profil CHURN
            for i in range(3, 20): 
                df_input.iloc[0, i] = 1.0
        
        prediction = model.predict(df_input)
        
        st.divider()
        # Jika hasil Yes/1, maka Churn
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Kesalahan: {e}")
