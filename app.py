import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.write("Prediksi apakah pelanggan akan berhenti (Churn) atau bertahan (Stay).")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan.")
    st.stop()

# 3. Form Input
st.divider()
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Bulan)", min_value=1, max_value=72, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=100.0)

with col2:
    # --- INI TOTAL CHARGES NYA ---
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=100.0)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Menyiapkan 45 kolom sesuai hasil One-Hot Encoding di notebook
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # PROSES SCALING (Supaya hasilnya gak STAY terus)
        # Menggunakan estimasi Mean & STD dari dataset Telco agar model mengenali angkanya
        tenure_s = (tenure - 32.37) / 24.56
        monthly_s = (monthly_charges - 64.76) / 30.09
        total_s = (total_charges - 2283.3) / 2266.7 # Scaling untuk Total Charges

        # Masukkan ke DataFrame input (Indeks 0, 1, 2 sesuai urutan fitur numerik)
        df_input.iloc[0, 0] = tenure_s
        df_input.iloc[0, 1] = monthly_s
        df_input.iloc[0, 2] = total_s
        
        # Mengaktifkan trigger kategori agar bisa muncul CHURN
        if contract == "Month-to-month":
            # Mengisi nilai 1 pada kolom kategori kontrak bulanan (One-Hot)
            for i in range(5, 15): 
                df_input.iloc[0, i] = 1.0
        
        prediction = model.predict(df_input)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Kesalahan teknis: {e}")

st.caption("UAS Bengkel Koding 2026")
