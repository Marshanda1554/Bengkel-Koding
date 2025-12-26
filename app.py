import streamlit as st
import pandas as pd
import joblib
import os

st.title("Aplikasi Prediksi Churn")

# 1. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
else:
    st.error("File model tidak ditemukan!")
    st.stop()

# 2. Input Sederhana
st.divider()
tenure = st.number_input("Tenure (Bulan)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# 3. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    try:
        # Cek jumlah fitur yang diminta model
        n_features = model.n_features_in_
        st.info(f"Info: Model kamu ternyata meminta {n_features} kolom.")
        
        # Buat data input sesuai jumlah yang diminta model
        df_input = pd.DataFrame([[0] * n_features])
        
        # Mengisi kolom (kita coba isi di beberapa posisi umum)
        if n_features >= 19:
            df_input.iloc[0, 4] = tenure
            df_input.iloc[0, n_features-2] = monthly
            df_input.iloc[0, n_features-1] = total
        
        prediction = model.predict(df_input)
        
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ Hasil: Pelanggan diprediksi CHURN")
        else:
            st.success("✅ Hasil: Pelanggan diprediksi STAY")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
