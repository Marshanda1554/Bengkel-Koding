import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul
st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini memprediksi apakah pelanggan akan berhenti atau tetap berlangganan.")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
else:
    st.error("File model_churn_rf.pkl tidak ditemukan!")
    st.stop()

# 3. Input Data
st.divider()
st.header("Input Data Pelanggan")
tenure = st.number_input("Tenure (Bulan)", 0, 100, 1)
monthly = st.number_input("Monthly Charges ($)", 0.0, 500.0, 150.0)
total = st.number_input("Total Charges ($)", 0.0, 10000.0, 150.0)

# 4. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    try:
        # SESUAI INFO: Model minta 45 kolom
        n_features = 45 
        df_input = pd.DataFrame([[0] * n_features])
        
        # Mengisi kolom penting (Tenure, Monthly, Total)
        # Kita isi di beberapa posisi yang biasanya ditempati fitur ini
        df_input.iloc[0, 0] = tenure
        df_input.iloc[0, 1] = monthly
        df_input.iloc[0, 2] = total
        
        # Melakukan prediksi
        prediction = model.predict(df_input)
        
        st.divider()
        st.subheader("Hasil Prediksi:")
        
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
            st.write("Saran: Berikan promo atau diskon agar pelanggan tidak berhenti.")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            st.write("Saran: Pertahankan layanan yang sudah ada.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
