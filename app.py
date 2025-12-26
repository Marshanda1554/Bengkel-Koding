import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul
st.title("Aplikasi Prediksi Churn")
st.write("Masukkan data pelanggan untuk melihat prediksi.")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error("File model_churn_rf.pkl tidak ditemukan di GitHub!")
    st.stop()

# 3. Input Data
st.divider()
tenure = st.number_input("Tenure (Bulan)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# 4. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    try:
        # Membuat data input dengan jumlah kolom yang sesuai (misal 20)
        # Sesuai instruksi sebelumnya, kita isi dengan angka nol sebagai dasar
        df_input = pd.DataFrame([[0] * 20]) 
        
        # Isi posisi kolom tenure, monthly, dan total
        # (Indeks 4, 18, 19 adalah standar fitur Telco Churn)
        df_input.iloc[0, 4] = tenure
        df_input.iloc[0, 18] = monthly_charges
        df_input.iloc[0, 19] = total_charges
        
        # Eksekusi Prediksi
        prediction = model.predict(df_input)
        
        st.subheader("Hasil Analisis:")
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ Pelanggan diprediksi akan STAY (Bertahan)")
            
    except Exception as e:
        st.warning(f"Terjadi penyesuaian data: {e}")
        # Jika error karena jumlah kolom bukan 20, coba otomatis pakai 19
        try:
            df_input_alt = pd.DataFrame([[0] * 19])
            df_input_alt.iloc[0, 4] = tenure
            df_input_alt.iloc[0, 17] = monthly_charges
            df_input_alt.iloc[0, 18] = total_charges
            prediction = model.predict(df_input_alt)
            if prediction[0] == 'Yes' or prediction[0] == 1:
                st.error("⚠️ Pelanggan diprediksi akan CHURN (Berhenti)")
            else:
                st.success("✅ Pelanggan diprediksi akan STAY (Bertahan)")
        except Exception as e2:
            st.error(f"Gagal total: {e2}")
