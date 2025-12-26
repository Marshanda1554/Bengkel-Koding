import streamlit as st
import pandas as pd
import joblib

# Judul
st.title("Aplikasi Prediksi Churn")

# Load Model
try:
    model = joblib.load('model_churn_rf.pkl')
    st.success("Model Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# Input Manual (Angka saja agar aman)
st.write("Masukkan Data (Gunakan Angka):")
tenure = st.number_input("Tenure (Bulan)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# Tombol Prediksi
if st.button("Prediksi"):
    try:
        # 1. Buat data input dengan 20 fitur (sesuai jumlah kolom saat training)
        # Kita isi dengan nilai 0 dulu sebagai dasar
        df_input = pd.DataFrame([[0] * 20]) 
        
        # 2. Masukkan input dari user ke posisi kolom yang tepat (disesuaikan dengan dataset Telco)
        # Indeks ini adalah perkiraan posisi kolom tenure, MonthlyCharges, dan TotalCharges
        df_input.iloc[0, 4] = tenure           # Indeks 4 biasanya tenure
        df_input.iloc[0, 18] = monthly_charges # Indeks 18 biasanya MonthlyCharges
        df_input.iloc[0, 19] = total_charges   # Indeks 19 biasanya TotalCharges
        
        # 3. Prediksi
        prediction = model.predict(df_input)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Tetap)")
            
        except ValueError as e:
        # Jika masih error jumlah kolom, kita bisa lihat berapa jumlah yang benar di sini
        st.error(f"Error jumlah kolom: {e}")
        except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
