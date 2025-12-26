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
    # Kita buat dummy data untuk fitur lainnya agar jumlah kolom pas
    # Sesuaikan jumlah kolom (misal 19 atau 20) sesuai training kamu
    # Di sini saya buatkan contoh 19 kolom sesuai standar Telco Churn
    input_data = [[0]*19] # Buat list berisi 19 angka nol
    input_data[0][4] = tenure # Sesuaikan urutan kolom tenure
    input_data[0][17] = monthly_charges
    input_data[0][18] = total_charges
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 'Yes' or prediction[0] == 1:
        st.error("Hasil: Pelanggan CHURN")
    else:
        st.success("Hasil: Pelanggan TETAP")
